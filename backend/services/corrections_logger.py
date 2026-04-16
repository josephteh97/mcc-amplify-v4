"""
Corrections Logger — Training Flywheel Data Collection

Every time a user edits or deletes an element in the 3D editor, this module
writes a row to a local SQLite database.  Each row captures:

  - The original element as produced by YOLO + AI (with bbox, confidence…)
  - The user's correction (changed fields, or a deletion flag)
  - Enough context (job_id, element type/index) to trace back to the source PDF

Over time the table becomes a labelled dataset for YOLO fine-tuning:
  PDF crop at bbox  →  YOLO said X  →  human said Y

Endpoints that expose this data:
  GET /corrections/stats   — counts by element type
  GET /corrections/export  — full JSON dump for offline training scripts
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

# Single shared DB file; kept alongside the other model artefacts
DB_PATH = Path("data/corrections.db")


class CorrectionsLogger:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ── Schema ────────────────────────────────────────────────────────────────

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS corrections (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp        TEXT    NOT NULL,
                    job_id           TEXT    NOT NULL,
                    element_type     TEXT    NOT NULL,
                    element_index    INTEGER NOT NULL,
                    original_element TEXT,
                    changes          TEXT,
                    is_delete        INTEGER NOT NULL DEFAULT 0
                )
            """)
            # Index for fast per-job lookups and export queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_corrections_job
                ON corrections (job_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_corrections_type
                ON corrections (element_type)
            """)
            conn.commit()

    # ── Write ─────────────────────────────────────────────────────────────────

    def log(
        self,
        job_id: str,
        element_type: str,
        element_index: int,
        original_element: dict,
        changes: dict,
        is_delete: bool = False,
    ) -> None:
        """
        Record one user correction.  Never raises — logging failures must
        not block the edit flow.

        Parameters
        ----------
        job_id          : pipeline job the element belongs to
        element_type    : "walls" | "columns" | "doors" | "windows" | …
        element_index   : 0-based index inside the element array
        original_element: full element dict before the correction
        changes         : dict of fields the user changed  (empty for deletes)
        is_delete       : True when the user deleted the element entirely
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO corrections
                        (timestamp, job_id, element_type, element_index,
                         original_element, changes, is_delete)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        datetime.now(timezone.utc).isoformat(),
                        job_id,
                        element_type,
                        element_index,
                        json.dumps(original_element),
                        json.dumps(changes),
                        1 if is_delete else 0,
                    ),
                )
                conn.commit()

            action = "DELETE" if is_delete else f"edit {list(changes.keys())}"
            logger.info(
                f"Correction logged: {element_type}[{element_index}] "
                f"job={job_id} action={action}"
            )
        except Exception as exc:
            logger.warning(f"Corrections logging failed (non-fatal): {exc}")

    # ── Read ──────────────────────────────────────────────────────────────────

    def export(self, limit: int = 5000) -> list[dict]:
        """
        Return all corrections newest-first as a list of dicts.
        `original_element` and `changes` are decoded from JSON.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM corrections ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()

        result = []
        for row in rows:
            d = dict(row)
            d["original_element"] = json.loads(d["original_element"] or "null")
            d["changes"]          = json.loads(d["changes"]          or "null")
            d["is_delete"]        = bool(d["is_delete"])
            result.append(d)
        return result

    def defaults(self, element_type: str) -> dict:
        """
        Return the most commonly corrected value for each field of `element_type`.

        Only considers edit corrections (not deletes).  A field must appear
        in at least 2 corrections to be included — single-sample data is too
        noisy to be a reliable firm default.

        Example return for "columns":
          { "width_mm": 800.0, "depth_mm": 800.0, "material": "Concrete" }
        """
        from collections import Counter, defaultdict

        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT changes FROM corrections "
                "WHERE element_type = ? AND is_delete = 0",
                (element_type,),
            ).fetchall()

        field_values: dict[str, list] = defaultdict(list)
        for (changes_json,) in rows:
            changes = json.loads(changes_json or "{}")
            for field, value in changes.items():
                if isinstance(value, (int, float, str, bool)):
                    field_values[field].append(value)

        result = {}
        for field, values in field_values.items():
            if len(values) >= 2:
                counter = Counter(str(v) for v in values)  # str for hashability
                most_common_str = counter.most_common(1)[0][0]
                # Restore original type from the first matching value
                original = next(v for v in values if str(v) == most_common_str)
                result[field] = original

        return result

    # ── YOLO training export ───────────────────────────────────────────────────

    # Maps element_type strings to YOLO class IDs (must match column-detect.pt)
    _YOLO_CLASS_IDS: dict[str, int] = {
        "walls":    0,
        "doors":    1,
        "windows":  2,
        "columns":  3,
        "rooms":    4,
    }

    def export_yolo_training_data(self, output_dir: str = "data/yolo_training") -> dict:
        """
        Generate YOLO-format training samples from human corrections.

        For every correction that:
          - has a `bbox` in the original_element dict (YOLO detection), AND
          - corresponds to a job whose render image exists at
            ``data/jobs/{job_id}/render.jpg``

        This method writes:
          ``{output_dir}/images/{job_id}_{index}.jpg``  — cropped element image
          ``{output_dir}/labels/{job_id}_{index}.txt``  — YOLO annotation line

        Deleted elements (is_delete=True) are written with an empty .txt
        (background sample — no positive annotation).

        Returns
        -------
        dict  {"samples_written": int, "skipped": int, "output_dir": str}
        """
        from pathlib import Path as _Path
        try:
            from PIL import Image as _PIL
        except ImportError:
            return {"error": "Pillow not installed", "samples_written": 0, "skipped": 0}

        out = _Path(output_dir)
        images_dir = out / "images"
        labels_dir = out / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        corrections = self.export(limit=50_000)
        written = 0
        skipped = 0

        for row in corrections:
            orig = row.get("original_element") or {}
            bbox = orig.get("bbox")          # expected: [x1, y1, x2, y2] pixels
            if not bbox or len(bbox) < 4:
                skipped += 1
                continue

            job_id   = row["job_id"]
            idx      = row["id"]
            el_type  = row["element_type"]
            is_del   = row["is_delete"]

            render_path = _Path(f"data/jobs/{job_id}/render.jpg")
            if not render_path.exists():
                skipped += 1
                continue

            try:
                img  = _PIL.open(render_path).convert("RGB")
                iw, ih = img.size

                x1, y1, x2, y2 = (
                    max(0, int(bbox[0])), max(0, int(bbox[1])),
                    min(iw, int(bbox[2])), min(ih, int(bbox[3])),
                )
                if x2 <= x1 or y2 <= y1:
                    skipped += 1
                    continue

                # Save cropped element image
                crop = img.crop((x1, y1, x2, y2))
                stem = f"{job_id}_{idx}"
                crop.save(str(images_dir / f"{stem}.jpg"), format="JPEG", quality=90)

                # Write YOLO annotation (relative coords of the crop itself)
                label_path = labels_dir / f"{stem}.txt"
                if is_del:
                    # False positive — empty label file (background sample)
                    label_path.write_text("")
                else:
                    class_id = self._YOLO_CLASS_IDS.get(el_type, -1)
                    if class_id < 0:
                        # Unknown class — write a metadata comment only
                        label_path.write_text(f"# unknown class: {el_type}\n")
                    else:
                        # The crop IS the full element — annotation is the full image
                        label_path.write_text(f"{class_id} 0.5 0.5 1.0 1.0\n")

                written += 1
            except Exception as exc:
                logger.warning(f"YOLO export skipped correction {idx}: {exc}")
                skipped += 1

        logger.info(
            f"YOLO training export: {written} samples written, "
            f"{skipped} skipped → {out}"
        )
        return {"samples_written": written, "skipped": skipped, "output_dir": str(out)}

    def stats(self) -> dict:
        """Summary counts — total, by element type, and delete vs edit ratio."""
        with sqlite3.connect(self.db_path) as conn:
            total   = conn.execute(
                "SELECT COUNT(*) FROM corrections"
            ).fetchone()[0]
            deletes = conn.execute(
                "SELECT COUNT(*) FROM corrections WHERE is_delete = 1"
            ).fetchone()[0]
            by_type = conn.execute(
                "SELECT element_type, COUNT(*) FROM corrections GROUP BY element_type"
            ).fetchall()
            recent  = conn.execute(
                "SELECT timestamp FROM corrections ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()

        return {
            "total":       total,
            "edits":       total - deletes,
            "deletes":     deletes,
            "by_type":     {row[0]: row[1] for row in by_type},
            "last_updated": recent[0] if recent else None,
        }
