#!/usr/bin/env python3
"""
YOLO Fine-tuning Script — Training Flywheel

Each user correction captured by CorrectionsLogger is one labelled training
example: the original YOLO detection bbox + what the user said was wrong.

  - Deleted elements  → false-positive bbox (excluded from positive labels)
  - Corrected elements → confirmed class + position label

Usage (run from the project root):
    python scripts/retrain_yolo.py
    python scripts/retrain_yolo.py --epochs 30 --min-corrections 20

Requirements:
    pip install ultralytics pyyaml pillow

What it needs on disk:
    data/corrections.db                     — corrections SQLite log
    data/jobs/{job_id}/render.jpg           — rendered floor plan image
    data/jobs/{job_id}/px_detections.json   — pixel-space YOLO detections

These files are written by the pipeline automatically since the checkpoint
feature was added (data/jobs/ directory).

Output:
    data/yolo_finetune/          — YOLO dataset structure
    ml/weights/yolov11_floorplan_ft_{timestamp}.pt  — fine-tuned weights
"""

import argparse
import json
import shutil
import sqlite3
import sys
import time
from pathlib import Path


# ── Class definitions must match the YOLO model ───────────────────────────────
CLASS_NAMES = ["wall", "door", "window", "column", "room"]
CLASS_TO_ID = {n: i for i, n in enumerate(CLASS_NAMES)}

# Recipe element types (plural) → YOLO class name (singular)
RECIPE_TYPE_TO_CLASS = {
    "walls":    "wall",
    "doors":    "door",
    "windows":  "window",
    "columns":  "column",
    "rooms":    "room",
}

# Singular form used by corrections_logger element_type field
SINGULAR_TO_CLASS = {
    "wall": "wall", "door": "door", "window": "window",
    "column": "column", "room": "room",
    # Accept plural too
    "walls": "wall", "doors": "door", "windows": "window",
    "columns": "column", "rooms": "room",
}


def load_corrections(db_path: Path) -> list:
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT job_id, element_type, element_index, "
        "       original_element, changes, is_delete "
        "FROM corrections ORDER BY timestamp"
    ).fetchall()
    conn.close()
    return [
        {
            "job_id":        r[0],
            "element_type":  r[1],
            "element_index": r[2],
            "original":      json.loads(r[3]),
            "changes":       json.loads(r[4]),
            "is_delete":     bool(r[5]),
        }
        for r in rows
    ]


def build_dataset(corrections: list, dataset_dir: Path, val_fraction: float = 0.2) -> int:
    """
    Build YOLO-format dataset from corrections + per-job checkpoint data.

    Returns the number of images successfully added to the dataset.
    """
    from PIL import Image as _PIL

    shutil.rmtree(dataset_dir, ignore_errors=True)
    for split in ("train", "val"):
        (dataset_dir / "images" / split).mkdir(parents=True)
        (dataset_dir / "labels" / split).mkdir(parents=True)

    # Group corrections by job_id
    jobs: dict[str, list] = {}
    for c in corrections:
        jobs.setdefault(c["job_id"], []).append(c)

    job_ids = sorted(jobs.keys())
    n_val   = max(1, int(len(job_ids) * val_fraction))
    val_set = set(job_ids[:n_val])

    valid_count = 0

    for job_id, job_corrections in jobs.items():
        render_path  = Path(f"data/jobs/{job_id}/render.jpg")
        detect_path  = Path(f"data/jobs/{job_id}/px_detections.json")

        if not render_path.exists():
            print(f"  [skip] {job_id[:8]}… — render.jpg not found")
            continue
        if not detect_path.exists():
            print(f"  [skip] {job_id[:8]}… — px_detections.json not found")
            continue

        with open(detect_path) as f:
            detections = json.load(f)

        # Set of (element_type, element_index) pairs the user deleted
        deleted = {
            (c["element_type"], c["element_index"])
            for c in job_corrections if c["is_delete"]
        }

        split   = "val" if job_id in val_set else "train"
        dest_img = dataset_dir / "images" / split / f"{job_id}.jpg"
        shutil.copy2(render_path, dest_img)

        with _PIL.open(render_path) as img:
            img_w, img_h = img.size

        labels = []

        for el_type_raw, elements in detections.items():
            el_type = el_type_raw  # may be "walls", "columns", etc.
            class_name = RECIPE_TYPE_TO_CLASS.get(el_type) or SINGULAR_TO_CLASS.get(el_type)
            if class_name is None:
                continue
            class_id = CLASS_TO_ID.get(class_name)
            if class_id is None:
                continue

            for idx, el in enumerate(elements):
                # Skip false positives that the user deleted
                if (el_type, idx) in deleted or (el_type.rstrip("s"), idx) in deleted:
                    continue

                bbox = el.get("bbox", [])
                if len(bbox) < 4:
                    continue

                x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
                cx = (x1 + x2) / 2 / img_w
                cy = (y1 + y2) / 2 / img_h
                bw = (x2 - x1) / img_w
                bh = (y2 - y1) / img_h

                # Skip degenerate boxes
                if not (0 < bw <= 1 and 0 < bh <= 1 and 0 <= cx <= 1 and 0 <= cy <= 1):
                    continue

                labels.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        dest_lbl = dataset_dir / "labels" / split / f"{job_id}.txt"
        dest_lbl.write_text("\n".join(labels))
        valid_count += 1
        print(f"  {job_id[:8]}… [{split}] — {len(labels)} labels, {len(deleted)} deleted")

    return valid_count


def write_data_yaml(dataset_dir: Path) -> Path:
    """Write the YOLO data.yaml config file."""
    try:
        import yaml
    except ImportError:
        print("PyYAML not installed — writing data.yaml manually.")
        yaml = None

    data = {
        "path":  str(dataset_dir.absolute()),
        "train": "images/train",
        "val":   "images/val",
        "names": CLASS_NAMES,
    }
    yaml_path = dataset_dir / "data.yaml"
    if yaml:
        with open(yaml_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
    else:
        lines = [
            f"path: {data['path']}",
            f"train: {data['train']}",
            f"val: {data['val']}",
            "names:",
        ] + [f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES)]
        yaml_path.write_text("\n".join(lines) + "\n")
    return yaml_path


def main():
    parser = argparse.ArgumentParser(
        description="Retrain YOLO on correction data from CorrectionsLogger."
    )
    parser.add_argument("--epochs",          type=int,   default=20,
                        help="Fine-tuning epochs (default: 20)")
    parser.add_argument("--output",          default="ml/weights/",
                        help="Directory to save fine-tuned weights")
    parser.add_argument("--min-corrections", type=int,   default=10,
                        help="Minimum corrections before retraining (default: 10)")
    parser.add_argument("--imgsz",           type=int,   default=640,
                        help="Training image size (default: 640)")
    parser.add_argument("--dry-run",         action="store_true",
                        help="Build dataset only; skip model training")
    args = parser.parse_args()

    # ── Validate inputs ───────────────────────────────────────────────────────
    db_path = Path("data/corrections.db")
    if not db_path.exists():
        print(f"ERROR: {db_path} not found. Run the pipeline and make corrections first.")
        sys.exit(1)

    weights_path = Path("ml/weights/yolov11_floorplan.pt")
    if not weights_path.exists() and not args.dry_run:
        print(f"ERROR: base weights not found at {weights_path}")
        sys.exit(1)

    # ── Load corrections ──────────────────────────────────────────────────────
    corrections = load_corrections(db_path)
    print(f"Loaded {len(corrections)} corrections from {db_path}")

    if len(corrections) < args.min_corrections:
        print(
            f"Only {len(corrections)} correction(s) found "
            f"(minimum: {args.min_corrections}). "
            "Make more corrections in the UI before retraining."
        )
        sys.exit(0)

    n_deleted  = sum(1 for c in corrections if c["is_delete"])
    n_edits    = len(corrections) - n_deleted
    print(f"  Edits: {n_edits}   Deletions (false-positives): {n_deleted}")

    # ── Build dataset ─────────────────────────────────────────────────────────
    dataset_dir = Path("data/yolo_finetune")
    print(f"\nBuilding YOLO dataset at {dataset_dir}/ ...")
    n_valid = build_dataset(corrections, dataset_dir)

    if n_valid == 0:
        print(
            "\nNo valid training samples found.\n"
            "Ensure the pipeline has run with checkpoint saving enabled "
            "(data/jobs/{job_id}/render.jpg and px_detections.json)."
        )
        sys.exit(1)

    yaml_path = write_data_yaml(dataset_dir)
    print(f"\nDataset complete: {n_valid} image(s).  Config: {yaml_path}")

    if args.dry_run:
        print("Dry-run mode — skipping training.")
        return

    # ── Fine-tune ─────────────────────────────────────────────────────────────
    print(f"\nStarting YOLO fine-tuning  ({args.epochs} epochs, imgsz={args.imgsz})...")
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    model = YOLO(str(weights_path))
    model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        project="ml/runs/finetune",
        name="correction_feedback",
        exist_ok=True,
        verbose=True,
    )

    # ── Save best weights ─────────────────────────────────────────────────────
    best = Path("ml/runs/finetune/correction_feedback/weights/best.pt")
    if best.exists():
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        ts  = int(time.time())
        dst = output_dir / f"yolov11_floorplan_ft_{ts}.pt"
        shutil.copy2(best, dst)
        print(f"\nFine-tuned weights saved: {dst}")
        print("To activate, update YOLO_WEIGHTS_PATH in your .env or rename the file to:")
        print(f"  {output_dir / 'yolov11_floorplan.pt'}")
    else:
        print("\nTraining complete but best.pt not found — check ml/runs/finetune/")


if __name__ == "__main__":
    main()
