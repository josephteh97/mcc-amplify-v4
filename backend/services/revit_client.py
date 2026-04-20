"""
Client to communicate with the Windows Revit 2023 Add-in server.

Two integration modes:

1. HTTP mode (default — add-in running inside Revit)
   ─────────────────────────────────────────────────
   The RevitModelBuilderAddin.dll runs as a Revit IExternalApplication.
   It listens on http://localhost:5000 (or the address in WINDOWS_REVIT_SERVER).
   POST /build-model → returns binary .rvt

2. Macro / file-drop mode (fallback — Macro Manager workflow)
   ─────────────────────────────────────────────────────────
   The Python server writes the transaction JSON to a shared directory
   (REVIT_SHARED_DIR, default C:\\RevitOutput on the Windows side, mapped via
   a network share or SSH tunnel on Linux).  The Revit macro reads pending.json,
   builds the model, and writes {job_id}.rvt + {job_id}.done.
   The Python server then reads the .rvt and returns it to the browser.

   Set REVIT_MODE=file in the environment to use this mode.
"""

import asyncio
import httpx
import json as _json
import os
import re
from pathlib import Path
from loguru import logger


def _rvt_stem(pdf_filename: str, job_id: str) -> str:
    if pdf_filename:
        stem = Path(pdf_filename).stem
        safe = re.sub(r"_+", "_", re.sub(r"[^\w\-]", "_", stem)).strip("_")
        if safe:
            return f"{safe}_{job_id}"
    return job_id


def _print_revit_warnings(job_id: str, warnings: list) -> None:
    """
    Print Revit build warnings prominently to the terminal so they are never
    missed in a scrolling log.  Always prints a summary line so the operator
    knows warnings were checked (even when count is zero).
    """
    sep = "=" * 60
    if not warnings:
        print(f"\n{sep}")
        print(f"  REVIT WARNINGS  job={job_id}  →  0 warnings (clean build)")
        print(f"{sep}\n", flush=True)
        logger.info(f"Revit build clean — no warnings for job {job_id}.")
        return

    print(f"\n{sep}")
    print(f"  REVIT WARNINGS  job={job_id}  →  {len(warnings)} warning(s)")
    print(sep)
    for i, w in enumerate(warnings, 1):
        print(f"  [{i}] {w}")
    print(f"{sep}\n", flush=True)

    logger.warning(
        f"Revit {len(warnings)} warning(s) for job {job_id}: "
        + " | ".join(warnings)
    )


class RevitClient:
    """Client for the Windows Revit 2023 API service."""

    def __init__(self):
        self.server_url  = os.getenv("WINDOWS_REVIT_SERVER", "http://localhost:5000")
        self.api_key     = os.getenv("REVIT_SERVER_API_KEY", "")
        self.timeout     = int(os.getenv("REVIT_TIMEOUT", "300"))
        self.mode        = os.getenv("REVIT_MODE", "http").lower()  # "http" | "file"
        # Shared directory visible from this Linux host (e.g. /mnt/revit_output)
        self.shared_dir  = Path(os.getenv("REVIT_SHARED_DIR", "/mnt/revit_output"))

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    async def check_health(self) -> bool:
        """Return True if the Windows Revit server (HTTP mode) is reachable."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.server_url}/health", timeout=5.0
                )
                return response.status_code == 200
        except Exception as e:
            logger.error(f"Cannot connect to Revit server: {e}")
            return False

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------

    async def build_model(
        self, transaction_path: str, job_id: str, pdf_filename: str = ""
    ) -> tuple:
        """
        Send a transaction JSON to the Revit server and retrieve the .rvt file.

        Returns:
            (rvt_path: str, warnings: list[str])
            warnings — Revit build warnings captured by the IFailuresPreprocessor;
                       empty list in file-drop mode or when no warnings occurred.
        """
        if self.mode == "file":
            rvt_path = await self._build_via_file_drop(transaction_path, job_id, pdf_filename)
            return rvt_path, []
        return await self._build_via_http(transaction_path, job_id, pdf_filename)

    # ── HTTP mode ─────────────────────────────────────────────────────────────

    _MAX_RETRIES = 3          # attempts before giving up
    _RETRY_BASE_S = 25        # base back-off: 25 s, 50 s — matches C# 20 s pending-wait

    async def _build_via_http(
        self, transaction_path: str, job_id: str, pdf_filename: str = ""
    ) -> str:
        """POST the transaction JSON to the add-in's HTTP server (with retry)."""
        with open(transaction_path, "r") as f:
            transaction_json = f.read()

        last_exc: Exception = RuntimeError("No attempt made")
        for attempt in range(1, self._MAX_RETRIES + 1):
            try:
                logger.info(
                    f"Sending build request to Revit server for job {job_id} "
                    f"(attempt {attempt}/{self._MAX_RETRIES})"
                )
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        f"{self.server_url}/build-model",
                        json={
                            "job_id":           job_id,
                            "pdf_filename":     pdf_filename,
                            "transaction_json": transaction_json,
                        },
                        headers={"X-API-Key": self.api_key},
                    )

                if response.status_code != 200:
                    raise Exception(
                        f"Revit server error {response.status_code}: {response.text}"
                    )

                # Verify the response is a real .rvt binary (Revit OLE header 0xD0CF…)
                content = response.content
                if len(content) < 8 or content[:4] != b"\xd0\xcf\x11\xe0":
                    raise Exception(
                        f"Revit server returned {len(content)} bytes that do not look "
                        f"like a valid .rvt file (got: {content[:64]!r}). "
                        "Check that the Add-in is loaded inside Revit 2023."
                    )

                rvt_path = f"data/models/rvt/{_rvt_stem(pdf_filename, job_id)}.rvt"
                Path(rvt_path).parent.mkdir(parents=True, exist_ok=True)
                with open(rvt_path, "wb") as f:
                    f.write(content)

                # Parse Revit build warnings from custom response header.
                # The C# WarningCollector silently dismisses dialogs and
                # encodes collected warning texts as a JSON array here.
                warnings: list = []
                raw_hdr = response.headers.get("x-revit-warnings", "[]")
                try:
                    parsed = _json.loads(raw_hdr)
                    if isinstance(parsed, list):
                        warnings = [str(w) for w in parsed]
                except Exception:
                    pass

                _print_revit_warnings(job_id, warnings)

                # Extract and cache available Revit families for the family
                # manifest endpoint (GET /api/revit/families).
                raw_fam = response.headers.get("x-revit-families", "")
                if raw_fam:
                    try:
                        families = _json.loads(raw_fam)
                        fam_path = Path("data/revit_families.json")
                        fam_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(fam_path, "w") as fj:
                            _json.dump(families, fj, indent=2)
                        col_count = len(families.get("structural_columns", []))
                        logger.info(
                            f"Revit family manifest cached: "
                            f"{col_count} column type(s) available."
                        )
                    except Exception as fex:
                        logger.debug(f"Family manifest parse failed: {fex}")

                logger.info(f"RVT file saved to {rvt_path} ({len(content):,} bytes)")
                return rvt_path, warnings

            except Exception as exc:
                last_exc = exc
                if attempt < self._MAX_RETRIES:
                    wait = self._RETRY_BASE_S * attempt
                    logger.warning(
                        f"Revit request failed (attempt {attempt}/{self._MAX_RETRIES}): "
                        f"{exc} — retrying in {wait} s"
                    )
                    await asyncio.sleep(wait)

        raise last_exc

    # ── File-drop / macro mode ────────────────────────────────────────────────

    async def _build_via_file_drop(
        self, transaction_path: str, job_id: str, pdf_filename: str = ""
    ) -> str:
        """
        Write pending.json to the shared directory and wait for the Revit macro
        to write the .rvt output.  Poll every 2 s for up to self.timeout seconds.
        """
        pending_path = self.shared_dir / "pending.json"
        done_path    = self.shared_dir / f"{job_id}.done"
        rvt_src_path = self.shared_dir / f"{job_id}.rvt"

        self.shared_dir.mkdir(parents=True, exist_ok=True)

        # Copy transaction JSON as pending.json
        with open(transaction_path, "r") as f:
            data = f.read()
        pending_path.write_text(data, encoding="utf-8")

        logger.info(
            f"[file-drop] Wrote pending.json — waiting for macro to build {job_id}.rvt"
        )

        # Poll for completion marker written by the macro
        elapsed = 0
        interval = 2
        while elapsed < self.timeout:
            await asyncio.sleep(interval)
            elapsed += interval
            if done_path.exists() or rvt_src_path.exists():
                break
        else:
            raise TimeoutError(
                f"Revit macro did not produce {job_id}.rvt within {self.timeout} s. "
                "Open Macro Manager and run 'BuildRevitModelFromJSON'."
            )

        # Clean up completion marker
        try:
            done_path.unlink(missing_ok=True)
        except Exception:
            pass

        rvt_path = f"data/models/rvt/{_rvt_stem(pdf_filename, job_id)}.rvt"
        Path(rvt_path).parent.mkdir(parents=True, exist_ok=True)

        with open(rvt_src_path, "rb") as fin, open(rvt_path, "wb") as fout:
            fout.write(fin.read())

        logger.info(f"RVT file saved to {rvt_path}")
        return rvt_path

    # ------------------------------------------------------------------
    # Session-based API (step-by-step MCP agent workflow)
    # ------------------------------------------------------------------
    # These methods wrap the new /session/* endpoints added in App.cs P2.
    # Each call is a thin HTTP wrapper — all reasoning lives in the agent.

    async def new_session(self) -> dict:
        """
        Open a new Revit document from the template.
        Returns { session_id, levels, template, message }.
        """
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(f"{self.server_url}/session/new", json={})
        r.raise_for_status()
        data = r.json()
        logger.info(f"Revit session opened: {data.get('session_id')}")
        return data

    async def load_family(self, session_id: str, rfa_path: str) -> dict:
        """
        Load an .rfa family file into an open session document.
        rfa_path must be a Windows path visible to the Revit machine.
        Returns { family_name, already_loaded, types: [...] }.
        """
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                f"{self.server_url}/session/{session_id}/load-family",
                json={"rfa_path": rfa_path},
            )
        r.raise_for_status()
        data = r.json()
        logger.info(
            f"Session {session_id}: loaded '{data.get('family_name')}' "
            f"({'already' if data.get('already_loaded') else 'newly'} in project)"
        )
        step_warns = data.get("warnings") or []
        if step_warns:
            _print_revit_warnings(f"{session_id}/load-family", step_warns)
        return data

    async def list_families(self, session_id: str) -> dict:
        """Return all families currently loaded in the session document."""
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(f"{self.server_url}/session/{session_id}/families")
        r.raise_for_status()
        return r.json()

    async def place_instance(
        self,
        session_id: str,
        family_name: str,
        type_name: str,
        x_mm: float,
        y_mm: float,
        z_mm: float = 0.0,
        level: str = "Level 0",
        top_level: str | None = None,
        parameters: dict | None = None,
    ) -> dict:
        """
        Place one FamilyInstance in the session document.
        Returns { element_id, placed: { category, family_name, type_name, ... } }.
        """
        payload = {
            "family_name": family_name,
            "type_name":   type_name,
            "x_mm": x_mm, "y_mm": y_mm, "z_mm": z_mm,
            "level": level,
        }
        if top_level:
            payload["top_level"] = top_level
        if parameters:
            payload["parameters"] = parameters
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                f"{self.server_url}/session/{session_id}/place",
                json=payload,
            )
        r.raise_for_status()
        data = r.json()
        logger.debug(
            f"Session {session_id}: placed {family_name}::{type_name} "
            f"@ ({x_mm:.0f},{y_mm:.0f}) → elemId={data.get('element_id')}"
        )
        step_warns = data.get("warnings") or []
        if step_warns:
            _print_revit_warnings(f"{session_id}/place:{family_name}", step_warns)
        return data

    async def set_parameter(
        self,
        session_id: str,
        element_id: str,
        parameter_name: str,
        value,
        value_type: str = "mm",
    ) -> dict:
        """
        Set a parameter on a placed element.
        value_type: "mm" (converted to internal feet) | "raw" | "string" | "int" | "id"
        """
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(
                f"{self.server_url}/session/{session_id}/set-param",
                json={
                    "element_id":     element_id,
                    "parameter_name": parameter_name,
                    "value":          value,
                    "value_type":     value_type,
                },
            )
        r.raise_for_status()
        return r.json()

    async def get_session_state(self, session_id: str) -> dict:
        """Return current levels, loaded families, and placed elements."""
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(f"{self.server_url}/session/{session_id}/state")
        r.raise_for_status()
        return r.json()

    async def export_session(
        self, session_id: str, job_id: str, keep_open: bool = False
    ) -> tuple[str, bool]:
        """
        Save the session document as .rvt and retrieve the bytes.
        Returns (rvt_path, session_still_open).
        By default the session is closed after export.
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(
                f"{self.server_url}/session/{session_id}/export",
                json={"keep_open": keep_open},
            )

        if r.status_code != 200:
            raise Exception(f"Session export failed {r.status_code}: {r.text}")

        content = r.content
        if len(content) < 8 or content[:4] != b"\xd0\xcf\x11\xe0":
            raise Exception(
                f"Session export returned {len(content)} bytes that don't look like .rvt"
            )

        rvt_path = f"data/models/rvt/{job_id}.rvt"
        Path(rvt_path).parent.mkdir(parents=True, exist_ok=True)
        with open(rvt_path, "wb") as f:
            f.write(content)

        still_open = r.headers.get("x-session-closed", "true").lower() != "true"
        logger.info(
            f"Session {session_id} exported → {rvt_path} ({len(content):,} bytes); "
            f"session {'still open' if still_open else 'closed'}"
        )
        return rvt_path, still_open

    async def list_rfa_files(
        self, folder: str = r"C:\MyDocuments\3. Revit Family Files"
    ) -> list:
        """
        List all .rfa files in a folder on the Revit machine.

        Returns a list of absolute Windows paths, e.g.
        ["C:\\MyDocuments\\3. Revit Family Files\\Columns\\Foo.rfa", ...]
        Returns [] if the folder doesn't exist or the server is unreachable.
        """
        import urllib.parse
        encoded = urllib.parse.quote(folder, safe="")
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(f"{self.server_url}/list-rfa?folder={encoded}")
        r.raise_for_status()
        return r.json().get("files", [])

    async def get_element_parameters(self, session_id: str, element_id: str) -> dict:
        """
        Return all parameters of a placed element so the agent can discover
        the correct names before calling set_parameter.

        Returns { element_id, count, parameters: [{name, storage_type,
                  is_read_only, value_mm?, value_int?, value_str?, value_id?}] }
        """
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(
                f"{self.server_url}/session/{session_id}/get-params",
                json={"element_id": element_id},
            )
        r.raise_for_status()
        return r.json()

    async def wall_join_all(self, session_id: str) -> dict:
        """
        Enable wall joins at both ends of every Wall in the session.
        Fixes display gaps and incorrect intersections at T-junctions / corners.

        Returns { ok, walls_total, walls_joined, warnings }.
        """
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                f"{self.server_url}/session/{session_id}/wall-join", json={}
            )
        r.raise_for_status()
        data = r.json()
        logger.info(
            f"Session {session_id}: wall-join-all — "
            f"{data.get('walls_joined')}/{data.get('walls_total')} walls joined"
        )
        step_warns = data.get("warnings") or []
        if step_warns:
            _print_revit_warnings(f"{session_id}/wall-join", step_warns)
        return data

    async def export_floor_plan_view(self, session_id: str) -> bytes:
        """
        Export the first floor plan view from the session as a PNG image.
        Returns raw PNG bytes (150 DPI, 2048 px wide).

        Call this BEFORE export_session (the document must still be open).
        """
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(
                f"{self.server_url}/session/{session_id}/export-view", json={}
            )
        if r.status_code != 200:
            raise Exception(f"export-view failed {r.status_code}: {r.text}")
        if not r.content or r.content[:4] != b"\x89PNG":
            raise Exception(
                f"export-view returned {len(r.content)} bytes that do not look like PNG"
            )
        logger.info(
            f"Session {session_id}: floor plan view exported ({len(r.content):,} bytes)"
        )
        return r.content

    async def close_session(self, session_id: str) -> dict:
        """Close an open session without saving."""
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(
                f"{self.server_url}/session/{session_id}/close", json={}
            )
        r.raise_for_status()
        logger.info(f"Session {session_id} closed.")
        return r.json()

    # ------------------------------------------------------------------
    # Render (unchanged from original)
    # ------------------------------------------------------------------

    async def render_model(self, rvt_path: str, job_id: str) -> str:
        """Send an existing .rvt to the Windows server for rendering."""
        logger.info(f"Sending render request for job {job_id}")

        with open(rvt_path, "rb") as f:
            rvt_content = f.read()

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.server_url}/render-model",
                data={"job_id": job_id},
                files={"file": (f"{job_id}.rvt", rvt_content, "application/octet-stream")},
                headers={"X-API-Key": self.api_key},
            )

        if response.status_code != 200:
            raise Exception(f"Revit rendering error: {response.text}")

        render_path = f"data/models/render/{job_id}.png"
        Path(render_path).parent.mkdir(parents=True, exist_ok=True)

        with open(render_path, "wb") as f:
            f.write(response.content)

        logger.info(f"Render saved to {render_path}")
        return render_path
