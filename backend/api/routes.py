"""
API Routes
"""

import asyncio
import json
import os
import time
import uuid
from pathlib import Path
from typing import Optional

# When false (default), human corrections are stored in the DB but never used
# to generate YOLO training data, preventing accidental model corruption.
_YOLO_RETRAIN_ENABLED = os.getenv("YOLO_RETRAIN_ENABLED", "false").lower() == "true"

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from loguru import logger
from pydantic import BaseModel

from api.websocket import manager as ws_manager
from services.core.orchestrator import PipelineOrchestrator
from services.corrections_logger import CorrectionsLogger
from services.job_store import JobStore
from services.revit_client import RevitClient
from utils.file_handler import save_upload_file

# Concurrency limiter — prevents resource exhaustion from parallel pipeline runs
_MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT_JOBS", "3"))
_pipeline_semaphore = asyncio.Semaphore(_MAX_CONCURRENT)

router = APIRouter()

# Persistent SQLite-backed job store with LRU eviction
job_store = JobStore(db_path="data/jobs.db", max_jobs=100)

orchestrator      = PipelineOrchestrator()
revit_client      = RevitClient()
corrections_log   = CorrectionsLogger()


class ProcessRequest(BaseModel):
    project_name: Optional[str] = None


# ── Upload ─────────────────────────────────────────────────────────────────────


@router.post("/upload")
async def upload_floor_plan(file: UploadFile = File(...)):
    """Accept a PDF floor plan and create a processing job."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")

    job_id = str(uuid.uuid4())
    file_path = await save_upload_file(file, job_id)

    job_store.put(job_id, {
        "status":     "uploaded",
        "progress":   0,
        "message":    "File uploaded",
        "filename":   file.filename,
        "created_at": time.time(),
    })
    logger.info(f"Uploaded: {file.filename} → job {job_id}")
    return {"job_id": job_id, "filename": file.filename, "message": "Uploaded successfully"}


# ── Process ────────────────────────────────────────────────────────────────────

@router.post("/process/{job_id}")
async def process_floor_plan(
    job_id: str,
    request: ProcessRequest,
    background_tasks: BackgroundTasks,
):
    """Start the AI pipeline for an uploaded PDF."""
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")

    filename = job.get("filename", f"{job_id}.pdf")
    ext = Path(filename).suffix.lower()
    file_path = f"data/uploads/{job_id}{ext}"

    if not Path(file_path).exists():
        raise HTTPException(404, "Uploaded file not found on disk")

    job_store.update(job_id, {"status": "processing", "progress": 5, "message": "Pipeline starting…"})

    background_tasks.add_task(_run_pipeline_task, job_id, file_path, request.project_name)
    return {"job_id": job_id, "status": "processing", "message": "Processing started"}


async def _run_pipeline_task(job_id: str, file_path: str, project_name: Optional[str]):
    """Background task — runs the full pipeline and pushes updates via WebSocket."""
    async with _pipeline_semaphore:
        await _run_pipeline_task_inner(job_id, file_path, project_name)


async def _run_pipeline_task_inner(job_id: str, file_path: str, project_name: Optional[str]):
    """Actual pipeline work, guarded by the concurrency semaphore."""

    def on_progress(pct: int, msg: str):
        job_store.update(job_id, {"progress": pct, "message": msg})
        asyncio.ensure_future(ws_manager.send_progress(job_id, {
            "type":     "progress",
            "job_id":   job_id,
            "progress": pct,
            "message":  msg,
        }))

    job = job_store.get(job_id) or {}
    pdf_filename = job.get("filename", "")
    try:
        result = await orchestrator.run_pipeline(
            file_path,
            job_id,
            project_name or "Project",
            pdf_filename=pdf_filename,
            progress_callback=on_progress,
        )
        job_store.update(job_id, {"status": "completed", "progress": 100, "result": result})
        await ws_manager.send_progress(job_id, {
            "type":     "completed",
            "job_id":   job_id,
            "progress": 100,
            "message":  "Processing complete — your RVT and glTF files are ready.",
            "result":   result,
        })
    except Exception as e:
        logger.error(f"Pipeline failed for job {job_id}: {e}")
        job_store.update(job_id, {"status": "failed", "progress": -1, "error": str(e)})
        await ws_manager.send_progress(job_id, {
            "type":    "failed",
            "job_id":  job_id,
            "progress": -1,
            "message": f"Processing failed: {e}",
        })


# ── Status ─────────────────────────────────────────────────────────────────────

@router.get("/status/{job_id}")
async def get_status(job_id: str):
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return job


# ── Downloads ──────────────────────────────────────────────────────────────────

@router.get("/download/rvt/{job_id}")
async def download_rvt(job_id: str):
    job = _require_completed(job_id)
    rvt_path = job.get("result", {}).get("files", {}).get("rvt")
    if not rvt_path or not Path(rvt_path).exists():
        raise HTTPException(404, "RVT file not available — Revit server may have been unreachable")
    return FileResponse(rvt_path, media_type="application/octet-stream", filename=Path(rvt_path).name)


@router.get("/download/gltf/{job_id}")
async def download_gltf(job_id: str):
    job = _require_completed(job_id)
    gltf_path = job.get("result", {}).get("files", {}).get("gltf")
    if not gltf_path or not Path(gltf_path).exists():
        raise HTTPException(404, "glTF file not found")
    return FileResponse(gltf_path, media_type="model/gltf-binary", filename=f"{job_id}.glb")


# ── RVT upload & render ────────────────────────────────────────────────────────

@router.post("/upload-rvt")
async def upload_rvt(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """Upload an existing RVT file for Revit rendering."""
    if not file.filename.lower().endswith('.rvt'):
        raise HTTPException(400, "Only .rvt files are supported")

    job_id = str(uuid.uuid4())
    file_path = await save_upload_file(file, job_id)

    job_store.put(job_id, {
        "status":     "uploaded_rvt",
        "progress":   0,
        "message":    "RVT uploaded, rendering queued",
        "filename":   file.filename,
        "file_path":  str(file_path),
        "created_at": time.time(),
    })
    background_tasks.add_task(_run_rvt_render_task, job_id, str(file_path))
    return {"job_id": job_id, "message": "RVT uploaded and rendering started"}


async def _run_rvt_render_task(job_id: str, rvt_path: str):
    try:
        job_store.update(job_id, {"status": "rendering", "progress": 10})
        render_path = await revit_client.render_model(rvt_path, job_id)
        job_store.update(job_id, {
            "status": "completed",
            "progress": 100,
            "result": {"files": {"render": render_path, "rvt": rvt_path}},
        })
    except Exception as e:
        logger.error(f"RVT render failed for job {job_id}: {e}")
        job_store.update(job_id, {"status": "failed", "error": str(e)})


@router.get("/download/render/{job_id}")
async def download_render(job_id: str):
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    render_path = job.get("result", {}).get("files", {}).get("render")
    if not render_path or not Path(render_path).exists():
        raise HTTPException(404, "Render not found")
    return FileResponse(render_path, media_type="image/png")


# ── Chat model info ────────────────────────────────────────────────────────────

@router.get("/chat/models")
async def chat_models():
    """Return available chat AI backends and the current server default."""
    from chat_agent.agent import get_available_models
    return get_available_models()


# ── Human-in-the-loop correction ───────────────────────────────────────────────

_ALLOWED_ELEMENT_TYPES = {"walls", "doors", "windows", "columns", "floors", "ceilings"}
_BLOCKED_FIELDS        = {"id", "level", "top_level", "host_wall_id",
                           "location", "start_point", "end_point", "boundary_points"}

# Per-element-type PATCH schema — must mirror frontend/src/components/EditPanel.jsx
# FIELD_DEFS. Changing this list is a breaking change for the edit UI.
_PATCH_FIELD_SCHEMA: dict[str, set[str]] = {
    "walls":    {"thickness", "height", "material", "is_structural"},
    "columns":  {"width", "depth", "height", "shape", "material"},
    "doors":    {"width", "height", "type_name"},
    "windows":  {"width", "height", "type_name"},
    "floors":   {"thickness"},
    "ceilings": {"thickness"},
}


@router.get("/model/{job_id}/recipe")
async def get_recipe(job_id: str):
    """Return the stored RevitTransaction recipe JSON for a completed job."""
    if not job_store.contains(job_id):
        raise HTTPException(404, "Job not found")
    tx = Path(f"data/models/rvt/{job_id}_transaction.json")
    if not tx.exists():
        raise HTTPException(404, "Recipe not found — pipeline may not have completed")
    with open(tx) as f:
        return json.load(f)


class RecipePatch(BaseModel):
    element_type:  str   # e.g. "walls", "columns"
    element_index: int   # 0-based index into the element array
    changes:       dict  # fields to update, e.g. {"thickness": 300}
    delete:        bool = False


@router.patch("/model/{job_id}/recipe")
async def patch_recipe(job_id: str, patch: RecipePatch):
    """
    Apply a single-element correction to the on-disk recipe, then
    immediately re-export the glTF (fast path, no Revit call).
    """
    if not job_store.contains(job_id):
        raise HTTPException(404, "Job not found")
    if patch.element_type not in _ALLOWED_ELEMENT_TYPES:
        raise HTTPException(400, f"Unknown element_type: {patch.element_type!r}")

    # Schema validation — reject unknown / blocked fields up-front so the
    # frontend gets a clear error instead of silent drops.
    if not patch.delete:
        allowed = _PATCH_FIELD_SCHEMA.get(patch.element_type, set())
        unknown = [k for k in patch.changes if k not in allowed]
        blocked = [k for k in patch.changes if k in _BLOCKED_FIELDS]
        if unknown:
            raise HTTPException(
                400,
                f"Unknown field(s) for {patch.element_type}: {unknown}. "
                f"Allowed: {sorted(allowed)}",
            )
        if blocked:
            raise HTTPException(
                400,
                f"Protected field(s) cannot be patched: {blocked}",
            )

    tx = Path(f"data/models/rvt/{job_id}_transaction.json")
    if not tx.exists():
        raise HTTPException(404, "Recipe not found on disk")

    with open(tx) as f:
        recipe = json.load(f)

    elems = recipe.get(patch.element_type, [])
    if not (0 <= patch.element_index < len(elems)):
        raise HTTPException(400, f"element_index {patch.element_index} out of range "
                                  f"(type '{patch.element_type}' has {len(elems)} elements)")

    # Snapshot the element BEFORE mutation so the logger stores YOLO's original output
    original_element = dict(elems[patch.element_index])

    if patch.delete:
        elems.pop(patch.element_index)
    else:
        for k, v in patch.changes.items():
            elems[patch.element_index][k] = v

    recipe[patch.element_type] = elems
    with open(tx, "w") as f:
        json.dump(recipe, f)

    # Log correction for training flywheel only when explicitly enabled.
    # Default is disabled (YOLO_RETRAIN_ENABLED=false) to prevent the
    # YOLO model from being retrained on incorrect detections.
    if _YOLO_RETRAIN_ENABLED:
        corrections_log.log(
            job_id          = job_id,
            element_type    = patch.element_type,
            element_index   = patch.element_index,
            original_element= original_element,
            changes         = patch.changes,
            is_delete       = patch.delete,
        )

    try:
        await orchestrator.rebuild_gltf(job_id)
    except Exception as e:
        raise HTTPException(500, f"glTF rebuild failed: {e}")

    # Return the patched recipe so the frontend can update state without a
    # follow-up GET round-trip.
    return {
        "status":  "ok",
        "job_id":  job_id,
        "message": "Recipe patched and glTF rebuilt",
        "recipe":  recipe,
    }


@router.post("/rebuild/{job_id}")
async def rebuild_rvt_endpoint(job_id: str, background_tasks: BackgroundTasks):
    """
    Send the (user-corrected) recipe to the Revit server and rebuild the RVT.
    Frontend polls /api/status/{job_id} for completion.
    """
    if not job_store.contains(job_id):
        raise HTTPException(404, "Job not found")
    job_store.update(job_id, {
        "status":   "rebuilding",
        "progress": 10,
        "message":  "Sending corrected model to Revit…",
    })
    background_tasks.add_task(_run_rebuild_task, job_id)
    return {"job_id": job_id, "status": "rebuilding"}


async def _run_rebuild_task(job_id: str):
    try:
        job = job_store.get(job_id) or {}
        pdf_filename = job.get("filename", "")
        rvt_path = await orchestrator.rebuild_rvt(job_id, pdf_filename)
        job_store.update(job_id, {
            "status":   "completed",
            "progress": 100,
            "message":  "Revit rebuild complete",
        })
        job_store.setdefault_nested(job_id, "result", "files", {"rvt": rvt_path})
        await ws_manager.send_progress(job_id, {
            "type":     "completed",
            "job_id":   job_id,
            "progress": 100,
            "message":  "Revit rebuild complete — corrected RVT ready.",
        })
    except Exception as e:
        logger.error(f"Rebuild failed for job {job_id}: {e}")
        job_store.update(job_id, {
            "status":   "completed",
            "progress": 100,
            "message":  f"Revit rebuild failed: {e}",
        })


# ── Agent build (P6: on-demand Claude agent BIM build) ─────────────────────────

@router.post("/agent-build/{job_id}")
async def agent_build_endpoint(job_id: str, background_tasks: BackgroundTasks):
    """
    Run the Claude MCP agent to build a Revit model from an existing completed job's
    recipe.  Useful for re-building without re-running the full pipeline.

    The agent uses the session API (step-by-step family loading and placement)
    rather than the batch build-model call.

    Frontend polls /api/status/{job_id} for progress.
    """
    if not job_store.contains(job_id):
        raise HTTPException(404, "Job not found")

    tx_path = Path(f"data/models/rvt/{job_id}_transaction.json")
    if not tx_path.exists():
        raise HTTPException(404, "Recipe not found — run the full pipeline first")

    job_store.update(job_id, {
        "status":   "agent_building",
        "progress": 5,
        "message":  "Claude agent starting Revit session…",
    })
    background_tasks.add_task(_run_agent_build_task, job_id, str(tx_path))
    return {"job_id": job_id, "status": "agent_building"}


async def _run_agent_build_task(job_id: str, transaction_path: str):
    """Background task: run RevitAgent on an existing transaction JSON."""
    def _on_progress(msg: str):
        job_store.update(job_id, {"message": f"Agent: {msg}"})

    try:
        from agents.revit_agent import RevitAgent
        with open(transaction_path) as f:
            recipe = json.load(f)

        agent  = RevitAgent()
        result = await agent.run(recipe, job_id, on_progress=_on_progress)

        if result["status"] == "done":
            job_store.update(job_id, {
                "status":   "completed",
                "progress": 100,
                "message":  f"Agent built {result['placed_count']} elements in {result['turns']} turns.",
            })
            job_store.setdefault_nested(job_id, "result", "files", {"rvt": result["rvt_path"]})
            await ws_manager.send_progress(job_id, {
                "type":     "completed",
                "job_id":   job_id,
                "progress": 100,
                "message":  f"Agent build complete — {result['placed_count']} elements placed.",
            })
        else:
            raise RuntimeError(result.get("error", "Agent did not produce an RVT"))

    except Exception as e:
        logger.error(f"Agent build failed for job {job_id}: {e}")
        job_store.update(job_id, {
            "status":   "completed",
            "progress": 100,
            "message":  f"Agent build failed: {e}",
        })


# ── Corrections (training flywheel) ────────────────────────────────────────────

@router.get("/corrections/stats")
async def corrections_stats():
    """Return aggregate counts of user corrections by element type."""
    return corrections_log.stats()


@router.get("/corrections/defaults/{element_type}")
async def corrections_defaults(element_type: str):
    """
    Return the firm's most commonly corrected field values for an element type.

    The frontend uses this to show "firm default: 800 mm" hints next to form
    fields so engineers can apply historical corrections in one click.
    Returns {} when there is not enough correction history (< 2 samples).
    """
    return corrections_log.defaults(element_type)


@router.get("/corrections/export")
async def corrections_export(limit: int = 5000):
    """
    Return all logged corrections as JSON.

    Each record contains:
      - original_element: the YOLO/AI output before the user's edit
        (includes bbox, center, confidence — enough to crop the source PDF)
      - changes: the fields the user corrected
      - is_delete: true if the user deleted the element entirely

    Use this endpoint to build a YOLO fine-tuning dataset:
      for row in records:
          crop PDF at row['original_element']['bbox']
          label = row['changes'] or DELETE
    """
    return corrections_log.export(limit=limit)


@router.post("/corrections/yolo-export")
async def corrections_yolo_export(output_dir: str = "data/yolo_training"):
    """
    Generate YOLO-format training samples from all human corrections.
    Disabled by default — set YOLO_RETRAIN_ENABLED=true in .env to enable.
    """
    if not _YOLO_RETRAIN_ENABLED:
        raise HTTPException(
            403,
            "YOLO retraining is disabled. Set YOLO_RETRAIN_ENABLED=true in .env to enable. "
            "The current ml/weights/column-detect.pt is the authoritative model."
        )
    result = corrections_log.export_yolo_training_data(output_dir=output_dir)
    if "error" in result:
        raise HTTPException(500, result["error"])
    return result


# ── Project profile ────────────────────────────────────────────────────────────

_PROFILE_PATH = Path("data/project_profile.json")
_DEFAULT_PROFILE = {
    "building_type":             "commercial",
    "typical_wall_height_mm":    2800.0,
    "typical_wall_thickness_mm": 200.0,
    "typical_column_size_mm":    800.0,
    "floor_to_floor_height_mm":  3000.0,
    "typical_door_width_mm":     900.0,
    "typical_sill_height_mm":    900.0,
    # Revit commit confidence gate — warn user before sending a risky model.
    # gate_block_on_fallback_grid : block on fallback grid detection
    # gate_block_on_scanned_pdf   : block when PDF appears to be scanned
    # gate_low_conf_threshold     : fraction of low-confidence elements that triggers a warning
    "gate_block_on_fallback_grid": True,
    "gate_block_on_scanned_pdf":   True,
    "gate_low_conf_threshold":     0.30,
}


@router.get("/project_profile")
async def get_project_profile():
    """Return the saved project profile, or the system defaults if none saved."""
    if _PROFILE_PATH.exists():
        with open(_PROFILE_PATH) as f:
            return json.load(f)
    return _DEFAULT_PROFILE


class ProjectProfileModel(BaseModel):
    building_type:             str   = "commercial"
    typical_wall_height_mm:    float = 2800.0
    typical_wall_thickness_mm: float = 200.0
    typical_column_size_mm:    float = 800.0
    floor_to_floor_height_mm:  float = 3000.0
    typical_door_width_mm:     float = 900.0
    typical_sill_height_mm:    float = 900.0
    # Revit commit confidence gate
    gate_block_on_fallback_grid: bool  = True
    gate_block_on_scanned_pdf:   bool  = True
    gate_low_conf_threshold:     float = 0.30


@router.post("/project_profile")
async def set_project_profile(profile: ProjectProfileModel):
    """Save the project profile.  Applied as dimension defaults on next pipeline run."""
    _PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_PROFILE_PATH, "w") as f:
        json.dump(profile.model_dump(), f, indent=2)
    logger.info(f"Project profile saved: {profile.model_dump()}")
    return {"status": "ok"}


# ── Revit family manifest ───────────────────────────────────────────────────────

@router.get("/revit/families")
async def get_revit_families():
    """
    Return available Revit families cached from the last successful build.
    Returns empty lists if no build has been completed yet.
    """
    families_path = Path("data/revit_families.json")
    if families_path.exists():
        with open(families_path) as f:
            return json.load(f)
    return {"structural_columns": [], "wall_types": [], "door_families": [], "window_families": []}


# ── RFA family library index ────────────────────────────────────────────────────

@router.get("/revit/library")
async def get_family_library(category: Optional[str] = None, keyword: Optional[str] = None):
    """
    Return the RFA family library index (generated by scripts/scan_family_library.py).
    Optional: ?category=OST_StructuralColumns  ?keyword=concrete
    """
    index_path = Path("data/family_library/index.json")
    if not index_path.exists():
        return {
            "total": 0, "families": [],
            "message": "Library not indexed yet. Run: python scripts/scan_family_library.py",
        }

    with open(index_path) as f:
        index = json.load(f)

    families = index.get("families", [])

    if category:
        families = [f for f in families
                    if f.get("category", "").lower() == category.lower()]
    if keyword:
        kw = keyword.lower()
        families = [
            f for f in families
            if kw in f.get("family_name", "").lower()
            or any(kw in t for t in f.get("tags", []))
            or any(kw in (t.get("type_name") or "").lower() for t in f.get("types", []))
        ]

    return {"total": len(families), "indexed_at": index.get("indexed_at"), "families": families}


# ── Stateful Revit session (MCP agent step-by-step workflow) ────────────────────


class _PlaceRequest(BaseModel):
    family_name: str
    type_name:   str
    x_mm:        float
    y_mm:        float
    z_mm:        float = 0.0
    level:       str   = "Level 0"
    top_level:   Optional[str] = None
    parameters:  Optional[dict] = None


class _SetParamRequest(BaseModel):
    element_id:     str
    parameter_name: str
    value:          object
    value_type:     str = "mm"


@router.post("/revit/session/new")
async def revit_new_session():
    """Open a new Revit document from template. Returns session_id."""
    try:
        return await revit_client.new_session()
    except Exception as e:
        raise HTTPException(502, f"Revit server error: {e}")


@router.post("/revit/session/{session_id}/load-family")
async def revit_load_family(session_id: str, body: dict):
    """Load an .rfa into an open session. Body: { rfa_path }."""
    try:
        rfa_path = body.get("rfa_path") or body.get("windows_rfa_path")
        if not rfa_path:
            raise HTTPException(400, "rfa_path is required")
        return await revit_client.load_family(session_id, rfa_path)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"Revit server error: {e}")


@router.get("/revit/session/{session_id}/families")
async def revit_list_families(session_id: str):
    """List all loaded families in a session."""
    try:
        return await revit_client.list_families(session_id)
    except Exception as e:
        raise HTTPException(502, f"Revit server error: {e}")


@router.post("/revit/session/{session_id}/place")
async def revit_place_instance(session_id: str, req: _PlaceRequest):
    """Place a single FamilyInstance in an open session."""
    try:
        return await revit_client.place_instance(
            session_id,
            req.family_name, req.type_name,
            req.x_mm, req.y_mm, req.z_mm,
            req.level, req.top_level, req.parameters,
        )
    except Exception as e:
        raise HTTPException(502, f"Revit server error: {e}")


@router.post("/revit/session/{session_id}/set-param")
async def revit_set_param(session_id: str, req: _SetParamRequest):
    """Set a parameter on a placed element in a session."""
    try:
        return await revit_client.set_parameter(
            session_id, req.element_id,
            req.parameter_name, req.value, req.value_type,
        )
    except Exception as e:
        raise HTTPException(502, f"Revit server error: {e}")


@router.get("/revit/session/{session_id}/state")
async def revit_session_state(session_id: str):
    """Return current state of an open session (levels, families, placed elements)."""
    try:
        return await revit_client.get_session_state(session_id)
    except Exception as e:
        raise HTTPException(502, f"Revit server error: {e}")


@router.post("/revit/session/{session_id}/close")
async def revit_close_session(session_id: str):
    """Close an open session without saving."""
    try:
        return await revit_client.close_session(session_id)
    except Exception as e:
        raise HTTPException(502, f"Revit server error: {e}")


# ── Health ─────────────────────────────────────────────────────────────────────

@router.get("/health")
async def health():
    return {"status": "healthy"}


# ── Helper ─────────────────────────────────────────────────────────────────────

def _require_completed(job_id: str) -> dict:
    """Return job data if completed; raise HTTPException otherwise."""
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job.get("status") != "completed":
        raise HTTPException(400, "Job not completed yet")
    return job
