"""
ChatAgent — conversational AI that monitors the pipeline and assists users.

Architecture:
  Message Router     → classify intent (status / technical / troubleshoot / admin)
  Context Manager    → per-user conversation history + job snapshot (memory store)
  Pipeline Observer  → subscribe to pipeline events → proactive notifications
  LLM               → Local Ollama models (no API key required)
                         qwen3_vl  → qwen3-vl:2b       (default)
                         gemma3_it → gemma3:4b-it-qat   (alternative)
                       Controlled by CHAT_MODEL_BACKEND env var (default: qwen3_vl)
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from datetime import datetime

# Qwen3 and some other reasoning models emit <think>…</think> blocks even when
# think=false is set. Compiled once here so the per-request hot path is cheap.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import WebSocket

# Load backend/.env so CHAT_MODEL_BACKEND is available regardless of import order
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")
from loguru import logger

from .context_manager import ContextManager
from .message_router import route
from .pipeline_observer import PipelineObserver, observer as global_observer


# ── Backend selection ─────────────────────────────────────────────────────────

_OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# Model registry: backend key → Ollama model tag
_OLLAMA_MODELS = {
    "qwen3_vl":  "qwen3-vl:2b",
    "gemma3_it": "gemma3:4b-it-qat",
}

_BACKEND = os.getenv("CHAT_MODEL_BACKEND", "qwen3_vl").lower().strip()


# ── Ollama initialisation ─────────────────────────────────────────────────────

_chat_session  = None   # requests.Session shared across all calls
_available_models: dict[str, str] = {}  # backend_key → model_tag (installed only)


def _init_ollama_chat():
    import requests

    try:
        r = requests.get(f"{_OLLAMA_URL}/api/tags", timeout=5)
        r.raise_for_status()
        installed = {m["name"] for m in r.json().get("models", [])}
    except Exception as exc:
        raise ConnectionError(f"Ollama server not reachable: {exc}")

    available = {k: v for k, v in _OLLAMA_MODELS.items() if v in installed}
    if not available:
        raise ConnectionError(
            f"No chat models found in Ollama. "
            f"Run: ollama pull qwen3-vl:2b && ollama pull gemma3:4b-it-qat"
        )
    return requests.Session(), available


try:
    _chat_session, _available_models = _init_ollama_chat()
    logger.info(
        "✓ Chat agent using Ollama models: {}",
        list(_available_models.values()),
    )
except Exception as exc:
    logger.warning(f"Ollama chat unavailable ({exc}) — chat responses will be disabled")

if _BACKEND not in _available_models and _available_models:
    _BACKEND = next(iter(_available_models))
    logger.warning(f"Requested chat model not available — defaulting to {_BACKEND}")


# ── Public helpers ─────────────────────────────────────────────────────────────

_MODEL_META = {
    "qwen3_vl":  {"display_name": "Qwen3-VL 2B",      "provider": "Ollama"},
    "gemma3_it": {"display_name": "Gemma3 4B IT QAT",  "provider": "Ollama"},
}


def get_available_models() -> dict:
    """Return available Ollama chat backends and the active default."""
    models = []
    for backend, meta in _MODEL_META.items():
        models.append({
            "backend":      backend,
            "display_name": meta["display_name"],
            "provider":     meta["provider"],
            "available":    backend in _available_models,
        })
    return {"models": models, "default": _BACKEND}


_MODEL_DISPLAY = _MODEL_META.get(_BACKEND, {}).get("display_name", _BACKEND)


# ── System prompt ──────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are an expert BIM engineer and Revit specialist embedded in Amplify AI — \
a system that converts PDF structural floor plans into native Revit (.RVT) BIM models and glTF web previews.

════════════════════════════════════════════════════════════
  YOUR EXPERTISE
════════════════════════════════════════════════════════════

You have deep knowledge of:

REVIT STRUCTURAL ELEMENTS & RULES
• Structural Column   — placed vertically between two levels; must sit exactly at a grid intersection;
                        family CJY_Concrete-Rectangular-Column (rect) or CJY_RC Round Column (circular);
                        minimum section 200 mm; width/depth snapped to standard sizes
                        (200 250 300 350 400 450 500 600 700 800 900 1000 1200 mm).
• Structural Framing  — beam placed along a Line between two XYZ points; family loaded from
                        OST_StructuralFraming category; default section 800×800 mm;
                        beam centre Z = top-of-column elevation (= storey height, default 2800 mm);
                        MUST NOT overlap a column centre within 1.5× the beam short-axis dimension
                        or Revit raises "Cannot keep elements joined" (37 errors in one transaction).
• Floor (Slab)        — structural floor slab; created with Floor.Create(); boundary polygon in mm;
                        elevation 0 mm (ground slab) or storey-height for upper slabs;
                        thickness default 200 mm.  Called "Floor" in the Revit Structure ribbon.
• Structural Wall     — created with Wall.Create() on a Line; height = storey height; thickness 200 mm default.
• Stair               — StairsType; requires run width, riser height, tread depth.
• Lift / Elevator     — modelled as a shaft opening bounded by structural walls.

REVIT LEVELS & GRIDS
• Levels are horizontal datums: "Level 0" = 0 mm (ground floor), "Level 1" = 2800–4000 mm (first floor).
• All structural columns span from Level 0 (base) to Level 1 (top).
• Grids are named reference lines: vertical lines numbered 1, 2, 3 … (left→right);
  horizontal lines lettered A, B, C … or AA, BB … (bottom→top).
• Every column MUST sit at a grid intersection; off-grid columns are rejected by the ValidationAgent.

COMMON REVIT ERRORS & HOW TO PREVENT THEM
• "Cannot keep elements joined" — beam centre too close to column centre in plan;
  auto-resolved by WarningCollector (SkipElements resolution) + prevented upstream by
  beam_column_join_conflict flag in ValidationAgent.
• "Identical instances in the same place" — two columns snapped to the same grid point;
  prevented by post-snap deduplication (rounds to 0.1 mm, first-seen wins).
• "ExternalEvent rejected (status: Pending)" — Revit has a modal dialog open or the
  previous ExternalEvent hasn't executed yet; fixed by 20 s pending-wait loop in C#.
• "No structural column family found" — the Revit template doesn't have CJY families loaded;
  LoadConcreteColumnFamilies() searches for them and logs a warning if absent.
• "Revit model build timed out" — BuildModel took > 120 s; reduce element count or check
  if Revit is processing something else.
• "Failed to deserialise transaction JSON" — malformed recipe; check the transaction JSON
  for null values in Location, StartPoint, or EndPoint fields.
• Floor/slab boundary loop is not closed — boundary_points polygon must have ≥ 3 points
  and the last edge must close back to the first point (handled automatically in CreateSlabs).

DfMA RULES — SINGAPORE SS CP 65
• Minimum structural bay: 3000 mm (bay_too_narrow flagged).
• Maximum structural bay: 12000 mm (bay_too_wide flagged).
• Columns must align with grid lines (off_grid flagged if > 80 px from any grid line).
• Isolated columns (no neighbour within 200 px) are flagged; off_grid + isolated = orphan → deleted.
• Beam-column join clearance: beam short-axis centre must be ≥ 1.5× beam short-dim from column centre.

COORDINATE SYSTEM
• All recipe coordinates are in millimetres (mm).
• X increases left→right (matches PDF grid numbering 1, 2, 3 …).
• Y increases bottom→top (Revit convention; PDF image Y is flipped during _px_to_world conversion).
• Z = 0 for ground-floor elements; Z = storey_height for top-of-column / beam datum.
• C# Add-in converts mm → Revit internal feet: 1 ft = 304.8 mm, constant MM = 1/304.8.

FAMILY NAMING CONVENTION
• Rectangular column : CJY_Concrete-Rectangular-Column / Symbol "{Width}x{Depth}mm" (e.g. 800x800mm)
• Circular column    : CJY_RC Round Column / Symbol "Φ{Diameter}" (e.g. Φ600)
• Beam              : first OST_StructuralFraming family found; prefers CJY_ prefix;
                      symbol name "Beam{MaxDim}x{MinDim}mm" (e.g. Beam800x800mm)
• Floor slab        : default FloorType from the Revit template (metric mm template preferred)

════════════════════════════════════════════════════════════
  PIPELINE STAGES
════════════════════════════════════════════════════════════

  Stage 1  — Security & size check (150 MP pixel budget, 300 DPI for A0/ANSI-E)
  Stage 2  — Source data: VectorProcessor (PDF paths + text) + StreamingProcessor (raster), parallel
  Stage 3  — 7 parallel detection agents:
               Grid Agent         — structural grid from PDF vector geometry (authoritative mm scale)
               Column Agent       — YOLO tiling inference (column-detect.pt)
               Framing Agent      — YOLO tiling inference (structural-framing-detect.pt, squareness filter OFF)
               Slab Agent         — YOLO tiling inference (slab-detect.pt, squareness filter OFF)
               Wall / Stair / Lift agents — stubs (return [] until models trained)
  Stage 4  — Detection Merger: HybridFusionPipeline snaps YOLO to vector; grid pixel alignment
  Stage 4c — Intelligence Middleware:
               TypeResolver        — cv2 contour analysis → circular / rectangular / L-shape
               CrossElementValidator — IoU overlap, grid distance, isolation checks
               ValidationAgent     — DfMA bay spacing, beam-column join-conflict detection
  Stage 5  — Semantic AI (Ollama aisingapore/Gemma-SEA-LION-v4-4B-VL):
               column annotation, materials, building type inference
  Stage 6  — Geometry Generation: px → world mm → _snap_to_nearest_grid → Revit recipe JSON
               (beams flagged beam_column_join_conflict are excluded)
  Stage 6.5— BIM Enrichment + Dedup: merge intelligence metadata; deduplicate snapped columns
  Stage 7a — RVT Export → Windows Revit C# Add-in (TCP :5000);
               WarningCollector auto-resolves join errors; AI correction loop (max 3 rounds)
  Stage 7b — glTF Export → .glb (Z-up to Y-up rotation); columns, framing, walls, slabs rendered

════════════════════════════════════════════════════════════
  YOUR ROLE
════════════════════════════════════════════════════════════

1. Monitor the pipeline and explain each stage in plain language.
2. When a Revit error occurs, diagnose the root cause using your structural knowledge and
   tell the user exactly what happened and how to fix it.
3. Answer questions about detected elements: count, positions, dimensions, confidence scores.
4. Proactively warn about conditions that will cause Revit errors (beam-column conflicts,
   off-grid columns, missing families, modal dialogs blocking the ExternalEvent).
5. Explain DfMA compliance status — which elements violate SS CP 65 and why.
6. Guide the user through the full workflow from PDF upload to opening the .rvt in Revit.

Be concise, precise, and structural-engineering-aware.
When you have live job data, reference actual element counts, grid labels, and mm values.
"""


class ChatAgent:
    """
    Full chat agent: routes messages, manages context, calls the LLM backend,
    and sends proactive pipeline notifications via WebSocket.
    """

    def __init__(self, pipeline_observer: Optional[PipelineObserver] = None):
        self.context  = ContextManager()
        self.observer = pipeline_observer or global_observer

        self._sessions: dict[str, WebSocket]  = {}
        self._job_to_user: dict[str, str]     = {}

        self._register_pipeline_handlers()

    # ── Session management ─────────────────────────────────────────────────────

    async def on_connect(self, user_id: str, websocket: WebSocket):
        self._sessions[user_id] = websocket
        await self._send(user_id, (
            f"Hi! I'm your Amplify AI assistant powered by {_MODEL_DISPLAY}. "
            "I'll keep you updated as your floor plan processes. "
            "Feel free to ask me anything!"
        ), auto=True)

    def on_disconnect(self, user_id: str):
        self._sessions.pop(user_id, None)
        self.context.delete(user_id)

    def link_job(self, user_id: str, job_id: str):
        self.context.set_job(user_id, job_id)
        self._job_to_user[job_id] = user_id

    async def send_reply(self, user_id: str, text: str) -> None:
        """Send a direct reply to a user via the current active session.

        Routes through _send so a mid-generation reconnect delivers the reply
        to the current socket rather than the stale one held by the caller.
        """
        await self._send(user_id, text)

    # ── Message handling ───────────────────────────────────────────────────────

    async def handle_message(
        self, user_id: str, message: str, context_data: dict
    ) -> str:
        job_id = context_data.get("job_id")
        if job_id:
            self.link_job(user_id, job_id)

        self.context.add_message(user_id, "user", message)

        # Per-call backend override from the frontend model selector.
        requested = context_data.get("model", "").lower().strip()
        call_backend = requested if requested in _available_models else _BACKEND

        intent   = route(message)
        enriched = self._build_user_content(message, user_id, intent)
        history  = self.context.get_history(user_id)

        try:
            reply = await self._call_ollama_chat(enriched, history, call_backend)
        except Exception as exc:
            logger.error(f"Chat LLM error ({call_backend}): {exc}")
            reply = "Sorry, I had trouble reaching the AI service. Please try again."

        self.context.add_message(user_id, "assistant", reply)
        return reply

    # ── LLM backend ───────────────────────────────────────────────────────────

    async def _call_ollama_chat(
        self, enriched: str, history: list, backend_key: str
    ) -> str:
        """Call local Ollama /api/chat with full conversation history."""
        if _chat_session is None:
            raise RuntimeError("Ollama session not initialised")

        model = _available_models.get(backend_key)
        if model is None:
            raise ValueError(f"Unknown chat backend: {backend_key!r}")

        # history[:-1]: exclude the current user turn — it's already merged into enriched
        messages = (
            [{"role": "system", "content": _SYSTEM_PROMPT}]
            + [{"role": t["role"], "content": t["content"]} for t in history[:-1]]
            + [{"role": "user", "content": enriched}]
        )

        payload = {
            "model":    model,
            "messages": messages,
            "stream":   False,
            "think":    False,   # disable qwen3 reasoning mode — adds latency with no benefit for chat
            "options":  {"num_predict": 800, "temperature": 0.7},
        }
        response = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: _chat_session.post(
                f"{_OLLAMA_URL}/api/chat", json=payload, timeout=120
            ),
        )
        response.raise_for_status()
        text = response.json().get("message", {}).get("content", "")
        return _THINK_RE.sub("", text).strip()

    # ── Context builder ────────────────────────────────────────────────────────

    def _build_user_content(self, message: str, user_id: str, intent: str) -> str:
        snapshot = self.context.get_job_snapshot(user_id)
        job_id   = self.context.get_current_job_id(user_id)

        state_block = ""
        if snapshot:
            state_block = f"\n\nCurrent job state:\n{json.dumps(snapshot, indent=2, default=str)}"
        elif job_id:
            state_block = f"\n\nJob ID: {job_id} (no snapshot data yet)"

        return f"[Intent: {intent}]\n\nUser: {message}{state_block}"

    # ── Proactive sending ──────────────────────────────────────────────────────

    async def _send(self, user_id: str, message: str, auto: bool = False, priority: str = "normal"):
        ws = self._sessions.get(user_id)
        if ws is None:
            return
        try:
            await ws.send_json({
                "type": "agent_message",
                "message": message,
                "metadata": {
                    "auto_generated": auto,
                    "priority": priority,
                    "timestamp": datetime.utcnow().isoformat(),
                },
            })
        except Exception as exc:
            logger.warning(f"Could not push message to {user_id}: {exc}")

    def _user_for_job(self, job_id: str) -> Optional[str]:
        return self._job_to_user.get(job_id)

    # ── Pipeline observer handlers ─────────────────────────────────────────────

    def _register_pipeline_handlers(self):

        @self.observer.on("stage_started")
        async def on_stage_started(job_id: str, stage: int, data: dict):
            user_id = self._user_for_job(job_id)
            if not user_id:
                return
            self.context.update_job_snapshot(user_id, {"current_stage": stage, "status": "processing"})
            await self._send(user_id, f"Stage {stage} started: {data.get('stage_name', '')}", auto=True)

        @self.observer.on("stage_completed")
        async def on_stage_completed(job_id: str, stage: int, output: dict):
            user_id = self._user_for_job(job_id)
            if not user_id:
                return
            self.context.update_job_snapshot(user_id, {"last_completed_stage": stage})
            if stage == 2:
                walls   = len(output.get("walls", []))
                doors   = len(output.get("doors", []))
                windows = len(output.get("windows", []))
                await self._send(
                    user_id,
                    f"Element detection done! Found {walls} walls, {doors} doors, {windows} windows.",
                    auto=True,
                )
            else:
                await self._send(user_id, f"Stage {stage} complete.", auto=True)

        @self.observer.on("warning")
        async def on_warning(job_id: str, warning_type: str, details: dict):
            user_id = self._user_for_job(job_id)
            if not user_id:
                return
            self.context.update_job_snapshot(user_id, {"last_warning": warning_type})
            if warning_type == "low_scale_confidence":
                msg = (
                    f"Low scale confidence ({details.get('confidence', '?')}%). "
                    "Tip: add a clear scale label like \"Scale: 1:100\" to your PDF."
                )
            elif warning_type == "large_file":
                msg = f"Large floor plan detected — DPI reduced to {details.get('dpi', '?')}."
            else:
                msg = f"Warning: {warning_type} — {details.get('message', '')}"
            await self._send(user_id, msg, auto=True, priority="high")

        @self.observer.on("error")
        async def on_error(job_id: str, error_type: str, details: dict):
            user_id = self._user_for_job(job_id)
            if not user_id:
                return
            self.context.update_job_snapshot(user_id, {"status": "failed", "last_error": error_type})
            await self._send(
                user_id,
                f"Processing error — {details.get('message', error_type)}. I'm here to help troubleshoot.",
                auto=True,
                priority="critical",
            )

        @self.observer.on("job_completed")
        async def on_job_completed(job_id: str, result: dict):
            user_id = self._user_for_job(job_id)
            if not user_id:
                return
            self.context.update_job_snapshot(user_id, {"status": "completed", "result": result})
            stats = result.get("stats", {})
            await self._send(
                user_id,
                f"Job complete! Elements: {stats.get('element_count', '?')}, "
                f"Scale: {stats.get('scale', '?')} ({stats.get('scale_source', '?')}). "
                "Your RVT and glTF files are ready to download.",
                auto=True,
                priority="high",
            )
