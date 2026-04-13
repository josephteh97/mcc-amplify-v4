"""
ChatAgent — conversational AI that monitors the pipeline and assists users.

Architecture:
  Message Router     → classify intent (status / technical / troubleshoot / admin)
  Context Manager    → per-user conversation history + job snapshot (memory store)
  Pipeline Observer  → subscribe to pipeline events → proactive notifications
  LLM               → NVIDIA NIM (DeepSeek V3.1, free) or Google Gemini API
                       Controlled by CHAT_MODEL_BACKEND env var (default: nvidia_nim)
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime
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

_BACKEND = os.getenv("CHAT_MODEL_BACKEND", "nvidia_nim").lower().strip()


# ── NVIDIA NIM client (OpenAI-compatible, free tier) ─────────────────────────

_nvidia_client = None
_NVIDIA_MODEL  = "deepseek-ai/deepseek-v3.1"
_NVIDIA_BASE   = "https://integrate.api.nvidia.com/v1"

def _init_nvidia_client():
    from utils.api_keys import get_nvidia_api_key
    api_key = get_nvidia_api_key()
    if not api_key:
        raise ValueError(
            "NVIDIA API key not found — add it to backend/nvidia_key.txt "
            "or set NVIDIA_API_KEY in backend/.env  (sign up free at https://build.nvidia.com)"
        )
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package not installed. Run: pip install openai")
    client = OpenAI(api_key=api_key, base_url=_NVIDIA_BASE)
    logger.info(f"✓ Chat agent using NVIDIA NIM ({_NVIDIA_MODEL})")
    return client


# ── Gemini client ─────────────────────────────────────────────────────────────

_gemini_client = None
_GEMINI_MODEL  = "gemini-2.5-flash"

def _init_gemini_client():
    try:
        from google import genai
        from utils.api_keys import get_google_api_key
    except ImportError as e:
        raise ImportError(f"google-genai package not installed: {e}")
    api_key = get_google_api_key()
    if not api_key or api_key == "[placeholder]":
        raise ValueError("GOOGLE_API_KEY not set — add it to backend/.env")
    client = genai.Client(api_key=api_key)
    logger.info(f"✓ Chat agent using Gemini API ({_GEMINI_MODEL})")
    return client


# ── Initialise ALL available backends so the user can switch freely ───────────

try:
    _nvidia_client = _init_nvidia_client()
except Exception as e:
    logger.warning(f"NVIDIA NIM unavailable ({e})")

try:
    _gemini_client = _init_gemini_client()
except Exception as e:
    logger.warning(f"Gemini unavailable ({e})")

# Adjust the default if the configured backend failed but the other is up
if _BACKEND == "nvidia_nim" and _nvidia_client is None:
    if _gemini_client is not None:
        logger.warning("NVIDIA NIM not ready — default falling back to Gemini")
        _BACKEND = "gemini_api"
    else:
        logger.error("No AI backends available — chat agent will return errors")
elif _BACKEND == "gemini_api" and _gemini_client is None:
    if _nvidia_client is not None:
        logger.warning("Gemini not ready — default falling back to NVIDIA NIM")
        _BACKEND = "nvidia_nim"
    else:
        logger.error("No AI backends available — chat agent will return errors")


# ── Public helpers ─────────────────────────────────────────────────────────────

_MODEL_META = {
    "nvidia_nim": {"display_name": "DeepSeek V3.1", "provider": "NVIDIA NIM"},
    "gemini_api": {"display_name": "Gemini 2.5 Flash", "provider": "Google"},
}


def get_available_models() -> dict:
    """Return the list of configured AI backends and which default is active."""
    models = []
    for backend, meta in _MODEL_META.items():
        client = _nvidia_client if backend == "nvidia_nim" else _gemini_client
        models.append({
            "backend":      backend,
            "display_name": meta["display_name"],
            "provider":     meta["provider"],
            "available":    client is not None,
        })
    return {"models": models, "default": _BACKEND}


_MODEL_DISPLAY = _MODEL_META.get(_BACKEND, {}).get("display_name", _BACKEND)


# ── System prompt ──────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are an AI assistant for Amplify AI — a system that converts PDF floor plans \
into professional 3D BIM models (Revit .RVT + glTF/GLB).

Your role:
1. Monitor the 7-stage processing pipeline and explain what each stage does.
2. Answer user questions about their upload: status, progress, detected elements, scale, errors.
3. Proactively notify users of warnings (low confidence, large file, missing scale annotation).
4. Troubleshoot issues and suggest improvements.
5. Guide users through the platform.

Pipeline stages:
  Stage 1 — Security & size check
  Stage 2 — Dual-track PDF processing (Track A: vector geometry, Track B: raster rendering + YOLO)
  Stage 3 — Hybrid fusion (aligning ML detections with precise vector geometry)
  Stage 4 — Structural grid detection (derives real-world scale from grid lines and dimension annotations; scale text like "1:100" is intentionally ignored as unreliable)
  Stage 5 — Semantic AI analysis (Claude/Gemini/local model validates and enriches detected elements)
  Stage 6 — 3D geometry generation (grid-snapped mm coordinates → parametric 3D solids)
  Stage 7 — BIM export (Revit server → .RVT; trimesh → .GLB)

Output formats:
  • glTF (.glb) — lightweight 3D web format for browser visualisation
  • RVT (.rvt)  — native Revit format, fully editable, requires Revit

Be concise, friendly, and specific. When you have live job data, reference actual values.
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

    # ── Message handling ───────────────────────────────────────────────────────

    async def handle_message(
        self, user_id: str, message: str, context_data: dict
    ) -> str:
        job_id = context_data.get("job_id")
        if job_id:
            self.link_job(user_id, job_id)

        self.context.add_message(user_id, "user", message)

        # Per-call backend override from the frontend model selector.
        # Falls back to the server-side default if the requested backend
        # has no client initialised (key missing / quota exceeded).
        requested = context_data.get("model", "").lower().strip()
        if requested == "nvidia_nim" and _nvidia_client is not None:
            call_backend = "nvidia_nim"
        elif requested == "gemini_api" and _gemini_client is not None:
            call_backend = "gemini_api"
        else:
            call_backend = _BACKEND   # server default

        intent   = route(message)
        enriched = self._build_user_content(message, user_id, intent)
        history  = self.context.get_history(user_id)

        try:
            if call_backend == "nvidia_nim":
                reply = await self._call_nvidia(enriched, history)
            else:
                reply = await self._call_gemini(enriched, history)
        except Exception as exc:
            logger.error(f"Chat LLM error ({call_backend}): {exc}")
            reply = "Sorry, I had trouble reaching the AI service. Please try again."

        self.context.add_message(user_id, "assistant", reply)
        return reply

    # ── LLM backends ──────────────────────────────────────────────────────────

    async def _call_nvidia(self, enriched: str, history: list) -> str:
        """Call NVIDIA NIM (OpenAI-compatible) with chat history."""
        messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
        for turn in history[:-1]:   # exclude current user turn (already in enriched)
            role = "user" if turn["role"] == "user" else "assistant"
            messages.append({"role": role, "content": turn["content"]})
        messages.append({"role": "user", "content": enriched})

        response = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: _nvidia_client.chat.completions.create(
                model=_NVIDIA_MODEL,
                messages=messages,
                max_tokens=800,
                temperature=0.7,
            ),
        )
        return response.choices[0].message.content.strip()

    async def _call_gemini(self, enriched: str, history: list) -> str:
        """Call Google Gemini API."""
        from google.genai import types as genai_types

        contents = []
        for turn in history[:-1]:
            role = "user" if turn["role"] == "user" else "model"
            contents.append({"role": role, "parts": [{"text": turn["content"]}]})
        contents.append({"role": "user", "parts": [{"text": enriched}]})

        response = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: _gemini_client.models.generate_content(
                model=_GEMINI_MODEL,
                contents=contents,
                config=genai_types.GenerateContentConfig(
                    system_instruction=_SYSTEM_PROMPT,
                    max_output_tokens=800,
                    temperature=0.7,
                ),
            ),
        )
        return response.text.strip()

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
