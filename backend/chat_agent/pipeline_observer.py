"""
Pipeline Observer — event pub/sub system that bridges pipeline stages to the chat agent.

Usage:
    observer = PipelineObserver()

    @observer.on("stage_completed")
    async def handler(job_id, stage, data): ...

    # Emit from orchestrator
    await observer.emit("stage_completed", job_id="abc", stage=3, data={...})
"""

import asyncio
from collections import defaultdict
from typing import Callable, Any
from loguru import logger


class PipelineObserver:
    """
    Lightweight event emitter for pipeline lifecycle events.

    Supported event names:
        stage_started       — a processing stage has begun
        stage_completed     — a processing stage finished successfully
        element_detected    — YOLO/vector found architectural elements
        warning             — non-fatal issue worth surfacing to the user
        error               — pipeline failure
        job_completed       — the full job finished
    """

    def __init__(self):
        self._listeners: dict[str, list[Callable]] = defaultdict(list)

    # ── Registration ──────────────────────────────────────────────────────────

    def on(self, event: str):
        """Decorator: register an async handler for *event*."""
        def decorator(fn: Callable):
            self._listeners[event].append(fn)
            return fn
        return decorator

    def subscribe(self, event: str, handler: Callable):
        """Register a handler programmatically."""
        self._listeners[event].append(handler)

    # ── Emission ──────────────────────────────────────────────────────────────

    async def emit(self, event: str, **kwargs: Any):
        """Fire all handlers registered for *event*, passing **kwargs."""
        handlers = self._listeners.get(event, [])
        for handler in handlers:
            try:
                await handler(**kwargs)
            except Exception as exc:
                logger.error(f"PipelineObserver handler error on '{event}': {exc}")

    # ── Convenience emitters (called by the orchestrator) ─────────────────────

    async def stage_started(self, job_id: str, stage: int, stage_name: str):
        await self.emit("stage_started", job_id=job_id, stage=stage, data={"stage_name": stage_name})

    async def stage_completed(self, job_id: str, stage: int, output: dict):
        await self.emit("stage_completed", job_id=job_id, stage=stage, output=output)

    async def element_detected(self, job_id: str, element_type: str, count: int):
        await self.emit("element_detected", job_id=job_id, element_type=element_type, count=count)

    async def warn(self, job_id: str, warning_type: str, details: dict):
        await self.emit("warning", job_id=job_id, warning_type=warning_type, details=details)

    async def error(self, job_id: str, error_type: str, details: dict):
        await self.emit("error", job_id=job_id, error_type=error_type, details=details)

    async def job_completed(self, job_id: str, result: dict):
        await self.emit("job_completed", job_id=job_id, result=result)


# Global singleton shared by the orchestrator and the chat agent
observer = PipelineObserver()
