"""
Context Manager — in-memory conversation history and session state (memory store).

Each user session stores:
  - conversation history (list of {role, content} dicts for Claude)
  - current job context (job_id, status, stage, etc.)

For production, swap the in-memory dict for Redis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
from loguru import logger


@dataclass
class SessionContext:
    user_id: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_active: datetime = field(default_factory=datetime.utcnow)

    # Conversation history in Anthropic multi-turn format
    history: list[dict] = field(default_factory=list)

    # Active job the user is watching
    current_job_id: Optional[str] = None
    job_snapshot: dict = field(default_factory=dict)  # Latest status / stage / detections


class ContextManager:
    """
    Manages per-user session state (memory store).

    thread-safety note: asyncio is single-threaded, so plain dict access is safe.
    """

    MAX_HISTORY      = 40   # keep last 40 messages (20 user + 20 assistant turns)
    SESSION_TTL_MIN  = 60   # expire sessions inactive for this many minutes

    def __init__(self):
        self._sessions: dict[str, SessionContext] = {}

    # ── Session lifecycle ─────────────────────────────────────────────────────

    def get_or_create(self, user_id: str) -> SessionContext:
        self._purge_stale()
        if user_id not in self._sessions:
            self._sessions[user_id] = SessionContext(user_id=user_id)
            logger.debug(f"New chat session: {user_id}")
        ctx = self._sessions[user_id]
        ctx.last_active = datetime.utcnow()
        return ctx

    def _purge_stale(self):
        """Remove sessions that have been inactive beyond SESSION_TTL_MIN."""
        cutoff = datetime.utcnow() - timedelta(minutes=self.SESSION_TTL_MIN)
        stale = [uid for uid, ctx in self._sessions.items()
                 if ctx.last_active < cutoff]
        for uid in stale:
            self._sessions.pop(uid, None)
        if stale:
            logger.debug(f"Purged {len(stale)} stale chat session(s)")

    def delete(self, user_id: str):
        self._sessions.pop(user_id, None)
        logger.debug(f"Chat session removed: {user_id}")

    # ── Conversation history ──────────────────────────────────────────────────

    def get_history(self, user_id: str) -> list[dict]:
        return self.get_or_create(user_id).history

    def add_message(self, user_id: str, role: str, content: str):
        """Append a message to the user's conversation history."""
        ctx = self.get_or_create(user_id)
        ctx.history.append({"role": role, "content": content})
        # Trim to keep only the most recent messages
        if len(ctx.history) > self.MAX_HISTORY:
            ctx.history = ctx.history[-self.MAX_HISTORY:]

    # ── Job context ───────────────────────────────────────────────────────────

    def set_job(self, user_id: str, job_id: str):
        ctx = self.get_or_create(user_id)
        ctx.current_job_id = job_id

    def update_job_snapshot(self, user_id: str, snapshot: dict):
        ctx = self.get_or_create(user_id)
        ctx.job_snapshot.update(snapshot)

    def get_job_snapshot(self, user_id: str) -> dict:
        return self.get_or_create(user_id).job_snapshot

    def get_current_job_id(self, user_id: str) -> Optional[str]:
        return self.get_or_create(user_id).current_job_id
