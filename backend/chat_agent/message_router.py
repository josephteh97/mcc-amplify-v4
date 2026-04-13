"""
Message Router — classifies incoming user messages and returns the appropriate
handler category so the chat agent can tailor its system prompt context.

Categories:
  status      — "how is my job?", "what's the progress?"
  technical   — "what does stage 3 do?", "what is glTF?"
  troubleshoot — "why did it fail?", "error in my plan"
  admin       — "queue length", "server load"
  general     — anything else
"""

from __future__ import annotations

import re

# Simple keyword maps — no external dependency needed for routing
_STATUS_PATTERNS = re.compile(
    r'\b(status|progress|how.*going|how.*long|eta|done|finish|complete|stage|current)\b',
    re.IGNORECASE,
)
_TECHNICAL_PATTERNS = re.compile(
    r'\b(what is|what does|explain|difference|gltf|rvt|revit|yolo|dpi|scale|vector|'
    r'stage \d|track [ab]|fusion|semantic|geometry)\b',
    re.IGNORECASE,
)
_TROUBLESHOOT_PATTERNS = re.compile(
    r'\b(fail|error|wrong|broken|issue|problem|crash|why did|can\'t|cannot|not working|retry)\b',
    re.IGNORECASE,
)
_ADMIN_PATTERNS = re.compile(
    r'\b(queue|server|load|active jobs|how many|memory|cpu|uptime)\b',
    re.IGNORECASE,
)


def route(message: str) -> str:
    """Return the handler category for *message*."""
    if _TROUBLESHOOT_PATTERNS.search(message):
        return "troubleshoot"
    if _STATUS_PATTERNS.search(message):
        return "status"
    if _TECHNICAL_PATTERNS.search(message):
        return "technical"
    if _ADMIN_PATTERNS.search(message):
        return "admin"
    return "general"
