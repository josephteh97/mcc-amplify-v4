"""Decision primitives shared across rule modules."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

ADMIT          = "admit"
ADMIT_WITH_FIX = "admit_with_fix"
REJECT         = "reject"


@dataclass
class Decision:
    action:   str                      # "admit" | "admit_with_fix" | "reject"
    reason:   str                      # short human-readable tag
    signals:  dict[str, Any] = field(default_factory=dict)   # scored signals (for overlay/audit)
    bbox_override:  tuple[float, float, float, float] | None = None   # (x1,y1,x2,y2) px
    metadata:       dict[str, Any] = field(default_factory=dict)      # e.g. {"material": "rc"}


def admit(reason: str, metadata: dict[str, Any] | None = None, **signals: Any) -> Decision:
    return Decision(action=ADMIT, reason=reason, signals=dict(signals), metadata=metadata or {})


def reject(reason: str, **signals: Any) -> Decision:
    return Decision(action=REJECT, reason=reason, signals=dict(signals))


def admit_with_fix(
    reason: str,
    bbox_override: tuple[float, float, float, float] | None = None,
    metadata: dict[str, Any] | None = None,
    **signals: Any,
) -> Decision:
    return Decision(
        action=ADMIT_WITH_FIX,
        reason=reason,
        signals=dict(signals),
        bbox_override=bbox_override,
        metadata=metadata or {},
    )
