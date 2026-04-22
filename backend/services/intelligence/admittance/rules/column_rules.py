"""
Admittance rule for columns.

Ports the legacy off-grid-deletion directive: columns must sit on a grid
intersection — a column flagged off_grid by CrossElementValidator is
almost always a YOLO false positive (column-cap hatch, circled note, etc.)
and is rejected. "Place wrongly is worse than missing one column."

No geometry fixes are applied; columns are only admit/reject.
"""
from __future__ import annotations

from backend.services.intelligence.admittance.context import ElementContext
from backend.services.intelligence.admittance.scoring import Decision, admit, reject
from backend.services.intelligence.cross_element_validator import OFF_GRID


def judge(det: dict, siblings: list[dict], ctx: ElementContext) -> Decision:
    flags = det.get("validation_flags") or []
    if OFF_GRID in flags:
        return reject("off_grid", flags=list(flags))
    return admit("on_grid")
