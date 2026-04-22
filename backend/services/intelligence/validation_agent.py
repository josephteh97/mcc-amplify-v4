"""
Validation Agent Middleware — DfMA bay-spacing + orphan detection.

Applies grid-level Singapore SS CP 65 rules that don't vary per element:
  - Minimum column spacing (min_bay_mm default: 3000 mm)
  - Maximum column spacing (max_bay_mm default: 12000 mm)
  - Orphan element detection (off_grid + isolated)

Per-element admit/reject judgment (including beam-column join conflicts,
off-grid column deletion, material tagging) now lives in
backend/services/intelligence/admittance/. See VALIDATION_AGENT.skill.md.

Element type vocabulary (aligns with Revit Structure panel):
  "column"            → Structural Column
  "structural_framing"→ Structural Framing (beam/lintel)
  "slab"              → Floor (Revit calls structural floor slabs "Floor")
  "wall"              → Structural Wall

Adds to each detection dict:
  dfma_violations: list[str]  — empty = compliant
  is_dfma_compliant: bool
  is_orphan: bool             — True = off_grid AND isolated

Does NOT remove elements. Does NOT modify coordinates.
"""
from __future__ import annotations

from loguru import logger

from backend.services.intelligence.cross_element_validator import OFF_GRID, ISOLATED

_MIN_BAY_MM = 3000.0
_MAX_BAY_MM = 12000.0


def enforce_rules(
    detections: list[dict],
    grid_info: dict | None = None,
    min_bay_mm: float = _MIN_BAY_MM,
    max_bay_mm: float = _MAX_BAY_MM,
) -> list[dict]:
    """
    Attach DfMA violation flags and orphan status to each detection.
    grid_info used to derive mm spacing between detected columns.
    Accepts a mixed list of columns + structural_framing for cross-type checks.
    """
    for det in detections:
        det["dfma_violations"] = []
        flags = det.get("validation_flags", [])
        det["is_orphan"] = OFF_GRID in flags and ISOLATED in flags

    if grid_info is not None:
        _check_bay_spacing(detections, grid_info, min_bay_mm, max_bay_mm)

    for det in detections:
        det["is_dfma_compliant"] = len(det["dfma_violations"]) == 0

    violations = sum(1 for d in detections if not d["is_dfma_compliant"])
    orphans = sum(1 for d in detections if d["is_orphan"])
    logger.info(
        "ValidationAgent: {} DfMA violations, {} orphan elements (of {} total)",
        violations, orphans, len(detections),
    )
    return detections


def _check_bay_spacing(
    detections: list[dict],
    grid_info: dict,
    min_bay_mm: float,
    max_bay_mm: float,
) -> None:
    """
    Use grid_info spacing to flag grids that violate bay size rules.
    Falls back gracefully if spacing data is unavailable.
    """
    x_spacings: list[float] = grid_info.get("x_spacings_mm", [])
    y_spacings: list[float] = grid_info.get("y_spacings_mm", [])

    violations: list[str] = []
    for sp in x_spacings + y_spacings:
        if sp < min_bay_mm:
            violations.append(f"bay_too_narrow_{sp:.0f}mm")
            logger.warning("Bay spacing %.0f mm < minimum %.0f mm (SS CP 65)", sp, min_bay_mm)
        if sp > max_bay_mm:
            violations.append(f"bay_too_wide_{sp:.0f}mm")
            logger.warning("Bay spacing %.0f mm > maximum %.0f mm (SS CP 65)", sp, max_bay_mm)

    # Bay spacing is a grid-level property — applies to all detections
    if violations:
        for det in detections:
            det["dfma_violations"].extend(violations)
