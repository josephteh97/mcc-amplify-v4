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

Does NOT remove elements (see remove_outside_grid for culling).
Does NOT modify coordinates.
"""
from __future__ import annotations

import os

from loguru import logger

from backend.services.intelligence.cross_element_validator import OFF_GRID, ISOLATED

_MIN_BAY_MM = 3000.0
_MAX_BAY_MM = 12000.0
_GRID_BOUNDS_TOLERANCE_PX: float = float(os.getenv("GRID_BOUNDS_TOLERANCE_PX", "50"))


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


def remove_outside_grid(
    detections: list[dict],
    grid_info: dict,
    tolerance_px: float = _GRID_BOUNDS_TOLERANCE_PX,
) -> tuple[list[dict], list[str]]:
    """
    Drop detections whose centre falls outside the outer-grid rectangle.

    Grid rect (pixels) =
      [min(x_lines_px) - tol, min(y_lines_px) - tol,
       max(x_lines_px) + tol, max(y_lines_px) + tol]

    `tolerance_px` absorbs drafting slop — a column sitting ON the outermost
    grid line can land a few pixels outside due to YOLO jitter.

    For each detection, use `center` if present, otherwise the bbox midpoint.
    Detections missing both fields are kept (something else will catch them).

    Returns (kept, actions) where actions is a human-readable reason per
    removal — mirrors recipe_sanitizer's return shape.

    If grid_info is missing or has fewer than 2 lines on either axis, the
    grid rect isn't well-defined — no-op and log a warning.

    TODO: polygon-aware culling — L-shaped / notched footprints have cells
    inside the outer grid rect but outside the building envelope. Extend with
    a boundary-polygon test once building-outline detection lands.
    """
    x_lines = (grid_info or {}).get("x_lines_px") or []
    y_lines = (grid_info or {}).get("y_lines_px") or []

    if len(x_lines) < 2 or len(y_lines) < 2:
        logger.warning(
            "remove_outside_grid: grid rect undefined "
            "(x_lines={}, y_lines={}) — no-op",
            len(x_lines), len(y_lines),
        )
        return detections, []

    x_lo = min(x_lines) - tolerance_px
    x_hi = max(x_lines) + tolerance_px
    y_lo = min(y_lines) - tolerance_px
    y_hi = max(y_lines) + tolerance_px

    kept: list[dict] = []
    actions: list[str] = []

    for det in detections:
        center = det.get("center")
        if center and len(center) >= 2:
            cx, cy = float(center[0]), float(center[1])
        else:
            bbox = det.get("bbox")
            if not bbox or len(bbox) < 4:
                kept.append(det)
                continue
            cx = (float(bbox[0]) + float(bbox[2])) / 2.0
            cy = (float(bbox[1]) + float(bbox[3])) / 2.0

        if x_lo <= cx <= x_hi and y_lo <= cy <= y_hi:
            kept.append(det)
        else:
            actions.append(
                f"{det.get('type', '?')} id={det.get('id', '?')} "
                f"@({cx:.0f},{cy:.0f}) outside grid rect "
                f"[{x_lo:.0f},{y_lo:.0f}]-[{x_hi:.0f},{y_hi:.0f}]"
            )

    if actions:
        logger.info(
            "ValidationAgent: removed {} detection(s) outside grid rect",
            len(actions),
        )
    return kept, actions


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
