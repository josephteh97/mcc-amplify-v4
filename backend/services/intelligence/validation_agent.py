"""
Validation Agent Middleware — DfMA rule enforcement, orphan and join-conflict detection.

Applies Singapore SS CP 65 structural rules to validated detections:
  - Minimum column spacing (min_bay_mm default: 3000 mm)
  - Maximum column spacing (max_bay_mm default: 12000 mm)
  - Grid alignment enforcement (columns at known grid intersections)
  - Orphan element detection (columns flagged off_grid + isolated)
  - Beam–column proximity: a beam whose centre lies within one beam-width
    of a column centre will cause a Revit "Cannot keep elements joined" error
    at transaction commit → flagged as beam_column_join_conflict

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
import math

from loguru import logger

from backend.services.intelligence.cross_element_validator import OFF_GRID, ISOLATED

_MIN_BAY_MM = 3000.0
_MAX_BAY_MM = 12000.0
# A beam whose centre is within this multiple of its own short-axis width from a column
# centre will produce a Revit geometry-join error at transaction commit.
_BEAM_COLUMN_JOIN_CLEARANCE_FACTOR = 1.5


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

    _check_beam_column_proximity(detections)

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


def _check_beam_column_proximity(detections: list[dict]) -> None:
    """
    Flag structural_framing elements whose centre is too close to a column centre.

    When a beam and a column overlap significantly in plan, Revit's auto-join
    algorithm fails with "Cannot keep elements joined" (37 unignorable errors).
    A beam whose centre is within CLEARANCE_FACTOR × beam_short_dim of any
    column centre is flagged as beam_column_join_conflict so the geometry
    generator can omit it rather than send it to Revit.

    Uses pixel-space centres (not yet converted to mm) because this check
    runs before geometry generation.
    """
    columns  = [d for d in detections if d.get("type") == "column"]
    framings = [d for d in detections if d.get("type") == "structural_framing"]

    if not columns or not framings:
        return

    for beam in framings:
        bc = beam.get("center", [0.0, 0.0])
        bbox = beam.get("bbox", [0.0, 0.0, 0.0, 0.0])
        if len(bbox) < 4:
            continue

        # Short-axis dimension in pixels — proxy for clearance zone
        short_dim = min(abs(bbox[2] - bbox[0]), abs(bbox[3] - bbox[1]))
        clearance = short_dim * _BEAM_COLUMN_JOIN_CLEARANCE_FACTOR

        for col in columns:
            cc = col.get("center", [0.0, 0.0])
            dist = math.hypot(bc[0] - cc[0], bc[1] - cc[1])
            if dist < clearance:
                beam["dfma_violations"].append("beam_column_join_conflict")
                # conflict_column_center is read by debug_overlay to draw a line
                # back to the offender; without it the renderer can't indicate which.
                beam["conflict_column_center"] = list(cc)
                logger.warning(
                    "Framing {} @ px({:.0f},{:.0f}) bbox={} conflicts with "
                    "column {} @ px({:.0f},{:.0f}) — dist {:.1f}px < clearance "
                    "{:.1f}px → excluded (would cause Revit join error)",
                    beam.get("id", "?"), bc[0], bc[1],
                    [f"{v:.0f}" for v in bbox],
                    col.get("id", "?"), cc[0], cc[1],
                    dist, clearance,
                )
                break   # one conflict is enough to flag the beam
