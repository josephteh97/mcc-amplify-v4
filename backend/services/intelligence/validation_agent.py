"""
Validation Agent Middleware — DfMA rule enforcement, orphan detection.

Applies Singapore SS CP 65 structural rules to validated detections:
  - Minimum column spacing (min_bay_mm default: 3000 mm)
  - Maximum column spacing (max_bay_mm default: 12000 mm)
  - Grid alignment enforcement (columns at known grid intersections)
  - Orphan element detection (columns flagged off_grid + isolated)
  - Beam placement: beams must sit at the TOP of columns, flush with
    the column top level line on the same floor (beam_not_at_column_top)
  - Slab placement: slabs/floors must sit at the BOTTOM of the floor,
    on the level line (slab_not_at_level)

Adds to each detection dict:
  dfma_violations: list[str]  — empty = compliant
  is_dfma_compliant: bool
  is_orphan: bool             — True = off_grid AND isolated

Does NOT remove elements. Does NOT modify coordinates.
"""
from __future__ import annotations

from loguru import logger

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
    """
    for det in detections:
        det["dfma_violations"] = []
        det["is_orphan"] = (
            "off_grid" in det.get("validation_flags", []) and
            "isolated" in det.get("validation_flags", [])
        )

    if grid_info is not None:
        _check_bay_spacing(detections, grid_info, min_bay_mm, max_bay_mm)

    _check_beam_placement(detections)
    _check_slab_placement(detections)

    for det in detections:
        det["is_dfma_compliant"] = len(det["dfma_violations"]) == 0

    violations = sum(1 for d in detections if not d["is_dfma_compliant"])
    orphans = sum(1 for d in detections if d["is_orphan"])
    logger.info(
        "ValidationAgent: %d DfMA violations, %d orphan elements (of %d total)",
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


def _check_beam_placement(detections: list[dict]) -> None:
    """
    Validate that beams are placed at the TOP of columns on the same floor.

    Each beam's bottom elevation must be flush with the top-of-column level
    line for its floor.  Beams that fail are flagged with
    ``beam_not_at_column_top``.
    """
    # Build a lookup: floor -> column top elevation(s)
    column_tops_by_floor: dict[str, list[float]] = {}
    for det in detections:
        if det.get("label") == "column":
            floor = det.get("floor")
            top_el = det.get("top_elevation_mm")
            if floor is not None and top_el is not None:
                column_tops_by_floor.setdefault(floor, []).append(top_el)

    for det in detections:
        if det.get("label") != "beam":
            continue

        floor = det.get("floor")
        bottom_el = det.get("bottom_elevation_mm")

        if floor is None or bottom_el is None:
            # Insufficient metadata — skip without flagging
            continue

        col_tops = column_tops_by_floor.get(floor, [])
        if not col_tops:
            # No columns on this floor to compare against
            continue

        # Beam bottom must match at least one column top on the same floor
        if not any(abs(bottom_el - ct) < 1e-3 for ct in col_tops):
            det["dfma_violations"].append("beam_not_at_column_top")
            logger.warning(
                "Beam '%s' bottom elevation %.1f mm not flush with any "
                "column top on floor %s",
                det.get("id", "?"), bottom_el, floor,
            )


def _check_slab_placement(detections: list[dict]) -> None:
    """
    Validate that slabs sit at the level line (bottom of the floor).

    Each slab's elevation must match the level-line elevation for its
    floor.  Slabs that fail are flagged with ``slab_not_at_level``.
    """
    # Build a lookup: floor -> level-line elevation
    level_by_floor: dict[str, float] = {}
    for det in detections:
        if det.get("label") == "level_line":
            floor = det.get("floor")
            elev = det.get("elevation_mm")
            if floor is not None and elev is not None:
                level_by_floor[floor] = elev

    for det in detections:
        if det.get("label") not in ("slab", "floor"):
            continue

        floor = det.get("floor")
        bottom_el = det.get("bottom_elevation_mm")

        if floor is None or bottom_el is None:
            continue

        level_el = level_by_floor.get(floor)
        if level_el is None:
            # No level line detected for this floor — skip
            continue

        if abs(bottom_el - level_el) >= 1e-3:
            det["dfma_violations"].append("slab_not_at_level")
            logger.warning(
                "Slab '%s' bottom elevation %.1f mm not at level line "
                "%.1f mm on floor %s",
                det.get("id", "?"), bottom_el, level_el, floor,
            )
