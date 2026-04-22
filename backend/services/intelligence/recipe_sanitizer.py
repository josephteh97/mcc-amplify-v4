"""
Recipe Sanitizer — deterministic pre-export cleanup to prevent known Revit errors.

Runs on the Revit recipe *before* it is written to transaction.json and sent to
the Windows machine.  No AI involved.  All fixes are geometric and rule-based.

Passes (in order):
  1. snap_and_filter_framing — snap beam endpoints to the nearest column's
                               insertion point, then TRIM each endpoint inward
                               by the column's half-dimension along the beam
                               axis so the beam body stops at the column face
                               (not its centre — which would otherwise clash
                               half-column-width into the column body).
                               REMOVE beams whose endpoints don't both land on
                               a column (would float), whose endpoints snapped
                               to the same column, whose post-snap axis
                               deviates beyond AXIS_TOLERANCE_MM from X/Y
                               (non-colinear columns), or whose span collapses
                               below MIN_BEAM_MM after face-trim.
  2. clamp_column_min_size   — ensure column width/depth >= COL_MIN_MM (200 mm).
                               Revit auto-deletes families below this threshold.

Thresholds are tunable via environment variables (defaults match SS CP 65 practice).
"""
from __future__ import annotations

import math
import os
from loguru import logger

_MIN_BEAM_MM:       float = float(os.getenv("MIN_BEAM_MM",       "500"))
_SNAP_BUFFER_MM:    float = float(os.getenv("SNAP_BUFFER_MM",    "150"))
_COL_MIN_MM:        float = float(os.getenv("COL_MIN_MM",        "200"))
# After snapping both endpoints to column centres, the beam axis should be
# either horizontal (dy≈0) or vertical (dx≈0). Allow small slop so near-colinear
# columns (within this tolerance) still produce a valid beam.
_AXIS_TOLERANCE_MM: float = float(os.getenv("AXIS_TOLERANCE_MM", "50"))


def sanitize_recipe(recipe: dict) -> tuple[dict, list[str]]:
    """
    Apply all sanitization passes to *recipe* in-place.

    Returns (recipe, actions).  actions is empty when nothing needed fixing.
    """
    col_boxes = _col_centers(recipe)

    recipe, a1 = _snap_and_filter_framing(recipe, col_boxes)
    recipe, a2 = _clamp_column_min_size(recipe)

    actions = a1 + a2
    if actions:
        logger.info("RecipeSanitizer: {} fix(es) applied before export", len(actions))
        for a in actions:
            logger.debug("  • {}", a)
    else:
        logger.debug("RecipeSanitizer: recipe clean — no pre-export fixes needed")

    return recipe, actions


def _col_centers(recipe: dict) -> list[tuple[float, float, float, float]]:
    """Return [(cx, cy, half_w, half_d), ...] for every column."""
    result = []
    for col in recipe.get("columns", []):
        loc = col.get("location", {})
        result.append((
            float(loc.get("x", 0.0)),
            float(loc.get("y", 0.0)),
            float(col.get("width",  800.0)) / 2.0,
            float(col.get("depth",  800.0)) / 2.0,
        ))
    return result


def _dist2d(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x2 - x1, y2 - y1)


_ENDPOINT_KEYS = ("start_point", "end_point")


def _reject(actions: list[str], index: int, reason: str) -> None:
    actions.append(f"framing[{index}] removed — {reason}")


def _snap_and_filter_framing(
    recipe: dict,
    centers: list[tuple[float, float, float, float]],
) -> tuple[dict, list[str]]:
    """
    Single pass over structural_framing. Snaps each endpoint within
    (column half-width + _SNAP_BUFFER_MM) to the nearest column centre, trims
    endpoints inward to the column face so the beam body doesn't clash into
    the column, then rejects beams that fail any of:
      - an endpoint didn't snap (floating / visible gap)
      - both endpoints snapped to the same column
      - post-snap axis is diagonal beyond _AXIS_TOLERANCE_MM (non-colinear columns)
      - span < _MIN_BEAM_MM after face-trim (columns physically overlap / too close)
    """
    framing = recipe.get("structural_framing", [])
    kept:    list[dict] = []
    actions: list[str]  = []

    if not centers or not framing:
        return recipe, actions

    for i, beam in enumerate(framing):
        snap_actions: list[str] = []
        # Map endpoint key → the column it snapped to: (cx, cy, half_w, half_d).
        snapped: dict[str, tuple[float, float, float, float]] = {}

        for pt_key in _ENDPOINT_KEYS:
            pt = beam.get(pt_key)
            if not isinstance(pt, dict):
                continue

            px = float(pt.get("x", 0.0))
            py = float(pt.get("y", 0.0))
            best_col:  tuple[float, float, float, float] | None = None
            best_dist: float = float("inf")

            for cx, cy, hw, hd in centers:
                # Fast axis pre-reject — skips the sqrt for distant columns
                if abs(px - cx) > hw + _SNAP_BUFFER_MM or abs(py - cy) > hd + _SNAP_BUFFER_MM:
                    continue
                d = _dist2d(px, py, cx, cy)
                if d < best_dist:
                    best_dist = d
                    best_col  = (cx, cy, hw, hd)

            if best_col:
                cx, cy, _, _ = best_col
                pt["x"] = cx
                pt["y"] = cy
                snapped[pt_key] = best_col
                snap_actions.append(
                    f"framing[{i}].{pt_key} snapped to column @ "
                    f"({cx:.0f}, {cy:.0f}) mm  [was {best_dist:.0f} mm away]"
                )

        missing = [k for k in _ENDPOINT_KEYS if k not in snapped]
        if missing:
            _reject(actions, i,
                f"{', '.join(missing)} not within {_SNAP_BUFFER_MM:.0f} mm "
                f"of any column (would float in model)")
            continue

        sp_col = snapped["start_point"]
        ep_col = snapped["end_point"]
        if sp_col[0] == ep_col[0] and sp_col[1] == ep_col[1]:
            _reject(actions, i, "both endpoints snapped to the same column")
            continue

        sp, ep = beam["start_point"], beam["end_point"]
        dx = ep["x"] - sp["x"]
        dy = ep["y"] - sp["y"]

        if min(abs(dx), abs(dy)) > _AXIS_TOLERANCE_MM:
            _reject(actions, i,
                f"diagonal beam after snap (dx={dx:.0f}mm, dy={dy:.0f}mm > "
                f"tolerance {_AXIS_TOLERANCE_MM:.0f}mm); endpoint columns are not colinear")
            continue

        # Trim endpoints from column centre → column face along the beam axis,
        # so the beam body ends flush with the column face instead of extruding
        # half-column-width into the column body.
        axis_is_x = abs(dx) >= abs(dy)
        if axis_is_x:
            sp_half = sp_col[2]   # start column's half-width
            ep_half = ep_col[2]
            direction = 1.0 if dx > 0 else -1.0
            sp["x"] += direction * sp_half
            ep["x"] -= direction * ep_half
        else:
            sp_half = sp_col[3]   # start column's half-depth
            ep_half = ep_col[3]
            direction = 1.0 if dy > 0 else -1.0
            sp["y"] += direction * sp_half
            ep["y"] -= direction * ep_half

        length = math.hypot(ep["x"] - sp["x"], ep["y"] - sp["y"])
        # If trim overshoots (columns physically overlap along beam axis) the
        # primary-axis delta flips sign — catch via post-trim length < min.
        if length < _MIN_BEAM_MM:
            _reject(actions, i,
                f"span {length:.0f} mm after column-face trim < minimum "
                f"{_MIN_BEAM_MM:.0f} mm (columns too close along beam axis)")
            continue

        snap_actions.append(
            f"framing[{i}] trimmed to column faces "
            f"(−{sp_half:.0f} mm start, −{ep_half:.0f} mm end)"
        )

        kept.append(beam)
        actions.extend(snap_actions)

    recipe["structural_framing"] = kept
    return recipe, actions


def _clamp_column_min_size(recipe: dict) -> tuple[dict, list[str]]:
    """Clamp column width/depth to _COL_MIN_MM (Revit rejects families below 200 mm)."""
    actions: list[str] = []
    for i, col in enumerate(recipe.get("columns", [])):
        for field in ("width", "depth"):
            v = float(col.get(field, 800.0))
            if v < _COL_MIN_MM:
                col[field] = _COL_MIN_MM
                actions.append(f"column[{i}].{field} clamped {v:.0f} → {_COL_MIN_MM:.0f} mm")
    return recipe, actions
