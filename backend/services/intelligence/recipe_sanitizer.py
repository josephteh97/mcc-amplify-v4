"""
Recipe Sanitizer — deterministic pre-export cleanup to prevent known Revit errors.

Runs on the Revit recipe *before* it is written to transaction.json and sent to
the Windows machine.  No AI involved.  All fixes are geometric and rule-based.

Passes (in order):
  1. snap_and_filter_framing — snap beam endpoints to the nearest column's
                               insertion point; REMOVE any beam whose endpoints
                               don't both land on a column (would float in the
                               Revit model) or whose post-snap axis deviates
                               beyond AXIS_TOLERANCE_MM from X/Y (diagonal
                               beams between non-colinear columns).
                               Prevents visible gaps, floating beams, and
                               tilted beams that aren't perpendicular to the
                               column axis.
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


def _snap_and_filter_framing(
    recipe: dict,
    centers: list[tuple[float, float, float, float]],
) -> tuple[dict, list[str]]:
    """
    Single pass over structural_framing:
      - Snap each endpoint within (column half-width + _SNAP_BUFFER_MM) to the
        nearest column centre. Structural beams should span column-to-column;
        YOLO pixel bboxes rarely land exactly on the Revit insertion point.
      - REMOVE any beam where either endpoint failed to snap to a column —
        a floating endpoint creates a visible gap or a beam dangling in
        mid-air (images 3 and 4 of the common-sense review).
      - REMOVE any beam whose post-snap axis is diagonal by more than
        _AXIS_TOLERANCE_MM — this happens when snap pulls the two endpoints
        to columns that aren't perfectly colinear, producing a tilted beam
        not perpendicular to the column axis (image 2).
      - REMOVE any beam whose span drops below _MIN_BEAM_MM after snapping
        (both endpoints snapped to the same column).
    """
    framing = recipe.get("structural_framing", [])
    kept:    list[dict] = []
    actions: list[str]  = []

    if not centers or not framing:
        return recipe, actions

    for i, beam in enumerate(framing):
        snap_actions: list[str] = []
        snapped = {"start_point": False, "end_point": False}

        for pt_key in ("start_point", "end_point"):
            pt = beam.get(pt_key)
            if not isinstance(pt, dict):
                continue

            px = float(pt.get("x", 0.0))
            py = float(pt.get("y", 0.0))
            best_col:  tuple[float, float] | None = None
            best_dist: float = float("inf")

            for cx, cy, hw, hd in centers:
                # Fast axis pre-reject — skips the sqrt for distant columns
                if abs(px - cx) > hw + _SNAP_BUFFER_MM or abs(py - cy) > hd + _SNAP_BUFFER_MM:
                    continue
                d = _dist2d(px, py, cx, cy)
                if d < best_dist:
                    best_dist = d
                    best_col  = (cx, cy)

            if best_col:
                cx, cy = best_col
                pt["x"] = cx
                pt["y"] = cy
                snapped[pt_key] = True
                snap_actions.append(
                    f"framing[{i}].{pt_key} snapped to column @ "
                    f"({cx:.0f}, {cy:.0f}) mm  [was {best_dist:.0f} mm away]"
                )

        if not (snapped["start_point"] and snapped["end_point"]):
            missing = [k for k, v in snapped.items() if not v]
            actions.append(
                f"framing[{i}] removed — {', '.join(missing)} not within "
                f"{_SNAP_BUFFER_MM:.0f} mm of any column (would float in model)"
            )
            continue

        sp = beam["start_point"]
        ep = beam["end_point"]
        dx = float(ep.get("x", 0.0)) - float(sp.get("x", 0.0))
        dy = float(ep.get("y", 0.0)) - float(sp.get("y", 0.0))
        length = math.hypot(dx, dy)

        off_axis = min(abs(dx), abs(dy))
        if off_axis > _AXIS_TOLERANCE_MM:
            actions.append(
                f"framing[{i}] removed — diagonal beam after snap "
                f"(dx={dx:.0f}mm, dy={dy:.0f}mm > tolerance {_AXIS_TOLERANCE_MM:.0f}mm); "
                f"endpoint columns are not colinear"
            )
            continue

        if length < _MIN_BEAM_MM:
            actions.append(
                f"framing[{i}] removed — span {length:.0f} mm "
                f"< minimum {_MIN_BEAM_MM:.0f} mm"
            )
            continue

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
