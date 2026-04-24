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
from collections import Counter
from loguru import logger

_MIN_BEAM_MM:       float = float(os.getenv("MIN_BEAM_MM",       "500"))
_SNAP_BUFFER_MM:    float = float(os.getenv("SNAP_BUFFER_MM",    "150"))
_COL_MIN_MM:        float = float(os.getenv("COL_MIN_MM",        "200"))
# After snapping both endpoints to column centres, the beam axis should be
# either horizontal (dy≈0) or vertical (dx≈0). Allow small slop so near-colinear
# columns (within this tolerance) still produce a valid beam.
_AXIS_TOLERANCE_MM: float = float(os.getenv("AXIS_TOLERANCE_MM", "50"))
# Secondary beams that frame into a primary beam (not a column) — if an endpoint
# didn't snap to a column, try snapping it onto the centreline of a primary beam
# that itself snapped to columns on both ends. Tighter than column snap because
# a secondary's bbox end typically lands right on the primary's line.
_BEAM_SNAP_MM:      float = float(os.getenv("BEAM_SNAP_MM",      "400"))


def sanitize_recipe(recipe: dict) -> tuple[dict, list[str], list[dict]]:
    """
    Apply all sanitization passes to *recipe* in-place.

    Returns (recipe, actions, rejected). `rejected` is a list of
    {id, reason, tag, original_start, original_end, snapped_keys} dicts —
    one per framing beam dropped, with the PRE-snap mm endpoints so a caller
    can overlay them on the source plan for diagnosis.
    """
    col_boxes = _col_centers(recipe)

    framing_in = len(recipe.get("structural_framing", []))
    recipe, a1, drop_reasons, rejected = _snap_and_filter_framing(recipe, col_boxes)
    recipe, a2 = _clamp_column_min_size(recipe)

    actions = a1 + a2
    if actions:
        logger.info("RecipeSanitizer: {} fix(es) applied before export", len(actions))
        for a in actions:
            logger.debug("  • {}", a)
    else:
        logger.debug("RecipeSanitizer: recipe clean — no pre-export fixes needed")

    if drop_reasons:
        framing_kept = len(recipe.get("structural_framing", []))
        dropped = sum(drop_reasons.values())
        logger.warning(
            "RecipeSanitizer: {} of {} framing beam(s) dropped ({} kept) — reasons: {}",
            dropped, framing_in, framing_kept,
            ", ".join(f"{reason}={count}" for reason, count in drop_reasons.most_common()),
        )

    return recipe, actions, rejected


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


def _reject(
    actions: list[str],
    index: int,
    reason: str,
    tag: str,
    counts: Counter,
    rejected: list[dict],
    entry: dict,
) -> None:
    actions.append(f"framing[{index}] removed — {reason}")
    counts[tag] += 1
    original = entry["original"]
    rejected.append({
        "id":             entry["beam"].get("id"),
        "reason":         reason,
        "tag":            tag,
        "original_start": original.get("start_point"),
        "original_end":   original.get("end_point"),
        "snapped_keys":   list(entry["snapped"].keys()),
    })


def _snap_and_filter_framing(
    recipe: dict,
    centers: list[tuple[float, float, float, float]],
) -> tuple[dict, list[str], Counter, list[dict]]:
    """
    Three passes over structural_framing:
      1. Column snap — each endpoint within (col_half + _SNAP_BUFFER_MM) of a
         column centre is moved onto that centre.
      2. Beam-to-beam snap — endpoints that didn't hit a column in pass 1 are
         projected onto the nearest *primary* beam's centreline (primary = both
         ends snapped to columns), within _BEAM_SNAP_MM. Recovers secondaries.
      3. Validate + face-trim + dedup. Rejects beams that fail any of:
           - an endpoint still floating after both snap passes
           - both endpoints snapped to the same point
           - post-snap axis diagonal beyond _AXIS_TOLERANCE_MM
           - duplicate span — same endpoints as an earlier kept beam
           - span < _MIN_BEAM_MM after column-face trim
         Column-snapped ends are trimmed inward by the column half-dimension so
         the beam body stops at the face; beam-snapped ends keep their projected
         position (beam lands on the primary's centreline).
    """
    framing = recipe.get("structural_framing", [])
    actions:  list[str]  = []
    drops:    Counter    = Counter()
    rejected: list[dict] = []
    seen_pairs: set[frozenset[tuple[float, float]]] = set()

    if not centers or not framing:
        return recipe, actions, drops, rejected

    per_beam: list[dict] = []
    for i, beam in enumerate(framing):
        # Snapshot pre-snap endpoints for diagnostics before any mutation.
        original = {
            k: dict(beam[k]) for k in _ENDPOINT_KEYS
            if isinstance(beam.get(k), dict)
        }
        snapped: dict[str, tuple[float, float, float, float]] = {}
        snap_actions: list[str] = []
        for pt_key in _ENDPOINT_KEYS:
            pt = beam.get(pt_key)
            if not isinstance(pt, dict):
                continue
            px = float(pt.get("x", 0.0))
            py = float(pt.get("y", 0.0))
            best_col:  tuple[float, float, float, float] | None = None
            best_dist: float = float("inf")
            for cx, cy, hw, hd in centers:
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
        per_beam.append({
            "beam": beam, "snapped": snapped, "actions": snap_actions,
            "original": original,
        })

    # --- Pass 2: beam-to-beam snap for endpoints that didn't find a column.
    # Only snap onto beams whose *both* endpoints snapped to columns (primaries).
    # Projection is perpendicular onto the primary's centreline, clamped to its
    # span so a secondary can't extend past the primary.
    primaries = [p for p in per_beam
                 if "start_point" in p["snapped"] and "end_point" in p["snapped"]]

    def _snap_to_primary(px: float, py: float) -> tuple[float, float, float] | None:
        """Return (proj_x, proj_y, dist_mm) of the nearest primary centreline, or None."""
        best: tuple[float, float, float] | None = None
        best_d = _BEAM_SNAP_MM
        for p in primaries:
            b = p["beam"]
            ax, ay = b["start_point"]["x"], b["start_point"]["y"]
            bx, by = b["end_point"]["x"],   b["end_point"]["y"]
            vx, vy = bx - ax, by - ay
            L2 = vx * vx + vy * vy
            if L2 < 1.0:
                continue
            t = ((px - ax) * vx + (py - ay) * vy) / L2
            t = max(0.0, min(1.0, t))
            qx = ax + t * vx
            qy = ay + t * vy
            d = math.hypot(px - qx, py - qy)
            if d < best_d:
                best_d = d
                best = (qx, qy, d)
        return best

    for i, entry in enumerate(per_beam):
        beam = entry["beam"]
        snapped = entry["snapped"]
        for pt_key in _ENDPOINT_KEYS:
            if pt_key in snapped:
                continue
            pt = beam.get(pt_key)
            if not isinstance(pt, dict):
                continue
            hit = _snap_to_primary(float(pt["x"]), float(pt["y"]))
            if hit is None:
                continue
            qx, qy, d = hit
            pt["x"] = qx
            pt["y"] = qy
            # Sentinel: half_w=half_d=0 means "snapped to beam, don't face-trim".
            snapped[pt_key] = (qx, qy, 0.0, 0.0)
            entry["actions"].append(
                f"framing[{i}].{pt_key} snapped to primary beam centreline @ "
                f"({qx:.0f}, {qy:.0f}) mm  [was {d:.0f} mm away]"
            )

    # --- Pass 3: validation, dedup, face-trim, keep list.
    kept: list[dict] = []
    for i, entry in enumerate(per_beam):
        beam = entry["beam"]
        snapped = entry["snapped"]
        snap_actions = entry["actions"]

        missing = [k for k in _ENDPOINT_KEYS if k not in snapped]
        if missing:
            _reject(actions, i,
                f"{', '.join(missing)} not within {_SNAP_BUFFER_MM:.0f} mm of any column "
                f"nor {_BEAM_SNAP_MM:.0f} mm of any primary beam (would float in model)",
                "floating_endpoint", drops, rejected, entry)
            continue

        sp_col = snapped["start_point"]
        ep_col = snapped["end_point"]
        if sp_col[0] == ep_col[0] and sp_col[1] == ep_col[1]:
            _reject(actions, i, "both endpoints snapped to the same column/beam point",
                "same_column", drops, rejected, entry)
            continue

        pair_key = frozenset({(sp_col[0], sp_col[1]), (ep_col[0], ep_col[1])})
        if pair_key in seen_pairs:
            _reject(actions, i,
                "duplicate span — another beam already snapped to the same endpoints",
                "duplicate_span", drops, rejected, entry)
            continue
        seen_pairs.add(pair_key)

        sp, ep = beam["start_point"], beam["end_point"]
        dx = ep["x"] - sp["x"]
        dy = ep["y"] - sp["y"]

        if min(abs(dx), abs(dy)) > _AXIS_TOLERANCE_MM:
            _reject(actions, i,
                f"diagonal beam after snap (dx={dx:.0f}mm, dy={dy:.0f}mm > "
                f"tolerance {_AXIS_TOLERANCE_MM:.0f}mm); endpoints are not colinear",
                "diagonal", drops, rejected, entry)
            continue

        # Face-trim only where the endpoint snapped to a column (half_w/half_d > 0).
        # Beam-to-beam endpoints keep their projected position (beam lands on the
        # primary's centreline — no face offset needed).
        axis_is_x = abs(dx) >= abs(dy)
        if axis_is_x:
            sp_half = sp_col[2]
            ep_half = ep_col[2]
            direction = 1.0 if dx > 0 else -1.0
            sp["x"] += direction * sp_half
            ep["x"] -= direction * ep_half
        else:
            sp_half = sp_col[3]
            ep_half = ep_col[3]
            direction = 1.0 if dy > 0 else -1.0
            sp["y"] += direction * sp_half
            ep["y"] -= direction * ep_half

        length = math.hypot(ep["x"] - sp["x"], ep["y"] - sp["y"])
        if length < _MIN_BEAM_MM:
            _reject(actions, i,
                f"span {length:.0f} mm after face trim < minimum "
                f"{_MIN_BEAM_MM:.0f} mm (endpoints too close along beam axis)",
                "too_short", drops, rejected, entry)
            continue

        if sp_half or ep_half:
            snap_actions.append(
                f"framing[{i}] trimmed to column faces "
                f"(−{sp_half:.0f} mm start, −{ep_half:.0f} mm end)"
            )

        kept.append(beam)
        actions.extend(snap_actions)

    recipe["structural_framing"] = kept
    return recipe, actions, drops, rejected


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
