"""
Admittance rule for structural_framing (beams).

Core judgment: when a beam bbox centre is close to a column centre, Revit's
join engine will throw "Cannot keep elements joined" at transaction commit.
Naive rejection drops real beams that merely terminate *at* a column face
(which is how most beams actually connect). We instead look for corroborating
signals that the beam is real, and if so, snap its end to the column face
and admit; only a conflict with no other evidence is rejected.

Signals combined (score → decision):
  +3  nearest-tag is in the legend and matches dashline-inferred material
  +2  nearest-tag present (even without legend match)
  +2  beam long axis aligns with a grid line through the conflict column
  +1  stroke style determined (dashed OR solid) — proves the beam is drawn
  +1  distance > 0.5 × beam short-dim (not perfectly coincident with column)

Decision thresholds:
  score ≥ 3 → ADMIT_WITH_FIX  (snap both endpoints to nearest-column faces)
  score == 2 → ADMIT           (keep as-is; borderline)
  score < 2  → REJECT          (insufficient corroboration — likely noise)

Material tagging runs on all admitted beams regardless of join conflict.
"""
from __future__ import annotations

from backend.services.intelligence.admittance.context import ElementContext
from backend.services.intelligence.admittance.scoring import (
    Decision, admit, reject, admit_with_fix,
)
from backend.services.intelligence.admittance.signals import (
    classify_stroke_style, find_nearest_tag, beam_axis_alignment, nearest_neighbor,
)


_JOIN_CLEARANCE_FACTOR = 1.5          # same trigger as the legacy rule
_TAG_SEARCH_RADIUS_PX  = 120.0
_GRID_ALIGN_TOL_PX     = 30.0
# Snap reach cap: a beam can stretch at most this multiple of its original
# long-axis length to reach a column. Prevents a short YOLO detection from
# snapping to a column in the next-next bay.
_SNAP_MAX_GROWTH = 3.0


def judge(det: dict, siblings: list[dict], ctx: ElementContext) -> Decision:
    bbox = det.get("bbox")
    if not bbox or len(bbox) < 4:
        return reject("no_bbox")

    bbox_t  = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    center  = det.get("center") or [(bbox_t[0] + bbox_t[2]) / 2, (bbox_t[1] + bbox_t[3]) / 2]
    short_dim = min(abs(bbox_t[2] - bbox_t[0]), abs(bbox_t[3] - bbox_t[1]))

    stroke = classify_stroke_style(bbox_t, ctx)
    stroke_material = {"dashed": "steel", "solid": "rc"}.get(stroke)

    tag, tag_material, _ = find_nearest_tag(bbox_t, ctx, max_dist_px=_TAG_SEARCH_RADIUS_PX)
    material = tag_material or stroke_material      # legend tag wins when both present

    col, col_dist = nearest_neighbor((center[0], center[1]), siblings, of_type="column")
    clearance = short_dim * _JOIN_CLEARANCE_FACTOR
    in_conflict = col is not None and col_dist < clearance

    md: dict = {}
    if material:
        md["material"] = material
    if tag:
        md["tag"] = tag

    aligned, axis = beam_axis_alignment(bbox_t, ctx.grid_info, tolerance_px=_GRID_ALIGN_TOL_PX)

    # Always attempt to snap endpoints to columns — closes the gap-to-column
    # that Revit would otherwise show as a floating beam. _snap_bbox_to_columns
    # is gated by the perpendicular band filter (no cross-gridline snaps) plus
    # _SNAP_MAX_GROWTH (no absurd extensions).
    snap_axis = axis or _infer_axis(bbox_t)
    snapped = _snap_bbox_to_columns(bbox_t, siblings, snap_axis)
    snapped_changed = snapped != bbox_t

    if not in_conflict:
        if snapped_changed:
            return admit_with_fix("snap_to_columns", bbox_override=snapped,
                                  metadata=md, stroke=stroke, tag=tag, material=material)
        return admit("no_conflict", metadata=md, stroke=stroke, tag=tag, material=material)

    score = _score(tag, material, aligned, stroke, col_dist, short_dim, ctx.legend_map)

    signals = dict(
        stroke=stroke, tag=tag, material=material,
        grid_aligned=aligned, axis=axis,
        col_id=(col.get("id") if col else None),
        col_dist=round(col_dist, 1),
        clearance=round(clearance, 1),
        score=score,
    )
    md["conflict_column_center"] = list(col.get("center", [])) if col else None

    if score >= 3:
        return admit_with_fix("join_conflict_resolved", bbox_override=snapped,
                              metadata=md, **signals)
    if score == 2:
        return admit("join_conflict_borderline_admitted", metadata=md, **signals)
    return reject("join_conflict_unsupported", **signals)


def _score(tag, material, aligned, stroke, col_dist, short_dim, legend_map) -> int:
    score = 0
    if tag and material and legend_map and tag in legend_map:
        score += 3
    elif tag:
        score += 2
    if aligned:
        score += 2
    if stroke in ("dashed", "solid"):
        score += 1
    if col_dist > 0.5 * short_dim:
        score += 1
    return score


def _infer_axis(bbox: tuple[float, float, float, float]) -> str:
    dx = abs(bbox[2] - bbox[0])
    dy = abs(bbox[3] - bbox[1])
    return "x" if dx >= dy else "y"


def _snap_bbox_to_columns(
    bbox: tuple[float, float, float, float],
    siblings: list[dict],
    axis: str,
) -> tuple[float, float, float, float]:
    """
    Extend bbox along `axis` so both ends reach the nearest column face.

    Only columns whose centre lies within one column-width perpendicular to
    the beam axis are considered (otherwise we'd snap to a column in a
    completely different gridline).
    """
    # Parameterise by axis — "x" uses bbox[0/2] along, [1/3] across; "y" swaps.
    along_lo, along_hi, across_lo, across_hi = (0, 2, 1, 3) if axis == "x" else (1, 3, 0, 2)

    beam_mid_across = (bbox[across_lo] + bbox[across_hi]) * 0.5
    beam_mid_along  = (bbox[along_lo]  + bbox[along_hi])  * 0.5
    beam_across     = abs(bbox[across_hi] - bbox[across_lo])

    candidates: list[tuple[float, float]] = []   # (col_lo_along, col_hi_along)
    for c in siblings:
        if c.get("type") != "column":
            continue
        cb = c.get("bbox")
        if not cb or len(cb) < 4:
            continue
        col_mid_across = (cb[across_lo] + cb[across_hi]) * 0.5
        col_half_across = abs(cb[across_hi] - cb[across_lo]) * 0.5
        if abs(col_mid_across - beam_mid_across) <= col_half_across + beam_across:
            candidates.append((cb[along_lo], cb[along_hi]))
    if not candidates:
        return bbox

    # Snap to column CENTRES — Revit convention is centreline-to-centreline
    # framing; the auto-join trims each beam end to the column face at commit.
    # (Snapping to the face directly breaks recipe_sanitizer's 150 mm
    # centre-tolerance filter.)
    def _center(cb):
        return (cb[0] + cb[1]) * 0.5
    befores = [cb for cb in candidates if _center(cb) < beam_mid_along]
    afters  = [cb for cb in candidates if _center(cb) >= beam_mid_along]
    new_lo = max(_center(cb) for cb in befores) if befores else bbox[along_lo]
    new_hi = min(_center(cb) for cb in afters)  if afters  else bbox[along_hi]
    if new_hi - new_lo < beam_across:       # sanity check — abandon if degenerate
        return bbox

    orig_len = bbox[along_hi] - bbox[along_lo]
    if orig_len > 0 and (new_hi - new_lo) > orig_len * _SNAP_MAX_GROWTH:
        return bbox

    patched = list(bbox)
    patched[along_lo] = new_lo
    patched[along_hi] = new_hi
    return tuple(patched)  # type: ignore[return-value]
