"""
Admittance rule for slabs (floor plates).

Slabs are large, roughly rectangular regions covering whole bays or zones.
Common YOLO failure modes we filter here:

  1. Fragment detections — tiny bboxes around hatch patterns, thickness
     annotations, or room-number bubbles that share the slab's visual
     texture but aren't slabs themselves.
  2. Duplicate detections — two overlapping bboxes of the same floor plate
     (different tiles saw the same slab at the tile boundary).
  3. Off-envelope detections — bboxes whose centre lies outside the grid
     extent (title block, legend, notes area).

No geometry fixes are applied; slabs are only admit/reject. Thickness and
elevation are assigned downstream in geometry_generator from legend or
project defaults.

Signals:
  • min_area_px    — reject if bbox area is below the floor (likely noise)
  • inside_grid    — reject if centre lies outside x_lines_px/y_lines_px extent
  • dup_overlap    — reject if another slab detection has IoU ≥ threshold
                     and a larger area (keeps the bigger/earlier detection)
  • nearest_tag    — metadata only; "S1", "FS1", etc. surfaced for audit
"""
from __future__ import annotations

from backend.services.intelligence.admittance.context import ElementContext
from backend.services.intelligence.admittance.scoring import Decision, admit, reject
from backend.services.intelligence.admittance.signals import find_nearest_tag
from backend.services.intelligence.cross_element_validator import _iou as _bbox_iou


# Tune these on real data; values chosen for 300 DPI renders where
# 1 mm ≈ 11.8 px, so 300 px ≈ 25 mm drawing ≈ 2.5 m real-world at 1:100.
_MIN_AREA_PX     = 300 * 300     # reject slivers smaller than this
_DUP_IOU_THRESH  = 0.5           # overlap at which two bboxes are "the same slab"
_TAG_SEARCH_RADIUS_PX = 200.0    # slabs are large; search farther for their tag


def judge(det: dict, siblings: list[dict], ctx: ElementContext) -> Decision:
    bbox = det.get("bbox")
    if not bbox or len(bbox) < 4:
        return reject("no_bbox")

    x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    area = w * h

    # 1. Area floor — drop hatch fragments and annotation-sized bboxes.
    if area < _MIN_AREA_PX:
        return reject("too_small", area_px=round(area, 0), min_area_px=_MIN_AREA_PX)

    # 2. Centre must lie inside the grid envelope. Slabs outside the grid
    #    are almost always title-block hatching or legend patches.
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    if not _center_inside_grid(cx, cy, ctx.grid_info):
        return reject("outside_grid_envelope", center=[round(cx, 1), round(cy, 1)])

    # 3. Deduplicate overlapping slab detections — keep the larger one.
    #    Only reject when a *larger* sibling overlaps us; otherwise this is
    #    the keeper and the smaller one will be rejected on its own pass.
    for other in siblings:
        if other is det or other.get("type") != "slab":
            continue
        ob = other.get("bbox")
        if not ob or len(ob) < 4:
            continue
        iou = _bbox_iou([x1, y1, x2, y2], [float(ob[0]), float(ob[1]), float(ob[2]), float(ob[3])])
        if iou < _DUP_IOU_THRESH:
            continue
        other_area = abs(ob[2] - ob[0]) * abs(ob[3] - ob[1])
        if other_area > area:
            return reject("duplicate_of_larger",
                          iou=round(iou, 3),
                          area_px=round(area, 0),
                          other_area_px=round(other_area, 0))

    # 4. Tag lookup is purely metadata — not scored.
    tag, _material, _ = find_nearest_tag(
        (x1, y1, x2, y2), ctx, max_dist_px=_TAG_SEARCH_RADIUS_PX,
    )
    md: dict = {}
    if tag:
        md["tag"] = tag

    return admit("slab_ok", metadata=md, tag=tag,
                 area_px=round(area, 0), width_px=round(w, 0), height_px=round(h, 0))


def _center_inside_grid(cx: float, cy: float, grid_info: dict) -> bool:
    """True when (cx, cy) lies inside the min/max extent of grid lines.

    Slabs bypass the upstream ``remove_outside_grid`` cull (that pass
    runs on columns + framing only), so this check exists as the first
    envelope filter for slab detections.

    If the grid has fewer than 2 lines on either axis the check is
    skipped (returns True) — we have no envelope to reject against.
    """
    xs = grid_info.get("x_lines_px") or []
    ys = grid_info.get("y_lines_px") or []
    if len(xs) < 2 or len(ys) < 2:
        return True
    return (min(xs) <= cx <= max(xs)) and (min(ys) <= cy <= max(ys))
