"""
Dashline validator — confirms each YOLO-proposed beam against the PDF vector
layer and overrides its bbox with the actual dashed-line extents.

The PDFs we process encode dashed beams as long runs of short individual line
segments (not a PostScript dash pattern — every path has `dashes="[] 0"`). So
we detect dashed lines geometrically: a cluster of ≥N short axis-aligned
segments inside a YOLO framing bbox is taken as the true beam.

A beam without dashline evidence is dropped. A beam with evidence has its
bbox tightened to the axis-min / axis-max of the matching segments — so the
downstream geometry generator writes the correct start/end mm coordinates.
"""
from __future__ import annotations

from bisect import bisect_left

from loguru import logger


# Segment length band (in PDF points) that matches typical CAD dash strokes.
_DASH_LEN_MIN_PT = 2.0
_DASH_LEN_MAX_PT = 20.0
# Minimum number of collinear dash segments required to confirm a beam.
_MIN_DASH_SEGMENTS = 3
# Padding (px) when testing segment-inside-bbox so dashes touching the bbox
# edge still count.
_BBOX_PAD_PX = 4.0
# Perpendicular distance (px) within which a segment is treated as lying on a
# grid/centerline and excluded from dashline evidence.
_GRIDLINE_EXCLUSION_PX = 20.0


def _segment_to_pixels(
    x1: float, y1: float, x2: float, y2: float,
    disp_w_pt: float, disp_h_pt: float, scale: float, rotation: int,
) -> tuple[float, float, float, float]:
    """Apply page rotation + DPI scale to bring a PDF-pt segment into image px."""
    def _one(cx: float, cy: float) -> tuple[float, float]:
        if rotation == 90:
            return (disp_w_pt - cy) * scale, cx * scale
        if rotation == 270:
            return cy * scale, (disp_h_pt - cx) * scale
        if rotation == 180:
            return (disp_w_pt - cx) * scale, (disp_h_pt - cy) * scale
        return cx * scale, cy * scale
    px1, py1 = _one(x1, y1)
    px2, py2 = _one(x2, y2)
    return px1, py1, px2, py2


def _collect_dash_segments(
    vector_data: dict, image_dpi: float,
    grid_info: dict | None = None,
) -> list[tuple[float, float, float, float, str]]:
    """Return all short axis-aligned segments in image-pixel coords.

    Tuple: (x1_px, y1_px, x2_px, y2_px, orient) where orient ∈ {'h', 'v'}.
    """
    page_rect = vector_data.get("page_rect") or [0, 0, 0, 0]
    disp_w_pt = float(page_rect[2] - page_rect[0])
    disp_h_pt = float(page_rect[3] - page_rect[1])
    rotation  = int(vector_data.get("page_rotation", 0) or 0)
    scale     = image_dpi / 72.0

    x_grid_px = sorted(float(v) for v in (grid_info or {}).get("x_lines_px", []))
    y_grid_px = sorted(float(v) for v in (grid_info or {}).get("y_lines_px", []))

    def _near_grid(v: float, grid: list[float]) -> bool:
        if not grid:
            return False
        i = bisect_left(grid, v)
        best = min(
            (abs(grid[j] - v) for j in (i - 1, i) if 0 <= j < len(grid)),
            default=float("inf"),
        )
        return best <= _GRIDLINE_EXCLUSION_PX

    out: list[tuple[float, float, float, float, str]] = []
    skipped_gridline = 0
    for path in vector_data.get("paths", []):
        items = path.get("items") or []
        if len(items) != 1:
            continue
        it = items[0]
        if it[0] != "l":
            continue
        p0, p1 = it[1], it[2]
        x1, y1 = float(p0.x), float(p0.y)
        x2, y2 = float(p1.x), float(p1.y)
        dx, dy = x2 - x1, y2 - y1
        length_pt = (dx * dx + dy * dy) ** 0.5
        if not (_DASH_LEN_MIN_PT <= length_pt <= _DASH_LEN_MAX_PT):
            continue
        # Axis-aligned filter (allow tiny skew). Orientation is finalised after
        # rotation; test here only to reject skewed strokes before transform.
        if abs(dx) < 3.0 * abs(dy) and abs(dy) < 3.0 * abs(dx):
            continue
        px1, py1, px2, py2 = _segment_to_pixels(
            x1, y1, x2, y2, disp_w_pt, disp_h_pt, scale, rotation,
        )
        if abs(px2 - px1) >= abs(py2 - py1):
            orient = "h"
            perp = (py1 + py2) / 2.0
            if _near_grid(perp, y_grid_px):
                skipped_gridline += 1
                continue
        else:
            orient = "v"
            perp = (px1 + px2) / 2.0
            if _near_grid(perp, x_grid_px):
                skipped_gridline += 1
                continue
        out.append((px1, py1, px2, py2, orient))

    if skipped_gridline:
        logger.debug(
            "DashlineValidator: excluded {} segment(s) lying on grid/centerlines",
            skipped_gridline,
        )
    return out


def validate_framing_dashlines(
    framing: list[dict],
    vector_data: dict,
    image_dpi: float,
    grid_info: dict | None = None,
) -> tuple[list[dict], list[dict]]:
    """Filter framing detections by vector-dashline evidence.

    For each detection:
      - search for ≥3 short axis-aligned segments inside its bbox along its
        long axis;
      - if found, overwrite the detection's bbox/center with the dashline's
        axis extents (short-axis thickness is preserved);
      - otherwise drop the detection.

    Returns (kept, dropped).
    """
    segments = _collect_dash_segments(vector_data, image_dpi, grid_info)
    if not segments:
        logger.warning("DashlineValidator: no dash candidate segments extracted.")
        return framing, []

    kept: list[dict] = []
    dropped: list[dict] = []
    for det in framing:
        bbox = det.get("bbox") or []
        if len(bbox) < 4:
            dropped.append(det)
            continue
        x1, y1, x2, y2 = (float(v) for v in bbox[:4])
        x_lo, x_hi = min(x1, x2) - _BBOX_PAD_PX, max(x1, x2) + _BBOX_PAD_PX
        y_lo, y_hi = min(y1, y2) - _BBOX_PAD_PX, max(y1, y2) + _BBOX_PAD_PX
        long_axis = "h" if (x_hi - x_lo) >= (y_hi - y_lo) else "v"

        matches = [
            (sx1, sy1, sx2, sy2)
            for (sx1, sy1, sx2, sy2, orient) in segments
            if orient == long_axis
            and x_lo <= min(sx1, sx2) and max(sx1, sx2) <= x_hi
            and y_lo <= min(sy1, sy2) and max(sy1, sy2) <= y_hi
        ]
        if len(matches) < _MIN_DASH_SEGMENTS:
            det["dashline_found"] = False
            dropped.append(det)
            continue

        xs = [c for m in matches for c in (m[0], m[2])]
        ys = [c for m in matches for c in (m[1], m[3])]
        if long_axis == "h":
            new_x1, new_x2 = min(xs), max(xs)
            mid_y = sum(ys) / len(ys)
            half_t = abs(y2 - y1) / 2.0
            new_bbox = [new_x1, mid_y - half_t, new_x2, mid_y + half_t]
        else:
            new_y1, new_y2 = min(ys), max(ys)
            mid_x = sum(xs) / len(xs)
            half_t = abs(x2 - x1) / 2.0
            new_bbox = [mid_x - half_t, new_y1, mid_x + half_t, new_y2]

        det["bbox"] = new_bbox
        det["center"] = [
            (new_bbox[0] + new_bbox[2]) / 2.0,
            (new_bbox[1] + new_bbox[3]) / 2.0,
        ]
        det["dashline_found"] = True
        det["dashline_segments"] = len(matches)
        kept.append(det)

    logger.info(
        "DashlineValidator: {} beam(s) kept (dashline found), {} dropped "
        "(no dashline inside bbox)",
        len(kept), len(dropped),
    )
    return kept, dropped
