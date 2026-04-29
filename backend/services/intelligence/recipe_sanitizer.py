"""
Recipe Sanitizer — deterministic pre-export cleanup to prevent known Revit errors.

Runs on the Revit recipe *before* it is written to transaction.json and sent to
the Windows machine.  No AI involved.  All fixes are geometric and rule-based.

Passes (in order):
  1. snap_and_filter_framing
        Pass A — perpendicular grid-line snap.
            For each beam endpoint, shift it perpendicular to the beam axis
            onto the nearest perpendicular grid line within half-grid-spacing.
            Then anchor the endpoint to a column or core wall on that grid
            intersection.  A beam normally sits *on* a grid line between two
            columns, so this corrects YOLO drift without inventing geometry.
        Pass B — dashline-confirmed extension (only on endpoints that didn't
            anchor in Pass A).
            If a dashed beam centreline in the vector data passes near the
            floating endpoint AND aligns with the beam axis, walk outward in
            grid-spacing steps (up to 4 if the other end is anchored, up to 2
            if both ends float) and probe each step for a column, core wall,
            or a closed rectangle the size of a column.  First hit wins.
        Reject taxonomy:
            no_dashline / dashline_no_anchor / out_of_grid /
            same_column / duplicate_span / diagonal / too_short
        Snapped beams are face-trimmed by the column half-dimension along the
        beam axis so the body stops at the column face, not its centre.
  2. clamp_column_min_size — ensure column width/depth >= COL_MIN_MM (200 mm).
                             Revit auto-deletes families below this threshold.

Thresholds tunable via env vars (defaults match SS CP 65 practice).
"""
from __future__ import annotations

import math
import os
from collections import Counter
from dataclasses import dataclass
from loguru import logger

from backend.services.intelligence.admittance.signals.dashline import _is_dashed
from backend.services.intelligence.grid_coords import interp_sorted, grid_lines_world_mm

_MIN_BEAM_MM:           float = float(os.getenv("MIN_BEAM_MM",       "500"))
_COL_MIN_MM:            float = float(os.getenv("COL_MIN_MM",        "200"))
_AXIS_TOLERANCE_MM:     float = float(os.getenv("AXIS_TOLERANCE_MM", "50"))
_PERP_SNAP_FRACTION:    float = float(os.getenv("PERP_SNAP_FRACTION", "0.5"))
_ALONG_ANCHOR_FRACTION: float = float(os.getenv("ALONG_ANCHOR_FRACTION", "0.5"))
_STEP_ANCHOR_FRACTION:  float = float(os.getenv("STEP_ANCHOR_FRACTION", "0.25"))
# Tight perpendicular tolerance — beam should sit ON the dashline. Stops us
# catching the dashline of a parallel beam one grid line over.
_DASHLINE_PERP_MM:      float = float(os.getenv("DASHLINE_PERP_MM",      "150"))
# Longitudinal slack as a fraction of beam width — covers cases where YOLO
# over-extends the bbox slightly past the centreline tip.
_DASHLINE_LONG_FRACTION: float = float(os.getenv("DASHLINE_LONG_FRACTION", "0.5"))
# Default beam width when the recipe entry omits it (matches geometry_generator).
_DEFAULT_BEAM_WIDTH_MM:  float = float(os.getenv("DEFAULT_BEAM_WIDTH_MM", "800"))
# cos 25° ≈ 0.9 — tightens up against near-perpendicular noise.
_DASHLINE_ALIGN_COS:    float = float(os.getenv("DASHLINE_ALIGN_COS", "0.9"))
_EXT_BUDGET_ONE_FLOAT:  int   = int(os.getenv("EXT_BUDGET_ONE_FLOAT",  "4"))
_EXT_BUDGET_BOTH_FLOAT: int   = int(os.getenv("EXT_BUDGET_BOTH_FLOAT", "2"))
# Vector rectangles considered "column-shaped" within this mm band; anything
# bigger is core wall / room / sheet border and skipped here.
_RECT_MIN_MM:           float = 200.0
_RECT_MAX_MM:           float = 1500.0


_ENDPOINT_KEYS = ("start_point", "end_point")


# ──────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────

def sanitize_recipe(
    recipe: dict,
    grid_info: dict | None = None,
    vector_data: dict | None = None,
    image_data: dict | None = None,
) -> tuple[dict, list[str], list[dict]]:
    """
    Apply all sanitization passes to *recipe* in-place.

    grid_info / vector_data / image_data unlock the Pass A grid-line snap and
    Pass B dashline-confirmed extension. Without them the sanitizer falls back
    to a degenerate "anchor must already be a column" rule (legacy behaviour).

    Returns (recipe, actions, rejected) — `rejected` contains pre-snap mm
    endpoints so a caller can overlay them on the source plan for diagnosis.
    """
    framing_in = len(recipe.get("structural_framing", []))

    ctx = _build_context(recipe, grid_info, vector_data, image_data)

    recipe, a1, drop_reasons, rejected = _snap_and_filter_framing(recipe, ctx)
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


# ──────────────────────────────────────────────────────────────────────────
# Context: grid lines, dashlines, rectangles, columns, core walls — all in mm.
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class _Ctx:
    columns:    list[tuple[float, float, float, float]]   # (cx, cy, half_w, half_d)
    core_walls: list[list[tuple[float, float]]]           # polygons
    rectangles: list[tuple[float, float, float, float]]   # (cx, cy, w, h) column-shaped
    # Beam-axis-aligned dashed segments, pre-bucketed by orientation so the hot
    # path is a pure interval + perpendicular-band check (no sqrt / cos test).
    # Each entry: (long_lo, long_hi, perp). Diagonal dashlines are dropped at
    # extract time — they can't confirm a beam axis.
    dashlines_x: list[tuple[float, float, float]]         # axis="x": long=x, perp=y
    dashlines_y: list[tuple[float, float, float]]         # axis="y": long=y, perp=x
    x_lines:    list[float]                               # mm, ascending
    y_lines:    list[float]                               # mm, world (Y-flipped, descending)
    dx_grid:    float = 0.0
    dy_grid:    float = 0.0

    @property
    def has_grid(self) -> bool:
        return bool(self.x_lines and self.y_lines and self.dx_grid > 0 and self.dy_grid > 0)

    @property
    def grid_x_min(self) -> float:
        return min(self.x_lines) if self.x_lines else float("-inf")

    @property
    def grid_x_max(self) -> float:
        return max(self.x_lines) if self.x_lines else float("inf")

    @property
    def grid_y_min(self) -> float:
        return min(self.y_lines) if self.y_lines else float("-inf")

    @property
    def grid_y_max(self) -> float:
        return max(self.y_lines) if self.y_lines else float("inf")


def _build_context(
    recipe: dict,
    grid_info: dict | None,
    vector_data: dict | None,
    image_data: dict | None,
) -> _Ctx:
    columns = _col_centers(recipe)
    core_walls = _core_wall_polygons(recipe)
    x_lines, y_lines, dx_grid, dy_grid = grid_lines_world_mm(grid_info)

    dashlines_x: list = []
    dashlines_y: list = []
    rectangles:  list = []
    if x_lines and y_lines and vector_data and image_data:
        dashlines_x, dashlines_y, rectangles = _extract_vector_features_mm(
            vector_data, grid_info, image_data,
        )

    return _Ctx(
        columns=columns,
        core_walls=core_walls,
        rectangles=rectangles,
        dashlines_x=dashlines_x,
        dashlines_y=dashlines_y,
        x_lines=x_lines,
        y_lines=y_lines,
        dx_grid=dx_grid,
        dy_grid=dy_grid,
    )


def _col_centers(recipe: dict) -> list[tuple[float, float, float, float]]:
    out: list[tuple[float, float, float, float]] = []
    for col in recipe.get("columns", []):
        loc = col.get("location", {})
        out.append((
            float(loc.get("x", 0.0)),
            float(loc.get("y", 0.0)),
            float(col.get("width", 800.0)) / 2.0,
            float(col.get("depth", 800.0)) / 2.0,
        ))
    return out


def _core_wall_polygons(recipe: dict) -> list[list[tuple[float, float]]]:
    out: list[list[tuple[float, float]]] = []
    for cw in recipe.get("core_walls", []):
        outline = cw.get("outline") or cw.get("polygon") or cw.get("points")
        if not outline:
            continue
        pts: list[tuple[float, float]] = []
        for p in outline:
            if isinstance(p, dict):
                pts.append((float(p.get("x", 0.0)), float(p.get("y", 0.0))))
            elif isinstance(p, (list, tuple)) and len(p) >= 2:
                pts.append((float(p[0]), float(p[1])))
        if len(pts) >= 3:
            out.append(pts)
    return out


# ──────────────────────────────────────────────────────────────────────────
# pt → mm conversion (rotation-aware, mirrors grid_detector._to_image_coords).
# ──────────────────────────────────────────────────────────────────────────

def _make_pt_to_mm(grid_info: dict, vector_data: dict, image_data: dict):
    """Return a closure (x_pt, y_pt) → (x_mm, y_mm) Y-flipped to match recipe."""
    page_rect = vector_data.get("page_rect") or [0.0, 0.0, 0.0, 0.0]
    rotation = int(vector_data.get("page_rotation", 0)) % 360
    scale = float(image_data.get("dpi", 300)) / 72.0

    page_x0 = float(page_rect[0])
    page_y0 = float(page_rect[1])
    page_w_pt = max(float(page_rect[2]) - page_x0, 1.0)
    page_h_pt = max(float(page_rect[3]) - page_y0, 1.0)

    # Pick the rotation transform once — avoids re-branching on every call
    # (this closure runs hundreds of thousands of times on big plans).
    if rotation == 90:
        def pt_to_px(x_pt: float, y_pt: float) -> tuple[float, float]:
            return (page_w_pt - (y_pt - page_y0)) * scale, (x_pt - page_x0) * scale
    elif rotation == 180:
        def pt_to_px(x_pt: float, y_pt: float) -> tuple[float, float]:
            return (page_w_pt - (x_pt - page_x0)) * scale, (page_h_pt - (y_pt - page_y0)) * scale
    elif rotation == 270:
        def pt_to_px(x_pt: float, y_pt: float) -> tuple[float, float]:
            return (y_pt - page_y0) * scale, (page_h_pt - (x_pt - page_x0)) * scale
    else:
        def pt_to_px(x_pt: float, y_pt: float) -> tuple[float, float]:
            return (x_pt - page_x0) * scale, (y_pt - page_y0) * scale

    x_lines_px = list(grid_info["x_lines_px"])
    y_lines_px = list(grid_info["y_lines_px"])
    x_sp = grid_info["x_spacings_mm"]
    y_sp = grid_info["y_spacings_mm"]
    x_world = [sum(x_sp[:i]) for i in range(len(x_lines_px))]
    y_world_raw = [sum(y_sp[:i]) for i in range(len(y_lines_px))]
    total_y = sum(y_sp)

    def pt_to_mm(x_pt: float, y_pt: float) -> tuple[float, float]:
        px, py = pt_to_px(x_pt, y_pt)
        x_mm = interp_sorted(px, x_lines_px, x_world)
        y_mm_raw = interp_sorted(py, y_lines_px, y_world_raw)
        return x_mm, total_y - y_mm_raw

    return pt_to_mm


# ──────────────────────────────────────────────────────────────────────────
# Vector feature extraction: dashlines + column-shaped rectangles, in mm.
# ──────────────────────────────────────────────────────────────────────────

def _extract_vector_features_mm(
    vector_data: dict,
    grid_info: dict,
    image_data: dict,
) -> tuple[
    list[tuple[float, float, float]],
    list[tuple[float, float, float]],
    list[tuple[float, float, float, float]],
]:
    """
    Walk vector_data.paths once and split into:
      - dashed line segments (beam centrelines), bucketed by orientation:
          dashlines_x: aligned with x-axis  (long_lo, long_hi, perp_y)
          dashlines_y: aligned with y-axis  (long_lo, long_hi, perp_x)
        Diagonal segments are dropped — they can't confirm a beam axis.
      - closed rectangles sized like columns (200..1500 mm), in mm — used as
        anchor candidates when YOLO/annotator missed a column
    """
    pt_to_mm = _make_pt_to_mm(grid_info, vector_data, image_data)
    dashlines_x: list[tuple[float, float, float]] = []
    dashlines_y: list[tuple[float, float, float]] = []
    rects:       list[tuple[float, float, float, float]] = []
    # Squared-form alignment threshold so we can classify by axis without sqrt.
    align_cos2 = _DASHLINE_ALIGN_COS * _DASHLINE_ALIGN_COS

    # Pre-filter rectangles in pt-space before paying for two pt→mm conversions.
    # Big plans contain ~100k paths (sheet borders, hatching, text glyphs);
    # the mm filter would drop them anyway, but only after the conversion.
    px_per_mm_x = float(grid_info.get("px_per_mm_x") or grid_info.get("pixels_per_mm") or 1.0)
    px_per_mm_y = float(grid_info.get("px_per_mm_y") or grid_info.get("pixels_per_mm") or 1.0)
    scale_pt_to_px = float(image_data.get("dpi", 300)) / 72.0
    if px_per_mm_x > 0 and px_per_mm_y > 0 and scale_pt_to_px > 0:
        mm_per_pt = scale_pt_to_px / max(px_per_mm_x, px_per_mm_y)
        # Generous bounds — we want to prune obvious non-candidates, not borderline.
        rect_pt_lo = (_RECT_MIN_MM / mm_per_pt) * 0.5
        rect_pt_hi = (_RECT_MAX_MM / mm_per_pt) * 2.0
    else:
        rect_pt_lo, rect_pt_hi = 0.0, float("inf")

    for path in vector_data.get("paths", []):
        items = path.get("items") or []
        is_dashed_path = _is_dashed(path.get("dashes"))
        rect_obj = path.get("rect")

        if rect_obj is not None:
            try:
                rx0 = float(rect_obj.x0 if hasattr(rect_obj, "x0") else rect_obj[0])
                ry0 = float(rect_obj.y0 if hasattr(rect_obj, "y0") else rect_obj[1])
                rx1 = float(rect_obj.x1 if hasattr(rect_obj, "x1") else rect_obj[2])
                ry1 = float(rect_obj.y1 if hasattr(rect_obj, "y1") else rect_obj[3])
            except (AttributeError, IndexError, TypeError, ValueError):
                rx0 = rx1 = ry0 = ry1 = 0.0
            if rx1 > rx0 and ry1 > ry0:
                w_pt = rx1 - rx0
                h_pt = ry1 - ry0
                if (rect_pt_lo <= w_pt <= rect_pt_hi
                        and rect_pt_lo <= h_pt <= rect_pt_hi):
                    ax, ay = pt_to_mm(rx0, ry0)
                    bx, by = pt_to_mm(rx1, ry1)
                    w_mm = abs(bx - ax)
                    h_mm = abs(by - ay)
                    if (
                        _RECT_MIN_MM <= w_mm <= _RECT_MAX_MM
                        and _RECT_MIN_MM <= h_mm <= _RECT_MAX_MM
                    ):
                        rects.append(((ax + bx) / 2.0, (ay + by) / 2.0, w_mm, h_mm))

        if is_dashed_path:
            for item in items:
                if not item or len(item) < 3 or item[0] != "l":
                    continue
                p1, p2 = item[1], item[2]
                try:
                    x1m, y1m = pt_to_mm(float(p1[0]), float(p1[1]))
                    x2m, y2m = pt_to_mm(float(p2[0]), float(p2[1]))
                except (TypeError, ValueError, IndexError):
                    continue
                dx = x2m - x1m
                dy = y2m - y1m
                L2 = dx * dx + dy * dy
                if L2 < 1.0:
                    continue
                # Bucket by orientation: cos²(angle) = component² / L².
                # x-aligned ⇒ |dx|/L ≥ ALIGN ⇒ dx² ≥ ALIGN² · L².
                if dx * dx >= align_cos2 * L2:
                    lo, hi = (x1m, x2m) if x1m <= x2m else (x2m, x1m)
                    dashlines_x.append((lo, hi, (y1m + y2m) * 0.5))
                elif dy * dy >= align_cos2 * L2:
                    lo, hi = (y1m, y2m) if y1m <= y2m else (y2m, y1m)
                    dashlines_y.append((lo, hi, (x1m + x2m) * 0.5))

    logger.debug(
        "RecipeSanitizer: vector features — {} x-aligned + {} y-aligned dashed "
        "segment(s), {} column-shaped rectangle(s)",
        len(dashlines_x), len(dashlines_y), len(rects),
    )
    return dashlines_x, dashlines_y, rects


# ──────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ──────────────────────────────────────────────────────────────────────────

def _dist2d(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x2 - x1, y2 - y1)


def _point_in_polygon(x: float, y: float, polygon: list[tuple[float, float]]) -> bool:
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
        ):
            inside = not inside
        j = i
    return inside


def _point_segment_dist(
    px: float, py: float,
    ax: float, ay: float,
    bx: float, by: float,
) -> float:
    vx, vy = bx - ax, by - ay
    L2 = vx * vx + vy * vy
    if L2 < 1e-9:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / L2))
    qx = ax + t * vx
    qy = ay + t * vy
    return math.hypot(px - qx, py - qy)


def _beam_axis(dx: float, dy: float) -> str:
    return "x" if abs(dx) >= abs(dy) else "y"


def _has_dashline_near(
    px: float, py: float,
    axis: str,
    beam_width_mm: float,
    ctx: _Ctx,
) -> bool:
    """
    Look for a beam centreline (dashed) that the endpoint sits on.

    The endpoint may extend up to half a beam width *past* either segment tip
    along the longitudinal direction (covers YOLO bboxes that overshoot the
    centreline) but must stay tight in the perpendicular direction (so we
    don't pick up a parallel beam's centreline one grid line over).
    """
    if axis == "x":
        segments = ctx.dashlines_x
        long_pos, perp_pos = px, py
    else:
        segments = ctx.dashlines_y
        long_pos, perp_pos = py, px
    if not segments:
        return False
    long_slack = beam_width_mm * _DASHLINE_LONG_FRACTION
    for seg_lo, seg_hi, seg_perp in segments:
        if abs(perp_pos - seg_perp) > _DASHLINE_PERP_MM:
            continue
        if seg_lo - long_slack <= long_pos <= seg_hi + long_slack:
            return True
    return False


def _find_column_anchor(
    pt_x: float, pt_y: float,
    ctx: _Ctx,
    radius_x: float,
    radius_y: float,
) -> tuple[float, float, float, float] | None:
    best: tuple[float, float, float, float] | None = None
    best_d = float("inf")
    for cx, cy, hw, hd in ctx.columns:
        if abs(pt_x - cx) > radius_x or abs(pt_y - cy) > radius_y:
            continue
        d = _dist2d(pt_x, pt_y, cx, cy)
        if d < best_d:
            best_d = d
            best = (cx, cy, hw, hd)
    return best


def _find_core_wall_anchor(pt_x: float, pt_y: float, ctx: _Ctx) -> bool:
    for cw in ctx.core_walls:
        if _point_in_polygon(pt_x, pt_y, cw):
            return True
    return False


def _find_rectangle_anchor(
    pt_x: float, pt_y: float,
    ctx: _Ctx,
    radius_x: float,
    radius_y: float,
) -> tuple[float, float] | None:
    best: tuple[float, float] | None = None
    best_d = float("inf")
    for cx, cy, _, _ in ctx.rectangles:
        if abs(pt_x - cx) > radius_x or abs(pt_y - cy) > radius_y:
            continue
        d = _dist2d(pt_x, pt_y, cx, cy)
        if d < best_d:
            best_d = d
            best = (cx, cy)
    return best


def _has_out_of_grid_endpoint(original: dict, ctx: _Ctx) -> bool:
    """True iff any pre-snap endpoint lies outside grid rect + 1-bay margin."""
    for k in _ENDPOINT_KEYS:
        ox = float(original.get(k, {}).get("x", 0.0))
        oy = float(original.get(k, {}).get("y", 0.0))
        if not (
            ctx.grid_x_min - ctx.dx_grid <= ox <= ctx.grid_x_max + ctx.dx_grid
            and ctx.grid_y_min - ctx.dy_grid <= oy <= ctx.grid_y_max + ctx.dy_grid
        ):
            return True
    return False


# ──────────────────────────────────────────────────────────────────────────
# Pass A: perpendicular grid-line snap → column / core-wall anchor.
# ──────────────────────────────────────────────────────────────────────────

def _snap_pass_a(
    pt: dict,
    axis: str,
    ctx: _Ctx,
) -> tuple[str, tuple[float, float, float, float] | None]:
    """
    Mutate *pt* in place if anchored. Returns:
      ("column",    col_tuple)
      ("core_wall", None)
      ("",          None)   — no anchor, fall through to Pass B
    """
    if not ctx.has_grid:
        col = _find_column_anchor(pt["x"], pt["y"], ctx, 1000.0, 1000.0)
        if col is not None:
            pt["x"], pt["y"] = col[0], col[1]
            return "column", col
        return "", None

    if _find_core_wall_anchor(pt["x"], pt["y"], ctx):
        return "core_wall", None

    if axis == "x":
        perp_lines, spacing, coord_key = ctx.y_lines, ctx.dy_grid, "y"
    else:
        perp_lines, spacing, coord_key = ctx.x_lines, ctx.dx_grid, "x"

    if not perp_lines or spacing <= 0:
        return "", None

    cur = pt[coord_key]
    nearest = min(perp_lines, key=lambda v: abs(cur - v))
    if abs(cur - nearest) > spacing * _PERP_SNAP_FRACTION:
        return "", None
    pt[coord_key] = nearest

    if _find_core_wall_anchor(pt["x"], pt["y"], ctx):
        return "core_wall", None

    radius = max(ctx.dx_grid, ctx.dy_grid) * _ALONG_ANCHOR_FRACTION
    col = _find_column_anchor(
        pt["x"], pt["y"], ctx,
        radius_x=radius if axis == "x" else ctx.dx_grid * 0.25,
        radius_y=ctx.dy_grid * 0.25 if axis == "x" else radius,
    )
    if col is not None:
        pt["x"], pt["y"] = col[0], col[1]
        return "column", col

    return "", None


# ──────────────────────────────────────────────────────────────────────────
# Pass B: dashline-confirmed extension. Pure geometry — caller formats logs.
# ──────────────────────────────────────────────────────────────────────────

def _snap_pass_b(
    pt: dict,
    axis: str,
    direction_sign: float,
    budget_units: int,
    beam_width_mm: float,
    ctx: _Ctx,
) -> tuple[str, tuple[float, float, float, float] | None, int]:
    """
    Walk outward, anchor on first hit. Mutates pt on success.

    Returns (kind, anchor_or_None, n_steps):
      ("column",    col_tuple,  n)
      ("core_wall", None,       n)
      ("rectangle", None,       n)
      ("no_dashline",        None, 0)  — no dashline near pt; not a real beam
      ("dashline_no_anchor", None, 0)  — confirmed but extension exhausted
    """
    if not _has_dashline_near(pt["x"], pt["y"], axis, beam_width_mm, ctx):
        return "no_dashline", None, 0

    step_x = ctx.dx_grid * direction_sign if axis == "x" else 0.0
    step_y = ctx.dy_grid * direction_sign if axis == "y" else 0.0
    rx = ctx.dx_grid * _STEP_ANCHOR_FRACTION
    ry = ctx.dy_grid * _STEP_ANCHOR_FRACTION

    for n in range(1, budget_units + 1):
        probe_x = pt["x"] + step_x * n
        probe_y = pt["y"] + step_y * n

        col = _find_column_anchor(probe_x, probe_y, ctx, rx, ry)
        if col is not None:
            pt["x"], pt["y"] = col[0], col[1]
            return "column", col, n

        if _find_core_wall_anchor(probe_x, probe_y, ctx):
            pt["x"], pt["y"] = probe_x, probe_y
            return "core_wall", None, n

        rect_hit = _find_rectangle_anchor(probe_x, probe_y, ctx, rx, ry)
        if rect_hit is not None:
            pt["x"], pt["y"] = rect_hit
            return "rectangle", None, n

    return "dashline_no_anchor", None, 0


# ──────────────────────────────────────────────────────────────────────────
# Snapshot + reject helpers
# ──────────────────────────────────────────────────────────────────────────

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
        "rescued_keys":   list(entry.get("rescued", {}).keys()),
    })


# ──────────────────────────────────────────────────────────────────────────
# Main pass — snap, anchor, validate, dedup, face-trim.
# ──────────────────────────────────────────────────────────────────────────

def _snap_and_filter_framing(
    recipe: dict,
    ctx: _Ctx,
) -> tuple[dict, list[str], Counter, list[dict]]:
    framing = recipe.get("structural_framing", [])
    actions:  list[str]  = []
    drops:    Counter    = Counter()
    rejected: list[dict] = []
    seen_pairs: set[frozenset[tuple[float, float]]] = set()

    if not framing:
        return recipe, actions, drops, rejected

    per_beam: list[dict] = []
    for i, beam in enumerate(framing):
        original = {
            k: dict(beam[k]) for k in _ENDPOINT_KEYS
            if isinstance(beam.get(k), dict)
        }
        snapped: dict[str, tuple[float, float, float, float]] = {}
        beam_actions: list[str] = []

        sp = beam.get("start_point")
        ep = beam.get("end_point")
        if not (isinstance(sp, dict) and isinstance(ep, dict)):
            per_beam.append({
                "beam": beam, "snapped": snapped, "actions": beam_actions,
                "original": original, "rescued": {},
            })
            continue

        dx_raw = float(ep.get("x", 0.0)) - float(sp.get("x", 0.0))
        dy_raw = float(ep.get("y", 0.0)) - float(sp.get("y", 0.0))
        axis = _beam_axis(dx_raw, dy_raw)

        for pt_key, pt in (("start_point", sp), ("end_point", ep)):
            kind, col = _snap_pass_a(pt, axis, ctx)
            if kind == "column" and col is not None:
                snapped[pt_key] = col
                beam_actions.append(
                    f"framing[{i}].{pt_key} snapped to column @ "
                    f"({col[0]:.0f}, {col[1]:.0f}) mm"
                )
            elif kind == "core_wall":
                snapped[pt_key] = (pt["x"], pt["y"], 0.0, 0.0)
                beam_actions.append(
                    f"framing[{i}].{pt_key} anchored to core wall @ "
                    f"({pt['x']:.0f}, {pt['y']:.0f}) mm"
                )

        # Both endpoints landed on the same column — un-snap the one whose
        # original position was further from the column centre. Leave its
        # coordinates at the column centre (a real grid intersection) so
        # Pass B's `pt + n·grid_spacing` probes land cleanly on neighbouring
        # column positions. The natural sign-from-pt-vs-other computation can't
        # tell which way to walk (both endpoints share the column position),
        # so we capture the direction here from the original endpoint.
        sp_col = snapped.get("start_point")
        ep_col = snapped.get("end_point")
        rescue_dir: dict[str, float] = {}
        if (
            sp_col is not None and ep_col is not None
            and sp_col[0] == ep_col[0] and sp_col[1] == ep_col[1]
            and sp_col[2] > 0 and ep_col[2] > 0
        ):
            cx, cy = sp_col[0], sp_col[1]
            d_sp = _dist2d(original["start_point"]["x"], original["start_point"]["y"], cx, cy)
            d_ep = _dist2d(original["end_point"]["x"],   original["end_point"]["y"],   cx, cy)
            unsnap_key = "start_point" if d_sp >= d_ep else "end_point"
            col_along = cx if axis == "x" else cy
            rescue_dir[unsnap_key] = (
                1.0 if original[unsnap_key][axis] >= col_along else -1.0
            )
            del snapped[unsnap_key]
            beam_actions.append(
                f"framing[{i}].{unsnap_key} un-snapped (would collapse onto same column "
                f"as the other end) — deferred to Pass B for outward grid walk"
            )

        per_beam.append({
            "beam": beam, "snapped": snapped, "actions": beam_actions,
            "original": original, "rescued": {}, "axis": axis,
            "rescue_dir": rescue_dir,
        })

    # Pass B — extend floating endpoints along beam axis if a dashline confirms.
    for i, entry in enumerate(per_beam):
        beam = entry["beam"]
        snapped = entry["snapped"]
        axis = entry.get("axis")
        if axis is None:
            continue

        floating = [k for k in _ENDPOINT_KEYS if k not in snapped]
        if not floating:
            continue

        budget = (
            _EXT_BUDGET_ONE_FLOAT if len(floating) == 1
            else _EXT_BUDGET_BOTH_FLOAT
        )

        sp = beam["start_point"]
        ep = beam["end_point"]
        beam_width_mm = float(beam.get("width") or _DEFAULT_BEAM_WIDTH_MM)
        rescue_dir = entry.get("rescue_dir") or {}
        for pt_key in floating:
            pt = beam[pt_key]
            other = ep if pt_key == "start_point" else sp
            override = rescue_dir.get(pt_key)
            if override is not None:
                sign = override
            else:
                sign = 1.0 if pt[axis] >= other[axis] else -1.0

            kind, col, n_steps = _snap_pass_b(pt, axis, sign, budget, beam_width_mm, ctx)
            if kind == "column" and col is not None:
                snapped[pt_key] = col
                entry["rescued"][pt_key] = "column"
                entry["actions"].append(
                    f"framing[{i}].{pt_key} extended {n_steps} grid unit(s) → "
                    f"column @ ({col[0]:.0f}, {col[1]:.0f}) mm"
                )
            elif kind in ("core_wall", "rectangle"):
                snapped[pt_key] = (pt["x"], pt["y"], 0.0, 0.0)
                entry["rescued"][pt_key] = kind
                entry["actions"].append(
                    f"framing[{i}].{pt_key} extended {n_steps} grid unit(s) → "
                    f"{kind} @ ({pt['x']:.0f}, {pt['y']:.0f}) mm"
                )
            else:
                entry.setdefault("fail_tags", {})[pt_key] = kind

    # Validate + dedup + face-trim + keep.
    kept: list[dict] = []
    for i, entry in enumerate(per_beam):
        beam = entry["beam"]
        snapped = entry["snapped"]
        beam_actions = entry["actions"]
        sp = beam.get("start_point")
        ep = beam.get("end_point")

        if not (isinstance(sp, dict) and isinstance(ep, dict)):
            _reject(actions, i, "missing endpoint dict",
                    "no_endpoints", drops, rejected, entry)
            continue

        if ctx.has_grid and _has_out_of_grid_endpoint(entry["original"], ctx):
            _reject(actions, i,
                "endpoint outside grid rect (would float in model)",
                "out_of_grid", drops, rejected, entry)
            continue

        missing = [k for k in _ENDPOINT_KEYS if k not in snapped]
        if missing:
            fail_tags = entry.get("fail_tags") or {}
            tag = fail_tags.get(missing[0]) or "no_dashline"
            reason = (
                "no dashline near floating endpoint — likely YOLO false positive"
                if tag == "no_dashline"
                else f"dashline confirmed but no anchor within "
                     f"{_EXT_BUDGET_ONE_FLOAT}-grid-unit budget"
            )
            _reject(actions, i, reason, tag, drops, rejected, entry)
            continue

        sp_anchor = snapped["start_point"]
        ep_anchor = snapped["end_point"]
        if sp_anchor[0] == ep_anchor[0] and sp_anchor[1] == ep_anchor[1]:
            _reject(actions, i, "both endpoints snapped to the same point",
                    "same_column", drops, rejected, entry)
            continue

        pair_key = frozenset({
            (round(sp_anchor[0], 1), round(sp_anchor[1], 1)),
            (round(ep_anchor[0], 1), round(ep_anchor[1], 1)),
        })
        if pair_key in seen_pairs:
            _reject(actions, i,
                "duplicate span — another beam already snapped to the same endpoints",
                "duplicate_span", drops, rejected, entry)
            continue
        seen_pairs.add(pair_key)

        dx = ep["x"] - sp["x"]
        dy = ep["y"] - sp["y"]
        if min(abs(dx), abs(dy)) > _AXIS_TOLERANCE_MM:
            _reject(actions, i,
                f"diagonal beam after snap (dx={dx:.0f}mm, dy={dy:.0f}mm > "
                f"tolerance {_AXIS_TOLERANCE_MM:.0f}mm)",
                "diagonal", drops, rejected, entry)
            continue

        # Face-trim only on column-anchored ends (half_w/half_d > 0).
        axis_is_x = abs(dx) >= abs(dy)
        if axis_is_x:
            sp_half, ep_half = sp_anchor[2], ep_anchor[2]
            direction = 1.0 if dx > 0 else -1.0
            sp["x"] += direction * sp_half
            ep["x"] -= direction * ep_half
        else:
            sp_half, ep_half = sp_anchor[3], ep_anchor[3]
            direction = 1.0 if dy > 0 else -1.0
            sp["y"] += direction * sp_half
            ep["y"] -= direction * ep_half

        length = math.hypot(ep["x"] - sp["x"], ep["y"] - sp["y"])
        if length < _MIN_BEAM_MM:
            _reject(actions, i,
                f"span {length:.0f} mm after face trim < minimum "
                f"{_MIN_BEAM_MM:.0f} mm",
                "too_short", drops, rejected, entry)
            continue

        if sp_half or ep_half:
            beam_actions.append(
                f"framing[{i}] trimmed to column faces "
                f"(−{sp_half:.0f} mm start, −{ep_half:.0f} mm end)"
            )

        kept.append(beam)
        actions.extend(beam_actions)

    recipe["structural_framing"] = kept
    return recipe, actions, drops, rejected


def _clamp_column_min_size(recipe: dict) -> tuple[dict, list[str]]:
    """Clamp column width/depth to _COL_MIN_MM (Revit rejects families below 200 mm)."""
    actions: list[str] = []
    for i, col in enumerate(recipe.get("columns", [])):
        for field_name in ("width", "depth"):
            v = float(col.get(field_name, 800.0))
            if v < _COL_MIN_MM:
                col[field_name] = _COL_MIN_MM
                actions.append(
                    f"column[{i}].{field_name} clamped {v:.0f} → {_COL_MIN_MM:.0f} mm"
                )
    return recipe, actions
