"""
Classify the dominant stroke style (dashed vs solid) of vector paths inside
or adjacent to a bounding box.

Drafting convention we rely on (Singapore structural drawings, SS CP 65):
  - Reinforced-concrete beams are drawn with SOLID outlines.
  - Structural steel beams are drawn with DASHED outlines.

Fitz exposes each stroke's dash pattern via path["dashes"] — an SVG-style
string like "[3] 0" (dashed) or "[]" / None (solid).
"""
from __future__ import annotations


def _is_dashed(dash_attr) -> bool:
    if dash_attr is None:
        return False
    s = str(dash_attr).strip()
    # fitz returns "[] 0" for solid lines; any non-empty bracket content = dashed.
    if not s or s == "[] 0" or s == "[]":
        return False
    return "[" in s and "]" in s and s[s.index("[") + 1 : s.index("]")].strip() != ""


def _iter_candidate_paths(ctx, bbox_px: tuple[float, float, float, float]):
    """Yield paths in the 3×3 bucket neighborhood around bbox centre."""
    bucket = ctx.BUCKET_PX
    cx = (bbox_px[0] + bbox_px[2]) * 0.5
    cy = (bbox_px[1] + bbox_px[3]) * 0.5
    bx, by = int(cx // bucket), int(cy // bucket)
    buckets = ctx._paths_bucketed
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for p in buckets.get((bx + dx, by + dy), ()):
                yield p


def classify_stroke_style(
    bbox_px: tuple[float, float, float, float],
    ctx,
) -> str:
    """Return "dashed" | "solid" | "unknown" for paths overlapping the bbox."""
    pt_to_px = ctx.pt_to_px
    if pt_to_px <= 0 or not ctx._paths_bucketed:
        return "unknown"

    inv = 1.0 / pt_to_px
    x1p, y1p, x2p, y2p = (v * inv for v in bbox_px)

    dashed = solid = 0
    for p in _iter_candidate_paths(ctx, bbox_px):
        r = p.get("rect")
        if r is None:
            continue
        px0, py0, px1, py1 = (r.x0, r.y0, r.x1, r.y1) if hasattr(r, "x0") else r
        if px1 < x1p or px0 > x2p or py1 < y1p or py0 > y2p:
            continue
        if _is_dashed(p.get("dashes")):
            dashed += 1
        else:
            solid += 1

    if dashed == 0 and solid == 0:
        return "unknown"
    if dashed >= max(2, solid):
        return "dashed"
    if solid > dashed:
        return "solid"
    return "unknown"
