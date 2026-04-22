"""
Find the nearest structural annotation text span to a detection and match it
against the legend map.

Structural drawings label each beam/column with a family tag (e.g. "RCB3",
"H-RCB1", "SB2") placed next to or on top of the element.
"""
from __future__ import annotations

import math
import re


# Matches tags like RCB3, H-RCB1, SB2, RC-B4, etc. Must contain ≥1 digit.
_TAG_RE = re.compile(r"^[A-Z]{1,3}[-]?[A-Z]{0,3}\d{1,3}[A-Z]?$")


def looks_like_tag(text: str) -> bool:
    s = (text or "").strip().upper()
    return bool(s) and len(s) <= 10 and _TAG_RE.match(s) is not None


def find_nearest_tag(
    bbox_px: tuple[float, float, float, float],
    ctx,
    max_dist_px: float = 80.0,
) -> tuple[str | None, str | None, float]:
    """
    Return (tag, material, distance_px).

    Uses ctx._tag_spans (pre-filtered) to avoid re-running the tag regex
    on every span for every detection.
    """
    spans = ctx._tag_spans
    pt_to_px = ctx.pt_to_px
    if not spans or pt_to_px <= 0:
        return None, None, math.inf

    legend_map = ctx.legend_map
    cx = (bbox_px[0] + bbox_px[2]) / 2.0
    cy = (bbox_px[1] + bbox_px[3]) / 2.0

    best: tuple[str, float] | None = None
    for span in spans:
        raw = (span.get("text") or "").strip()
        b = span.get("bbox")
        if not b or len(b) < 4:
            continue
        # text bbox is in PDF pts; convert to px
        tcx = (b[0] + b[2]) / 2.0 * pt_to_px
        tcy = (b[1] + b[3]) / 2.0 * pt_to_px
        d = math.hypot(tcx - cx, tcy - cy)
        if d <= max_dist_px and (best is None or d < best[1]):
            best = (raw.upper(), d)

    if best is None:
        return None, None, math.inf

    tag, dist = best
    material = None
    if legend_map:
        material = legend_map.get(tag)
    if material is None:
        # Prefix heuristic: anything with "RC" → rc, "SB" or "S-" → steel.
        upper = tag.upper()
        if "RC" in upper:
            material = "rc"
        elif upper.startswith("SB") or upper.startswith("S-"):
            material = "steel"
    return tag, material, dist
