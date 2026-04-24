"""
Debug overlay renderer for admittance decisions.

Renders every framing element whose admittance decision is ADMIT_WITH_FIX
or REJECT so the user can see what the admittance agent did and why:

  - GREEN bbox   = admitted with geometry fix (e.g. snapped to column face)
  - RED   bbox   = rejected
  - YELLOW line  = links the beam centre to the conflicting column centre

The label on each box shows the element id + decision reason.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from backend.services.intelligence.admittance import ADMIT_WITH_FIX, REJECT


def save_join_conflict_overlay(
    image: np.ndarray,
    detections: list[dict],
    out_path: str | Path,
) -> int:
    """Write an overlay PNG highlighting admittance decisions on framing.

    Returns the number of elements drawn (0 = no overlay written).
    """
    interesting = [
        d for d in detections
        if d.get("type") == "structural_framing"
        and (d.get("admittance_decision") or {}).get("action") in (ADMIT_WITH_FIX, REJECT)
    ]
    if not interesting:
        return 0

    overlay = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for beam in interesting:
        bbox = beam.get("bbox") or []
        if len(bbox) < 4:
            continue
        x1, y1, x2, y2 = (int(v) for v in bbox)
        bc = beam.get("center") or [(x1 + x2) / 2, (y1 + y2) / 2]

        decision = beam.get("admittance_decision") or {}
        action   = decision.get("action", "")
        reason   = decision.get("reason", "")

        color = (0, 200, 0) if action == ADMIT_WITH_FIX else (0, 0, 255)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness=3)

        # Yellow line to conflicting column (if known)
        cc = (beam.get("admittance_metadata") or {}).get("conflict_column_center")
        if cc and len(cc) >= 2:
            cv2.line(overlay, (int(bc[0]), int(bc[1])),
                     (int(cc[0]), int(cc[1])), (0, 255, 255), thickness=2)
            cv2.circle(overlay, (int(cc[0]), int(cc[1])), 12, (0, 255, 255), thickness=2)

        label = f"{beam.get('id', '?')} {action}:{reason}"
        cv2.putText(overlay, label, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)
    logger.info(
        "Saved admittance debug overlay → {} ({} framing element(s) highlighted)",
        out_path, len(interesting),
    )
    return len(interesting)


# Per-tag colour (BGR) so the rejection reason is visible at a glance.
_REJECT_COLORS = {
    "floating_endpoint": (0,   0,   255),  # red
    "same_column":       (0,   140, 255),  # orange
    "duplicate_span":    (255, 0,   255),  # magenta
    "diagonal":          (0,   255, 255),  # yellow
    "too_short":         (255, 255, 0),    # cyan
}


def save_sanitizer_rejected_overlay(
    image: np.ndarray,
    rejected: list[dict],
    grid_info: dict,
    out_path: str | Path,
) -> int:
    """Render every sanitizer-rejected beam's pre-snap endpoints on the plan.

    Each rejected entry carries original (pre-snap) mm endpoints. We convert
    back to image pixels via grid_info and draw:
      - coloured line between the two endpoints (colour = reason tag)
      - small dot on each endpoint; a HOLLOW dot marks the endpoint(s) that
        failed to snap (the "floating" side) so secondary-onto-secondary and
        column-miss cases are distinguishable at a glance
      - short "<id>:<tag>" label
    """
    if not rejected:
        return 0

    x_lines_px = grid_info.get("x_lines_px") or []
    y_lines_px = grid_info.get("y_lines_px") or []
    x_sp       = grid_info.get("x_spacings_mm") or []
    y_sp       = grid_info.get("y_spacings_mm") or []
    if len(x_lines_px) < 2 or len(y_lines_px) < 2:
        logger.warning("Sanitizer overlay skipped — grid_info lacks line positions.")
        return 0

    # World-mm position of each grid line (matches geometry_generator._px_to_world).
    x_world = [sum(x_sp[:i]) for i in range(len(x_lines_px))]
    total_y = sum(y_sp)
    y_world = [total_y - sum(y_sp[:i]) for i in range(len(y_lines_px))]

    def _mm_to_px(xm: float, ym: float) -> tuple[int, int]:
        return (
            int(round(_interp(xm, x_world, x_lines_px))),
            int(round(_interp(ym, y_world, y_lines_px))),
        )

    overlay = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    drawn = 0
    for r in rejected:
        sp, ep = r.get("original_start"), r.get("original_end")
        if not (isinstance(sp, dict) and isinstance(ep, dict)):
            continue
        x1, y1 = _mm_to_px(float(sp["x"]), float(sp["y"]))
        x2, y2 = _mm_to_px(float(ep["x"]), float(ep["y"]))
        tag = r.get("tag", "")
        color = _REJECT_COLORS.get(tag, (0, 0, 255))
        cv2.line(overlay, (x1, y1), (x2, y2), color, 3)

        snapped = set(r.get("snapped_keys", []))
        for key, (xx, yy) in (("start_point", (x1, y1)), ("end_point", (x2, y2))):
            cv2.circle(overlay, (xx, yy), 10, color,
                       thickness=-1 if key in snapped else 3)

        label = f"{r.get('id', '?')}:{tag}"
        cv2.putText(overlay, label, (min(x1, x2), max(0, min(y1, y2) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        drawn += 1

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)
    logger.info(
        "Saved sanitizer-rejection overlay → {} ({} beam(s); filled=snapped, hollow=floated)",
        out_path, drawn,
    )
    return drawn


def _interp(v: float, xs: list[float], ys: list[int]) -> float:
    """Piecewise-linear interpolate v through (xs -> ys). xs need not be sorted."""
    pairs = sorted(zip(xs, ys), key=lambda p: p[0])
    xs_s = [p[0] for p in pairs]
    ys_s = [p[1] for p in pairs]
    if v <= xs_s[0]:
        x0, x1 = xs_s[0], xs_s[1]
        y0, y1 = ys_s[0], ys_s[1]
    elif v >= xs_s[-1]:
        x0, x1 = xs_s[-2], xs_s[-1]
        y0, y1 = ys_s[-2], ys_s[-1]
    else:
        i = 1
        while i < len(xs_s) and xs_s[i] < v:
            i += 1
        x0, x1 = xs_s[i - 1], xs_s[i]
        y0, y1 = ys_s[i - 1], ys_s[i]
    if x1 == x0:
        return float(y0)
    return y0 + (y1 - y0) * (v - x0) / (x1 - x0)
