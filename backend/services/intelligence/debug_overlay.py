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
from backend.services.intelligence.grid_coords import interp_sorted


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
    "no_dashline":         (0,   0,   255),  # red    — likely YOLO false positive
    "dashline_no_anchor":  (0,   140, 255),  # orange — real beam, no column found
    "out_of_grid":         (128, 0,   128),  # purple — endpoint outside grid rect
    "same_column":         (0,   200, 200),  # mustard
    "duplicate_span":      (255, 0,   255),  # magenta
    "diagonal":            (0,   255, 255),  # yellow
    "too_short":           (255, 255, 0),    # cyan
    "no_endpoints":        (128, 128, 128),  # grey
    # Legacy tag (pre-grid-aware sanitizer); kept so old debug runs still render.
    "floating_endpoint":   (0,   0,   255),
}


_PASS_A_COLOR = (0, 200, 0)        # green
_PASS_B_COLOR = (255, 140, 0)       # blue (BGR)


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
      - per endpoint:
          GREEN filled dot  = Pass A column / core-wall snap succeeded
          BLUE  filled dot  = Pass B dashline-confirmed extension succeeded
          HOLLOW dot in reject colour = endpoint floated (no anchor)
      - short "<id>:<tag>" label (reject reason)
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
    # X is already ascending; Y is descending after the world-flip — re-pair-sort
    # the Y axis so interp_sorted's bisect can use it.
    x_world = [sum(x_sp[:i]) for i in range(len(x_lines_px))]
    total_y = sum(y_sp)
    y_world = [total_y - sum(y_sp[:i]) for i in range(len(y_lines_px))]
    y_pairs = sorted(zip(y_world, y_lines_px))
    y_world_asc  = [p[0] for p in y_pairs]
    y_px_by_y_mm = [p[1] for p in y_pairs]

    def _mm_to_px(xm: float, ym: float) -> tuple[int, int]:
        return (
            int(round(interp_sorted(xm, x_world,      x_lines_px))),
            int(round(interp_sorted(ym, y_world_asc,  y_px_by_y_mm))),
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
        rescued = set(r.get("rescued_keys", []))
        for key, (xx, yy) in (("start_point", (x1, y1)), ("end_point", (x2, y2))):
            if key in rescued:
                cv2.circle(overlay, (xx, yy), 10, _PASS_B_COLOR, thickness=-1)
            elif key in snapped:
                cv2.circle(overlay, (xx, yy), 10, _PASS_A_COLOR, thickness=-1)
            else:
                cv2.circle(overlay, (xx, yy), 10, color, thickness=3)

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


