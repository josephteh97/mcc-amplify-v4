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
