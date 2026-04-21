"""
Debug overlay renderer for ValidationAgent exclusions.

Given the rendered PDF raster and the post-enforce_rules detection list,
saves a PNG with rejected beams drawn in red and a line back to the column
centre that triggered the proximity check. Lets the user visually locate
which beams were dropped to avoid "Cannot keep elements joined" in Revit.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from loguru import logger


def save_join_conflict_overlay(
    image: np.ndarray,
    detections: list[dict],
    out_path: str | Path,
) -> int:
    """Write an overlay PNG highlighting beams flagged beam_column_join_conflict.

    Returns the number of conflicts drawn (0 = overlay skipped, no file written).
    """
    conflicts = [
        d for d in detections
        if d.get("type") == "structural_framing"
        and "beam_column_join_conflict" in d.get("dfma_violations", [])
    ]
    if not conflicts:
        return 0

    overlay = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for beam in conflicts:
        bbox = beam.get("bbox") or []
        if len(bbox) < 4:
            continue
        x1, y1, x2, y2 = (int(v) for v in bbox)
        bc = beam.get("center") or [(x1 + x2) / 2, (y1 + y2) / 2]
        cc = beam.get("conflict_column_center")

        # Red filled bbox for the rejected beam
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), thickness=3)

        # Yellow line from beam centre to the conflicting column centre
        if cc and len(cc) >= 2:
            cv2.line(overlay, (int(bc[0]), int(bc[1])),
                     (int(cc[0]), int(cc[1])), (0, 255, 255), thickness=2)
            cv2.circle(overlay, (int(cc[0]), int(cc[1])), 12, (0, 255, 255), thickness=2)

        label = beam.get("id", "?")
        cv2.putText(overlay, label, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)
    logger.info(
        "Saved join-conflict debug overlay → {} ({} beam(s) highlighted)",
        out_path, len(conflicts),
    )
    return len(conflicts)
