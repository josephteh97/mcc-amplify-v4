"""
Cross-Element Validator Middleware — post type-resolution, pre-geometry.

Runs three checks on the enriched detection list:
  1. IoU overlap  — flag pairs with IoU > 0.5 (duplicate detections)
  2. Grid distance — flag detections whose pixel center is > max_grid_dist_px from any grid line
  3. Neighbourhood consensus — flag isolated columns (no neighbour within isolation_radius_px)

Adds to each detection dict:
  validation_flags: list[str]  — empty list = clean, else names of triggered checks
  is_valid: bool               — True if validation_flags is empty

Does NOT modify coordinates, type fields, or confidence. Does NOT discard detections —
the Validation Agent decides what to do with flagged elements.
"""
from __future__ import annotations
from itertools import combinations

import numpy as np
from loguru import logger

# Validation flag names — exported so downstream code (orchestrator's off-grid
# deletion pass) doesn't duplicate these literals.
OFF_GRID      = "off_grid"
ISOLATED      = "isolated"
IOU_OVERLAP   = "iou_overlap"


def validate_elements(
    detections: list[dict],
    grid_info: dict | None = None,
    max_grid_dist_px: float = 80.0,
    isolation_radius_px: float = 200.0,
    iou_threshold: float = 0.50,
) -> list[dict]:
    """
    Validate detections and attach validation_flags + is_valid to each dict.
    grid_info: the grid_info dict from GridDetector (optional; skips grid check if None).
    """
    for det in detections:
        det["validation_flags"] = []

    _check_iou_overlaps(detections, iou_threshold)
    if grid_info is not None:
        _check_grid_distance(detections, grid_info, max_grid_dist_px)
    _check_isolation(detections, isolation_radius_px)

    for det in detections:
        det["is_valid"] = len(det["validation_flags"]) == 0

    valid_count = sum(1 for d in detections if d["is_valid"])
    logger.info(
        "CrossElementValidator: %d/%d detections passed all checks",
        valid_count, len(detections),
    )
    return detections


def _iou(b1: list[float], b2: list[float]) -> float:
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter == 0:
        return 0.0
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter)


def _check_iou_overlaps(detections: list[dict], threshold: float) -> None:
    for i, j in combinations(range(len(detections)), 2):
        if _iou(detections[i]["bbox"], detections[j]["bbox"]) > threshold:
            for idx in (i, j):
                if IOU_OVERLAP not in detections[idx]["validation_flags"]:
                    detections[idx]["validation_flags"].append(IOU_OVERLAP)


def _check_grid_distance(
    detections: list[dict], grid_info: dict, max_dist_px: float
) -> None:
    """Flag detections whose center is more than max_dist_px from every grid line."""
    x_lines: list[float] = grid_info.get("x_lines_px", [])
    y_lines: list[float] = grid_info.get("y_lines_px", [])
    if not x_lines and not y_lines:
        return
    for det in detections:
        cx, cy = det["center"]
        dx = min(abs(cx - xl) for xl in x_lines) if x_lines else 0.0
        dy = min(abs(cy - yl) for yl in y_lines) if y_lines else 0.0
        if dx > max_dist_px and dy > max_dist_px:
            det["validation_flags"].append(OFF_GRID)


def _check_isolation(detections: list[dict], radius_px: float) -> None:
    """Flag columns with no neighbouring column within radius_px."""
    centers = np.array([d["center"] for d in detections], dtype=float)
    if len(centers) < 2:
        return
    for i, det in enumerate(detections):
        dists = np.linalg.norm(centers - centers[i], axis=1)
        dists[i] = np.inf
        if dists.min() > radius_px:
            det["validation_flags"].append(ISOLATED)
