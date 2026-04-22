"""Proximity signals — nearest neighbor of a given type."""
from __future__ import annotations

import math


def nearest_neighbor(
    center: tuple[float, float],
    detections: list[dict],
    of_type: str,
    exclude_id: str | None = None,
) -> tuple[dict | None, float]:
    """
    Return (neighbor_det, distance_px). If none found returns (None, inf).
    """
    best: dict | None = None
    best_d = math.inf
    cx, cy = center
    for d in detections:
        if d.get("type") != of_type:
            continue
        if exclude_id and d.get("id") == exclude_id:
            continue
        nc = d.get("center") or [0.0, 0.0]
        dist = math.hypot(nc[0] - cx, nc[1] - cy)
        if dist < best_d:
            best_d = dist
            best = d
    return best, best_d
