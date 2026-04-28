"""
Shared grid-coordinate helpers used by sanitizer + debug overlay.

`grid_info` arrays from grid_detector arrive sorted (post `_sort_lines`), so
both helpers assume monotonically increasing inputs and use bisect — avoids
re-sorting tens of thousands of times inside the recipe sanitizer's vector
feature extraction hot loop.
"""
from __future__ import annotations

from bisect import bisect_left


def interp_sorted(v: float, xs: list[float], ys: list[float]) -> float:
    """Piecewise-linear interpolate v through (xs → ys); xs must be sorted."""
    n = len(xs)
    if n == 0:
        return 0.0
    if n == 1:
        return float(ys[0])
    if v <= xs[0]:
        x0, x1, y0, y1 = xs[0], xs[1], ys[0], ys[1]
    elif v >= xs[-1]:
        x0, x1, y0, y1 = xs[-2], xs[-1], ys[-2], ys[-1]
    else:
        i = bisect_left(xs, v)
        if xs[i] == v:
            return float(ys[i])
        x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
    if x1 == x0:
        return float(y0)
    return y0 + (y1 - y0) * (v - x0) / (x1 - x0)


def grid_lines_world_mm(
    grid_info: dict | None,
) -> tuple[list[float], list[float], float, float]:
    """
    Return (x_lines_mm, y_lines_mm, dx_median, dy_median) in Revit world coords.

    y_lines_mm is Y-flipped (image-Y-down → Revit-Y-up) so it matches what
    `geometry_generator._px_to_world` produces for beam endpoints.

    Returns four empty/zero values when grid_info is missing or incomplete.
    """
    if not grid_info:
        return [], [], 0.0, 0.0
    x_lines_px = grid_info.get("x_lines_px") or []
    y_lines_px = grid_info.get("y_lines_px") or []
    x_sp = grid_info.get("x_spacings_mm") or []
    y_sp = grid_info.get("y_spacings_mm") or []
    if len(x_lines_px) < 2 or len(y_lines_px) < 2 or not x_sp or not y_sp:
        return [], [], 0.0, 0.0
    x_lines_mm = [sum(x_sp[:i]) for i in range(len(x_lines_px))]
    total_y = sum(y_sp)
    y_lines_mm = [total_y - sum(y_sp[:i]) for i in range(len(y_lines_px))]
    dx = sum(x_sp) / len(x_sp)
    dy = sum(y_sp) / len(y_sp)
    return x_lines_mm, y_lines_mm, dx, dy
