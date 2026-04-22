"""Grid-alignment signal — does the beam's long axis sit on a structural grid line?"""
from __future__ import annotations


def beam_axis_alignment(
    bbox_px: tuple[float, float, float, float],
    grid_info: dict,
    tolerance_px: float = 30.0,
) -> tuple[bool, str | None]:
    """
    Return (is_aligned, axis). axis is "x" (horizontal beam on a grid row)
    or "y" (vertical beam on a grid column) or None.
    """
    x1, y1, x2, y2 = bbox_px
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)

    x_lines = grid_info.get("x_lines_px") or []
    y_lines = grid_info.get("y_lines_px") or []

    if dx >= dy:
        # Horizontal beam — check if its y-centre matches a horizontal grid line
        for gy in y_lines:
            if abs(cy - gy) <= tolerance_px:
                return True, "x"
    else:
        for gx in x_lines:
            if abs(cx - gx) <= tolerance_px:
                return True, "y"
    return False, None
