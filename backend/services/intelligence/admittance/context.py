"""ElementContext — bundle of everything signals/rules need beyond the detection itself."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ElementContext:
    """
    Read-only context passed into each rule. Rules never mutate this.

    vector_data      — output of VectorProcessor.extract (paths + text)
    grid_info        — grid detector output (x_lines_px, y_lines_px, spacings)
    legend_map       — {tag → material} built by legend_parser (e.g. "RCB3"→"rc")
    raster           — optional rendered page (H×W×3 BGR) for vision fallbacks
    page_width_pt,
    page_height_pt   — PDF point dimensions (for pt↔px scaling)
    dpi              — render DPI used for raster

    Spatial indices (populated by admittance.judge() before rule dispatch):
    _paths_bucketed  — dict[(bx, by) → list[path]] keyed by 256-px tile on
                       path rect centre, so dashline lookup touches a handful
                       of paths instead of ~110 000.
    _tag_spans       — pre-filtered text spans matching the structural-mark
                       regex (small list, avoids regex work per detection).
    """
    vector_data: dict[str, Any] = field(default_factory=dict)
    grid_info:   dict[str, Any] = field(default_factory=dict)
    legend_map:  dict[str, str] = field(default_factory=dict)
    raster:      np.ndarray | None = None
    page_width_pt:  float = 0.0
    page_height_pt: float = 0.0
    dpi:            float = 300.0

    _paths_bucketed: dict[tuple[int, int], list[dict]] = field(default_factory=dict)
    _tag_spans:      list[dict] = field(default_factory=list)

    BUCKET_PX: int = 256

    @property
    def pt_to_px(self) -> float:
        """PDF point → raster pixel scale (same for x and y at 300 DPI)."""
        return self.dpi / 72.0
