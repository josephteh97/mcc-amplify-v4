"""Shared signal detectors — primitives each rule module composes."""
from backend.services.intelligence.admittance.signals.dashline import classify_stroke_style
from backend.services.intelligence.admittance.signals.legend_tag import find_nearest_tag
from backend.services.intelligence.admittance.signals.grid_alignment import beam_axis_alignment
from backend.services.intelligence.admittance.signals.proximity import nearest_neighbor

__all__ = [
    "classify_stroke_style",
    "find_nearest_tag",
    "beam_axis_alignment",
    "nearest_neighbor",
]
