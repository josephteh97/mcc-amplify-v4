"""
Detection Agent base class and shared context.

Each structural element type has its own DetectionAgent subclass.
All agents receive the same DetectionContext and return detection dicts
with at minimum: {element_type, bbox, center, confidence}.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np


@dataclass
class DetectionContext:
    """Immutable input bundle passed to every detection agent."""
    pdf_path: str
    image: np.ndarray           # full-page RGB raster at image_dpi
    image_dpi: int
    vector_data: dict           # VectorProcessor.extract() output
    schedule_texts: list[str] = field(default_factory=list)


class DetectionAgent(ABC):
    """Base for all per-element structural detection agents."""

    #: Element type string produced by this agent (used as "type" key in dicts).
    element_type: str

    @abstractmethod
    async def detect(self, ctx: DetectionContext) -> list[dict]:
        """
        Run detection and return element dicts.

        Each dict must contain:
          element_type : str              — matches self.element_type
          bbox         : [x1, y1, x2, y2] — pixel coords
          center       : [cx, cy]          — pixel coords
          confidence   : float

        Return [] when no trained model exists yet.
        """
        ...


class UntrainedDetectionAgent(DetectionAgent):
    """
    Placeholder for an agent whose model is not yet trained.
    Returns [] immediately; no GPU/CPU work performed.
    Replace with a real agent class once a trained model is available.
    """

    def __init__(self, element_type: str):
        self.element_type = element_type

    async def detect(self, ctx: DetectionContext) -> list[dict]:
        return []
