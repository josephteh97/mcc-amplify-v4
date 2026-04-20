"""
YoloDetectionAgent — generic tiling-YOLO detection for any structural element type.

Parameterise at construction time with the element type label and the filter
thresholds appropriate for that element's shape:

  column:             min_squareness=0.75  max_side=80   (square, small)
  structural_framing: min_squareness=0.0   max_side=300  (rectangular, longer)
"""
from __future__ import annotations
import asyncio
from loguru import logger

from backend.services.yolo_runner import run_yolo
from .base import DetectionAgent, DetectionContext


class YoloDetectionAgent(DetectionAgent):
    """
    Runs tiling YOLO inference for a single structural element type.
    Pass min_squareness=0.0 to disable the squareness filter (e.g. for beams).
    """

    def __init__(
        self,
        yolo_model,
        element_type: str,
        *,
        min_squareness: float = 0.75,
        min_side: int = 10,
        max_side: int = 80,
    ):
        self.element_type = element_type
        self._yolo = yolo_model
        self._filter_kwargs = {
            "min_squareness": min_squareness,
            "min_side":       min_side,
            "max_side":       max_side,
        }

    async def detect(self, ctx: DetectionContext) -> list[dict]:
        if self._yolo is None:
            logger.warning(
                "YoloDetectionAgent({}): no model loaded — skipping", self.element_type
            )
            return []
        image_data = {
            "image":  ctx.image,
            "width":  ctx.image.shape[1],
            "height": ctx.image.shape[0],
            "dpi":    ctx.image_dpi,
        }
        detections = await asyncio.to_thread(
            run_yolo,
            self._yolo,
            image_data,
            self.element_type,
            **self._filter_kwargs,
        )
        logger.info(
            "YoloDetectionAgent({}): {} detection(s)", self.element_type, len(detections)
        )
        return detections
