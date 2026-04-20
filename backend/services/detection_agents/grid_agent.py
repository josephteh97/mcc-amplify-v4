"""
Grid Detection Agent — derives the structural column grid from PDF vector geometry.

Returns grid_info (coordinate metadata) rather than element dicts, so it exposes
detect_grid() instead of detect(). GridDimensionMissingError is never swallowed —
the pipeline must abort if real mm annotations cannot be found.
"""
from __future__ import annotations
import asyncio
from loguru import logger

from backend.services.grid_detector import GridDetector, GridDimensionMissingError
from .base import DetectionContext


class GridDetectionAgent:
    """Wraps GridDetector to run alongside element detection agents."""

    def __init__(self, grid_detector: GridDetector):
        self._detector = grid_detector

    async def detect_grid(self, ctx: DetectionContext) -> dict:
        """
        Return grid_info dict from structural grid detection.
        Raises GridDimensionMissingError if dimension annotations are absent —
        caller must not proceed without real coordinates.
        """
        image_data = {
            "image":  ctx.image,
            "width":  ctx.image.shape[1],
            "height": ctx.image.shape[0],
            "dpi":    ctx.image_dpi,
        }
        try:
            grid_info = await asyncio.to_thread(
                self._detector.detect, ctx.vector_data, image_data,
            )
        except GridDimensionMissingError:
            raise
        except Exception as exc:
            logger.warning(f"GridAgent: detection failed ({exc}) — using fallback grid")
            return self._detector._fallback_grid(ctx.image.shape[1], ctx.image.shape[0])

        logger.info(
            "GridAgent: {} V-lines × {} H-lines  source={}  confidence={:.2f}",
            len(grid_info.get("x_lines_px", [])),
            len(grid_info.get("y_lines_px", [])),
            grid_info.get("source", "unknown"),
            grid_info.get("grid_confidence", 0.0),
        )
        return grid_info
