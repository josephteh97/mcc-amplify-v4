"""
Type Resolver Middleware — post-detection, pre-geometry.

Enriches each column detection dict with:
  resolved_type: "circular" | "rectangular" | "L-shape" | "unknown"
  nominal_diameter_mm: float | None   (circular only)
  nominal_width_mm: float | None      (rectangular only)
  nominal_depth_mm: float | None      (rectangular only)
  type_confidence: float

Input:  list[dict]  — raw detections from YOLO pipeline
Output: list[dict]  — same list, each dict mutated in place with type fields added

Does NOT modify "center", "bbox", "confidence", or "element_type" fields.
Does NOT affect coordinate values. Safe to call before GeometryGenerator.
"""
from __future__ import annotations
import cv2
import numpy as np
from loguru import logger

_CIRCULAR_CIRCULARITY_THRESHOLD = 0.78
_MIN_CONTOUR_AREA_PX = 30


def resolve_types(
    detections: list[dict],
    image: np.ndarray,
) -> list[dict]:
    """
    Enrich detection dicts with resolved column type and nominal dimensions.
    Safe to call even if detections is empty.
    """
    # Pre-convert to grayscale once to avoid per-crop cvtColor calls
    gray_full = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    for det in detections:
        try:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            crop = gray_full[y1:y2, x1:x2]
            if crop.size == 0:
                _set_unknown(det)
                continue
            resolved = _classify_crop(crop)
            det.update(resolved)
        except Exception as exc:
            logger.warning("Type resolution failed for detection: {}", exc)
            _set_unknown(det)
    return detections


_UNKNOWN = {
    "resolved_type": "unknown",
    "nominal_diameter_mm": None,
    "nominal_width_mm": None,
    "nominal_depth_mm": None,
    "type_confidence": 0.0,
}


def _set_unknown(det: dict) -> None:
    det.update(_UNKNOWN)


def _classify_crop(crop: np.ndarray) -> dict:
    """Classify a grayscale crop as circular, rectangular, or L-shape."""
    _, binary = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return dict(_UNKNOWN)

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < _MIN_CONTOUR_AREA_PX:
        return dict(_UNKNOWN)

    perimeter = cv2.arcLength(largest, True)
    circularity = (4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0
    h, w = crop.shape[:2]

    if circularity >= _CIRCULAR_CIRCULARITY_THRESHOLD:
        return {
            "resolved_type": "circular",
            "nominal_diameter_mm": None,   # filled by SemanticAnalyzer if available
            "nominal_width_mm": None,
            "nominal_depth_mm": None,
            "type_confidence": round(circularity, 3),
        }
    # Rectangular / L-shape — use bounding box aspect ratio as first heuristic
    aspect = w / h if h > 0 else 1.0
    resolved_type = "rectangular" if 0.5 <= aspect <= 2.0 else "L-shape"
    return {
        "resolved_type": resolved_type,
        "nominal_diameter_mm": None,
        "nominal_width_mm": w,   # pixel units — SemanticAnalyzer / grid scale converts later
        "nominal_depth_mm": h,
        "type_confidence": round(1.0 - circularity, 3),
    }
