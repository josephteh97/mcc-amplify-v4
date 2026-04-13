"""
Hybrid Fusion Layer: Combining Vector Precision with ML Intelligence

Current implementation
----------------------
Level 1 — Normalize ML detections to PDF point space (using rendered DPI).
Level 2 — Snap wall endpoints to nearby axis-aligned vector lines (practical
           improvement over raw YOLO boxes).
Level 3 — Semantic enrichment (handled by SemanticAnalyzer in Stage 5).

Output includes `refined_px` — the refined detections converted back to pixel
space so downstream stages (grid detector, geometry generator) can use them
directly without any coordinate-space confusion.
"""

import numpy as np
from typing import Dict, List, Any
from loguru import logger


class SpatialAlignmentEngine:
    """Convert between PDF point space and pixel space."""

    def __init__(self):
        self.dpi = 72
        self.scale_factor = 1.0

    def set_dpi(self, dpi: int):
        self.dpi = dpi
        self.scale_factor = dpi / 72.0

    def px_to_pt(self, coords: List[float]) -> List[float]:
        return [c / self.scale_factor for c in coords]

    def pt_to_px(self, coords: List[float]) -> List[float]:
        return [c * self.scale_factor for c in coords]

    def bbox_px_to_pt(self, bbox: List[float]) -> List[float]:
        return [c / self.scale_factor for c in bbox]

    def bbox_pt_to_px(self, bbox: List[float]) -> List[float]:
        return [c * self.scale_factor for c in bbox]


class HybridFusionPipeline:

    def __init__(self):
        self.aligner = SpatialAlignmentEngine()

    async def fuse(
        self,
        vector_data: Dict,
        ml_detections: List[Dict],
        metadata: Dict,
    ) -> Dict:
        dpi = metadata.get("dpi", 72)
        self.aligner.set_dpi(dpi)

        paths = vector_data.get("paths", [])
        logger.info(
            f"Fusion: {len(paths)} vector paths × {len(ml_detections)} ML detections "
            f"(DPI={dpi})"
        )

        # Level 1 — convert YOLO pixel bboxes to PDF point space
        pts_detections = self._normalize_to_points(ml_detections)

        # Level 2 — snap wall endpoints to axis-aligned vector lines
        refined_pts = self._snap_walls_to_vectors(pts_detections, paths)

        # Convert refined PDF-point detections back to pixel space for downstream
        refined_px = self._points_to_pixels(refined_pts)

        return {
            "refined_px":   refined_px,      # pixel space — for geometry/grid
            "refined_pts":  refined_pts,     # PDF point space — for debugging
            "raw_vectors":  vector_data,
            "metadata":     metadata,
        }

    # ── Level 1 ───────────────────────────────────────────────────────────────

    def _normalize_to_points(self, detections: List[Dict]) -> List[Dict]:
        """Convert pixel bboxes → PDF point bboxes."""
        result = []
        for det in detections:
            bbox_pt = self.aligner.bbox_px_to_pt(det["bbox"])
            result.append({**det, "bbox": bbox_pt, "geometry_source": "ml_approximate"})
        return result

    # ── Level 2 ───────────────────────────────────────────────────────────────

    def _snap_walls_to_vectors(
        self, detections: List[Dict], paths: List[Dict]
    ) -> List[Dict]:
        """
        For wall detections: find the dominant axis-aligned vector line inside
        the bbox and snap the wall endpoints to it.

        Vector lines that are horizontal (dy<dx) snap to a Y coordinate;
        vertical lines (dy>dx) snap to an X coordinate.
        If no matching vector is found the detection is returned unchanged.
        """
        if not paths:
            return detections

        # Build flat list of axis-aligned line segments from vector paths.
        # PyMuPDF drawing items are tuples: ('m', pt), ('l', pt), ('c', p1, p2, p3), …
        # We extract consecutive moveto/lineto pairs to find straight segments.
        h_lines: List[tuple] = []   # (x1, y, x2) — horizontal
        v_lines: List[tuple] = []   # (x, y1, y2) — vertical
        for path in paths:
            items = path.get("items", [])
            last_pt = None
            for item in items:
                op = item[0] if item else ""
                if op == "m":                        # move-to: new pen position
                    pt = item[1]
                    last_pt = (float(pt.x), float(pt.y)) if hasattr(pt, "x") else (float(pt[0]), float(pt[1]))
                elif op == "l" and last_pt:          # line-to: segment to extract
                    pt = item[1]
                    x2 = float(pt.x) if hasattr(pt, "x") else float(pt[0])
                    y2 = float(pt.y) if hasattr(pt, "x") else float(pt[1])
                    x1, y1 = last_pt
                    dx, dy = abs(x2 - x1), abs(y2 - y1)
                    if dy < 5 and dx > 20:
                        h_lines.append((min(x1, x2), (y1 + y2) / 2, max(x1, x2)))
                    elif dx < 5 and dy > 20:
                        v_lines.append(((x1 + x2) / 2, min(y1, y2), max(y1, y2)))
                    last_pt = (x2, y2)
                else:
                    last_pt = None  # curves break line continuity

        refined = []
        for det in detections:
            if det.get("type", "").lower() != "wall":
                refined.append(det)
                continue
            refined.append(self._snap_wall(det, h_lines, v_lines))
        return refined

    def _snap_wall(
        self,
        det: Dict,
        h_lines: List[tuple],
        v_lines: List[tuple],
    ) -> Dict:
        x1, y1, x2, y2 = det["bbox"]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w,  h  = x2 - x1, y2 - y1

        # Look for matching vector line overlapping the bbox
        if w >= h:  # horizontal wall — snap Y to a nearby h_line
            best = self._nearest_hline(cx, cy, x1, y1, x2, y2, h_lines)
            if best is not None:
                lx1, ly, lx2 = best
                snapped_bbox = [
                    max(x1, lx1), ly - h / 2,
                    min(x2, lx2), ly + h / 2,
                ]
                return {**det, "bbox": snapped_bbox, "geometry_source": "vector_snapped"}
        else:       # vertical wall — snap X to a nearby v_line
            best = self._nearest_vline(cx, cy, x1, y1, x2, y2, v_lines)
            if best is not None:
                lx, ly1, ly2 = best
                snapped_bbox = [
                    lx - w / 2, max(y1, ly1),
                    lx + w / 2, min(y2, ly2),
                ]
                return {**det, "bbox": snapped_bbox, "geometry_source": "vector_snapped"}

        return det  # no match — return unchanged

    def _nearest_hline(self, cx, cy, bbox_x1, bbox_y1, bbox_x2, bbox_y2, h_lines):
        """
        Find the nearest horizontal vector line whose Y sits inside the wall bbox.

        Overlap check (lx1 <= bbox_x2 and lx2 >= bbox_x1) instead of requiring
        the bbox center to lie within the line's X extent.  This correctly handles
        walls that extend beyond the endpoints of a vector line.
        """
        best, best_dist = None, float("inf")
        for line in h_lines:
            lx1, ly, lx2 = line
            if bbox_y1 <= ly <= bbox_y2 and lx1 <= bbox_x2 and lx2 >= bbox_x1:
                d = abs(ly - cy)
                if d < best_dist:
                    best, best_dist = line, d
        return best

    def _nearest_vline(self, cx, cy, bbox_x1, bbox_y1, bbox_x2, bbox_y2, v_lines):
        """
        Find the nearest vertical vector line whose X sits inside the wall bbox.

        Overlap check (ly1 <= bbox_y2 and ly2 >= bbox_y1) for the same reason
        as _nearest_hline — walls can be longer than the matching vector segment.
        """
        best, best_dist = None, float("inf")
        for line in v_lines:
            lx, ly1, ly2 = line
            if bbox_x1 <= lx <= bbox_x2 and ly1 <= bbox_y2 and ly2 >= bbox_y1:
                d = abs(lx - cx)
                if d < best_dist:
                    best, best_dist = line, d
        return best

    # ── Convert back to pixels ────────────────────────────────────────────────

    def _points_to_pixels(self, detections: List[Dict]) -> List[Dict]:
        """Convert refined PDF-point bboxes back to pixel space."""
        result = []
        for det in detections:
            bbox_px = self.aligner.bbox_pt_to_px(det["bbox"])
            result.append({**det, "bbox": bbox_px})
        return result
