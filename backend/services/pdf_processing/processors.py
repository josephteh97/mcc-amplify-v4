"""
PDF Processing Layer: Track A (Vector) & Track B (Streaming Raster)
"""
import fitz
import asyncio
import re
import numpy as np
import math
import gc
from typing import Dict, Any, List
from loguru import logger

from backend.services.security.secure_renderer import (
    SecurityError,
    SecurePDFRenderer,
)

# Reuse authoritative constants from the security layer
MAX_PIXELS            = SecurePDFRenderer.MAX_PIXEL_COUNT
MAX_MEMORY_MB         = SecurePDFRenderer.MAX_MEMORY_MB
MAX_DIMENSION_INCHES  = SecurePDFRenderer.MAX_DIMENSION_INCHES
MAX_ASPECT_RATIO      = SecurePDFRenderer.MAX_ASPECT_RATIO
TILE_PX               = 2000  # tile side in pixels (keeps each tile < ~12 MB)


class VectorProcessor:
    """Track A: Extract precise vector geometry from PDF."""

    def extract(self, pdf_path: str) -> Dict[str, Any]:
        logger.info("Extracting vector data…")
        doc = None
        try:
            doc  = fitz.open(pdf_path)
            page = doc[0]

            paths = page.get_drawings()
            vector_data: Dict[str, Any] = {"paths": [], "text": []}

            for path in paths:
                vector_data["paths"].append({
                    "type":  path.get("type", ""),
                    "items": path.get("items", []),
                    "color": path.get("color"),
                    "width": path.get("width", 0),
                    "rect":  path.get("rect"),
                })

            for block in page.get_text("dict")["blocks"]:
                if block["type"] == 0:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            vector_data["text"].append({
                                "text": span["text"],
                                "bbox": span["bbox"],
                                "size": span["size"],
                                "font": span["font"],
                            })

            # Store page dimensions in PDF points for the grid detector.
            # PyMuPDF normalises the origin to top-left, so rect.x0/y0 is
            # usually 0.  We store all four corners for robustness.
            vector_data["page_rect"] = [
                page.rect.x0, page.rect.y0,
                page.rect.x1, page.rect.y1,
            ]
            # Store page rotation (0, 90, 180, 270) so the grid detector can
            # swap its axes when the PDF was exported in a rotated orientation.
            vector_data["page_rotation"] = page.rotation

            logger.info(
                f"Extracted {len(vector_data['paths'])} paths, "
                f"{len(vector_data['text'])} text blocks, "
                f"page {page.rect.width:.0f}×{page.rect.height:.0f} pt"
            )
            return vector_data

        except Exception as e:
            logger.error(f"Vector extraction failed: {e}")
            raise
        finally:
            if doc is not None:
                doc.close()
                fitz.TOOLS.store_shrink(0)
                gc.collect()

    def extract_all_pages_text(self, pdf_path: str) -> List[dict]:
        """
        Extract text from every page of the PDF beyond page 0 and classify
        each page as a column schedule or not.

        A page is classified as a schedule when it contains ≥ 3 text items
        that each combine a column type-mark (e.g. "C1", "B3") with a
        dimension pattern ("800×800", "300∅", "Ø500").  This reliably picks
        up "Column Schedule" tabular pages without misidentifying notes pages.

        Returns:
            List of dicts, one per non-floor-plan page:
                {
                    "page_idx":     int,   # 0-based page index
                    "text_items":   list,  # [{"text": str, "bbox": list}, …]
                    "is_schedule":  bool,
                    "schedule_hits":int,   # number of type-mark+dimension matches
                }
            Empty list for single-page PDFs.
        """
        _re_dim  = re.compile(
            r'\d{2,4}\s*[xX×]\s*\d{2,4}'   # rectangular: 800x800
            r'|\d{2,4}\s*[Ø⌀∅]'            # number-first circular: 300∅
            r'|[Ø⌀∅]\s*\d{2,4}',           # symbol-first circular: Ø300
        )
        _re_mark = re.compile(r'\b[A-Z]{1,3}\d{1,3}\b')

        results = []
        doc = None
        try:
            doc = fitz.open(pdf_path)
            if len(doc) <= 1:
                return results     # single-page PDF — nothing extra to scan

            for page_idx in range(1, len(doc)):
                page = doc[page_idx]
                text_items = []
                for block in page.get_text("dict")["blocks"]:
                    if block["type"] == 0:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                text_items.append({
                                    "text": span["text"],
                                    "bbox": span["bbox"],
                                })

                schedule_hits = sum(
                    1 for t in text_items
                    if _re_mark.search(t["text"]) and _re_dim.search(t["text"])
                )
                is_schedule = schedule_hits >= 3

                results.append({
                    "page_idx":    page_idx,
                    "text_items":  text_items,
                    "is_schedule": is_schedule,
                    "schedule_hits": schedule_hits,
                })

                if is_schedule:
                    logger.info(
                        f"PDF page {page_idx + 1}: column schedule detected "
                        f"({schedule_hits} type+dimension entries, "
                        f"{len(text_items)} total text items)"
                    )
                else:
                    logger.debug(
                        f"PDF page {page_idx + 1}: not a schedule "
                        f"({schedule_hits} matches, {len(text_items)} text items)"
                    )

            return results

        except Exception as e:
            logger.warning(f"extract_all_pages_text failed: {e}")
            return []
        finally:
            if doc is not None:
                doc.close()


class StreamingProcessor:
    """Track B: Safe raster rendering with tiled memory management."""

    def __init__(self, ml_detector=None):
        self.ml_detector    = ml_detector
        self.vector_processor = VectorProcessor()

    async def render_safe(self, pdf_path: str, dpi: int = 150) -> Dict[str, Any]:
        """
        Render first page to a numpy RGB array.
        Uses tiled rendering when the full image would exceed MAX_MEMORY_MB.
        Never holds more than one tile + the canvas in RAM at once.
        """
        logger.info(f"Rendering PDF at {dpi} DPI…")
        doc = None
        try:
            doc  = fitz.open(pdf_path)
            page = doc[0]

            w_in = page.rect.width  / 72.0
            h_in = page.rect.height / 72.0

            # ── Page dimension attack prevention ──────────────────────────
            if w_in > MAX_DIMENSION_INCHES or h_in > MAX_DIMENSION_INCHES:
                raise SecurityError(
                    f"Page dimensions {w_in:.1f}\" x {h_in:.1f}\" exceed "
                    f"{MAX_DIMENSION_INCHES}\" limit — possible dimension attack"
                )

            short_side = min(w_in, h_in)
            long_side  = max(w_in, h_in)
            if short_side > 0 and (long_side / short_side) > MAX_ASPECT_RATIO:
                raise SecurityError(
                    f"Aspect ratio {long_side / short_side:.1f}:1 exceeds "
                    f"{MAX_ASPECT_RATIO}:1 limit — possible crafted PDF"
                )

            # ── Cap DPI so total pixels stay within budget ─────────────────
            natural_px = (w_in * dpi) * (h_in * dpi)
            if natural_px > MAX_PIXELS:
                dpi = max(72, int(math.sqrt(MAX_PIXELS / (w_in * h_in))))
                logger.warning(f"DPI capped to {dpi} to respect pixel budget")

            target_w = int(w_in * dpi)
            target_h = int(h_in * dpi)
            est_mb   = (target_w * target_h * 3 * 1.5) / (1024 * 1024)

            if est_mb <= MAX_MEMORY_MB:
                # ── Direct render ──────────────────────────────────────────
                logger.info(f"Direct render: {target_w}×{target_h} (~{est_mb:.0f}MB)")
                mat = fitz.Matrix(dpi / 72, dpi / 72)
                pix = page.get_pixmap(matrix=mat)
                img = np.frombuffer(pix.samples, dtype=np.uint8) \
                        .reshape(pix.height, pix.width, 3).copy()
                del pix
            else:
                # ── Tiled render ───────────────────────────────────────────
                logger.info(
                    f"Tiled render: {target_w}×{target_h} (~{est_mb:.0f}MB) "
                    f"using {TILE_PX}px tiles"
                )
                img = self._render_tiled(page, dpi, target_w, target_h)

            fitz.TOOLS.store_shrink(0)
            gc.collect()

            logger.info(f"Rendered: {img.shape[1]}×{img.shape[0]} at {dpi} DPI")
            return {"image": img, "width": img.shape[1], "height": img.shape[0], "dpi": dpi}

        except Exception as e:
            logger.error(f"Rendering failed: {e}")
            raise
        finally:
            if doc is not None:
                doc.close()
                fitz.TOOLS.store_shrink(0)
                gc.collect()

    # ── Tiled renderer ────────────────────────────────────────────────────────

    def _render_tiled(
        self,
        page,
        dpi: int,
        target_w: int,
        target_h: int,
    ) -> np.ndarray:
        """
        Render the page in TILE_PX × TILE_PX tiles and stitch into one array.
        Each tile is freed immediately after being copied into the canvas.
        """
        scale    = dpi / 72.0
        tile_pt  = TILE_PX / scale          # tile size in PDF points
        canvas   = np.zeros((target_h, target_w, 3), dtype=np.uint8)

        y_pt, y_px = 0.0, 0
        while y_pt < page.rect.height:
            x_pt, x_px = 0.0, 0
            while x_pt < page.rect.width:
                clip = fitz.Rect(
                    x_pt, y_pt,
                    min(x_pt + tile_pt, page.rect.width),
                    min(y_pt + tile_pt, page.rect.height),
                )
                mat  = fitz.Matrix(scale, scale)
                tile = page.get_pixmap(matrix=mat, clip=clip)
                arr  = np.frombuffer(tile.samples, dtype=np.uint8) \
                         .reshape(tile.height, tile.width, 3)

                h, w = arr.shape[:2]
                canvas[y_px:y_px + h, x_px:x_px + w] = arr

                del tile, arr          # free tile immediately
                fitz.TOOLS.store_shrink(0)

                x_pt += tile_pt
                x_px += TILE_PX

            y_pt += tile_pt
            y_px += TILE_PX
            gc.collect()               # per-row GC pass

        return canvas
