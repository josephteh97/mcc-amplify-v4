"""
Security & DoS Prevention Layer
"""

import os
import fitz  # PyMuPDF
import asyncio
import psutil
import gc
from typing import Dict, Optional
from loguru import logger
from async_timeout import timeout


class SecurityError(Exception):
    pass


class SecurePDFRenderer:
    """Aggressive DoS prevention for massive floor plans"""

    MAX_PIXEL_COUNT  = 25_000_000  # 25 MP — handles A0/A1 engineering drawings at 100-130 DPI
    MAX_MEMORY_MB    = 400         # bytes budget before mandatory tiling kicks in
    MAX_FILE_SIZE_MB = 100
    TIMEOUT_SECONDS  = 30

    ABSOLUTE_MIN_DPI = 72
    ABSOLUTE_MAX_DPI = 300

    def __init__(self):
        self.rejected_count      = 0
        self.tiling_forced_count = 0

    async def safe_render(self, pdf_path: str) -> Dict:
        """
        Inspect the PDF and decide the safe render strategy.
        Returns only metadata (dpi, method) — does NOT hold doc/page open.
        """

        # LAYER 1: file size
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"File not found: {pdf_path}")

        file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
        if file_size_mb > self.MAX_FILE_SIZE_MB:
            raise SecurityError(
                f"File too large: {file_size_mb:.1f}MB > {self.MAX_FILE_SIZE_MB}MB limit"
            )

        # LAYER 2: open with timeout, inspect, then CLOSE immediately
        try:
            async with timeout(self.TIMEOUT_SECONDS):
                doc = fitz.open(pdf_path)
                page = doc[0]
                width_inches  = page.rect.width  / 72.0
                height_inches = page.rect.height / 72.0
                doc.close()
                del doc, page
                fitz.TOOLS.store_shrink(0)
                gc.collect()
        except asyncio.TimeoutError:
            raise SecurityError("PDF parsing timeout — possible malicious file")
        except SecurityError:
            raise
        except Exception as e:
            raise SecurityError(f"Failed to open PDF: {e}")

        area_sq_ft = (width_inches * height_inches) / 144
        logger.info(
            f"📐 Page size: {width_inches:.1f}\" x {height_inches:.1f}\" "
            f"({area_sq_ft:.1f} sq ft)"
        )

        # LAYER 3: pick safe DPI
        safe_dpi = self._pick_safe_dpi(width_inches, height_inches)

        if safe_dpi is None:
            logger.warning("🔴 Page too large even at 72 DPI — mandatory tiling")
            self.tiling_forced_count += 1
            return {"method": "tiled", "dpi": 72}

        # LAYER 4: estimated memory check
        estimated_mb = self._estimate_mb(width_inches, height_inches, safe_dpi)
        if estimated_mb > self.MAX_MEMORY_MB:
            logger.warning(
                f"⚠️ Estimated {estimated_mb:.1f}MB > {self.MAX_MEMORY_MB}MB — forcing tiled"
            )
            self.tiling_forced_count += 1
            return {"method": "tiled", "dpi": safe_dpi}

        logger.info(f"✅ Direct render OK at {safe_dpi} DPI (~{estimated_mb:.0f}MB)")
        return {"method": "direct", "dpi": safe_dpi}

    # ── helpers ───────────────────────────────────────────────────────────────

    def _pick_safe_dpi(self, w_in: float, h_in: float) -> Optional[int]:
        for dpi in [300, 200, 150, 100, 72]:
            pixels = (w_in * dpi) * (h_in * dpi)
            mb     = (pixels * 3) / (1024 * 1024)
            if pixels <= self.MAX_PIXEL_COUNT and mb <= self.MAX_MEMORY_MB:
                if dpi < 100:
                    logger.warning(f"⚠️ Large plan — DPI reduced to {dpi}")
                elif dpi < 150:
                    logger.info(f"Large plan — rendering at {dpi} DPI")
                return dpi
        return None

    def _estimate_mb(self, w_in: float, h_in: float, dpi: int) -> float:
        """RGB bytes + 50 % MuPDF overhead."""
        pixels = (w_in * dpi) * (h_in * dpi)
        return (pixels * 3 * 1.5) / (1024 * 1024)


class ResourceMonitor:
    """Active memory monitoring during pipeline execution"""

    def __init__(self):
        self.peak_memory_mb = 0
        self.monitoring     = False

    def start(self):
        self.monitoring = True
        asyncio.create_task(self._monitor_loop())

    def stop(self):
        self.monitoring = False

    async def _monitor_loop(self):
        while self.monitoring:
            try:
                process   = psutil.Process(os.getpid())
                memory_mb = process.memory_info().rss / (1024 * 1024)
                self.peak_memory_mb = max(self.peak_memory_mb, memory_mb)

                if memory_mb > 4096:
                    logger.error(f"🔴 MEMORY EXCEEDED: {memory_mb:.0f}MB")

                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Monitor error: {e}")
                break
