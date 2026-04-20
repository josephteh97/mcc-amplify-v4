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
    """Aggressive DoS prevention for massive floor plans.

    Supports standard engineering formats up to A0 (841mm x 1189mm,
    ~33.1" x 46.8") and comparable ANSI E / Arch E sizes.  At 300 DPI
    an A0 page would produce ~139 MP, far exceeding the 25 MP budget,
    so the renderer automatically steps DPI down until the pixel count
    fits.  Intermediate DPI values (250, 125) ensure A0 sheets land
    near the optimal ~127 DPI sweet spot instead of jumping straight
    to 100 DPI.

    Hardened against:
        - Page dimension attacks: pages wider or taller than 60 inches
          (1524 mm) are rejected outright — no standard engineering
          format exceeds this.
        - Aspect ratio attacks: pages with an aspect ratio > 4:1 are
          rejected; legitimate engineering drawings are <=~ 2:1.
        - Embedded resource bombs: if page 0 contains > 200 embedded
          objects (images + fonts) the renderer forces tiled mode to
          limit peak memory from resource decompression.
        - DPI inflation: DPI is always capped so total pixel count
          stays within MAX_PIXEL_COUNT (150 MP).
    """

    MAX_PIXEL_COUNT  = 150_000_000  # 150 MP — 300 DPI at A0/ANSI-E; larger sheets are DPI-reduced then tiled
    MAX_MEMORY_MB    = 600          # MB budget before mandatory tiling kicks in
    MAX_FILE_SIZE_MB = 100
    TIMEOUT_SECONDS  = 30

    ABSOLUTE_MIN_DPI = 72
    ABSOLUTE_MAX_DPI = 300

    MAX_DIMENSION_INCHES  = 60    # reject pages wider or taller than this
    MAX_ASPECT_RATIO      = 4.0   # reject pages with aspect ratio > 4:1
    MAX_EMBEDDED_OBJECTS  = 200   # force tiled rendering above this count

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
        embedded_object_count = 0
        try:
            async with timeout(self.TIMEOUT_SECONDS):
                doc = fitz.open(pdf_path)
                page = doc[0]
                width_inches  = page.rect.width  / 72.0
                height_inches = page.rect.height / 72.0

                # Count embedded objects (images + fonts) on page 0
                embedded_object_count = len(page.get_images(full=True)) + len(page.get_fonts())

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

        # LAYER 2a: page dimension attack prevention
        if width_inches > self.MAX_DIMENSION_INCHES or height_inches > self.MAX_DIMENSION_INCHES:
            self.rejected_count += 1
            raise SecurityError(
                f"Page dimensions {width_inches:.1f}\" x {height_inches:.1f}\" exceed "
                f"{self.MAX_DIMENSION_INCHES}\" limit — possible dimension attack"
            )

        # LAYER 2b: aspect ratio attack prevention
        short_side = min(width_inches, height_inches)
        long_side  = max(width_inches, height_inches)
        if short_side > 0 and (long_side / short_side) > self.MAX_ASPECT_RATIO:
            self.rejected_count += 1
            raise SecurityError(
                f"Aspect ratio {long_side / short_side:.1f}:1 exceeds "
                f"{self.MAX_ASPECT_RATIO}:1 limit — possible crafted PDF"
            )

        area_sq_ft = (width_inches * height_inches) / 144
        logger.info(
            f"📐 Page size: {width_inches:.1f}\" x {height_inches:.1f}\" "
            f"({area_sq_ft:.1f} sq ft)"
        )

        # LAYER 2c: embedded resource bomb detection
        force_tiled = False
        if embedded_object_count > self.MAX_EMBEDDED_OBJECTS:
            logger.warning(
                f"⚠️ {embedded_object_count} embedded objects on page 0 "
                f"(threshold {self.MAX_EMBEDDED_OBJECTS}) — forcing tiled render"
            )
            force_tiled = True

        # LAYER 3: pick safe DPI
        safe_dpi = self._pick_safe_dpi(width_inches, height_inches)

        if safe_dpi is None:
            logger.warning("🔴 Page too large even at 72 DPI — mandatory tiling")
            self.tiling_forced_count += 1
            return {"method": "tiled", "dpi": 72}

        # LAYER 4: estimated memory check + embedded resource bomb override
        estimated_mb = self._estimate_mb(width_inches, height_inches, safe_dpi)
        if force_tiled or estimated_mb > self.MAX_MEMORY_MB:
            reason = (
                f"embedded resource bomb ({embedded_object_count} objects)"
                if force_tiled
                else f"estimated {estimated_mb:.1f}MB > {self.MAX_MEMORY_MB}MB"
            )
            logger.warning(f"⚠️ {reason} — forcing tiled render at {safe_dpi} DPI")
            self.tiling_forced_count += 1
            return {"method": "tiled", "dpi": safe_dpi}

        logger.info(f"✅ Direct render OK at {safe_dpi} DPI (~{estimated_mb:.0f}MB)")
        return {"method": "direct", "dpi": safe_dpi}

    # ── helpers ───────────────────────────────────────────────────────────────

    def _pick_safe_dpi(self, w_in: float, h_in: float) -> Optional[int]:
        # Intermediate steps (250, 125) give much better A0 results:
        # A0 at 127 DPI = ~24.9 MP (just under 25 MP budget).
        # The old 300->200->150->100 ladder jumped from 150 DPI (too many
        # pixels for A0) straight to 100 DPI, missing the sweet spot.
        for dpi in [300, 250, 200, 150, 125, 100, 72]:
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

    # Warn at 75% of physical RAM (minimum 4 GB so small-machine installs still get warned)
    _WARN_THRESHOLD_MB: int = max(4096, int(psutil.virtual_memory().total / 1024 / 1024 * 0.75))

    async def _monitor_loop(self):
        while self.monitoring:
            try:
                process   = psutil.Process(os.getpid())
                memory_mb = process.memory_info().rss / (1024 * 1024)
                self.peak_memory_mb = max(self.peak_memory_mb, memory_mb)

                if memory_mb > self._WARN_THRESHOLD_MB:
                    logger.error(
                        f"🔴 MEMORY EXCEEDED: {memory_mb:.0f}MB "
                        f"(threshold {self._WARN_THRESHOLD_MB}MB / "
                        f"{psutil.virtual_memory().total // 1024 // 1024}MB total)"
                    )

                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Monitor error: {e}")
                break
