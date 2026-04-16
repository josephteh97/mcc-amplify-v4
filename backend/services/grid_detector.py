"""
Grid Line Detector — PyMuPDF text extraction (v26, 2026-03-20)

Detection logic
===============
Grid label text is read directly from the PDF vector layer using text items
already extracted by VectorProcessor (vector_data["text"]).  No image
processing, no deep-learning model, no Hough circles.

  - V-lines  (vertical,   numbered):  integer labels 1 … N in the top/bottom margins
  - H-lines  (horizontal, lettered):  1-2 char alpha labels A … DD in the left/right margins

Label + position pairing (v26 approach)
-----------------------------------------
Each grid label is paired with its pixel coordinate **at detection time**, not
in a separate second pass.  For every unique label string found in the margin:

    position = median pixel coordinate of all occurrences of that label

No separate clustering step.  No label-to-position matching pass.
The label and its coordinate are inseparable from the moment of detection.

This approach is robust to:
  - Labels that appear in only one margin (common for double-letter labels
    like AA, BB that only print on one side of the drawing).
  - Small positional variations between left/right (or top/bottom) copies.
  - Any label naming convention — labels are treated as opaque strings.

Sorting for gap computation is by numeric pixel coordinate only, never by
label string.  This avoids any alphabetical ordering bugs for sequences like
…Y, Z, AA, BB… where naïve string sort would misplace Z after AA.

Coordinate transform (rotation-aware)
--------------------------------------
PyMuPDF returns text bbox coordinates in the page's natural (pre-rotation) space.
The rendered image pixel space is derived from the display rectangle.

  rotation=90  (stored portrait, displayed landscape — most engineering drawings):
      image_x = (page_w_pt − cy_pdf) × scale
      image_y = cx_pdf × scale

  rotation=0  (no rotation):
      image_x = cx_pdf × scale
      image_y = cy_pdf × scale

  rotation=180:
      image_x = (page_w_pt − cx_pdf) × scale
      image_y = (page_h_pt − cy_pdf) × scale

  rotation=270:
      image_x = cy_pdf × scale
      image_y = (page_h_pt − cx_pdf) × scale

where (cx_pdf, cy_pdf) = bbox centre in natural PDF coords,
      page_w_pt        = page.rect.width  (display width in points),
      scale            = dpi / 72.

Spacing detection (modal-based, v26)
--------------------------------------
Structural drawings commonly print both individual bay annotations (e.g. 8400)
AND cumulative annotations (e.g. 16800, 25200) in the same margin region.
All candidate dimension values are collected per gap, then the value closest to
the **modal annotation** across all gaps is selected.  This robustly prefers
individual bay dimensions without any hard-coded threshold.

Output interface (unchanged from previous version)
---------------------------------------------------
  x_lines_px     — V-line pixel X positions, left→right
  y_lines_px     — H-line pixel Y positions, top→bottom
  x_labels       — numeric label strings ("1", "2", …)
  y_labels       — alpha   label strings ("A", "B", …, "AA", "BB", …)
  x_spacings_mm  — dimension-annotation spacing per V-bay (mm)
  y_spacings_mm  — dimension-annotation spacing per H-bay (mm)
  origin_px      — (x_lines_px[0], y_lines_px[0])
  px_per_mm_x    — pixels-per-mm along X axis
  px_per_mm_y    — pixels-per-mm along Y axis
  pixels_per_mm  — average of both axes
  page_rotation  — int (0 / 90 / 180 / 270)
  has_grid       — True
  source         — "text_labels"
  scale_string   — "grid-derived"
"""

import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger


class GridDimensionMissingError(ValueError):
    """Raised when grid lines are found but dimension annotations are missing."""


# ── Constants ──────────────────────────────────────────────────────────────────
DEFAULT_BAY_MM  = 6000
FALLBACK_BAY_MM = 8400
_DIM_RE         = re.compile(r'^\s*(\d{3,5})\s*$')
_H_LABEL_RE     = re.compile(r'^[A-Za-z]{1,2}$')


# ── Main class ─────────────────────────────────────────────────────────────────

class GridDetector:

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, vector_data: Dict, image_data: Dict) -> Dict:
        dpi   = float(image_data.get("dpi", 150))
        img_w = int(image_data["width"])
        img_h = int(image_data["height"])
        scale = dpi / 72.0          # pixels per point

        page_rect = vector_data.get("page_rect")
        if page_rect:
            page_x0   = float(page_rect[0])
            page_y0   = float(page_rect[1])
            disp_w_pt = max(float(page_rect[2]) - page_x0, 1.0)
            disp_h_pt = max(float(page_rect[3]) - page_y0, 1.0)
        else:
            page_x0, page_y0, disp_w_pt, disp_h_pt = self._estimate_page_bounds(
                vector_data.get("paths", [])
            )

        page_rotation = int(vector_data.get("page_rotation", 0))
        texts         = vector_data.get("text", [])

        logger.info(
            f"Grid detection (text-based v26): page {disp_w_pt:.0f}×{disp_h_pt:.0f} pt, "
            f"rotation={page_rotation}°"
        )

        # ── Detect V-lines (numbered) and H-lines (lettered) from text labels ─
        x_lines_px, x_labels, y_lines_px, y_labels = self._extract_lines_from_text(
            texts, img_w, img_h, scale, disp_w_pt, disp_h_pt, page_rotation
        )

        if len(x_lines_px) < 2 or len(y_lines_px) < 2:
            logger.warning("Too few grid lines detected — using fallback grid.")
            return self._fallback_grid(img_w, img_h)

        # Sort V-lines left→right, H-lines top→bottom (by pixel coordinate only)
        x_lines_px, x_labels = self._sort_lines(x_lines_px, x_labels, numeric=True)
        y_lines_px, y_labels = self._sort_lines(y_lines_px, y_labels, numeric=False)

        logger.info(f"FINAL X labels (V-lines, numbered, left→right): {x_labels}")
        logger.info(f"FINAL Y labels (H-lines, lettered, top→bottom): {y_labels}")

        # ── Spacings from dimension annotations ─────────────────────────────────
        x_spacings_mm = self._detect_spacings(
            x_lines_px, "x", texts, scale, disp_w_pt, disp_h_pt, page_rotation
        )
        y_spacings_mm = self._detect_spacings(
            y_lines_px, "y", texts, scale, disp_w_pt, disp_h_pt, page_rotation
        )

        px_per_mm_x = self._compute_px_per_mm(x_lines_px, x_spacings_mm)
        px_per_mm_y = self._compute_px_per_mm(y_lines_px, y_spacings_mm)

        return {
            "x_lines_px":      x_lines_px,
            "y_lines_px":      y_lines_px,
            "x_labels":        x_labels,
            "y_labels":        y_labels,
            "x_spacings_mm":   x_spacings_mm,
            "y_spacings_mm":   y_spacings_mm,
            "origin_px":       (x_lines_px[0], y_lines_px[0]),
            "px_per_mm_x":     px_per_mm_x,
            "px_per_mm_y":     px_per_mm_y,
            "pixels_per_mm":   (px_per_mm_x + px_per_mm_y) / 2.0,
            "page_rotation":   page_rotation,
            "has_grid":        True,
            "source":          "text_labels",
            "scale_string":    "grid-derived",
        }

    def pixel_to_world(self, px: float, py: float, grid_info: Dict) -> Tuple[float, float]:
        x_lines = grid_info.get("x_lines_px", [])
        y_lines = grid_info.get("y_lines_px", [])
        x_sp    = grid_info.get("x_spacings_mm", [])
        y_sp    = grid_info.get("y_spacings_mm", [])
        x_mm    = self._interp_world(px, x_lines, x_sp)
        y_mm    = self._interp_world(py, y_lines, y_sp)
        return x_mm, y_mm

    def align_pixels_to_columns(self, grid_info: Dict, column_detections: List[Dict]) -> Dict:
        """Grid lines are fixed. This method intentionally does nothing."""
        return grid_info

    # ------------------------------------------------------------------
    # Coordinate transform (rotation-aware)
    # ------------------------------------------------------------------

    @staticmethod
    def _to_image_coords(
        bbox: List[float],
        disp_w_pt: float,
        disp_h_pt: float,
        scale: float,
        rotation: int,
    ) -> Tuple[float, float]:
        """
        Convert a PDF text bbox centre to image-pixel (image_x, image_y).

        bbox       — [x0, y0, x1, y1] in natural (pre-rotation) PDF points
        disp_w_pt  — page.rect.width  (display width in points)
        disp_h_pt  — page.rect.height (display height in points)
        scale      — dpi / 72  (pixels per point)
        rotation   — page rotation in degrees (0, 90, 180, 270)
        """
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        if rotation == 90:
            return (disp_w_pt - cy) * scale, cx * scale
        if rotation == 270:
            return cy * scale, (disp_h_pt - cx) * scale
        if rotation == 180:
            return (disp_w_pt - cx) * scale, (disp_h_pt - cy) * scale
        # rotation == 0
        return cx * scale, cy * scale

    # ------------------------------------------------------------------
    # Text-based grid line extraction  (v26 — pair at detection time)
    # ------------------------------------------------------------------

    def _extract_lines_from_text(
        self,
        texts: List[Dict],
        img_w: int,
        img_h: int,
        scale: float,
        disp_w_pt: float,
        disp_h_pt: float,
        rotation: int,
    ) -> Tuple[List[float], List[str], List[float], List[str]]:
        """
        v26 algorithm: pair each grid label with its pixel position at the
        moment of detection.

        For every unique label string found in the margin:
            position = median pixel coordinate of all occurrences of that label
                       (left margin + right margin for H-labels;
                        top margin  + bottom margin for V-labels)

        No separate clustering step.  No label-to-position matching pass.
        Ordering for gap computation is by pixel coordinate only — never by
        label string — so sequences like …Z, AA, BB… are always handled
        correctly regardless of alphabetical ordering.

        Returns
        -------
        (v_positions, v_labels, h_positions, h_labels)
        """
        MARGIN_X_TOL = int(round(40 * scale))   # ~83 px at 150 DPI
        LABEL_Y_TOL  = 50                        # px — margin band half-width

        def to_px(t):
            bbox = t.get("bbox")
            if not bbox or len(bbox) < 4:
                return None
            return self._to_image_coords(bbox, disp_w_pt, disp_h_pt, scale, rotation)

        def _median(vals):
            s = sorted(vals)
            return s[len(s) // 2]

        def median_filter_x(group, tol):
            if not group:
                return group
            med_x = _median([x for _, x, _ in group])
            return [(t, x, y) for t, x, y in group if abs(x - med_x) <= tol]

        def median_filter_y(group, tol=LABEL_Y_TOL):
            if len(group) < 2:
                return group
            med_y = _median([y for _, _, y in group])
            return [(t, x, y) for t, x, y in group if abs(y - med_y) <= tol]

        # ── Step 1: H-labels (1-2 char alphabetic) → establish margin X bounds ──
        h_raw = []
        for t in texts:
            raw = t.get("text", "").strip()
            if not raw or not _H_LABEL_RE.match(raw):
                continue
            coords = to_px(t)
            if coords is None:
                continue
            ix, iy = coords
            if 0 < ix < img_w and 0 < iy < img_h:
                h_raw.append((raw, ix, iy))

        mid_x   = img_w / 2.0
        left_h  = median_filter_x([(t, x, y) for t, x, y in h_raw if x < mid_x],  MARGIN_X_TOL)
        right_h = median_filter_x([(t, x, y) for t, x, y in h_raw if x >= mid_x], MARGIN_X_TOL)

        BX0 = int(_median([x for _, x, _ in left_h]))  if left_h  else int(img_w * 0.08)
        BX1 = int(_median([x for _, x, _ in right_h])) if right_h else int(img_w * 0.88)

        all_h_ys = sorted(y for _, _, y in (left_h + right_h))
        BY0 = int(all_h_ys[0])  if all_h_ys else int(img_h * 0.15)
        BY1 = int(all_h_ys[-1]) if all_h_ys else int(img_h * 0.92)

        # ── Step 2: V-label pass to find V_TOP_Y / V_BOT_Y ─────────────────────
        v_all = []
        for t in texts:
            raw = t.get("text", "").strip()
            try:
                val = int(raw)
            except ValueError:
                continue
            if val < 1 or val > 99:
                continue
            coords = to_px(t)
            if coords is None:
                continue
            ix, iy = coords
            if BX0 - 200 < ix < BX1 + 200:
                v_all.append((raw, ix, iy))

        v_top_raw = [(t, x, y) for t, x, y in v_all if y < img_h * 0.5]
        v_bot_raw = [(t, x, y) for t, x, y in v_all if y >= img_h * 0.5]
        v_top = median_filter_y(v_top_raw)
        v_bot = median_filter_y(v_bot_raw)

        V_BOT_Y = int(_median([y for _, _, y in v_bot])) if v_bot else BY1 + 50
        V_TOP_Y = int(_median([y for _, _, y in v_top])) if v_top else BY0 - 50

        logger.info(
            f"H-labels: left={len(left_h)}, right={len(right_h)}  "
            f"BX0={BX0} BX1={BX1}  V_TOP_Y={V_TOP_Y} V_BOT_Y={V_BOT_Y}"
        )
        logger.info(
            f"V-labels: top={len(v_top)}, bottom={len(v_bot)}  "
            f"V_TOP_Y={V_TOP_Y} V_BOT_Y={V_BOT_Y}"
        )
        _h_left_count  = len(left_h)
        _h_right_count = len(right_h)
        _h_confirmed   = max(_h_left_count, _h_right_count)
        _h_delta       = abs(_h_left_count - _h_right_count)
        if _h_delta == 0:
            logger.info(f"H-lines: dual left+right confirmation → {_h_confirmed}")
        else:
            logger.info(
                f"H-lines: dual left+right confirmation → {_h_confirmed}"
                f"  (right={_h_right_count}, delta={_h_delta})"
            )
        if not left_h:
            logger.warning("No left-margin H-labels — BX0 is a fallback estimate")
        if not right_h:
            logger.warning("No right-margin H-labels — BX1 is a fallback estimate")
        if not v_top:
            logger.warning("No top V-labels — V_TOP_Y is a fallback estimate")
        if not v_bot:
            logger.warning("No bottom V-labels — V_BOT_Y is a fallback estimate")

        # ── Step 3: V-lines — pair each label with its median X position ─────────
        # For every unique numeric label, collect image_x from all occurrences
        # that fall in the top or bottom margin band.  Position = median image_x.
        # Sorting is by pixel coordinate, never by label string.
        v_label_xs: Dict[str, List[float]] = {}
        for t in texts:
            raw = t.get("text", "").strip()
            try:
                val = int(raw)
            except ValueError:
                continue
            if val < 1 or val > 99:
                continue
            coords = to_px(t)
            if coords is None:
                continue
            ix, iy = coords
            if ix < BX0 or ix > BX1:
                continue
            # Must be in top or bottom margin band
            in_top = abs(iy - V_TOP_Y) <= LABEL_Y_TOL
            in_bot = abs(iy - V_BOT_Y) <= LABEL_Y_TOL
            if not (in_top or in_bot):
                continue
            v_label_xs.setdefault(raw, []).append(ix)

        v_pairs = sorted(
            [(_median(xs), lbl) for lbl, xs in v_label_xs.items()],
            key=lambda p: p[0],
        )
        v_positions = [p[0] for p in v_pairs]
        v_labels    = [p[1] for p in v_pairs]

        logger.info(
            f"V-lines: {len(v_pairs)} unique labels paired at detection time"
        )
        logger.info(f"V-lines: dual top+bot confirmation → {len(v_positions)}")

        # ── Step 4: H-lines — pair each label with its median Y position ─────────
        # For every unique alpha label, collect image_y from all occurrences
        # that fall near the left or right margin column AND within the grid
        # vertical extent.  Position = median image_y.
        # Labels that appear in only one margin (e.g. "AA", "BB" on one side
        # only) are handled correctly — their single occurrence is used as-is.
        # Sorting is by pixel coordinate, never by label string.
        h_label_ys: Dict[str, List[float]] = {}
        for t in texts:
            raw = t.get("text", "").strip()
            if not raw or not _H_LABEL_RE.match(raw):
                continue
            coords = to_px(t)
            if coords is None:
                continue
            ix, iy = coords
            # Must be within the vertical grid extent
            if iy < V_TOP_Y or iy > V_BOT_Y:
                continue
            # Must be near the left OR right margin column
            near_left  = abs(ix - BX0) <= MARGIN_X_TOL
            near_right = abs(ix - BX1) <= MARGIN_X_TOL
            if not (near_left or near_right):
                continue
            h_label_ys.setdefault(raw, []).append(iy)

        h_pairs = sorted(
            [(_median(ys), lbl) for lbl, ys in h_label_ys.items()],
            key=lambda p: p[0],
        )
        h_positions = [p[0] for p in h_pairs]
        h_labels    = [p[1] for p in h_pairs]

        logger.info(f"Final: {len(v_positions)} V-lines, {len(h_positions)} H-lines")
        logger.info(
            f"H-lines paired (label → image_y px): "
            f"{[(lbl, round(iy)) for lbl, iy in zip(h_labels, h_positions)]}"
        )

        # Warn about duplicate labels (may indicate duplicate text in PDF)
        for label_list, axis_name in [(v_labels, "V"), (h_labels, "H")]:
            seen: set = set()
            for lbl in label_list:
                if lbl in seen:
                    logger.warning(
                        f"Duplicate {axis_name}-line label '{lbl}' detected — "
                        f"check PDF for repeated text"
                    )
                seen.add(lbl)

        return v_positions, v_labels, h_positions, h_labels

    # ------------------------------------------------------------------
    # Sort and validate
    # ------------------------------------------------------------------

    @staticmethod
    def _sort_lines(
        positions: List[float],
        labels: List[str],
        numeric: bool,
    ) -> Tuple[List[float], List[str]]:
        """Sort grid lines by pixel position (ascending).  For numeric labels,
        also try re-sorting by label integer value if that keeps positions sorted."""
        pairs      = sorted(zip(positions, labels), key=lambda p: p[0])
        positions  = [p[0] for p in pairs]
        labels     = [p[1] for p in pairs]

        if numeric and all(l.isdigit() for l in labels):
            by_label   = sorted(zip(labels, positions), key=lambda p: int(p[0]))
            sorted_lbl = [p[0] for p in by_label]
            sorted_px  = [p[1] for p in by_label]
            if sorted_px == sorted(sorted_px):
                positions, labels = sorted_px, sorted_lbl

        return positions, labels

    # ------------------------------------------------------------------
    # Spacing detection  (v26 — modal-based, robust to cumulative annotations)
    # ------------------------------------------------------------------

    def _detect_spacings(
        self,
        line_positions: List[float],
        axis: str,                  # "x" → V-line bays, "y" → H-line bays
        texts: List[Dict],
        scale: float,
        disp_w_pt: float,
        disp_h_pt: float,
        rotation: int,
    ) -> List[float]:
        """
        Find dimension annotation text (3-5 digit integers, e.g. "8400") whose
        image-space centre falls in each bay, and record it as the bay spacing.

        Structural drawings often print both individual bay annotations (e.g. 8400)
        AND cumulative annotations (e.g. 16800, 25200) in the same margin region.
        To avoid picking cumulative values over individual ones, ALL candidate
        dimension values are collected per gap, then the value **closest to the
        modal annotation** across all bays is selected.  The modal value is the
        most commonly occurring annotation, which is the individual bay dimension
        for uniform grids.

        Missing bays are inferred from the annotated px/mm ratio (±50% tolerance),
        then fall back to FALLBACK_BAY_MM.
        """
        n_gaps = max(0, len(line_positions) - 1)
        if n_gaps == 0:
            return []

        # Collect ALL candidate dimension values per gap (may be multiple)
        gap_candidates: List[List[float]] = [[] for _ in range(n_gaps)]

        for t in texts:
            raw = t.get("text", "").strip()
            m   = _DIM_RE.match(raw)
            if not m:
                continue
            dim_mm = int(m.group(1))
            if dim_mm < 300 or dim_mm > 30_000:
                continue
            bbox = t.get("bbox")
            if not bbox or len(bbox) < 4:
                continue

            ix, iy = self._to_image_coords(bbox, disp_w_pt, disp_h_pt, scale, rotation)
            text_pos_px = ix if axis == "x" else iy

            for i in range(n_gaps):
                gap_px  = abs(line_positions[i + 1] - line_positions[i])
                mid_px  = (line_positions[i] + line_positions[i + 1]) / 2.0
                half_px = max(gap_px / 2.0, 1.0)
                if abs(text_pos_px - mid_px) < half_px:
                    gap_candidates[i].append(float(dim_mm))
                    break

        # Determine the modal annotation value and median gap pixel size.
        # For a uniform grid the modal value is the individual bay dimension.
        # For each gap we compute an *expected* mm value proportional to its
        # pixel width relative to the median gap pixel width.  Candidates that
        # deviate from the expected value by more than PLAUS_TOL are considered
        # cumulative or spurious annotations and are discarded; the gap then
        # falls through to the pixel-ratio inference step below.
        PLAUS_TOL = 0.30   # 30% tolerance — rejects 2× cumulative annotations

        all_values = [v for cands in gap_candidates for v in cands]
        spacings: List[Optional[float]] = [None] * n_gaps

        if all_values:
            counts   = Counter(all_values)
            modal_mm = float(counts.most_common(1)[0][0])

            gap_pxs       = [abs(line_positions[i + 1] - line_positions[i])
                             for i in range(n_gaps)]
            median_gap_px = sorted(gap_pxs)[len(gap_pxs) // 2]

            logger.debug(
                f"{axis}-axis: modal annotation = {modal_mm:.0f} mm, "
                f"median gap = {median_gap_px:.1f} px "
                f"(from {len(all_values)} candidates across {n_gaps} gaps)"
            )

            for i, cands in enumerate(gap_candidates):
                if not cands:
                    continue
                # Expected mm for this specific gap (handles non-uniform grids)
                expected_mm = modal_mm * gap_pxs[i] / max(median_gap_px, 1.0)
                plausible   = [v for v in cands
                               if abs(v - expected_mm) / max(expected_mm, 1.0) <= PLAUS_TOL]
                if plausible:
                    spacings[i] = min(plausible, key=lambda v: abs(v - expected_mm))
                else:
                    logger.debug(
                        f"{axis}-axis gap {i}: annotation(s) {cands} rejected as "
                        f"implausible (expected ≈{expected_mm:.0f} mm) — will infer"
                    )

        # Infer missing spacings from annotated px/mm ratio (±50% tolerance)
        annotated = [
            (abs(line_positions[i + 1] - line_positions[i]), s)
            for i, s in enumerate(spacings) if s is not None
        ]
        if annotated:
            avg_ratio = sum(px / mm for px, mm in annotated if mm > 0) / len(annotated)
            for i in range(n_gaps):
                if spacings[i] is not None:
                    continue
                gap_px    = abs(line_positions[i + 1] - line_positions[i])
                best_mm, best_err = None, float("inf")
                for _, mm in annotated:
                    expected = mm * avg_ratio
                    if expected <= 0:
                        continue
                    ratio = gap_px / expected
                    if 0.50 <= ratio <= 1.50:
                        err = abs(ratio - 1.0)
                        if err < best_err:
                            best_err, best_mm = err, mm
                if best_mm is not None:
                    spacings[i] = best_mm

        missing = [i for i, s in enumerate(spacings) if s is None]
        if missing:
            logger.warning(
                f"{len(missing)}/{n_gaps} {axis}-axis gaps missing annotation "
                f"→ using {FALLBACK_BAY_MM} mm"
            )
            for i in missing:
                spacings[i] = float(FALLBACK_BAY_MM)

        return [float(s) for s in spacings]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_px_per_mm(
        self, line_positions: List[float], spacings_mm: List[float]
    ) -> float:
        if len(line_positions) < 2 or not spacings_mm:
            return 150.0 / 25.4
        ratios = [
            abs(line_positions[i + 1] - line_positions[i]) / sp
            for i, sp in enumerate(spacings_mm) if sp > 0
        ]
        return float(np.mean(ratios)) if ratios else (150.0 / 25.4)

    def _interp_world(
        self, pos: float, lines: List[float], spacings: List[float]
    ) -> float:
        if not lines:
            return 0.0
        idx = len(lines) - 1
        for i in range(len(lines) - 1):
            if pos < lines[i + 1]:
                idx = i
                break
        world = sum(spacings[:idx]) if spacings and idx > 0 else 0.0
        if idx < len(lines) - 1 and idx < len(spacings):
            cell_px = lines[idx + 1] - lines[idx]
            if cell_px > 0:
                world += (pos - lines[idx]) / cell_px * spacings[idx]
        return world

    def _estimate_page_bounds(self, paths: List[Dict]):
        xs0, ys0, xs1, ys1 = [], [], [], []
        for p in paths:
            r = p.get("rect")
            if r is None:
                continue
            try:
                if hasattr(r, 'x0'):
                    x0, y0, x1, y1 = float(r.x0), float(r.y0), float(r.x1), float(r.y1)
                else:
                    x0, y0, x1, y1 = float(r[0]), float(r[1]), float(r[2]), float(r[3])
                xs0.append(x0); ys0.append(y0)
                xs1.append(x1); ys1.append(y1)
            except Exception:
                continue
        if not xs0:
            return 0.0, 0.0, 3370.0, 2384.0
        return min(xs0), min(ys0), max(xs1) - min(xs0), max(ys1) - min(ys0)

    @staticmethod
    def _make_alpha_defaults(n: int) -> List[str]:
        """A, B, …, Z, AA, BB, CC, DD, … (double-same-letter after Z)."""
        result = []
        for i in range(n):
            if i < 26:
                result.append(chr(65 + i))
            else:
                c = chr(65 + (i - 26))
                result.append(c + c)
        return result

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------

    def _fallback_grid(self, img_w: int, img_h: int) -> Dict:
        """Uniform fallback grid using fixed defaults (no interactive prompt — web server context)."""
        n_x, n_y = 42, 30

        logger.warning(f"Using fallback grid: {n_x} V × {n_y} H lines")

        x_lines = [round(img_w * i / (n_x - 1)) for i in range(n_x)]
        y_lines = [round(img_h * i / (n_y - 1)) for i in range(n_y)]
        sp_x    = [float(DEFAULT_BAY_MM)] * (n_x - 1)
        sp_y    = [float(DEFAULT_BAY_MM)] * (n_y - 1)

        return {
            "x_lines_px":            x_lines,
            "y_lines_px":            y_lines,
            "x_labels":              [str(i + 1) for i in range(n_x)],
            "y_labels":              self._make_alpha_defaults(n_y),
            "x_spacings_mm":         sp_x,
            "y_spacings_mm":         sp_y,
            "origin_px":             (x_lines[0], y_lines[0]),
            "px_per_mm_x":           img_w / ((n_x - 1) * DEFAULT_BAY_MM),
            "px_per_mm_y":           img_h / ((n_y - 1) * DEFAULT_BAY_MM),
            "pixels_per_mm":         (img_w / ((n_x - 1) * DEFAULT_BAY_MM) +
                                      img_h / ((n_y - 1) * DEFAULT_BAY_MM)) / 2.0,
            "page_rotation":         0,
            "has_grid":              False,
            "source":                "fallback",
            "scale_string":          "fallback-grid",
            "grid_confidence":       0.0,
            "grid_confidence_label": "Fallback",
        }

