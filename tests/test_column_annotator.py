"""
Tests for the column annotation parser (extracted from orchestrator).
Runs without YOLO weights or AI backends — uses synthetic data only.
"""
import numpy as np
import pytest
from backend.services.column_annotator import annotate_columns, _RE_RECT, _RE_CIRC, _RE_MARK, _is_beam_label


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_vector_data(text_items=None, page_rect=None):
    """Build a minimal vector_data dict."""
    return {
        "paths": [],
        "text": text_items or [],
        "page_rect": page_rect or [0, 0, 600, 800],
        "page_rotation": 0,
    }


def _make_image_data(w=600, h=800):
    return {"image": np.zeros((h, w, 3), dtype=np.uint8), "width": w, "height": h, "dpi": 150}


def _make_detections(columns):
    """Build a detections dict with only columns."""
    return {"walls": [], "doors": [], "windows": [], "columns": columns, "rooms": []}


def _col(cx, cy, size=30):
    return {
        "id": 0,
        "confidence": 0.9,
        "center": [cx, cy],
        "bbox": [cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2],
        "dimensions": {"width_px": size, "height_px": size},
    }


# ── Regex unit tests ──────────────────────────────────────────────────────────

class TestRegexPatterns:
    def test_rect_pattern_x(self):
        m = _RE_RECT.search("C1 800x800")
        assert m and m.group(1) == "800" and m.group(2) == "800"

    def test_rect_pattern_times(self):
        m = _RE_RECT.search("C2 400×600")
        assert m and m.group(1) == "400" and m.group(2) == "600"

    def test_circ_pattern_prefix(self):
        m = _RE_CIRC.search("Ø300")
        assert m and m.group(1) == "300"

    def test_circ_pattern_suffix(self):
        m = _RE_CIRC.search("500Ø")
        assert m and m.group(2) == "500"

    def test_circ_dia_prefix(self):
        m = _RE_CIRC.search("dia 200")
        assert m and m.group(1) == "200"

    def test_mark_pattern(self):
        m = _RE_MARK.search("C1 800x800")
        assert m and m.group(1) == "C1"

    def test_mark_pattern_multi_char(self):
        m = _RE_MARK.search("RCB1")
        assert m and m.group(1) == "RCB1"


class TestBeamLabelFilter:
    def test_rcb_is_beam(self):
        assert _is_beam_label("RCB2 800×300")

    def test_gb_is_beam(self):
        assert _is_beam_label("GB1")

    def test_column_mark_not_beam(self):
        assert not _is_beam_label("C1 800×800")

    def test_k_mark_not_beam(self):
        assert not _is_beam_label("K14 500×500")


# ── Pass 1: Schedule scan ─────────────────────────────────────────────────────

class TestScheduleScan:
    def test_schedule_from_page_text(self):
        """Pass 1 should build a schedule from text items containing mark+dims."""
        text_items = [
            {"text": "C1 800x800", "bbox": [10, 10, 100, 20], "size": 8, "font": "Arial"},
            {"text": "C2 Ø300", "bbox": [10, 30, 100, 40], "size": 8, "font": "Arial"},
        ]
        dets = _make_detections([_col(50, 50)])
        vector = _make_vector_data(text_items)
        image = _make_image_data()

        result = annotate_columns(dets, vector, image)
        col = result["columns"][0]
        # Column near text "C1 800x800" should get 800x800
        assert col["width_mm"] == 800.0
        assert col["depth_mm"] == 800.0
        assert col.get("is_circular") is False

    def test_schedule_from_extra_pages(self):
        """Pass 1 supplement: schedule entries from extra pages should be used."""
        extra = ["C5 600x600"]
        # Column text nearby has only the mark "C5", no dimensions
        text_items = [
            {"text": "C5", "bbox": [45, 45, 55, 55], "size": 8, "font": "Arial"},
        ]
        dets = _make_detections([_col(50, 50)])
        vector = _make_vector_data(text_items)
        image = _make_image_data()

        result = annotate_columns(dets, vector, image, extra_schedule_texts=extra)
        col = result["columns"][0]
        assert col["width_mm"] == 600.0
        assert col["type_mark"] == "C5"

    def test_beam_mark_excluded_from_schedule(self):
        """Beam marks (RCB, GB, etc.) should not enter the schedule."""
        text_items = [
            {"text": "RCB2 800×300", "bbox": [10, 10, 100, 20], "size": 8, "font": "Arial"},
        ]
        dets = _make_detections([_col(50, 50)])
        vector = _make_vector_data(text_items)
        image = _make_image_data()

        result = annotate_columns(dets, vector, image)
        col = result["columns"][0]
        # Should default to 800mm since "RCB2" is a beam label
        assert col["width_mm"] == 800.0
        assert "type_mark" not in col


# ── Pass 2: Proximity annotation ──────────────────────────────────────────────

class TestProximityAnnotation:
    def test_inline_label_rectangular(self):
        """Column near text 'C1 400×600' should get 400×600."""
        text_items = [
            {"text": "C1 400×600", "bbox": [48, 48, 90, 55], "size": 8, "font": "Arial"},
        ]
        dets = _make_detections([_col(50, 50)])
        vector = _make_vector_data(text_items)
        image = _make_image_data()

        result = annotate_columns(dets, vector, image)
        col = result["columns"][0]
        assert col["width_mm"] == 400.0
        assert col["depth_mm"] == 600.0
        assert col["type_mark"] == "C1"

    def test_inline_label_circular(self):
        """Column near text 'C20 Ø200' should get diameter 200."""
        text_items = [
            {"text": "C20 Ø200", "bbox": [48, 48, 90, 55], "size": 8, "font": "Arial"},
        ]
        dets = _make_detections([_col(50, 50)])
        vector = _make_vector_data(text_items)
        image = _make_image_data()

        result = annotate_columns(dets, vector, image)
        col = result["columns"][0]
        assert col["is_circular"] is True
        assert col["diameter_mm"] == 200.0
        assert col["type_mark"] == "C20"

    def test_far_text_not_matched(self):
        """Text far from column should not be matched."""
        text_items = [
            {"text": "C1 400×600", "bbox": [550, 750, 595, 760], "size": 8, "font": "Arial"},
        ]
        dets = _make_detections([_col(50, 50)])
        vector = _make_vector_data(text_items)
        image = _make_image_data()

        result = annotate_columns(dets, vector, image)
        col = result["columns"][0]
        # Should fall back to default 800mm
        assert col["width_mm"] == 800.0


# ── Pass 4: Single-scheme fallback ────────────────────────────────────────────

class TestSingleSchemeFallback:
    def test_single_scheme_applied(self):
        """When exactly one schedule type exists, all unresolved columns get it."""
        # Schedule page has C1 500x500
        extra = ["C1 500x500"]
        # Two columns, no nearby text
        dets = _make_detections([_col(50, 50), _col(200, 200)])
        vector = _make_vector_data([])
        image = _make_image_data()

        result = annotate_columns(dets, vector, image, extra_schedule_texts=extra)
        assert result["columns"][0]["width_mm"] == 500.0
        assert result["columns"][1]["width_mm"] == 500.0

    def test_multiple_schemes_no_fallback(self):
        """When multiple schedule types exist, unresolved columns get the safe default."""
        extra = ["C1 500x500", "C2 300x300"]
        dets = _make_detections([_col(50, 50)])
        vector = _make_vector_data([])
        image = _make_image_data()

        result = annotate_columns(dets, vector, image, extra_schedule_texts=extra)
        col = result["columns"][0]
        # Should fall back to 800mm default (not either schedule type)
        assert col["width_mm"] == 800.0


# ── Pass 5: Safe default ──────────────────────────────────────────────────────

class TestSafeDefault:
    def test_no_text_defaults_to_800mm(self):
        """Column with no nearby text should get 800mm safe default."""
        dets = _make_detections([_col(50, 50)])
        vector = _make_vector_data([])
        image = _make_image_data()

        result = annotate_columns(dets, vector, image)
        col = result["columns"][0]
        assert col["width_mm"] == 800.0
        assert col["depth_mm"] == 800.0
        assert col.get("is_circular") is False


# ── Coordinate immutability ───────────────────────────────────────────────────

class TestCoordinateImmutability:
    def test_center_preserved(self):
        """annotate_columns must not alter the column center coordinates."""
        original_center = [123.456, 789.012]
        text_items = [
            {"text": "C1 400×600", "bbox": [120, 785, 140, 795], "size": 8, "font": "Arial"},
        ]
        dets = _make_detections([_col(123.456, 789.012)])
        vector = _make_vector_data(text_items)
        image = _make_image_data()

        result = annotate_columns(dets, vector, image)
        assert result["columns"][0]["center"] == original_center

    def test_bbox_preserved(self):
        """annotate_columns must not alter the column bbox coordinates."""
        col = _col(100, 200, 40)
        original_bbox = list(col["bbox"])
        dets = _make_detections([col])
        vector = _make_vector_data([])
        image = _make_image_data()

        result = annotate_columns(dets, vector, image)
        assert result["columns"][0]["bbox"] == original_bbox
