"""Unit tests for backend.services.intelligence.slab_thickness_parser
and the GeometryGenerator._resolve_slab_thickness integration."""
from __future__ import annotations

import fitz

from backend.services.geometry_generator import GeometryGenerator
from backend.services.intelligence.slab_thickness_parser import (
    extract_notes_legend,
    locate_zone_labels,
    resolve_code_thickness,
)


_DOC_CACHE: list = []   # keep fitz.Document alive so the Page isn't orphaned


def _build_page(texts):
    """texts: list[tuple[str, float, float]] — (string, x_pt, y_pt)."""
    doc = fitz.open()
    _DOC_CACHE.append(doc)
    page = doc.new_page(width=842, height=595)
    for s, x, y in texts:
        page.insert_text((x, y), s, fontsize=8)
    return page


class TestExtractNotesLegend:
    def test_nsp_codes_same_line(self):
        page = _build_page([
            ("NOTES:", 600, 50),
            ("NSP2 = 250 mm", 600, 70),
            ("NSP3 = 200 mm", 600, 85),
        ])
        assert extract_notes_legend(page) == {"NSP2": 250.0, "NSP3": 200.0}

    def test_self_describing_codes(self):
        page = _build_page([
            ("NOTES:", 600, 50),
            ("300CIS slabs are cast-in-situ", 600, 70),
        ])
        assert extract_notes_legend(page) == {"300CIS": 300.0}

    def test_no_notes_returns_empty(self):
        page = _build_page([("SITE PLAN", 100, 100)])
        assert extract_notes_legend(page) == {}

    def test_rebar_token_not_confused_for_thickness(self):
        page = _build_page([
            ("NOTES:", 600, 50),
            ("Reinforcement T16 unless noted", 600, 70),
            ("NSP2 = 250 mm", 600, 85),
        ])
        assert extract_notes_legend(page) == {"NSP2": 250.0}

    def test_two_notes_blocks_later_wins(self):
        page = _build_page([
            ("GENERAL NOTES", 600, 50),
            ("NSP2 = 250 mm", 600, 70),
            ("STRUCTURAL NOTES", 600, 300),
            ("NSP2 = 230 mm", 600, 320),
        ])
        assert extract_notes_legend(page)["NSP2"] == 230.0


class TestLocateZoneLabels:
    def test_excludes_labels_inside_notes_block(self):
        page = _build_page([
            ("NOTES:", 600, 50),
            ("NSP2 = 250 mm", 600, 70),   # inside NOTES
            ("NSP2", 150, 300),              # on plan
            ("300CIS", 400, 300),            # on plan
        ])
        labels = locate_zone_labels(page)
        codes = sorted(c for c, _, _ in labels)
        assert codes == ["300CIS", "NSP2"]
        for _, x, _ in labels:
            assert x < 500

    def test_empty_page(self):
        doc = fitz.open()
        _DOC_CACHE.append(doc)
        assert locate_zone_labels(doc.new_page()) == []


class TestResolveCodeThickness:
    def test_self_describing(self):
        assert resolve_code_thickness("300CIS", None) == 300.0
        assert resolve_code_thickness("250CIS", None) == 250.0

    def test_lookup_hit(self):
        assert resolve_code_thickness("NSP2", {"NSP2": 250.0}) == 250.0

    def test_lookup_miss_returns_none(self):
        assert resolve_code_thickness("NSP2", {}) is None

    def test_unknown_code_returns_none(self):
        assert resolve_code_thickness("RANDOM", {}) is None


class TestResolveSlabThicknessIntegration:
    """End-to-end PIP + fallback on GeometryGenerator._resolve_slab_thickness."""

    _square = [
        {"x": 0.0,    "y": 0.0},
        {"x": 1000.0, "y": 0.0},
        {"x": 1000.0, "y": 1000.0},
        {"x": 0.0,    "y": 1000.0},
    ]

    def test_label_inside_region_self_describing(self):
        g = GeometryGenerator()
        t = g._resolve_slab_thickness(
            self._square,
            zone_labels_mm=[("300CIS", 500.0, 500.0)],
            legend={},
        )
        assert t == 300.0

    def test_label_inside_region_nsp_lookup(self):
        g = GeometryGenerator()
        t = g._resolve_slab_thickness(
            self._square,
            zone_labels_mm=[("NSP2", 500.0, 500.0)],
            legend={"NSP2": 250.0},
        )
        assert t == 250.0

    def test_label_outside_region_falls_back(self):
        g = GeometryGenerator()
        t = g._resolve_slab_thickness(
            self._square,
            zone_labels_mm=[("NSP2", 5000.0, 5000.0)],
            legend={"NSP2": 250.0},
        )
        assert t == float(g.default_floor_thickness)

    def test_no_labels_falls_back(self):
        g = GeometryGenerator()
        assert g._resolve_slab_thickness(self._square, [], {}) == float(g.default_floor_thickness)

    def test_nsp_label_without_legend_falls_back(self):
        g = GeometryGenerator()
        t = g._resolve_slab_thickness(
            self._square,
            zone_labels_mm=[("NSP99", 500.0, 500.0)],
            legend={"NSP2": 250.0},
        )
        assert t == float(g.default_floor_thickness)


class TestBeamSlabThickness:
    def test_beam_midpoint_inside_slab_uses_its_thickness(self):
        g = GeometryGenerator()
        slab_regions = [{
            "boundary_points": [
                {"x": 0.0, "y": 0.0}, {"x": 1000.0, "y": 0.0},
                {"x": 1000.0, "y": 1000.0}, {"x": 0.0, "y": 1000.0},
            ],
            "thickness": 300.0,
        }]
        assert g._beam_slab_thickness(500.0, 500.0, slab_regions) == 300.0

    def test_beam_midpoint_outside_all_slabs_falls_back(self):
        g = GeometryGenerator()
        slab_regions = [{
            "boundary_points": [
                {"x": 0.0, "y": 0.0}, {"x": 100.0, "y": 0.0},
                {"x": 100.0, "y": 100.0}, {"x": 0.0, "y": 100.0},
            ],
            "thickness": 300.0,
        }]
        assert g._beam_slab_thickness(500.0, 500.0, slab_regions) == float(g.default_floor_thickness)

    def test_no_slabs_falls_back(self):
        g = GeometryGenerator()
        assert g._beam_slab_thickness(500.0, 500.0, None) == float(g.default_floor_thickness)
