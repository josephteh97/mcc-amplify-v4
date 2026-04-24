"""Unit tests for GeometryGenerator — structural framing builder."""
import math
import pytest

from backend.services.geometry_generator import GeometryGenerator


# Minimal grid_info for pixel→mm conversion — matches a simple 2-bay grid
_GRID = {
    "x_lines_px":    [100.0, 500.0, 900.0],
    "y_lines_px":    [100.0, 500.0, 900.0],
    "x_spacings_mm": [4000.0, 4000.0],
    "y_spacings_mm": [4000.0, 4000.0],
    "px_per_mm_x":   0.1,   # 10 px per mm  (400 px / 4000 mm)
    "px_per_mm_y":   0.1,
    "origin_x_px":   100.0,
    "origin_y_px":   100.0,
}

_LEVEL0_ELEV = 0   # Level 0 (ground) elevation (mm); beam is referenced here


def _make_beam(idx, x1, y1, x2, y2, conf=0.85):
    return {
        "id":         idx,
        "confidence": conf,
        "center":     [(x1 + x2) / 2, (y1 + y2) / 2],
        "bbox":       [float(x1), float(y1), float(x2), float(y2)],
    }


@pytest.fixture
def gen():
    return GeometryGenerator()


class TestBuildStructuralFramingParameters:

    def test_n_inputs_produce_n_outputs(self, gen):
        beams = [
            _make_beam(0, 100, 290, 900, 310),   # horizontal
            _make_beam(1, 290, 100, 310, 900),   # vertical
        ]
        result = gen._build_structural_framing_parameters(beams, _GRID, _LEVEL0_ELEV)
        assert len(result) == 2

    def test_start_ne_end(self, gen):
        beams = [_make_beam(0, 100, 290, 900, 310)]
        result = gen._build_structural_framing_parameters(beams, _GRID, _LEVEL0_ELEV)
        assert len(result) == 1
        s, e = result[0]["start_point"], result[0]["end_point"]
        span = math.sqrt((e["x"] - s["x"]) ** 2 + (e["y"] - s["y"]) ** 2)
        assert span > 1.0

    def test_horizontal_beam_axis(self, gen):
        # Wide horizontal bbox → should span in X
        beams = [_make_beam(0, 100, 290, 900, 310)]  # 800 px wide, 20 px tall
        result = gen._build_structural_framing_parameters(beams, _GRID, _LEVEL0_ELEV)
        s, e = result[0]["start_point"], result[0]["end_point"]
        dx = abs(e["x"] - s["x"])
        dy = abs(e["y"] - s["y"])
        assert dx > dy, "Horizontal beam should span in X"

    def test_vertical_beam_axis(self, gen):
        # Tall vertical bbox → should span in Y
        beams = [_make_beam(0, 290, 100, 310, 900)]  # 20 px wide, 800 px tall
        result = gen._build_structural_framing_parameters(beams, _GRID, _LEVEL0_ELEV)
        s, e = result[0]["start_point"], result[0]["end_point"]
        dy = abs(e["y"] - s["y"])
        dx = abs(e["x"] - s["x"])
        assert dy > dx, "Vertical beam should span in Y"

    def test_z_at_beam_top(self, gen):
        """Insertion Z is the beam TOP elevation = Level0 + slab_thickness,
        flush with the ground-slab top surface."""
        beams = [_make_beam(0, 100, 290, 900, 310)]
        result = gen._build_structural_framing_parameters(beams, _GRID, _LEVEL0_ELEV)
        expected_z = _LEVEL0_ELEV + gen.default_floor_thickness
        assert result[0]["start_point"]["z"] == pytest.approx(expected_z)
        assert result[0]["end_point"]["z"]   == pytest.approx(expected_z)
        assert result[0]["level"] == "Level 0"

    def test_degenerate_bbox_skipped(self, gen):
        # Bbox with zero span → world coords are equal → skipped
        beams = [_make_beam(0, 300, 300, 300, 300)]
        result = gen._build_structural_framing_parameters(beams, _GRID, _LEVEL0_ELEV)
        assert len(result) == 0

    def test_output_has_required_keys(self, gen):
        beams = [_make_beam(0, 100, 290, 900, 310)]
        result = gen._build_structural_framing_parameters(beams, _GRID, _LEVEL0_ELEV)
        entry = result[0]
        for key in ("id", "start_point", "end_point", "width", "depth", "level", "family_type"):
            assert key in entry, f"Missing key: {key}"

    def test_empty_input_returns_empty(self, gen):
        assert gen._build_structural_framing_parameters([], _GRID, _LEVEL0_ELEV) == []
