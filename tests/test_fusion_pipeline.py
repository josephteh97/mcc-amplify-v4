"""
Tests for the HybridFusionPipeline.
Validates coordinate transforms and wall-snapping logic.
"""
import pytest
from backend.services.fusion.pipeline import HybridFusionPipeline, SpatialAlignmentEngine


# ── SpatialAlignmentEngine tests ──────────────────────────────────────────────

class TestSpatialAlignmentEngine:
    def test_default_dpi_is_72(self):
        engine = SpatialAlignmentEngine()
        assert engine.dpi == 72
        assert engine.scale_factor == 1.0

    def test_set_dpi_updates_scale(self):
        engine = SpatialAlignmentEngine()
        engine.set_dpi(150)
        assert engine.dpi == 150
        assert engine.scale_factor == pytest.approx(150 / 72.0)

    def test_px_to_pt_identity_at_72dpi(self):
        engine = SpatialAlignmentEngine()
        engine.set_dpi(72)
        assert engine.px_to_pt([100.0, 200.0]) == [100.0, 200.0]

    def test_px_to_pt_at_150dpi(self):
        engine = SpatialAlignmentEngine()
        engine.set_dpi(150)
        result = engine.px_to_pt([300.0])
        # 300 px / (150/72) = 300 / 2.0833 ≈ 144.0
        assert result[0] == pytest.approx(300.0 / (150 / 72.0), rel=1e-3)

    def test_roundtrip_px_pt_px(self):
        """Converting px→pt→px should return the original coordinates."""
        engine = SpatialAlignmentEngine()
        engine.set_dpi(200)
        original = [500.0, 700.0, 600.0, 800.0]
        pts = engine.bbox_px_to_pt(original)
        back = engine.bbox_pt_to_px(pts)
        for a, b in zip(original, back):
            assert a == pytest.approx(b, abs=0.001)


# ── HybridFusionPipeline Level 1: Normalization ──────────────────────────────

class TestFusionLevel1:
    @pytest.mark.asyncio
    async def test_normalize_preserves_detection_count(self):
        pipe = HybridFusionPipeline()
        detections = [
            {"type": "column", "bbox": [100, 200, 130, 230], "confidence": 0.9},
            {"type": "wall", "bbox": [50, 50, 500, 55], "confidence": 0.8},
        ]
        result = await pipe.fuse(
            vector_data={"paths": [], "text": []},
            ml_detections=detections,
            metadata={"dpi": 150, "width": 1000, "height": 800},
        )
        assert len(result["refined_px"]) == 2

    @pytest.mark.asyncio
    async def test_roundtrip_preserves_bbox(self):
        """Pixel → point → pixel should be near-identity."""
        pipe = HybridFusionPipeline()
        det = {"type": "column", "bbox": [100.0, 200.0, 130.0, 230.0], "confidence": 0.9}
        result = await pipe.fuse(
            vector_data={"paths": [], "text": []},
            ml_detections=[det],
            metadata={"dpi": 150},
        )
        out_bbox = result["refined_px"][0]["bbox"]
        for a, b in zip(det["bbox"], out_bbox):
            assert a == pytest.approx(b, abs=0.5)


# ── HybridFusionPipeline Level 2: Wall snapping ──────────────────────────────

def _fake_path_hline(y, x1, x2):
    """Build a PyMuPDF-style path dict for a horizontal line at y from x1 to x2."""
    class Pt:
        def __init__(self, x, y):
            self.x = x
            self.y = y
    return {
        "type": "", "color": None, "width": 1, "rect": None,
        "items": [("m", Pt(x1, y)), ("l", Pt(x2, y))],
    }


def _fake_path_vline(x, y1, y2):
    """Build a PyMuPDF-style path dict for a vertical line at x from y1 to y2."""
    class Pt:
        def __init__(self, x, y):
            self.x = x
            self.y = y
    return {
        "type": "", "color": None, "width": 1, "rect": None,
        "items": [("m", Pt(x, y1)), ("l", Pt(x, y2))],
    }


class TestFusionLevel2WallSnap:
    @pytest.mark.asyncio
    async def test_horizontal_wall_snaps_to_hline(self):
        """A horizontal wall bbox should snap Y to a nearby horizontal vector line."""
        pipe = HybridFusionPipeline()
        # Wall bbox: wide (horizontal), centered at y=102
        wall = {"type": "wall", "bbox": [50.0, 98.0, 400.0, 106.0], "confidence": 0.8}
        # Vector hline at y=100
        paths = [_fake_path_hline(100, 40, 420)]
        result = await pipe.fuse(
            vector_data={"paths": paths, "text": []},
            ml_detections=[wall],
            metadata={"dpi": 72},
        )
        snapped = result["refined_px"][0]
        # The wall should be snapped — geometry_source should be vector_snapped
        assert snapped.get("geometry_source") == "vector_snapped"
        # The Y center should now be at 100 (the vector line)
        snapped_cy = (snapped["bbox"][1] + snapped["bbox"][3]) / 2
        assert snapped_cy == pytest.approx(100.0, abs=1.0)

    @pytest.mark.asyncio
    async def test_vertical_wall_snaps_to_vline(self):
        """A vertical wall bbox should snap X to a nearby vertical vector line."""
        pipe = HybridFusionPipeline()
        wall = {"type": "wall", "bbox": [198.0, 50.0, 206.0, 400.0], "confidence": 0.8}
        paths = [_fake_path_vline(200, 40, 420)]
        result = await pipe.fuse(
            vector_data={"paths": paths, "text": []},
            ml_detections=[wall],
            metadata={"dpi": 72},
        )
        snapped = result["refined_px"][0]
        assert snapped.get("geometry_source") == "vector_snapped"
        snapped_cx = (snapped["bbox"][0] + snapped["bbox"][2]) / 2
        assert snapped_cx == pytest.approx(200.0, abs=1.0)

    @pytest.mark.asyncio
    async def test_column_not_snapped(self):
        """Columns should pass through Level 2 unchanged (no snapping)."""
        pipe = HybridFusionPipeline()
        col = {"type": "column", "bbox": [100.0, 200.0, 130.0, 230.0], "confidence": 0.9}
        paths = [_fake_path_hline(215, 80, 150)]
        result = await pipe.fuse(
            vector_data={"paths": paths, "text": []},
            ml_detections=[col],
            metadata={"dpi": 72},
        )
        out = result["refined_px"][0]
        assert out.get("geometry_source") != "vector_snapped"

    @pytest.mark.asyncio
    async def test_no_snap_when_no_paths(self):
        """Without vector paths, walls should keep ml_approximate source."""
        pipe = HybridFusionPipeline()
        wall = {"type": "wall", "bbox": [50.0, 98.0, 400.0, 106.0], "confidence": 0.8}
        result = await pipe.fuse(
            vector_data={"paths": [], "text": []},
            ml_detections=[wall],
            metadata={"dpi": 72},
        )
        out = result["refined_px"][0]
        assert out.get("geometry_source") == "ml_approximate"

    @pytest.mark.asyncio
    async def test_empty_detections(self):
        """Empty detection list should return empty refined_px."""
        pipe = HybridFusionPipeline()
        result = await pipe.fuse(
            vector_data={"paths": [], "text": []},
            ml_detections=[],
            metadata={"dpi": 150},
        )
        assert result["refined_px"] == []
