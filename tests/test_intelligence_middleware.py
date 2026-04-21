"""
Tests for the intelligence middleware layer.
Runs without YOLO weights — uses synthetic detection dicts.
"""
import numpy as np
import pytest
from backend.services.intelligence.type_resolver import resolve_types
from backend.services.intelligence.cross_element_validator import validate_elements
from backend.services.intelligence.validation_agent import enforce_rules
from backend.services.intelligence.bim_translator_enricher import enrich_recipe
from backend.services.intelligence.recipe_sanitizer import sanitize_recipe


def _make_detection(cx=100.0, cy=100.0, w=30, h=30):
    return {
        "element_type": "column",
        "bbox": [cx - w/2, cy - h/2, cx + w/2, cy + h/2],
        "center": [cx, cy],
        "confidence": 0.85,
    }


def test_type_resolver_adds_fields():
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    dets = [_make_detection(100, 100)]
    result = resolve_types(dets, img)
    assert "resolved_type" in result[0]
    assert "type_confidence" in result[0]


def test_cross_validator_empty_input():
    result = validate_elements([], grid_info=None)
    assert result == []


def test_cross_validator_flags_overlap():
    d1 = _make_detection(100, 100, 40, 40)
    d2 = _make_detection(105, 105, 40, 40)  # overlapping
    result = validate_elements([d1, d2], grid_info=None)
    assert any("iou_overlap" in d["validation_flags"] for d in result)


def test_validation_agent_adds_dfma_fields():
    d = _make_detection()
    d["validation_flags"] = []
    result = enforce_rules([d], grid_info=None)
    assert "is_dfma_compliant" in result[0]
    assert "is_orphan" in result[0]


def test_bim_translator_enricher_safe_skip_on_mismatch():
    recipe = {"columns": [{"x": 1000, "y": 2000, "z": 0}]}
    dets = [_make_detection(), _make_detection()]  # mismatch: 2 dets, 1 recipe column
    result = enrich_recipe(recipe, dets)
    # Should NOT have enriched (mismatch guard)
    assert "resolved_type" not in result["columns"][0]


def test_bim_translator_enricher_matches_correctly():
    recipe = {"columns": [{"x": 1000, "y": 2000, "z": 0}]}
    det = _make_detection()
    det.update({
        "resolved_type": "circular", "type_confidence": 0.9,
        "is_valid": True, "is_dfma_compliant": True,
        "is_orphan": False, "validation_flags": [], "dfma_violations": [],
    })
    result = enrich_recipe(recipe, [det])
    assert result["columns"][0]["resolved_type"] == "circular"
    assert result["columns"][0]["x"] == 1000   # coordinate untouched


def _col(x, y, w=800, d=800):
    return {"location": {"x": x, "y": y, "z": 0}, "width": w, "depth": d}


def _beam(sx, sy, ex, ey, w=400):
    return {
        "start_point": {"x": sx, "y": sy, "z": 3000},
        "end_point":   {"x": ex, "y": ey, "z": 3000},
        "width":       w,
        "depth":       800,
    }


class TestRecipeSanitizerFramingRules:
    """Each rule mirrors a failure mode seen in the common-sense geometry review."""

    def test_both_endpoints_snap_to_columns_is_kept(self):
        recipe = {
            "columns": [_col(0, 0), _col(5000, 0)],
            "structural_framing": [_beam(0, 0, 5000, 0)],
        }
        result, actions = sanitize_recipe(recipe)
        assert len(result["structural_framing"]) == 1

    def test_floating_beam_with_no_nearby_column_is_removed(self):
        # image 4: beam hangs in mid-air, neither endpoint near a column
        recipe = {
            "columns": [_col(0, 0), _col(50000, 0)],  # columns far from beam
            "structural_framing": [_beam(10000, 0, 15000, 0)],
        }
        result, actions = sanitize_recipe(recipe)
        assert result["structural_framing"] == []
        assert any("not within" in a and "would float" in a for a in actions)

    def test_beam_with_one_endpoint_gap_is_removed(self):
        # image 3: one endpoint snaps to column, the other has a visible gap.
        # Gap must exceed half_width (400) + SNAP_BUFFER_MM (150) = 550mm to
        # escape snapping — use 700mm short of the far column.
        recipe = {
            "columns": [_col(0, 0), _col(5000, 0)],
            "structural_framing": [_beam(0, 0, 4300, 0)],
        }
        result, actions = sanitize_recipe(recipe)
        assert result["structural_framing"] == []
        assert any("end_point" in a and "not within" in a for a in actions)

    def test_diagonal_beam_between_non_colinear_columns_is_removed(self):
        # image 2: snap pulls endpoints to columns that aren't perfectly aligned
        recipe = {
            "columns": [_col(0, 0), _col(5000, 200)],  # 200mm y-offset
            "structural_framing": [_beam(0, 0, 5000, 200)],
        }
        result, actions = sanitize_recipe(recipe)
        assert result["structural_framing"] == []
        assert any("diagonal beam" in a for a in actions)

    def test_slight_off_axis_within_tolerance_is_kept(self):
        # near-colinear columns (within _AXIS_TOLERANCE_MM=50mm) still produce valid beam
        recipe = {
            "columns": [_col(0, 0), _col(5000, 20)],  # 20mm y-offset — within tolerance
            "structural_framing": [_beam(0, 0, 5000, 20)],
        }
        result, actions = sanitize_recipe(recipe)
        assert len(result["structural_framing"]) == 1

    def test_both_endpoints_snap_to_same_column_is_removed(self):
        recipe = {
            "columns": [_col(0, 0)],
            "structural_framing": [_beam(100, 0, 200, 0)],  # both snap to column @ (0,0)
        }
        result, actions = sanitize_recipe(recipe)
        assert result["structural_framing"] == []


def test_coordinate_immutability():
    """Column placement pipeline coordinates must not be altered by any middleware."""
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    det = _make_detection(123.456, 789.012)
    original_center = list(det["center"])

    det = resolve_types([det], img)[0]
    assert det["center"] == original_center, "TypeResolver must not alter center"

    det["validation_flags"] = []
    det = validate_elements([det], grid_info=None)[0]
    assert det["center"] == original_center, "CrossElementValidator must not alter center"

    det = enforce_rules([det], grid_info=None)[0]
    assert det["center"] == original_center, "ValidationAgent must not alter center"
