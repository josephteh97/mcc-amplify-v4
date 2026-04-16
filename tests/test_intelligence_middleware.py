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
