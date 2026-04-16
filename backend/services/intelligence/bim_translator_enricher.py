"""
BIM Translator Enricher — post-geometry, pre-RVT export.

Takes the Revit recipe dict (built by GeometryGenerator) and the enriched
detection list (from Validation Agent), and merges type + validation metadata
into the recipe's column entries by matching on element index.

Appended fields per column in the Revit recipe:
  resolved_type:       str    ("circular" | "rectangular" | "L-shape" | "unknown")
  type_confidence:     float
  is_valid:            bool
  is_dfma_compliant:   bool
  is_orphan:           bool
  validation_flags:    list[str]
  dfma_violations:     list[str]

Never modifies: x, y, z, level, family, type_name, or any geometry field.
The Revit C# Add-in ignores unknown fields in the JSON — safe to add freely.
"""
from __future__ import annotations

from loguru import logger

_METADATA_KEYS = (
    "resolved_type", "type_confidence", "is_valid", "is_dfma_compliant",
    "is_orphan", "validation_flags", "dfma_violations",
)


def enrich_recipe(recipe: dict, detections: list[dict]) -> dict:
    """
    Merge detection metadata into the Revit recipe's column entries.

    recipe:     the dict returned by GeometryGenerator (keys: walls, columns, etc.)
    detections: the enriched list[dict] from ValidationAgent.enforce_rules()

    Returns the mutated recipe dict (same object, safe to chain).
    """
    columns_in_recipe: list[dict] = recipe.get("columns", [])

    if not columns_in_recipe:
        logger.debug("BIMTranslatorEnricher: no columns in recipe to enrich")
        return recipe

    if len(columns_in_recipe) != len(detections):
        logger.warning(
            "BIMTranslatorEnricher: recipe has %d columns but %d detections — "
            "enrichment skipped to avoid index mismatch",
            len(columns_in_recipe), len(detections),
        )
        return recipe

    for recipe_col, det in zip(columns_in_recipe, detections):
        for key in _METADATA_KEYS:
            recipe_col[key] = det.get(key)

    logger.info(
        "BIMTranslatorEnricher: enriched %d column entries in Revit recipe",
        len(columns_in_recipe),
    )
    return recipe
