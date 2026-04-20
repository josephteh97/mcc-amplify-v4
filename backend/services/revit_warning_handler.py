"""
Revit Warning Handler — deterministic pattern-based correction before the AI fallback.

The orchestrator calls handle_warnings() on every non-final correction attempt.
It matches warning strings against known Revit error patterns and applies surgical
recipe edits without an LLM call.  Any warnings it cannot handle are returned as
*unresolved* so SemanticAnalyzer can take a second pass.

Known patterns handled
──────────────────────
  "cannot keep elements joined"        → remove the framing element whose endpoint
                                         overlaps a column bounding box the most.
  "identical instances in the same"    → already deduped pre-export; log only.
  "ExternalEvent" / "Pending"          → Revit modal-dialog state; no recipe fix.
  "one or more unresolved references"  → family not loaded; remove that element type.
  "the column is too short"            → raise column height to MIN_COLUMN_HEIGHT_MM.
  "highlights elements whose"          → Revit join-highlight alias → same as join error.
"""
from __future__ import annotations

import os
import re
from typing import Callable
from loguru import logger

from backend.services.intelligence.recipe_sanitizer import _col_centers

_MIN_COLUMN_HEIGHT_MM: float = float(os.getenv("MIN_COLUMN_HEIGHT_MM", "1000"))

# Explicit singular substrings used to identify element types in Revit warning text.
# rstrip("s") is NOT used — "structural_framing" has no trailing s and would not singularize.
_ELEMENT_SINGULAR: dict[str, str] = {
    "structural_framing": "framing",
    "columns":            "column",
    "walls":              "wall",
}


def handle_warnings(
    warnings: list[str],
    recipe: dict,
) -> tuple[dict, list[str], list[str]]:
    """
    Apply deterministic fixes for known Revit warning patterns.

    Mutates *recipe* in place; returns the same object.  The caller (orchestrator)
    already owns an isolated copy — no deepcopy needed here.

    Returns
    -------
    (recipe, applied_actions, unresolved_warnings)
        applied_actions     — human-readable description of each deterministic fix.
        unresolved_warnings — warnings that matched no pattern; pass to SemanticAnalyzer.
    """
    applied:    list[str] = []
    unresolved: list[str] = []

    # Build column geometry once for all handlers that need it (_fix_join_error).
    col_boxes = _col_centers(recipe)

    for warn in warnings:
        matched = False
        for pattern, handler in _PATTERNS:
            if pattern.search(warn):
                recipe, action = handler(recipe, warn, col_boxes)
                if action:
                    applied.append(action)
                matched = True
                break

        if not matched:
            unresolved.append(warn)

    if applied:
        logger.info("RevitWarningHandler: %d deterministic fix(es) applied", len(applied))
        for a in applied:
            logger.debug("  • %s", a)
    if unresolved:
        logger.info(
            "RevitWarningHandler: %d warning(s) unresolved → passed to AI",
            len(unresolved),
        )

    return recipe, applied, unresolved


# ── Handlers ──────────────────────────────────────────────────────────────────
# Signature: (recipe, warning_text, col_boxes) → (recipe, action_str | None)

def _fix_join_error(
    recipe: dict,
    warning: str,
    col_boxes: list[tuple[float, float, float, float]],
) -> tuple[dict, str | None]:
    """
    "Cannot keep elements joined" — remove the framing element whose endpoint is
    deepest inside a column bounding box.

    RecipeSanitizer's endpoint-snapping pass should prevent this, but a join error
    can still occur when two columns are very close together.  Removing the
    offending beam is the safe fallback.
    """
    framing = recipe.get("structural_framing", [])
    if not framing:
        logger.debug("_fix_join_error: no structural_framing in recipe — skipping")
        return recipe, None

    worst_idx:   int | None = None
    worst_score: float      = 0.0

    for i, beam in enumerate(framing):
        for pt_key in ("start_point", "end_point"):
            pt = beam.get(pt_key, {})
            px = float(pt.get("x", 0.0))
            py = float(pt.get("y", 0.0))
            for cx, cy, hw, hd in col_boxes:
                # Skip entirely-outside boxes without computing overlap
                if abs(px - cx) >= hw or abs(py - cy) >= hd:
                    continue
                score = min(hw - abs(px - cx), hd - abs(py - cy))
                if score > worst_score:
                    worst_score = score
                    worst_idx   = i

    if worst_idx is not None:
        removed = framing.pop(worst_idx)
        recipe["structural_framing"] = framing
        sp = removed.get("start_point", {})
        ep = removed.get("end_point",   {})
        return recipe, (
            f"framing[{worst_idx}] removed — endpoint {worst_score:.0f} mm inside column; "
            f"span ({sp.get('x',0):.0f},{sp.get('y',0):.0f})→"
            f"({ep.get('x',0):.0f},{ep.get('y',0):.0f}) mm"
        )

    # Pattern matched but no overlapping beam found — let AI handle it.
    logger.warning(
        "RevitWarningHandler: join-error pattern matched but no "
        "overlapping framing endpoint found — warning passed to AI"
    )
    return recipe, None


def _fix_identical(
    recipe: dict,
    warning: str,
    col_boxes: list[tuple[float, float, float, float]],
) -> tuple[dict, str | None]:
    logger.info(
        "RevitWarningHandler: identical-instances warning — "
        "already deduplicated in pipeline; no action needed"
    )
    return recipe, "identical-instances: pre-export dedup already applied (no action)"


def _fix_transient(
    recipe: dict,
    warning: str,
    col_boxes: list[tuple[float, float, float, float]],
) -> tuple[dict, str | None]:
    # No recipe change is possible — Revit has a modal dialog open.
    # The 25 s base back-off in RevitClient gives the user time to dismiss it.
    logger.warning(
        "RevitWarningHandler: ExternalEvent/Pending — Revit may have a modal dialog open. "
        "Dismiss it in Revit then the next retry will succeed.  No recipe change made."
    )
    return recipe, "ExternalEvent/Pending: transient Revit state — no recipe fix; retry will proceed"


def _fix_missing_family(
    recipe: dict,
    warning: str,
    col_boxes: list[tuple[float, float, float, float]],
) -> tuple[dict, str | None]:
    lower = warning.lower()
    for key, singular in _ELEMENT_SINGULAR.items():
        if singular in lower or key in lower:
            count_before = len(recipe.get(key, []))
            if count_before:
                recipe[key] = []
                return recipe, (
                    f"missing-family: cleared {count_before} {key} entry/entries "
                    "— family RFA not in project"
                )
    return recipe, None


def _fix_short_column(
    recipe: dict,
    warning: str,
    col_boxes: list[tuple[float, float, float, float]],
) -> tuple[dict, str | None]:
    patched = 0
    for col in recipe.get("columns", []):
        h = float(col.get("height", _MIN_COLUMN_HEIGHT_MM))
        if h < _MIN_COLUMN_HEIGHT_MM:
            col["height"] = _MIN_COLUMN_HEIGHT_MM
            patched += 1
    if patched:
        return recipe, (
            f"short-column: raised height to {_MIN_COLUMN_HEIGHT_MM:.0f} mm "
            f"on {patched} column(s)"
        )
    return recipe, None


# ── Pattern → handler registry ────────────────────────────────────────────────
# Defined after the handler functions so function references resolve correctly.
# Tried in order; first match wins.
_PATTERNS: list[tuple[re.Pattern, Callable]] = [
    (re.compile(r"cannot (be )?kept? (elements )?joined|cannot keep.*joined|"
                r"highlights elements whose.*join", re.I),   _fix_join_error),
    (re.compile(r"identical.*instance|instance.*identical",  re.I),   _fix_identical),
    (re.compile(r"ExternalEvent.*[Pp]ending|event.*rejected.*pending", re.I), _fix_transient),
    (re.compile(r"unresolved references?|missing.*family|family.*not.*load", re.I), _fix_missing_family),
    (re.compile(r"column is too short|too short to",         re.I),   _fix_short_column),
]
