"""
Build a {tag → material} lookup by scanning the drawing's NOTES / LEGEND
block.

Strategy:
  1. Fast path — regex-scan vector_data text spans. Structural drawings
     typically list beam tags in a notes column (e.g. "RCB3  300x600 RC",
     "SB2  UB 305x165 STEEL"). We pair tags with the material keyword
     that appears nearest in the same line.
  2. Vision fallback (optional) — if fewer than 2 tags are recovered
     from text and a raster + vision LLM are available, crop the
     top-right quadrant (where notes usually live) and ask the semantic
     analyzer to return JSON.

Both paths are best-effort; admittance/framing_rules still works without
a legend (falls back to prefix heuristics).
"""
from __future__ import annotations

import json
import re
from typing import Any

from loguru import logger


_TAG_RE = re.compile(r"\b([A-Z]{1,3}-?[A-Z]{0,3}\d{1,3}[A-Z]?)\b")
_RC_RE  = re.compile(r"\b(RC|R\.?C\.?|REINFORCED[\s-]?CONCRETE|CONCRETE)\b", re.IGNORECASE)
_STEEL_RE = re.compile(r"\b(STEEL|S\.?S\.?|UB|UC|H-?BEAM|UNIVERSAL)\b", re.IGNORECASE)


def parse_legend(vector_data: dict) -> dict[str, str]:
    """
    Return a {TAG: material} map where material ∈ {"rc", "steel"}.
    """
    texts = vector_data.get("text") or []
    if not texts:
        return {}

    # Group spans by approximate line (same y within 6pt).
    lines: list[list[dict]] = []
    for span in sorted(texts, key=lambda s: (s.get("bbox", [0, 0])[1], s.get("bbox", [0, 0])[0])):
        bbox = span.get("bbox") or [0, 0, 0, 0]
        y = (bbox[1] + bbox[3]) / 2.0
        if lines and abs(y - _line_y(lines[-1])) < 6.0:
            lines[-1].append(span)
        else:
            lines.append([span])

    legend: dict[str, str] = {}
    for line in lines:
        text = " ".join((s.get("text") or "").strip() for s in line)
        if not text:
            continue
        material: str | None = None
        if _RC_RE.search(text):
            material = "rc"
        elif _STEEL_RE.search(text):
            material = "steel"
        if not material:
            continue
        for tag in _TAG_RE.findall(text.upper()):
            # Skip common false matches (grid labels, dimension tags)
            if len(tag) < 3 or tag.isdigit():
                continue
            legend.setdefault(tag, material)

    if legend:
        logger.info("Legend parser (text): {} tag(s) recovered", len(legend))
    return legend


def _line_y(line: list[dict]) -> float:
    b = line[-1].get("bbox") or [0, 0, 0, 0]
    return (b[1] + b[3]) / 2.0


def enrich_with_vision(
    legend: dict[str, str],
    raster,
    semantic_analyzer: Any | None,
) -> dict[str, str]:
    """
    Optional second pass — crop the raster top-right quadrant and ask the
    vision LLM to extract a tag/material map. Only invoked when the text
    pass returned < 2 tags.
    """
    if len(legend) >= 2 or semantic_analyzer is None or raster is None:
        return legend
    try:
        from PIL import Image
        h, w = raster.shape[:2]
        # Notes typically live in the top-right quadrant of structural drawings.
        crop = raster[0 : h // 2, w // 2 : w]
        img = Image.fromarray(crop[:, :, ::-1])  # BGR→RGB
        prompt = (
            "This is the notes/legend block of a Singapore structural drawing. "
            "Extract every beam tag and its material. "
            "Return STRICT JSON of the form "
            '{"tags": [{"tag": "RCB3", "material": "rc"}, {"tag": "SB2", "material": "steel"}]}. '
            'Material must be exactly "rc" or "steel". No prose.'
        )
        response = semantic_analyzer._call_ollama(prompt, img, max_tokens=1024)
        parsed = json.loads(_extract_json(response))
        for entry in parsed.get("tags", []):
            tag = (entry.get("tag") or "").upper().strip()
            mat = (entry.get("material") or "").lower().strip()
            if tag and mat in ("rc", "steel"):
                legend.setdefault(tag, mat)
        logger.info("Legend parser (vision): {} tag(s) total after vision pass", len(legend))
    except Exception as e:
        logger.warning("Legend vision-fallback failed: {}", e)
    return legend


def _extract_json(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    return text[start : end + 1] if start != -1 and end > start else "{}"
