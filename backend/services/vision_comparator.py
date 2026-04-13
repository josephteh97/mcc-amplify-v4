"""
Vision Comparator — Closed-Loop Accuracy Verification
======================================================

After the Claude agent builds a Revit model and exports a floor plan view image,
this module uses the same vision LLM (Gemini / Claude) to compare the generated
floor plan with the original PDF render and report discrepancies.

Usage (from orchestrator)
-------------------------
    comparator = VisionComparator()
    report = await comparator.compare(
        original_image_path = "data/jobs/{job_id}/render.jpg",
        revit_png_bytes      = <bytes from export_floor_plan_view>,
        job_id               = job_id,
    )
    # report = {
    #   "match_score": 0.85,           # 0.0–1.0 overall similarity
    #   "matches":     [...],           # elements present in both
    #   "missing":     [...],           # in PDF but not in Revit model
    #   "extra":       [...],           # in Revit model but not in PDF
    #   "notes":       "...",           # free-text summary from the LLM
    # }

The report is stored at ``data/jobs/{job_id}/vision_diff.json`` and included
in the pipeline result stats.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Optional

from loguru import logger

_BACKEND = os.getenv("SEMANTIC_MODEL_BACKEND", "gemini_api")

_COMPARISON_PROMPT = """\
You are a BIM quality-control expert comparing two architectural floor plan images.

Image 1 is the ORIGINAL floor plan (from a PDF — it may contain text labels,
hatch patterns, dimension lines, and layout annotations).

Image 2 is the GENERATED Revit floor plan (a clean vector export — it shows
only the structural elements that were placed by the BIM tool).

Your task: compare them and identify:
1. Elements that appear correctly in BOTH (matches)
2. Elements present in the ORIGINAL but MISSING from the Revit model
3. Elements present in the Revit model that are NOT in the original (extras)
4. Any obvious geometric errors (wrong position, wrong orientation, wrong size)

Respond ONLY with a valid JSON object — no markdown fences, no preamble:
{
  "match_score": <float 0.0–1.0, overall structural similarity>,
  "matches": [
    {"type": "column|wall|door|window", "description": "<brief location>"}
  ],
  "missing": [
    {"type": "column|wall|door|window", "description": "<what is missing and where>"}
  ],
  "extra": [
    {"type": "column|wall|door|window", "description": "<what is extra and where>"}
  ],
  "geometric_errors": [
    {"type": "...", "description": "<position/size error>"}
  ],
  "notes": "<one-sentence overall summary>"
}
"""


class VisionComparator:
    """
    Compares an original PDF floor plan render with a Revit-exported view
    using a multimodal LLM and returns a structured diff report.
    """

    def __init__(self):
        self._backend = _BACKEND
        logger.info(f"VisionComparator using backend: {self._backend}")

    async def compare(
        self,
        original_image_path: str,
        revit_png_bytes: bytes,
        job_id: str,
    ) -> dict:
        """
        Compare the original floor plan with the Revit-generated view.

        Parameters
        ----------
        original_image_path : str
            Path to the original render saved by the pipeline checkpoint
            (typically ``data/jobs/{job_id}/render.jpg``).
        revit_png_bytes : bytes
            Raw PNG bytes from ``RevitClient.export_floor_plan_view()``.
        job_id : str
            Job ID — used for saving the diff report to disk.

        Returns
        -------
        dict  {match_score, matches, missing, extra, geometric_errors, notes}
              or {error: str} if the comparison could not be completed.
        """
        orig_path = Path(original_image_path)
        if not orig_path.exists():
            logger.warning(
                f"VisionComparator: original image not found at {orig_path} — skipping"
            )
            return {"error": "original image not found", "match_score": None}

        if not revit_png_bytes:
            return {"error": "empty revit_png_bytes", "match_score": None}

        try:
            if self._backend == "gemini_api":
                report = await self._compare_gemini(orig_path, revit_png_bytes)
            elif self._backend == "anthropic_api":
                report = await self._compare_anthropic(orig_path, revit_png_bytes)
            else:
                logger.warning(
                    f"VisionComparator: backend '{self._backend}' does not support "
                    "vision comparison — skipping"
                )
                return {"error": f"backend '{self._backend}' not supported", "match_score": None}
        except Exception as exc:
            logger.error(f"VisionComparator failed for job {job_id}: {exc}")
            return {"error": str(exc), "match_score": None}

        # Persist the diff report alongside the other job artefacts
        diff_path = Path(f"data/jobs/{job_id}/vision_diff.json")
        diff_path.parent.mkdir(parents=True, exist_ok=True)
        with open(diff_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(
            f"Vision diff for job {job_id}: score={report.get('match_score')}, "
            f"missing={len(report.get('missing', []))}, "
            f"extra={len(report.get('extra', []))}"
        )
        return report

    # ── Gemini backend ─────────────────────────────────────────────────────────

    async def _compare_gemini(self, orig_path: Path, revit_bytes: bytes) -> dict:
        import asyncio
        from google import genai
        from google.genai import types as gtypes
        from utils.api_keys import get_google_api_key

        api_key = get_google_api_key()
        client  = genai.Client(api_key=api_key)

        orig_bytes = orig_path.read_bytes()
        orig_mime  = "image/jpeg" if orig_path.suffix.lower() in (".jpg", ".jpeg") else "image/png"

        parts = [
            gtypes.Part.from_bytes(data=orig_bytes,  mime_type=orig_mime),
            gtypes.Part.from_bytes(data=revit_bytes, mime_type="image/png"),
            gtypes.Part.from_text(_COMPARISON_PROMPT),
        ]

        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[gtypes.Content(role="user", parts=parts)],
            ),
        )
        return _parse_llm_response(response.text)

    # ── Anthropic backend ──────────────────────────────────────────────────────

    async def _compare_anthropic(self, orig_path: Path, revit_bytes: bytes) -> dict:
        import asyncio
        import anthropic

        api_key = os.getenv("ANTHROPIC_API_KEY")
        client  = anthropic.Anthropic(api_key=api_key)

        orig_bytes = orig_path.read_bytes()
        orig_mime  = "image/jpeg" if orig_path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
        orig_b64   = base64.standard_b64encode(orig_bytes).decode()
        revit_b64  = base64.standard_b64encode(revit_bytes).decode()

        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type":   "image",
                            "source": {"type": "base64", "media_type": orig_mime, "data": orig_b64},
                        },
                        {
                            "type":   "image",
                            "source": {"type": "base64", "media_type": "image/png", "data": revit_b64},
                        },
                        {"type": "text", "text": _COMPARISON_PROMPT},
                    ],
                }],
            ),
        )
        return _parse_llm_response(response.content[0].text)


# ── Shared JSON parser ─────────────────────────────────────────────────────────

def _parse_llm_response(text: str) -> dict:
    """Extract the JSON object from the LLM's text response."""
    import re
    # Strip any markdown code fences the model may have added
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fallback: find the first {...} block
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    logger.warning(f"VisionComparator: could not parse LLM response as JSON: {text[:200]}")
    return {
        "match_score": None,
        "matches": [], "missing": [], "extra": [], "geometric_errors": [],
        "notes": text[:500],
        "parse_error": True,
    }
