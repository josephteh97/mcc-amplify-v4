"""
Slab Thickness Parser — extract per-region slab thickness from a born-digital
PDF drawing page.

Construction drawings encode slab thickness via zone codes dropped on plan
regions:
  - Lookup codes       — `NSP\\d+`:  resolved against a NOTES legend block.
  - Self-describing    — `\\d{3}CIS`:  leading digits are the thickness in mm
                          (e.g. `300CIS` → 300 mm, cast-in-situ).

Two public functions, no class needed:

  extract_notes_legend(page) → {CODE: thickness_mm}
      Parses the NOTES block(s) on the page and returns a {code → thickness}
      map. Later-occurring legend entries override earlier ones (a drawing
      with separate GENERAL NOTES and STRUCTURAL NOTES blocks will keep the
      STRUCTURAL value).

  locate_zone_labels(page) → [(CODE, x_center_pt, y_center_pt), ...]
      Finds zone labels placed on the plan (not inside any NOTES block),
      returning the word-centre in PDF points. Caller is responsible for
      converting to mm via the page's grid scale.

Algorithm — NOTES block detection:
  An anchor is any token matching `(?i)^NOTES:?$`. The block is grown from
  the anchor downward: each candidate word (y0 > anchor.y0 AND
  |x0 - anchor.x0| < COLUMN_TOL_PT) is accepted only if its y0 is within
  LINE_BREAK_GAP_PT of the previous accepted word. This cleanly separates
  the NOTES column from on-plan content printed further below in the same
  x-range.

Thresholds — tunable via env vars (defaults for 150 DPI structural plates):
  SLAB_LEGEND_COLUMN_TOL_PT    (default 300)  — horizontal column tolerance
  SLAB_LEGEND_LINE_GAP_PT      (default 100)  — max vertical gap within a
                                                 NOTES block
  SLAB_THICKNESS_MIN_MM        (default 100)  — legend integer plausibility
  SLAB_THICKNESS_MAX_MM        (default 600)

Deterministic. No OCR, no LLM — all parsing is text-layer extraction via
PyMuPDF's `page.get_text("words")`.
"""
from __future__ import annotations

import os
import re

import fitz
from loguru import logger


_LOOKUP_RE   = re.compile(r"^NSP\d+$",            re.IGNORECASE)
_SELFDESC_RE = re.compile(r"^(\d{3})CIS$",        re.IGNORECASE)
_NOTES_RE    = re.compile(r"^NOTES:?$",           re.IGNORECASE)
_INT_RE      = re.compile(r"^\d+$")

_COLUMN_TOL_PT     = float(os.getenv("SLAB_LEGEND_COLUMN_TOL_PT", "300"))
_LINE_BREAK_GAP_PT = float(os.getenv("SLAB_LEGEND_LINE_GAP_PT",   "100"))
_MIN_THICK_MM      = int(os.getenv("SLAB_THICKNESS_MIN_MM", "100"))
_MAX_THICK_MM      = int(os.getenv("SLAB_THICKNESS_MAX_MM", "600"))


def extract_notes_legend(
    page: fitz.Page,
    words: list[tuple] | None = None,
) -> dict[str, float]:
    """Return {CODE: thickness_mm} parsed from the NOTES block(s) on *page*.

    If *words* is provided, re-use it to skip a second `page.get_text("words")`
    parse (callers that also run locate_zone_labels should share the list).
    """
    if words is None:
        words = _safe_words(page)
    if not words:
        return {}

    legend: dict[str, float] = {}
    for _rect, block_words in _find_notes_blocks(words):
        _parse_block_legend(block_words, legend)

    logger.debug("extract_notes_legend: {} entries", len(legend))
    return legend


def locate_zone_labels(
    page: fitz.Page,
    words: list[tuple] | None = None,
) -> list[tuple[str, float, float]]:
    """Return [(CODE, cx_pt, cy_pt), ...] for zone labels NOT inside a NOTES block.

    If *words* is provided, re-use it to skip a second `page.get_text("words")`.
    """
    if words is None:
        words = _safe_words(page)
    if not words:
        return []

    block_rects = [rect for rect, _ in _find_notes_blocks(words)]
    labels: list[tuple[str, float, float]] = []
    for w in words:
        txt = (w[4] or "").strip()
        if not (_LOOKUP_RE.match(txt) or _SELFDESC_RE.match(txt)):
            continue
        if _in_any_rect(w, block_rects):
            continue
        cx = (w[0] + w[2]) / 2.0
        cy = (w[1] + w[3]) / 2.0
        labels.append((txt.upper(), cx, cy))

    logger.debug("locate_zone_labels: {} labels", len(labels))
    return labels


def resolve_code_thickness(code: str, legend: dict[str, float] | None) -> float | None:
    """Return thickness_mm for *code*, or None if it can't be resolved.

    Self-describing (`\\d{3}CIS`) yields the leading integer.
    Lookup (`NSP\\d+`) returns legend[code.upper()] if present.
    """
    up = (code or "").upper()
    m = _SELFDESC_RE.match(up)
    if m:
        return float(m.group(1))
    if _LOOKUP_RE.match(up) and legend and up in legend:
        return float(legend[up])
    return None


# ── Internal helpers ──────────────────────────────────────────────────────────

def _safe_words(page: fitz.Page) -> list[tuple]:
    try:
        return page.get_text("words")
    except Exception as exc:
        logger.warning("slab_thickness_parser: page.get_text failed: {}", exc)
        return []


def _find_notes_blocks(
    words: list[tuple],
) -> list[tuple[tuple[float, float, float, float], list[tuple]]]:
    """Return [(block_rect, block_words)] for every NOTES anchor on the page.

    Block is grown from the anchor downward; growth stops at a vertical
    gap > _LINE_BREAK_GAP_PT so plan content printed in the same column
    (far below) is not swept into the NOTES block.
    """
    blocks = []
    for anchor in words:
        if not _NOTES_RE.match((anchor[4] or "").strip()):
            continue
        candidates = sorted(
            (w for w in words
             if w[1] > anchor[1] and abs(w[0] - anchor[0]) < _COLUMN_TOL_PT),
            key=lambda w: w[1],
        )
        block_words: list[tuple] = []
        last_y = anchor[1]
        for w in candidates:
            if w[1] - last_y > _LINE_BREAK_GAP_PT:
                break
            block_words.append(w)
            last_y = max(last_y, w[1])
        if not block_words:
            continue
        rect = (
            min(w[0] for w in block_words),
            min(w[1] for w in block_words),
            max(w[2] for w in block_words),
            max(w[3] for w in block_words),
        )
        blocks.append((rect, block_words))
    return blocks


def _parse_block_legend(block_words: list[tuple], legend: dict[str, float]) -> None:
    """Populate *legend* with entries parsed from one NOTES block's words.

    A clause's total slab thickness is the SUM of every in-range integer
    in that clause, so notes that combine an RC slab with a topping
    (e.g. "ALL NSP2 SHALL BE 130 THK + TOPPING 120 THK" → 250 mm) yield
    the full structural depth rather than just the RC layer.

    Clauses are split on "ALL" (each numbered note starts with "ALL") and
    on comma-terminated tokens, so a single numbered note that declares
    two zones on one wrapped line — "ALL NSP2 ... 130 + TOPPING 120, ALL
    NSP5 ... 150 + TOPPING 200." — partitions cleanly into NSP2=250 and
    NSP5=350 instead of one polluted sum.
    """
    for clause in _split_clauses(block_words):
        nsp_tokens = [w for w in clause if _LOOKUP_RE.match((w[4] or "").strip())]
        if not nsp_tokens:
            continue
        total = 0
        for w in clause:
            txt = (w[4] or "").strip()
            # Bare _INT_RE keeps list markers like "1." / "7." out of the
            # sum: their period prevents the match, so even if MIN_THICK_MM
            # is ever lowered we won't fold note-numbering into thicknesses.
            if not _INT_RE.match(txt):
                continue
            val = int(txt)
            if _MIN_THICK_MM <= val <= _MAX_THICK_MM:
                total += val
        if total <= 0:
            continue
        for nsp in nsp_tokens:
            legend[(nsp[4] or "").upper()] = float(total)

    # Self-describing codes record directly, no lookup needed.
    for w in block_words:
        m = _SELFDESC_RE.match((w[4] or "").strip())
        if m:
            legend[(w[4] or "").upper()] = float(m.group(1))


def _split_clauses(block_words: list[tuple]) -> list[list[tuple]]:
    """Partition NOTES words into clauses in reading order.

    Standard path: a clause begins at every "ALL" keyword (each numbered
    structural note begins with "ALL") and after any comma-terminated
    token. This survives PyMuPDF line-wrapping (a wrapped fragment is on
    a different line_no but still belongs to the same clause) and cleanly
    separates "ALL NSP2 ..., ALL NSP5 ..." into independent clauses.

    Fallback: if the block contains no "ALL" tokens (a drawing using
    different phrasing), partition per PyMuPDF line so we don't sum
    integers across unrelated rules into one polluted total.
    """
    ordered = sorted(block_words, key=lambda w: (w[5], w[6], w[0]))
    has_all = any((w[4] or "").strip().upper() == "ALL" for w in ordered)
    if not has_all:
        lines: dict[tuple[int, int], list[tuple]] = {}
        for w in ordered:
            lines.setdefault((w[5], w[6]), []).append(w)
        return list(lines.values())

    clauses: list[list[tuple]] = []
    current: list[tuple] = []
    for w in ordered:
        txt = (w[4] or "").strip()
        if txt.upper() == "ALL":
            if current:
                clauses.append(current)
            current = [w]
            continue
        current.append(w)
        if txt.endswith(","):
            clauses.append(current)
            current = []
    if current:
        clauses.append(current)
    return clauses


def _in_any_rect(word: tuple, rects: list[tuple[float, float, float, float]]) -> bool:
    wx, wy = word[0], word[1]
    return any(rx0 <= wx <= rx1 and ry0 <= wy <= ry1
               for (rx0, ry0, rx1, ry1) in rects)
