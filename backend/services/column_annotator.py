"""
Column Annotation Parser — extracted from PipelineOrchestrator.

5-pass strategy to match PDF text labels to YOLO-detected columns:
  Pass 1 — Global schedule scan (page 0 + schedule pages)
  Pass 2 — Per-column proximity search + schedule lookup
  Pass 3 — Vision LLM crop fallback (vector strokes invisible to text layer)
  Pass 4 — Single-scheme fallback (all columns share one type)
  Pass 5 — Safe structural default (800 mm)

Coordinate contract: this module NEVER modifies "center", "bbox", or any
coordinate field.  It only adds annotation metadata (width_mm, depth_mm,
is_circular, type_mark, diameter_mm).
"""

import copy
import math
import os
import re
from loguru import logger

# ── Column annotation regex patterns ─────────────────────────────────────────
# Matches: "800x800", "800X800", "800×800", "800*800"
_RE_RECT = re.compile(r'(\d{2,4})\s*[xX×*]\s*(\d{2,4})')
# Matches circular diameter in both orderings:
#   symbol-first: "Ø200", "⌀300", "∅300", "dia 200", "dia.200", "phi200"  → group 1
#   number-first: "300∅", "500Ø", "200⌀"  (CAD export format)             → group 2
_RE_CIRC = re.compile(
    r'(?:Ø|⌀|∅|dia\.?\s*|phi\s*)(\d{2,4})'   # group 1: symbol before number
    r'|(\d{2,4})\s*[Ø⌀∅]',                    # group 2: number before symbol
    re.IGNORECASE,
)
# Matches column type mark: any 1-3 uppercase letters followed by 1-3 digits
# e.g. "C1", "C20", "B3", "K14" — covers non-standard naming conventions
_RE_MARK = re.compile(r'\b([A-Z]{1,3}\d{1,3})\b')

# Beam / slab / lintel prefixes that must NOT be accepted as column marks.
_BEAM_MARK_PREFIX = re.compile(r'^(RCB|GB|SB|TB|FB|RB|SL|LB|L|B)\d', re.IGNORECASE)


def _is_beam_label(txt: str) -> bool:
    """True when *txt* carries a mark that identifies a beam/slab, not a column."""
    m = _RE_MARK.search(txt)
    return bool(m and _BEAM_MARK_PREFIX.match(m.group(1)))


def _scan_schedule_texts(texts, schedule: dict) -> None:
    """Parse type-mark + dimension pairs from *texts* into *schedule* dict."""
    for txt in texts:
        mark_m = _RE_MARK.search(txt)
        if not mark_m:
            continue
        mark = mark_m.group(1)
        if _BEAM_MARK_PREFIX.match(mark):
            continue
        rect_m = _RE_RECT.search(txt)
        circ_m = _RE_CIRC.search(txt)
        if rect_m and mark not in schedule:
            schedule[mark] = (float(rect_m.group(1)), float(rect_m.group(2)), False)
        elif circ_m and mark not in schedule:
            diam = float(circ_m.group(1) or circ_m.group(2))
            schedule[mark] = (diam, diam, True)


def annotate_columns(
    detections: dict,
    vector_data: dict,
    image_data: dict,
    extra_schedule_texts: list | None = None,
    semantic_ai=None,
) -> dict:
    """
    Parse column type marks and dimensions from PDF text annotations.

    Parameters
    ----------
    detections         : structured dict with "columns" key
    vector_data        : VectorProcessor output (text, page_rect, etc.)
    image_data         : StreamingProcessor output (image, width, height, dpi)
    extra_schedule_texts: text items from schedule-classified pages
    semantic_ai        : SemanticAnalyzer instance (needed for Pass 3 LLM calls)

    Returns
    -------
    Updated detections dict with annotated columns.
    """
    _SAFE_DEFAULT_MM = 800.0

    text_items = vector_data.get("text", [])
    page_rect  = vector_data.get("page_rect", [0, 0, 595, 842])
    img_w      = image_data.get("width", 1)
    img_h      = image_data.get("height", 1)

    pt_w = page_rect[2] - page_rect[0]
    pt_h = page_rect[3] - page_rect[1]
    sx   = img_w / pt_w if pt_w > 0 else 1.0
    sy   = img_h / pt_h if pt_h > 0 else 1.0

    # Project all text items into pixel space once
    text_px = []
    for t in text_items:
        bx = t.get("bbox", [0, 0, 0, 0])
        cx = (bx[0] + bx[2]) / 2 * sx
        cy = (bx[1] + bx[3]) / 2 * sy
        text_px.append((cx, cy, t["text"]))

    # ── Pass 1: Build global type-mark → dimension table ──────────────────
    schedule: dict[str, tuple] = {}
    _scan_schedule_texts((txt for _, _, txt in text_px), schedule)
    _scan_schedule_texts(extra_schedule_texts or [], schedule)

    if schedule:
        logger.info(
            f"Column schedule scan found {len(schedule)} type definition(s): "
            + ", ".join(
                f"{k}={'×'.join(str(int(v)) for v in schedule[k][:2])}"
                for k in sorted(schedule)
            )
        )

    # ── Pass 2: Annotate each detected column ─────────────────────────────
    columns      = copy.deepcopy(detections.get("columns", []))
    by_proximity = 0
    by_schedule  = 0
    by_llm       = 0
    by_default   = 0
    by_schedule_fallback = 0

    MAX_LLM_CALLS = int(os.getenv("VISION_ANNOTATION_MAX_CALLS", "10"))
    llm_calls     = 0
    llm_cache: dict = {}
    img_np = image_data.get("image")

    def _make_crop(col_dict):
        if img_np is None:
            return None
        from PIL import Image as _PIL
        img_h_f, img_w_f = img_np.shape[:2]
        bbox = col_dict.get("bbox", [])
        cx_f, cy_f = col_dict.get("center", [0.0, 0.0])
        bw_f = max(abs(bbox[2] - bbox[0]), 50) if len(bbox) >= 4 else 50
        bh_f = max(abs(bbox[3] - bbox[1]), 50) if len(bbox) >= 4 else 50
        pad_f = max(bw_f, bh_f) * 2.5
        x0_f = max(0, int(cx_f - bw_f / 2 - pad_f))
        y0_f = max(0, int(cy_f - bh_f / 2 - pad_f))
        x1_f = min(img_w_f, int(cx_f + bw_f / 2 + pad_f))
        y1_f = min(img_h_f, int(cy_f + bh_f / 2 + pad_f))
        if x1_f - x0_f < 20 or y1_f - y0_f < 20:
            return None
        return _PIL.fromarray(img_np[y0_f:y1_f, x0_f:x1_f])

    def _apply(col, w, d, is_circ, mark=None):
        col["width_mm"]    = w
        col["depth_mm"]    = d
        col["is_circular"] = is_circ
        if is_circ:
            col["diameter_mm"] = w
        if mark:
            col["type_mark"] = mark

    unresolved = []

    for col in columns:
        cx, cy = col.get("center", [0.0, 0.0])
        bbox   = col.get("bbox", [0, 0, 0, 0])
        col_w  = abs(bbox[2] - bbox[0]) if len(bbox) >= 4 else 100
        col_h  = abs(bbox[3] - bbox[1]) if len(bbox) >= 4 else 100

        search_r = max(col_w, col_h, 200) * 3.0

        nearby = sorted(
            [(d, txt)
             for tx, ty, txt in text_px
             if (d := math.hypot(tx - cx, ty - cy)) < search_r],
            key=lambda x: x[0],
        )

        matched = False

        # (a) Proximity — text item contains both mark and dimensions
        for _, txt in nearby:
            if _is_beam_label(txt):
                continue
            rect_m = _RE_RECT.search(txt)
            if rect_m:
                mark = _RE_MARK.search(txt)
                _apply(col,
                       float(rect_m.group(1)), float(rect_m.group(2)),
                       False, mark.group(1) if mark else None)
                by_proximity += 1
                matched = True
                break
            circ_m = _RE_CIRC.search(txt)
            if circ_m:
                diam = float(circ_m.group(1) or circ_m.group(2))
                mark = _RE_MARK.search(txt)
                _apply(col, diam, diam, True, mark.group(1) if mark else None)
                by_proximity += 1
                matched = True
                break

        if matched:
            continue

        # (b) Proximity — nearby text has only a type mark → look up schedule
        for _, txt in nearby:
            if _is_beam_label(txt):
                continue
            mark_m = _RE_MARK.search(txt)
            if mark_m:
                mark = mark_m.group(1)
                if _BEAM_MARK_PREFIX.match(mark):
                    continue
                if mark in schedule:
                    w, d, is_circ = schedule[mark]
                    _apply(col, w, d, is_circ, mark)
                    by_schedule += 1
                    matched = True
                    break

        if matched:
            continue

        # ── Pass 3: Vision LLM crop ──────────────────────────────────────
        if semantic_ai is not None and llm_calls < MAX_LLM_CALLS:
            tm = col.get("type_mark")
            if tm and tm in llm_cache:
                cached = llm_cache[tm]
                _apply(col, cached["w"], cached["d"], cached["is_circ"], tm)
                by_llm += 1
                matched = True
            else:
                crop = _make_crop(col)
                if crop is not None:
                    llm_calls += 1
                    result_ann = semantic_ai.read_element_annotation(crop)
                    resolved_mark = result_ann.get("type_mark") or tm

                    if result_ann.get("is_circular") and result_ann.get("diameter_mm"):
                        w = d = float(result_ann["diameter_mm"])
                        _apply(col, w, d, True, resolved_mark)
                        cache_key = resolved_mark or f"_llm{llm_calls}"
                        llm_cache[cache_key] = {"w": w, "d": d, "is_circ": True}
                        by_llm += 1
                        matched = True

                    elif result_ann.get("width_mm") and result_ann.get("depth_mm"):
                        w = float(result_ann["width_mm"])
                        d = float(result_ann["depth_mm"])
                        _apply(col, w, d, False, resolved_mark)
                        cache_key = resolved_mark or f"_llm{llm_calls}"
                        llm_cache[cache_key] = {"w": w, "d": d, "is_circ": False}
                        by_llm += 1
                        matched = True

                    else:
                        logger.debug(
                            f"LLM crop returned no dimensions for column "
                            f"{col.get('id')} (mark={tm}) — column remains unresolved."
                        )

        if not matched:
            unresolved.append(col)

    # ── Pass 4: Single-scheme fallback ────────────────────────────────────
    if unresolved and len(schedule) == 1:
        single_type, (w, d, is_circ) = next(iter(schedule.items()))
        logger.info(
            f"Applying single schedule type '{single_type}' ({w:.0f}×{d:.0f}mm) "
            f"to {len(unresolved)} unresolved columns"
        )
        for col in unresolved:
            _apply(col, w, d, is_circ, single_type)
        by_schedule_fallback = len(unresolved)
        unresolved = []

    # ── Pass 5: Safe structural default ───────────────────────────────────
    for col in unresolved:
        _apply(col, _SAFE_DEFAULT_MM, _SAFE_DEFAULT_MM, False)
        by_default += 1

    if llm_calls >= MAX_LLM_CALLS:
        remaining = sum(1 for c in columns if "width_mm" not in c)
        if remaining:
            logger.warning(
                f"Vision LLM cap reached ({MAX_LLM_CALLS} calls) — "
                f"{remaining} element(s) still unresolved will use the safe "
                f"structural default ({_SAFE_DEFAULT_MM:.0f} mm). "
                f"For higher accuracy (e.g. cost estimation), set env var "
                f"VISION_ANNOTATION_MAX_CALLS={MAX_LLM_CALLS * 5} or higher."
            )

    total = len(columns)
    logger.info(
        f"Column annotation: {by_proximity} proximity, "
        f"{by_schedule} via schedule table, "
        f"{by_llm} via vision LLM ({llm_calls} API call(s)), "
        f"{by_schedule_fallback} via single-scheme fallback, "
        f"{by_default} defaulted to {_SAFE_DEFAULT_MM:.0f}mm "
        f"(total {total})"
    )

    result = dict(detections)
    result["columns"] = columns
    return result
