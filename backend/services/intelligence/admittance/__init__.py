"""
Admittance framework — unified YOLO-detection triage across element types.

YOLO is imperfect: it misses real elements, invents phantom ones, and clips
legitimate geometry (e.g. a beam detected only under its label text, cut
short of the column it actually frames into). The admittance layer sits
between detection and geometry generation, combining weak signals
(vector-path dashing, nearest-text legend match, grid alignment, neighbour
proximity) into a per-element Decision: admit, admit-with-fix (geometry
patch), or reject.

Every element type gets its own `rules/<type>_rules.py` module that
consumes shared `signals/` primitives and returns Decisions. The judge()
entrypoint dispatches by `det["type"]`.

See VALIDATION_AGENT.skill.md for the full signal catalogue, scoring
conventions, and how to add a new element type.
"""
from __future__ import annotations

from loguru import logger

from backend.services.intelligence.admittance.context import ElementContext
from backend.services.intelligence.admittance.scoring import Decision, ADMIT, REJECT, ADMIT_WITH_FIX
from backend.services.intelligence.admittance.rules import framing_rules, column_rules
from backend.services.intelligence.admittance.signals.legend_tag import looks_like_tag


_RULE_DISPATCH = {
    "structural_framing": framing_rules.judge,
    "column":             column_rules.judge,
}


def _build_indices(context: ElementContext) -> None:
    """Populate spatial indices on the context so signals query O(local) buckets."""
    bucket = context.BUCKET_PX
    pt_to_px = context.pt_to_px or 1.0

    # Paths bucketed by rect centre in px
    paths_bucketed: dict[tuple[int, int], list[dict]] = {}
    for p in context.vector_data.get("paths") or []:
        r = p.get("rect")
        if r is None:
            continue
        x0, y0, x1, y1 = (r.x0, r.y0, r.x1, r.y1) if hasattr(r, "x0") else r
        cx_px = (x0 + x1) * 0.5 * pt_to_px
        cy_px = (y0 + y1) * 0.5 * pt_to_px
        paths_bucketed.setdefault((int(cx_px // bucket), int(cy_px // bucket)), []).append(p)
    context._paths_bucketed = paths_bucketed

    # Pre-filter text spans to those that look like structural tags
    context._tag_spans = [
        s for s in (context.vector_data.get("text") or [])
        if looks_like_tag((s.get("text") or "").strip())
    ]


def judge(detections: list[dict], context: ElementContext) -> list[dict]:
    """
    Attach an admittance decision to each detection and apply in-place fixes.

    - ADMIT            → detection kept as-is
    - ADMIT_WITH_FIX   → detection kept with mutated bbox/center/metadata
    - REJECT           → detection retained in list but marked; caller filters

    Returns the same list (for chaining). Callers filter rejects via
        [d for d in dets if d["admittance_decision"]["action"] != "reject"]
    """
    _build_indices(context)

    counts = {"admit": 0, "admit_with_fix": 0, "reject": 0, "skipped": 0}
    for det in detections:
        rule = _RULE_DISPATCH.get(det.get("type"))
        if rule is None:
            det["admittance_decision"] = {"action": "admit", "reason": "no_rule_for_type"}
            counts["skipped"] += 1
            continue
        decision = rule(det, detections, context)
        _apply_decision(det, decision)
        counts[decision.action] += 1

    logger.info(
        "Admittance: {} admitted, {} admitted-with-fix, {} rejected, {} no-rule",
        counts["admit"], counts["admit_with_fix"], counts["reject"], counts["skipped"],
    )
    return detections


def _apply_decision(det: dict, decision: Decision) -> None:
    det["admittance_decision"] = {
        "action":   decision.action,
        "reason":   decision.reason,
        "signals":  decision.signals,
    }
    if decision.action == ADMIT_WITH_FIX and decision.bbox_override is not None:
        x1, y1, x2, y2 = decision.bbox_override
        det["bbox"]   = [x1, y1, x2, y2]
        det["center"] = [(x1 + x2) / 2.0, (y1 + y2) / 2.0]
    if decision.metadata:
        det.setdefault("admittance_metadata", {}).update(decision.metadata)


__all__ = ["judge", "ElementContext", "Decision", "ADMIT", "REJECT", "ADMIT_WITH_FIX"]
