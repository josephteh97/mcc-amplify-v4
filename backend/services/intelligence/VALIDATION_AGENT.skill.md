---
name: validation-agent
description: Admittance framework — unified per-element triage across YOLO-detected structural elements (columns, framing, and future wall/slab/stairs/lift).
type: skill
---

# Validation Agent — Admittance Framework

## Purpose

YOLO is imperfect. It misses thin walls, invents phantom columns at cap-hatch
patterns, and clips beams short of the columns they actually frame into.
A single hard-coded rule ("distance < threshold → reject") drops real
elements along with the junk.

The **admittance framework** replaces that per-rule logic with a pluggable
judgment pass. Each element type has a rule module that combines weak
signals (vector dashing, nearest legend tag, grid alignment, neighbour
proximity) into one of three decisions:

- **`admit`** — keep as-is.
- **`admit_with_fix`** — keep, but mutate bbox / metadata (e.g. snap beam
  endpoints to column faces, tag material as `rc` / `steel`).
- **`reject`** — drop before geometry generation.

The framework is **generalisable** — columns, framing, walls, slabs,
stairs, lifts all plug in the same way.

## Where it lives

```
backend/services/intelligence/admittance/
    __init__.py            # judge(detections, context) — public entrypoint
    context.py             # ElementContext dataclass
    scoring.py             # Decision + admit/reject/admit_with_fix helpers
    legend_parser.py       # {tag → material} from notes block
    signals/
        dashline.py        # stroke style (dashed = steel, solid = rc)
        legend_tag.py      # nearest tag span → material via legend map
        grid_alignment.py  # long axis aligned with grid line?
        proximity.py       # nearest-neighbor of a given type
    rules/
        framing_rules.py   # beam-column join judgment + material tag
        column_rules.py    # off-grid deletion
        slab_rules.py      # area floor, grid-envelope check, overlap dedup
        # wall_rules.py    # TODO
```

## Framing rule (reference implementation)

A beam whose centre lies within `1.5 × short_dim` of a column centre
triggers Revit's "Cannot keep elements joined" error. Hard rejection
drops real beams that simply terminate at a column face — which is how
most beams actually connect. We instead score corroborating signals:

| Signal                                                        | Score |
| ------------------------------------------------------------- | :---: |
| Nearest tag in the legend map + dashline-inferred material agrees | +3 |
| Nearest tag present (no legend match)                         | +2 |
| Beam long axis aligned with a grid line                       | +2 |
| Stroke style resolvable (dashed or solid)                     | +1 |
| Distance > 0.5 × short_dim (not perfectly coincident)         | +1 |

**Thresholds:**
- Score ≥ 3 → `admit_with_fix` (snap both endpoints to column faces)
- Score == 2 → `admit` (borderline — keep, no geometry change)
- Score <  2 → `reject` (no corroboration — likely YOLO noise)

Material is tagged on every admitted beam via
`admittance_metadata["material"]` ∈ `{"rc", "steel"}` so
`geometry_generator` can emit the correct family name
(`RCBeam800x600mm`, `SteelBeam300x150mm`, …).

## Drawing conventions relied on (Singapore structural drawings)

- **RC beams** draw with **solid** outlines; tags like `RCB1`, `H-RCB3`.
- **Steel beams** draw with **dashed** outlines; tags like `SB1`, `SB2`.
- **Notes / legend block** typically sits in the top-right quadrant and
  contains lines like `RCB3  300x600 RC` or `SB2  UB 305x165 STEEL`.
- **Grid labels** are letters (AA…DD) along one axis, numbers (1…42)
  along the other. Beam long axis nearly always sits on a labelled line.

`legend_parser.parse_legend()` scans text spans for these conventions.
If text parsing fails (fewer than 2 tags), `enrich_with_vision()` falls
back to cropping the raster top-right quadrant and asking the SEA-LION
vision model for a `{tags: [{tag, material}]}` JSON response.

## Adding a new element type

1. **Decide the signals** — what makes a real vs phantom instance of
   this type distinguishable? Reuse existing signals where possible;
   add a new module under `signals/` only if the primitive is novel.
2. **Write `rules/<type>_rules.py`** exposing
   `judge(det, siblings, ctx) -> Decision`. Keep it short — compose
   signals, score, return one of `admit() / admit_with_fix() / reject()`.
3. **Register** in `admittance/__init__.py::_RULE_DISPATCH`.
4. **Document** the convention + scoring weights in this file so future
   maintainers can tune without re-reverse-engineering.
5. **Test** with a PDF that exercises the new type and inspect the
   admittance overlay (`data/debug/<job>_join_conflicts.png`).

## What the framework does NOT do

- **DfMA bay-spacing checks** (grid-level: min_bay_mm, max_bay_mm)
  remain in `validation_agent.enforce_rules`. Those are properties of
  the grid, not any single detection.
- **Grid detection** is upstream — admittance assumes `grid_info` is
  already authoritative.
- **Semantic inference** (building type, floor count, construction
  system) lives in `semantic_analyzer.py`, not here.

## Diagnostics

Every decision is recorded on the detection dict:

```python
det["admittance_decision"] = {
    "action":   "admit" | "admit_with_fix" | "reject",
    "reason":   "no_conflict" | "join_conflict_resolved" | "off_grid" | ...,
    "signals":  {"stroke": "solid", "tag": "RCB3", "score": 5, ...},
}
det["admittance_metadata"] = {"material": "rc", "tag": "RCB3", ...}   # admit_with_fix only
```

Inspect the debug overlay at `data/debug/<job_id>_join_conflicts.png`:
- Green bbox   → admitted with geometry fix
- Red   bbox   → rejected
- Yellow line  → link to the conflicting column centre

## Tuning knobs

- `framing_rules._JOIN_CLEARANCE_FACTOR` (default 1.5) — how close to a
  column before conflict is suspected.
- `framing_rules._TAG_SEARCH_RADIUS_PX` (default 120) — how far from
  beam centre we search for a tag label.
- `framing_rules._GRID_ALIGN_TOL_PX` (default 30) — grid-line alignment
  tolerance.

Tune cautiously — the scoring thresholds assume these values. If you
change a signal's effective range, revisit the thresholds too.
