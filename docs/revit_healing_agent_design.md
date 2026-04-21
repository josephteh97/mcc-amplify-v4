# Revit Self-Healing Validation Agent — Phase 0 Design Document

**Date:** 2026-04-20  
**Status:** Phase 0 complete — gate review before Phase 1 implementation

---

## 1. What Does Not Exist (and What the Prompt Assumed)

The original brief referenced `pipeline_controller.py` with agent registration, `@memory_first` wrapping,
and inter-agent negotiation.  **None of this exists.**  A full audit found zero occurrences of:

| Symbol assumed by prompt | Reality |
|---|---|
| `pipeline_controller.py` | File does not exist anywhere in the repo |
| `REGISTRY` / `@register_agent` | No such pattern in any `.py` file |
| `@memory_first` decorator | Does not exist |
| `coordinator` / `negotiat*` patterns | Not present |
| MCP server process | `revit_agent.py` uses embedded Anthropic SDK; no separate process |

All orchestration lives in a single class: `PipelineOrchestrator` in
`backend/services/core/orchestrator.py` (~870 lines).  Direct method calls, no plugin registry.

This means the "agent-coordinator" pattern described in the brief must be **designed and built from
scratch** as part of Phase 2.

---

## 2. Current Warning Propagation Path

```
Revit C# Add-in (Windows :5000)
  └─ IFailuresPreprocessor.PreprocessFailures()         App.cs:1597
       └─ WarningCollector.PreprocessFailures()         App.cs:1590–1603
            collects: List<(string text, List<int> elementIds)>
            AUTO-RESOLVED warnings prefixed: "[JOIN-RESOLVED: ...]"
  └─ HTTP response header serialization                 App.cs:403
       X-Revit-Warnings: JSON string array (text only — element IDs not yet piped)
       binary .rvt in response body

revit_client.py (Ubuntu)
  └─ build_model() receives HTTP response               revit_client.py:168–173
       raw_hdr = response.headers.get("x-revit-warnings", "[]")
       parsed  = json.loads(raw_hdr)
       warnings = [str(w) for w in parsed if isinstance(parsed, list)]
       returns: tuple[str, list[str]]  → (rvt_path, warnings)

rvt_exporter.export() wraps revit_client.build_model()
  └─ returns (rvt_path, revit_warnings) to orchestrator

orchestrator.py correction loop                        orchestrator.py:451–527
  for _attempt in range(3):
    rvt_path, revit_warnings = await self.rvt_exporter.export(...)
    if not revit_warnings or _attempt == 2: break
    # Step 1 — deterministic handler
    current_recipe, det_actions, unresolved = handle_revit_warnings(revit_warnings, current_recipe)
    # Step 2 — AI fallback for unresolved
    corrections = await self.semantic_ai.analyze_revit_warnings(unresolved, current_recipe)
    current_recipe = self._apply_revit_corrections(current_recipe, corrections)
```

### Key invariants

- `[JOIN-RESOLVED: ...]` warnings are already resolved by Revit; they arrive in the list but must
  never trigger re-correction.  The `_fix_join_error` regex does not match this prefix (tested).
- Element IDs are collected in C# (`WarningDetails`) but **not yet serialized** to Python.
  This is a **blocker** for the clash resolver (Phase 1 Add-in work).
- Pre-export: `sanitize_recipe()` runs before any Revit call (`orchestrator.py:414`); it snaps beam
  endpoints to column centers and clamps undersized columns.

---

## 3. Existing Retry Policy

| Dimension | Current behaviour |
|---|---|
| Attempts | 3 whole-recipe retries (attempts 0, 1, 2) |
| Step 1 | Deterministic handler (`revit_warning_handler.handle_warnings`) |
| Step 2 | AI fallback (`semantic_ai.analyze_revit_warnings`) for unresolved |
| Short-circuit | If deterministic fixes all warnings → `continue` (no AI call) |
| Exhausted | Warnings accepted as-is; `rvt_status = "warnings_accepted"` |
| Per-element retry | None — whole recipe only |
| Back-off | None — immediate retry |
| Transient / modal | `_fix_transient` logs but does no recipe change; 25 s back-off lives in `RevitClient` |

**Gap:** no exponential back-off between attempts; no distinction between recoverable and fatal
warnings on attempt 3 (both result in `warnings_accepted`).

---

## 4. Existing SQLite Stores

| Store | File | Table | Key columns |
|---|---|---|---|
| `JobStore` | `data/jobs.db` | `jobs` | job_id TEXT PK, data TEXT, created_at REAL, accessed_at REAL |
| `CorrectionsLogger` | `data/corrections.db` | `corrections` | id INTEGER PK, timestamp REAL, job_id TEXT, element_type TEXT, element_index INTEGER, original_element TEXT, changes TEXT, is_delete INTEGER |

Phase 2 will need a new `revit_warnings.db` (or a new table in `corrections.db`) to store per-warning
history for the memory-first lookup pattern.

---

## 5. Deterministic Handler — Patterns Already Implemented

`backend/services/revit_warning_handler.py` covers:

| Pattern | Handler | Effect |
|---|---|---|
| `cannot keep.*joined` / `highlights elements whose.*join` | `_fix_join_error` | Remove framing element deepest inside a column bbox |
| `identical.*instance` | `_fix_identical` | Log-only; dedup already done pre-export |
| `ExternalEvent.*Pending` | `_fix_transient` | Log only; 25 s back-off in RevitClient handles this |
| `unresolved references?` / `missing.*family` | `_fix_missing_family` | Clear that element type from recipe |
| `column is too short` | `_fix_short_column` | Raise height to `MIN_COLUMN_HEIGHT_MM` (default 1000 mm) |

`_col_centers()` shared between `recipe_sanitizer.py` and `revit_warning_handler.py` via import.

---

## 6. Gaps — What Phases 1–4 Must Build

### Phase 1 — Add-in additions (C#, Windows)

**Blocker A: Element ID enrichment**

`WarningCollector` already collects `List<int> elementIds` per warning (App.cs:1590–1591) but the
serialization at App.cs:403 only writes the text string.

```
// Current (App.cs ~403)
var warningTexts = _warningCollector.Warnings.Select(w => w.Text).ToList();
response.Headers.Add("X-Revit-Warnings", JsonConvert.SerializeObject(warningTexts));

// Target (v2)
var warningObjects = _warningCollector.Warnings.Select(w => new {
    text       = w.Text,
    elementIds = w.ElementIds,
}).ToList();
response.Headers.Add("X-Revit-Warnings",         JsonConvert.SerializeObject(warningObjects));
response.Headers.Add("X-Revit-Warnings-Version", "2");
```

`revit_client.py` must detect `X-Revit-Warnings-Version: 2` and parse accordingly (backwards
compatible: default to v1 string-list if header absent).

**Blocker B: Geometry/bounding-box query endpoint**

No such endpoint exists today.  The clash resolver needs to ask Revit:
"What is the current bounding box of element #42?"

New endpoint:

```
POST /session/{id}/query-elements
Body: { "element_ids": [int]?, "category": string? }     // either filter; both optional = all
Response: {
  "elements": [
    {
      "id": 123456,
      "category": "StructuralColumns",
      "family":   "M_Concrete-Rectangular-Column",
      "type":     "300 x 600mm",
      "bbox":     { "min": {"x": ..., "y": ..., "z": ...}, "max": {...} },    // mm
      "location": { "x": ..., "y": ..., "z": ... }                             // mm
    }
  ],
  "truncated": true    // present + true only when a category/all scan hit QUERY_ELEMENTS_MAX_SCAN
}
```

Implementation notes:
- Must execute on the Revit thread via the existing `CommandHandler` / `RunOnRevitThread` plumbing.
- Use `doc.GetElement(id).get_BoundingBox(null)` for bounds; return `null` for elements without geometry.
- Convert feet → mm at the C# boundary so Python sees consistent units.
- Filter via `FilteredElementCollector(doc).OfCategory(BuiltInCategory.OST_*)` when `element_ids` is omitted.
- Unfiltered/category scans are capped at `QUERY_ELEMENTS_MAX_SCAN` (5000) elements; response
  includes `"truncated": true` when hit.  Callers needing a larger set must pass explicit `element_ids`.

Approx 60 lines of C# in App.cs.  Ship with a minimal fixture test (load known .rvt → assert bbox
shape for one column).  Hard blocker for Phase 3 `clash_resolver.py`.

### Phase 2 — Agent core (Python, Ubuntu)

`pipeline_controller.py` does not exist and must be created as a **thin coordinator** that:

1. Wraps `PipelineOrchestrator` without modifying the frozen column pipeline
2. Provides the `@memory_first` pattern: check `revit_warnings.db` for a known fix before calling AI
3. Registers the new agent modules via a lightweight dict (no decorator magic needed)
4. Routes unresolved warnings to the appropriate specialist handler

Suggested minimal structure:

```python
# backend/services/core/pipeline_controller.py
class PipelineController:
    def __init__(self, orchestrator: PipelineOrchestrator):
        self._orch = orchestrator
        self._memory = RevitWarningMemory("data/revit_warnings.db")
        self._agents: dict[str, WarningHandler] = {
            "join_error":       JoinErrorHandler(),
            "missing_family":   MissingFamilyHandler(),
            "short_column":     ShortColumnHandler(),
            "clash":            ClashResolver(),   # Phase 3, requires Blocker B
        }

    async def run(self, job_id, pdf_path, observer):
        # delegates entirely to orchestrator; intercepts only the warning-correction step
        return await self._orch.run(job_id, pdf_path, observer,
                                    warning_hook=self._handle_warnings)

    async def _handle_warnings(self, warnings, recipe):
        # memory-first: known fix → apply without AI
        # unknown: route to specialist agent or AI fallback
        ...
```

### Phase 2.5 — Contracts fixed before implementation starts

These are decisions that must be nailed down at the Phase 0 gate so Phase 2 doesn't silently pick
them later.

**`RevitFailure` dataclass** — canonical shape for all downstream components:

```python
@dataclass
class RevitFailure:
    raw_text: str                         # warning string from header
    element_ids: list[int]                # [] on v1, populated on v2
    severity: Literal["warning", "resolved", "error"]
    auto_resolved: bool                   # True iff raw_text starts with "[JOIN-RESOLVED:"
    signature: str                        # normalized form for KB + memory key
    transaction_context: str | None       # which /build-model or /place call produced it
```

`auto_resolved=True` rows go to `revit_healing_audit` and are skipped by the healing loop.

**`FailureCategory` enum** — classification output driving handler dispatch:

`OFF_AXIS | OVERLAP | MISSING_HOST | CONSTRAINT_VIOLATION | TYPE_RESOLUTION | SKETCH_INVALID | CLASH | UNKNOWN`

Classification order: `signature_regex` match from KB → LLM fallback with constrained enum output.

**Healing-loop defaults**:

- `MAX_ATTEMPTS = 4` per element (up from 3 whole-recipe today).
- `LLM_BUDGET = 2` calls per element; exceeding either ceiling escalates.
- This is a **per-element** retry model — a significant shift from today's whole-recipe retry loop
  in `orchestrator.py:451`. The old loop stays as the outer envelope; the new per-element loop sits
  inside `PipelineController._handle_warnings` once element IDs (v2) are available.

**Memory / KB lookup order**:

1. `memory.lookup((element.category, failure.signature))` — memoized plan.
2. `kb.lookup(failure.signature)` — YAML regex match in `revit_practice_kb.yaml`.
3. LLM fallback — constrained JSON matching `CorrectionPlan` schema. LLM plans are **mirrored to
   `docs/kb_candidates.jsonl`** for deliberate human promotion into the YAML KB. The KB grows by
   review, not silently.

**Element registry** — `Protocol`-based, not decorator-based:

```python
class ElementHandler(Protocol):
    category: str
    def placement_params(self, src: SourceGeometry) -> PlacementParams: ...
    def validate_post_placement(self, element_id: int) -> list[RevitFailure]: ...

REGISTRY: dict[str, ElementHandler] = {}
def register(handler: ElementHandler) -> None: REGISTRY[handler.category] = handler
```

Ship 5 concrete handlers + `GenericElementHandler` KB-only fallback.  New category = new file +
one `register()` call.

**SQLite storage policy** — migration files, never inline `CREATE TABLE`:

No migration infrastructure exists in the repo today — today's stores (`JobStore`,
`CorrectionsLogger`) each call `CREATE TABLE IF NOT EXISTS` in their `__init__`.  To honour the
brief's "migration file, not inline CREATE TABLE" rule we must also create lightweight migration
scaffolding:

```
backend/db/
  migrations/
    001_revit_healing_memory.sql
    002_revit_healing_audit.sql
  migrator.py           ← applies pending .sql files by filename order, tracks in schema_migrations
```

Two new tables:

```sql
-- 001_revit_healing_memory.sql
CREATE TABLE revit_healing_memory (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    element_category  TEXT    NOT NULL,   -- "column", "structural_framing", ...
    failure_signature TEXT    NOT NULL,   -- normalized from RevitFailure.signature
    plan_json         TEXT    NOT NULL,   -- CorrectionPlan serialized
    source            TEXT    NOT NULL,   -- "memory" | "kb" | "llm"
    success_count     INTEGER NOT NULL DEFAULT 0,
    fail_count        INTEGER NOT NULL DEFAULT 0,
    last_used_at      REAL    NOT NULL,
    UNIQUE(element_category, failure_signature)
);

-- 002_revit_healing_audit.sql
CREATE TABLE revit_healing_audit (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id        TEXT    NOT NULL,
    attempt       INTEGER NOT NULL,
    failure_json  TEXT    NOT NULL,   -- RevitFailure serialized
    plan_json     TEXT    NOT NULL,   -- CorrectionPlan applied
    outcome       TEXT    NOT NULL,   -- "success" | "fail" | "escalated" | "auto_resolved"
    logged_at     REAL    NOT NULL
);
```

### Phase 3 — Handlers + clash resolver

Five specialist handlers + generic fallback:
`ColumnHandler`, `StructuralFramingHandler`, `SlabHandler`, `StairsHandler`, `LiftHandler`,
`GenericElementHandler`.

**Clash resolver pair matrix** — default resolutions when two bboxes intersect:

| Pair | Resolution |
|---|---|
| column × column | merge if same family + centers within 50 mm; else shift lower-priority along grid |
| beam × column | trim beam to column face (Revit native join via existing session API) |
| beam × slab | set beam Z-justification to top; if still clashing, lower beam by slab thickness |
| slab × column | cut slab around column (auto-join) |
| slab × lift | insert shaft opening cut through all slabs in lift span |
| slab × stairs | insert floor opening matching stair footprint + headroom extension |
| stairs × beam | raise beam or reroute stair run; flag if neither feasible |

No rule match → classify as `CLASH` → escalate with both IDs + bboxes logged to
`revit_healing_audit`.  Pair intersection is computed in Python from cached `/query-elements`
responses — no per-pair Revit round-trip.

### Phase 4 — Tests

- Unit: KB regex matching, handler classify, memory-first lookup
- Integration: replay test fixtures from `data/corrections.db`
- Golden placement: assert no framing collides with column bboxes post-sanitizer

---

## 7. Integration Plan (Minimal Diff)

The goal is zero changes to the frozen column pipeline and zero changes to
`grid_detector`/`geometry_generator` (except the separate framing stub fix).

```
Files to CREATE
  backend/services/core/pipeline_controller.py      ← new thin coordinator (brief assumed it exists)
  backend/agents/revit_validation_agent.py          ← main agent: parser + classifier + planner + loop
  backend/agents/revit_feedback_parser.py           ← header warnings → list[RevitFailure]
  backend/agents/failure_classifier.py              ← RevitFailure → FailureCategory
  backend/agents/correction_planner.py              ← memory → KB → LLM lookup order
  backend/agents/clash_resolver.py                  ← BLOCKED on Phase 1 Blocker B
  backend/agents/handlers/__init__.py               ← REGISTRY + register()
  backend/agents/handlers/column.py
  backend/agents/handlers/structural_framing.py
  backend/agents/handlers/slab.py
  backend/agents/handlers/stairs.py
  backend/agents/handlers/lift.py
  backend/agents/handlers/generic.py                ← KB-only fallback
  backend/services/revit_healing_memory.py          ← SQLite store (opens via migrator)
  backend/db/migrator.py                            ← new: applies pending .sql migrations
  backend/db/migrations/001_revit_healing_memory.sql
  backend/db/migrations/002_revit_healing_audit.sql
  config/revit_practice_kb.yaml                     ← SS CP 65 rules, regex patterns
  docs/kb_candidates.jsonl                          ← starts empty; LLM plans mirrored here
  tests/unit/test_revit_feedback_parser.py
  tests/unit/test_failure_classifier.py
  tests/unit/test_correction_planner.py             ← memory-first path, KB path, LLM mock path
  tests/unit/test_revit_healing_memory.py
  tests/integration/test_revit_correction_replay.py ← replay 10 captured warning headers
  tests/integration/test_query_elements_endpoint.py ← Phase 1 fixture test
  tests/golden/test_healing_golden.py               ← known-failing placement → heal → clean

Files to MODIFY (minimal)
  revit_server/RevitAddin/App.cs                    ← Phase 1: v2 warning header + /query-elements
  backend/services/revit_client.py                  ← Phase 1: v1/v2 schema detection
  backend/services/core/orchestrator.py             ← wire warning_hook parameter
  backend/main.py  (or wherever PipelineOrchestrator is instantiated)
                                                    ← swap in PipelineController

Files NOT to touch
  backend/services/intelligence/recipe_sanitizer.py  ← already clean, Phase 0 complete
  backend/services/revit_warning_handler.py          ← already clean (folds into handler later)
  backend/services/intelligence/column_detector.py   ← frozen
  backend/services/geometry_generator.py             ← frozen (framing stub fix = separate PR)
  backend/services/intelligence/grid_detector.py     ← frozen
```

---

## 8. Open Questions for Review

1. **`pipeline_controller.py` injection point** — should `PipelineController` replace
   `PipelineOrchestrator` at the FastAPI route level, or wrap it internally?  Replacing at the route
   level is cleaner but touches `main.py`; wrapping internally keeps the diff smaller.

2. **Warning memory keying** — key on canonical warning text (after stripping element IDs)?  Or hash
   the (warning_text, element_type) pair?  The latter handles "column is too short" differently for
   columns vs. walls.

3. **Clash resolver gate** — Phase 3 `clash_resolver.py` is blocked until Phase 1 Blocker B lands.
   Should Phase 2 ship a stub that logs "clash resolver unavailable — element ID query endpoint not
   yet implemented" and routes to AI fallback?  Or hold Phase 2 until both Add-in changes are done?

4. **Back-off** — should the 3-attempt loop gain exponential back-off (e.g. 2 s, 4 s)?  Currently
   there is none.  For `ExternalEvent/Pending` (user has a modal open) this would meaningfully help.

5. **`_build_structural_framing_parameters` stub** — `geometry_generator.py` line 582–583 returns
   `[]` unconditionally, dropping all framing detections.  This is a separate bug fix (not part of
   the healing agent) but should be tracked as a parallel PR so that the healing agent has real
   framing data to test against.

6. **Per-element vs whole-recipe retry** — the brief's healing loop retries `place_element(element)`
   per-element with `MAX_ATTEMPTS=4`.  The existing orchestrator retries the whole recipe with
   `range(3)`.  Switching to per-element requires either (a) the MCP-style step-by-step agent
   builder (already exists behind `USE_AGENT_BUILDER=true`) or (b) a new per-element placement API
   in the Add-in.  Recommend: run the new healing loop *on top of* the existing whole-recipe loop
   when `USE_AGENT_BUILDER=false`; the outer loop provides whole-recipe attempts and the inner
   healing loop handles per-element corrections within each attempt.

7. **Migration infrastructure ownership** — no migrator exists today; `JobStore` /
   `CorrectionsLogger` each issue `CREATE TABLE IF NOT EXISTS` in their `__init__`.  Introducing a
   migrator for just the healing agent leaves the older stores on the inline pattern.  Either
   migrate them too (cleaner, larger diff) or scope the migrator to new tables only (smaller diff,
   two patterns coexist).  Recommend scoping to new tables for Phase 2; retrofit older stores
   later.

8. **Handler Protocol granularity** — `validate_post_placement(element_id)` assumes element IDs are
   available, which only happens after Add-in v2 ships.  During the v1 transitional period
   handlers can fall back to category-scan validation (slower); this should be called out in the
   `ElementHandler` docstring rather than left implicit.
