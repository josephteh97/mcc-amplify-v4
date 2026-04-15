"""
Revit BIM Tool implementations (P3).

These async functions are the canonical tool implementations shared by:
  - backend/mcp/server.py   — MCP protocol server (external agents / Claude Desktop)
  - backend/agents/revit_agent.py — embedded Claude tool-use loop (pipeline P5/P6)

All coordinates are in millimetres.  The Revit C# add-in converts to internal feet.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger

# RevitClient is imported lazily so this module can be imported without a live
# Revit server (e.g. during tests or when running only the MCP server binary).
_LIBRARY_INDEX        = Path(__file__).resolve().parents[2] / "data" / "family_library" / "index.json"
_REVIT_DOCS           = Path(__file__).resolve().parents[2] / "data" / "revit_docs"
_FALLBACK_FAMILY_FOLDER = r"C:\MyDocuments\3. Revit Family Files"

# Cached index — parsed once per process; set to None to force a reload.
_index_cache: dict | None = None


def _load_index() -> dict:
    global _index_cache
    if _index_cache is None and _LIBRARY_INDEX.exists():
        with open(_LIBRARY_INDEX) as f:
            _index_cache = json.load(f)
    return _index_cache or {}

# OST category → filename hint words used for fuzzy matching in the fallback folder
_CATEGORY_HINTS: dict = {
    "ost_structuralcolumns": ["column", "col"],
    "ost_structuralframing": ["beam", "framing", "uc", "ub", "rhs", "chs"],
    "ost_doors":             ["door"],
    "ost_windows":           ["window"],
    "ost_walls":             ["wall"],
    "ost_floors":            ["floor", "slab"],
}


# ── Lazy RevitClient singleton ────────────────────────────────────────────────

_client = None


def _get_client():
    global _client
    if _client is None:
        from services.revit_client import RevitClient
        _client = RevitClient()
    return _client


# ── Tool: search_family_library ───────────────────────────────────────────────

async def search_family_library(
    category: str | None = None,
    keyword: str | None = None,
) -> dict:
    """
    Search the local RFA family library index.

    Two-tier search:
    1. Primary — data/family_library/index.json (built by scan_family_library.py).
    2. Fallback — C:\\MyDocuments\\3. Revit Family Files on the Revit machine,
       queried via the /list-rfa endpoint and matched by filename.
    If neither tier finds anything, returns {"not_found": true} — the agent
    must skip that element rather than fabricating a family name.

    Parameters
    ----------
    category : str, optional
        Revit OST category filter, e.g. "OST_StructuralColumns", "OST_Doors".
    keyword : str, optional
        Free-text keyword matched against family_name, tags, and type_name fields.

    Returns
    -------
    dict  {"total": int, "families": [...]}
    Each family entry includes: family_name, category, windows_rfa_path, types[].
    """
    # ── Tier 1: primary index ──────────────────────────────────────────────────
    index = _load_index()
    if index:
        families: list[dict] = index.get("families", [])

        if category:
            families = [
                fam for fam in families
                if fam.get("category", "").lower() == category.lower()
            ]
        if keyword:
            kw = keyword.lower()
            families = [
                fam for fam in families
                if kw in fam.get("family_name", "").lower()
                or any(kw in tag for tag in fam.get("tags", []))
                or any(kw in (t.get("type_name") or "").lower() for t in fam.get("types", []))
            ]

        if families:
            # Rank by relevance before returning
            families = _score_and_rank(families, category=category, keyword=keyword)
            top = ", ".join(f.get("family_name", "?") for f in families[:3])
            logger.info(
                f"[search_family_library] cat={category!r} kw={keyword!r} "
                f"→ {len(families)} hits, top: {top}"
            )
            return {
                "total":      len(families),
                "indexed_at": index.get("indexed_at"),
                "families":   families[:15],
                "source":     "primary_index",
            }

    # ── Tier 2: fallback — scan user folder on Windows machine ────────────────
    logger.info(
        f"[search_family_library] Primary index miss "
        f"(category={category!r}, keyword={keyword!r}). "
        f"Trying fallback folder: {_FALLBACK_FAMILY_FOLDER}"
    )
    try:
        all_rfa = await _get_client().list_rfa_files(_FALLBACK_FAMILY_FOLDER)
    except Exception as exc:
        logger.warning(f"[search_family_library] Fallback scan failed: {exc}")
        all_rfa = []

    if all_rfa:
        matched = _match_rfa_by_name(all_rfa, category, keyword)
        if matched:
            top = ", ".join(m.get("family_name", "?") for m in matched[:3])
            logger.info(
                f"[search_family_library] fallback folder hit → "
                f"{len(matched)} matches, top: {top}"
            )
            return {
                "total":    len(matched),
                "families": matched,
                "source":   "user_folder",
                "note":     (
                    "Found in fallback folder. types[] will be populated after "
                    "revit_load_family is called."
                ),
            }

    # ── Tier 3: nothing found — agent must skip this element ──────────────────
    return {
        "total":     0,
        "families":  [],
        "not_found": True,
        "message": (
            f"No .rfa found for category={category!r}, keyword={keyword!r} "
            f"in the primary library index or fallback folder "
            f"({_FALLBACK_FAMILY_FOLDER}). "
            "Skip this element — do NOT fabricate a family name."
        ),
    }


def _score_and_rank(
    families: list[dict],
    category: str | None,
    keyword: str | None,
) -> list[dict]:
    """
    Rank *families* by relevance to the search query.

    Scoring criteria (higher is better):
      +4  exact category match
      +3  keyword in family_name (exact word, not substring)
      +2  keyword in any type_name
      +1  keyword in any tag
      +2  size tokens in keyword match a type's dimension fields
      -1  has_windows_rfa_path is False (sidecar-only — .rfa may not exist yet)

    Returns the list sorted descending by score.
    """
    import re as _re

    kw = keyword.lower().strip() if keyword else ""
    cat_lower = category.lower() if category else ""

    # Extract numeric tokens from keyword (e.g. "800x800" → [800, 800])
    kw_nums = [int(n) for n in _re.findall(r'\d+', kw)] if kw else []

    def _score(fam: dict) -> float:
        score = 0.0

        # Category match
        if cat_lower and fam.get("category", "").lower() == cat_lower:
            score += 4.0

        if kw:
            name_lower = fam.get("family_name", "").lower()
            # Exact word match in family name
            if _re.search(r'\b' + _re.escape(kw) + r'\b', name_lower):
                score += 3.0
            elif kw in name_lower:
                score += 1.5

            # Tag match
            tags = [t.lower() for t in fam.get("tags", [])]
            if any(kw in t for t in tags):
                score += 1.0

            # Type name match
            for t in fam.get("types", []):
                if kw in (t.get("type_name") or "").lower():
                    score += 2.0
                    break

        # Size proximity: reward families whose type dimensions are close to
        # the numbers in the keyword (e.g. "800x800" → prefer 800×800 over 800×600).
        #
        # Scoring strategy:
        #   - If keyword has 2 numbers (WxD), check both match; bonus for both close.
        #   - If keyword has 1 number (diameter / square), check closest dimension.
        if kw_nums:
            best_type_score = 0.0
            for t in fam.get("types", []):
                dim_vals = []
                for field in ("width_mm", "depth_mm", "diameter_mm", "b_mm", "h_mm"):
                    v = t.get(field)
                    if v is not None:
                        dim_vals.append(float(v))
                if not dim_vals:
                    continue

                if len(kw_nums) >= 2 and len(dim_vals) >= 2:
                    # Both keyword dimensions present — score how well BOTH match.
                    # Sort both lists so largest aligns with largest (WxD convention).
                    kn_sorted = sorted(kw_nums[:2], reverse=True)
                    dv_sorted = sorted(dim_vals[:2], reverse=True)
                    diff_sum  = sum(abs(kn - dv) for kn, dv in zip(kn_sorted, dv_sorted))
                    # +2 if both within 50 mm; degrading score up to 500 mm total error
                    type_score = max(0.0, 2.0 - diff_sum / 500.0)
                else:
                    # Single number — just find closest dimension
                    diffs = [abs(kw_nums[0] - d) for d in dim_vals]
                    type_score = max(0.0, 2.0 - min(diffs) / 500.0)

                if type_score > best_type_score:
                    best_type_score = type_score

            score += best_type_score

        # Prefer families where the Windows .rfa path is known
        if not fam.get("windows_rfa_path") and not fam.get("rfa_local", False):
            score -= 1.0

        return score

    return sorted(families, key=_score, reverse=True)


def _match_rfa_by_name(
    rfa_paths: list,
    category: str | None,
    keyword: str | None,
) -> list:
    """
    Filter Windows .rfa paths by category hints and keyword using filename matching.
    Returns family stubs compatible with the primary index format.
    """
    cat_hints: list = []
    if category:
        cat_hints = _CATEGORY_HINTS.get(category.lower(), [])

    kw = keyword.lower() if keyword else None

    results = []
    for path in rfa_paths:
        # Extract stem from Windows path without relying on pathlib (cross-platform)
        stem = path.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
        if stem.lower().endswith(".rfa"):
            stem = stem[:-4]
        name = stem.lower()

        hit_cat = (not cat_hints) or any(h in name for h in cat_hints)
        hit_kw  = (kw is None) or (kw in name)

        if hit_cat and hit_kw:
            results.append({
                "family_name":      stem,
                "category":         category or "",
                "windows_rfa_path": path,
                "types":            [],   # populated by revit_load_family
                "source":           "user_folder",
            })

    return results[:15]


# ── Tool: revit_new_session ───────────────────────────────────────────────────

async def revit_new_session() -> dict:
    """
    Open a new Revit document from the metric template.

    Returns
    -------
    dict  {"session_id": str, "levels": [...], "template": str, "message": str}
    """
    data = await _get_client().new_session()
    logger.info(f"[MCP] new_session → {data.get('session_id')}")
    return data


# ── Tool: revit_load_family ───────────────────────────────────────────────────

async def revit_load_family(session_id: str, windows_rfa_path: str) -> dict:
    """
    Load an .rfa family file into an open Revit session.

    Parameters
    ----------
    session_id       : str  Session token from revit_new_session.
    windows_rfa_path : str  Absolute Windows path to the .rfa file on the
                            Revit machine (e.g. C:\\Families\\Column_Concrete.rfa).
                            Use windows_rfa_path from search_family_library results.

    Returns
    -------
    dict  {"family_name": str, "already_loaded": bool, "types": [...]}
    """
    data = await _get_client().load_family(session_id, windows_rfa_path)
    logger.info(f"[MCP] load_family '{data.get('family_name')}' into {session_id}")
    return data


# ── Tool: revit_list_families ─────────────────────────────────────────────────

async def revit_list_families(session_id: str) -> dict:
    """
    List all families currently loaded in an open session document.

    Returns
    -------
    dict  {"families": [{family_name, category, types: [...]}, ...]}
    """
    return await _get_client().list_families(session_id)


# ── Tool: revit_place_instance ────────────────────────────────────────────────

async def revit_place_instance(
    session_id:  str,
    family_name: str,
    type_name:   str,
    x_mm:        float,
    y_mm:        float,
    z_mm:        float = 0.0,
    level:       str   = "Level 0",
    top_level:   str | None = None,
    parameters:  dict | None = None,
) -> dict:
    """
    Place one FamilyInstance in the open Revit session.

    Parameters
    ----------
    session_id  : str    Session token.
    family_name : str    Family name exactly as returned by revit_list_families.
    type_name   : str    Type name exactly as listed under the family.
    x_mm, y_mm  : float  Insertion point in millimetres (world coordinates).
    z_mm        : float  Elevation offset from base level (default 0).
    level       : str    Base level name (default "Level 0").
    top_level   : str    Top level for structural columns (e.g. "Level 1").
    parameters  : dict   Optional {param_name: value_mm} pairs to set on placement.

    Returns
    -------
    dict  {"element_id": str, "placed": {...}}
    """
    data = await _get_client().place_instance(
        session_id, family_name, type_name,
        x_mm, y_mm, z_mm, level, top_level, parameters,
    )
    logger.debug(
        f"[MCP] placed {family_name}::{type_name} @ ({x_mm:.0f},{y_mm:.0f}) "
        f"→ elem_id={data.get('element_id')}"
    )
    return data


# ── Tool: revit_set_parameter ─────────────────────────────────────────────────

async def revit_set_parameter(
    session_id:     str,
    element_id:     str,
    parameter_name: str,
    value:          Any,
    value_type:     str = "mm",
) -> dict:
    """
    Set a parameter on an already-placed element.

    Parameters
    ----------
    session_id     : str   Session token.
    element_id     : str   Element ID from revit_place_instance response.
    parameter_name : str   Revit parameter name (e.g. "b", "h", "Mark").
    value          : any   The value to set.
    value_type     : str   Unit hint — "mm" (auto-converts to internal feet),
                           "raw" (pass through), "string", "int", "id".

    Returns
    -------
    dict  {"ok": true, "element_id": str, "parameter_name": str}
    """
    return await _get_client().set_parameter(
        session_id, element_id, parameter_name, value, value_type
    )


# ── Tool: revit_get_parameters ────────────────────────────────────────────────

async def revit_get_parameters(session_id: str, element_id: str) -> dict:
    """
    List all parameters of a placed element.

    Call this BEFORE revit_set_parameter to discover the correct parameter
    names for a given element.  Editable parameters are listed first.

    Parameters
    ----------
    session_id : str  Session token.
    element_id : str  Element ID from revit_place_instance response.

    Returns
    -------
    dict  {"element_id": str, "count": int,
           "parameters": [{name, storage_type, is_read_only, value_mm?}]}
    """
    return await _get_client().get_element_parameters(session_id, element_id)


# ── Tool: revit_wall_join_all ─────────────────────────────────────────────────

async def revit_wall_join_all(session_id: str) -> dict:
    """
    Enable wall joins at both ends of every wall in the session.

    Call this once after all walls have been placed.  It fixes display
    gaps and incorrect intersections at T-junctions and corners.

    Parameters
    ----------
    session_id : str  Session token.

    Returns
    -------
    dict  {"ok": bool, "walls_total": int, "walls_joined": int}
    """
    return await _get_client().wall_join_all(session_id)


# ── Tool: revit_get_state ─────────────────────────────────────────────────────

async def revit_get_state(session_id: str) -> dict:
    """
    Query the current state of an open Revit session.

    Returns
    -------
    dict  {"levels": [...], "loaded_families": [...], "placed_elements": [...]}
    Useful to verify placements or resume an interrupted session.
    """
    return await _get_client().get_session_state(session_id)


# ── Tool: revit_export_session ────────────────────────────────────────────────

async def revit_export_session(session_id: str, job_id: str) -> dict:
    """
    Save the Revit document as .rvt and close the session.

    Parameters
    ----------
    session_id : str  Session token.
    job_id     : str  Pipeline job ID — output saved to data/models/rvt/{job_id}.rvt.

    Returns
    -------
    dict  {"rvt_path": str, "session_still_open": bool}
    """
    rvt_path, still_open = await _get_client().export_session(session_id, job_id)
    logger.info(f"[MCP] export_session {session_id} → {rvt_path}")
    return {"rvt_path": rvt_path, "session_still_open": still_open}


# ── Tool registry (used by MCP server and agent loop) ────────────────────────

#: Maps tool name → (callable, input_schema)
TOOL_REGISTRY: dict[str, tuple] = {
    "search_family_library": (
        search_family_library,
        {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": (
                        "Revit OST category filter. Common values: "
                        "OST_StructuralColumns, OST_Walls, OST_Doors, "
                        "OST_Windows, OST_StructuralFraming, OST_Floors."
                    ),
                },
                "keyword": {
                    "type": "string",
                    "description": "Free-text search across family name, tags, and type names.",
                },
            },
        },
    ),
    "revit_new_session": (
        revit_new_session,
        {"type": "object", "properties": {}},
    ),
    "revit_load_family": (
        revit_load_family,
        {
            "type": "object",
            "properties": {
                "session_id":       {"type": "string"},
                "windows_rfa_path": {
                    "type": "string",
                    "description": "Absolute Windows path to the .rfa file (from library search).",
                },
            },
            "required": ["session_id", "windows_rfa_path"],
        },
    ),
    "revit_list_families": (
        revit_list_families,
        {
            "type": "object",
            "properties": {"session_id": {"type": "string"}},
            "required": ["session_id"],
        },
    ),
    "revit_place_instance": (
        revit_place_instance,
        {
            "type": "object",
            "properties": {
                "session_id":  {"type": "string"},
                "family_name": {"type": "string"},
                "type_name":   {"type": "string"},
                "x_mm":        {"type": "number", "description": "X coordinate in millimetres."},
                "y_mm":        {"type": "number", "description": "Y coordinate in millimetres."},
                "z_mm":        {"type": "number", "description": "Z offset from base level in mm (default 0)."},
                "level":       {"type": "string", "description": "Base level name (default: Level 0)."},
                "top_level":   {"type": "string", "description": "Top level for columns (e.g. Level 1)."},
                "parameters":  {
                    "type": "object",
                    "description": "Optional {param_name: value_mm} pairs set on placement.",
                },
            },
            "required": ["session_id", "family_name", "type_name", "x_mm", "y_mm"],
        },
    ),
    "revit_set_parameter": (
        revit_set_parameter,
        {
            "type": "object",
            "properties": {
                "session_id":     {"type": "string"},
                "element_id":     {"type": "string"},
                "parameter_name": {"type": "string"},
                "value":          {"description": "Value to set."},
                "value_type":     {
                    "type": "string",
                    "enum": ["mm", "raw", "string", "int", "id"],
                    "description": "Unit conversion hint (default: mm).",
                },
            },
            "required": ["session_id", "element_id", "parameter_name", "value"],
        },
    ),
    "revit_get_parameters": (
        revit_get_parameters,
        {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "element_id": {
                    "type": "string",
                    "description": "Element ID from revit_place_instance response.",
                },
            },
            "required": ["session_id", "element_id"],
        },
    ),
    "revit_wall_join_all": (
        revit_wall_join_all,
        {
            "type": "object",
            "properties": {"session_id": {"type": "string"}},
            "required": ["session_id"],
        },
    ),
    "revit_get_state": (
        revit_get_state,
        {
            "type": "object",
            "properties": {"session_id": {"type": "string"}},
            "required": ["session_id"],
        },
    ),
    "revit_export_session": (
        revit_export_session,
        {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "job_id":     {"type": "string", "description": "Pipeline job ID for output naming."},
            },
            "required": ["session_id", "job_id"],
        },
    ),
}


async def call_tool(name: str, arguments: dict) -> Any:
    """Dispatch a tool call by name.  Raises KeyError on unknown tool."""
    fn, _ = TOOL_REGISTRY[name]
    return await fn(**arguments)
