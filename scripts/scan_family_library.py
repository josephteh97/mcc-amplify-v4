#!/usr/bin/env python3
"""
scan_family_library.py — RFA Family Library Scanner
====================================================

Walks a directory of Autodesk Revit family files (.rfa) and builds a
structured JSON index used by the MCP agent when selecting which family
to load for each element.

How it works
------------
1. Recursively scans ``--library-root`` (default: data/family_library)
2. For every .rfa file it tries, in order:
     a) Read a sidecar  <stem>.json  for rich hand-curated metadata
     b) Parse the filename itself using known naming conventions
     c) Infer the Revit category from the parent directory name
3. Writes the merged result to ``--output`` (default: data/family_library/index.json)

Filename patterns understood
----------------------------
Rectangular columns / beams:
  M_Concrete-Rectangular-Column.rfa
  UC_152x152x23.rfa          →  152 x 152 mm, 23 kg/m universal column
  RHS_200x100x6.rfa          →  200 x 100 mm rectangular hollow section

Circular columns / pipes:
  CHS_219x8.rfa              →  219 mm diameter, 8 mm thick
  M_Concrete-Round-Column.rfa
  Ø300.rfa  /  300dia.rfa    →  300 mm diameter

Doors / windows:
  M_Door-Single-Flush.rfa
  M_Fixed-900x1200.rfa       →  900 mm wide, 1200 mm high

Running on Windows (full library scan)
---------------------------------------
  python scripts/scan_family_library.py \\
      --library-root "C:\\ProgramData\\Autodesk\\RVT 2023\\Libraries" \\
      --output data/family_library/index.json

Running on Linux (hand-curated index only)
-------------------------------------------
  python scripts/scan_family_library.py \\
      --library-root data/family_library \\
      --output data/family_library/index.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Category inference: directory name → Revit OST category
# ---------------------------------------------------------------------------

_DIR_TO_CATEGORY: dict[str, dict] = {
    "structural_columns": {
        "ost": "OST_StructuralColumns",
        "tags": ["column", "structural"],
        "placement": "column",
    },
    "structural_framing": {
        "ost": "OST_StructuralFraming",
        "tags": ["beam", "structural", "framing"],
        "placement": "beam",
    },
    "walls": {
        "ost": "OST_Walls",
        "tags": ["wall"],
        "placement": "wall",
    },
    "doors": {
        "ost": "OST_Doors",
        "tags": ["door", "opening"],
        "placement": "hosted",
    },
    "windows": {
        "ost": "OST_Windows",
        "tags": ["window", "opening", "glazing"],
        "placement": "hosted",
    },
    "floors": {
        "ost": "OST_Floors",
        "tags": ["floor", "slab"],
        "placement": "floor",
    },
    "ceilings": {
        "ost": "OST_Ceilings",
        "tags": ["ceiling"],
        "placement": "ceiling",
    },
    "roofs": {
        "ost": "OST_Roofs",
        "tags": ["roof"],
        "placement": "roof",
    },
    "generic_models": {
        "ost": "OST_GenericModel",
        "tags": [],
        "placement": "point",
    },
}

# Also infer from filename keywords when the directory isn't categorised
_NAME_TO_CATEGORY: list[tuple[re.Pattern, str]] = [
    (re.compile(r"column|pillar|col_", re.I), "OST_StructuralColumns"),
    (re.compile(r"beam|framing|girder|rafter|joist", re.I), "OST_StructuralFraming"),
    (re.compile(r"\bdoor\b", re.I), "OST_Doors"),
    (re.compile(r"\bwindow\b|\bglaz", re.I), "OST_Windows"),
    (re.compile(r"\bfloor\b|\bslab\b", re.I), "OST_Floors"),
    (re.compile(r"\bwall\b", re.I), "OST_Walls"),
]

# ---------------------------------------------------------------------------
# Dimension extraction from filenames
# ---------------------------------------------------------------------------

# 300x400  or  300X400  →  (300.0, 400.0) rectangular
_RE_RECT = re.compile(r"(\d{2,4})[xX×](\d{2,4})")
# 219x8 as diameter×thickness for CHS
_RE_CHS  = re.compile(r"CHS[_\-]?(\d{2,4})[xX×](\d{1,3})", re.I)
# Ø300  or  300dia  or  300∅ →  300.0 circular
_RE_CIRC = re.compile(
    r"(?:Ø|⌀|∅|dia\.?\s*|phi\s*)(\d{2,4})"
    r"|(\d{2,4})\s*[Ø⌀∅]",
    re.I,
)
# Weight kg/m for steel sections (UC 152x152x23 → 23)
_RE_WEIGHT = re.compile(r"(\d+(?:\.\d+)?)\s*(?:kg|kN)", re.I)
# Height for doors/windows: 900x2100 (width=900, height=2100)
_RE_WH = re.compile(r"(\d{3,4})[xX×](\d{3,4})")

# ---------------------------------------------------------------------------
# Material inference
# ---------------------------------------------------------------------------

_MATERIAL_KEYWORDS = {
    "concrete":  re.compile(r"concrete|beton|RC", re.I),
    "steel":     re.compile(r"steel|UC|UB|CHS|RHS|SHS|HEA|HEB|IPE|UPE", re.I),
    "timber":    re.compile(r"timber|wood|glulam|CLT", re.I),
    "masonry":   re.compile(r"masonry|brick|block|CMU", re.I),
    "aluminium": re.compile(r"alumin", re.I),
}

# ---------------------------------------------------------------------------
# Shape inference
# ---------------------------------------------------------------------------

def _infer_shape(stem: str) -> str:
    s = stem.lower()
    if re.search(r"round|circ|chs|hollow\s*section|∅|dia", s):
        return "circular"
    if re.search(r"rect|square|rectangular", s):
        return "rectangular"
    if re.search(r"\buc\b|\bub\b|\bi\s*section|ipea?|hea?|heb?", s):
        return "I-section"
    if re.search(r"rhs|shs|hollow", s):
        return "rectangular_hollow"
    return "rectangular"   # safe default for columns


def _infer_material(stem: str) -> str:
    for mat, pat in _MATERIAL_KEYWORDS.items():
        if pat.search(stem):
            return mat
    return "concrete"      # structural default


# ---------------------------------------------------------------------------
# Type generation from filename
# ---------------------------------------------------------------------------

def _types_from_name(stem: str, shape: str) -> list[dict]:
    """
    Try to extract one or more FamilySymbol type entries from the filename stem.
    Returns [] if nothing can be parsed.
    """
    types: list[dict] = []

    # Circular (CHS / round column)
    circ = _RE_CIRC.search(stem)
    if circ:
        d_mm = float(circ.group(1) or circ.group(2))
        types.append({
            "type_name": f"Ø{int(d_mm)}",
            "is_circular": True,
            "diameter_mm": d_mm,
        })
        return types

    # CHS (circular hollow section — diameter × thickness)
    chs = _RE_CHS.search(stem)
    if chs:
        d_mm = float(chs.group(1))
        t_mm = float(chs.group(2))
        types.append({
            "type_name": f"CHS {int(d_mm)}×{int(t_mm)}",
            "is_circular": True,
            "diameter_mm": d_mm,
            "thickness_mm": t_mm,
        })
        return types

    # Rectangular: look for all WxD pairs in the stem
    for m in _RE_RECT.finditer(stem):
        w, d = float(m.group(1)), float(m.group(2))
        types.append({
            "type_name": f"{int(w)}×{int(d)}mm",
            "is_circular": False,
            "width_mm": w,
            "depth_mm": d,
        })

    return types


# ---------------------------------------------------------------------------
# Per-file record builder
# ---------------------------------------------------------------------------

def _infer_category(stem: str, parent_dir: str) -> dict:
    """Return category dict by directory name first, then filename keywords."""
    cat = _DIR_TO_CATEGORY.get(parent_dir.lower())
    if cat:
        return cat
    for pat, ost in _NAME_TO_CATEGORY:
        if pat.search(stem):
            return {"ost": ost, "tags": [], "placement": "point"}
    return {"ost": "OST_GenericModel", "tags": [], "placement": "point"}


def _build_record(rfa_path: Path, library_root: Path) -> dict:
    """
    Build a single family index record for *rfa_path*.
    Sidecar JSON (same stem, .json extension) is merged on top of inferred data.
    """
    stem       = rfa_path.stem
    parent_dir = rfa_path.parent.name
    rel_path   = str(rfa_path.relative_to(library_root)).replace("\\", "/")

    cat_info = _infer_category(stem, parent_dir)
    shape    = _infer_shape(stem)
    material = _infer_material(stem)
    types    = _types_from_name(stem, shape)

    # Absolute Windows path (valid when running on the Revit machine)
    windows_rfa_path = str(rfa_path).replace("/", "\\")

    record: dict = {
        "path":             rel_path,
        "windows_rfa_path": windows_rfa_path,
        "family_name":      stem,
        "category":         cat_info["ost"],
        "placement":        cat_info["placement"],
        "shape":            shape,
        "material":         material,
        "tags":             list(cat_info.get("tags", [])),
        "types":            types,
    }

    # Add material/shape tags
    if material not in record["tags"]:
        record["tags"].append(material)
    if shape not in record["tags"]:
        record["tags"].append(shape)

    # Merge sidecar JSON (wins over inferred values)
    sidecar = rfa_path.with_suffix(".json")
    if sidecar.exists():
        try:
            with open(sidecar, encoding="utf-8") as f:
                extra = json.load(f)
            # Merge: sidecar wins for everything except 'path'
            for k, v in extra.items():
                if k == "path":
                    continue
                if k == "tags" and isinstance(v, list):
                    record["tags"] = list({*record["tags"], *v})
                elif k == "types" and isinstance(v, list):
                    # Sidecar types replace inferred types
                    record["types"] = v
                else:
                    record[k] = v
            record["sidecar"] = True
        except Exception as e:
            log.warning(f"Could not read sidecar {sidecar.name}: {e}")

    return record


# ---------------------------------------------------------------------------
# Full library scan
# ---------------------------------------------------------------------------

def _build_record_from_sidecar(sidecar: Path, library_root: Path) -> dict | None:
    """
    Build an index record from a standalone sidecar .json (no local .rfa present).
    The .rfa exists on Windows; the sidecar is the sole source of truth on Linux.
    The record is marked  "rfa_local": false  so callers know the .rfa must be
    fetched from Windows at build time.
    """
    try:
        with open(sidecar, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        log.warning(f"Could not read sidecar {sidecar}: {e}")
        return None

    # family_name is required
    if "family_name" not in data:
        data["family_name"] = sidecar.stem

    # Infer missing fields from sidecar content / filename
    stem       = sidecar.stem
    parent_dir = sidecar.parent.name
    cat_info   = _infer_category(stem, parent_dir)

    record: dict = {
        "path":             str(sidecar.relative_to(library_root)).replace("\\", "/"),
        "windows_rfa_path": data.get("windows_rfa_path", ""),
        "rfa_local":        False,   # .rfa exists only on Windows
        "family_name":      data.get("family_name", stem),
        "category":         data.get("category", cat_info["ost"]),
        "placement":        data.get("placement", cat_info["placement"]),
        "shape":            data.get("shape", _infer_shape(stem)),
        "material":         data.get("material", _infer_material(stem)),
        "tags":             data.get("tags", list(cat_info.get("tags", []))),
        "types":            data.get("types", []),
        "sidecar":          True,
    }

    # Pass through any extra sidecar fields (description, parameter_map, notes, …)
    for k, v in data.items():
        if k not in record:
            record[k] = v

    return record


def scan(library_root: Path) -> list[dict]:
    records: list[dict] = []

    # --- Pass 1: local .rfa files (Windows or copied locally) ---------------
    rfa_files = sorted(library_root.rglob("*.rfa"))
    log.info(f"Found {len(rfa_files)} local .rfa file(s) under {library_root}")
    indexed_stems: set[str] = set()

    for rfa in rfa_files:
        try:
            rec = _build_record(rfa, library_root)
            records.append(rec)
            indexed_stems.add(rfa.stem.lower())
            log.debug(f"  {rec['path']}  →  {rec['category']}")
        except Exception as e:
            log.error(f"Skipped {rfa.name}: {e}")

    # --- Pass 2: sidecar-only .json files (no local .rfa counterpart) -------
    #    Skip files already covered by a local .rfa  (same stem, same directory).
    #    Also skip the output index itself.
    sidecar_files = sorted(library_root.rglob("*.json"))
    for sc in sidecar_files:
        if sc.name == "index.json":
            continue
        # Already covered by a real .rfa?
        if sc.stem.lower() in indexed_stems:
            continue
        rec = _build_record_from_sidecar(sc, library_root)
        if rec is None:
            continue
        records.append(rec)
        log.debug(f"  {rec['path']}  (sidecar-only) →  {rec['category']}")

    if not records:
        log.warning(
            "No families indexed. Add .rfa files or .json sidecar files to "
            f"{library_root}/<category>/ subdirectories."
        )

    # Sort: by category, then family_name
    records.sort(key=lambda r: (r["category"], r["family_name"]))
    return records


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Scan an RFA library directory and write index.json"
    )
    p.add_argument(
        "--library-root",
        default="data/family_library",
        help="Root directory to scan for .rfa files (default: data/family_library)",
    )
    p.add_argument(
        "--output",
        default="data/family_library/index.json",
        help="Output JSON path (default: data/family_library/index.json)",
    )
    p.add_argument(
        "--pretty",
        action="store_true",
        default=True,
        help="Pretty-print output JSON (default: true)",
    )
    p.add_argument("--debug", action="store_true", help="Verbose logging")
    args = p.parse_args(argv)

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    library_root = Path(args.library_root).resolve()
    output_path  = Path(args.output)

    if not library_root.exists():
        log.error(f"Library root does not exist: {library_root}")
        return 1

    families = scan(library_root)

    index = {
        "version":      2,
        "indexed_at":   datetime.now(tz=timezone.utc).isoformat(),
        "library_root": str(library_root),
        "total":        len(families),
        "families":     families,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    indent = 2 if args.pretty else None
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=indent, ensure_ascii=False)

    log.info(f"Wrote {len(families)} families → {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
