# Revit 2023 — Element Placement Guide

## Recommended placement order

Build the model in this order to avoid Revit constraint errors:

```
1. Grids          (already in batch transaction — verify via get_state)
2. Levels         (already created by new_session)
3. Structural Columns
4. Structural Framing (beams) if present
5. Walls          (use batch transaction JSON — system family)
6. Doors          (must be hosted inside a wall — place after walls)
7. Windows        (must be hosted inside a wall — place after walls)
8. Floors / Slabs (use batch transaction JSON)
9. Rooms / Spaces (analytical only — no geometry)
```

## Structural columns

Columns require both a base level and a top level:

```json
{
  "family_name": "Concrete-Rectangular-Column",
  "type_name":   "300x300mm",
  "x_mm": 6000, // follow/align to the first number of type name
  "y_mm": 3000, // follow/align to the second number of type name
  "level": "Level 0",
  "top_level": "Level 1"
}
```

Place a column at each detected grid intersection.
Use the `center_mm` field from the geometry transaction if available.

## Doors and windows

Doors and windows are **wall-hosted** in Revit.  The C# add-in will
automatically find the nearest wall to host each element.  Provide the
door/window centre point as `x_mm`, `y_mm`.

```json
{
  "family_name": "Single-Flush",
  "type_name":   "900 x 2100mm",
  "x_mm": 3500,
  "y_mm": 0,
  "level": "Level 0"
}
```

If placement fails with a "no host wall found" error, the coordinates
may be slightly off the wall centre-line.  Adjust by ±50 mm.

## Coordinate system

All coordinates come from the `geometry_transaction.json` produced by
stage 6 of the pipeline.  They are already in millimetres relative to
the structural grid origin.

Key fields in the transaction JSON:
- `columns[].center` or `columns[].location` → {x, y, z} in mm
- `walls[].start_point` / `.end_point` → wall endpoints in mm
- `doors[].location` → insertion point in mm
- `windows[].location` → insertion point in mm

## Session workflow

```
new_session()
  → load_family() for each unique column/door/window family needed
  → place_instance() for each element (columns first, then doors/windows)
  → [optional] set_parameter() for size corrections
  → get_state()  to verify element count
  → export_session() to save .rvt and close
```

## Handling missing families

If `search_family_library` returns no results:
1. Use a generic substitute (e.g., "Generic Column 400x400" always exists).
2. After placement, call `set_parameter` to adjust the dimensions.
3. Log the missing family name so the library can be expanded.

Do NOT abort the session — always place a substitute and continue.

## Error recovery

- **"Family not found"**: The family was not loaded.  Call `load_family` first.
- **"No host wall"**: Adjust door/window x_mm/y_mm by ±50 mm.
- **"Level not found"**: Use "Level 0" or "Level 1" — always present.
- **Timeout**: The Revit thread may be busy.  Call `get_state` to check, then retry.
