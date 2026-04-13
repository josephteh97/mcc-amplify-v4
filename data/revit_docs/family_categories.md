# Revit 2023 — Family Categories (OST Codes)

Use these category strings in `search_family_library(category=...)`.

## Structural

| OST Code | Description | Typical family file pattern |
|----------|-------------|----------------------------|
| `OST_StructuralColumns` | Structural columns (concrete, steel, timber) | `*Column*`, `*Col*` |
| `OST_StructuralFraming` | Beams, girders, braces | `*Beam*` |
| `OST_StructuralFoundation` | Footings, pile caps | `*Footing*`, `*Foundation*` |

## Architectural

| OST Code | Description | Typical family file pattern |
|----------|-------------|----------------------------|
| `OST_Doors` | Door families | `*Door*` |
| `OST_Windows` | Window families | `*Window*`, `*Window*` |
| `OST_Furniture` | Furniture | `*Desk*`, `*Chair*` |
| `OST_Casework` | Built-in casework, cabinets | `*Cabinet*`, `*Counter*` |
| `OST_Stairs` | Stair elements | `*Stair*` |
| `OST_GenericModel` | Generic model families | varies |

## MEP

| OST Code | Description |
|----------|-------------|
| `OST_MechanicalEquipment` | HVAC equipment |
| `OST_ElectricalEquipment` | Electrical panels |
| `OST_PipingSystem` | Piping |

## System families (NOT loadable — controlled via transaction JSON)
These are system families built into every Revit project.
Do not try to search for or load them via `revit_load_family`.

| Type | How to create |
|------|---------------|
| Walls | Defined in the `walls` array of the geometry transaction JSON |
| Floors | Defined in the `floors` array |
| Ceilings | Defined in the `ceilings` array |
| Roofs | Not yet implemented |
| Levels | Always created as Level 0 (0 mm) and Level 1 (3000 mm) |
| Grids | Defined in the `grids` array |

## Family selection strategy

When placing elements from a floor plan:
1. Call `search_family_library(category="OST_StructuralColumns")` to find columns.
2. Pick a type whose dimensions most closely match the detected column size.
3. If no exact match, load the nearest family and use `revit_set_parameter` to
   adjust `b`/`h`/`d` after placement.
4. Prefer families with `"recommended": true` in the library index.
