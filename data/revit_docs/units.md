# Revit 2023 — Units & Measurement Reference

## Internal unit system
Revit 2023 stores all lengths internally in **decimal feet** (not mm, not inches).

The add-in conversion constant:
```
MM_TO_FEET = 1.0 / 304.8
feet_value  = mm_value * MM_TO_FEET
```

**All tool coordinates (`x_mm`, `y_mm`, `z_mm`) are in millimetres.**
The C# add-in performs the mm → internal feet conversion automatically.
Never pass raw feet to the tools.

## Common parameter types (`value_type` in `revit_set_parameter`)

| value_type | When to use | Example |
|------------|-------------|---------|
| `"mm"`     | Any length — auto-converted to internal feet | `{"value": 800, "value_type": "mm"}` |
| `"string"` | Text parameters (Mark, Comments) | `{"value": "C1", "value_type": "string"}` |
| `"int"`    | Integer parameters (count, phase) | `{"value": 2, "value_type": "int"}` |
| `"raw"`    | Unitless or already-converted values | rare |
| `"id"`     | ElementId parameters | `{"value": "123456", "value_type": "id"}` |

## Structural column parameters

| Parameter | Description | Typical value (mm) |
|-----------|-------------|-------------------|
| `b`       | Column width (rectangular) | 400–800 |
| `h`       | Column depth (rectangular) | 400–800 |
| `d`       | Column diameter (circular) | 300–600 |
| `Mark`    | Column type mark (string) | "C1", "C20" |

## Door parameters

| Parameter | Description | Typical (mm) |
|-----------|-------------|--------------|
| `Width`   | Door leaf width | 900–1200 |
| `Height`  | Door leaf height | 2100 |

## Window parameters

| Parameter | Description | Typical (mm) |
|-----------|-------------|--------------|
| `Width`   | Window width | 600–1800 |
| `Height`  | Window height | 900–1500 |
| `Sill Height` | Sill above floor level | 900 |

## Level elevations
Default levels created by the add-in:
- **Level 0** (Ground Floor) — elevation 0 mm
- **Level 1** (First Floor) — elevation 3000 mm

All elements must reference a valid level name.
Structural columns require both `level` (base) and `top_level` (top).

## Coordinate origin
The origin (0, 0, 0) in Revit world space corresponds to the pixel origin of the
detected structural grid.  All `x_mm` / `y_mm` values are positive offsets from
the grid's lower-left intersection.
