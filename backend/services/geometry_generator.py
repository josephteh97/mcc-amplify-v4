"""
Semantic 3D Generation
Converts 2D detected elements to Semantic 3D parameters for Revit Solid Modeling.

Coordinate system
-----------------
All pixel coordinates are converted to real-world millimetres using the
structural grid detected by GridDetector.  There is NO dependency on any
scale text printed on the floor plan — only the grid line positions and their
dimension annotations matter.

Default levels
--------------
When the number of storeys is unknown (the common case for a single floor plan),
Level 0  (Ground Floor, elevation = 0 mm)  and
Level 1  (First Floor,  elevation = 3000 mm)
are always created.  Grid lines are placed at Level 0.
"""

import os

import numpy as np
from typing import Dict, List, Tuple, Optional
from loguru import logger

from backend.services.grid_detector import GridDetector
from backend.services.intelligence.admittance import REJECT
from backend.services.intelligence.slab_thickness_parser import resolve_code_thickness


# Default floor-to-floor height when no storey height annotation is present.
DEFAULT_STOREY_HEIGHT_MM = 5000              # actual case: refer to arch drawing

# Standard SQUARE column section sizes (mm).
# Used as snap targets when no PDF annotation is found for square sections —
# structural engineers specify these standard sizes, so snapping eliminates
# family proliferation caused by YOLO bbox jitter (e.g. 298 → 300 mm).
STANDARD_SQUARE_COLUMN_SIZES = [200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000, 1200]

# Standard CIRCULAR column diameters (mm).
STANDARD_CIRCULAR_COLUMN_DIAMETERS = [300, 350, 400, 450, 500, 600, 700, 800, 900, 1000, 1200]

# Columns with aspect ratio within this fractional threshold are treated as square.
# 20% absorbs YOLO bbox noise (e.g. 720×880 reported for a true 800×800).
SQUARE_ASPECT_THRESHOLD = 0.20  # 20%


def _nearest(v: float, candidates: List[float]) -> float:
    """Return the element of *candidates* closest to *v* by absolute difference."""
    return float(min(candidates, key=lambda c: abs(c - v)))


def _point_in_polygon(x: float, y: float, polygon: List[Dict]) -> bool:
    """Ray-casting PIP. *polygon* is [{"x": mm, "y": mm}, ...]."""
    n = len(polygon)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]["x"], polygon[i]["y"]
        xj, yj = polygon[j]["x"], polygon[j]["y"]
        if ((yi > y) != (yj > y)) and x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi:
            inside = not inside
        j = i
    return inside


def level_elevation(levels: List[Dict], name: str, default: float = 0.0) -> float:
    """Return the elevation (mm) of the level named *name*, or *default* if not found."""
    return next(
        (float(l["elevation"]) for l in levels if l.get("name") == name),
        float(default),
    )


def normalize_column_dimensions(
    width_mm: float,
    depth_mm: float,
    column_shape: str = "rectangular",
    annotated_dimensions: Optional[Tuple[float, float]] = None,
) -> Tuple[float, float, str]:
    """Normalize column dimensions with shape-specific validation.

    Returns:
        (width_mm, depth_mm, family_suffix)
    """
    # print(f"[NORM] w={width_mm:.0f} d={depth_mm:.0f} shape={column_shape} ann={annotated_dimensions}")

    # ── Shape-specific validation of annotations ─────────────────────────────
    if annotated_dimensions is not None:
        ann_w, ann_d = annotated_dimensions

        if column_shape == "circular":
            diameter = max(ann_w, ann_d)
            nearest = _nearest(diameter, STANDARD_CIRCULAR_COLUMN_DIAMETERS)
            annotated_dimensions = (float(nearest), float(nearest))
            if nearest != diameter:
                logger.debug(f"[CIRCULAR] diameter={diameter} → snapped to {nearest}")

        elif column_shape == "square":
            if ann_w != ann_d:
                avg = (ann_w + ann_d) / 2
                nearest = _nearest(avg, STANDARD_SQUARE_COLUMN_SIZES)
                annotated_dimensions = (float(nearest), float(nearest))
                logger.debug(f"[SQUARE] {ann_w}x{ann_d} mismatch → corrected to {nearest}x{nearest}")
            else:
                nearest = _nearest(ann_w, STANDARD_SQUARE_COLUMN_SIZES)
                annotated_dimensions = (float(nearest), float(nearest))

        elif column_shape == "rectangular":
            diff = abs(ann_w - ann_d)
            if diff == 0:
                nearest = _nearest(ann_w, STANDARD_SQUARE_COLUMN_SIZES)
                annotated_dimensions = (float(nearest), float(nearest))
                if nearest != ann_w:
                    logger.debug(f"[RECT→SQUARE] {ann_w}x{ann_d} → {nearest}x{nearest}")
            elif diff <= 100:
                avg = (ann_w + ann_d) / 2
                nearest = _nearest(avg, STANDARD_SQUARE_COLUMN_SIZES)
                annotated_dimensions = (float(nearest), float(nearest))
                logger.debug(f"[RECT→SQUARE] {ann_w}x{ann_d} diff={diff}mm → {nearest}x{nearest}")
            else:
                w_rounded = round(ann_w / 50) * 50
                d_rounded = round(ann_d / 50) * 50
                annotated_dimensions = (float(w_rounded), float(d_rounded))
                if w_rounded != ann_w or d_rounded != ann_d:
                    logger.debug(f"[RECTANGULAR] {ann_w}x{ann_d} → {w_rounded}x{d_rounded}")

    # Priority 1: validated annotated dimensions
    if annotated_dimensions is not None:
        ann_w, ann_d = annotated_dimensions
        if ann_w > 0 and ann_d > 0:
            if abs(ann_w - ann_d) < 10:
                size = max(ann_w, ann_d)
                return float(size), float(size), f"RECT{int(size)}x{int(size)}mm"
            return float(ann_w), float(ann_d), f"RECT{int(min(ann_w, ann_d))}x{int(max(ann_w, ann_d))}mm"

    # Priority 2: circular
    if column_shape == "circular":
        diameter = max(width_mm, depth_mm)
        nearest = _nearest(diameter, STANDARD_CIRCULAR_COLUMN_DIAMETERS)
        return float(nearest), float(nearest), f"CIRC{int(nearest)}"

    # Priority 3: square by aspect ratio
    if width_mm > 0 and depth_mm > 0:
        aspect = min(width_mm, depth_mm) / max(width_mm, depth_mm)
    else:
        aspect = 1.0
    if column_shape == "square" or (1.0 - aspect) <= SQUARE_ASPECT_THRESHOLD:
        avg = (width_mm + depth_mm) / 2
        nearest = _nearest(avg, STANDARD_SQUARE_COLUMN_SIZES)
        return float(nearest), float(nearest), f"RECT{int(nearest)}x{int(nearest)}mm"

    # Priority 4: rectangular — round to nearest 50 mm
    w_rounded = round(width_mm / 50) * 50
    d_rounded = round(depth_mm / 50) * 50
    return (
        float(w_rounded),
        float(d_rounded),
        f"RECT{int(min(w_rounded, d_rounded))}x{int(max(w_rounded, d_rounded))}mm",
    )


class GeometryGenerator:
    """Build Semantic 3D parameters for native Revit solid objects."""

    def __init__(self):
        self.grid_detector = GridDetector()

        # Default architectural standards (mm) — overridable via apply_profile()
        self.default_wall_height      = 2800
        self.default_wall_thickness   = 200
        self.default_door_height      = 2100
        self.default_window_height    = 1500
        self.default_sill_height      = 900
        self.default_floor_thickness  = 200
        self._storey_height_override  = None   # set by apply_profile()
        # Minimum structural column section (mm). 200 mm is the Revit extrusion floor.
        # Stored as an instance attribute so apply_profile() can raise it per-job
        # without mutating the class and affecting other GeometryGenerator instances.
        self._min_column_mm           = 800.0
        # Default beam cross-section dimensions — 800×800 matches column default.
        # We do NOT derive beam section from the YOLO bbox short-side: the bbox's
        # on-plan short side is a drafting-line thickness, not a structural section,
        # and trusting it produces hallucinated type names like "1050x800mm".
        self.default_beam_width       = float(os.getenv("DEFAULT_BEAM_WIDTH_MM", "800"))
        self.default_beam_depth       = float(os.getenv("DEFAULT_BEAM_DEPTH_MM", "800"))

    def apply_profile(self, profile: dict) -> None:
        """
        Override per-instance dimension defaults from a project profile dict.
        Called by the orchestrator once per run if data/project_profile.json exists.

        Recognised keys (all in mm, all optional):
            typical_wall_height_mm      → default_wall_height
            typical_wall_thickness_mm   → default_wall_thickness
            typical_sill_height_mm      → default_sill_height
            floor_to_floor_height_mm    → Level 1 elevation (storey height)
            typical_column_size_mm      → _MIN_COLUMN_MM clamp floor
        """
        _map = {
            "typical_wall_height_mm":    "default_wall_height",
            "typical_wall_thickness_mm": "default_wall_thickness",
            "typical_sill_height_mm":    "default_sill_height",
        }
        for profile_key, attr in _map.items():
            v = profile.get(profile_key)
            if v and isinstance(v, (int, float)) and v > 0:
                setattr(self, attr, float(v))

        fth = profile.get("floor_to_floor_height_mm")
        if fth and isinstance(fth, (int, float)) and 2000 <= fth <= 8000:
            self._storey_height_override = float(fth)

        col_sz = profile.get("typical_column_size_mm")
        if col_sz and isinstance(col_sz, (int, float)) and col_sz > self._min_column_mm:
            # Raise the minimum clamp if the project profile indicates larger columns.
            # Never lower it below the hard floor of 200 mm (Revit extrusion limit).
            self._min_column_mm = float(col_sz)
            logger.info(f"apply_profile: _min_column_mm raised to {self._min_column_mm:.0f} mm")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def build(
        self,
        enriched_data: Dict,
        grid_info: Dict,
        zone_labels_mm: Optional[List[Tuple[str, float, float]]] = None,
        slab_legend: Optional[Dict[str, float]] = None,
    ) -> Dict:
        """
        Build Semantic 3D parameters from enriched analysis data.

        Args:
            enriched_data  : data from Claude/Gemini analysis merged with YOLO
            grid_info      : grid_info dict from GridDetector.detect()
            zone_labels_mm : optional [(code, x_mm, y_mm)] zone labels placed
                             on the plan (not in a NOTES block). Used to look
                             up per-slab thickness.
            slab_legend    : optional {CODE: thickness_mm} from the NOTES block.

        Returns:
            Dict matching the RevitTransaction JSON schema consumed by
            ModelBuilder.cs on the Windows Revit server.
        """
        logger.info("Generating Semantic 3D parameters for Revit (grid-based)…")

        levels = self._build_default_levels(enriched_data)
        level0_elev = level_elevation(levels, "Level 0", 0.0)

        # Build slabs first so structural_framing can resolve per-beam Z by
        # testing the beam midpoint against each slab's polygon.
        slabs_built = self._build_slab_parameters(
            enriched_data.get("slabs", []), grid_info,
            zone_labels_mm=zone_labels_mm, slab_legend=slab_legend,
        )

        geometry = {
            "levels":   levels,
            "grids":    self._build_grid_commands(grid_info),

            # ── Structural elements ────────────────────────────────────
            "columns":            self._build_column_parameters(
                                      enriched_data.get("columns", []), grid_info),
            "walls":              self._build_wall_parameters(
                                      enriched_data.get("walls", []), grid_info),
            "structural_framing": self._build_structural_framing_parameters(
                                      enriched_data.get("structural_framing", []),
                                      grid_info, level0_elev,
                                      slab_regions=slabs_built),
            "stairs":             self._build_stairs_parameters(
                                      enriched_data.get("stairs", [])),
            "lifts":              self._build_lift_parameters(
                                      enriched_data.get("lifts", [])),
            "slabs":  slabs_built,

            "metadata": enriched_data.get("metadata", {}),
        }

        logger.info(
            f"Generated: {len(geometry['columns'])} columns, "
            f"{len(geometry['structural_framing'])} framing, "
            f"{len(geometry['walls'])} walls, "
            f"{len(geometry['stairs'])} stairs, "
            f"{len(geometry['lifts'])} lifts, "
            f"{len(geometry['slabs'])} slabs, "
            f"{len(geometry['grids'])} grid lines, "
            f"levels: {[l['name'] for l in geometry['levels']]}"
        )
        return geometry

    # ------------------------------------------------------------------
    # Levels
    # ------------------------------------------------------------------

    def _build_default_levels(self, enriched_data: Dict) -> List[Dict]:
        """
        Always create Level 0 (Ground Floor) and Level 1 (First Floor).
        Priority for storey height:
          1. project profile override (apply_profile was called)
          2. semantic analysis (room ceiling_height)
          3. DEFAULT_STOREY_HEIGHT_MM (3000 mm)
        """
        # 1. Project profile override
        if self._storey_height_override:
            return [
                {"name": "Level 0", "elevation": 0},
                {"name": "Level 1", "elevation": int(self._storey_height_override)},
            ]

        # 2. Try to read storey height from semantic metadata
        ceiling_h = DEFAULT_STOREY_HEIGHT_MM
        for room in enriched_data.get("rooms", []):
            h = room.get("ceiling_height") or room.get("target_height")
            if h and isinstance(h, (int, float)) and 2000 <= h <= 6000:
                ceiling_h = int(h)
                break

        return [
            {"name": "Level 0", "elevation": 0},
            {"name": "Level 1", "elevation": ceiling_h},
        ]

    # ------------------------------------------------------------------
    # Structural grid
    # ------------------------------------------------------------------

    def _build_grid_commands(self, grid_info: Dict) -> List[Dict]:
        """
        Build Revit Grid creation commands from the detected grid.

        Vertical grid lines (constant X) use the numeric labels (1, 2, 3…).
        Horizontal grid lines (constant Y) use alphabetic labels (A, B, C…).
        Both extend the full width/height of the drawing.
        """
        grids = []

        x_lines   = grid_info.get("x_lines_px", [])
        y_lines   = grid_info.get("y_lines_px", [])
        x_labels  = grid_info.get("x_labels", [])
        y_labels  = grid_info.get("y_labels", [])
        x_sp      = grid_info.get("x_spacings_mm", [])
        y_sp      = grid_info.get("y_spacings_mm", [])

        # Total drawing extents in mm
        if x_sp:
            total_x_mm = sum(x_sp)
        else:
            total_x_mm = DEFAULT_STOREY_HEIGHT_MM * 4
            logger.warning(
                "Grid X spacings are empty — cannot determine real drawing width. "
                f"Falling back to {total_x_mm:.0f} mm. "
                "All grid lines will stack at x=0; geometry will be misaligned. "
                "Check that dimension annotations exist between grid lines."
            )
        if y_sp:
            total_y_mm = sum(y_sp)
        else:
            total_y_mm = DEFAULT_STOREY_HEIGHT_MM * 4
            logger.warning(
                "Grid Y spacings are empty — cannot determine real drawing height. "
                f"Falling back to {total_y_mm:.0f} mm. "
                "All grid lines will stack at y=0; geometry will be misaligned. "
                "Check that dimension annotations exist between grid lines."
            )

        # Vertical grid lines — run full height (Y direction)
        for i, _ in enumerate(x_lines):
            x_mm  = sum(x_sp[:i])
            label = x_labels[i] if i < len(x_labels) else str(i + 1)
            grids.append({
                "name":  label,
                "start": {"x": x_mm, "y": -10000,        "z": 0.0},
                "end":   {"x": x_mm, "y": total_y_mm + 10000, "z": 0.0},
            })

        # Horizontal grid lines — run full width (X direction)
        # y_lines_px[0] is the topmost line in the image (smallest pixel Y).
        # Image Y increases downward; Revit Y increases upward.  The topmost
        # architectural line must therefore receive the LARGEST Revit Y, so the
        # accumulation order is always flipped regardless of page rotation.
        for i, _ in enumerate(y_lines):
            y_mm = total_y_mm - sum(y_sp[:i])
            label = y_labels[i] if i < len(y_labels) else chr(65 + i)
            grids.append({
                "name":  label,
                "start": {"x": -10000,     "y": y_mm, "z": 0.0},
                "end":   {"x": total_x_mm, "y": y_mm, "z": 0.0},
            })

        return grids

    # ------------------------------------------------------------------
    # Element placement (all coordinates via grid)
    # ------------------------------------------------------------------

    def _px_to_world(self, px: float, py: float, grid_info: Dict) -> Tuple[float, float]:
        """Convert pixel coords to real-world mm with Y-axis inversion.

        pixel_to_world returns raw_y_mm increasing downward (image origin = top-left).
        Revit's Y axis increases upward, so flip: revit_y = total_y_mm - raw_y_mm.
        """
        x_mm, raw_y_mm = self.grid_detector.pixel_to_world(px, py, grid_info)
        total_y_mm = sum(grid_info.get("y_spacings_mm", []))
        return x_mm, total_y_mm - raw_y_mm

    def pt_to_world(
        self, x_pt: float, y_pt: float, grid_info: Dict, image_dpi: float,
    ) -> Tuple[float, float]:
        """Convert a PDF-point coordinate to Revit world mm (Y-flipped)."""
        s = image_dpi / 72.0
        return self._px_to_world(x_pt * s, y_pt * s, grid_info)

    def _snap_to_nearest_grid(
        self, x_mm: float, y_mm: float, grid_info: Dict
    ) -> Tuple[float, float]:
        """
        Snap world coordinates to the nearest structural grid intersection.

        YOLO detects column symbols by their visual bbox, which often includes
        nearby label text.  For a /Rotate 90 PDF this creates a systematic
        sub-bay offset between the detected centre and the true grid intersection.
        Snapping to the nearest intersection is safe for structural floor plans
        where every column sits at a grid crossing.

        A half-bay tolerance gate prevents accidentally snapping genuinely
        off-grid elements (e.g. transfer columns in complex structures).
        """
        x_sp = grid_info.get("x_spacings_mm", [])
        y_sp = grid_info.get("y_spacings_mm", [])
        n_x = len(grid_info.get("x_lines_px", []))
        n_y = len(grid_info.get("y_lines_px", []))

        if n_x == 0 or n_y == 0:
            return x_mm, y_mm

        # World x positions of vertical grid lines (left → right)
        x_grid = [sum(x_sp[:i]) for i in range(n_x)]

        # World y positions of horizontal grid lines — always Y-flipped to match
        # _px_to_world and _build_grid_commands (image Y↓ → Revit Y↑).
        total_y = sum(y_sp)
        y_grid = [total_y - sum(y_sp[:i]) for i in range(n_y)]

        # Tolerance: half the smallest bay in each axis
        all_sp = (x_sp or []) + (y_sp or [])
        min_bay = min(all_sp) if all_sp else DEFAULT_STOREY_HEIGHT_MM
        tol = min_bay / 2.0

        snapped_x = _nearest(x_mm, x_grid)
        snapped_y = _nearest(y_mm, y_grid)

        new_x = snapped_x if abs(snapped_x - x_mm) <= tol else x_mm
        new_y = snapped_y if abs(snapped_y - y_mm) <= tol else y_mm

        if new_x != x_mm or new_y != y_mm:
            logger.debug(
                f"Column grid snap: ({x_mm:.0f}, {y_mm:.0f}) → ({new_x:.0f}, {new_y:.0f}) mm"
            )

        return new_x, new_y

    def _build_wall_parameters(
        self, walls_2d: List[Dict], grid_info: Dict
    ) -> List[Dict]:
        """Generate parameters for Revit Wall.Create (Solid Modeling)."""
        walls_params = []
        for wall in walls_2d:
            endpoints = wall.get("endpoints", [[0, 0], [0, 0]])
            s0, s1 = endpoints[0], endpoints[1]

            sx_mm, sy_mm = self._px_to_world(s0[0], s0[1], grid_info)
            ex_mm, ey_mm = self._px_to_world(s1[0], s1[1], grid_info)

            # Convert pixel thickness to mm using average px/mm
            px_per_mm = grid_info.get("pixels_per_mm", 1.0)
            if px_per_mm > 0:
                thickness_mm = wall.get("thickness", self.default_wall_thickness * px_per_mm) / px_per_mm
            else:
                thickness_mm = self.default_wall_thickness

            walls_params.append({
                "id":            wall.get("id"),
                "start_point":   {"x": sx_mm, "y": sy_mm, "z": 0.0},
                "end_point":     {"x": ex_mm, "y": ey_mm, "z": 0.0},
                "thickness":     round(thickness_mm, 1),
                "height":        wall.get("ceiling_height", self.default_wall_height),
                "material":      wall.get("material", "Concrete"),
                "is_structural": wall.get("structural", False),
                "function":      wall.get("wall_function", "Interior"),
                "level":         "Level 0",
            })
        return walls_params

    def _build_opening_parameters(
        self, openings_2d: List[Dict], grid_info: Dict, o_type: str
    ) -> List[Dict]:
        """Generate parameters for FamilyInstance creation (Doors / Windows)."""
        opening_params = []
        for op in openings_2d:
            center = op.get("center", [0.0, 0.0])
            cx_mm, cy_mm = self._px_to_world(center[0], center[1], grid_info)

            # Width / height from pixel bbox, converted via px_per_mm
            px_per_mm = grid_info.get("pixels_per_mm", 1.0)
            bbox = op.get("bbox", [0, 0, 0, 0])
            w_px = abs(bbox[2] - bbox[0]) if len(bbox) >= 4 else 0.0
            h_px = abs(bbox[3] - bbox[1]) if len(bbox) >= 4 else 0.0
            width_mm  = (w_px / px_per_mm) if px_per_mm > 0 else (900 if o_type == "door" else 1200)
            height_mm = (h_px / px_per_mm) if px_per_mm > 0 else (self.default_door_height if o_type == "door" else self.default_window_height)

            param = {
                "id":           op.get("id"),
                "location":     {
                    "x": cx_mm,
                    "y": cy_mm,
                    "z": 0.0 if o_type == "door" else self.default_sill_height,
                },
                "width":        round(max(width_mm,  200.0), 1),
                "height":       round(max(height_mm, 400.0), 1),
                "type_name":    op.get("door_type" if o_type == "door" else "window_type", "Standard"),
                "host_wall_id": op.get("host_wall_id"),
                "level":        "Level 0",
            }
            if o_type == "door":
                param["swing_direction"] = op.get("swing_direction", "Right")
            opening_params.append(param)
        return opening_params

    def _build_column_parameters(
        self, columns_2d: List[Dict], grid_info: Dict
    ) -> List[Dict]:
        """Generate parameters for Revit Column creation at grid intersections."""
        column_params = []

        for col in columns_2d:
            center = col.get("center")
            if not center or (center[0] == 0.0 and center[1] == 0.0):
                logger.warning(
                    f"Column {col.get('id', '?')}: 'center' field missing or at origin — "
                    "placing at (0, 0). Check YOLO detection or fusion output."
                )
                center = center or [0.0, 0.0]
            cx_mm, cy_mm = self._px_to_world(center[0], center[1], grid_info)
            cx_mm, cy_mm = self._snap_to_nearest_grid(cx_mm, cy_mm, grid_info)

            # Prefer annotation-extractor key ("column_shape") over LLM key ("shape") so
            # precise PDF text reads aren't overridden by vaguer visual classification.
            shape = (
                "circular" if col.get("is_circular")
                else col.get("column_shape") or col.get("shape", "rectangular")
            )

            # ── Dimension source priority ──────────────────────────────────────
            # 1. Circular annotation  (Ø200 → diameter_mm)
            # 2. Rectangular/square annotation  (C1 800×800 → width_mm/depth_mm)
            # 3. No annotation → safe structural default (_min_column_mm)
            # Circular columns always snap to STANDARD_CIRCULAR_COLUMN_DIAMETERS
            # regardless of annotated value; annotated_dims stays None so Priority 2
            # fires inside normalize_column_dimensions.
            annotated_dims: Optional[Tuple[float, float]] = None
            has_rect_annotation = False
            if col.get("is_circular") and col.get("diameter_mm"):
                diam     = float(col["diameter_mm"])
                width_mm = diam
                depth_mm = diam
            elif col.get("width_mm") and col.get("depth_mm"):
                width_mm = float(col["width_mm"])
                depth_mm = float(col["depth_mm"])
                has_rect_annotation = True
            else:
                # No PDF annotation found — use safe structural default.
                # YOLO pixel bboxes are too noisy for dimension extraction on
                # structural drawings.
                width_mm = self._min_column_mm
                depth_mm = self._min_column_mm

            # Enforce minimum — prevents Revit family extrusion errors
            width_mm = max(width_mm, self._min_column_mm)
            depth_mm = max(depth_mm, self._min_column_mm)

            if has_rect_annotation:
                annotated_dims = (width_mm, depth_mm)

            width_mm, depth_mm, family_suffix = normalize_column_dimensions(
                width_mm, depth_mm,
                column_shape=shape,
                annotated_dimensions=annotated_dims,
            )

            # Symbol uses the family's max×min mm convention (e.g. "800x300mm")
            # so C# finds the existing type instead of falling back to duplication.
            if shape == "circular":
                family_name = "CJY_RC Round Column"
                symbol_name = f"Φ{int(width_mm)}"
            else:
                family_name = "CJY_Concrete-Rectangular-Column"
                big, small = max(width_mm, depth_mm), min(width_mm, depth_mm)
                symbol_name = f"{int(big)}x{int(small)}mm"

            width_r   = round(width_mm, 1)
            depth_r   = round(depth_mm, 1)
            material  = col.get("material", "Concrete")

            column_params.append({
                "id":          col.get("id"),
                "type_mark":   col.get("type_mark"),
                "Parameters": {
                    "Family":   family_name,
                    "Symbol":   symbol_name,
                    "Location": {"X": cx_mm, "Y": cy_mm, "Z": 0.0},
                    "Level":    "Level 0",
                    "TopLevel": "Level 1",
                },
                "Properties": {
                    "Width":    width_r,
                    "Depth":    depth_r,
                    "Material": material,
                },
                "family_type": family_suffix,
                "location":    {"x": cx_mm, "y": cy_mm, "z": 0.0},
                "width":       width_r,
                "depth":       depth_r,
                "height":      col.get("ceiling_height", self.default_wall_height),
                "shape":       shape,
                "material":    material,
                "level":       "Level 0",
                "top_level":   "Level 1",
            })

        return column_params

    # ------------------------------------------------------------------
    # Structural element stubs — populated once detection agents are trained
    # ------------------------------------------------------------------

    def _build_structural_framing_parameters(
        self,
        elements: List[Dict],
        grid_info: Dict,
        level0_elev: float,
        slab_regions: Optional[List[Dict]] = None,
    ) -> List[Dict]:
        """Convert detected beam bboxes to Revit StructuralFraming recipe entries.

        The YOLO bbox is used ONLY for the beam's centre-line (long axis + span).
        The cross-section is the project default (DEFAULT_BEAM_WIDTH_MM ×
        DEFAULT_BEAM_DEPTH_MM, 800×800 for concrete to match the column default)
        unless the admittance layer supplies `section_width_mm` / `section_depth_mm`
        from a parsed annotation label — we never infer the section from the
        bbox short side (it's a drafting-line thickness, not a structural size,
        and produces hallucinated type names like "1050x800mm").

        Z convention: the insertion Z is the beam TOP elevation (NOT the
        centroid). Columns and slab both sit on Level 0 (z=0); the slab
        spans 0 → +slab_thickness, and the beam TOP is flush with the slab
        top at +slab_thickness. The beam body hangs depth_mm downward
        from there into the foundation zone below Level 0.

            z_mm = Level0 + slab_thickness     # = beam top elevation

        Empirical note: the Add-in leaves Z_JUSTIFICATION=Center (1), but
        for the project's RC framing family the insertion curve actually
        lands at the beam TOP in the rendered model — the family's
        internal reference plane is top-referenced. If the family is ever
        swapped (e.g. for steel), re-verify that this still holds and
        adjust either this formula or the Z_JUSTIFICATION value.

        The slab_thickness used per-beam is the resolved zone thickness
        from NSP/CIS codes parsed off the drawing's NOTES/legend (see
        `_beam_slab_thickness`); `default_floor_thickness` (200 mm) is the
        fallback when the beam midpoint doesn't land in any annotated zone
        or the parser couldn't read a code for it.
        """
        params: List[Dict] = []

        for elem in elements:
            # Admittance layer may have rejected this element; orchestrator
            # usually filters those, but guard here too.
            decision = elem.get("admittance_decision") or {}
            if decision.get("action") == REJECT:
                continue
            bbox = elem.get("bbox")
            if not bbox or len(bbox) < 4:
                continue

            x1_px, y1_px, x2_px, y2_px = bbox[0], bbox[1], bbox[2], bbox[3]

            x1_mm, y1_mm = self._px_to_world(x1_px, y1_px, grid_info)
            x2_mm, y2_mm = self._px_to_world(x2_px, y2_px, grid_info)

            dx = abs(x2_mm - x1_mm)
            dy = abs(y2_mm - y1_mm)
            mid_x = (x1_mm + x2_mm) / 2.0
            mid_y = (y1_mm + y2_mm) / 2.0

            metadata = elem.get("admittance_metadata") or {}
            width_mm = float(metadata.get("section_width_mm") or self.default_beam_width)
            depth_mm = float(metadata.get("section_depth_mm") or self.default_beam_depth)

            slab_thickness = self._beam_slab_thickness(mid_x, mid_y, slab_regions)
            z_mm = level0_elev + slab_thickness

            if dx >= dy:
                start = {"x": min(x1_mm, x2_mm), "y": mid_y, "z": z_mm}
                end   = {"x": max(x1_mm, x2_mm), "y": mid_y, "z": z_mm}
            else:
                start = {"x": mid_x, "y": min(y1_mm, y2_mm), "z": z_mm}
                end   = {"x": mid_x, "y": max(y1_mm, y2_mm), "z": z_mm}

            span = abs(end["x"] - start["x"]) + abs(end["y"] - start["y"])
            if span < 10.0:   # skip degenerate detections (< 10 mm span)
                continue

            max_dim = max(width_mm, depth_mm)
            min_dim = min(width_mm, depth_mm)

            # Default to concrete (RC): CJY_RC Structural Framing is the project's
            # standard beam family. Only switch to steel when the admittance layer
            # explicitly identified the material as steel.
            material = metadata.get("material") or "rc"
            family_prefix = {"rc": "RCBeam", "steel": "SteelBeam"}.get(material, "RCBeam")
            entry = {
                "id":          elem.get("id"),
                "start_point": start,
                "end_point":   end,
                "width":       round(width_mm, 1),
                "depth":       round(depth_mm, 1),
                "level":       "Level 0",
                "family_type": f"{family_prefix}{int(max_dim)}x{int(min_dim)}mm",
            }
            if material:
                entry["material"] = material
            params.append(entry)

        return params

    def _build_stairs_parameters(self, elements: List[Dict]) -> List[Dict]:
        return []

    def _build_lift_parameters(self, elements: List[Dict]) -> List[Dict]:
        return []

    def _build_slab_parameters(
        self,
        slabs_2d: List[Dict],
        grid_info: Dict,
        zone_labels_mm: Optional[List[Tuple[str, float, float]]] = None,
        slab_legend: Optional[Dict[str, float]] = None,
    ) -> List[Dict]:
        """Generate Revit slab parameters from SlabDetectionAgent output.

        Each detected slab bbox becomes a flat boundary polygon. Thickness is
        resolved from a containing zone label (if any) via NOTES legend or a
        self-describing code; otherwise falls back to default_floor_thickness.
        """
        params: List[Dict] = []
        for i, slab in enumerate(slabs_2d):
            bbox = slab.get("bbox")
            if not bbox or len(bbox) < 4:
                continue
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            boundary_mm = [
                {"x": xm, "y": ym}
                for xm, ym in (
                    self._px_to_world(pt[0], pt[1], grid_info)
                    for pt in [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                )
            ]
            thickness = self._resolve_slab_thickness(
                boundary_mm, zone_labels_mm, slab_legend,
            )
            params.append({
                "id":              slab.get("id", f"slab_{i}"),
                "boundary_points": boundary_mm,
                "thickness":       thickness,
                "elevation":       0.0,
                "level":           "Level 0",
            })
        return params

    # ------------------------------------------------------------------
    # Per-region slab thickness resolution
    # ------------------------------------------------------------------

    def _resolve_slab_thickness(
        self,
        region_polygon_mm: List[Dict],
        zone_labels_mm: Optional[List[Tuple[str, float, float]]],
        legend: Optional[Dict[str, float]],
    ) -> float:
        """Return slab thickness for *region_polygon_mm* in mm.

        Tests each zone label's centre against the region polygon (ray-cast
        PIP). The first containing label is resolved via
        slab_thickness_parser.resolve_code_thickness; falls through to
        self.default_floor_thickness on any miss.
        """
        if not zone_labels_mm or len(region_polygon_mm) < 3:
            return float(self.default_floor_thickness)
        for code, lx, ly in zone_labels_mm:
            if not _point_in_polygon(lx, ly, region_polygon_mm):
                continue
            t = resolve_code_thickness(code, legend)
            if t is not None:
                return t
        return float(self.default_floor_thickness)

    def _beam_slab_thickness(
        self,
        mid_x: float,
        mid_y: float,
        slab_regions: Optional[List[Dict]],
    ) -> float:
        """Return thickness of the first slab region containing (mid_x, mid_y),
        or the uniform default when no region matches."""
        if not slab_regions:
            return float(self.default_floor_thickness)
        for slab in slab_regions:
            poly = slab.get("boundary_points") or []
            if len(poly) >= 3 and _point_in_polygon(mid_x, mid_y, poly):
                return float(slab.get("thickness", self.default_floor_thickness))
        return float(self.default_floor_thickness)
