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

import numpy as np
from typing import Dict, List, Tuple, Optional
from loguru import logger
from pathlib import Path
import json

from backend.services.grid_detector import GridDetector


# Default floor-to-floor height when no storey height annotation is present.
DEFAULT_STOREY_HEIGHT_MM = 3000


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
        self._min_column_mm           = 200.0

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

    async def build(self, enriched_data: Dict, grid_info: Dict) -> Dict:
        """
        Build Semantic 3D parameters from enriched analysis data.

        Args:
            enriched_data : data from Claude/Gemini analysis merged with YOLO
            grid_info     : grid_info dict from GridDetector.detect()
                            (replaces the old scale_info / pixels_per_mm approach)

        Returns:
            Dict matching the RevitTransaction JSON schema consumed by
            ModelBuilder.cs on the Windows Revit server.
        """
        logger.info("Generating Semantic 3D parameters for Revit (grid-based)…")

        geometry = {
            # ── Levels: always provide Ground Floor + First Floor ──────
            "levels": self._build_default_levels(enriched_data),

            # ── Structural grid lines ──────────────────────────────────
            "grids": self._build_grid_commands(grid_info),

            # ── Building elements ──────────────────────────────────────
            "walls":    self._build_wall_parameters(
                            enriched_data.get("walls", []), grid_info),
            "doors":    self._build_opening_parameters(
                            enriched_data.get("doors", []), grid_info, "door"),
            "windows":  self._build_opening_parameters(
                            enriched_data.get("windows", []), grid_info, "window"),
            "rooms":    self._build_room_parameters(
                            enriched_data.get("rooms", []), grid_info),
            "columns":  self._build_column_parameters(
                            enriched_data.get("columns", []), grid_info),
            **self._build_slabs(enriched_data.get("rooms", []), grid_info),

            "metadata": enriched_data.get("metadata", {}),
        }

        logger.info(
            f"Generated: {len(geometry['walls'])} walls, "
            f"{len(geometry['doors'])} doors, "
            f"{len(geometry['windows'])} windows, "
            f"{len(geometry['columns'])} columns, "
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

        snapped_x = min(x_grid, key=lambda gx: abs(gx - x_mm))
        snapped_y = min(y_grid, key=lambda gy: abs(gy - y_mm))

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
        px_per_mm = grid_info.get("pixels_per_mm", 1.0)

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

            # ── Dimension priority ─────────────────────────────────────────────
            # 1. Circular annotation  (e.g. "Ø200" read by AI or text parser)
            # 2. Rectangular annotation (e.g. "C1 800x800")
            # 3. YOLO pixel bbox → mm via pixels_per_mm  (least reliable)
            if col.get("is_circular") and col.get("diameter_mm"):
                diam     = float(col["diameter_mm"])
                width_mm = diam
                depth_mm = diam
            elif col.get("width_mm") and col.get("depth_mm"):
                width_mm = float(col["width_mm"])
                depth_mm = float(col["depth_mm"])
            else:
                # No PDF annotation was found by _annotate_columns_from_vector_text.
                # YOLO pixel bboxes are too noisy for dimension extraction on
                # structural drawings — use the safe structural default instead.
                width_mm = self._min_column_mm
                depth_mm = self._min_column_mm

            # Enforce minimum — prevents Revit family extrusion errors
            width_mm = max(width_mm, self._min_column_mm)
            depth_mm = max(depth_mm, self._min_column_mm)

            shape = "circular" if col.get("is_circular") else col.get("column_shape", "rectangular")

            column_params.append({
                "id":        col.get("id"),
                "type_mark": col.get("type_mark"),
                "location":  {"x": cx_mm, "y": cy_mm, "z": 0.0},
                "width":     round(width_mm, 1),
                "depth":     round(depth_mm, 1),
                "height":    col.get("ceiling_height", self.default_wall_height),
                "shape":     shape,
                "material":  col.get("material", "Concrete"),
                "level":     "Level 0",
                "top_level": "Level 1",
            })
        return column_params

    def _build_room_parameters(
        self, rooms_2d: List[Dict], grid_info: Dict
    ) -> List[Dict]:
        """Generate parameters for Revit Room creation."""
        room_params = []
        for room in rooms_2d:
            center = room.get("center", [0.0, 0.0])
            cx_mm, cy_mm = self._px_to_world(center[0], center[1], grid_info)

            room_params.append({
                "id":            room.get("id"),
                "name":          room.get("name", "Unnamed Room"),
                "purpose":       room.get("purpose", "General"),
                "center_point":  {"x": cx_mm, "y": cy_mm, "z": 0.0},
                "area_sqm":      room.get("area_sqm", 0),
                "target_height": room.get("ceiling_height", self.default_wall_height),
                "level":         "Level 0",
            })
        return room_params

    def _build_slabs(self, rooms_2d: List[Dict], grid_info: Dict) -> Dict:
        """
        Generate floor and ceiling slab parameters in a single pass over rooms.

        Uses explicit polygon boundary when available; falls back to the YOLO
        bounding-box corners so every detected room still gets a slab.
        Returns {"floors": [...], "ceilings": [...]}.
        """
        floors: List[Dict] = []
        ceilings: List[Dict] = []

        for i, room in enumerate(rooms_2d):
            if "boundary" in room:
                boundary_px = room["boundary"]
            else:
                bbox = room.get("bbox")
                if not bbox or len(bbox) < 4:
                    logger.warning(
                        f"Room {i} ({room.get('name', '?')}): no 'boundary' polygon and no "
                        "valid 'bbox' — floor/ceiling slabs skipped."
                    )
                    continue
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                boundary_px = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

            boundary_mm = [
                {"x": xm, "y": ym}
                for xm, ym in (self._px_to_world(pt[0], pt[1], grid_info) for pt in boundary_px)
            ]
            ceiling_elev = room.get("ceiling_height", self.default_wall_height)

            floors.append({
                "id":              f"floor_{i}",
                "boundary_points": boundary_mm,
                "thickness":       self.default_floor_thickness,
                "elevation":       0.0,
                "level":           "Level 0",
            })
            ceilings.append({
                "id":              f"ceiling_{i}",
                "boundary_points": boundary_mm,
                "thickness":       20,
                "elevation":       ceiling_elev,
                "level":           "Level 0",
            })

        return {"floors": floors, "ceilings": ceilings}
