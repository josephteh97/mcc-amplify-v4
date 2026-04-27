"""
Stage 7: glTF Exporter
Exports 3D geometry to glTF/GLB for web viewing.

Geometry exported:
  • Walls     — grey boxes oriented along wall axis
  • Columns   — grey boxes or cylinders
  • Doors     — brown thin boxes at door location
  • Windows   — light-blue thin boxes at sill height
  • Floors    — beige flat boxes from room boundary
  • Ceilings  — light-grey flat boxes at ceiling elevation
"""

import numpy as np
import trimesh
from pathlib import Path
from loguru import logger


class GltfExporter:

    async def export(self, geometry_data: dict, output_path: str) -> str:
        """Export geometry dict (from GeometryGenerator) to a .glb file."""
        logger.info(f"Exporting glTF to {output_path}")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        scene = trimesh.Scene()

        # Level name → elevation mm. Used so columns extend to their top-level
        # elevation (matching Revit) rather than a standalone `height` field —
        # otherwise beams placed at Level 1 elevation float above column tops.
        level_elev = {
            l["name"]: float(l.get("elevation", 0))
            for l in geometry_data.get("levels", [])
        }

        for idx, wall in enumerate(geometry_data.get("walls", [])):
            m = self._wall_mesh(wall)
            if m is not None:
                m.visual.face_colors = [200, 200, 200, 255]
                scene.add_geometry(m, geom_name=f"wall_{idx}")

        for idx, col in enumerate(geometry_data.get("columns", [])):
            m = self._column_mesh(col, level_elev)
            if m is not None:
                m.visual.face_colors = [150, 150, 150, 255]
                scene.add_geometry(m, geom_name=f"column_{idx}")

        for idx, door in enumerate(geometry_data.get("doors", [])):
            m = self._opening_mesh(door, depth=100.0)
            if m is not None:
                m.visual.face_colors = [139, 90, 43, 255]    # brown
                scene.add_geometry(m, geom_name=f"door_{idx}")

        for idx, win in enumerate(geometry_data.get("windows", [])):
            m = self._opening_mesh(win, depth=50.0)
            if m is not None:
                m.visual.face_colors = [135, 206, 235, 200]  # sky-blue
                scene.add_geometry(m, geom_name=f"window_{idx}")

        for idx, slab in enumerate(geometry_data.get("slabs", [])):
            m = self._slab_mesh(slab)
            if m is not None:
                m.visual.face_colors = [220, 210, 190, 255]  # warm beige
                scene.add_geometry(m, geom_name=f"slab_{idx}")

        for idx, beam in enumerate(geometry_data.get("structural_framing", [])):
            m = self._beam_mesh(beam)
            if m is not None:
                m.visual.face_colors = [90, 75, 60, 255]    # dark concrete brown
                scene.add_geometry(m, geom_name=f"beam_{idx}")

        if len(scene.geometry) == 0:
            logger.warning("No geometry produced — adding placeholder floor plane")
            plane = trimesh.creation.box(extents=[1000, 1000, 1])
            plane.visual.face_colors = [180, 180, 180, 255]
            scene.add_geometry(plane)

        # BIM uses Z-up; glTF uses Y-up. Rotate the whole scene −90° about X
        # so columns extrude upward in viewers instead of toward the camera.
        z_up_to_y_up = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
        scene.apply_transform(z_up_to_y_up)

        scene.export(output_path)
        logger.info(
            f"glTF export complete — "
            f"columns:{len(geometry_data.get('columns', []))} "
            f"framing:{len(geometry_data.get('structural_framing', []))} "
            f"walls:{len(geometry_data.get('walls', []))} "
            f"slabs:{len(geometry_data.get('slabs', []))}"
        )
        return output_path

    # ── Mesh builders ──────────────────────────────────────────────────────────

    def _axis_box_mesh(self, start: dict, end: dict, cross_w: float, cross_h: float, z_center: float):
        """Shared builder: oriented box along a start→end axis at the given z centre."""
        dx   = end["x"] - start["x"]
        dy   = end["y"] - start["y"]
        span = float(np.sqrt(dx ** 2 + dy ** 2))
        if span < 1.0:
            return None
        angle = float(np.arctan2(dy, dx))
        cx    = (start["x"] + end["x"]) / 2
        cy    = (start["y"] + end["y"]) / 2
        box   = trimesh.creation.box(extents=[span, cross_w, cross_h])
        T     = trimesh.transformations.translation_matrix([cx, cy, z_center])
        R     = trimesh.transformations.rotation_matrix(angle, [0, 0, 1])
        box.apply_transform(trimesh.transformations.concatenate_matrices(T, R))
        return box

    def _wall_mesh(self, wall: dict):
        try:
            thickness = float(wall.get("thickness", 200))
            height    = float(wall.get("height", 2800))
            return self._axis_box_mesh(
                wall["start_point"], wall["end_point"],
                thickness, height, height / 2,
            )
        except Exception as exc:
            logger.debug(f"Wall mesh skipped: {exc}")
            return None

    def _column_mesh(self, col: dict, level_elev: dict):
        try:
            loc   = col["location"]
            width = float(col.get("width",  300))
            depth = float(col.get("depth",  300))

            # Prefer base/top-level elevations (match Revit) over standalone
            # `height` — latter is a room ceiling metric that undershoots the
            # storey. Fall back to `height` only when levels aren't provided.
            base_elev = level_elev.get(col.get("level", "Level 0"), 0.0)
            top_elev  = level_elev.get(col.get("top_level", "Level 1"))
            if top_elev is None:
                top_elev = base_elev + float(col.get("height", 2800))
            height = top_elev - base_elev
            if height <= 0:
                return None

            if col.get("shape") == "circular":
                mesh = trimesh.creation.cylinder(radius=width / 2, height=height)
            else:
                mesh = trimesh.creation.box(extents=[width, depth, height])
            T = trimesh.transformations.translation_matrix(
                [loc["x"], loc["y"], base_elev + height / 2])
            mesh.apply_transform(T)
            return mesh
        except Exception as exc:
            logger.debug(f"Column mesh skipped: {exc}")
            return None

    def _opening_mesh(self, opening: dict, depth: float = 100.0):
        """Door or window — thin box placed at the given location."""
        try:
            loc    = opening["location"]
            width  = float(opening.get("width",  900))
            height = float(opening.get("height", 2100))
            z      = float(loc.get("z", 0))
            box    = trimesh.creation.box(extents=[width, depth, height])
            T = trimesh.transformations.translation_matrix([loc["x"], loc["y"], z + height / 2])
            box.apply_transform(T)
            return box
        except Exception as exc:
            logger.debug(f"Opening mesh skipped: {exc}")
            return None

    def _slab_mesh(self, slab: dict):
        """Floor or ceiling slab — flat box from boundary bounding rect.

        Convention: `elevation` is the slab TOP elevation (matches Revit's
        Floor.Create default, which extrudes the body downward from the
        boundary). The mesh therefore centres at elevation - thickness/2 so
        the preview shows the slab top flush with Level 0 (and flush with
        the beam top), body hanging into the foundation zone below.
        """
        try:
            pts = slab.get("boundary_points", [])
            if len(pts) < 3:
                return None
            xs  = [p["x"] for p in pts]
            ys  = [p["y"] for p in pts]
            w   = max(xs) - min(xs)
            d   = max(ys) - min(ys)
            if w < 1.0 or d < 1.0:
                return None
            cx        = (min(xs) + max(xs)) / 2
            cy        = (min(ys) + max(ys)) / 2
            thickness = float(slab.get("thickness", 200))
            elevation = float(slab.get("elevation", 0))
            mesh = trimesh.creation.box(extents=[w, d, thickness])
            T = trimesh.transformations.translation_matrix([cx, cy, elevation - thickness / 2])
            mesh.apply_transform(T)
            return mesh
        except Exception as exc:
            logger.debug(f"Slab mesh skipped: {exc}")
            return None

    def _beam_mesh(self, beam: dict):
        """Structural beam — box extruded along the start→end axis."""
        try:
            width = float(beam.get("width", 200))
            depth = float(beam.get("depth", 800))
            z     = float(beam["start_point"].get("z", 2800))
            return self._axis_box_mesh(
                beam["start_point"], beam["end_point"],
                width, depth, z - depth / 2,
            )
        except Exception as exc:
            logger.debug(f"Beam mesh skipped: {exc}")
            return None
