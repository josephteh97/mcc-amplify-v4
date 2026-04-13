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

        for idx, wall in enumerate(geometry_data.get("walls", [])):
            m = self._wall_mesh(wall)
            if m is not None:
                m.visual.face_colors = [200, 200, 200, 255]
                scene.add_geometry(m, geom_name=f"wall_{idx}")

        for idx, col in enumerate(geometry_data.get("columns", [])):
            m = self._column_mesh(col)
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

        for idx, slab in enumerate(geometry_data.get("floors", [])):
            m = self._slab_mesh(slab)
            if m is not None:
                m.visual.face_colors = [220, 210, 190, 255]  # warm beige
                scene.add_geometry(m, geom_name=f"floor_{idx}")

        for idx, slab in enumerate(geometry_data.get("ceilings", [])):
            m = self._slab_mesh(slab)
            if m is not None:
                m.visual.face_colors = [240, 240, 240, 220]  # off-white
                scene.add_geometry(m, geom_name=f"ceiling_{idx}")

        if len(scene.geometry) == 0:
            logger.warning("No geometry produced — adding placeholder floor plane")
            plane = trimesh.creation.box(extents=[1000, 1000, 1])
            plane.visual.face_colors = [180, 180, 180, 255]
            scene.add_geometry(plane)

        scene.export(output_path)
        logger.info(
            f"glTF export complete — walls:{len(geometry_data.get('walls', []))} "
            f"doors:{len(geometry_data.get('doors', []))} "
            f"windows:{len(geometry_data.get('windows', []))} "
            f"columns:{len(geometry_data.get('columns', []))} "
            f"floors:{len(geometry_data.get('floors', []))}"
        )
        return output_path

    # ── Mesh builders ──────────────────────────────────────────────────────────

    def _wall_mesh(self, wall: dict):
        try:
            s, e = wall["start_point"], wall["end_point"]
            dx = e["x"] - s["x"]
            dy = e["y"] - s["y"]
            length = float(np.sqrt(dx ** 2 + dy ** 2))
            if length < 1.0:
                return None
            angle = float(np.arctan2(dy, dx))
            thickness = float(wall.get("thickness", 200))
            height    = float(wall.get("height", 2800))
            box = trimesh.creation.box(extents=[length, thickness, height])
            cx  = (s["x"] + e["x"]) / 2
            cy  = (s["y"] + e["y"]) / 2
            T   = trimesh.transformations.translation_matrix([cx, cy, height / 2])
            R   = trimesh.transformations.rotation_matrix(angle, [0, 0, 1])
            box.apply_transform(trimesh.transformations.concatenate_matrices(T, R))
            return box
        except Exception as exc:
            logger.debug(f"Wall mesh skipped: {exc}")
            return None

    def _column_mesh(self, col: dict):
        try:
            loc    = col["location"]
            width  = float(col.get("width",  300))
            depth  = float(col.get("depth",  300))
            height = float(col.get("height", 2800))
            if col.get("shape") == "circular":
                mesh = trimesh.creation.cylinder(radius=width / 2, height=height)
            else:
                mesh = trimesh.creation.box(extents=[width, depth, height])
            T = trimesh.transformations.translation_matrix([loc["x"], loc["y"], height / 2])
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
        """Floor or ceiling slab — flat box from boundary bounding rect."""
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
            T = trimesh.transformations.translation_matrix([cx, cy, elevation + thickness / 2])
            mesh.apply_transform(T)
            return mesh
        except Exception as exc:
            logger.debug(f"Slab mesh skipped: {exc}")
            return None
