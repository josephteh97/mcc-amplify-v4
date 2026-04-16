"""
Core Orchestrator: Manages the Dual-Track Pipeline

Stages executed:
  1. Security check            (services/security)
  2. Dual-Track PDF processing  (services/pdf_processing)
       Track A – VectorProcessor  (precise geometry + text for grid detection)
       Track B – StreamingProcessor (raster render) + inline YOLO detection
  3. Hybrid Fusion              (services/fusion)
  4. Grid Detection             (services/grid_detector — NEVER reads scale text)
  5. Semantic AI analysis       (services/semantic_analyzer)
  6. 3D Geometry generation     (services/geometry_generator)
  7. BIM Export                 (services/exporters)

NOTE: Scale is derived exclusively from the structural grid lines and their
dimension annotations.  The "scale" text printed on a floor plan is intentionally
ignored because it is often unreliable.
"""

import asyncio
import copy
import json
import math
import os
import re
from pathlib import Path
from typing import Callable, Optional
from loguru import logger

# ── Column annotation regex patterns ─────────────────────────────────────────
# Matches: "800x800", "800X800", "800×800", "800*800"
_RE_RECT = re.compile(r'(\d{2,4})\s*[xX×*]\s*(\d{2,4})')
# Matches circular diameter in both orderings:
#   symbol-first: "Ø200", "⌀300", "∅300", "dia 200", "dia.200", "phi200"  → group 1
#   number-first: "300∅", "500Ø", "200⌀"  (CAD export format)             → group 2
_RE_CIRC = re.compile(
    r'(?:Ø|⌀|∅|dia\.?\s*|phi\s*)(\d{2,4})'   # group 1: symbol before number
    r'|(\d{2,4})\s*[Ø⌀∅]',                    # group 2: number before symbol
    re.IGNORECASE,
)
# Matches column type mark: any 1-3 uppercase letters followed by 1-3 digits
# e.g. "C1", "C20", "B3", "K14" — covers non-standard naming conventions
_RE_MARK = re.compile(r'\b([A-Z]{1,3}\d{1,3})\b')

# Beam / slab / lintel prefixes that must NOT be accepted as column marks.
# Without this filter the column annotator happily scrapes "RCB2 800×300"
# from a nearby beam schedule and applies it to an actual column.
_BEAM_MARK_PREFIX = re.compile(r'^(RCB|GB|SB|TB|FB|RB|SL|LB|L|B)\d', re.IGNORECASE)


def _is_beam_label(txt: str) -> bool:
    """True when *txt* carries a mark that identifies a beam/slab, not a column."""
    m = _RE_MARK.search(txt)
    return bool(m and _BEAM_MARK_PREFIX.match(m.group(1)))

# ── Dual-track infrastructure ──────────────────────────────────────────────────
from backend.services.security.secure_renderer import SecurePDFRenderer, ResourceMonitor
from backend.services.pdf_processing.processors import VectorProcessor, StreamingProcessor
from backend.services.fusion.pipeline import HybridFusionPipeline

# ── AI logic services ─────────────────────────────────────────────────────────
from backend.services.grid_detector import GridDetector, GridDimensionMissingError
from backend.services.semantic_analyzer import SemanticAnalyzer
from backend.services.geometry_generator import GeometryGenerator
from backend.services.exporters.rvt_exporter import RvtExporter
from backend.services.exporters.gltf_exporter import GltfExporter
from backend.services.vision_comparator import VisionComparator

# ── Intelligence layer ────────────────────────────────────────────────────────
from backend.services.intelligence.type_resolver import resolve_types
from backend.services.intelligence.cross_element_validator import validate_elements
from backend.services.intelligence.validation_agent import enforce_rules
from backend.services.intelligence.bim_translator_enricher import enrich_recipe


class PipelineOrchestrator:
    """
    Central brain of the Hybrid AI System.
    Orchestrates:
      Security → PDF Processing → YOLO → Fusion → Grid Detection → AI → Geometry → Export
    """

    def __init__(self):
        self.security         = SecurePDFRenderer()
        self.vector_processor = VectorProcessor()
        self.stream_processor = StreamingProcessor()
        self.fusion           = HybridFusionPipeline()
        self.grid_detector    = GridDetector()
        self.semantic_ai      = SemanticAnalyzer()
        self.geometry_gen     = GeometryGenerator()
        self.rvt_exporter     = RvtExporter()
        self.gltf_exporter    = GltfExporter()
        self.vision_cmp       = VisionComparator()

        # Load YOLO model (weights at ml/weights/column-detect.pt)
        self.yolo = None
        yolo_path = Path(__file__).parent.parent.parent / "ml" / "weights" / "column-detect.pt"
        if yolo_path.exists():
            try:
                from ultralytics import YOLO
                self.yolo = YOLO(str(yolo_path))
                logger.info(f"YOLO model loaded: {yolo_path.name}")
            except Exception as e:
                logger.warning(f"YOLO load failed ({e}) — detection will be skipped")
        else:
            logger.warning(f"YOLO weights not found at {yolo_path} — detection will be skipped")

    # ──────────────────────────────────────────────────────────────────────────

    async def run_pipeline(
        self,
        pdf_path: str,
        job_id: str,
        project_name: str = "Project",
        pdf_filename: str = "",
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ):
        """
        Main execution flow.  Calls progress_callback(pct, message) at each
        stage so the frontend progress bar stays alive.
        """

        def progress(pct: int, msg: str):
            logger.info(f"[{pct}%] {msg}")
            if progress_callback:
                progress_callback(pct, msg)

        logger.info(f"🚀 Starting Hybrid Pipeline — Job {job_id}")
        monitor = ResourceMonitor()
        monitor.start()

        try:
            # ── Stage 1: Security Check ────────────────────────────────────────
            progress(5, "Security & size check…")
            secure_context = await self.security.safe_render(pdf_path)
            safe_dpi = secure_context.get("dpi", 150)

            # ── Stage 2a: Track A — Vector extraction ─────────────────────────
            progress(15, "Track A: extracting vector geometry…")
            vector_data = self.vector_processor.extract(pdf_path)

            # ── Scanned PDF check ─────────────────────────────────────────────
            # A PDF with fewer than 50 vector paths is almost certainly a scanned
            # raster image.  Grid detection will fall back to a uniform grid and
            # coordinate accuracy will be significantly reduced.
            path_count = len(vector_data.get("paths", []))
            is_scanned = path_count < 50
            if is_scanned:
                logger.warning(
                    f"Scanned/raster PDF detected ({path_count} vector paths). "
                    "Grid detection will use fallback coordinates — accuracy reduced."
                )

            # ── Stage 2a (supplement): Multi-page schedule scan ───────────────
            # Structural drawing sets often put the column schedule on a
            # separate page.  Scan all pages now so Pass 1 of the column
            # annotation step can use type-mark → dimension entries from
            # those schedule pages, even though the floor plan is on page 0.
            progress(20, "Scanning all PDF pages for column schedules…")
            extra_pages      = self.vector_processor.extract_all_pages_text(pdf_path)
            schedule_page_texts = [
                item["text"]
                for page in extra_pages if page["is_schedule"]
                for item in page["text_items"]
            ]
            if schedule_page_texts:
                n_sched = sum(1 for p in extra_pages if p["is_schedule"])
                logger.info(
                    f"Cross-page schedule: {len(schedule_page_texts)} text items "
                    f"from {n_sched} schedule page(s) merged into annotation pass."
                )

            # ── Stage 2b: Track B — Raster rendering ──────────────────────────
            # Request 300 DPI so 800 mm columns at 1:400 appear as ~24 px —
            # the same scale the YOLO model was trained on.  The renderer's
            # MAX_PIXELS cap will reduce DPI automatically for very large sheets
            # (A0 caps at ~127 DPI); _run_yolo handles that by upsampling.
            progress(25, "Track B: raster rendering…")
            image_data = await self.stream_processor.render_safe(pdf_path, dpi=300)

            # ── Stage 2c: YOLO element detection on rendered image ─────────────
            progress(35, "Detecting elements (YOLO)…")
            ml_detections = self._run_yolo(image_data)

            # ── Checkpoint: save render + detections for YOLO training ─────────
            self._save_job_checkpoint(job_id, "render.jpg", image_data["image"])
            self._save_job_checkpoint(job_id, "px_detections.json", ml_detections)

            # ── Stage 3: Hybrid Fusion ─────────────────────────────────────────
            progress(45, "Fusing vector & ML detections…")
            fused_data = await self.fusion.fuse(
                vector_data,
                ml_detections,
                {
                    "width":  image_data["width"],
                    "height": image_data["height"],
                    "dpi":    image_data.get("dpi", safe_dpi),
                },
            )
            # Use refined pixel-space detections if fusion produced them;
            # fall back to raw YOLO detections if fusion returned nothing useful.
            refined_detections = fused_data.get("refined_px") or ml_detections

            # ── Stage 4: Grid Detection ────────────────────────────────────────
            # We derive the real-world coordinate system from structural grid
            # lines and their dimension annotations — never from scale text.
            progress(55, "Detecting structural grid lines…")
            grid_info = self._detect_grid(vector_data, image_data)

            # ── Stage 4b: Align grid pixel reference to YOLO column centres ──────
            # The structural grid datum (mm spacings) comes from vector-detected
            # dashed centre lines + PDF dimension annotations.  The PIXEL positions
            # of those lines can be 10–40 px off from the rendered column centres
            # due to rasterisation differences, which causes a world-coordinate
            # offset of hundreds of mm (e.g. 30 px / 280 px × 8400 mm ≈ 900 mm).
            #
            # Strategy (two-step, both non-fatal):
            #   Step 1 — align_pixels_to_columns: update ONLY pixel positions to
            #             match YOLO column centres while keeping all mm spacings.
            #             Never fails, silently skips per-axis if count mismatches.
            #   Step 2 — refine_with_columns: full re-detection of spacings from
            #             the updated positions.  Attempted but skipped if it raises
            #             GridDimensionMissingError (new gaps have no annotations).
            column_raw = [d for d in refined_detections if d.get("type") == "column"]
            if len(column_raw) >= 2:
                # Step 1: always run — aligns pixel reference, never fails.
                grid_info = self.grid_detector.align_pixels_to_columns(
                    grid_info, column_raw
                )

                # NOTE: refine_with_columns is intentionally NOT called here.
                # The structural grid (number of lines + mm spacings) is derived
                # exclusively from the PDF vector data — it is the authoritative
                # datum.  Columns are slaves to the grid, never the reverse.
                # align_pixels_to_columns (Step 1) is sufficient: it updates pixel
                # reference positions only when the YOLO column count matches the
                # detected grid line count, preserving all mm spacings.
                logger.info(
                    f"Grid pixel alignment complete — grid is PDF-authoritative "
                    f"({len(grid_info.get('x_lines_px',[]))} V × "
                    f"{len(grid_info.get('y_lines_px',[]))} H lines kept from PDF)."
                )

            # ── Stage 4c: Intelligence middleware (post-detection, pre-geometry) ──
            progress(58, "Intelligence layer: type resolution & validation…")
            _column_dets = []
            if column_raw and image_data.get("image") is not None:
                _column_dets = resolve_types(column_raw, image_data["image"])
                _column_dets = validate_elements(
                    _column_dets,
                    grid_info=grid_info,
                    max_grid_dist_px=float(os.getenv("MAX_GRID_DIST_PX", "80")),
                    isolation_radius_px=float(os.getenv("ISOLATION_RADIUS_PX", "200")),
                )
                _column_dets = enforce_rules(
                    _column_dets,
                    grid_info=grid_info,
                    min_bay_mm=float(os.getenv("MIN_BAY_MM", "3000")),
                    max_bay_mm=float(os.getenv("MAX_BAY_MM", "12000")),
                )

            # ── Stage 5: Semantic AI Analysis ─────────────────────────────────
            # Build structured element dict from pixel-space detections
            # so the geometry generator can snap them to the grid.
            progress(60, "AI semantic analysis…")
            structured_elements = self._format_for_geometry(refined_detections)

            # ── Stage 3b: Column annotation parsing ───────────────────────────
            # Match PDF text labels (e.g. "C1 800x800", "C20 Ø200") to the
            # YOLO-detected columns so geometry_generator uses real dimensions.
            # Must run AFTER _format_for_geometry() so structured_elements is
            # a dict with a "columns" key, not a raw list from YOLO.
            structured_elements = self._annotate_columns_from_vector_text(
                structured_elements, vector_data, image_data,
                extra_schedule_texts=schedule_page_texts,
            )
            enriched_data = await self.semantic_ai.analyze(
                image_data,
                structured_elements,
                grid_info,
            )
            # ── Checkpoint: save enriched data for debugging / re-runs ──────────
            self._save_job_checkpoint(job_id, "enriched.json", enriched_data)

            # ── Stage 6: 3D Geometry Generation ───────────────────────────────
            # Apply project profile defaults before generating geometry so that
            # user-configured wall heights, storey heights, etc. are respected.
            progress(75, "Generating 3D geometry…")
            _profile_path = Path("data/project_profile.json")
            if _profile_path.exists():
                with open(_profile_path) as _pf:
                    _profile = json.load(_pf)
                self.geometry_gen.apply_profile(_profile)
                logger.info(
                    f"Project profile applied: building_type={_profile.get('building_type')}, "
                    f"wall_h={_profile.get('typical_wall_height_mm')}mm, "
                    f"storey={_profile.get('floor_to_floor_height_mm')}mm"
                )
            recipe = await self.geometry_gen.build(enriched_data, grid_info)

            # ── Stage 6.5: BIM Translator Enrichment ─────────────────────────
            if _column_dets:
                recipe = enrich_recipe(recipe, _column_dets)

            # ── Pre-clash validation ───────────────────────────────────────────
            validation_warnings = self._validate_recipe(recipe)

            # ── Stage 7: BIM Export ────────────────────────────────────────────
            progress(88, "Exporting RVT & glTF…")
            transaction_path = f"data/models/rvt/{job_id}_transaction.json"
            Path(transaction_path).parent.mkdir(parents=True, exist_ok=True)
            # Embed job_id so the Revit macro knows the output filename
            recipe["job_id"] = job_id
            with open(transaction_path, "w") as f:
                json.dump(recipe, f)

            # glTF — always attempted; must succeed for the job to be useful
            gltf_path = f"data/models/gltf/{job_id}.glb"
            gltf_out  = await self.gltf_exporter.export(recipe, gltf_path)

            # RVT — optional: Revit server may be unreachable; don't fail the job.
            # Two build modes controlled by USE_AGENT_BUILDER env var:
            #   "true"  → Claude MCP agent places elements step-by-step (P5/P6)
            #   default → Batch build_model call (original path, unchanged)
            rvt_path    = None
            vision_diff = None
            _use_agent  = os.getenv("USE_AGENT_BUILDER", "").lower() == "true"
            try:
                if _use_agent:
                    rvt_path, vision_diff = await self._run_agent_export(recipe, job_id, progress, pdf_filename)
                else:
                    current_recipe = recipe
                    for _attempt in range(3):          # attempt 0, 1, 2
                        rvt_path, revit_warnings = await self.rvt_exporter.export(
                            transaction_path, job_id, pdf_filename
                        )

                        if not revit_warnings or _attempt == 2:
                            if revit_warnings:
                                logger.warning(
                                    f"Revit warnings remain after {_attempt + 1} correction "
                                    f"attempt(s) — accepted as-is: {revit_warnings}"
                                )
                            break

                        # Ask the AI what to change
                        progress(
                            90 + _attempt * 3,
                            f"Revit warnings — AI correcting (round {_attempt + 1}/2)…",
                        )
                        corrections = await self.semantic_ai.analyze_revit_warnings(
                            revit_warnings, current_recipe
                        )
                        if not corrections.get("corrections"):
                            logger.info(
                                "AI found no actionable corrections for Revit warnings "
                                f"— proceeding with current RVT: {revit_warnings}"
                            )
                            break

                        logger.info(
                            f"AI correction summary: {corrections.get('summary')} "
                            f"({len(corrections['corrections'])} change(s))"
                        )
                        current_recipe = self._apply_revit_corrections(
                            current_recipe, corrections
                        )
                        # Persist the corrected transaction for the next send
                        with open(transaction_path, "w") as f:
                            json.dump(current_recipe, f)

            except Exception as rvt_err:
                import traceback
                logger.warning(
                    f"RVT export skipped — {type(rvt_err).__name__}: {rvt_err}\n"
                    f"{traceback.format_exc()}"
                )

            progress(100, "Complete!")
            logger.info(
                f"✅ Pipeline complete — glTF: {gltf_out} | RVT: {rvt_path or 'skipped'}"
            )

            return {
                "job_id":  job_id,
                "status":  "completed",
                "files":   {"rvt": rvt_path, "gltf": gltf_out},
                "stats":   {
                    "method":                secure_context.get("method"),
                    "dpi":                   safe_dpi,
                    "element_count":         len(refined_detections),
                    "yolo_detections":       len(ml_detections),
                    "grid_source":           grid_info["source"],
                    "grid_lines":            f"{len(grid_info['x_lines_px'])}V × {len(grid_info['y_lines_px'])}H",
                    "has_grid":              grid_info["has_grid"],
                    "is_scanned":            is_scanned,
                    "grid_confidence":       grid_info.get("grid_confidence", 0.0),
                    "grid_confidence_label": grid_info.get("grid_confidence_label", "Unknown"),
                    "intelligence_valid":    sum(1 for d in _column_dets if d.get("is_valid", True)) if _column_dets else 0,
                    "intelligence_flagged":  sum(1 for d in _column_dets if not d.get("is_valid", True)) if _column_dets else 0,
                    "validation_warnings":   validation_warnings,
                    "vision_diff":           vision_diff,
                },
            }

        except Exception as e:
            import traceback
            logger.error(
                f"Pipeline failed: {type(e).__name__}: {e}\n"
                + traceback.format_exc()
            )
            raise
        finally:
            monitor.stop()

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _run_agent_export(
        self,
        recipe: dict,
        job_id: str,
        progress_fn: Callable[[int, str], None],
        pdf_filename: str = "",
    ) -> tuple[str | None, dict | None]:
        """
        P6: Use the Claude MCP agent to build the Revit model step-by-step.

        Called when USE_AGENT_BUILDER=true.  Falls back to (None, None) on
        any error so the pipeline still delivers the glTF to the user.

        Returns
        -------
        (rvt_path, vision_diff)
            rvt_path    — path to the saved .rvt file, or None on failure
            vision_diff — structured accuracy report from VisionComparator,
                          or None if the comparison could not run
        """
        try:
            from agents.revit_agent import RevitAgent
        except ImportError as e:
            logger.warning(f"RevitAgent import failed ({e}) — falling back to batch export")
            rvt_path, _ = await self.rvt_exporter.export(
                f"data/models/rvt/{job_id}_transaction.json", job_id, pdf_filename
            )
            return rvt_path, None

        progress_fn(89, "Agent builder: Claude is placing Revit elements…")

        def _on_agent_progress(msg: str):
            progress_fn(90, f"Agent: {msg}")

        agent = RevitAgent()
        result = await agent.run(recipe, job_id, on_progress=_on_agent_progress)

        if result["status"] != "done":
            logger.warning(
                f"Agent export failed after {result['turns']} turns: {result.get('error')}"
            )
            return None, None

        logger.info(
            f"Agent export complete — {result['placed_count']} elements placed, "
            f"{result['turns']} turns, rvt={result['rvt_path']}"
        )

        # ── Closed-loop vision comparison ─────────────────────────────────────
        # Export the Revit floor plan view and compare with the original render
        # to detect missing / misplaced elements.
        vision_diff = None
        session_id  = result.get("session_id")   # populated if agent kept session open
        original_render = f"data/jobs/{job_id}/render.jpg"

        if session_id:
            # Session still open — request the floor plan image before closing
            try:
                progress_fn(95, "Vision comparison: exporting Revit floor plan…")
                from services.revit_client import RevitClient
                rc = RevitClient()
                revit_png = await rc.export_floor_plan_view(session_id)
                progress_fn(96, "Vision comparison: comparing with original PDF…")
                vision_diff = await self.vision_cmp.compare(
                    original_image_path=original_render,
                    revit_png_bytes=revit_png,
                    job_id=job_id,
                )
                score = vision_diff.get("match_score")
                logger.info(f"Vision diff complete — match_score={score}")
            except Exception as ve:
                logger.warning(f"Vision comparison skipped: {ve}")

        return result["rvt_path"], vision_diff

    def _save_job_checkpoint(self, job_id: str, filename: str, data) -> None:
        """
        Persist intermediate pipeline data to data/jobs/{job_id}/{filename}.

        - .json files: serialised with json.dump (non-serialisable values skipped)
        - .jpg  files: numpy RGB array saved via PIL
        Never raises — checkpoint failures must not abort the pipeline.
        """
        try:
            job_dir = Path(f"data/jobs/{job_id}")
            job_dir.mkdir(parents=True, exist_ok=True)
            dest = job_dir / filename
            if filename.endswith(".json"):
                with open(dest, "w") as f:
                    json.dump(data, f, default=str)
            elif filename.endswith(".jpg"):
                from PIL import Image as _PIL
                _PIL.fromarray(data).save(str(dest), format="JPEG", quality=85)
        except Exception as e:
            logger.warning(f"Checkpoint save failed ({filename}): {e}")

    def _detect_grid(self, vector_data: dict, image_data: dict) -> dict:
        """
        Detect the structural column grid from PDF vector paths.

        The grid line positions and the spacing dimension annotations printed
        on the drawing are the ONLY source of real-world scale.  The scale
        label (e.g. "1:100") is intentionally ignored.
        """
        try:
            grid_info = self.grid_detector.detect(vector_data, image_data)
            logger.info(
                f"Grid detected: {len(grid_info['x_lines_px'])} vertical lines, "
                f"{len(grid_info['y_lines_px'])} horizontal lines "
                f"(source: {grid_info['source']})"
            )
            return grid_info
        except GridDimensionMissingError:
            # Grid lines were found but dimension annotations are missing.
            # Re-raise — BIM generation must NOT proceed without real coordinates.
            raise
        except Exception as e:
            logger.warning(f"Grid detection failed ({e}) — using fallback grid")
            return self.grid_detector._fallback_grid(
                image_data["width"], image_data["height"]
            )

    @staticmethod
    def _enhance_for_yolo(img_rgb):
        """
        Apply CLAHE contrast enhancement so that faint engineering-drawing
        lines become clearly visible before YOLO inference.

        Converts to LAB colour space, enhances the L channel with CLAHE
        (clipLimit=2, 8×8 tile grid), then converts back to RGB.
        Falls back to a simple percentile stretch if cv2 is unavailable.
        """
        import numpy as np
        try:
            import cv2
            lab   = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l     = clahe.apply(l)
            enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
        except Exception:
            p1, p99 = np.percentile(img_rgb, [1, 99])
            if p99 > p1:
                enhanced = np.clip(
                    (img_rgb.astype(float) - p1) / (p99 - p1) * 255, 0, 255
                ).astype(np.uint8)
            else:
                enhanced = img_rgb
        return enhanced

    def _run_yolo(self, image_data: dict) -> list:
        """
        Tiling YOLO inference — mirrors inspect_detections.ipynb exactly.

        The full rendered image is sliced into 1280×1280 px tiles with 200 px
        overlap.  Each tile is fed to YOLO at imgsz=1280 (scale=1.0×, no
        internal rescaling), so columns appear at the same ~24 px they had
        during training.  Detections are mapped back to global pixel coordinates
        and merged with NMS.

        Why NOT whole-image inference:
          • YOLO internally rescales the image to imgsz=640.  A 6000×4200 image
            shrinks ~9×, turning 24 px columns into ~3 px — invisible to the model.

        Why upsample when DPI < 300:
          • Training used 300 DPI renders; 800 mm columns appear as ~24 px.
          • At the renderer's 25 MP cap, A0 sheets land at ~127 DPI → ~10 px columns.
          • Upsampling to the 300 DPI equivalent restores the correct pixel scale
            before tiling, without changing any rendering logic.
        """
        if self.yolo is None or image_data is None:
            return []
        try:
            import numpy as np
            import torch
            from PIL import Image
            from torchvision.ops import nms as torch_nms

            img_np     = image_data["image"]          # H×W×3 uint8 numpy array
            render_dpi = image_data.get("dpi", 150)

            # ── Enhance contrast ──────────────────────────────────────────────
            enhanced = self._enhance_for_yolo(img_np)

            # ── Upsample to 300 DPI equivalent if rendered below target ───────
            # Training DPI = 300; if renderer capped to a lower DPI, columns are
            # proportionally smaller.  Upsample so they land at ~24 px in tiles.
            TARGET_DPI = 300
            if render_dpi < TARGET_DPI * 0.85:          # only if >15 % off
                scale  = TARGET_DPI / render_dpi        # e.g. 300/127 ≈ 2.36
                new_w  = int(enhanced.shape[1] * scale)
                new_h  = int(enhanced.shape[0] * scale)
                try:
                    resample = Image.Resampling.LANCZOS
                except AttributeError:
                    resample = Image.LANCZOS
                pil_img = Image.fromarray(enhanced).resize((new_w, new_h), resample)
                logger.info(
                    f"YOLO upsample: {enhanced.shape[1]}×{enhanced.shape[0]} "
                    f"({render_dpi} DPI) → {new_w}×{new_h} ({TARGET_DPI} DPI eq.)"
                )
                coord_scale = render_dpi / TARGET_DPI   # scale detections back later
            else:
                pil_img     = Image.fromarray(enhanced)
                coord_scale = 1.0

            W, H = pil_img.size

            # ── Sliding-window tiling ─────────────────────────────────────────
            TILE_SIZE = 1280
            OVERLAP   = 200
            CONF      = 0.25
            IOU       = 0.45
            step      = TILE_SIZE - OVERLAP

            raw_boxes, raw_confs = [], []
            ys = list(range(0, H, step))
            xs = list(range(0, W, step))
            total_tiles = len(xs) * len(ys)

            for y0 in ys:
                for x0 in xs:
                    x1 = min(x0 + TILE_SIZE, W);  xa = max(0, x1 - TILE_SIZE)
                    y1 = min(y0 + TILE_SIZE, H);  ya = max(0, y1 - TILE_SIZE)
                    tile = pil_img.crop((xa, ya, x1, y1))

                    res = self.yolo.predict(
                        source=tile, imgsz=TILE_SIZE,
                        conf=CONF, iou=IOU, verbose=False,
                    )[0]

                    for box, c in zip(res.boxes.xyxy.cpu().numpy(),
                                      res.boxes.conf.cpu().numpy()):
                        raw_boxes.append([
                            float(box[0]) + xa, float(box[1]) + ya,
                            float(box[2]) + xa, float(box[3]) + ya,
                        ])
                        raw_confs.append(float(c))

            logger.info(f"YOLO tiling: {total_tiles} tiles, {len(raw_boxes)} raw detections")

            if not raw_boxes:
                return []

            # ── Global NMS ────────────────────────────────────────────────────
            b_t  = torch.tensor(raw_boxes, dtype=torch.float32)
            c_t  = torch.tensor(raw_confs, dtype=torch.float32)
            keep = torch_nms(b_t, c_t, iou_threshold=IOU).numpy()
            b_nms = b_t.numpy()[keep]
            c_nms = c_t.numpy()[keep]

            # ── Squareness + size filter ──────────────────────────────────────
            # Columns are near-square; elongated elements (walls, beams) are not.
            # Size bounds: 10–80 px in upsampled (300 DPI eq.) space.
            MIN_SQ   = 0.75
            MIN_SIDE = 10
            MAX_SIDE = 80

            detections = []
            for box, conf in zip(b_nms, c_nms):
                x1, y1, x2, y2 = box
                w = x2 - x1;  h = y2 - y1
                sq = min(w, h) / max(w, h) if max(w, h) > 0 else 0
                if sq >= MIN_SQ and MIN_SIDE <= w <= MAX_SIDE and MIN_SIDE <= h <= MAX_SIDE:
                    # Scale coordinates back to original (pre-upsample) pixel space
                    detections.append({
                        "type":       "column",
                        "bbox":       [
                            float(x1 * coord_scale), float(y1 * coord_scale),
                            float(x2 * coord_scale), float(y2 * coord_scale),
                        ],
                        "confidence": float(conf),
                    })

            logger.info(
                f"YOLO tiling: {len(raw_boxes)} raw → {len(keep)} after NMS "
                f"→ {len(detections)} columns after filter"
            )
            return detections

        except Exception as e:
            logger.warning(f"YOLO inference failed: {e} — continuing without detections")
            return []

    def _format_for_geometry(self, detections: list) -> dict:
        """
        Convert flat YOLO detections (pixel-space bboxes) into the structured
        dict that geometry_generator and semantic_analyzer expect.

        Element positions stay in pixel coordinates here; the geometry
        generator will convert them to real-world mm using the grid.
        """
        output = {"walls": [], "doors": [], "windows": [], "columns": [], "rooms": []}

        for det in detections:
            el_type = det.get("type", "").lower().rstrip("s")  # normalise plural
            bbox    = det.get("bbox", [0.0, 0.0, 0.0, 0.0])

            if len(bbox) < 4:
                continue

            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            w  = abs(x2 - x1)
            h  = abs(y2 - y1)

            base = {
                "id":         len(output.get(el_type + "s", [])),
                "confidence": det.get("confidence", 0.0),
                "center":     [cx, cy],
                "bbox":       bbox,          # pixel coords, grid-snapped downstream
            }

            if el_type == "wall":
                # Infer wall axis from aspect ratio
                if w >= h:
                    endpoints = [[x1, cy], [x2, cy]]
                    thickness = h
                else:
                    endpoints = [[cx, y1], [cx, y2]]
                    thickness = w
                base.update({"endpoints": endpoints, "thickness": thickness})
                output["walls"].append(base)

            elif el_type == "door":
                output["doors"].append(base)

            elif el_type == "window":
                output["windows"].append(base)

            elif el_type == "column":
                base["dimensions"] = {"width_px": w, "height_px": h}
                output["columns"].append(base)

            elif el_type == "room":
                output["rooms"].append(base)

        return output

    def _annotate_columns_from_vector_text(
        self,
        detections: dict,
        vector_data: dict,
        image_data: dict,
        extra_schedule_texts: list | None = None,
    ) -> dict:
        """
        Parse column type marks and dimensions from PDF text annotations.

        Structural drawings typically place column sizes in THREE ways:
          A) Inline labels next to each column symbol: "C1 800×800"
          B) A schedule/table anywhere on the sheet (same page or other pages):
                 C1  800×800
                 C2  600×600
             …with the column symbol only showing "C1" without dimensions.
          C) Numbers rendered as vector strokes — invisible to PyMuPDF's text
             layer.  For these, Pass 3 crops the raster image around the
             element and asks the vision LLM to read the annotation directly.

        This method handles both with a two-pass strategy:

        Pass 1 — Global schedule scan (whole page + extra schedule pages):
            Scan every text item on page 0 AND any text from schedule-classified
            pages (extra_schedule_texts).  Any item containing BOTH a column
            type mark (e.g. "C1") AND a dimension pattern ("800×800" / "Ø200")
            is added to a lookup table: { "C1": (800, 800, False), … }

        Pass 2 — Per-column annotation:
            For each detected column, try in order:
            a) Proximity search within 3× the column bbox — finds inline labels
            b) Proximity search for a bare type mark (without dimensions), then
               look up dimensions in the global schedule table

        Pass 3 — Vision LLM crop (fallback when text layer is missing):
            For each column still unresolved after Pass 2, crop the raster image
            around the element and ask the vision LLM to read the annotation.
            Results are cached by type_mark so similar columns pay only one API
            call.  Capped at MAX_LLM_CALLS distinct calls per run.

        Pass 4 — Single‑scheme fallback:
            If exactly one definition exists in the global schedule, assign it to
            every column that still has no type mark (i.e. unresolved after all
            above steps).  This catches cases where every column shares the same
            type (e.g. all are RCB1 800×800) but individual annotations were
            missing or missed.
        """
        _SAFE_DEFAULT_MM = 800.0   # fallback when all annotation passes fail

        text_items = vector_data.get("text", [])
        page_rect  = vector_data.get("page_rect", [0, 0, 595, 842])
        img_w      = image_data.get("width", 1)
        img_h      = image_data.get("height", 1)

        pt_w = page_rect[2] - page_rect[0]
        pt_h = page_rect[3] - page_rect[1]
        sx   = img_w / pt_w if pt_w > 0 else 1.0
        sy   = img_h / pt_h if pt_h > 0 else 1.0

        # Project all text items into pixel space once
        text_px = []
        for t in text_items:
            bx = t.get("bbox", [0, 0, 0, 0])
            cx = (bx[0] + bx[2]) / 2 * sx
            cy = (bx[1] + bx[3]) / 2 * sy
            text_px.append((cx, cy, t["text"]))

        # ── Pass 1: Build global type-mark → dimension table ──────────────────
        # Handles schedule tables / legends anywhere on the drawing.
        schedule: dict[str, tuple] = {}   # { "C1": (w_mm, d_mm, is_circ) }
        for _, _, txt in text_px:
            mark_m = _RE_MARK.search(txt)
            if not mark_m:
                continue
            mark = mark_m.group(1)
            if _BEAM_MARK_PREFIX.match(mark):
                continue
            rect_m = _RE_RECT.search(txt)
            circ_m = _RE_CIRC.search(txt)
            if rect_m and mark not in schedule:
                schedule[mark] = (float(rect_m.group(1)), float(rect_m.group(2)), False)
            elif circ_m and mark not in schedule:
                diam = float(circ_m.group(1) or circ_m.group(2))
                schedule[mark] = (diam, diam, True)

        # ── Pass 1 (supplement): Also scan text from schedule pages ───────────
        # These come from extra_schedule_texts (pages 1+ of the PDF classified
        # as column schedules).  We only need the text strings here, not pixel
        # coordinates, because the schedule table is on a different page.
        for txt in (extra_schedule_texts or []):
            mark_m = _RE_MARK.search(txt)
            if not mark_m:
                continue
            mark   = mark_m.group(1)
            if _BEAM_MARK_PREFIX.match(mark):
                continue
            rect_m = _RE_RECT.search(txt)
            circ_m = _RE_CIRC.search(txt)
            if rect_m and mark not in schedule:
                schedule[mark] = (float(rect_m.group(1)), float(rect_m.group(2)), False)
            elif circ_m and mark not in schedule:
                diam = float(circ_m.group(1) or circ_m.group(2))
                schedule[mark] = (diam, diam, True)

        if schedule:
            logger.info(
                f"Column schedule scan found {len(schedule)} type definition(s): "
                + ", ".join(f"{k}={'×'.join(str(int(v)) for v in schedule[k][:2])}" for k in sorted(schedule))
            )

        # ── Pass 2: Annotate each detected column ──────────────────────────────
        columns      = copy.deepcopy(detections.get("columns", []))
        by_proximity = 0
        by_schedule  = 0
        by_llm       = 0
        by_default   = 0
        by_schedule_fallback = 0   # new counter for single‑scheme fallback

        # Pass 3 state — shared across the column loop.
        # Cap is configurable via VISION_ANNOTATION_MAX_CALLS env var so that as
        # element types expand (beams, MEP, embedded plates, etc.) the budget can
        # be raised without code changes.  Default 10 is intentionally conservative
        # for cost control; raise to 50-100 for production accuracy.
        MAX_LLM_CALLS = int(os.getenv("VISION_ANNOTATION_MAX_CALLS", "10"))
        llm_calls     = 0
        llm_cache: dict = {}   # resolved_mark → {"w", "d", "is_circ"}
        img_np = image_data.get("image")  # numpy array; None for scanned PDFs

        def _make_crop(col_dict):
            """Crop raster image around column with 2.5× padding.  Returns PIL Image or None."""
            if img_np is None:
                return None
            from PIL import Image as _PIL
            img_h_f, img_w_f = img_np.shape[:2]
            bbox = col_dict.get("bbox", [])
            cx_f, cy_f = col_dict.get("center", [0.0, 0.0])
            bw_f = max(abs(bbox[2] - bbox[0]), 50) if len(bbox) >= 4 else 50
            bh_f = max(abs(bbox[3] - bbox[1]), 50) if len(bbox) >= 4 else 50
            pad_f = max(bw_f, bh_f) * 2.5
            x0_f = max(0, int(cx_f - bw_f / 2 - pad_f))
            y0_f = max(0, int(cy_f - bh_f / 2 - pad_f))
            x1_f = min(img_w_f, int(cx_f + bw_f / 2 + pad_f))
            y1_f = min(img_h_f, int(cy_f + bh_f / 2 + pad_f))
            if x1_f - x0_f < 20 or y1_f - y0_f < 20:
                return None
            return _PIL.fromarray(img_np[y0_f:y1_f, x0_f:x1_f])

        def _apply(col, w, d, is_circ, mark=None):
            col["width_mm"]    = w
            col["depth_mm"]    = d
            col["is_circular"] = is_circ
            if is_circ:
                col["diameter_mm"] = w
            if mark:
                col["type_mark"] = mark

        unresolved = []   # columns that still have no type mark after all attempts

        for col in columns:
            cx, cy = col.get("center", [0.0, 0.0])
            bbox   = col.get("bbox", [0, 0, 0, 0])
            col_w  = abs(bbox[2] - bbox[0]) if len(bbox) >= 4 else 100
            col_h  = abs(bbox[3] - bbox[1]) if len(bbox) >= 4 else 100

            # Wider search than before: 3× the bbox, minimum 200 px
            search_r = max(col_w, col_h, 200) * 3.0

            nearby = sorted(
                [(math.hypot(tx - cx, ty - cy), txt)
                 for tx, ty, txt in text_px
                 if math.hypot(tx - cx, ty - cy) < search_r],
                key=lambda x: x[0],
            )

            matched = False

            # (a) Proximity — text item contains both mark and dimensions
            for _, txt in nearby:
                if _is_beam_label(txt):
                    continue
                rect_m = _RE_RECT.search(txt)
                if rect_m:
                    mark = _RE_MARK.search(txt)
                    _apply(col,
                           float(rect_m.group(1)), float(rect_m.group(2)),
                           False, mark.group(1) if mark else None)
                    by_proximity += 1
                    matched = True
                    break
                circ_m = _RE_CIRC.search(txt)
                if circ_m:
                    diam = float(circ_m.group(1) or circ_m.group(2))
                    mark = _RE_MARK.search(txt)
                    _apply(col, diam, diam, True, mark.group(1) if mark else None)
                    by_proximity += 1
                    matched = True
                    break

            if matched:
                continue

            # (b) Proximity — nearby text has only a type mark → look up schedule
            for _, txt in nearby:
                if _is_beam_label(txt):
                    continue
                mark_m = _RE_MARK.search(txt)
                if mark_m:
                    mark = mark_m.group(1)
                    if _BEAM_MARK_PREFIX.match(mark):
                        continue
                    if mark in schedule:
                        w, d, is_circ = schedule[mark]
                        _apply(col, w, d, is_circ, mark)
                        by_schedule += 1
                        matched = True
                        break

            if matched:
                continue

            # ── Pass 3: Vision LLM crop ────────────────────────────────────────
            # Text layer had nothing — ask the vision model to read the pixels.
            if llm_calls < MAX_LLM_CALLS:
                tm = col.get("type_mark")
                # Cache hit: same type mark already resolved by an earlier LLM call
                if tm and tm in llm_cache:
                    cached = llm_cache[tm]
                    _apply(col, cached["w"], cached["d"], cached["is_circ"], tm)
                    by_llm += 1
                    matched = True
                else:
                    crop = _make_crop(col)
                    if crop is not None:
                        llm_calls += 1
                        result_ann = self.semantic_ai.read_element_annotation(crop)
                        resolved_mark = result_ann.get("type_mark") or tm

                        if result_ann.get("is_circular") and result_ann.get("diameter_mm"):
                            w = d = float(result_ann["diameter_mm"])
                            _apply(col, w, d, True, resolved_mark)
                            cache_key = resolved_mark or f"_llm{llm_calls}"
                            llm_cache[cache_key] = {"w": w, "d": d, "is_circ": True}
                            by_llm += 1
                            matched = True

                        elif result_ann.get("width_mm") and result_ann.get("depth_mm"):
                            w = float(result_ann["width_mm"])
                            d = float(result_ann["depth_mm"])
                            _apply(col, w, d, False, resolved_mark)
                            cache_key = resolved_mark or f"_llm{llm_calls}"
                            llm_cache[cache_key] = {"w": w, "d": d, "is_circ": False}
                            by_llm += 1
                            matched = True

                        else:
                            logger.debug(
                                f"LLM crop returned no dimensions for column "
                                f"{col.get('id')} (mark={tm}) — column remains unresolved."
                            )

            if not matched:
                # Column still has no type mark – remember it for later fallback
                unresolved.append(col)

        # ── Pass 4: Single‑scheme fallback ────────────────────────────────────
        # If exactly one definition exists in the schedule, apply it to all
        # unresolved columns.  This catches homogeneous designs where every column
        # is the same type (e.g. all RCB1 800×800) but individual annotations were
        # missing.
        if unresolved and len(schedule) == 1:
            single_type, (w, d, is_circ) = next(iter(schedule.items()))
            logger.info(
                f"Applying single schedule type '{single_type}' ({w:.0f}×{d:.0f}mm) "
                f"to {len(unresolved)} unresolved columns"
            )
            for col in unresolved:
                _apply(col, w, d, is_circ, single_type)
            by_schedule_fallback = len(unresolved)
            unresolved = []   # all resolved

        # ── Pass 5: Safe structural default ───────────────────────────────────
        # Any column still unresolved after all passes gets the safe default.
        for col in unresolved:
            _apply(col, _SAFE_DEFAULT_MM, _SAFE_DEFAULT_MM, False)
            by_default += 1

        if llm_calls >= MAX_LLM_CALLS:
            remaining = sum(1 for c in columns if "width_mm" not in c)
            if remaining:
                logger.warning(
                    f"Vision LLM cap reached ({MAX_LLM_CALLS} calls) — "
                    f"{remaining} element(s) still unresolved will use the safe "
                    f"structural default ({_SAFE_DEFAULT_MM:.0f} mm). "
                    f"For higher accuracy (e.g. cost estimation), set env var "
                    f"VISION_ANNOTATION_MAX_CALLS={MAX_LLM_CALLS * 5} or higher."
                )

        total = len(columns)
        logger.info(
            f"Column annotation: {by_proximity} proximity, "
            f"{by_schedule} via schedule table, "
            f"{by_llm} via vision LLM ({llm_calls} API call(s)), "
            f"{by_schedule_fallback} via single‑scheme fallback, "
            f"{by_default} defaulted to {_SAFE_DEFAULT_MM:.0f}mm "
            f"(total {total})"
        )

        result = dict(detections)
        result["columns"] = columns
        return result

    def _apply_revit_corrections(self, recipe: dict, corrections: dict) -> dict:
        """
        Apply AI-suggested corrections to the transaction recipe dict.

        Each correction item:
          { "element_type": "columns", "element_index": 0,
            "field": "width", "new_value": 400 }

        Only numeric fields on known element types are patched to prevent
        the AI from corrupting structural keys like "level" or "id".
        """
        import copy
        patched = copy.deepcopy(recipe)
        allowed_types  = {"columns", "walls", "doors", "windows", "floors"}
        numeric_fields = {"width", "depth", "height", "thickness",
                          "elevation", "area_sqm", "sill_height"}

        for corr in corrections.get("corrections", []):
            el_type = corr.get("element_type", "")
            idx     = corr.get("element_index")
            field   = corr.get("field", "")
            value   = corr.get("new_value")

            if el_type not in allowed_types:
                logger.debug(f"Skipping correction for unknown type '{el_type}'")
                continue
            if field not in numeric_fields:
                logger.debug(f"Skipping non-numeric field correction '{field}'")
                continue
            if not isinstance(value, (int, float)):
                continue

            elements = patched.get(el_type, [])
            if isinstance(idx, int) and 0 <= idx < len(elements):
                old = elements[idx].get(field)
                elements[idx][field] = value
                logger.info(
                    f"Correction applied: {el_type}[{idx}].{field} "
                    f"{old} → {value}"
                )

        return patched

    def _validate_recipe(self, recipe: dict) -> list:
        """
        Pre-clash validation: scan the geometry recipe for common structural
        problems before sending to Revit.

        Returns a list of human-readable warning strings (empty → all good).
        Warnings are informational — the pipeline always continues.
        """
        warnings = []

        # ── Columns: minimum 200 mm ──────────────────────────────────────────
        for idx, col in enumerate(recipe.get("columns", [])):
            w = col.get("width", 200)
            d = col.get("depth", 200)
            if w < 200 or d < 200:
                warnings.append(
                    f"Column {idx}: {w:.0f}×{d:.0f} mm is below the 200 mm safe "
                    "minimum for Revit column families — may be auto-deleted."
                )

        # ── Walls: very short walls are likely phantom detections ────────────
        for idx, wall in enumerate(recipe.get("walls", [])):
            sp = wall.get("start_point") or {}
            ep = wall.get("end_point")   or {}
            dx = ep.get("x", 0) - sp.get("x", 0)
            dy = ep.get("y", 0) - sp.get("y", 0)
            length = math.hypot(dx, dy)
            if 0 < length < 100:
                warnings.append(
                    f"Wall {idx}: very short ({length:.0f} mm). "
                    "Possible phantom detection — consider deleting."
                )

        # ── Doors: unusually wide openings may be misclassified walls ───────
        for idx, door in enumerate(recipe.get("doors", [])):
            w = door.get("width", 900)
            if w > 2000:
                warnings.append(
                    f"Door {idx}: width {w:.0f} mm > 2000 mm is unusually large. "
                    "May be misclassified — review in the 3D editor."
                )

        # ── Windows: similarly flag oversized openings ───────────────────────
        for idx, win in enumerate(recipe.get("windows", [])):
            w = win.get("width", 1200)
            if w > 3000:
                warnings.append(
                    f"Window {idx}: width {w:.0f} mm > 3000 mm is unusually large. "
                    "May be misclassified — review in the 3D editor."
                )

        if warnings:
            logger.warning(
                f"Pre-clash validation: {len(warnings)} issue(s) found"
            )
            for w in warnings:
                logger.warning(f"  • {w}")
        else:
            logger.info("Pre-clash validation: no issues found ✓")

        return warnings

    # ------------------------------------------------------------------
    # Human-in-the-loop rebuild (fast path: glTF only; slow path: + RVT)
    # ------------------------------------------------------------------

    async def rebuild_gltf(self, job_id: str) -> str:
        """
        Re-export glTF from the stored recipe after user corrections.
        Fast path — no YOLO, no AI, no Revit call (~1-2 s).
        """
        transaction_path = f"data/models/rvt/{job_id}_transaction.json"
        if not Path(transaction_path).exists():
            raise FileNotFoundError(f"Recipe not found: {transaction_path}")
        with open(transaction_path) as f:
            recipe = json.load(f)
        gltf_path = f"data/models/gltf/{job_id}.glb"
        await self.gltf_exporter.export(recipe, gltf_path)
        logger.info(f"glTF rebuilt for job {job_id}")
        return gltf_path

    async def rebuild_rvt(self, job_id: str, pdf_filename: str = "") -> str:
        """
        Send the (user-corrected) on-disk recipe to the Revit server.
        Slow path — triggers the full Windows Revit build.
        """
        transaction_path = f"data/models/rvt/{job_id}_transaction.json"
        if not Path(transaction_path).exists():
            raise FileNotFoundError(f"Recipe not found: {transaction_path}")
        rvt_path, warnings = await self.rvt_exporter.export(transaction_path, job_id, pdf_filename)
        if warnings:
            logger.warning(f"Revit rebuild warnings for {job_id}: {warnings}")
        logger.info(f"RVT rebuilt for job {job_id}")
        return rvt_path