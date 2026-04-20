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
import json
import math
import os
from pathlib import Path
from typing import Callable, Optional
from loguru import logger

# Extracted modules
from backend.services.column_annotator import annotate_columns
from backend.services.yolo_runner import load_yolo, run_yolo

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
from backend.services.intelligence.cross_element_validator import validate_elements, OFF_GRID
from backend.services.intelligence.validation_agent import enforce_rules
from backend.services.intelligence.bim_translator_enricher import enrich_recipe

# ── Observer (fire-and-forget event bus for chat agent) ──────────────────────
from backend.chat_agent.pipeline_observer import observer


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
        yolo_path = Path(__file__).parent.parent.parent / "ml" / "weights" / "column-detect.pt"
        self.yolo = load_yolo(yolo_path)

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

        def emit(coro):
            """Fire-and-forget observer emission — never blocks the pipeline."""
            asyncio.create_task(coro)

        logger.info(f"🚀 Starting Hybrid Pipeline — Job {job_id}")
        monitor = ResourceMonitor()
        monitor.start()

        try:
            # ── Stage 1: Security Check ────────────────────────────────────────
            emit(observer.stage_started(job_id, 1, "Security & size check"))
            progress(5, "Security & size check…")
            secure_context = await self.security.safe_render(pdf_path)
            safe_dpi = secure_context.get("dpi", 150)
            emit(observer.stage_completed(job_id, 1, {"dpi": safe_dpi, "method": secure_context.get("method")}))

            # ── Stage 2: Dual-track PDF extraction (parallel) ─────────────────
            # Track A (vector geometry), multi-page schedule text scan, and
            # Track B (300 DPI raster render) are all I/O-bound PDF reads
            # that don't depend on each other.  Run them concurrently via
            # asyncio.gather to cut PDF extraction wall-time roughly in half
            # on multi-page drawing sets.  300 DPI keeps 800 mm columns at
            # 1:400 at ~24 px — the scale YOLO was trained on; the renderer's
            # MAX_PIXELS cap auto-reduces DPI for A0 sheets (~127 DPI) and
            # run_yolo handles that by upsampling.
            emit(observer.stage_started(job_id, 2, "Dual-track extraction (vector + raster)"))
            progress(15, "Dual-track: vector + schedule scan + raster render…")
            vector_data, extra_pages, image_data = await asyncio.gather(
                asyncio.to_thread(self.vector_processor.extract, pdf_path),
                asyncio.to_thread(self.vector_processor.extract_all_pages_text, pdf_path),
                self.stream_processor.render_safe(pdf_path, dpi=300),
            )

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
                emit(observer.warn(job_id, "scanned_pdf", {"vector_path_count": path_count}))

            # ── Multi-page schedule merge ─────────────────────────────────────
            # Structural drawing sets often put the column schedule on a
            # separate page.  Pass 1 of the column annotation step uses the
            # type-mark → dimension entries harvested above so it still works
            # even when the floor plan is on page 0.
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

            emit(observer.stage_completed(job_id, 2, {
                "vector_paths": path_count,
                "schedule_pages": sum(1 for p in extra_pages if p["is_schedule"]),
                "is_scanned": is_scanned,
            }))

            # ── Stage 3: YOLO element detection on rendered image ─────────────
            emit(observer.stage_started(job_id, 3, "YOLO element detection"))
            progress(35, "Detecting elements (YOLO)…")
            ml_detections = run_yolo(self.yolo, image_data)

            # ── Checkpoint: save render + detections for YOLO training ─────────
            self._save_job_checkpoint(job_id, "render.jpg", image_data["image"])
            self._save_job_checkpoint(job_id, "px_detections.json", ml_detections)

            # Report detection counts by type for the chat agent
            _det_counts: dict[str, int] = {}
            for _d in ml_detections:
                _t = _d.get("type", "unknown")
                _det_counts[_t] = _det_counts.get(_t, 0) + 1
            for _t, _n in _det_counts.items():
                emit(observer.element_detected(job_id, _t, _n))
            emit(observer.stage_completed(job_id, 3, {"yolo_total": len(ml_detections), "by_type": _det_counts}))

            # ── Stage 3: Hybrid Fusion ─────────────────────────────────────────
            emit(observer.stage_started(job_id, 4, "Hybrid fusion (vector + ML)"))
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
            emit(observer.stage_completed(job_id, 4, {"refined_count": len(refined_detections)}))

            # ── Stage 4: Grid Detection ────────────────────────────────────────
            # We derive the real-world coordinate system from structural grid
            # lines and their dimension annotations — never from scale text.
            emit(observer.stage_started(job_id, 5, "Grid detection"))
            progress(55, "Detecting structural grid lines…")
            grid_info = self._detect_grid(vector_data, image_data)
            if grid_info.get("source") in ("fallback", "uniform_fallback"):
                emit(observer.warn(job_id, "fallback_grid", {
                    "source": grid_info.get("source"),
                    "confidence": grid_info.get("grid_confidence", 0.0),
                }))

            # ── Stage 4b: Align grid pixel reference to YOLO column centres ──────
            # The structural grid datum (mm spacings) comes from vector-detected
            # dashed centre lines + PDF dimension annotations.  The PIXEL positions
            # of those lines can be 10–40 px off from the rendered column centres
            # due to rasterisation differences, which causes a world-coordinate
            # offset of hundreds of mm (e.g. 30 px / 280 px × 8400 mm ≈ 900 mm).
            #
            # Align grid pixel positions to YOLO column centres while
            # keeping all mm spacings from the PDF vector layer intact.
            column_raw = [d for d in refined_detections if d.get("type") == "column"]
            if len(column_raw) >= 2:
                grid_info = self.grid_detector.align_pixels_to_columns(
                    grid_info, column_raw
                )
                logger.info(
                    f"Grid pixel alignment complete — grid is PDF-authoritative "
                    f"({len(grid_info.get('x_lines_px',[]))} V × "
                    f"{len(grid_info.get('y_lines_px',[]))} H lines kept from PDF)."
                )
            emit(observer.stage_completed(job_id, 5, {
                "grid_source": grid_info.get("source"),
                "x_lines": len(grid_info.get("x_lines_px", [])),
                "y_lines": len(grid_info.get("y_lines_px", [])),
                "confidence": grid_info.get("grid_confidence", 0.0),
            }))

            # ── Stage 4c: Intelligence middleware (post-detection, pre-geometry) ──
            emit(observer.stage_started(job_id, 6, "Intelligence middleware"))
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

            # ── Off-grid column deletion (Validation Agent enforcement) ─────
            # A column whose pixel centre is farther than max_grid_dist_px from
            # every grid line is almost certainly a YOLO false positive.  User
            # directive: "Place wrongly is worse than missing one column."
            # We delete rather than flag.  The dict identity is shared between
            # _column_dets and refined_detections (resolve_types mutates in
            # place), so the off_grid flag is already visible on both.
            if _column_dets:
                before = len(refined_detections)
                refined_detections = [
                    d for d in refined_detections
                    if not (
                        d.get("type") == "column"
                        and OFF_GRID in d.get("validation_flags", [])
                    )
                ]
                _column_dets = [
                    d for d in _column_dets
                    if OFF_GRID not in d.get("validation_flags", [])
                ]
                deleted = before - len(refined_detections)
                if deleted:
                    logger.warning(
                        f"🗑️  Deleted {deleted} off-grid column(s) — "
                        "Validation Agent rejected (columns cannot be outside the grid)."
                    )
                    emit(observer.warn(job_id, "off_grid_columns_deleted", {"count": deleted}))
            emit(observer.stage_completed(job_id, 6, {
                "column_dets_kept": len(_column_dets),
                "dfma_violations": sum(1 for d in _column_dets if not d.get("is_dfma_compliant", True)),
            }))

            # ── Stage 5: Semantic AI Analysis ─────────────────────────────────
            # Build structured element dict from pixel-space detections
            # so the geometry generator can snap them to the grid.
            emit(observer.stage_started(job_id, 7, "Semantic AI analysis"))
            progress(60, "AI semantic analysis…")
            structured_elements = self._format_for_geometry(refined_detections)

            # ── Stage 3b: Column annotation parsing ───────────────────────────
            # Match PDF text labels (e.g. "C1 800x800", "C20 Ø200") to the
            # YOLO-detected columns so geometry_generator uses real dimensions.
            # Must run AFTER _format_for_geometry() so structured_elements is
            # a dict with a "columns" key, not a raw list from YOLO.
            structured_elements = annotate_columns(
                structured_elements, vector_data, image_data,
                extra_schedule_texts=schedule_page_texts,
                semantic_ai=self.semantic_ai,
            )
            enriched_data = await self.semantic_ai.analyze(
                image_data,
                structured_elements,
                grid_info,
            )
            # ── Checkpoint: save enriched data for debugging / re-runs ──────────
            self._save_job_checkpoint(job_id, "enriched.json", enriched_data)
            emit(observer.stage_completed(job_id, 7, {"enriched_elements": len(enriched_data) if hasattr(enriched_data, "__len__") else None}))

            # ── Stage 6: 3D Geometry Generation ───────────────────────────────
            # Apply project profile defaults before generating geometry so that
            # user-configured wall heights, storey heights, etc. are respected.
            emit(observer.stage_started(job_id, 8, "3D geometry generation"))
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

            # ── Deduplicate columns that snapped to the same grid intersection ─
            # Done AFTER enrich_recipe so intelligence metadata is already merged.
            # Revit rejects identical-location instances ("identical instances in
            # the same place" warning). Round to 1 dp to handle float near-equals.
            _cols = recipe.get("columns", [])
            if _cols:
                _seen: set = set()
                _unique = []
                for _c in _cols:
                    _loc = _c.get("location", {})
                    _key = (round(_loc.get("x", 0), 1), round(_loc.get("y", 0), 1))
                    if _key not in _seen:
                        _seen.add(_key)
                        _unique.append(_c)
                if len(_unique) < len(_cols):
                    logger.info(
                        f"Deduplicated {len(_cols)} → {len(_unique)} columns after grid snap"
                    )
                    recipe["columns"] = _unique

            # ── Pre-clash validation ───────────────────────────────────────────
            validation_warnings = self._validate_recipe(recipe)
            if validation_warnings:
                emit(observer.warn(job_id, "pre_clash_validation", {
                    "count": len(validation_warnings),
                    "warnings": validation_warnings[:10],  # cap payload size
                }))
            emit(observer.stage_completed(job_id, 8, {
                "columns": len(recipe.get("columns", [])),
                "walls": len(recipe.get("walls", [])),
                "doors": len(recipe.get("doors", [])),
                "windows": len(recipe.get("windows", [])),
                "pre_clash_warnings": len(validation_warnings),
            }))

            # ── Stage 7: BIM Export ────────────────────────────────────────────
            emit(observer.stage_started(job_id, 9, "BIM export (RVT + glTF)"))
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
            # rvt_status: "success" | "warnings_accepted" | "skipped" | "failed"
            rvt_status  = "skipped"
            rvt_warnings_final: list = []
            _use_agent  = os.getenv("USE_AGENT_BUILDER", "").lower() == "true"
            try:
                if _use_agent:
                    rvt_path, vision_diff = await self._run_agent_export(recipe, job_id, progress, pdf_filename)
                    rvt_status = "success" if rvt_path else "failed"
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
                                rvt_warnings_final = revit_warnings
                                rvt_status = "warnings_accepted"
                                emit(observer.warn(job_id, "revit_warnings", {
                                    "attempts": _attempt + 1,
                                    "warnings": revit_warnings,
                                }))
                            else:
                                rvt_status = "success"
                            break

                        # Ask the AI what to change
                        progress(
                            90 + _attempt * 3,
                            f"Revit warnings — AI correcting (round {_attempt + 1}/3)…",
                        )
                        corrections = await self.semantic_ai.analyze_revit_warnings(
                            revit_warnings, current_recipe
                        )
                        if not corrections.get("corrections"):
                            logger.info(
                                "AI found no actionable corrections for Revit warnings "
                                f"— proceeding with current RVT: {revit_warnings}"
                            )
                            rvt_warnings_final = revit_warnings
                            rvt_status = "warnings_accepted"
                            emit(observer.warn(job_id, "revit_warnings", {
                                "attempts": _attempt + 1,
                                "warnings": revit_warnings,
                                "ai_corrections": "none_actionable",
                            }))
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
                rvt_status = "failed"
                emit(observer.warn(job_id, "rvt_export_failed", {
                    "error_type": type(rvt_err).__name__,
                    "message": str(rvt_err),
                }))

            emit(observer.stage_completed(job_id, 9, {
                "gltf": gltf_out,
                "rvt": rvt_path,
                "rvt_status": rvt_status,
            }))

            progress(100, "Complete!")
            logger.info(
                f"✅ Pipeline complete — glTF: {gltf_out} | RVT: {rvt_path or 'skipped'} ({rvt_status})"
            )

            result = {
                "job_id":     job_id,
                "status":     "completed",
                "files":      {"rvt": rvt_path, "gltf": gltf_out},
                "rvt_status": rvt_status,
                "stats":      {
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
                    "rvt_warnings":          rvt_warnings_final,
                },
            }
            emit(observer.job_completed(job_id, result))
            return result

        except Exception as e:
            import traceback
            logger.error(
                f"Pipeline failed: {type(e).__name__}: {e}\n"
                + traceback.format_exc()
            )
            emit(observer.error(job_id, type(e).__name__, {"message": str(e)}))
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