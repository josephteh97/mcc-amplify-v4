"""
Core Orchestrator: Agent-Based Structural Detection Pipeline

Stages executed:
  1. Security check       (services/security)
  2. Source data          (VectorProcessor + StreamingProcessor — parallel I/O)
  3. Parallel agents      (7 detection agents via asyncio.gather)
       GridDetectionAgent          — structural grid from vector geometry
       ColumnDetectionAgent        — YOLO tiling inference
       WallDetectionAgent          — vector path analysis (via fusion)
       StructuralFramingDetectionAgent — YOLO tiling inference
       StairsDetectionAgent        — stub, model pending
       LiftDetectionAgent          — stub, model pending
       SlabDetectionAgent          — stub, model pending
  4. Detection Merger     (HybridFusionPipeline + grid pixel alignment)
  4c. Intelligence layer  (TypeResolver → CrossElementValidator → ValidationAgent)
  5. BIM Enrichment       (BIMTranslatorEnricher + element deduplication)
  6. 3D Geometry          (GeometryGenerator — px → mm → Revit recipe)
  7. BIM Export           (RvtExporter + GltfExporter)

Scale is derived exclusively from structural grid lines and dimension annotations.
Scale text printed on the drawing (e.g. "1:100") is intentionally ignored.
"""

import asyncio
import gc
import json
import math
import os
from pathlib import Path
from typing import Callable, Optional
from loguru import logger

from backend.services.column_annotator import annotate_columns
from backend.services.yolo_runner import load_yolo

from backend.services.security.secure_renderer import SecurePDFRenderer, ResourceMonitor
from backend.services.pdf_processing.processors import VectorProcessor, StreamingProcessor
from backend.services.fusion.pipeline import HybridFusionPipeline
from backend.services.grid_detector import GridDetector
from backend.services.semantic_analyzer import SemanticAnalyzer
from backend.services.geometry_generator import GeometryGenerator
from backend.services.exporters.rvt_exporter import RvtExporter
from backend.services.exporters.gltf_exporter import GltfExporter
from backend.services.vision_comparator import VisionComparator

# ── Detection agents ──────────────────────────────────────────────────────────
from backend.services.detection_agents import (
    DetectionContext,
    GridDetectionAgent,
    YoloDetectionAgent,
    UntrainedDetectionAgent,
)

# ── Intelligence layer ────────────────────────────────────────────────────────
from backend.services.intelligence.type_resolver import resolve_types
from backend.services.intelligence.cross_element_validator import validate_elements
from backend.services.intelligence.validation_agent import enforce_rules, remove_outside_grid
from backend.services.intelligence.debug_overlay import (
    save_join_conflict_overlay,
    save_sanitizer_rejected_overlay,
)
from backend.services.intelligence.admittance import judge as admittance_judge
from backend.services.intelligence.admittance import ElementContext, REJECT
from backend.services.intelligence.admittance.legend_parser import parse_legend, enrich_with_vision
from backend.services.intelligence.bim_translator_enricher import enrich_recipe
from backend.services.intelligence.recipe_sanitizer import sanitize_recipe
from backend.services.revit_warning_handler import handle_warnings as handle_revit_warnings

# ── Observer (fire-and-forget event bus for chat agent) ──────────────────────
from backend.chat_agent.pipeline_observer import observer


def _drop_by_id(seq: list[dict], dropped_ids: set[int]) -> list[dict]:
    """Filter dicts out of *seq* whose Python id() is in *dropped_ids*."""
    return [d for d in seq if id(d) not in dropped_ids]


class PipelineOrchestrator:
    """
    Orchestrates: Security → Source Data → Parallel Agents → Merger
                  → Intelligence → BIM Enrichment → Geometry → Export
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

        weights = Path(__file__).parent.parent.parent / "ml" / "weights"

        # ── Detection agents (one per structural element type) ────────────────
        self.grid_agent               = GridDetectionAgent(self.grid_detector)
        self.column_agent             = YoloDetectionAgent(
            load_yolo(weights / "column-detect.pt"), "column",
        )
        self.structural_framing_agent = YoloDetectionAgent(
            load_yolo(weights / "structural-framing-detect.pt"), "structural_framing",
            min_squareness=0.0,   # beams are rectangular, not square
            max_side=300,
        )
        self.wall_agent  = UntrainedDetectionAgent("wall")    # walls extracted by fusion pipeline
        self.stairs_agent = UntrainedDetectionAgent("stairs")
        self.lift_agent   = UntrainedDetectionAgent("lift")
        self.slab_agent   = UntrainedDetectionAgent("slab")

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

            # ── Stage 2: Source data acquisition (parallel I/O) ───────────────
            emit(observer.stage_started(job_id, 2, "Source data acquisition"))
            progress(15, "Vector extraction + raster render…")
            vector_data, extra_pages, image_data = await asyncio.gather(
                asyncio.to_thread(self.vector_processor.extract, pdf_path),
                asyncio.to_thread(self.vector_processor.extract_all_pages_text, pdf_path),
                self.stream_processor.render_safe(pdf_path, dpi=300),
            )

            path_count = len(vector_data.get("paths", []))
            is_scanned = path_count < 50
            if is_scanned:
                logger.warning(
                    f"Scanned/raster PDF detected ({path_count} vector paths). "
                    "Grid detection will use fallback coordinates — accuracy reduced."
                )
                emit(observer.warn(job_id, "scanned_pdf", {"vector_path_count": path_count}))

            schedule_page_texts = [
                item["text"]
                for page in extra_pages if page["is_schedule"]
                for item in page["text_items"]
            ]
            if schedule_page_texts:
                n_sched = sum(1 for p in extra_pages if p["is_schedule"])
                logger.info(
                    f"Cross-page schedule: {len(schedule_page_texts)} text items "
                    f"from {n_sched} schedule page(s) merged."
                )
            emit(observer.stage_completed(job_id, 2, {
                "vector_paths": path_count,
                "schedule_pages": sum(1 for p in extra_pages if p["is_schedule"]),
                "is_scanned": is_scanned,
            }))

            # ── Stage 3: Parallel detection agents ────────────────────────────
            emit(observer.stage_started(job_id, 3, "Parallel element detection agents"))
            progress(30, "Running element detection agents…")
            ctx = DetectionContext(
                pdf_path=pdf_path,
                image=image_data["image"],
                image_dpi=image_data.get("dpi", safe_dpi),
                vector_data=vector_data,
                schedule_texts=schedule_page_texts,
            )
            (
                grid_info,
                col_dets,
                wall_dets,
                sf_dets,
                stair_dets,
                lift_dets,
                slab_dets,
            ) = await asyncio.gather(
                self.grid_agent.detect_grid(ctx),
                self.column_agent.detect(ctx),
                self.wall_agent.detect(ctx),
                self.structural_framing_agent.detect(ctx),
                self.stairs_agent.detect(ctx),
                self.lift_agent.detect(ctx),
                self.slab_agent.detect(ctx),
            )
            self._save_job_checkpoint(job_id, "render.jpg", image_data["image"])
            self._save_job_checkpoint(job_id, "px_detections.json", col_dets)

            _det_counts: dict[str, int] = {}
            for _dets, _lbl in (
                (col_dets,   "column"),
                (wall_dets,  "wall"),
                (sf_dets,    "structural_framing"),
                (stair_dets, "stairs"),
                (lift_dets,  "lift"),
                (slab_dets,  "slab"),
            ):
                if _dets:
                    _det_counts[_lbl] = len(_dets)
                    emit(observer.element_detected(job_id, _lbl, len(_dets)))

            if grid_info.get("source") in ("fallback", "uniform_fallback"):
                emit(observer.warn(job_id, "fallback_grid", {
                    "source": grid_info.get("source"),
                    "confidence": grid_info.get("grid_confidence", 0.0),
                }))
            emit(observer.stage_completed(job_id, 3, {"agents_run": 7, "by_type": _det_counts}))

            # Release CLAHE/PIL copies created by YOLO agents — the raw numpy
            # image is still needed downstream (resolve_types), but the
            # intermediate copies inside run_yolo have already been returned
            # as tensors and can be GC'd now.
            gc.collect()

            # ── Stage 4: Detection Merger + Parser ────────────────────────────
            emit(observer.stage_started(job_id, 4, "Detection merger + parser"))
            progress(45, "Fusing detections with vector geometry…")
            fused_data = await self.fusion.fuse(
                vector_data,
                col_dets,
                {
                    "width":  image_data["width"],
                    "height": image_data["height"],
                    "dpi":    image_data.get("dpi", safe_dpi),
                },
            )
            # Refined column detections from fusion; other agents appended as-is.
            refined_detections = (
                (fused_data.get("refined_px") or col_dets)
                + wall_dets + sf_dets + stair_dets + lift_dets + slab_dets
            )
            emit(observer.stage_completed(job_id, 4, {"refined_count": len(refined_detections)}))

            # ── Stage 4b: Align grid pixel reference to column centres ─────────
            # Grid mm spacings come from PDF vector annotations (authoritative).
            # Pixel positions can be 10–40 px off from rendered column centres
            # due to rasterisation; align them while preserving all mm spacings.
            column_raw = [d for d in refined_detections if d.get("type") == "column"]
            if len(column_raw) >= 2:
                grid_info = self.grid_detector.align_pixels_to_columns(
                    grid_info, column_raw
                )
                logger.info(
                    f"Grid pixel alignment complete — "
                    f"{len(grid_info.get('x_lines_px',[]))} V × "
                    f"{len(grid_info.get('y_lines_px',[]))} H lines (PDF-authoritative)."
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
            framing_raw = [d for d in refined_detections if d.get("type") == "structural_framing"]
            raster = image_data.get("image")
            _column_dets = []
            if column_raw and raster is not None:
                _column_dets = resolve_types(column_raw, raster)
                _column_dets = validate_elements(
                    _column_dets,
                    grid_info=grid_info,
                    max_grid_dist_px=float(os.getenv("MAX_GRID_DIST_PX", "80")),
                    isolation_radius_px=float(os.getenv("ISOLATION_RADIUS_PX", "200")),
                )

            # ── DfMA bay-spacing checks (still handled by legacy agent) ─────
            # Admittance layer does per-element judgment; grid-level DfMA
            # rules (bay min/max) remain on enforce_rules.
            # Cull before enforce_rules so violation counts don't include
            # soon-to-drop title-block / legend / border noise.
            pre_cull = _column_dets + framing_raw
            all_dets, out_of_grid_actions = remove_outside_grid(pre_cull, grid_info)
            if out_of_grid_actions:
                kept_ids = {id(d) for d in all_dets}
                dropped_ids = {id(d) for d in pre_cull if id(d) not in kept_ids}
                refined_detections = _drop_by_id(refined_detections, dropped_ids)
                _column_dets = _drop_by_id(_column_dets, dropped_ids)
                emit(observer.warn(job_id, "outside_grid_culled", {
                    "count": len(out_of_grid_actions),
                    "examples": out_of_grid_actions[:5],
                }))

            enforce_rules(
                all_dets,
                grid_info=grid_info,
                min_bay_mm=float(os.getenv("MIN_BAY_MM", "3000")),
                max_bay_mm=float(os.getenv("MAX_BAY_MM", "12000")),
            )

            # ── Admittance framework: per-element triage ──────────────────────
            legend_map = parse_legend(vector_data)
            legend_map = enrich_with_vision(legend_map, raster, self.semantic_ai)
            page_rect = vector_data.get("page_rect", [0, 0, 0, 0])
            ctx = ElementContext(
                vector_data=vector_data,
                grid_info=grid_info,
                legend_map=legend_map,
                raster=raster,
                page_width_pt=page_rect[2] - page_rect[0],
                page_height_pt=page_rect[3] - page_rect[1],
                dpi=float(image_data.get("dpi", safe_dpi)),
            )
            admittance_judge(all_dets, ctx)

            if raster is not None:
                save_join_conflict_overlay(
                    raster, all_dets, f"data/debug/{job_id}_join_conflicts.png",
                )

            rejected_ids = {
                id(d) for d in all_dets
                if (d.get("admittance_decision") or {}).get("action") == REJECT
            }
            before = len(refined_detections)
            refined_detections = _drop_by_id(refined_detections, rejected_ids)
            _column_dets = _drop_by_id(_column_dets, rejected_ids)
            deleted = before - len(refined_detections)
            if deleted:
                rejected = [d for d in all_dets if id(d) in rejected_ids]
                lines = []
                for d in rejected:
                    dec = d.get("admittance_decision") or {}
                    c = d.get("center") or []
                    cxy = f"({c[0]:.0f},{c[1]:.0f})" if len(c) >= 2 else "?"
                    lines.append(
                        f"   • {d.get('type','?')} id={d.get('id','?')} @{cxy} — {dec.get('reason','?')}"
                    )
                logger.warning(
                    "🗑️  Admittance rejected {} element(s):\n{}", deleted, "\n".join(lines)
                )
                emit(observer.warn(job_id, "admittance_rejections", {"count": deleted}))
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
            dpi = float(image_data.get("dpi", safe_dpi))
            zone_labels_mm = [
                (code, *self.geometry_gen.pt_to_world(x, y, grid_info, dpi))
                for (code, x, y) in vector_data.get("zone_labels", [])
            ]
            recipe = await self.geometry_gen.build(
                enriched_data, grid_info,
                zone_labels_mm=zone_labels_mm,
                slab_legend=vector_data.get("slab_legend") or {},
            )

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
                "columns":            len(recipe.get("columns", [])),
                "walls":              len(recipe.get("walls", [])),
                "core_walls":         len(recipe.get("core_walls", [])),
                "structural_framing": len(recipe.get("structural_framing", [])),
                "stairs":             len(recipe.get("stairs", [])),
                "lifts":              len(recipe.get("lifts", [])),
                "slabs":              len(recipe.get("slabs", [])),
                "pre_clash_warnings": len(validation_warnings),
            }))

            # ── Pre-export sanitizer ───────────────────────────────────────────
            # Fix known geometry problems (beam-column overlaps, short beams,
            # undersized columns) before the recipe reaches the Windows machine.
            framing_before_sanitize = len(recipe.get("structural_framing", []))
            recipe, sanitizer_actions, sanitizer_rejected = sanitize_recipe(recipe)
            if sanitizer_actions:
                framing_after_sanitize = len(recipe.get("structural_framing", []))
                emit(observer.warn(job_id, "pre_export_sanitizer", {
                    "count":   len(sanitizer_actions),
                    "actions": sanitizer_actions,
                    "framing_dropped": framing_before_sanitize - framing_after_sanitize,
                    "framing_kept":    framing_after_sanitize,
                }))
            if sanitizer_rejected:
                try:
                    save_sanitizer_rejected_overlay(
                        image_data["image"],
                        sanitizer_rejected,
                        grid_info,
                        f"data/debug/{job_id}_sanitizer_rejected.png",
                    )
                except Exception as exc:
                    logger.warning("Sanitizer-rejection overlay failed: {}", exc)

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

                        progress(
                            90 + _attempt * 3,
                            f"Revit warnings — correcting (round {_attempt + 1}/3)…",
                        )

                        # ── Step 1: deterministic pattern-based handler ────────
                        current_recipe, det_actions, unresolved = handle_revit_warnings(
                            revit_warnings, current_recipe
                        )
                        if det_actions:
                            logger.info(
                                f"Deterministic corrections: {len(det_actions)} fix(es) — "
                                + ", ".join(det_actions[:3])
                                + (" …" if len(det_actions) > 3 else "")
                            )
                            emit(observer.warn(job_id, "revit_deterministic_corrections", {
                                "attempt": _attempt + 1,
                                "actions": det_actions,
                            }))
                            with open(transaction_path, "w") as f:
                                json.dump(current_recipe, f)

                            if not unresolved:
                                # All warnings handled — retry Revit without AI call
                                continue

                        # ── Step 2: AI for remaining unresolved warnings ───────
                        warnings_for_ai = unresolved if det_actions else revit_warnings
                        corrections = await self.semantic_ai.analyze_revit_warnings(
                            warnings_for_ai, current_recipe
                        )
                        if not corrections.get("corrections"):
                            logger.info(
                                "AI found no actionable corrections for Revit warnings "
                                f"— proceeding with current RVT: {warnings_for_ai}"
                            )
                            rvt_warnings_final = revit_warnings
                            rvt_status = "warnings_accepted"
                            emit(observer.warn(job_id, "revit_warnings", {
                                "attempts":        _attempt + 1,
                                "warnings":        revit_warnings,
                                "ai_corrections":  "none_actionable",
                                "det_actions":     det_actions,
                            }))
                            break

                        logger.info(
                            f"AI correction summary: {corrections.get('summary')} "
                            f"({len(corrections['corrections'])} change(s))"
                        )
                        current_recipe = self._apply_revit_corrections(
                            current_recipe, corrections
                        )
                        # Persist the combined (deterministic + AI) corrections
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
                    "yolo_detections":       len(col_dets),
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
        Convert agent detection dicts (pixel-space) into the structured dict
        that geometry_generator expects.  Positions stay in pixel coords here;
        GeometryGenerator converts them to real-world mm via the grid.

        Structural elements: column, wall, structural_framing, stairs, lift, slab.
        Rooms are retained as a fallback input for slab boundary generation
        when SlabDetectionAgent returns no detections.
        """
        output = {
            "walls":               [],
            "columns":             [],
            "structural_framing":  [],
            "stairs":              [],
            "lifts":               [],
            "slabs":               [],
            "rooms":               [],  # fallback for slab boundary generation
        }

        for det in detections:
            el_type = det.get("type", "").lower().rstrip("s")
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
                "bbox":       bbox,
            }

            if el_type == "wall":
                # Axis inferred from bounding-box aspect ratio
                if w >= h:
                    endpoints = [[x1, cy], [x2, cy]]
                    thickness = h
                else:
                    endpoints = [[cx, y1], [cx, y2]]
                    thickness = w
                base.update({"endpoints": endpoints, "thickness": thickness})
                output["walls"].append(base)

            elif el_type == "column":
                base["dimensions"] = {"width_px": w, "height_px": h}
                output["columns"].append(base)

            elif el_type == "structural_framing":
                output["structural_framing"].append(base)

            elif el_type == "stair":
                output["stairs"].append(base)

            elif el_type == "lift":
                output["lifts"].append(base)

            elif el_type == "slab":
                output["slabs"].append(base)

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
        allowed_types  = {"columns", "walls", "structural_framing", "stairs", "lifts", "slabs"}
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