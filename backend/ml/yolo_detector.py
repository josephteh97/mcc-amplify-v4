# YOLO detection is performed inline inside PipelineOrchestrator._run_yolo()
# (backend/services/core/orchestrator.py) using the ultralytics library directly.
#
# This file is intentionally minimal — it exists as a placeholder for a future
# standalone YoloDetector class if the detection logic needs to be reused or
# tested independently of the full orchestrator.
