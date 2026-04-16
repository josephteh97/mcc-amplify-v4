"""
Centralized configuration — single source of truth for ports, paths, and limits.

All hardcoded values that were previously scattered across modules live here.
Import as:  from backend.config import cfg
"""

import os
from pathlib import Path


class _Config:
    """Read-once config from environment with sensible defaults."""

    # ── Network ──────────────────────────────────────────────────────────────
    APP_HOST: str = os.getenv("APP_HOST", "0.0.0.0")
    APP_PORT: int = int(os.getenv("APP_PORT", "8000"))

    REVIT_SERVER_URL: str = os.getenv("WINDOWS_REVIT_SERVER", "http://localhost:5000")
    REVIT_API_KEY: str = os.getenv("REVIT_SERVER_API_KEY", "")
    REVIT_TIMEOUT: int = int(os.getenv("REVIT_TIMEOUT", "300"))
    REVIT_MODE: str = os.getenv("REVIT_MODE", "http").lower()
    REVIT_SHARED_DIR: Path = Path(os.getenv("REVIT_SHARED_DIR", "/mnt/revit_output"))
    REVIT_HEARTBEAT_INTERVAL: int = int(os.getenv("REVIT_HEARTBEAT_INTERVAL", "30"))

    CORS_ORIGINS: list[str] = os.getenv(
        "CORS_ORIGINS", "http://localhost:5173"
    ).split(",")

    # ── Job store ────────────────────────────────────────────────────────────
    MAX_JOBS: int = int(os.getenv("MAX_JOBS", "100"))
    JOB_DB_PATH: Path = Path(os.getenv("JOB_DB_PATH", "data/jobs.db"))

    # ── PDF / rendering limits ───────────────────────────────────────────────
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "100"))
    MAX_PIXEL_COUNT: int = int(os.getenv("MAX_PIXEL_COUNT", "25000000"))
    MAX_MEMORY_MB: int = int(os.getenv("MAX_MEMORY_MB", "400"))
    MAX_PAGE_INCHES: float = float(os.getenv("MAX_PAGE_INCHES", "60"))
    RENDER_TIMEOUT: int = int(os.getenv("RENDER_TIMEOUT", "30"))

    # ── YOLO ─────────────────────────────────────────────────────────────────
    YOLO_WEIGHTS: Path = Path(
        os.getenv("YOLO_WEIGHTS_PATH", "ml/weights/column-detect.pt")
    )
    YOLO_TILE_SIZE: int = 1280
    YOLO_OVERLAP: int = 200
    YOLO_CONF: float = float(os.getenv("DETECTION_CONFIDENCE", "0.25"))
    YOLO_IOU: float = float(os.getenv("NMS_THRESHOLD", "0.45"))
    YOLO_TARGET_DPI: int = 300

    # ── Semantic AI ──────────────────────────────────────────────────────────
    SEMANTIC_RETRY_ATTEMPTS: int = int(os.getenv("SEMANTIC_RETRY_ATTEMPTS", "2"))
    SEMANTIC_RETRY_BACKOFF: float = float(os.getenv("SEMANTIC_RETRY_BACKOFF", "3.0"))

    # ── Concurrency ──────────────────────────────────────────────────────────
    MAX_CONCURRENT_JOBS: int = int(os.getenv("MAX_CONCURRENT_JOBS", "3"))


cfg = _Config()
