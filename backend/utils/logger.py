"""
Logging Configuration
"""

import os
import sys
from pathlib import Path

from loguru import logger

# Resolve log directory relative to this file so logs always land in
# backend/logs/ regardless of which directory the process is started from.
_BACKEND_DIR = Path(__file__).resolve().parents[1]
_DEFAULT_LOG  = str(_BACKEND_DIR / "logs" / "app.log")


def setup_logger():
    """Configure loguru — console + rotating file."""
    logger.remove()

    log_level = os.getenv("LOG_LEVEL", "INFO")

    # Console
    logger.add(
        sys.stderr,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        level=log_level,
    )

    # File — absolute path, 10 MB rotation, 1-week retention
    log_file = os.getenv("LOG_FILE", _DEFAULT_LOG)
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_file,
        rotation="10 MB",
        retention="1 week",
        level=log_level,
    )

    return logger
