"""
FloorPlanPipeline — thin wrapper around PipelineOrchestrator.
Called by api/routes.py background tasks.
"""

from typing import Dict, Any, Optional, Callable
from loguru import logger

from backend.services.core.orchestrator import PipelineOrchestrator


class FloorPlanPipeline:
    """Delegates to the Hybrid AI Orchestrator and forwards progress callbacks."""

    def __init__(self):
        self.orchestrator = PipelineOrchestrator()

    async def process(
        self,
        pdf_path: str,
        job_id: str,
        project_name: Optional[str] = None,
        pdf_filename: str = "",
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> Dict[str, Any]:
        return await self.orchestrator.run_pipeline(
            pdf_path,
            job_id,
            project_name or "Project",
            pdf_filename=pdf_filename,
            progress_callback=progress_callback,
        )
