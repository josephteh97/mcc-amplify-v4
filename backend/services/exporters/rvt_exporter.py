"""
Stage 7: RVT Exporter
Handles communication with Windows Revit Server to generate .RVT files
"""

from backend.services.revit_client import RevitClient

class RvtExporter:
    def __init__(self):
        self.client = RevitClient()

    async def export(
        self, transaction_path: str, job_id: str, pdf_filename: str = ""
    ) -> tuple:
        """
        Build model on Windows Revit server.

        Returns:
            (rvt_path: str, warnings: list[str])
            warnings — Revit build warnings (column too thin, etc.) captured by
                       the C# IFailuresPreprocessor; empty when all is clean.
        """
        return await self.client.build_model(transaction_path, job_id, pdf_filename)
