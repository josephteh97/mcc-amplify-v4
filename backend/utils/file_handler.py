"""
File Handling Utilities
"""

import aiofiles
from pathlib import Path
from fastapi import UploadFile
from loguru import logger

_CHUNK_SIZE = 1024 * 1024   # 1 MB — keeps peak RAM from doubling for large uploads
_PDF_MAGIC  = b"%PDF"


async def save_upload_file(file: UploadFile, job_id: str) -> Path:
    """
    Stream an uploaded file to disk in 1 MB chunks and validate it is a real PDF.

    Args:
        file:   FastAPI UploadFile object
        job_id: Unique job identifier

    Returns:
        Path to the saved file

    Raises:
        ValueError: if the saved file is not a valid PDF (wrong magic bytes)
    """
    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)

    ext       = Path(file.filename).suffix or ".pdf"
    file_path = upload_dir / f"{job_id}{ext}"

    try:
        async with aiofiles.open(file_path, "wb") as out_file:
            while True:
                chunk = await file.read(_CHUNK_SIZE)
                if not chunk:
                    break
                await out_file.write(chunk)

        logger.info(f"Saved upload to {file_path} ({file_path.stat().st_size:,} bytes)")

        # Validate PDF magic bytes (%PDF) — catches files renamed to .pdf
        with open(file_path, "rb") as f:
            header = f.read(4)
        if header != _PDF_MAGIC:
            file_path.unlink(missing_ok=True)
            raise ValueError(
                f"Uploaded file is not a valid PDF (got: {header!r}). "
                "Please upload a genuine PDF floor plan."
            )

        return file_path

    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Failed to save upload: {e}")
        file_path.unlink(missing_ok=True)
        raise
