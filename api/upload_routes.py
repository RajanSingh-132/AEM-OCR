"""Upload / extraction endpoints (PRD section 3)."""

import asyncio
import logging
import time
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from OCRAI.config import ALLOWED_EXTENSIONS, WORD_EXTENSIONS
from OCRAI.extraction import ExtractionError, extract_document
from OCRAI.schemas import MODULES

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1")

WORD_CONTENT_TYPES = {
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/octet-stream",
}


def validate_upload(filename: str, content_type: str | None) -> None:
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Only PDF, image, or Word files ({', '.join(sorted(ALLOWED_EXTENSIONS))}) are allowed.",
        )
    content_type = (content_type or "").split(";")[0].strip().lower()
    if not content_type:
        return
    if content_type == "application/pdf" or content_type.startswith("image/"):
        return
    if ext in WORD_EXTENSIONS and content_type in WORD_CONTENT_TYPES:
        return
    raise HTTPException(
        status_code=400,
        detail="Invalid content-type. Expected application/pdf, an image type, or a Word document type.",
    )


def validate_module(module: str) -> str:
    normalized = (module or "").strip().upper()
    if normalized not in MODULES:
        raise HTTPException(status_code=400, detail=f"module must be one of: {', '.join(MODULES)}.")
    return normalized


@router.post("/upload/pdf_dynamic_extract")
async def pdf_dynamic_extract(
    file: UploadFile = File(...),
    module: str = Form(..., description="ACE (U.S. eManifest) or ACI (Canada eManifest)", examples=["ACE"]),
):
    """Extract a PDF, image or Word file into the ACE or ACI eManifest JSON template.

    Values found in the document are filled in; values not in the document are "".
    """
    request_start = time.perf_counter()
    filename = file.filename or ""
    log.info("=" * 70)
    log.info("NEW REQUEST pdf_dynamic_extract: %s (%s), module %s", filename, file.content_type, module)
    try:
        module = validate_module(module)
        validate_upload(filename, file.content_type)
        file_bytes = await file.read()
        log.info("[1/3] Upload OK: %.2f MB", len(file_bytes) / 1024 / 1024)
        result = await asyncio.to_thread(extract_document, file_bytes, filename, module)
        log.info("DONE in %.2fs", time.perf_counter() - request_start)
        return result
    except ExtractionError as exc:
        log.warning("  FAILED in %.2fs: %s", time.perf_counter() - request_start, exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except HTTPException as exc:
        log.warning("  Rejected (%d): %s", exc.status_code, exc.detail)
        raise
    except Exception as exc:
        log.exception("  Unexpected error")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        await file.close()
        log.info("=" * 70)


