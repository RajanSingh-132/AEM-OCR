"""OCR AI endpoints."""

import logging
import time

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from OCRAI.config import GEMINI_MODEL, MAX_FILE_MB
from OCRAI.llm_service import GeminiError, gemini_to_json
from OCRAI.ocr import UnsupportedFileError, extract_text
from OCRAI.prompts import DEFAULT_PROMPT

log = logging.getLogger(__name__)

router = APIRouter()


async def read_upload(file: UploadFile) -> bytes:
    data = await file.read()
    if not data:
        log.warning("  Rejected: file is empty")
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(data) > MAX_FILE_MB * 1024 * 1024:
        log.warning("  Rejected: %.2f MB is over the %d MB limit", len(data) / 1024 / 1024, MAX_FILE_MB)
        raise HTTPException(status_code=413, detail=f"File is larger than {MAX_FILE_MB} MB.")
    return data


async def run_extraction(file: UploadFile, data: bytes) -> list[dict]:
    try:
        # OCR is CPU-bound; keep it off the event loop.
        return await run_in_threadpool(extract_text, file.filename or "", data)
    except UnsupportedFileError as exc:
        log.warning("  Rejected: %s", exc)
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    except Exception as exc:
        log.error("  Could not read file: %s", exc)
        raise HTTPException(status_code=422, detail=f"Could not read file: {exc}") from exc


@router.post("/extract")
async def extract(
    file: UploadFile = File(...),
    prompt: str = Form(DEFAULT_PROMPT, description="What to extract / the JSON structure you want."),
    include_text: bool = Form(False, description="Also return the extracted page text."),
):
    """Extract text from the whole file, then let Gemini return the JSON asked for in the prompt."""
    request_start = time.perf_counter()
    log.info("=" * 70)
    log.info("NEW REQUEST /extract: %s", file.filename)

    log.info("[1/3] Reading upload...")
    start = time.perf_counter()
    file_bytes = await read_upload(file)
    upload_time = time.perf_counter() - start
    log.info("[1/3] Upload OK: %.2f MB in %.2fs", len(file_bytes) / 1024 / 1024, upload_time)

    log.info("[2/3] Extracting text...")
    start = time.perf_counter()
    pages = await run_extraction(file, file_bytes)
    extract_time = time.perf_counter() - start
    ocr_pages = [p["page"] for p in pages if p["method"] == "ocr"]
    log.info(
        "[2/3] Extraction OK: %d page(s) in %.2fs (%d via digital text, %d via OCR - OCR is the slow part)",
        len(pages), extract_time, len(pages) - len(ocr_pages), len(ocr_pages),
    )
    if not any(p["text"].strip() for p in pages):
        log.warning("  Rejected: no text could be extracted from any page")
        raise HTTPException(status_code=422, detail="No text could be extracted from the file.")

    log.info("[3/3] Converting to JSON with Gemini (%s prompt)...", "default" if prompt == DEFAULT_PROMPT else "custom")
    start = time.perf_counter()
    try:
        data = await gemini_to_json(pages, prompt)
    except GeminiError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    gemini_time = time.perf_counter() - start
    log.info("[3/3] Gemini OK in %.2fs", gemini_time)

    total_time = time.perf_counter() - request_start
    log.info(
        "DONE in %.2fs | upload %.2fs (%.0f%%) | extraction %.2fs (%.0f%%) | gemini %.2fs (%.0f%%)",
        total_time,
        upload_time, upload_time / total_time * 100,
        extract_time, extract_time / total_time * 100,
        gemini_time, gemini_time / total_time * 100,
    )
    log.info("=" * 70)

    result = {
        "file_name": file.filename,
        "total_pages": len(pages),
        "ocr_pages": ocr_pages,
        "model": GEMINI_MODEL,
        "data": data,
    }
    if include_text:
        result["pages"] = pages
    return result
