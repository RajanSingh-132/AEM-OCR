"""Vision OCR helper (PRD 5.1): image bytes -> plain text via the LLM."""

import io
import logging
import time

from PIL import Image

from OCRAI import llm_client
from OCRAI.prompts import NO_TEXT_MARKER, OCR_INSTRUCTION

log = logging.getLogger(__name__)

JPEG_MAGIC = b"\xff\xd8\xff"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def normalize_image(image_bytes: bytes) -> tuple[bytes, str]:
    """Re-encode as PNG so the LLM always gets a valid image. Returns (bytes, mime type)."""
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            if image.mode not in ("RGB", "RGBA"):
                image = image.convert("RGB")
            out = io.BytesIO()
            image.save(out, format="PNG")
            return out.getvalue(), "image/png"
    except Exception:
        if image_bytes.startswith(JPEG_MAGIC):
            return image_bytes, "image/jpeg"
        if image_bytes.startswith(PNG_MAGIC):
            return image_bytes, "image/png"
        raise ValueError("invalid image data")


def _run_ocr(image_bytes: bytes, name: str) -> str:
    """OCR one image. Returns "" when it has no readable text; raises on failure."""
    start = time.perf_counter()
    data, mime_type = normalize_image(image_bytes)
    text = llm_client.generate_from_image(OCR_INSTRUCTION, data, mime_type).strip()
    if NO_TEXT_MARKER in text:
        text = ""
    log.info("    Vision OCR %s (%d KB): %d chars in %.2fs", name, len(image_bytes) // 1024, len(text), time.perf_counter() - start)
    return text


def ocr_image(image_bytes: bytes, name: str) -> str | None:
    """OCR for the PDF and Word loaders: the text, or None if there is none or OCR failed.

    The loaders must not mistake the placeholder messages of extract_text_from_image
    for real text, so they use this instead.
    """
    try:
        return _run_ocr(image_bytes, name) or None
    except Exception:
        log.exception("    Vision OCR failed for %s", name)
        return None


def extract_text_from_image(image_bytes: bytes, filename: str) -> str:
    """OCR an uploaded image file. Never raises."""
    try:
        text = _run_ocr(image_bytes, filename)
    except Exception:
        log.exception("    Vision OCR failed for %s", filename)
        return f"Image file {filename}"
    return text or f"No readable text found in image {filename}"
