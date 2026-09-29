"""Text extraction: digital PDF text via PyMuPDF, everything else via RapidOCR."""

import io
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import pymupdf
from PIL import Image, ImageSequence
from rapidocr_onnxruntime import RapidOCR

from OCRAI.config import IMAGE_EXTENSIONS, MIN_TEXT_CHARS, PDF_EXTENSIONS, PDF_OCR_DPI

log = logging.getLogger(__name__)

# Loading the ONNX models takes a few seconds, so it is done once at startup.
_start = time.perf_counter()
ocr_engine = RapidOCR()
log.info("RapidOCR engine loaded in %.2fs (one-time startup cost)", time.perf_counter() - _start)


class UnsupportedFileError(ValueError):
    pass


def preprocess(img: np.ndarray) -> np.ndarray:
    """Grayscale, upscale small images and lightly denoise to help OCR."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = img.shape[:2]
    if max(h, w) < 1000:
        scale = 1000 / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        log.info("    Upscaled %dx%d -> %dx%d px (small images OCR poorly)", w, h, img.shape[1], img.shape[0])
    img = cv2.fastNlMeansDenoising(img, h=10)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def ocr_image(img: np.ndarray) -> tuple[str, float | None]:
    """Run OCR on an RGB image array. Returns (text, average confidence)."""
    start = time.perf_counter()
    prepared = preprocess(img)
    log.info(
        "    Preprocess (grayscale + denoise) %dx%d px: %.2fs - denoise cost grows with pixel count",
        prepared.shape[1], prepared.shape[0], time.perf_counter() - start,
    )

    start = time.perf_counter()
    result, _ = ocr_engine(prepared)
    elapsed = time.perf_counter() - start
    if not result:
        log.info("    OCR: %.2fs - no text found", elapsed)
        return "", None
    lines = [text for _box, text, _score in result]
    confidence = sum(float(score) for *_, score in result) / len(result)
    log.info(
        "    OCR: %.2fs - %d text lines, avg confidence %.2f (time grows with number of lines)",
        elapsed, len(lines), confidence,
    )
    return "\n".join(lines), round(confidence, 4)


def extract_pdf(data: bytes) -> list[dict]:
    pages = []
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        total = doc.page_count
        log.info("  PDF opened: %d page(s)", total)
        for number, page in enumerate(doc, start=1):
            page_start = time.perf_counter()
            text = page.get_text("text").strip()
            if len(text) >= MIN_TEXT_CHARS:
                pages.append({"page": number, "method": "text", "confidence": None, "text": text})
                log.info(
                    "  Page %d/%d: digital text (%d chars >= %d) -> read directly, no OCR needed: %.2fs",
                    number, total, len(text), MIN_TEXT_CHARS, time.perf_counter() - page_start,
                )
                continue

            log.info(
                "  Page %d/%d: only %d chars of digital text (< %d) -> treated as scanned, running OCR",
                number, total, len(text), MIN_TEXT_CHARS,
            )
            start = time.perf_counter()
            pix = page.get_pixmap(dpi=PDF_OCR_DPI, colorspace=pymupdf.csRGB, alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
            log.info(
                "    Render to image at %d DPI (%dx%d px): %.2fs",
                PDF_OCR_DPI, pix.width, pix.height, time.perf_counter() - start,
            )
            ocr_text, confidence = ocr_image(img)
            pages.append({"page": number, "method": "ocr", "confidence": confidence, "text": ocr_text})
            log.info(
                "  Page %d/%d done via OCR: %d chars in %.2fs",
                number, total, len(ocr_text), time.perf_counter() - page_start,
            )
    return pages


def extract_image(data: bytes) -> list[dict]:
    pages = []
    with Image.open(io.BytesIO(data)) as image:
        total = getattr(image, "n_frames", 1)
        log.info("  Image opened: %s %dx%d px, %d frame(s)", image.format, image.width, image.height, total)
        # Multi-page TIFFs yield several frames; other formats yield one.
        for number, frame in enumerate(ImageSequence.Iterator(image), start=1):
            page_start = time.perf_counter()
            log.info("  Page %d/%d: image -> always needs OCR", number, total)
            img = np.array(frame.convert("RGB"))
            ocr_text, confidence = ocr_image(img)
            pages.append({"page": number, "method": "ocr", "confidence": confidence, "text": ocr_text})
            log.info(
                "  Page %d/%d done via OCR: %d chars in %.2fs",
                number, total, len(ocr_text), time.perf_counter() - page_start,
            )
    return pages


def extract_text(filename: str, data: bytes) -> list[dict]:
    ext = Path(filename).suffix.lower()
    if ext in PDF_EXTENSIONS:
        log.info("  File type '%s' -> PDF path (digital text first, OCR only for scanned pages)", ext)
        return extract_pdf(data)
    if ext in IMAGE_EXTENSIONS:
        log.info("  File type '%s' -> image path (OCR on every page)", ext)
        return extract_image(data)
    raise UnsupportedFileError(
        f"Unsupported file type '{ext}'. Allowed: {sorted(PDF_EXTENSIONS | IMAGE_EXTENSIONS)}"
    )
