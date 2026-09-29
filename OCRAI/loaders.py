"""Dispatch an uploaded file to the right loader by extension. Returns one text per page."""

from pathlib import Path

from OCRAI.config import ALLOWED_EXTENSIONS, IMAGE_EXTENSIONS, PDF_EXTENSIONS, WORD_EXTENSIONS
from OCRAI.ocr import extract_text_from_image
from OCRAI.pdf_loader import load_pdf
from OCRAI.word_loader import load_word


def load_pages(data: bytes, filename: str) -> list[str]:
    ext = Path(filename).suffix.lower()
    if ext in PDF_EXTENSIONS:
        return load_pdf(data, filename)
    if ext in IMAGE_EXTENSIONS:
        return [extract_text_from_image(data, filename)]
    if ext in WORD_EXTENSIONS:
        return load_word(data, filename)
    raise ValueError(f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}")
