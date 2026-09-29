"""Settings and constants, loaded from .env."""

import os

from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL")
MAX_FILE_MB = int(os.getenv("MAX_FILE_MB", "25"))
PDF_OCR_DPI = int(os.getenv("PDF_OCR_DPI", "200"))
# A PDF page with less digital text than this is treated as scanned and OCR'd.
MIN_TEXT_CHARS = 30

PDF_EXTENSIONS = {".pdf"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# Prompts live in OCRAI/prompts.py.

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing. Add it to the .env file.")
