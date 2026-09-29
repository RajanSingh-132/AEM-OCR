"""Settings and constants, loaded from .env."""

import os

from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
# Vision OCR model (required); must accept images.
GEMINI_VISION_MODEL = os.getenv("GEMINI_VISION_MODEL", "").strip()

_thinking_budget = os.getenv("GEMINI_THINKING_BUDGET", "0").strip()
GEMINI_THINKING_BUDGET = int(_thinking_budget) if _thinking_budget else None
MAX_PROMPT_CHARS = 100_000
MAX_OUTPUT_TOKENS = 32_768

PDF_EXTENSIONS = {".pdf"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
WORD_EXTENSIONS = {".docx", ".doc"}
ALLOWED_EXTENSIONS = PDF_EXTENSIONS | IMAGE_EXTENSIONS | WORD_EXTENSIONS

