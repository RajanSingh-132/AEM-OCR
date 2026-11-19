"""Settings and constants, loaded from .env."""

import os

from dotenv import load_dotenv

load_dotenv()

# Primary LLM (tried first). Optional: without a key, Gemini is used directly.
GROQ_API_KEY = os.getenv("GROQ_API_KEY").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL")
# Must accept images (gpt-oss models do not).
GROQ_VISION_MODEL = os.getenv("GROQ_VISION_MODEL")

# xAI Groq: primary LLM for the AEM chat assistant only (POST /api/v1/chat); Groq is its fallback.
# Optional: without a key, the chat uses Groq -> Gemini as before.
XAI_API_KEY = os.getenv("XAI_API_KEY", "").strip()
XAI_MODEL = os.getenv("XAI_MODEL", "").strip()
XAI_BASE_URL = os.getenv("XAI_BASE_URL", "").strip()
XAI_TIMEOUT_SECONDS = float(os.getenv("XAI_TIMEOUT_SECONDS", "").strip() or 60)

# Claude (Anthropic): primary LLM for PDF / image / Word extraction (vision OCR + JSON); Gemini is its fallback.
# Optional: without a key, extraction uses Gemini only.
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
CLAUDE_MODEL = os.getenv("LLM_MODEL", "").strip()
CLAUDE_TIMEOUT_SECONDS = float(os.getenv("CLAUDE_TIMEOUT_SECONDS", "").strip() or 300)

# Fallback LLM (used when the primary fails).
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL")
# Vision OCR model (required); must accept images.
GEMINI_VISION_MODEL = os.getenv("GEMINI_VISION_MODEL", "").strip()

_thinking_budget = os.getenv("GEMINI_THINKING_BUDGET").strip()
GEMINI_THINKING_BUDGET = int(_thinking_budget) if _thinking_budget else None
MAX_PROMPT_CHARS = 100_000
MAX_OUTPUT_TOKENS = 32_768

PDF_EXTENSIONS = {".pdf"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
WORD_EXTENSIONS = {".docx", ".doc"}
ALLOWED_EXTENSIONS = PDF_EXTENSIONS | IMAGE_EXTENSIONS | WORD_EXTENSIONS

