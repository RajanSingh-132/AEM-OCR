"""The LLM (Gemini) used for vision OCR and structured extraction.

The client is created once at import. Calls are synchronous because the pipeline
runs in a worker thread. Errors are re-raised as LLMError with a vendor-neutral
message; the full error is only written to the server log.
"""

import logging
import time

from google import genai
from google.genai import types

from OCRAI.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GEMINI_THINKING_BUDGET,
    GEMINI_VISION_MODEL,
    MAX_OUTPUT_TOKENS,
)

log = logging.getLogger(__name__)

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing. Add it to the .env file.")
if not GEMINI_VISION_MODEL:
    raise RuntimeError(
        "GEMINI_VISION_MODEL is missing. Add it to the .env file, e.g. GEMINI_VISION_MODEL=gemini-2.5-flash"
    )

_client = genai.Client(api_key=GEMINI_API_KEY)


class LLMError(RuntimeError):
    pass


def _config(**kwargs) -> types.GenerateContentConfig:
    if GEMINI_THINKING_BUDGET is not None:
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=GEMINI_THINKING_BUDGET)
    return types.GenerateContentConfig(
        temperature=0,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        **kwargs,
    )


def _generate(model: str, contents, config: types.GenerateContentConfig, label: str) -> str:
    start = time.perf_counter()
    try:
        response = _client.models.generate_content(model=model, contents=contents, config=config)
    except Exception as exc:
        log.error("    LLM %s failed after %.2fs: %s", label, time.perf_counter() - start, exc)
        raise LLMError(f"LLM request failed ({type(exc).__name__}).") from exc

    usage = response.usage_metadata
    finish = response.candidates[0].finish_reason if response.candidates else None
    log.info(
        "    LLM %s: %.2fs | tokens in %s, out %s, thinking %s | finish %s",
        label, time.perf_counter() - start,
        getattr(usage, "prompt_token_count", None),
        getattr(usage, "candidates_token_count", None),
        getattr(usage, "thoughts_token_count", None),
        getattr(finish, "name", finish),
    )
    if finish is not None and getattr(finish, "name", "") == "MAX_TOKENS":
        log.warning("    LLM %s hit the output token limit; the reply is cut off", label)
    return response.text or ""


def generate_json(prompt: str) -> str:
    """Structured extraction: prompt in, raw JSON text out."""
    config = _config(max_output_tokens=MAX_OUTPUT_TOKENS, response_mime_type="application/json")
    return _generate(GEMINI_MODEL, prompt, config, "extraction")


def generate_from_image(instruction: str, image_bytes: bytes, mime_type: str) -> str:
    """Vision OCR: one message with the instruction and the image."""
    contents = [instruction, types.Part.from_bytes(data=image_bytes, mime_type=mime_type)]
    return _generate(GEMINI_VISION_MODEL, contents, _config(), "vision OCR")
