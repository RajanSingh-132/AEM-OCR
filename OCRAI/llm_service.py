"""Send extracted document text to Gemini and get JSON back."""

import json
import logging
import time

from google import genai
from google.genai import types

from OCRAI.config import GEMINI_API_KEY, GEMINI_MODEL
from OCRAI.prompts import SYSTEM_INSTRUCTION

log = logging.getLogger(__name__)

client = genai.Client(api_key=GEMINI_API_KEY)


class GeminiError(RuntimeError):
    pass


async def gemini_to_json(pages: list[dict], prompt: str) -> dict | list:
    document = "\n\n".join(f"--- Page {p['page']} ---\n{p['text']}" for p in pages)
    contents = f"INSTRUCTIONS:\n{prompt}\n\nDOCUMENT:\n{document}"
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        response_mime_type="application/json",
        temperature=0,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    log.info(
        "  Sending to %s: prompt %d chars + document %d chars (%d page(s)) - waiting for reply...",
        GEMINI_MODEL, len(prompt), len(document), len(pages),
    )
    start = time.perf_counter()
    try:
        response = await client.aio.models.generate_content(
            model=GEMINI_MODEL, contents=contents, config=config
        )
    except Exception as exc:
        log.error("  Gemini request failed after %.2fs: %s", time.perf_counter() - start, exc)
        raise GeminiError(f"Gemini request failed: {exc}") from exc

    # Token counts explain the time: more input/output/thinking tokens = longer call.
    usage = response.usage_metadata
    log.info(
        "  Gemini replied in %.2fs | tokens: input %s, output %s, thinking %s",
        time.perf_counter() - start,
        getattr(usage, "prompt_token_count", None),
        getattr(usage, "candidates_token_count", None),
        getattr(usage, "thoughts_token_count", None),
    )

    try:
        return json.loads(response.text)
    except (TypeError, json.JSONDecodeError) as exc:
        log.error("  Gemini reply was not valid JSON")
        raise GeminiError(f"Gemini did not return valid JSON: {response.text!r}") from exc
