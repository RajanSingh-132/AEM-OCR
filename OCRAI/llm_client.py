"""The LLMs used for vision OCR and structured extraction: Groq first, Gemini as fallback.

Every call goes to the primary (Groq) once; if that fails, it goes to the fallback (Gemini)
once. If both fail, LLMError is raised. There are no retries.

Clients are created once at import. Calls are synchronous because the pipeline runs in a
worker thread. LLMError messages are vendor-neutral ("primary" / "fallback"); the full
errors, with provider names, are only written to the server log.
"""

import base64
import logging
import time

from google import genai
from google.genai import types
from groq import Groq

from OCRAI.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GEMINI_THINKING_BUDGET,
    GEMINI_VISION_MODEL,
    GROQ_API_KEY,
    GROQ_MODEL,
    GROQ_VISION_MODEL,
    MAX_OUTPUT_TOKENS,
)

log = logging.getLogger(__name__)

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing. Add it to the .env file.")
if not GEMINI_VISION_MODEL:
    raise RuntimeError(
        "GEMINI_VISION_MODEL is missing. Add it to the .env file, e.g. GEMINI_VISION_MODEL=gemini-2.5-flash"
    )

# No retries on either client: each call is sent exactly once.
_groq = Groq(api_key=GROQ_API_KEY, max_retries=0) if GROQ_API_KEY else None
_gemini = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options=types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=1)),
)
if _groq is None:
    log.warning("GROQ_API_KEY not set: using Gemini only")


class LLMError(RuntimeError):
    pass


# ---------- Groq (primary) ----------

def _groq_generate(model: str, messages: list, label: str, **kwargs) -> str:
    start = time.perf_counter()
    response = _groq.chat.completions.create(model=model, messages=messages, temperature=0, **kwargs)
    choice = response.choices[0]
    usage = response.usage
    log.info(
        "    LLM %s [Groq %s]: %.2fs | tokens in %s, out %s | finish %s",
        label, model, time.perf_counter() - start,
        getattr(usage, "prompt_tokens", None), getattr(usage, "completion_tokens", None), choice.finish_reason,
    )
    if choice.finish_reason == "length":
        raise LLMError("reply was cut off at the output token limit")
    return choice.message.content or ""


# ---------- Gemini (fallback) ----------

def _gemini_config(**kwargs) -> types.GenerateContentConfig:
    if GEMINI_THINKING_BUDGET is not None:
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=GEMINI_THINKING_BUDGET)
    return types.GenerateContentConfig(
        temperature=0,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        **kwargs,
    )


def _gemini_generate(model: str, contents, config: types.GenerateContentConfig, label: str) -> str:
    start = time.perf_counter()
    response = _gemini.models.generate_content(model=model, contents=contents, config=config)
    usage = response.usage_metadata
    finish = response.candidates[0].finish_reason if response.candidates else None
    log.info(
        "    LLM %s [Gemini %s]: %.2fs | tokens in %s, out %s, thinking %s | finish %s",
        label, model, time.perf_counter() - start,
        getattr(usage, "prompt_token_count", None),
        getattr(usage, "candidates_token_count", None),
        getattr(usage, "thoughts_token_count", None),
        getattr(finish, "name", finish),
    )
    if finish is not None and getattr(finish, "name", "") == "MAX_TOKENS":
        log.warning("    LLM %s hit the output token limit; the reply is cut off", label)
    return response.text or ""


# ---------- Primary -> fallback ----------

def _first_success(label: str, attempts: list[tuple[str, str, object]]) -> str:
    """Run (role, provider, call) in order; return the first success, else raise LLMError."""
    failures = []
    for role, provider, call in attempts:
        start = time.perf_counter()
        try:
            return call()
        except Exception as exc:
            log.warning("    LLM %s [%s] failed after %.2fs: %s", label, provider, time.perf_counter() - start, exc)
            failures.append(f"{role}: {type(exc).__name__}")
    raise LLMError(f"LLM request failed ({'; '.join(failures)}).")


def generate_json(prompt: str) -> str:
    """Structured extraction: prompt in, raw JSON text out."""
    attempts = []
    if _groq is not None:
        attempts.append(("primary", "Groq", lambda: _groq_generate(
            GROQ_MODEL, [{"role": "user", "content": prompt}], "extraction",
            response_format={"type": "json_object"}, max_completion_tokens=MAX_OUTPUT_TOKENS,
        )))
    config = _gemini_config(max_output_tokens=MAX_OUTPUT_TOKENS, response_mime_type="application/json")
    attempts.append(("fallback", "Gemini", lambda: _gemini_generate(GEMINI_MODEL, prompt, config, "extraction")))
    return _first_success("extraction", attempts)


def generate_text(system_prompt: str, user_prompt: str, max_tokens: int = 4096) -> str:
    """Chat answer: system prompt + user message in, plain text out."""
    attempts = []
    if _groq is not None:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        attempts.append(("primary", "Groq", lambda: _groq_generate(
            GROQ_MODEL, messages, "chat", max_completion_tokens=max_tokens,
        )))
    config = _gemini_config(system_instruction=system_prompt, max_output_tokens=max_tokens)
    attempts.append(("fallback", "Gemini", lambda: _gemini_generate(GEMINI_MODEL, user_prompt, config, "chat")))
    return _first_success("chat", attempts)


def generate_from_image(instruction: str, image_bytes: bytes, mime_type: str) -> str:
    """Vision OCR: one message with the instruction and the image."""
    attempts = []
    if _groq is not None:
        data_url = f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode()}"
        messages = [{"role": "user", "content": [
            {"type": "text", "text": instruction},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]}]
        attempts.append(("primary", "Groq", lambda: _groq_generate(GROQ_VISION_MODEL, messages, "vision OCR")))
    contents = [instruction, types.Part.from_bytes(data=image_bytes, mime_type=mime_type)]
    attempts.append(("fallback", "Gemini", lambda: _gemini_generate(
        GEMINI_VISION_MODEL, contents, _gemini_config(), "vision OCR"
    )))
    return _first_success("vision OCR", attempts)
