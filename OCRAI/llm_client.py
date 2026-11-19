"""The LLMs used for vision OCR, structured extraction and the AEM chat assistant.

    OCR / extraction (/upload/pdf_dynamic_extract): Claude -> Gemini   (Claude only if ANTHROPIC_API_KEY + LLM_MODEL are set)
    AEM chat (/api/v1/chat: planner+answer): xAI Groq -> Groq -> Gemini   (Groq only if XAI_API_KEY is set)

Every call goes to each provider in that order once, until one succeeds. If all fail,
LLMError is raised. There are no retries.

Clients are created once at import. Calls are synchronous because the pipeline runs in a
worker thread. LLMError messages are vendor-neutral ("primary" / "fallback"); the full
errors, with provider names, are only written to the server log.
"""

import base64
import logging
import time

import anthropic
import httpx
from google import genai
from google.genai import types
from groq import Groq

from OCRAI.config import (
    ANTHROPIC_API_KEY,
    CLAUDE_MODEL,
    CLAUDE_TIMEOUT_SECONDS,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GEMINI_THINKING_BUDGET,
    GEMINI_VISION_MODEL,
    GROQ_API_KEY,
    GROQ_MODEL,
    MAX_OUTPUT_TOKENS,
    XAI_API_KEY,
    XAI_BASE_URL,
    XAI_MODEL,
    XAI_TIMEOUT_SECONDS,
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
# xAI's API is OpenAI-compatible; plain HTTP, no extra SDK.
_xai = httpx.Client(
    base_url=XAI_BASE_URL, timeout=XAI_TIMEOUT_SECONDS,
    headers={"Authorization": f"Bearer {XAI_API_KEY}"},
) if XAI_API_KEY and XAI_MODEL and XAI_BASE_URL else None
if _xai is None:
    log.warning("XAI_API_KEY / XAI_MODEL / XAI_BASE_URL not set: the chat assistant uses Groq -> Gemini")
# Claude: no SDK retries, so a failure falls through to Gemini at once (like the other clients).
_claude = anthropic.Anthropic(
    api_key=ANTHROPIC_API_KEY, max_retries=0, timeout=CLAUDE_TIMEOUT_SECONDS,
) if ANTHROPIC_API_KEY and CLAUDE_MODEL else None
if _claude is None:
    log.warning("ANTHROPIC_API_KEY / LLM_MODEL not set: document extraction uses Gemini only")


class LLMError(RuntimeError):
    pass


# ---------- xAI Groq (chat primary) ----------

def _xai_generate(messages: list, label: str, max_tokens: int, json_mode: bool = False) -> str:
    start = time.perf_counter()
    body = {"model": XAI_MODEL, "messages": messages, "temperature": 0, "max_tokens": max_tokens}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    response = _xai.post("/chat/completions", json=body)
    if response.status_code != 200:
        raise LLMError(f"HTTP {response.status_code}: {response.text[:300]}")
    data = response.json()
    choice = data["choices"][0]
    usage = data.get("usage") or {}
    log.info(
        "    LLM %s [Groq %s]: %.2fs | tokens in %s, out %s | finish %s",
        label, XAI_MODEL, time.perf_counter() - start,
        usage.get("prompt_tokens"), usage.get("completion_tokens"), choice.get("finish_reason"),
    )
    if choice.get("finish_reason") == "length":
        raise LLMError("reply was cut off at the output token limit")
    return choice["message"].get("content") or ""


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


# ---------- Claude (extraction primary) ----------

def _claude_generate(content: list, label: str, max_tokens: int = MAX_OUTPUT_TOKENS) -> str:
    """One Claude call. Streams so a large max_tokens never hits the SDK's HTTP timeout guard."""
    start = time.perf_counter()
    with _claude.messages.stream(
        model=CLAUDE_MODEL, max_tokens=max_tokens, messages=[{"role": "user", "content": content}],
    ) as stream:
        message = stream.get_final_message()
    usage = message.usage
    log.info(
        "    LLM %s [Claude %s]: %.2fs | tokens in %s, out %s | stop %s",
        label, CLAUDE_MODEL, time.perf_counter() - start,
        getattr(usage, "input_tokens", None), getattr(usage, "output_tokens", None), message.stop_reason,
    )
    if message.stop_reason == "max_tokens":
        raise LLMError("reply was cut off at the output token limit")
    if message.stop_reason == "refusal":
        raise LLMError("the model declined the request")
    return "".join(block.text for block in message.content if block.type == "text")


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


def generate_json(prompt: str, for_chat: bool = False) -> str:
    """Prompt in, raw JSON text out.

    Document extraction (default): Claude -> Gemini.
    Chat planner (for_chat=True):  Groq (xAI) -> Groq -> Gemini.
    """
    attempts = []
    label = "chat planner" if for_chat else "extraction"
    if for_chat:
        if _xai is not None:
            attempts.append(("primary", "Groq", lambda: _xai_generate(
                [{"role": "user", "content": prompt}], label, MAX_OUTPUT_TOKENS, json_mode=True,
            )))
        if _groq is not None:
            attempts.append(("fallback" if attempts else "primary", "Groq", lambda: _groq_generate(
                GROQ_MODEL, [{"role": "user", "content": prompt}], label,
                response_format={"type": "json_object"}, max_completion_tokens=MAX_OUTPUT_TOKENS,
            )))
    elif _claude is not None:
        attempts.append(("primary", "Claude", lambda: _claude_generate([{"type": "text", "text": prompt}], label)))
    config = _gemini_config(max_output_tokens=MAX_OUTPUT_TOKENS, response_mime_type="application/json")
    attempts.append(("fallback", "Gemini", lambda: _gemini_generate(GEMINI_MODEL, prompt, config, label)))
    return _first_success(label, attempts)


def generate_text(system_prompt: str, user_prompt: str, max_tokens: int = 4096) -> str:
    """Chat answer: system prompt + user message in, plain text out. Groq -> Groq -> Gemini."""
    attempts = []
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    if _xai is not None:
        attempts.append(("primary", "Groq", lambda: _xai_generate(messages, "chat", max_tokens)))
    if _groq is not None:
        attempts.append(("fallback" if attempts else "primary", "Groq", lambda: _groq_generate(
            GROQ_MODEL, messages, "chat", max_completion_tokens=max_tokens,
        )))
    config = _gemini_config(system_instruction=system_prompt, max_output_tokens=max_tokens)
    attempts.append(("fallback", "Gemini", lambda: _gemini_generate(GEMINI_MODEL, user_prompt, config, "chat")))
    return _first_success("chat", attempts)


def generate_from_image(instruction: str, image_bytes: bytes, mime_type: str) -> str:
    """Vision OCR: one message with the image and the instruction. Claude -> Gemini."""
    attempts = []
    if _claude is not None:
        content = [
            {"type": "image", "source": {
                "type": "base64", "media_type": mime_type,
                "data": base64.standard_b64encode(image_bytes).decode("utf-8"),
            }},
            {"type": "text", "text": instruction},
        ]
        attempts.append(("primary", "Claude", lambda: _claude_generate(content, "vision OCR", max_tokens=16_000)))
    contents = [instruction, types.Part.from_bytes(data=image_bytes, mime_type=mime_type)]
    attempts.append(("fallback", "Gemini", lambda: _gemini_generate(
        GEMINI_VISION_MODEL, contents, _gemini_config(), "vision OCR"
    )))
    return _first_success("vision OCR", attempts)
