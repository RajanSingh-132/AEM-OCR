"""The extraction pipeline: file -> page texts -> LLM -> JSON in the module's template (ACE / ACI)."""

import logging
import time

from OCRAI import llm_client
from OCRAI.config import MAX_PROMPT_CHARS
from OCRAI.json_parser import parse_llm_json
from OCRAI.loaders import load_pages
from OCRAI.prompts import build_extraction_prompt
from OCRAI.schemas import conform_to_template

log = logging.getLogger(__name__)


class ExtractionError(RuntimeError):
    pass


def extract_document(file_bytes: bytes, filename: str, module: str) -> dict:
    """Returns the module template filled from the document. Raises ExtractionError on failure."""
    log.info("[2/3] Loading text...")
    start = time.perf_counter()
    try:
        pages = load_pages(file_bytes, filename)
    except ValueError as exc:
        raise ExtractionError(str(exc)) from exc
    except Exception as exc:
        log.exception("  Could not read %s", filename)
        raise ExtractionError(f"Could not read '{filename}'. The file may be corrupted.") from exc
    full_text = "\n".join(page for page in pages if page.strip()).strip()
    log.info("[2/3] Text loaded: %d page(s), %d chars in %.2fs", len(pages), len(full_text), time.perf_counter() - start)
    if not full_text:
        raise ExtractionError(f"No text could be extracted from '{filename}'. The file may be corrupted or empty.")

    if len(full_text) > MAX_PROMPT_CHARS:
        log.warning("  Text is %d chars; only the first %d are sent to the LLM", len(full_text), MAX_PROMPT_CHARS)
    prompt = build_extraction_prompt(module, full_text[:MAX_PROMPT_CHARS])

    log.info("[3/3] Filling %s template...", module)
    start = time.perf_counter()
    try:
        raw = llm_client.generate_json(prompt)
    except llm_client.LLMError as exc:
        raise ExtractionError(str(exc)) from exc
    result = conform_to_template(parse_llm_json(raw), module)
    log.info("[3/3] %s template filled in %.2fs (reply %d chars)", module, time.perf_counter() - start, len(raw))
    return result
