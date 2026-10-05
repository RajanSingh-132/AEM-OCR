import logging
import time

from OCRAI import llm_client
from OCRAI.config import MAX_PROMPT_CHARS
from OCRAI.json_parser import parse_llm_json
from OCRAI.loaders import load_pages
from OCRAI.prompts import EXTRACTION_PROMPT

log = logging.getLogger(__name__)


def extract_document(file_bytes: bytes, filename: str) -> dict | list:
    try:
        log.info("[2/3] Loading text...")
        start = time.perf_counter()
        pages = load_pages(file_bytes, filename)
        full_text = "\n".join(page for page in pages if page.strip()).strip()
        log.info("[2/3] Text loaded: %d page(s), %d chars in %.2fs", len(pages), len(full_text), time.perf_counter() - start)
        if not full_text:
            return {"error": f"No text could be extracted from '{filename}'. The file may be corrupted or empty."}

        if len(full_text) > MAX_PROMPT_CHARS:
            log.warning("  Text is %d chars; only the first %d are sent to the LLM", len(full_text), MAX_PROMPT_CHARS)
        prompt = EXTRACTION_PROMPT.replace("{text}", full_text[:MAX_PROMPT_CHARS])

        log.info("[3/3] Structured extraction...")
        start = time.perf_counter()
        raw = llm_client.generate_json(prompt)
        result = parse_llm_json(raw)
        log.info("[3/3] Extraction done in %.2fs (reply %d chars)", time.perf_counter() - start, len(raw))
        return result
    except Exception as exc:
        log.exception("  Pipeline failed for %s", filename)
        return {"error": str(exc)}
