"""Answer a question from the eManifest records: retrieve -> prompt -> LLM."""

import logging
import time

from Ai_Assistant.config import CHAT_MAX_ANSWER_TOKENS
from Ai_Assistant.prompts import SYSTEM_PROMPT, build_user_prompt
from Ai_Assistant.services.retriever import search
from OCRAI import llm_client

log = logging.getLogger(__name__)


def answer_question(question: str, top_k: int, company_id: int) -> dict:
    """Returns {"answer": str, "sources": [records used]}. Raises on retrieval or LLM failure."""
    records = search(question, top_k, company_id)

    log.info("[3/3] Generating answer from %d record(s)...", len(records))
    start = time.perf_counter()
    answer = llm_client.generate_text(
        SYSTEM_PROMPT, build_user_prompt(question, records), max_tokens=CHAT_MAX_ANSWER_TOKENS,
    ).strip()
    log.info("[3/3] Answer ready in %.2fs (%d chars)", time.perf_counter() - start, len(answer))

    sources = [
        {"id": r["id"], "trip_number": r["trip_number"], "status": r["status"], "port": r["port"],
         "score": round(r["score"], 4)}
        for r in records
    ]
    return {"answer": answer, "sources": sources}
