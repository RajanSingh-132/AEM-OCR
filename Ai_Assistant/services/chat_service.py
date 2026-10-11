"""Answer a question: pick collection -> retrieve -> prompt -> LLM."""

import logging
import time

from Ai_Assistant.config import CHAT_MAX_ANSWER_TOKENS, CHAT_TOP_K
from Ai_Assistant.prompts import OUT_OF_SCOPE_MESSAGE, SYSTEM_PROMPT, build_user_prompt
from Ai_Assistant.services.collection_router import resolve_collection
from Ai_Assistant.services.retriever import search
from OCRAI import llm_client

log = logging.getLogger(__name__)


def answer_question(question: str, company_id: int) -> dict:
    """Returns {"answer": str}.

    Raises CompanyDataNotFound (no data for the company), LLMError, or search errors.
    """
    spec = resolve_collection(question)
    if spec is None:
        return {"answer": OUT_OF_SCOPE_MESSAGE}

    records = search(question, CHAT_TOP_K, company_id, spec)

    log.info("[3/3] Generating answer from %d record(s)...", len(records))
    start = time.perf_counter()
    answer = llm_client.generate_text(
        SYSTEM_PROMPT, build_user_prompt(question, records), max_tokens=CHAT_MAX_ANSWER_TOKENS,
    ).strip()
    log.info("[3/3] Answer ready in %.2fs (%d chars)", time.perf_counter() - start, len(answer))
    return {"answer": answer}
