"""Answer a question: session -> standalone question -> pick collection -> retrieve -> LLM -> save turn."""

import logging
import time

from Ai_Assistant.config import (
    CHAT_HISTORY_ANSWER_CHARS,
    CHAT_MAX_ANSWER_TOKENS,
    CHAT_MAX_QUESTION_CHARS,
    CHAT_TOP_K,
)
from Ai_Assistant.prompts import (
    COMPANY_NOT_READY_MESSAGE,
    OUT_OF_SCOPE_MESSAGE,
    SYSTEM_PROMPT,
    build_rewrite_prompt,
    build_user_prompt,
    format_history,
)
from Ai_Assistant.services.chat_memory import save_turn, start_session
from Ai_Assistant.services.collection_router import resolve_collection
from Ai_Assistant.services.mongo_client import CompanyDataNotFound
from Ai_Assistant.services.retriever import search
from OCRAI import llm_client

log = logging.getLogger(__name__)

REWRITE_SYSTEM_PROMPT = "You rewrite follow-up questions into standalone questions. Reply with the question only."


def standalone_question(question: str, history: str) -> str:
    """The follow-up rewritten to stand on its own. On any failure the original question is used."""
    if not history:
        return question
    start = time.perf_counter()
    try:
        rewritten = llm_client.generate_text(REWRITE_SYSTEM_PROMPT, build_rewrite_prompt(history, question), max_tokens=1024)
    except llm_client.LLMError as exc:
        log.warning("  Follow-up rewrite failed, using the question as asked: %s", exc)
        return question
    # Collapse all whitespace (the LLM sometimes uses non-breaking spaces) and drop quotes.
    rewritten = " ".join(rewritten.split()).strip('"').strip()
    if not rewritten or len(rewritten) > CHAT_MAX_QUESTION_CHARS:
        return question
    log.info("  Follow-up rewritten in %.2fs: %r", time.perf_counter() - start, rewritten)
    return rewritten


def answer_question(question: str, company_id: int, session_id: str | None = None) -> dict:
    """Returns {"answer": str, "session_id": str}. Raises LLMError or search errors."""
    session_id, turns = start_session(session_id, company_id)
    history = format_history(turns, CHAT_HISTORY_ANSWER_CHARS)
    query = standalone_question(question, history)

    answer = _answer(question, query, history, company_id)
    save_turn(session_id, question, query, answer)
    return {"answer": answer, "session_id": session_id}


def _answer(question: str, query: str, history: str, company_id: int) -> str:
    spec = resolve_collection(query)
    if spec is None:
        return OUT_OF_SCOPE_MESSAGE

    try:
        records = search(query, CHAT_TOP_K, company_id, spec)
    except CompanyDataNotFound as exc:
        log.warning("  No data for company %d (%s) -> 'working on the data' answer", company_id, exc)
        return COMPANY_NOT_READY_MESSAGE.format(company_id=company_id)

    log.info("[3/3] Generating answer from %d record(s)...", len(records))
    start = time.perf_counter()
    answer = llm_client.generate_text(
        SYSTEM_PROMPT, build_user_prompt(question, records, history, query), max_tokens=CHAT_MAX_ANSWER_TOKENS,
    ).strip()
    log.info("[3/3] Answer ready in %.2fs (%d chars)", time.perf_counter() - start, len(answer))
    return answer
