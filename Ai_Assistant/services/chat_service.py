"""Answer a question: session -> plan -> company collection -> exact query or similarity search
-> answer LLM -> save turn.

How many records reach the answer LLM depends on the question:
  lookup   all records with the named IDs (up to QUERY_HARD_LIMIT)
  list     exact query over all records + real total; the first `limit` in the asked order
  count    numbers only (total, or counts per field)
  semantic only records similar enough to the question (score cut-off)
"""

import logging
import time

from Ai_Assistant.config import (
    CHAT_HISTORY_ANSWER_CHARS,
    CHAT_MAX_ANSWER_TOKENS,
    PLANNER_HISTORY_ANSWER_CHARS,
    QUERY_HARD_LIMIT,
)
from Ai_Assistant.prompts import (
    COMPANY_NOT_READY_MESSAGE,
    OUT_OF_SCOPE_MESSAGE,
    SYSTEM_PROMPT,
    build_user_prompt,
    format_history,
)
from Ai_Assistant.services.chat_memory import save_turn, start_session
from Ai_Assistant.services.mongo_client import (
    CompanyDataNotFound,
    ensure_vector_index_once,
    get_company_collection,
)
from Ai_Assistant.services.query_planner import Plan, plan_question
from Ai_Assistant.services.retriever import similarity_search
from Ai_Assistant.services.structured_query import (
    QueryResult,
    build_match,
    count_records,
    describe_conditions,
    fetch_records,
)
from OCRAI import llm_client

log = logging.getLogger(__name__)


def answer_question(question: str, company_id: int, session_id: str | None = None) -> dict:
    """Returns {"answer": str, "session_id": str}. Raises LLMError or database errors."""
    session_id, turns = start_session(session_id, company_id)
    plan = plan_question(question, format_history(turns, PLANNER_HISTORY_ANSWER_CHARS), company_id)
    answer = _answer(question, plan, format_history(turns, CHAT_HISTORY_ANSWER_CHARS), company_id)
    save_turn(session_id, question, plan.standalone_question, answer)
    return {"answer": answer, "session_id": session_id}


def _answer(question: str, plan: Plan, history: str, company_id: int) -> str:
    spec = plan.collection
    if spec is None:
        return OUT_OF_SCOPE_MESSAGE
    try:
        collection = get_company_collection(company_id, spec.name)
    except CompanyDataNotFound as exc:
        log.warning("  No data for company %d (%s) -> 'working on the data' answer", company_id, exc)
        return COMPANY_NOT_READY_MESSAGE.format(company_id=company_id)
    log.info("  Company %d -> %s.%s", company_id, collection.database.name, collection.name)
    ensure_vector_index_once(collection, spec.vector_index, spec.filter_fields, spec.embedding_field)

    result = fetch_data(plan, collection, company_id)

    log.info("[3/3] Generating answer...")
    start = time.perf_counter()
    answer = llm_client.generate_text(
        SYSTEM_PROMPT, build_user_prompt(question, result, history, plan.standalone_question),
        max_tokens=CHAT_MAX_ANSWER_TOKENS,
    ).strip()
    log.info("[3/3] Answer ready in %.2fs (%d chars)", time.perf_counter() - start, len(answer))
    return answer


def fetch_data(plan: Plan, collection, company_id: int) -> QueryResult:
    """Run the plan: exact query for lookup / list / count, similarity search for semantic."""
    spec = plan.collection
    if plan.intent == "semantic":
        return similarity_search(collection, spec, plan.standalone_question, company_id)

    ids = plan.ids if plan.intent == "lookup" else []
    match = build_match(spec, company_id, plan.filters, ids)
    if plan.intent == "count":
        result = count_records(collection, spec, match, plan.group_by)
    else:
        limit = plan.limit or (QUERY_HARD_LIMIT if plan.intent == "lookup" else None)
        result = fetch_records(collection, spec, match, plan.sort, limit, plan.intent)
        if result.total == 0:
            # Nothing matched exactly (e.g. a misspelled name): offer the closest records,
            # clearly labeled as not matching.
            result.related = similarity_search(collection, spec, plan.standalone_question, company_id).records
    result.description.insert(1, f"Conditions: {describe_conditions(plan.filters, ids)}")
    return result
