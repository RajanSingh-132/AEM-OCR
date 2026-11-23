"""Answer a question: session -> plan -> company collection -> exact query or similarity search
-> answer LLM -> save turn.

How many records reach the answer LLM depends on the question:
  lookup   all records with the named IDs (up to QUERY_HARD_LIMIT)
  list     exact query over all records + real total; the first `limit` in the asked order
  count    numbers only (total, or counts per field)
  semantic only records similar enough to the question (score cut-off)
"""

import logging
import re
import time
from dataclasses import replace

from Ai_Assistant.collection_mapping import COLLECTIONS, clear_collection, named_programs, needs_program_choice
from Ai_Assistant.config import (
    CHAT_HISTORY_ANSWER_CHARS,
    CHAT_MAX_ANSWER_TOKENS,
    PLANNER_HISTORY_ANSWER_CHARS,
    QUERY_HARD_LIMIT,
)
from Ai_Assistant.prompts import (
    COMPANY_NOT_READY_MESSAGE,
    OUT_OF_SCOPE_MESSAGE,
    PROGRAM_CHOICE_MESSAGE,
    PROGRAM_CHOICE_RETRY_MESSAGE,
    PROGRAM_CHOICE_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_program_choice_prompt,
    build_user_prompt,
    format_history,
)
from Ai_Assistant.services.chat_memory import save_turn, start_session
from Ai_Assistant.services.id_resolver import resolve_ids
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

_NOTHING = {"_id": {"$in": []}}  # matches no document (identifiers that resolved to nothing)


def answer_question(question: str, company_id: int, session_id: str | None = None) -> dict:
    """Returns {"answer": str, "session_id": str}. Raises LLMError or database errors."""
    session_id, turns = start_session(session_id, company_id)
    last = turns[-1] if turns else {}
    pending = last.get("pending_question")    # we asked "ACE or ACI?" for this question
    answered = last.get("program_question")   # we just answered this question for ACE / ACI / both
    program_question = None
    if pending and len(question.split()) <= _CHOICE_REPLY_MAX_WORDS:
        # Reply to "ACE or ACI?": ACE / ACI / Both continue the waiting question; anything
        # unclear ("trip", "ok") is asked again.
        choice = _program_choice(question)
        if choice is None:
            log.info("  Reply '%s' does not say ACE or ACI -> asking again", question)
            return _ask_program(pending, question, session_id, retry=True)
        program_question = pending
    elif answered and _is_program_switch(question):
        # Follow-up after that answer: "and ACI?", "both for?", "what about ACE" -> same
        # question for that program. Anything else is a new question.
        choice = _program_choice(question)
        program_question = answered
    if program_question:
        if choice == "both":
            return _answer_for_both(program_question, turns, company_id, session_id)
        question = f"{program_question} (for {choice.record_label}s)"
        log.info("  Reply -> '%s'", question)
    if needs_program_choice(question):
        # Status question without ACE / ACI: ask instead of guessing.
        log.info("  Status question without ACE / ACI -> asking which one")
        return _ask_program(question, question, session_id, retry=False)
    plan = plan_question(question, format_history(turns, PLANNER_HISTORY_ANSWER_CHARS), company_id)
    answer = _answer(question, plan, format_history(turns, CHAT_HISTORY_ANSWER_CHARS), company_id)
    save_turn(session_id, question, plan.standalone_question, answer, program_question=program_question)
    return {"answer": answer, "session_id": session_id}


_CHOICE_REPLY_MAX_WORDS = 4  # "ACE", "for ACI", "eManifest please"; longer = a new question
_BOTH_WORDS = re.compile(r"(?<![A-Za-z])(both|all|dono|donon|donno)(?![A-Za-z])", re.IGNORECASE)


def _program_choice(reply: str):
    """Reply to "ACE or ACI?": "both", the collection it names clearly (ACE / PAPS / USA /
    eManifest -> emanifest, ACI / PARS / Canada -> trips), or None ("trip", "ok", ...)."""
    if _BOTH_WORDS.search(reply) or len(named_programs(reply)) == 2:
        return "both"
    return clear_collection(reply)


# Words that may surround the program in a follow-up ("and what about ACI trips?", "ACI ke liye").
_SWITCH_FILLERS = {
    "and", "for", "what", "about", "how", "the", "then", "also", "please", "pls", "now", "same",
    "again", "too", "of", "in", "same", "one", "side", "trip", "trips", "record", "records",
    "data", "ke", "liye", "ka", "ki", "bhi", "ab", "aur", "ok", "okay",
}


def _is_program_switch(question: str) -> bool:
    """A short follow-up that only names ACE / ACI / both ("and ACI?", "both for?", "ACI ke liye")."""
    if len(question.split()) > _CHOICE_REPLY_MAX_WORDS:
        return False
    words = [w for w in re.findall(r"[A-Za-z.]+", question) if w.lower().strip(".") not in _SWITCH_FILLERS]
    return bool(words) and all(_program_choice(w) is not None for w in words)


def _ask_program(pending: str, question: str, session_id: str, retry: bool) -> dict:
    """Ask ACE or ACI (again, with a "sorry" when retry) and remember the waiting question."""
    message = _program_choice_message(pending, question if retry else None)
    save_turn(session_id, question, question, message, pending_question=pending)
    return {"answer": message, "session_id": session_id}


def _program_choice_message(pending: str, unclear_reply: str | None) -> str:
    """The "ACE or ACI?" message, written by the LLM for this question; the fixed message if
    the LLM fails or its reply does not ask for ACE and ACI."""
    fallback = PROGRAM_CHOICE_RETRY_MESSAGE if unclear_reply else PROGRAM_CHOICE_MESSAGE
    try:
        message = llm_client.generate_text(
            PROGRAM_CHOICE_SYSTEM_PROMPT, build_program_choice_prompt(pending, unclear_reply), max_tokens=400,
        ).strip()
    except Exception as exc:
        log.warning("  ACE / ACI question: LLM failed (%s) -> fixed message", exc)
        return fallback
    if not message or len(message) > 400 or "ACE" not in message or "ACI" not in message:
        log.warning("  ACE / ACI question: unusable LLM reply %r -> fixed message", message[:200])
        return fallback
    return message


def _answer_for_both(original: str, turns: list[dict], company_id: int, session_id: str) -> dict:
    """The waiting question answered for ACE (emanifest) first, then ACI (trips), in one reply."""
    log.info("  Reply 'both' to the ACE / ACI question -> answering '%s' for every collection", original)
    planner_history = format_history(turns, PLANNER_HISTORY_ANSWER_CHARS)
    answer_history = format_history(turns, CHAT_HISTORY_ANSWER_CHARS)
    parts = []
    for spec in COLLECTIONS.values():  # emanifest (ACE) first, then trips (ACI)
        question = f"{original} (for {spec.record_label}s)"
        plan = plan_question(question, planner_history, company_id)
        parts.append(f"**{spec.record_label}s**\n\n{_answer(question, plan, answer_history, company_id)}")
    answer = "\n\n".join(parts)
    combined = f"{original} (for ACE and ACI)"
    save_turn(session_id, combined, combined, answer, program_question=original)
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

    is_id_lookup = plan.intent == "lookup" and bool(plan.ids)
    result = fetch_data(plan, collection, company_id, related_on_empty=not is_id_lookup)
    if is_id_lookup and result.total == 0:
        # The IDs are not in this collection: they may belong to the other one (e.g. "trip 45444"
        # asked as a trip, but it is an ACE eManifest trip number).
        result = _lookup_in_other_collections(plan, company_id) or _with_related(result, plan, collection, company_id)

    log.info("[3/3] Generating answer...")
    start = time.perf_counter()
    answer = llm_client.generate_text(
        SYSTEM_PROMPT, build_user_prompt(question, result, history, plan.standalone_question),
        max_tokens=CHAT_MAX_ANSWER_TOKENS,
    ).strip()
    log.info("[3/3] Answer ready in %.2fs (%d chars)", time.perf_counter() - start, len(answer))
    return answer


def _with_related(result: QueryResult, plan: Plan, collection, company_id: int) -> QueryResult:
    """Attach the closest records (clearly labeled as not matching) to an empty exact result."""
    result.related = similarity_search(collection, plan.collection, plan.standalone_question, company_id).records
    return result


def _lookup_in_other_collections(plan: Plan, company_id: int) -> QueryResult | None:
    """Look the plan's IDs up in the company's other collections; the first that has them wins."""
    for spec in COLLECTIONS.values():
        if spec is plan.collection:
            continue
        try:
            collection = get_company_collection(company_id, spec.name)
        except CompanyDataNotFound:
            continue
        resolution = resolve_ids(collection, spec, company_id, plan.ids, build_match(spec, company_id, []))
        if resolution.condition is None:
            continue
        log.info("  IDs not found as %s -> found as %s in %s.%s",
                 plan.collection.record_label, spec.record_label, collection.database.name, collection.name)
        ensure_vector_index_once(collection, spec.vector_index, spec.filter_fields, spec.embedding_field)
        # Filters were written for the first collection's fields, so only the IDs carry over.
        other_plan = replace(plan, collection=spec, filters=[], sort=None, group_by=None)
        result = fetch_data(other_plan, collection, company_id, related_on_empty=False)
        result.description.insert(0, f"Note: no {plan.collection.record_label} record has these IDs; "
                                     f"they were found as {spec.record_label} records instead")
        return result
    return None


def fetch_data(plan: Plan, collection, company_id: int, related_on_empty: bool = True) -> QueryResult:
    """Run the plan: exact query for lookup / list / count, similarity search for semantic."""
    spec = plan.collection
    if plan.intent == "semantic":
        return similarity_search(collection, spec, plan.standalone_question, company_id)

    match = build_match(spec, company_id, plan.filters)
    id_lines = []
    if plan.ids:
        # SCAC + trip number / trip number / other IDs / SCAC only, resolved against the real data.
        resolution = resolve_ids(collection, spec, company_id, plan.ids, match)
        id_lines = resolution.lines
        match = build_match(spec, company_id, plan.filters, resolution.condition or _NOTHING)
    if plan.intent == "count":
        result = count_records(collection, spec, match, plan.group_by)
    else:
        limit = plan.limit or (QUERY_HARD_LIMIT if plan.intent == "lookup" else None)
        result = fetch_records(collection, spec, match, plan.sort, limit, plan.intent)
        if result.total == 0 and related_on_empty:
            # Nothing matched exactly (e.g. a misspelled name): offer the closest records,
            # clearly labeled as not matching.
            _with_related(result, plan, collection, company_id)
    result.description[1:1] = [f"Conditions: {describe_conditions(plan.filters)}",
                               *(f"Identifier {line}" for line in id_lines)]
    return result
