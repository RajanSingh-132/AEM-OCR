"""Conversation memory in MongoDB: the backend creates sessions; the frontend sends the id back.

One document per session in CHAT_MEMORY_DB.CHAT_MEMORY_COLLECTION:
    _id         session id (random UUID, created here)
    company_id  owner; a session is only ever loaded for the same company
    created_at, updated_at
    turns       [{question, standalone_question, answer, at}, ...] (last CHAT_MAX_STORED_TURNS)
                + pending_question on a turn that asked "ACE or ACI?" (the question still to answer)
                + program_question on a turn that answered such a question for ACE / ACI / both
                  (so "and ACI?" can ask it again for the other program)
Sessions not used for CHAT_SESSION_TTL_DAYS are deleted by MongoDB (TTL index on updated_at).
"""

import logging
import uuid
from datetime import datetime, timezone
from functools import lru_cache

from pymongo.collection import Collection

from Ai_Assistant.config import (
    CHAT_HISTORY_TURNS,
    CHAT_MAX_STORED_TURNS,
    CHAT_MEMORY_COLLECTION,
    CHAT_MEMORY_DB,
    CHAT_SESSION_TTL_DAYS,
)
from Ai_Assistant.services.mongo_client import get_client

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _sessions() -> Collection:
    collection = get_client()[CHAT_MEMORY_DB][CHAT_MEMORY_COLLECTION]
    collection.create_index("updated_at", expireAfterSeconds=CHAT_SESSION_TTL_DAYS * 24 * 3600)
    return collection


def start_session(session_id: str | None, company_id: int) -> tuple[str, list[dict]]:
    """(session id, last CHAT_HISTORY_TURNS turns).

    A missing, unknown, expired or other-company session id starts a new, empty session.
    """
    if session_id:
        doc = _sessions().find_one(
            {"_id": session_id, "company_id": company_id},
            {"turns": {"$slice": -CHAT_HISTORY_TURNS}},
        )
        if doc is not None:
            turns = doc.get("turns", [])
            log.info("  Session %s: %d earlier turn(s) loaded", session_id, len(turns))
            return session_id, turns
        log.info("  Session %s not found for company %d -> starting a new one", session_id, company_id)

    new_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    _sessions().insert_one({"_id": new_id, "company_id": company_id, "created_at": now, "updated_at": now, "turns": []})
    log.info("  New session %s", new_id)
    return new_id, []


def save_turn(session_id: str, question: str, standalone_question: str, answer: str,
              pending_question: str | None = None, program_question: str | None = None) -> None:
    """Append one question/answer to the session. Never raises: memory must not break the chat.

    pending_question: set when the answer asked "ACE or ACI?"; the question waiting for that choice.
    program_question: set when the answer was that question answered for ACE / ACI / both.
    """
    now = datetime.now(timezone.utc)
    turn = {"question": question, "standalone_question": standalone_question, "answer": answer, "at": now}
    if pending_question:
        turn["pending_question"] = pending_question
    if program_question:
        turn["program_question"] = program_question
    try:
        _sessions().update_one(
            {"_id": session_id},
            {"$push": {"turns": {"$each": [turn], "$slice": -CHAT_MAX_STORED_TURNS}}, "$set": {"updated_at": now}},
        )
    except Exception:
        log.exception("  Could not save the turn to session %s", session_id)
