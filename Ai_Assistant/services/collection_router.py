"""Map a question to a collection: keyword aliases first, then ONE LLM call as fallback."""

import logging
import time

from Ai_Assistant.collection_mapping import COLLECTIONS, CollectionSpec, match_by_keywords
from Ai_Assistant.prompts import build_collection_router_prompt
from OCRAI import llm_client
from OCRAI.json_parser import parse_llm_json

log = logging.getLogger(__name__)


def resolve_collection(question: str) -> CollectionSpec | None:
    """The collection the question is about, or None if it belongs to no collection.

    Raises llm_client.LLMError if the keyword match fails and the LLM cannot be reached.
    """
    if match := match_by_keywords(question):
        spec, alias = match
        log.info("[0/3] Collection '%s' (keyword '%s')", spec.name, alias)
        return spec

    start = time.perf_counter()
    reply = parse_llm_json(llm_client.generate_json(build_collection_router_prompt(question, COLLECTIONS)))
    name = reply.get("collection") if isinstance(reply, dict) else None
    spec = COLLECTIONS.get(str(name).strip().lower()) if name else None
    log.info(
        "[0/3] No keyword match -> LLM decided: %s (%.2fs)",
        f"collection '{spec.name}'" if spec else "no collection", time.perf_counter() - start,
    )
    return spec
