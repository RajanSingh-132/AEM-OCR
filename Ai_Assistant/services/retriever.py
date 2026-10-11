"""Similarity search (for vague questions): exact ID matches + Atlas Vector Search with a score cut-off.

Embeddings capture meaning, not exact codes, so identifiers in the question (trip numbers,
truck numbers, plates) are also looked up exactly and those records always come first.
Vector matches are kept only if they are similar enough (VECTOR_MIN_SCORE) and close to the
best match (VECTOR_SCORE_WINDOW), so the number of records depends on the question.
"""

import logging
import re
import time

from pymongo.collection import Collection
from pymongo.errors import OperationFailure

from Ai_Assistant.collection_mapping import CollectionSpec
from Ai_Assistant.config import (
    CHAT_NUM_CANDIDATES,
    VECTOR_MAX_RECORDS,
    VECTOR_MIN_SCORE,
    VECTOR_SCORE_WINDOW,
)
from Ai_Assistant.services.embeddings import embed_text
from Ai_Assistant.services.structured_query import QueryResult, exact_any

log = logging.getLogger(__name__)

# Token that looks like an identifier: 4+ letters/digits/-/_ and at least one digit.
_ID_TOKEN = re.compile(r"(?=[\w-]*\d)[\w-]{4,}")


def _projection(spec: CollectionSpec) -> dict:
    return {"_id": 1, "text": f"${spec.text_field}", **{n: f"${f}" for n, f in spec.summary_fields.items()}}


def _to_record(doc: dict, score: float, spec: CollectionSpec) -> dict:
    return {"id": doc["_id"], "text": doc.get("text") or "", **{n: doc.get(n) for n in spec.summary_fields}, "score": score}


def exact_id_matches(collection: Collection, spec: CollectionSpec, question: str, company_id: int) -> list[dict]:
    """Records whose trip number, truck number, plate etc. appears literally in the question."""
    tokens = sorted(set(_ID_TOKEN.findall(question)))
    if not tokens:
        return []
    query = {spec.company_field: company_id, "$or": [{field: exact_any(tokens)} for field in spec.id_fields]}
    pipeline = [{"$match": query}, {"$project": _projection(spec)}]
    return [_to_record(doc, 1.0, spec) for doc in collection.aggregate(pipeline)]


def similarity_search(collection: Collection, spec: CollectionSpec, question: str, company_id: int) -> QueryResult:
    start = time.perf_counter()
    vector = embed_text(question)
    log.info("  Question embedded in %.2fs", time.perf_counter() - start)

    start = time.perf_counter()
    exact = exact_id_matches(collection, spec, question, company_id)
    pipeline = [
        {"$vectorSearch": {
            "index": spec.vector_index,
            "path": spec.embedding_field,
            "queryVector": vector,
            "numCandidates": CHAT_NUM_CANDIDATES,
            "limit": VECTOR_MAX_RECORDS * 2,
            "filter": {spec.company_field: company_id},
        }},
        {"$project": {**_projection(spec), "score": {"$meta": "vectorSearchScore"}}},
    ]
    try:
        similar = [_to_record(doc, doc["score"], spec) for doc in collection.aggregate(pipeline)]
    except OperationFailure as exc:
        # e.g. a just-created index is still building (INITIAL_SYNC): answer from exact matches only.
        log.warning("  Vector search unavailable on %s.%s: %s", collection.database.name, collection.name,
                    exc.details.get("errmsg", exc) if exc.details else exc)
        similar = []

    best = similar[0]["score"] if similar else 0.0
    cutoff = max(VECTOR_MIN_SCORE, best - VECTOR_SCORE_WINDOW)
    seen = {r["id"] for r in exact}
    kept = [r for r in similar if r["score"] >= cutoff and r["id"] not in seen]
    records = (exact + kept)[:max(VECTOR_MAX_RECORDS, len(exact))]
    log.info(
        "[2/3] Similarity search: %d exact ID + %d of %d similar (score >= %.3f, best %.3f) -> %d record(s) in %.2fs",
        len(exact), len(kept), len(similar), cutoff, best, len(records), time.perf_counter() - start,
    )

    description = [
        "Search: similarity search (the closest matches by meaning; NOT a complete list and NOT a total)",
        f"Records shown below: {len(records)}",
    ]
    if exact:
        description.append(f"{len(exact)} of them matched an ID in the question exactly (shown first)")
    return QueryResult(search_type="similarity", description=description, records=records)
