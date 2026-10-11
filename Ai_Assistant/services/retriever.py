"""Search over a company's collection: exact ID lookup + Atlas Vector Search.

Embeddings capture meaning, not exact codes, so identifiers in the question (trip numbers,
truck numbers, plates) are also looked up exactly and those records always come first.
Field names (text, ids, company, index) come from the collection's CollectionSpec.
"""

import logging
import re
import time

from pymongo.collection import Collection
from pymongo.errors import OperationFailure

from Ai_Assistant.collection_mapping import CollectionSpec
from Ai_Assistant.config import CHAT_NUM_CANDIDATES
from Ai_Assistant.services.embeddings import embed_text
from Ai_Assistant.services.mongo_client import ensure_vector_index_once, get_company_collection

log = logging.getLogger(__name__)

# Token that looks like an identifier: 4+ letters/digits/-/_ and at least one digit.
_ID_TOKEN = re.compile(r"(?=[\w-]*\d)[\w-]{4,}")


def _projection(spec: CollectionSpec) -> dict:
    return {
        "_id": 1,
        "text": f"${spec.text_field}",
        **{name: f"${field}" for name, field in spec.summary_fields.items()},
    }


def _to_record(doc: dict, score: float, spec: CollectionSpec) -> dict:
    return {
        "id": doc["_id"],
        "text": doc.get("text") or "",
        **{name: doc.get(name) for name in spec.summary_fields},
        "score": score,
    }


def exact_matches(collection: Collection, spec: CollectionSpec, question: str, company_id: int) -> list[dict]:
    """Records whose trip number, truck number, plate etc. appears literally in the question."""
    found = _ID_TOKEN.findall(question)
    tokens = list({*found, *(t.upper() for t in found)})
    if not tokens:
        return []
    query = {spec.company_field: company_id, "$or": [{field: {"$in": tokens}} for field in spec.id_fields]}
    pipeline = [{"$match": query}, {"$project": _projection(spec)}]
    return [_to_record(doc, 1.0, spec) for doc in collection.aggregate(pipeline)]


def search(question: str, top_k: int, company_id: int, spec: CollectionSpec) -> list[dict]:
    """Best records for the question: [{id, text, <summary fields>, score}].

    company_id is the database name (company 5 -> database "5") and also filters inside it,
    so a company only ever sees its own records. No data for the company -> CompanyDataNotFound.
    """
    collection = get_company_collection(company_id, spec.name)  # fails fast, before any embedding cost
    log.info("  Company %d -> %s.%s", company_id, collection.database.name, collection.name)
    ensure_vector_index_once(collection, spec.vector_index, spec.filter_fields, spec.embedding_field)

    start = time.perf_counter()
    vector = embed_text(question)
    log.info("[1/3] Question embedded in %.2fs", time.perf_counter() - start)

    start = time.perf_counter()
    exact = exact_matches(collection, spec, question, company_id)

    pipeline = [
        {"$vectorSearch": {
            "index": spec.vector_index,
            "path": spec.embedding_field,
            "queryVector": vector,
            "numCandidates": max(CHAT_NUM_CANDIDATES, top_k),
            "limit": top_k,
            "filter": {spec.company_field: company_id},
        }},
        {"$project": {**_projection(spec), "score": {"$meta": "vectorSearchScore"}}},
    ]
    try:
        similar = [_to_record(doc, doc["score"], spec) for doc in collection.aggregate(pipeline)]
    except OperationFailure as exc:
        # e.g. a just-created index is still building (INITIAL_SYNC): answer from exact matches only.
        log.warning("  Vector search unavailable on %s.%s: %s", collection.database.name, collection.name, exc.details.get("errmsg", exc) if exc.details else exc)
        similar = []

    # Exact ID matches first, then the most similar records, without duplicates.
    seen = {r["id"] for r in exact}
    records = (exact + [r for r in similar if r["id"] not in seen])[:max(top_k, len(exact))]
    log.info(
        "[2/3] Search: %d exact ID match(es) + vector search -> %d record(s) in %.2fs%s",
        len(exact), len(records), time.perf_counter() - start,
        f" (best similarity {similar[0]['score']:.3f})" if similar else "",
    )
    return records
