"""Search over the stored eManifest records: exact ID lookup + Atlas Vector Search.

Embeddings capture meaning, not exact codes, so identifiers in the question (trip numbers,
truck numbers, plates) are also looked up exactly and those records always come first.
"""

import logging
import re
import time

from Ai_Assistant.config import CHAT_NUM_CANDIDATES, VECTOR_INDEX_NAME
from Ai_Assistant.services.embeddings import embed_text
from pymongo.collection import Collection

from Ai_Assistant.services.mongo_client import get_company_collection

log = logging.getLogger(__name__)

# Identifier fields matched exactly against tokens in the question.
ID_FIELDS = ("data.TripNum", "data.conveyanceNumber", "data.LpOne", "data.ShipmentControl", "data.ReferenceID")
# Token that looks like an identifier: 4+ letters/digits/-/_ and at least one digit.
_ID_TOKEN = re.compile(r"(?=[\w-]*\d)[\w-]{4,}")
_PROJECTION = {"_id": 1, "text": 1, "trip_number": "$data.TripNum", "status": "$data.Status", "port": "$data.USPortName"}


def _to_record(doc: dict, score: float) -> dict:
    return {"id": doc["_id"], **{k: doc.get(k) for k in ("trip_number", "status", "port", "text")}, "score": score}


def exact_matches(collection: Collection, question: str, company_id: int) -> list[dict]:
    """Records whose trip number, truck number, plate etc. appears literally in the question."""
    tokens = {t for t in _ID_TOKEN.findall(question)} | {t.upper() for t in _ID_TOKEN.findall(question)}
    if not tokens:
        return []
    query = {"data.CompanyId": company_id, "$or": [{field: {"$in": list(tokens)}} for field in ID_FIELDS]}
    return [_to_record(doc, 1.0) for doc in collection.aggregate([{"$match": query}, {"$project": _PROJECTION}])]


def search(question: str, top_k: int, company_id: int) -> list[dict]:
    """Best records for the question: [{id, trip_number, status, port, text, score}].

    company_id picks the database/collection (COMPANY_ROUTES) and also filters inside it,
    so a company only ever sees its own manifests. Unknown company -> UnknownCompanyError.
    """
    collection = get_company_collection(company_id)  # fails fast, before any embedding cost
    log.info("  Company %d -> %s.%s", company_id, collection.database.name, collection.name)

    start = time.perf_counter()
    vector = embed_text(question)
    log.info("[1/3] Question embedded in %.2fs", time.perf_counter() - start)

    start = time.perf_counter()
    exact = exact_matches(collection, question, company_id)

    pipeline = [
        {"$vectorSearch": {
            "index": VECTOR_INDEX_NAME,
            "path": "embedding",
            "queryVector": vector,
            "numCandidates": max(CHAT_NUM_CANDIDATES, top_k),
            "limit": top_k,
            "filter": {"data.CompanyId": company_id},
        }},
        {"$project": {**_PROJECTION, "score": {"$meta": "vectorSearchScore"}}},
    ]
    similar = [_to_record(doc, doc["score"]) for doc in collection.aggregate(pipeline)]

    # Exact ID matches first, then the most similar records, without duplicates.
    seen = {r["id"] for r in exact}
    records = (exact + [r for r in similar if r["id"] not in seen])[:max(top_k, len(exact))]
    log.info(
        "[2/3] Search: %d exact ID match(es) + vector search -> %d record(s) in %.2fs%s",
        len(exact), len(records), time.perf_counter() - start,
        f" (best similarity {similar[0]['score']:.3f})" if similar else "",
    )
    return records
