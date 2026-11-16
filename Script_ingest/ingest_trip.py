"""Ingest ACI trip JSON records into MongoDB with a 1024-d Titan V2 embedding per record.

Company -> database mapping:
    database   = the company id as text (company 5 -> database "5")
    collection = "trips" (the collection the chat assistant reads for ACI trips)

Each record becomes one document in the same layout as the synced trips in the other company
databases, so POST /api/v1/chat works on it unchanged:
    <the trip fields, flat>   TripNo, scac, FPOA, arrivalDate, TripStatusName, TruckNo, ... (internal
                              base-entity fields such as CreatedOn / IsDeleted are dropped)
    namespace       "aem_trips"
    page_content    "Field: value" lines (non-empty fields, sorted, true/false in lowercase); THIS is embedded
    contentHash     sha256 of the stored fields; unchanged records are skipped on the next run
    metadata        type, company_id, trips_id, source_document, collection, database, ...
    embedding       1024 floats (Titan V2, normalized)
    syncedAt        when the document was stored (UTC)
Re-running is safe: documents are matched on (CompanyId, Id) and replaced, never duplicated.
The Atlas Vector Search index "trips_vector_index" is created if it does not exist yet.

Input file (default: trips.txt in the project root): a JSON array, {"Data": [...]} (the API
response), or JSON objects one after another.

Run from the project root:
    python -m Script_ingest.ingest_trip --company-id 5            # all records -> database "5"
    python -m Script_ingest.ingest_trip                           # database = each record's CompanyId
    python -m Script_ingest.ingest_trip --company-id 5 --dry-run  # show the texts; no AWS / MongoDB
    python -m Script_ingest.ingest_trip --company-id 5 --limit 5  # only the first 5 records
    python -m Script_ingest.ingest_trip --company-id 5 --force    # re-embed even unchanged records
    python -m Script_ingest.ingest_trip --file other.json --company-id 7
"""

import argparse
import logging
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from pymongo import ReplaceOne

from Ai_Assistant.collection_mapping import TRIPS
from Ai_Assistant.config import (
    BEDROCK_EMBEDDING_MODEL,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_WORKERS,
    company_db_name,
)
from Ai_Assistant.services.embeddings import embed_text
from Ai_Assistant.services.mongo_client import ensure_vector_index, get_collection, ping
from Script_ingest.ingest_emanifest import content_hash, load_records

log = logging.getLogger("Ai_Assistant.ingest_trip")

DEFAULT_FILE = Path(__file__).resolve().parent.parent / "emanifest.txt"
NAMESPACE = "aem_trips"

# Base-entity / UI fields of the API export that carry no trip information; not stored.
DROPPED_FIELDS = {
    "CompanyName", "CreateDate", "LastUpdate", "CreateUser", "UpdateUser", "IsActive", "IsDeleted",
    "CreatedOn", "CreatedBy", "ModifiedOn", "ModifiedBy", "InsertUpdateUserId", "Message", "Value1", "Value2",
}
# Stored, but not part of the embedded text (same as the synced data).
NOT_EMBEDDED_FIELDS = {"LastModifiedDate", "CreatedModifiedOnDate"}


# ---------- building one document ----------

def _filled(value) -> bool:
    return value is not None and str(value).strip() != ""


def _text(value) -> str:
    """true/false in lowercase (as in the synced data), everything else as text."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value).strip()


def clean_record(record: dict, company_id: int) -> dict:
    fields = {k: v for k, v in record.items() if k not in DROPPED_FIELDS}
    fields["CompanyId"] = company_id
    return fields


def build_page_content(fields: dict) -> str:
    """'Field: value' lines of the non-empty fields, sorted by field name."""
    return "\n".join(
        f"{name}: {_text(value)}"
        for name, value in sorted(fields.items())
        if name not in NOT_EMBEDDED_FIELDS and _filled(value)
    )


def build_document(fields: dict, page_content: str, embedding: list[float], source: str, now: datetime) -> dict:
    company_id = fields["CompanyId"]
    return {
        **fields,
        "namespace": NAMESPACE,
        "page_content": page_content,
        "contentHash": content_hash(fields),
        "metadata": {
            "type": NAMESPACE,
            "company_id": company_id,
            "trips_id": fields["Id"],
            "source_document": source,
            "collection": TRIPS.name,
            "database": company_db_name(company_id),
            "ingested_at": now.isoformat(),
            "structured": True,
            "embedding_dimensions": EMBEDDING_DIMENSIONS,
        },
        TRIPS.embedding_field: embedding,
        "syncedAt": now,
    }


# ---------- ingest ----------

def _embed(item: tuple[dict, str]) -> tuple[dict, str, list[float] | None]:
    fields, text = item
    try:
        return fields, text, embed_text(text)
    except Exception as exc:
        log.error("  Embedding failed for trip Id %s: %s", fields.get("Id"), exc)
        return fields, text, None


def group_by_company(records: list[dict], company_id: int | None) -> dict[int, list[dict]]:
    groups: dict[int, list[dict]] = defaultdict(list)
    for record in records:
        if company_id is not None:
            target = company_id
            if record.get("CompanyId") not in (None, company_id):
                log.warning("  Id %s has CompanyId %s -> stored under company %s",
                            record["Id"], record.get("CompanyId"), company_id)
        elif record.get("CompanyId") is None:
            log.warning("  Id %s has no CompanyId and --company-id was not given -> skipped", record["Id"])
            continue
        else:
            target = int(record["CompanyId"])
        groups[target].append(clean_record(record, target))
    return groups


def ingest(path: Path, company_id: int | None, limit: int | None, force: bool) -> None:
    run_start = time.perf_counter()

    log.info("[1/4] Connecting to MongoDB...")
    ping()

    records = load_records(path)[:limit]
    log.info("[2/4] Loaded %d trip record(s) from %s", len(records), path.name)
    groups = group_by_company(records, company_id)
    log.info("  Target: %s", ", ".join(
        f"company {cid} -> {company_db_name(cid)}.{TRIPS.name} ({len(items)})" for cid, items in groups.items()
    ) or "nothing")

    todo, skipped = [], 0
    for cid, items in groups.items():
        existing = {}
        if not force:
            collection = get_collection(company_db_name(cid), TRIPS.name)
            cursor = collection.find({"CompanyId": cid, "Id": {"$in": [f["Id"] for f in items]}},
                                     {"Id": 1, "contentHash": 1, "has_emb": {"$isArray": f"${TRIPS.embedding_field}"}})
            existing = {d["Id"]: d for d in cursor}
        for fields in items:
            doc = existing.get(fields["Id"])
            if doc and doc.get("has_emb") and doc.get("contentHash") == content_hash(fields):
                skipped += 1
            else:
                todo.append((fields, build_page_content(fields)))
    if skipped:
        log.info("  %d record(s) unchanged since the last run -> skipped (use --force to re-embed)", skipped)

    log.info("[3/4] Embedding %d record(s) with %s (%d dims)...", len(todo), BEDROCK_EMBEDDING_MODEL, EMBEDDING_DIMENSIONS)
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=EMBEDDING_WORKERS) as pool:
        results = list(pool.map(_embed, todo))
    embedded = [(f, t, e) for f, t, e in results if e is not None]
    failed = len(results) - len(embedded)
    log.info("[3/4] Embedded %d record(s) in %.2fs (%d failed)", len(embedded), time.perf_counter() - start, failed)

    log.info("[4/4] Saving to MongoDB...")
    now = datetime.now(timezone.utc)
    source = f"file:{path.name}"
    operations: dict[int, list[ReplaceOne]] = defaultdict(list)
    for fields, text, embedding in embedded:
        operations[fields["CompanyId"]].append(ReplaceOne(
            {"CompanyId": fields["CompanyId"], "Id": fields["Id"]},
            build_document(fields, text, embedding, source, now),
            upsert=True,
        ))
    for cid in groups:
        collection = get_collection(company_db_name(cid), TRIPS.name)
        start = time.perf_counter()
        if operations[cid]:
            result = collection.bulk_write(operations[cid], ordered=False)
            log.info("[4/4] %s.%s: %d new, %d replaced in %.2fs", collection.database.name, collection.name,
                     result.upserted_count, result.modified_count, time.perf_counter() - start)
        ensure_vector_index(collection, TRIPS.vector_index, TRIPS.filter_fields, TRIPS.embedding_field)
        log.info("  %s.%s now has %d document(s)", collection.database.name, collection.name,
                 collection.count_documents({"CompanyId": cid}))

    log.info("DONE in %.2fs | %d stored, %d unchanged, %d failed",
             time.perf_counter() - run_start, len(embedded), skipped, failed)


def dry_run(path: Path, company_id: int | None, limit: int | None) -> None:
    records = load_records(path)[:limit or 3]
    for cid, items in group_by_company(records, company_id).items():
        for fields in items:
            print("-" * 60)
            print(f"-> database {company_db_name(cid)}, collection {TRIPS.name}, Id {fields['Id']}")
            print(build_page_content(fields))
    print("-" * 60)
    print(f"Dry run: {len(records)} record(s) shown; nothing embedded or stored.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest ACI trip records into MongoDB with embeddings.")
    parser.add_argument("--file", type=Path, default=DEFAULT_FILE, help="trip JSON file (default: trips.txt)")
    parser.add_argument("--company-id", type=int, help="store everything in this company's database "
                                                       "(default: each record's own CompanyId)")
    parser.add_argument("--limit", type=int, help="only the first N records")
    parser.add_argument("--force", action="store_true", help="re-embed records even if unchanged")
    parser.add_argument("--dry-run", action="store_true", help="print the texts; no AWS or MongoDB calls")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    for noisy in ("pymongo", "botocore", "boto3", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.dry_run:
        dry_run(args.file, args.company_id, args.limit)
    else:
        ingest(args.file, args.company_id, args.limit, args.force)


if __name__ == "__main__":
    main()
