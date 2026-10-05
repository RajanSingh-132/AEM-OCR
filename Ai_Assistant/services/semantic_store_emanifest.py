"""Store eManifest records in MongoDB with a 1024-d text embedding per record.

Each record becomes one document, in this field order:
    _id        the eManifest "Id" (re-running replaces instead of duplicating)
    data       the original JSON record, unchanged and in the same field order as the file
    text       readable summary of the record; this is what was embedded (semantic search)
    embedding  1024 floats
    stored_at  when the document was stored (UTC date and time)
Documents are written whole (replace, not $set), which keeps the field order.
Records whose data has not changed are skipped (no new embedding cost); --force re-embeds all.

Run from the project root:
    python -m Ai_Assistant.services.semantic_store_emanifest              # store everything
    python -m Ai_Assistant.services.semantic_store_emanifest --dry-run    # show texts only, no AWS/MongoDB
    python -m Ai_Assistant.services.semantic_store_emanifest --limit 5    # first 5 records
"""

import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from pymongo import ReplaceOne

from Ai_Assistant.config import BEDROCK_EMBEDDING_MODEL, EMANIFEST_FILE, EMBEDDING_DIMENSIONS, EMBEDDING_WORKERS
from Ai_Assistant.services.embeddings import embed_text
from Ai_Assistant.services.mongo_client import ensure_vector_index, get_collection, ping

log = logging.getLogger("Ai_Assistant.store")

# Fields that go into the embedded text, with readable labels, in reading order.
TEXT_FIELDS = (
    ("TripNum", "Trip number"),
    ("Status", "Status"),
    ("prevstatus", "Previous status"),
    ("Scac", "SCAC"),
    ("CarrierName", "Carrier"),
    ("CompanyName", "Company"),
    ("USPortName", "US port of arrival"),
    ("InTransit", "In transit"),
    ("ManifestType", "Manifest type"),
    ("ManifestCode", "Manifest code"),
    ("MessageType", "Message type"),
    ("DriverFullName", "Driver"),
    ("conveyanceNumber", "Truck / conveyance number"),
    ("LpOne", "Truck license plate"),
    ("Trailers", "Trailers"),
    ("ShipmentControl", "Shipment control number"),
    ("ReferenceID", "Reference ID"),
    ("AceID", "ACE ID"),
    ("SubmitTime", "Submitted"),
    ("FinalizedTime", "Finalized"),
    ("ArriveBorderTime", "Arrived at border"),
    ("AmendmentCode", "Amendment code"),
    ("Remarks", "Remarks"),
    ("noteCount", "Notes"),
    ("Version", "Version"),
    ("CreatedModifiedOnDate", "Last modified"),
    ("CreatedModifiedByName", "Modified by"),
)
SEAL_FIELDS = ("SealOne", "SealTwo", "SealThree", "SealFour", "SealFive", "SealSix")
YES_NO = {"Y": "Yes", "N": "No"}


def _filled(value) -> bool:
    return value not in (None, "", "0001-01-01T00:00:00")


def parse_arrival(record: dict) -> datetime | None:
    """ArrivalDate "YYYYMMDD" + ArrivalTime "HHMM" -> datetime (None if missing or invalid)."""
    date, clock = str(record.get("ArrivalDate") or ""), str(record.get("ArrivalTime") or "0000")
    try:
        return datetime.strptime(date + clock.zfill(4), "%Y%m%d%H%M")
    except ValueError:
        return None


def build_text(record: dict) -> str:
    """Readable "Label: value" summary of the record; this is what gets embedded."""
    lines = ["ACE eManifest"]
    arrival = parse_arrival(record)
    for field, label in TEXT_FIELDS:
        value = record.get(field)
        if not _filled(value):
            continue
        if field == "InTransit":
            value = YES_NO.get(str(value).upper(), value)
        lines.append(f"{label}: {value}")
        if field == "USPortName" and arrival:
            lines.append(f"Arrival: {arrival:%Y-%m-%d %H:%M} ({arrival:%A, %B %d, %Y})")
    seals = [str(record[f]) for f in SEAL_FIELDS if _filled(record.get(f))]
    if seals:
        lines.append(f"Seals: {', '.join(seals)}")
    return "\n".join(lines)


def load_records(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig") as f:
        data = json.load(f)
    records = data["Data"] if isinstance(data, dict) else data
    total = data.get("RecordsTotal") if isinstance(data, dict) else None
    log.info("[2/4] Loaded %d record(s) from %s%s", len(records), path.name, f" (source total: {total})" if total else "")
    return [r for r in records if r.get("Id") is not None]


def _embed(item: tuple[dict, str]) -> tuple[dict, str, list[float] | None]:
    record, text = item
    try:
        return record, text, embed_text(text)
    except Exception as exc:
        log.error("  Embedding failed for Id %s: %s", record.get("Id"), exc)
        return record, text, None


def store(path: Path, limit: int | None, force: bool) -> None:
    run_start = time.perf_counter()

    log.info("[1/4] Connecting to MongoDB...")
    ping()
    collection = get_collection()

    records = load_records(path)[:limit]
    items = [(record, build_text(record)) for record in records]

    log.info("[3/4] Embedding with %s (%d dims)...", BEDROCK_EMBEDDING_MODEL, EMBEDDING_DIMENSIONS)
    existing = {}
    if not force:
        ids = [record["Id"] for record, _ in items]
        existing = {doc["_id"]: doc for doc in collection.find({"_id": {"$in": ids}}, {"data": 1, "text": 1})}

    def unchanged(record: dict, text: str) -> bool:
        doc = existing.get(record["Id"])
        return doc is not None and doc.get("data") == record and doc.get("text") == text

    todo = [(r, t) for r, t in items if force or not unchanged(r, t)]
    skipped = len(items) - len(todo)
    if skipped:
        log.info("  %d record(s) unchanged since last run -> skipped", skipped)

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=EMBEDDING_WORKERS) as pool:
        results = list(pool.map(_embed, todo))
    embedded = [(r, t, e) for r, t, e in results if e is not None]
    failed = len(results) - len(embedded)
    log.info("[3/4] Embedded %d record(s) in %.2fs (%d failed)", len(embedded), time.perf_counter() - start, failed)

    log.info("[4/4] Saving to MongoDB...")
    start = time.perf_counter()
    now = datetime.now(timezone.utc)
    operations = [
        # Whole-document replace keeps this field order ($set would sort fields alphabetically).
        ReplaceOne(
            {"_id": record["Id"]},
            {"_id": record["Id"], "data": record, "text": text, "embedding": embedding, "stored_at": now},
            upsert=True,
        )
        for record, text, embedding in embedded
    ]
    if operations:
        result = collection.bulk_write(operations, ordered=False)
        log.info(
            "[4/4] Saved in %.2fs: %d new, %d replaced",
            time.perf_counter() - start, result.upserted_count, result.modified_count,
        )
    ensure_vector_index(collection)

    log.info(
        "DONE in %.2fs | %d stored, %d unchanged, %d failed | collection now has %d document(s)",
        time.perf_counter() - run_start, len(embedded), skipped, failed, collection.count_documents({}),
    )


def dry_run(path: Path, limit: int | None) -> None:
    records = load_records(path)[:limit or 3]
    for record in records:
        print("-" * 60)
        print(f"_id {record['Id']}  arrival {parse_arrival(record)}")
        print(build_text(record))
    print("-" * 60)
    print(f"Dry run: {len(records)} record(s) shown; nothing embedded or stored.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Store eManifest records in MongoDB with embeddings.")
    parser.add_argument("--file", type=Path, default=EMANIFEST_FILE, help="eManifest JSON export")
    parser.add_argument("--limit", type=int, help="only the first N records")
    parser.add_argument("--force", action="store_true", help="re-embed records even if unchanged")
    parser.add_argument("--dry-run", action="store_true", help="print the texts; no AWS or MongoDB calls")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    for noisy in ("pymongo", "botocore", "boto3", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.dry_run:
        dry_run(args.file, args.limit)
    else:
        store(args.file, args.limit, args.force)


if __name__ == "__main__":
    main()
