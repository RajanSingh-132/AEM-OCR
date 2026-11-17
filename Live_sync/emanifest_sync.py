"""eManifest live sync: AEM API -> MongoDB <company id>.emanifest (with 1024-d embeddings), every minute.

Everything for this sync is in this one file: settings, API client, sync rules, scheduler, logging.

Run from the project root:
    python -m Live_sync.emanifest_sync                         # forever, every minute (what NSSM will run)
    python -m Live_sync.emanifest_sync --once                  # one cycle, then exit (testing)
    python -m Live_sync.emanifest_sync --once --full           # read ALL pages once (first load / catch up)
    python -m Live_sync.emanifest_sync --once --companies 5    # only these companies (default: SYNC_COMPANY_IDS)

Settings in .env:
    AEM_API_BASE_URL, AEM_API_KEY      the AEM API (the key is sent as the X-API-KEY header)
    SYNC_COMPANY_IDS=2014,5            companies synced in parallel; company 5 -> MongoDB database "5"
    optional: SYNC_INTERVAL_SECONDS (60), SYNC_PAGE_SIZE (15),
              AEM_API_TIMEOUT_SECONDS (60), EMANIFEST_SEARCH_VALUE

Rules for every record from the API (key = CompanyId + Id):
    1. Id not in MongoDB                              -> embed and store
    2. Id in MongoDB, CreatedModifiedOnDate different -> delete the stored document(s), embed and store the new one
    3. Id in MongoDB, CreatedModifiedOnDate the same  -> skip (no embedding, no write)

The API returns records sorted by last change, newest first. A cycle reads page 1 and goes on to the
next page as long as every record on the page was new or changed (there is no page limit, so a
backlog, e.g. after the service was stopped, is never skipped). The first page containing an
unchanged record means everything after it is already in MongoDB. --full reads every page.
If embedding a record fails, its old document is kept and the record is retried in the next cycle.

Each cycle starts one minute after the previous one started (cycles never overlap). One company
failing (API down, MongoDB error) does not stop the others or the next cycle. Ctrl+C / service
stop ends the process after the running cycle. Logs: console + Live_sync/logs/emanifest_sync.log.

Documents have the same layout as Script_ingest/ingest_emanifest.py (and the synced data), so the
chat assistant reads them unchanged.
"""

import argparse
import logging
import os
import signal
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from logging.handlers import RotatingFileHandler
from pathlib import Path

import httpx
from dotenv import load_dotenv
from pymongo import DeleteMany, InsertOne

from Ai_Assistant.collection_mapping import EMANIFEST
from Ai_Assistant.config import EMBEDDING_WORKERS, company_db_name
from Ai_Assistant.services.embeddings import embed_text
from Ai_Assistant.services.mongo_client import ensure_vector_index_once, get_collection, ping
from Live_sync.alerts import Alerter
from Script_ingest.ingest_emanifest import build_document, build_page_content, clean_record

load_dotenv()
log = logging.getLogger("Live_sync.emanifest")


# ---------- settings (.env) ----------

def _int(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    return int(value) if value else default


AEM_API_BASE_URL = os.getenv("AEM_API_BASE_URL", "").strip().rstrip("/")
AEM_API_KEY = os.getenv("AEM_API_KEY", "").strip()
AEM_API_TIMEOUT_SECONDS = _int("AEM_API_TIMEOUT_SECONDS", 60)

SYNC_COMPANY_IDS = [int(x) for x in os.getenv("SYNC_COMPANY_IDS", "").replace(" ", "").split(",") if x]
SYNC_INTERVAL_SECONDS = _int("SYNC_INTERVAL_SECONDS", 60)  # one cycle every minute (start to start)
SYNC_PAGE_SIZE = _int("SYNC_PAGE_SIZE", 15)                # records per API page (the API allows up to 100)

EMANIFEST_ENDPOINT = "/api/EManifest/GetEManifestAll_ai"
EMANIFEST_SEARCH_VALUE = os.getenv("EMANIFEST_SEARCH_VALUE", "").strip() or "Km9wdGlvbjF8"
EMANIFEST_SORT_COLUMN = "lastUpdate"  # newest change first
MODIFIED_FIELD = "CreatedModifiedOnDate"

LOG_FILE = Path(__file__).resolve().parent / "logs" / "emanifest_sync.log"


# ---------- AEM API ----------

class AemApiError(RuntimeError):
    """The AEM API could not be reached or answered with an error."""


@lru_cache(maxsize=1)
def _http() -> httpx.Client:
    if not AEM_API_KEY:
        raise AemApiError("Missing in .env: AEM_API_KEY")
    return httpx.Client(
        base_url=AEM_API_BASE_URL,
        headers={"X-API-KEY": AEM_API_KEY, "Accept": "application/json"},
        timeout=AEM_API_TIMEOUT_SECONDS,
    )


def _params(company_id: int, page: int) -> dict:
    # No FromDate/StartDate/ToDate/EndDate: the API ignores them (tested), so every page is by last change.
    return {
        "CompanyId": company_id, "Id": 0, "SearchValue": EMANIFEST_SEARCH_VALUE,
        "PageNo": page, "PageSize": SYNC_PAGE_SIZE,
        "SortColumn": EMANIFEST_SORT_COLUMN, "SortDirection": "DESC",
    }


def fetch_page(company_id: int, page: int) -> tuple[list[dict], int | None]:
    """One page of eManifests -> (records in "Data", "RecordsTotal"). Raises AemApiError on any failure."""
    try:
        response = _http().get(EMANIFEST_ENDPOINT, params=_params(company_id, page))
    except httpx.HTTPError as exc:
        raise AemApiError(f"request failed ({type(exc).__name__})") from exc
    if response.status_code != 200:
        raise AemApiError(f"HTTP {response.status_code}")
    try:
        body = response.json()
    except ValueError as exc:
        raise AemApiError("response is not JSON") from exc

    if isinstance(body, list):
        return body, None
    if not isinstance(body, dict):
        raise AemApiError(f"unexpected response type {type(body).__name__}")
    records = body.get("Data") or []
    if not isinstance(records, list):
        raise AemApiError("'Data' is not a list")
    return records, body.get("RecordsTotal")


# ---------- sync rules ----------

@dataclass
class SyncStats:
    company_id: int
    fetched: int = 0
    inserted: int = 0
    updated: int = 0
    unchanged: int = 0
    failed: int = 0
    pages: int = 0

    def __str__(self) -> str:
        return (f"company {self.company_id}: {self.fetched} fetched ({self.pages} page(s)) | "
                f"{self.inserted} new, {self.updated} updated, {self.unchanged} unchanged, {self.failed} failed")


def _modified(value) -> str:
    return "" if value is None else str(value).strip()


def _stored_versions(collection, company_id: int, ids: list) -> dict:
    """{Id: {CreatedModifiedOnDate of each stored document with that Id}} (one query)."""
    stored: dict = defaultdict(set)
    for doc in collection.find({"CompanyId": company_id, "Id": {"$in": ids}}, {"_id": 0, "Id": 1, MODIFIED_FIELD: 1}):
        stored[doc["Id"]].add(_modified(doc.get(MODIFIED_FIELD)))
    return stored


def _embed(item: tuple[dict, str]) -> tuple[dict, str, list[float] | None]:
    fields, text = item
    try:
        return fields, text, embed_text(text)
    except Exception as exc:
        log.error("  Embedding failed for company %s eManifest Id %s: %s", fields.get("CompanyId"), fields.get("Id"), exc)
        return fields, text, None


def _process_page(collection, company_id: int, records: list[dict], stats: SyncStats, workers: int) -> bool:
    """Apply rules 1-3 to one page. Returns True when every record on the page was new or changed."""
    records = [r for r in records if isinstance(r, dict) and r.get("Id") is not None]
    # Same Id twice on a page: keep the first (= most recent change).
    records = list({r["Id"]: r for r in reversed(records)}.values())[::-1]
    stored = _stored_versions(collection, company_id, [r["Id"] for r in records])

    todo: list[tuple[dict, str]] = []
    is_update: dict = {}
    for record in records:
        fields = clean_record(record, company_id)
        versions = stored.get(fields["Id"])
        if versions is None:                                       # rule 1: new
            is_update[fields["Id"]] = False
        elif versions != {_modified(fields.get(MODIFIED_FIELD))}:  # rule 2: changed
            is_update[fields["Id"]] = True
        else:                                                      # rule 3: unchanged
            stats.unchanged += 1
            continue
        todo.append((fields, build_page_content(fields)))

    if todo:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_embed, todo))
        now = datetime.now(timezone.utc)
        source = f"api:{AEM_API_BASE_URL}{EMANIFEST_ENDPOINT}?CompanyId={company_id}"
        operations = []
        for fields, text, embedding in results:
            if embedding is None:
                stats.failed += 1
                continue
            document = build_document(fields, text, embedding, source, now)
            document["metadata"]["source_document"] = source
            if is_update[fields["Id"]]:
                operations.append(DeleteMany({"CompanyId": company_id, "Id": fields["Id"]}))
                stats.updated += 1
            else:
                stats.inserted += 1
            operations.append(InsertOne(document))
        if operations:
            # ordered=True: each delete runs before the insert of the new version.
            collection.bulk_write(operations, ordered=True)

    return len(todo) == len(records) and len(records) > 0


def sync_company(company_id: int, full: bool = False, workers: int = EMBEDDING_WORKERS) -> SyncStats:
    start = time.perf_counter()
    stats = SyncStats(company_id)
    collection = get_collection(company_db_name(company_id), EMANIFEST.name)

    page = 1
    while not _stop.is_set():
        records, total = fetch_page(company_id, page)
        stats.pages, stats.fetched = page, stats.fetched + len(records)
        if not records:
            break
        all_new_or_changed = _process_page(collection, company_id, records, stats, workers)
        last_page = len(records) < SYNC_PAGE_SIZE or (total is not None and stats.fetched >= total)
        if last_page or (not full and not all_new_or_changed):
            break
        page += 1

    if stats.inserted or stats.updated:
        ensure_vector_index_once(collection, EMANIFEST.vector_index, EMANIFEST.filter_fields, EMANIFEST.embedding_field)
    log.info("  eManifest %s in %.2fs", stats, time.perf_counter() - start)
    return stats


# ---------- scheduler ----------

_stop = threading.Event()


def _sync_safely(company_id: int, full: bool, workers: int) -> None:
    try:
        sync_company(company_id, full=full, workers=workers)
    except Exception as exc:
        log.error("  eManifest sync failed for company %s: %s", company_id, exc)


def run_cycle(company_ids: list[int], full: bool) -> None:
    start = time.perf_counter()
    # Share the embedding parallelism between the companies running at the same time.
    workers = max(1, EMBEDDING_WORKERS // len(company_ids))
    log.info("Sync cycle started for companies %s", ", ".join(map(str, company_ids)))
    with ThreadPoolExecutor(max_workers=len(company_ids)) as pool:
        for company_id in company_ids:
            pool.submit(_sync_safely, company_id, full, workers)
    log.info("Sync cycle finished in %.2fs", time.perf_counter() - start)


def _setup_logging() -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s  %(levelname)-7s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    console = logging.StreamHandler()
    file = RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=5, encoding="utf-8")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in (console, file):
        handler.setFormatter(formatter)
        root.addHandler(handler)
    for noisy in ("pymongo", "botocore", "boto3", "urllib3", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _request_stop(signum, _frame) -> None:
    log.info("Stop requested (signal %s); finishing the current cycle...", signum)
    _stop.set()


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync eManifests from the AEM API into MongoDB every minute.")
    parser.add_argument("--once", action="store_true", help="run one cycle and exit")
    parser.add_argument("--full", action="store_true", help="read all pages, not only the new changes")
    parser.add_argument("--companies", help="comma-separated company ids (default: SYNC_COMPANY_IDS in .env)")
    args = parser.parse_args()

    _setup_logging()
    company_ids = [int(x) for x in args.companies.split(",") if x.strip()] if args.companies else SYNC_COMPANY_IDS
    if not company_ids:
        raise SystemExit("No companies to sync: set SYNC_COMPANY_IDS in .env (e.g. 2014,5) or pass --companies")

    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):  # SIGBREAK: Ctrl+Break / NSSM console stop on Windows
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), _request_stop)

    if args.once:  # manual test run: no alert emails
        ping()
        run_cycle(company_ids, args.full)
        return

    alerter = Alerter("AEM eManifest Sync", LOG_FILE)
    logging.getLogger().addHandler(alerter)
    alerter.service_started(company_ids)
    try:
        ping()
        log.info("eManifest live sync running every %ds for companies %s",
                 SYNC_INTERVAL_SECONDS, ", ".join(map(str, company_ids)))
        while not _stop.is_set():
            started = time.monotonic()
            try:
                run_cycle(company_ids, args.full)
            except Exception as exc:  # never let one bad cycle kill the service
                log.exception("Sync cycle crashed: %s", exc)
            alerter.end_cycle()
            _stop.wait(max(0.0, SYNC_INTERVAL_SECONDS - (time.monotonic() - started)))
    except BaseException as exc:
        log.exception("eManifest live sync crashed: %s", exc)
        alerter.service_stopped(f"crashed: {type(exc).__name__}: {exc}", crashed=True)
        raise
    log.info("eManifest live sync stopped")
    alerter.service_stopped("stop requested (service stop / Ctrl+C)")


if __name__ == "__main__":
    main()
