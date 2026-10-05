"""MongoDB connection (one cached client) and the Atlas Vector Search index."""

import logging
import time
from functools import lru_cache

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import OperationFailure
from pymongo.operations import SearchIndexModel

from Ai_Assistant.config import (
    COMPANY_ROUTES,
    EMBEDDING_DIMENSIONS,
    MONGO_COLLECTION,
    MONGO_DB,
    MONGO_URI,
    VECTOR_FILTER_FIELDS,
    VECTOR_INDEX_NAME,
    require,
)

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_client() -> MongoClient:
    require(("MONGO_URI", MONGO_URI), ("DB", MONGO_DB), ("COLLECTION_NAME", MONGO_COLLECTION))
    return MongoClient(MONGO_URI, appname="AEM-AI", serverSelectionTimeoutMS=10_000)


class UnknownCompanyError(LookupError):
    pass


def get_collection(db_name: str = MONGO_DB, collection_name: str = MONGO_COLLECTION) -> Collection:
    return get_client()[db_name][collection_name]


def get_company_collection(company_id: int) -> Collection:
    """The collection holding this company's eManifests (routing from COMPANY_ROUTES)."""
    route = COMPANY_ROUTES.get(company_id)
    if route is None:
        raise UnknownCompanyError(f"No eManifest data is configured for company {company_id}.")
    return get_collection(*route)


def ping() -> None:
    """Check the connection; raises if MongoDB cannot be reached."""
    start = time.perf_counter()
    get_client().admin.command("ping")
    log.info("MongoDB connected: db '%s', collection '%s' (%.2fs)", MONGO_DB, MONGO_COLLECTION, time.perf_counter() - start)


def ensure_vector_index(collection: Collection) -> None:
    """Create the Atlas Vector Search index on "embedding", or update it if its definition changed.

    Atlas builds the index in the background; it becomes queryable after a minute or so.
    """
    definition = {
        "fields": [
            {"type": "vector", "path": "embedding", "numDimensions": EMBEDDING_DIMENSIONS, "similarity": "cosine"},
            *({"type": "filter", "path": field} for field in VECTOR_FILTER_FIELDS),
        ]
    }
    try:
        existing = {index["name"]: index for index in collection.list_search_indexes()}
    except OperationFailure as exc:
        log.warning("Could not list search indexes (Atlas Vector Search unavailable?): %s", exc)
        return

    try:
        index = existing.get(VECTOR_INDEX_NAME)
        if index is None:
            collection.create_search_index(
                SearchIndexModel(definition=definition, name=VECTOR_INDEX_NAME, type="vectorSearch")
            )
            log.info("Vector index '%s' created (%d dims, cosine); Atlas is building it", VECTOR_INDEX_NAME, EMBEDDING_DIMENSIONS)
        elif index.get("latestDefinition") != definition:
            collection.update_search_index(VECTOR_INDEX_NAME, definition)
            log.info("Vector index '%s' updated to the new definition; Atlas is rebuilding it", VECTOR_INDEX_NAME)
        else:
            log.info("Vector index '%s' is up to date", VECTOR_INDEX_NAME)
    except OperationFailure as exc:
        log.warning("Could not create/update vector index '%s': %s", VECTOR_INDEX_NAME, exc)
