"""MongoDB connection (one cached client) and the Atlas Vector Search index."""

import logging
import time
from functools import lru_cache

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import OperationFailure
from pymongo.operations import SearchIndexModel

from Ai_Assistant.config import (
    EMBEDDING_DIMENSIONS,
    MONGO_COLLECTION,
    MONGO_URI,
    VECTOR_FILTER_FIELDS,
    VECTOR_INDEX_NAME,
    company_db_name,
    require,
)

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_client() -> MongoClient:
    require(("MONGO_URI", MONGO_URI), ("COLLECTION_NAME", MONGO_COLLECTION))
    return MongoClient(MONGO_URI, appname="AEM-AI", serverSelectionTimeoutMS=10_000)


class CompanyDataNotFound(LookupError):
    """The company's database / collection does not exist yet or holds no records."""


def get_collection(db_name: str, collection_name: str = MONGO_COLLECTION) -> Collection:
    return get_client()[db_name][collection_name]


def get_company_collection(company_id: int, collection_name: str) -> Collection:
    """The collection in the database named after the company id (company 5 -> database "5").

    Raises CompanyDataNotFound when that database / collection does not exist or is empty.
    """
    db = get_client()[company_db_name(company_id)]
    if not db.list_collection_names(filter={"name": collection_name}):
        raise CompanyDataNotFound(f"No '{db.name}.{collection_name}' collection")
    collection = db[collection_name]
    if collection.estimated_document_count() == 0:
        raise CompanyDataNotFound(f"'{db.name}.{collection_name}' is empty")
    return collection


def ping() -> None:
    """Check the connection; raises if MongoDB cannot be reached."""
    start = time.perf_counter()
    get_client().admin.command("ping")
    log.info("MongoDB connected (%.2fs)", time.perf_counter() - start)


def ensure_vector_index(
    collection: Collection,
    index_name: str = VECTOR_INDEX_NAME,
    filter_fields: tuple[str, ...] = VECTOR_FILTER_FIELDS,
    embedding_field: str = "embedding",
) -> None:
    """Create the Atlas Vector Search index, or update it if its definition changed.

    Atlas builds the index in the background; it becomes queryable after a minute or so.
    """
    definition = {
        "fields": [
            {"type": "vector", "path": embedding_field, "numDimensions": EMBEDDING_DIMENSIONS, "similarity": "cosine"},
            *({"type": "filter", "path": field} for field in filter_fields),
        ]
    }
    where = f"{collection.database.name}.{collection.name}"
    try:
        existing = {index["name"]: index for index in collection.list_search_indexes()}
    except OperationFailure as exc:
        log.warning("Could not list search indexes on %s (Atlas Vector Search unavailable?): %s", where, exc)
        return

    try:
        index = existing.get(index_name)
        if index is None:
            collection.create_search_index(SearchIndexModel(definition=definition, name=index_name, type="vectorSearch"))
            log.info("Vector index '%s' created on %s (%d dims, cosine); Atlas is building it", index_name, where, EMBEDDING_DIMENSIONS)
        elif index.get("latestDefinition") != definition:
            collection.update_search_index(index_name, definition)
            log.info("Vector index '%s' on %s updated; Atlas is rebuilding it", index_name, where)
        else:
            log.info("Vector index '%s' on %s is up to date", index_name, where)
    except OperationFailure as exc:
        log.warning("Could not create/update vector index '%s' on %s: %s", index_name, where, exc)


_checked_indexes: set[tuple[str, str, str]] = set()


def ensure_vector_index_once(collection: Collection, index_name: str, filter_fields: tuple[str, ...],
                             embedding_field: str) -> None:
    """ensure_vector_index, but only the first time per collection in this server process."""
    key = (collection.database.name, collection.name, index_name)
    if key not in _checked_indexes:
        ensure_vector_index(collection, index_name, filter_fields, embedding_field)
        _checked_indexes.add(key)
