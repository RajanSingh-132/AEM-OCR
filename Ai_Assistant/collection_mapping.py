"""Which MongoDB collection a question is about, and how that collection is laid out.

A question is mapped to a collection in two steps (see services/collection_router.py):
  1. keyword match: one of the collection's aliases appears in the question
  2. otherwise, ONE LLM call decides from the descriptions below whether the question
     belongs to a collection (or to none)

The database is always the company id (company 5 -> database "5"); the collection comes
from here. To add a collection, add a CollectionSpec to COLLECTIONS.
"""

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class CollectionSpec:
    name: str                       # MongoDB collection name (case-sensitive)
    description: str                # what it holds; the LLM uses this to match questions
    aliases: tuple[str, ...]        # words that map a question to this collection directly
    text_field: str                 # readable text that was embedded (sent to the LLM)
    company_field: str              # field holding the company id (always filtered on)
    id_fields: tuple[str, ...]      # identifiers looked up exactly when they appear in a question
    summary_fields: dict = field(default_factory=dict)  # name -> field, shown in logs
    vector_index: str = ""          # Atlas Vector Search index name
    filter_fields: tuple[str, ...] = ()  # fields the vector search may pre-filter on
    embedding_field: str = "embedding"


EMANIFEST_ALIASES = [
    "emanifest",
    "Emanifest",
    "eManifest",
    "EManifest",
    "EMANIFEST",
    "emanifests",
    "Emanifests",
    "eManifests",
    "EMANIFESTS",
    "e-manifest",
    "E-manifest",
    "E-MANIFEST",
    "e-manifests",
    "E-Manifests",
    "E-MANIFESTS",
    "e manifest",
    "E manifest",
    "E Manifest",
    "e manifests",
    "E manifests",
    "E Manifests",
    "electronic manifest",
    "Electronic Manifest",
    "ELECTRONIC MANIFEST",
    "electronic manifests",
    "Electronic Manifests",
    "manifest",
    "manifests",
    "e manifest record",
    "e manifest records",
    "manifest record",
    "manifest records",
    "manifest details",
    "emanifest details",
    "emanifest data",
    "emanifest module",
]

EMANIFEST = CollectionSpec(
    name="emanifest",
    description=(
        "ACE eManifest records (U.S. CBP truck eManifests): trips and trip numbers, filing status "
        "(Draft, Ready, ...), carrier and SCAC, U.S. port of arrival, arrival date and time, "
        "in-transit flag, manifest type and code, driver, truck / conveyance number, license plate, "
        "trailers, seals, shipment control and reference numbers, remarks, version, and who "
        "created or modified a record and when."
    ),
    aliases=tuple(EMANIFEST_ALIASES),
    text_field="page_content",
    company_field="CompanyId",
    id_fields=("TripNum", "conveyanceNumber", "LpOne", "ShipmentControl", "ReferenceID"),
    summary_fields={"trip_number": "TripNum", "status": "Status", "port": "USPortName"},
    vector_index="emanifest_vector_index",
    filter_fields=("CompanyId", "USPort", "Status", "ArrivalDate", "TripNum", "Scac"),
)

COLLECTIONS: dict[str, CollectionSpec] = {spec.name: spec for spec in (EMANIFEST,)}


def _alias_pattern(alias: str) -> re.Pattern:
    # Case-insensitive, whole words only ("manifest" must not match inside "manifestation"),
    # and any spacing between words ("e  manifest").
    words = r"\s+".join(re.escape(word) for word in alias.split())
    return re.compile(rf"(?<![A-Za-z0-9]){words}(?![A-Za-z0-9])", re.IGNORECASE)


# Longest aliases first, so the log shows the most specific match.
_ALIAS_PATTERNS = sorted(
    {(alias.lower(), spec.name) for spec in COLLECTIONS.values() for alias in spec.aliases},
    key=lambda item: -len(item[0]),
)
_ALIAS_PATTERNS = [(alias, name, _alias_pattern(alias)) for alias, name in _ALIAS_PATTERNS]


def match_by_keywords(question: str) -> tuple[CollectionSpec, str] | None:
    """(collection, matched alias) when an alias appears in the question, else None."""
    for alias, name, pattern in _ALIAS_PATTERNS:
        if pattern.search(question):
            return COLLECTIONS[name], alias
    return None
