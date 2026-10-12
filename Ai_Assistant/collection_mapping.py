"""Which MongoDB collection a question is about, and how that collection is laid out.

A question is mapped to a collection by:
  1. keyword match: one of the collection's aliases appears in the question (always wins)
  2. otherwise the query planner (ONE LLM call, services/query_planner.py) decides from the
     descriptions below whether the question belongs to a collection (or to none)

The database is always the company id (company 5 -> database "5"); the collection comes
from here. To add a collection, add a CollectionSpec to COLLECTIONS.

query_fields is the whitelist of fields the planner may filter, sort or group on. Each field
has a kind, which decides the allowed operators and how values are matched (see
services/structured_query.py). Field descriptions and formats below were checked against the
real data in 5.emanifest.
"""

import re
from dataclasses import dataclass, field

# Field kinds:
#   "id"       exact identifier (case-insensitive, surrounding spaces ignored)     ops: eq, in
#   "code"     short code, exact (case-insensitive)                                ops: eq, in
#   "enum"     one of a small set of values read live from the data                ops: eq, in
#   "text"     free text; "contains" matches part of it (case-insensitive)         ops: eq, contains
#   "date8"    date stored as "YYYYMMDD"                                           ops: eq, gte, lte, between
#   "datetime" timestamp stored as "YYYY-MM-DD HH:MM:SS"                           ops: eq, gte, lte, between
OPS_BY_KIND = {
    "id": ("eq", "in"),
    "code": ("eq", "in"),
    "enum": ("eq", "in"),
    "text": ("eq", "contains"),
    "date8": ("eq", "gte", "lte", "between"),
    "datetime": ("eq", "gte", "lte", "between"),
}


@dataclass(frozen=True)
class QueryField:
    kind: str                # see OPS_BY_KIND
    description: str         # shown to the planner
    groupable: bool = False  # may be used in "count per <field>" questions


@dataclass(frozen=True)
class SortSpec:
    description: str
    # (field, how to compare): "string" sorts the stored text as is; "int" converts to a
    # number first (e.g. ArrivalTime "515" must sort before "1500").
    keys: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class CollectionSpec:
    name: str                       # MongoDB collection name (case-sensitive)
    description: str                # what it holds; the planner uses this to match questions
    aliases: tuple[str, ...]        # words that map a question to this collection directly
    text_field: str                 # readable text of one record (embedded; sent to the LLM)
    company_field: str              # field holding the company id (always filtered on, in code)
    id_fields: tuple[str, ...]      # identifiers looked up exactly
    # (code field, trip field): users often write both as one value, e.g. SCAC "19E5" + trip
    # number "TRPRPXG5S" = "19E5TRPRPXG5S" (see services/id_resolver.py).
    trip_key: tuple[str, str] | None = None
    record_label: str = "record"    # what one document is called in answers
    query_fields: dict[str, QueryField] = field(default_factory=dict)
    sorts: dict[str, SortSpec] = field(default_factory=dict)
    default_sort: str = ""          # sort used for lists when the question names none
    summary_fields: dict = field(default_factory=dict)  # name -> field, shown in logs
    vector_index: str = ""          # Atlas Vector Search index name
    filter_fields: tuple[str, ...] = ()  # fields the vector index can pre-filter on
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
    trip_key=("Scac", "TripNum"),
    record_label="eManifest",
    query_fields={
        "TripNum": QueryField("id", "trip number, e.g. '45444', 'TRP562334'"),
        "Status": QueryField("enum", "current status of the eManifest", groupable=True),
        "prevstatus": QueryField("enum", "previous status of the eManifest", groupable=True),
        "Scac": QueryField("code", "carrier SCAC code, 4 characters, e.g. '8AVL', '19E5'", groupable=True),
        "CarrierName": QueryField("text", "carrier name, e.g. 'AVAAL TECHNOLOGY SOLUTIONS INC'", groupable=True),
        "USPort": QueryField("code", "U.S. port of arrival code, 4 digits, e.g. '3801', '0101'", groupable=True),
        "USPortName": QueryField(
            "text", "U.S. port of arrival name, stored like 'DETROIT,MI' or '0101 PORTLAND,ME'; "
                    "use 'contains' with the city name", groupable=True),
        "ArrivalDate": QueryField("date8", "arrival date at the U.S. border", groupable=True),
        "DriverFullName": QueryField("text", "driver name(s), e.g. 'WILLIAMS FRANCIS'", groupable=True),
        "conveyanceNumber": QueryField("id", "truck / conveyance (unit) number"),
        "LpOne": QueryField("id", "truck license plate"),
        "Trailers": QueryField("text", "trailer numbers with plates, stored like '101 (56U 4544)'"),
        "ManifestType": QueryField("code", "manifest type code: '1' or '2'", groupable=True),
        "MessageType": QueryField("code", "message type code: '309' or '358'", groupable=True),
        "CreatedModifiedByName": QueryField("text", "user who last created or modified the eManifest", groupable=True),
        "CreatedModifiedOnDate": QueryField("datetime", "when the eManifest was last created or modified"),
    },
    sorts={
        "arrival": SortSpec("arrival date and time", (("ArrivalDate", "string"), ("ArrivalTime", "int"))),
        "modified": SortSpec("last created / modified time", (("CreatedModifiedOnDate", "string"),)),
    },
    default_sort="modified",
    summary_fields={"trip_number": "TripNum", "status": "Status", "port": "USPortName"},
    vector_index="emanifest_vector_index",
    filter_fields=("CompanyId", "USPort", "Status", "ArrivalDate", "TripNum", "Scac"),
)

COLLECTIONS: dict[str, CollectionSpec] = {spec.name: spec for spec in (EMANIFEST,)}

# Every query field must use a known kind (fails at startup, not at question time).
for _spec in COLLECTIONS.values():
    for _name, _qf in _spec.query_fields.items():
        if _qf.kind not in OPS_BY_KIND:
            raise ValueError(f"{_spec.name}.{_name}: unknown field kind '{_qf.kind}'")
    if _spec.default_sort and _spec.default_sort not in _spec.sorts:
        raise ValueError(f"{_spec.name}: default_sort '{_spec.default_sort}' is not in sorts")


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
