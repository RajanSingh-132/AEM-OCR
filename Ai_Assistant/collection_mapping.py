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
    # Weaker words: they decide only when no collection's `aliases` word is in the question
    # (e.g. "trip": every eManifest has a trip number too, so "eManifest of trip X" stays eManifest).
    weak_aliases: tuple[str, ...] = ()
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
        "ACE eManifests (U.S. CBP truck eManifests, entering the United States): eManifest trip "
        "numbers, filing status (Draft, Ready, ...), carrier and SCAC, U.S. port of arrival, arrival date and time, "
        "in-transit flag, manifest type and code, driver, truck / conveyance number, license plate, "
        "trailers, seals, shipment control and reference numbers, remarks, version, and who "
        "created or modified a record and when."
    ),
    aliases=tuple(EMANIFEST_ALIASES),
    text_field="page_content",
    company_field="CompanyId",
    id_fields=("TripNum", "conveyanceNumber", "LpOne", "ShipmentControl", "ReferenceID"),
    trip_key=("Scac", "TripNum"),
    record_label="ACE eManifest",
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

TRIP_ALIASES = [
    "Trip",
    "trip",
    "TRIP",
    "Trips",
    "trips",
    "Trip details",
    "Trip information",
    "Trip number",
    "Trip status",
    "ACI Trip",
    "ACI trip",
    "Canadian trip",
]
# ACI-specific words always mean the trips collection; the plain "trip" words decide only
# when no eManifest word is in the question.
_TRIP_STRONG = {"aci trip", "canadian trip"}

# ACI trips (Canada). Checked against 2014.trips: ports exist only as FPOA codes (port names
# are empty), status is in TripStatusName (the Status field is empty), and driver, carrier
# and cargo control fields are empty, so they are not offered as filters.
TRIPS = CollectionSpec(
    name="trips",
    description=(
        "ACI trips (CBSA truck trips, entering Canada): trip number, SCAC, first port of arrival "
        "code (FPOA, 4 digits; port names are not stored), arrival date and time, trip status "
        "(Draft, Sending, Rejected, Deleting), truck number and plate, trailers with plates, truck "
        "seal, flags (empty truck, LVS, IIT, fly truck, postal), and who created or modified the "
        "trip and when."
    ),
    aliases=tuple(a for a in TRIP_ALIASES if a.lower() in _TRIP_STRONG),
    weak_aliases=tuple(a for a in TRIP_ALIASES if a.lower() not in _TRIP_STRONG),
    text_field="page_content",
    company_field="CompanyId",
    id_fields=("TripNo", "TruckNo", "TruckPlate"),
    trip_key=("scac", "TripNo"),
    record_label="ACI trip",
    query_fields={
        "TripNo": QueryField("id", "trip number, e.g. '5465634'"),
        "TripStatusName": QueryField("enum", "current status of the trip", groupable=True),
        "scac": QueryField("code", "carrier SCAC code, e.g. 'AFM1', 'AF-1'", groupable=True),
        "FPOA": QueryField("code", "first port of arrival in Canada, 4-digit code, e.g. '0318'", groupable=True),
        "arrivalDate": QueryField("date8", "arrival date at the Canadian border", groupable=True),
        "TruckNo": QueryField("id", "truck (unit) number"),
        "TruckPlate": QueryField("id", "truck license plate"),
        "Trailers": QueryField("text", "trailer numbers with plates, stored like 'AS23165 (SDA6549846)'"),
        "CreatedModifiedByName": QueryField("text", "user who last created or modified the trip", groupable=True),
        "CreatedModifiedOnDate": QueryField("datetime", "when the trip was last created or modified"),
    },
    sorts={
        "arrival": SortSpec("arrival date and time", (("arrivalDate", "string"), ("arrivalTime", "int"))),
        "modified": SortSpec("last created / modified time", (("CreatedModifiedOnDate", "string"),)),
    },
    default_sort="modified",
    summary_fields={"trip_number": "TripNo", "status": "TripStatusName", "port": "FPOA"},
    vector_index="trips_vector_index",
    filter_fields=("CompanyId", "FPOA", "TripStatusName", "arrivalDate", "TripNo", "scac"),
)

COLLECTIONS: dict[str, CollectionSpec] = {spec.name: spec for spec in (EMANIFEST, TRIPS)}

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


def _patterns(attr: str) -> list[tuple[str, str, re.Pattern]]:
    # Longest aliases first, so the log shows the most specific match.
    pairs = sorted({(a.lower(), spec.name) for spec in COLLECTIONS.values() for a in getattr(spec, attr)},
                   key=lambda item: -len(item[0]))
    return [(alias, name, _alias_pattern(alias)) for alias, name in pairs]


# Strong aliases of every collection are checked before any weak alias.
_ALIAS_PATTERNS = _patterns("aliases") + _patterns("weak_aliases")


def match_by_keywords(question: str) -> tuple[CollectionSpec, str] | None:
    """(collection, matched alias) when an alias appears in the question, else None.

    Priority: strong aliases ("ACI trip", "emanifest", "manifest", ...) of any collection,
    then weak aliases ("trip", "trip details", ...).
    """
    for alias, name, pattern in _ALIAS_PATTERNS:
        if pattern.search(question):
            return COLLECTIONS[name], alias
    return None
