"""Which MongoDB collection a question is about, and how that collection is laid out.

A question is mapped to a collection by:
  1. keyword match: one of the collection's aliases appears in the question (always wins).
     Program words come first, in three levels (a level that points to both collections is
     skipped, so the next one decides), then the record words:
       a. program:   ACE / ACEs / PAPS -> emanifest;  ACI / ACIs / PARS -> trips
       b. direction: into / to / entering / towards USA, USA-bound, from Canada -> emanifest;
                     into / to / entering / towards Canada, Canada-bound, from USA -> trips
       c. country:   USA / US / America / American / CBP -> emanifest;
                     Canada / Canadian / CBSA -> trips
     So "ACE trips" -> emanifest and "ACI eManifest" -> trips, whatever record word follows.
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
    # Program words, checked before every other alias (see the module docstring), in order:
    # program names (ACE, PARS, ...), direction ("into USA", "from Canada", ...), country /
    # agency ("Canadian", "CBP", ...). So "ACE trips" -> emanifest, "ACI eManifest" -> trips.
    program_aliases: tuple[str, ...] = ()
    direction_aliases: tuple[str, ...] = ()
    country_aliases: tuple[str, ...] = ()
    # Statuses only this program has (ACE "Ready" / "Held", ACI "InChange"): checked after the
    # strong aliases and before the weak ones, so "how many trips are ready?" -> emanifest.
    status_aliases: tuple[str, ...] = ()
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

# ---------- program words (ACE = into the USA, ACI = into Canada) ----------
# "US" in capitals only ("show us trips" is English, "show US trips" is the country): any
# alias with the word "US" is matched case-sensitively (see _alias_pattern).
_US_NAMES = ("USA", "U.S.A", "US", "U.S", "United States", "America")
_CANADA_NAMES = ("Canada",)
_TOWARDS = ("into", "to", "entering", "toward", "towards", "bound for")


def _direction(to_names, from_names) -> tuple[str, ...]:
    """'into USA', 'to the USA', 'USA-bound', ... for to_names; 'from Canada', ... for from_names."""
    words = []
    for name in to_names:
        words += [f"{verb} {the}{name}" for verb in _TOWARDS for the in ("", "the ")]
        words += [f"{name}-bound", f"{name} bound", f"{name}bound"]
    for name in from_names:
        words += [f"from {the}{name}" for the in ("", "the ")]
    return tuple(words)

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
    program_aliases=("ACE", "ACEs", "PAPS"),
    direction_aliases=_direction(_US_NAMES, _CANADA_NAMES),  # into the USA / from Canada
    country_aliases=_US_NAMES + ("American", "CBP"),
    # ACE statuses: Draft, Ready, Sending, Accepted, Rejected, Deleting, Deleted, Arrived, Held,
    # Released, Failed (+ Arrived with amending). Only these are not ACI statuses:
    status_aliases=("Ready", "Held", "Amending", "Amended", "Amendment"),
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
    program_aliases=("ACI", "ACIs", "PARS"),
    direction_aliases=_direction(_CANADA_NAMES, _US_NAMES),  # into Canada / from the USA
    country_aliases=_CANADA_NAMES + ("Canadian", "CBSA"),
    # ACI statuses: Draft, Sending, Accepted, Rejected, Arrived, Released, Deleting, Deleted,
    # InChange, Failed. Only this one is not an ACE status:
    status_aliases=("InChange", "In-Change"),
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


def _case_sensitive(alias: str) -> bool:
    # "US" is the country only in capitals; "us" is English ("show us", "send to us").
    return "US" in alias.split()


def _alias_pattern(alias: str) -> re.Pattern:
    # Whole words only ("manifest" must not match inside "manifestation", "ACE" not inside
    # "place"), any spacing between words ("e  manifest"), "-" or space in "USA-bound".
    # Case-insensitive, except aliases with the word "US".
    words = r"\s+".join(re.escape(word) for word in alias.split()).replace(r"\-", r"[-\s]?")
    flags = 0 if _case_sensitive(alias) else re.IGNORECASE
    return re.compile(rf"(?<![A-Za-z0-9]){words}(?![A-Za-z0-9])", flags)


def _patterns(attr: str) -> list[tuple[str, str, re.Pattern]]:
    # One pattern per alias (case variants merged); longest first, so the log shows the most
    # specific match.
    pairs = {(a if _case_sensitive(a) else a.lower(), spec.name)
             for spec in COLLECTIONS.values() for a in getattr(spec, attr)}
    return [(alias, name, _alias_pattern(alias)) for alias, name in sorted(pairs, key=lambda p: -len(p[0]))]


# Program words (three levels), then strong aliases, then program-only statuses, then weak aliases.
_PROGRAM_LEVELS = [_patterns("program_aliases"), _patterns("direction_aliases"), _patterns("country_aliases")]
_STRONG_PATTERNS = _patterns("aliases")
_STATUS_PATTERNS = _patterns("status_aliases")
_WEAK_PATTERNS = _patterns("weak_aliases")


def _single_side(question: str, patterns) -> tuple[CollectionSpec, str] | None:
    """(collection, alias) when the patterns that match all point to ONE collection, else None."""
    hits = {}
    for alias, name, pattern in patterns:
        if name not in hits and pattern.search(question):
            hits[name] = alias
    if len(hits) == 1:
        name, alias = next(iter(hits.items()))
        return COLLECTIONS[name], alias
    return None


def named_programs(question: str) -> set[str]:
    """Collections whose program names (ACE / PAPS, ACI / PARS) appear in the question."""
    return {name for _, name, pattern in _PROGRAM_LEVELS[0] if pattern.search(question)}


def match_program(question: str) -> tuple[CollectionSpec, str] | None:
    """(collection, word) when the question names ACE or ACI (program, direction or country words)."""
    for level in _PROGRAM_LEVELS:
        hit = _single_side(question, level)
        if hit:
            return hit
    return None


def match_by_keywords(question: str) -> tuple[CollectionSpec, str] | None:
    """(collection, matched alias) when an alias appears in the question, else None.

    Priority: program words in three levels (program name, direction, country / agency; see
    the module docstring), then strong aliases ("ACI trip", "emanifest", "manifest", ...),
    then statuses only one program has ("Ready", "Held" / "InChange"), then weak aliases
    ("trip", ...). A level whose words point to both collections ("ACE and ACI") does not
    decide; the next level does.
    """
    hit = match_program(question)
    if hit:
        return hit
    for alias, name, pattern in _STRONG_PATTERNS:
        if pattern.search(question):
            return COLLECTIONS[name], alias
    hit = _single_side(question, _STATUS_PATTERNS)
    if hit:
        return hit
    for alias, name, pattern in _WEAK_PATTERNS:
        if pattern.search(question):
            return COLLECTIONS[name], alias
    return None


# ---------- status questions that could be ACE or ACI ----------
# Words of a status question; the statuses here exist in BOTH programs (ACE-only / ACI-only
# statuses are status_aliases above and decide on their own).
_STATUS_QUESTION = re.compile(
    r"(?<![A-Za-z0-9])(status|statuses|status-?wise|draft|drafts|sending|accepted|rejected|"
    r"deleting|deleteing|deleted|arrived|released|failed)(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_WEAK_WORDS = {alias for alias, _, _ in _WEAK_PATTERNS}

# Misspelled status words ("dratf", "rejcted", "arived", "stauts") count too: one letter wrong,
# missing, extra, or two letters swapped. Statuses of one program only ("ready", "held") are
# left out: "read", "head", "hold" are one letter away. Real words one letter away from a
# status are not statuses ("trips that arrive tomorrow").
_FUZZY_STATUS_WORDS = ("status", "draft", "sending", "accepted", "rejected", "deleting",
                       "deleted", "arrived", "released", "failed")
_NOT_STATUS_WORDS = {"arrive", "arrives", "release", "releases", "delete", "deletes", "deleter"}


def _one_edit_apart(a: str, b: str) -> bool:
    """True when a and b differ by one letter changed / added / removed, or two neighbours swapped."""
    if a == b or abs(len(a) - len(b)) > 1:
        return a == b
    if len(a) == len(b):
        diff = [i for i in range(len(a)) if a[i] != b[i]]
        return len(diff) == 1 or (len(diff) == 2 and diff[1] == diff[0] + 1
                                  and a[diff[0]] == b[diff[1]] and a[diff[1]] == b[diff[0]])
    short, long_ = (a, b) if len(a) < len(b) else (b, a)
    return any(long_[:i] + long_[i + 1:] == short for i in range(len(long_)))


def _is_status_question(question: str) -> bool:
    if _STATUS_QUESTION.search(question):
        return True
    for word in re.findall(r"[A-Za-z]{4,}", question.lower()):
        if word in _NOT_STATUS_WORDS:
            continue
        if any(_one_edit_apart(word, status) for status in _FUZZY_STATUS_WORDS):
            return True
    return False


def _has_identifier(question: str) -> bool:
    # A trip number / plate / SCAC+trip value: letters and digits mixed ("TRP562334",
    # "19E5TRPRPXG5S") or 5+ digits ("45444"). Dates ("2026-11-20") and counts ("top 5") are not.
    for token in re.findall(r"[A-Za-z0-9]+", question):
        has_digit = any(c.isdigit() for c in token)
        if has_digit and (len(token) >= 5 or any(c.isalpha() for c in token)):
            return True
    return False


def needs_program_choice(question: str) -> bool:
    """True for a status question that does not say ACE or ACI ("draft trips status wise
    summary", "status summary"): both programs have trips and the same statuses, so the
    assistant asks which one instead of guessing. Questions with an identifier ("status of
    trip 45444") keep the normal flow: the ID is looked up in both collections.
    """
    if not _is_status_question(question) or _has_identifier(question):
        return False
    return clear_collection(question) is None


def clear_collection(question: str) -> CollectionSpec | None:
    """The collection when the question says it clearly (ACE / ACI words, "eManifest",
    "manifest", ACE-only or ACI-only statuses); None for nothing or only "trip" words, which
    both programs use."""
    hit = match_by_keywords(question)
    return hit[0] if hit and hit[1] not in _WEAK_WORDS else None
