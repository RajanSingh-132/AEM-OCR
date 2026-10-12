"""Resolve identifiers from a question into exact conditions (code, not LLM).

Users mostly write an eManifest as SCAC + trip number in one value ("19E5TRPRPXG5S",
"19E5 TRPRPXG5S", "19E5-TRPRPXG5S"), but the data stores them in two fields (Scac, TripNum).
Users also write only the trip number, only the SCAC, or both separately ("SCAC 19E5 trip
TRPRPXG5S"). For every value, all exact readings are tried against the real data:

  1. SCAC + trip number   the value starts with a REAL SCAC code (read from the data) and the
                          rest is the trip number; or a SCAC value + another value given separately
  2. trip number          TripNum equals the value (some stored trip numbers already include
                          the SCAC, e.g. "8AVLJKLJ7877")
  3. other identifiers    truck / conveyance number, license plate, shipment control, reference
  4. SCAC only            only when 1-3 found nothing and the value is a real SCAC code

Every reading that matches at least one record is kept and labeled (one value can mean two
different manifests, e.g. "19E556546" = SCAC 19E5 + trip 56546, and also trip "19E556546").
"""

import logging
import re
import time
from dataclasses import dataclass, field

from pymongo.collection import Collection

from Ai_Assistant.collection_mapping import CollectionSpec
from Ai_Assistant.config import ENUM_VALUES_CACHE_SECONDS
from Ai_Assistant.services.structured_query import exact_any

log = logging.getLogger(__name__)

_MAX_VALUES = 20


@dataclass
class IdResolution:
    condition: dict | None                       # OR of all matching readings; None = nothing matched
    lines: list[str] = field(default_factory=list)  # one description per value, for the answer LLM


def normalize(value: str) -> str:
    """Upper case without spaces, hyphens, slashes, dots or underscores."""
    return re.sub(r"[\s\-/._]", "", str(value)).upper()


# ---------- real SCAC codes (cached per company) ----------

_scac_cache: dict[tuple, tuple[float, set[str]]] = {}


def known_codes(collection: Collection, spec: CollectionSpec, company_id: int) -> set[str]:
    if not spec.trip_key:
        return set()
    key = (collection.database.name, collection.name, company_id)
    cached = _scac_cache.get(key)
    if cached and time.monotonic() - cached[0] < ENUM_VALUES_CACHE_SECONDS:
        return cached[1]
    codes = {normalize(v) for v in collection.distinct(spec.trip_key[0], {spec.company_field: company_id}) if v}
    _scac_cache[key] = (time.monotonic(), codes)
    return codes


# ---------- resolution ----------

def _id_label(spec: CollectionSpec, id_field: str) -> str:
    if spec.trip_key and id_field == spec.trip_key[1]:
        return "trip number"
    query_field = spec.query_fields.get(id_field)
    return query_field.description.split(",")[0] if query_field else id_field


def _variants(value: str) -> list[str]:
    raw = " ".join(str(value).split())
    return list(dict.fromkeys([raw, normalize(raw)]))


def resolve_ids(collection: Collection, spec: CollectionSpec, company_id: int,
                values: list[str], base_match: dict, allow_code_only: bool = True) -> IdResolution:
    """Exact conditions for the identifiers in `values`; counts are within `base_match`.

    allow_code_only=False skips reading 4 (a SCAC alone selecting all of its eManifests).
    """
    values = [v for v in dict.fromkeys(str(v).strip() for v in values) if v][:_MAX_VALUES]
    if not values:
        return IdResolution(None)
    code_field, trip_field = spec.trip_key if spec.trip_key else (None, None)
    codes = known_codes(collection, spec, company_id)

    def count(condition: dict) -> int:
        return collection.count_documents({"$and": [base_match, condition]})

    matched: list[dict] = []
    lines: list[str] = []
    used: set[str] = set()

    # A SCAC given as its own value + another value: "SCAC 19E5 trip TRPRPXG5S", ["19E5", "TRPRPXG5S"].
    if code_field:
        for code_value in values:
            code = normalize(code_value)
            if code not in codes:
                continue
            for other in values:
                if other == code_value or other in used:
                    continue
                condition = {code_field: exact_any([code]), trip_field: exact_any(_variants(other))}
                n = count(condition)
                if n:
                    matched.append(condition)
                    lines.append(f"'{code_value}' + '{other}' = SCAC {code} + trip number {other}: {n} record(s)")
                    used.update((code_value, other))

    for value in values:
        if value in used:
            continue
        norm = normalize(value)
        readings: list[tuple[dict, str]] = []

        # 1. SCAC + trip number written as one value.
        if code_field:
            for size in sorted({len(c) for c in codes}, reverse=True):
                prefix, rest = norm[:size], norm[size:]
                if rest and prefix in codes:
                    readings.append(({code_field: exact_any([prefix]), trip_field: exact_any([rest])},
                                     f"SCAC {prefix} + trip number {rest}"))
        # 2. Trip number as stored. 3. Other identifiers.
        for id_field in spec.id_fields:
            readings.append(({id_field: exact_any(_variants(value))}, f"{_id_label(spec, id_field)} '{value}'"))

        found = [(condition, label, n) for condition, label in readings if (n := count(condition))]
        # 4. SCAC only, when nothing more specific matched.
        if not found and allow_code_only and code_field and norm in codes:
            condition = {code_field: exact_any([norm])}
            if n := count(condition):
                found.append((condition, f"all eManifests of SCAC {norm}", n))

        if found:
            matched.extend(condition for condition, _, _ in found)
            meanings = "; ".join(f"{label}: {n} record(s)" for _, label, n in found)
            lines.append(f"'{value}' matched {meanings}")
        else:
            lines.append(f"'{value}' matched no record (not as SCAC + trip number, trip number or other ID)")

    log.info("  Identifiers: %s", " | ".join(lines))
    return IdResolution({"$or": matched} if matched else None, lines)
