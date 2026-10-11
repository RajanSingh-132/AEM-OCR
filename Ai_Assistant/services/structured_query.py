"""Exact MongoDB queries built from a validated plan: lookup, list, count, count per field.

Everything here is code, not LLM: the company filter is always added here, user values are
regex-escaped (never interpreted as query operators), and the result description handed to
the answer LLM is written here, so totals and conditions are exactly what was executed.
"""

import logging
import re
import time
from dataclasses import dataclass, field

from pymongo.collection import Collection

from Ai_Assistant.collection_mapping import CollectionSpec
from Ai_Assistant.config import (
    ENUM_VALUES_CACHE_SECONDS,
    GROUP_MAX_ROWS,
    LIST_DEFAULT_LIMIT,
    QUERY_HARD_LIMIT,
)
from Ai_Assistant.services.mongo_client import CompanyDataNotFound, get_company_collection

log = logging.getLogger(__name__)


@dataclass
class Filter:
    field: str
    op: str                   # eq | in | contains | gte | lte | between
    value: str | list[str]    # dates are ISO "YYYY-MM-DD"; "in"/"between" take a list


@dataclass
class QueryResult:
    search_type: str          # "exact" | "count" | "groups" | "similarity"
    description: list[str]    # what was executed, as lines written by code
    total: int | None = None  # exact total over all of the company's records (not for similarity)
    records: list[dict] = field(default_factory=list)
    groups: list[tuple[str, int]] = field(default_factory=list)
    related: list[dict] = field(default_factory=list)  # closest records when an exact query found nothing


# ---------- conditions ----------

def _exact(value: str) -> dict:
    """Case-insensitive whole-value match; surrounding spaces in stored values are ignored."""
    return {"$regex": rf"^\s*{re.escape(value.strip())}\s*$", "$options": "i"}


def exact_any(values: list[str]) -> dict:
    return {"$in": [re.compile(rf"^\s*{re.escape(v.strip())}\s*$", re.IGNORECASE) for v in values]}


def _condition(kind: str, f: Filter) -> dict:
    if kind in ("date8", "datetime"):
        start, end = (f.value if f.op == "between" else (f.value, f.value))
        if kind == "date8":  # stored "YYYYMMDD"
            low, high = start.replace("-", ""), end.replace("-", "")
        else:                # stored "YYYY-MM-DD HH:MM:SS"
            low, high = f"{start} 00:00:00", f"{end} 23:59:59"
        if f.op == "gte":
            return {f.field: {"$gte": low}}
        if f.op == "lte":
            return {f.field: {"$lte": high}}
        if kind == "date8" and f.op == "eq":
            return {f.field: low}
        return {f.field: {"$gte": low, "$lte": high}}
    if f.op == "in":
        return {f.field: exact_any(f.value)}
    if f.op == "contains":
        return {f.field: {"$regex": re.escape(f.value.strip()), "$options": "i"}}
    return {f.field: _exact(f.value)}


def build_match(spec: CollectionSpec, company_id: int, filters: list[Filter], ids: list[str]) -> dict:
    conditions = [{spec.company_field: company_id}]
    conditions += [_condition(spec.query_fields[f.field].kind, f) for f in filters]
    if ids:
        conditions.append({"$or": [{field: exact_any(ids)} for field in spec.id_fields]})
    return {"$and": conditions}


_OP_TEXT = {"eq": "is", "in": "is one of", "contains": "contains", "gte": "is on or after",
            "lte": "is on or before", "between": "is between"}


def describe_conditions(filters: list[Filter], ids: list[str]) -> str:
    parts = []
    if ids:
        parts.append(f"ID (trip number, truck number or plate) is one of {', '.join(repr(i) for i in ids)}")
    for f in filters:
        if f.op == "between":
            value = f"{f.value[0]} and {f.value[1]}"
        elif f.op == "in":
            value = ", ".join(repr(v) for v in f.value)
        else:
            value = repr(f.value)
        parts.append(f"{f.field} {_OP_TEXT[f.op]} {value}")
    return "; ".join(parts) if parts else "none (all records)"


# ---------- queries ----------

def _projection(spec: CollectionSpec) -> dict:
    return {"_id": 1, "text": f"${spec.text_field}", **{n: f"${f}" for n, f in spec.summary_fields.items()}}


def _sort_stages(spec: CollectionSpec, sort_name: str, order: str) -> list[dict]:
    direction = -1 if order == "desc" else 1
    added, sort = {}, {}
    for field_name, how in spec.sorts[sort_name].keys:
        if how == "int":
            key = f"_sort_{field_name}"
            added[key] = {"$convert": {"input": f"${field_name}", "to": "int", "onError": 0, "onNull": 0}}
            sort[key] = direction
        else:
            sort[field_name] = direction
    sort["_id"] = direction  # stable order for equal keys
    return ([{"$addFields": added}] if added else []) + [{"$sort": sort}]


def fetch_records(collection: Collection, spec: CollectionSpec, match: dict,
                  sort: tuple[str, str] | None, limit: int | None, label: str) -> QueryResult:
    """Exact query: real total over all matching records + the first `limit` in sort order."""
    start = time.perf_counter()
    sort_name, order = sort or (spec.default_sort, "desc")
    limit = min(limit or LIST_DEFAULT_LIMIT, QUERY_HARD_LIMIT)
    total = collection.count_documents(match)
    pipeline = [{"$match": match}, *_sort_stages(spec, sort_name, order), {"$limit": limit},
                {"$project": _projection(spec)}]
    records = [{"id": d["_id"], "text": d.get("text") or "", **{n: d.get(n) for n in spec.summary_fields}}
               for d in collection.aggregate(pipeline)]
    order_text = "latest first" if order == "desc" else "earliest first"
    log.info("[2/3] Exact %s query: %d match(es), %d returned in %.2fs", label, total, len(records), time.perf_counter() - start)
    return QueryResult(
        search_type="exact",
        description=[
            f"Search: exact database query over ALL {spec.record_label} records of this company",
            f"Sorted by: {spec.sorts[sort_name].description}, {order_text}",
            f"Total matching records: {total}",
            f"Records shown below: {len(records)}" + (f" (the first {len(records)} of {total} in this order)" if total > len(records) else " (all of them)"),
        ],
        total=total,
        records=records,
    )


def count_records(collection: Collection, spec: CollectionSpec, match: dict, group_by: str | None) -> QueryResult:
    start = time.perf_counter()
    total = collection.count_documents(match)
    if not group_by:
        log.info("[2/3] Exact count: %d in %.2fs", total, time.perf_counter() - start)
        return QueryResult(
            search_type="count",
            description=[f"Search: exact count over ALL {spec.record_label} records of this company",
                         f"Total matching records: {total}"],
            total=total,
        )

    pipeline = [{"$match": match}, {"$group": {"_id": f"${group_by}", "n": {"$sum": 1}}}, {"$sort": {"n": -1, "_id": 1}}]
    rows = list(collection.aggregate(pipeline))
    groups = [(str(r["_id"]).strip() if r["_id"] not in (None, "") else "(empty)", r["n"]) for r in rows]
    shown = groups[:GROUP_MAX_ROWS]
    log.info("[2/3] Exact count per %s: %d group(s), total %d in %.2fs", group_by, len(groups), total, time.perf_counter() - start)
    return QueryResult(
        search_type="groups",
        description=[
            f"Search: exact count over ALL {spec.record_label} records of this company, grouped by {group_by}",
            f"Total matching records: {total}",
            f"Distinct {group_by} values: {len(groups)}"
            + (f" (the {len(shown)} largest are shown)" if len(groups) > len(shown) else ""),
        ],
        total=total,
        groups=shown,
    )


# ---------- allowed values of enum fields (read from the data, cached) ----------

_enum_cache: dict[tuple, tuple[float, dict[str, list[str]]]] = {}


def enum_values(company_id: int, spec: CollectionSpec) -> dict[str, list[str]]:
    """{field: [values in the data]} for the spec's enum fields; {} if the company has no data."""
    key = (company_id, spec.name)
    cached = _enum_cache.get(key)
    if cached and time.monotonic() - cached[0] < ENUM_VALUES_CACHE_SECONDS:
        return cached[1]
    try:
        collection = get_company_collection(company_id, spec.name)
    except CompanyDataNotFound:
        return {}
    values = {
        name: sorted({str(v).strip() for v in collection.distinct(name, {spec.company_field: company_id})
                      if v not in (None, "")})
        for name, qf in spec.query_fields.items() if qf.kind == "enum"
    }
    _enum_cache[key] = (time.monotonic(), values)
    return values
