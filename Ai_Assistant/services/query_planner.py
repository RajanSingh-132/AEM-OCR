"""Query planner: ONE LLM call turns the question (+ conversation) into a plan; code validates it.

The plan says which collection, the standalone question (follow-ups resolved), the intent
(lookup / list / count / semantic), filters, sort, limit and group_by. The LLM only proposes:
every part is checked against the collection's whitelist here, and anything invalid is dropped
(and logged). The exact queries are then built in code (structured_query.py).
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import date, datetime

from Ai_Assistant.collection_mapping import COLLECTIONS, OPS_BY_KIND, CollectionSpec, match_by_keywords
from Ai_Assistant.config import CHAT_MAX_QUESTION_CHARS, LATEST_DEFAULT_LIMIT
from Ai_Assistant.prompts import build_planner_prompt
from Ai_Assistant.services.structured_query import Filter, enum_values
from OCRAI import llm_client
from OCRAI.json_parser import parse_llm_json

log = logging.getLogger(__name__)

INTENTS = ("lookup", "list", "count", "semantic")
_MAX_VALUE_CHARS = 100
_MAX_LIST_VALUES = 20


@dataclass
class Plan:
    collection: CollectionSpec | None
    standalone_question: str
    intent: str = "semantic"
    ids: list[str] = field(default_factory=list)
    filters: list[Filter] = field(default_factory=list)
    sort: tuple[str, str] | None = None     # (sort name, "asc" | "desc")
    limit: int | None = None
    group_by: str | None = None

    def summary(self) -> str:
        if self.collection is None:
            return "no collection"
        parts = [f"collection={self.collection.name}", f"intent={self.intent}"]
        if self.ids:
            parts.append(f"ids={self.ids}")
        if self.filters:
            parts.append("filters=" + "; ".join(f"{f.field} {f.op} {f.value!r}" for f in self.filters))
        if self.sort:
            parts.append(f"sort={self.sort[0]} {self.sort[1]}")
        if self.limit:
            parts.append(f"limit={self.limit}")
        if self.group_by:
            parts.append(f"group_by={self.group_by}")
        return " | ".join(parts)


# ---------- validation ----------

def _clean_text(value) -> str | None:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        return None
    text = " ".join(str(value).split())
    return text if 0 < len(text) <= _MAX_VALUE_CHARS else None


def _iso_date(value) -> str | None:
    text = _clean_text(value)
    if text is None:
        return None
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            pass
    return None


def _validate_filter(raw, spec: CollectionSpec) -> Filter | None:
    if not isinstance(raw, dict):
        return None
    name, op = raw.get("field"), str(raw.get("op", "")).lower()
    qf = spec.query_fields.get(name)
    if qf is None or op not in OPS_BY_KIND[qf.kind]:
        return None
    value = raw.get("value")
    if qf.kind in ("date8", "datetime"):
        if op == "between":
            if not isinstance(value, list) or len(value) != 2:
                return None
            dates = [_iso_date(v) for v in value]
            if None in dates:
                return None
            return Filter(name, op, sorted(dates))
        iso = _iso_date(value)
        return Filter(name, op, iso) if iso else None
    if op == "in":
        values = [_clean_text(v) for v in (value if isinstance(value, list) else [value])]
        values = [v for v in values if v][:_MAX_LIST_VALUES]
        return Filter(name, op, values) if values else None
    text = _clean_text(value)
    return Filter(name, op, text) if text else None


def _validate(raw: dict, question: str, keyword_spec: CollectionSpec | None) -> Plan:
    name = raw.get("collection")
    spec = keyword_spec or (COLLECTIONS.get(str(name).strip().lower()) if name else None)

    standalone = _clean_question(raw.get("standalone_question")) or question
    if spec is None:
        return Plan(collection=None, standalone_question=standalone)

    intent = str(raw.get("intent", "")).lower()
    if intent not in INTENTS:
        intent = "semantic"

    ids = []
    for value in raw.get("ids") or []:
        text = _clean_text(value)
        if text and text not in ids:
            ids.append(text)
    ids = ids[:_MAX_LIST_VALUES]

    filters = []
    for item in raw.get("filters") or []:
        f = _validate_filter(item, spec)
        if f:
            filters.append(f)
        else:
            log.warning("  Planner filter dropped (not allowed / invalid): %r", item)

    sort = None
    raw_sort = raw.get("sort")
    if isinstance(raw_sort, dict) and raw_sort.get("by") in spec.sorts:
        sort = (raw_sort["by"], "asc" if str(raw_sort.get("order")).lower() == "asc" else "desc")
    elif raw_sort:
        log.warning("  Planner sort dropped: %r", raw_sort)

    limit = raw.get("limit")
    limit = limit if isinstance(limit, int) and not isinstance(limit, bool) and limit > 0 else None

    group_by = raw.get("group_by")
    if group_by and not (group_by in spec.query_fields and spec.query_fields[group_by].groupable):
        log.warning("  Planner group_by dropped: %r", group_by)
        group_by = None

    # Make the plan consistent.
    if intent == "lookup" and not ids:
        intent = "list" if filters else "semantic"
    if intent != "count":
        group_by = None
    if intent == "list" and sort and limit is None:
        limit = LATEST_DEFAULT_LIMIT  # "latest / oldest / next ..." without a number

    return Plan(spec, standalone, intent, ids, filters, sort, limit, group_by)


def _clean_question(value) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.split()).strip('"').strip()
    return text if 0 < len(text) <= CHAT_MAX_QUESTION_CHARS else None


# ---------- planning ----------

def _single_collection() -> CollectionSpec | None:
    return next(iter(COLLECTIONS.values())) if len(COLLECTIONS) == 1 else None

def plan_question(question: str, history: str, company_id: int) -> Plan:
    """Plan for the question. Raises llm_client.LLMError only when no fallback is possible."""
    keyword = match_by_keywords(question)
    keyword_spec = keyword[0] if keyword else None
    if keyword:
        log.info("  Keyword '%s' -> collection '%s'", keyword[1], keyword_spec.name)

    values = {name: enum_values(company_id, spec) for name, spec in COLLECTIONS.items()}
    prompt = build_planner_prompt(question, history, COLLECTIONS, values, date.today(),
                                  keyword_spec.name if keyword_spec else None)
    start = time.perf_counter()
    try:
        raw = parse_llm_json(llm_client.generate_json(prompt))
    except llm_client.LLMError:
        spec = keyword_spec or _single_collection()
        if spec is None:
            raise
        log.warning("  Planner unavailable -> similarity search with the question as asked")
        return Plan(spec, question)

    if not isinstance(raw, dict) or "raw_extracted_text" in raw:
        log.warning("  Planner reply was not valid JSON -> similarity search with the question as asked")
        return Plan(keyword_spec or _single_collection(), question)

    plan = _validate(raw, question, keyword_spec)
    log.info("[1/3] Plan in %.2fs: %s", time.perf_counter() - start, plan.summary())
    if plan.standalone_question != question:
        log.info("  Standalone question: %r", plan.standalone_question)
    return plan
