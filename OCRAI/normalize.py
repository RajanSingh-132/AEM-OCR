"""Post-processing of the LLM's JSON (PRD 5.7): shipment keys, commodity cleanup, multi-line split.

Key spellings (including the misspellings) are part of the frontend contract.
"""

import re

SHIPMENT_KEYS = (
    "commodity", "pickup_location", "pickup_date", "pickup_time", "pickup_refrence_no", "distance",
    "delivery_location", "delivery_date", "delivery_time", "delivery_refrence_no", "ValueOfgoods",
    "Equipment", "No.OfPackage", "weight", "temperature", "dimention", "pickupNote", "DeliveryNotes",
    "Copmliancehandling",
)
PER_LINE_KEYS = ("commodity", "No.OfPackage", "weight", "dimention")

_COUNT = re.compile(r"\b\d+(\.\d+)?\s*(pcs?|pieces?|pkgs?|packages?|pallets?|skids?|qty)\b")
_WEIGHT = re.compile(r"\b\d+([.,]\d+)?\s*(lbs?|lb|kg|kgs|pounds?)\b")
_UNIT_WORDS = re.compile(r"\b(lbs?|kgs?|pounds?|pcs|pieces|and)\b|,")
_PC_WORD = re.compile(r"\bpcs?\b")
_WEIGHT_WORD = re.compile(r"\b(lbs?|kg)\b")


def is_invalid_commodity(value) -> bool:
    """True when the "commodity" is really a package count or a weight."""
    if isinstance(value, list):
        return any(is_invalid_commodity(item) for item in value)
    if value is None:
        return False
    text = str(value).lower()
    if _COUNT.search(text):
        return True
    if _WEIGHT.search(text) and not re.search(r"[a-z]{3,}", _UNIT_WORDS.sub(" ", text)):
        return True
    return bool(_PC_WORD.search(text) and _WEIGHT_WORD.search(text))


def _is_empty(value) -> bool:
    return value is None or value == "" or value == []


def _commodity_values(shipment: dict) -> list | None:
    """The commodity list when it has 2+ non-empty values (i.e. the shipment must be split)."""
    commodity = shipment.get("commodity")
    if isinstance(commodity, list) and sum(not _is_empty(c) for c in commodity) >= 2:
        return commodity
    return None


def sanitize_shipment(shipment: dict) -> dict:
    row = dict(shipment)
    for key in SHIPMENT_KEYS:
        row.setdefault(key, None)
    commodity = row["commodity"]
    if isinstance(commodity, list):
        # 0 or 1 real values: unwrap (2+ values are split before this is called).
        values = [c for c in commodity if not _is_empty(c)]
        if len(values) <= 1:
            commodity = values[0] if values else None
    if is_invalid_commodity(commodity):
        commodity = None
    row["commodity"] = commodity
    return row


def split_by_commodity(shipment: dict, commodities: list) -> list[dict]:
    """One row per commodity; per-line list values are zipped by position, scalars copied."""
    rows = []
    for i, commodity in enumerate(commodities):
        if _is_empty(commodity):
            continue
        row = dict(shipment)
        for key in PER_LINE_KEYS:
            value = shipment.get(key)
            if isinstance(value, list):
                value = value[i] if i < len(value) else None
                value = None if _is_empty(value) else value
            row[key] = value
        row["commodity"] = commodity
        rows.append(sanitize_shipment(row))
    return rows


def normalize_shipment(shipment):
    if isinstance(shipment, dict):
        commodities = _commodity_values(shipment)
        return split_by_commodity(shipment, commodities) if commodities else sanitize_shipment(shipment)

    if isinstance(shipment, list):
        items = [item for item in shipment if isinstance(item, dict)]
        if len(shipment) == 1 and items and (commodities := _commodity_values(items[0])):
            return split_by_commodity(items[0], commodities)
        rows = [sanitize_shipment(item) for item in items]
        if len(rows) == 1:
            return rows[0]
        return rows or sanitize_shipment({})

    if shipment is None:
        return sanitize_shipment({})
    return shipment


def normalize_extract(data: dict) -> dict:
    if "shipment" in data:
        data["shipment"] = normalize_shipment(data["shipment"])
    if "commodity" in data and is_invalid_commodity(data["commodity"]):
        data["commodity"] = None
    return data
