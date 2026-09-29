"""Robust parsing of the LLM's reply into JSON (PRD 5.6). The structure is kept as-is."""

import json
import re


def _try_load(text: str):
    try:
        value = json.loads(text)
        return json.loads(value) if isinstance(value, str) else value
    except (TypeError, ValueError):
        return None


def parse_llm_json(text: str) -> dict | list:
    original = text or ""
    cleaned = re.sub(r"^```(?:json)?\s*", "", original.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    # The whole reply may be JSON wrapped in a quoted string.
    if len(cleaned) >= 2 and cleaned.startswith('"') and cleaned.endswith('"'):
        unwrapped = None
        try:
            unwrapped = json.loads(cleaned)
        except ValueError:
            pass
        if isinstance(unwrapped, str):
            cleaned = unwrapped.strip()

    candidates = [cleaned]
    first, last = cleaned.find("{"), cleaned.rfind("}")
    if first != -1 and last > first:
        candidates.append(cleaned[first:last + 1])
    for candidate in candidates:
        value = _try_load(candidate)
        if isinstance(value, dict):
            return value

    # A list of objects is kept; a bare scalar array (e.g. ["0.000","0.000"]) never is.
    first, last = cleaned.find("["), cleaned.rfind("]")
    if first != -1 and last > first:
        value = _try_load(cleaned[first:last + 1])
        if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
            return value

    objects = []
    for line in cleaned.splitlines():
        line = line.strip()
        if line.startswith("{"):
            value = _try_load(line.rstrip(","))
            if isinstance(value, dict):
                objects.append(value)
    if len(objects) == 1:
        return objects[0]
    if objects:
        return objects

    return {"raw_extracted_text": original}
