"""Fixed response templates per module (ACE / ACI) and conforming the LLM reply to them.

The response always has exactly the template's keys, in the template's order:
- a value found in the document -> that value, as a string
- a value not in the document   -> "" (or the template default, e.g. driverRole "Driver")
- extra keys from the LLM       -> dropped
- a list with no real items     -> one blank item, so the frontend form still renders
"""

import copy

ACE_TEMPLATE = {
    "module": "ACE",
    "form": {
        "scac": "",
        "tripNumber": "",
        "portName": "",
        "portCode": "",
        "estArrivalDate": "",
        "estArrivalTime": "",
    },
    "trucks": [{
        "truckNumber": "",
        "vinNumber": "",
        "vehicleType": "",
        "licensePlateNumber": "",
        "country": "",
        "provinceState": "",
    }],
    "drivers": [{
        "firstName": "",
        "lastName": "",
        "driverRole": "Driver",
        "driverLicense": "",
        "citizenship": "",
        "addressLine1": "",
        "city": "",
        "provinceState": "",
        "country": "",
    }],
    "shipments": [{
        "scac": "",
        "shipmentControlNumber": "",
        "shipperName": "",
        "shipperAddress": "",
        "shipperCity": "",
        "shipperCountry": "",
        "consigneeName": "",
        "consigneeAddress": "",
        "consigneeCity": "",
        "consigneeCountry": "",
        "commodities": [{
            "description": "",
            "quantity": "",
            "quantityUnit": "",
            "weight": "",
            "weightUnit": "",
            "countryOfOrigin": "",
            "marksAndNumbers": "",
        }],
    }],
    "trailers": [{
        "trailerNumber": "",
        "trailerType": "",
        "shipmentControl": "",
        "commodity": "",
        "quantity": "",
        "weight": "",
        "licensePlates": [{
            "plateNumber": "",
            "country": "",
            "provinceState": "",
        }],
    }],
}

_PARTY = {
    "name": "",
    "address": "",
    "city": "",
    "province": "",
    "country": "",
    "postal": "",
    "contact": "",
    "phone": "",
}

ACI_TEMPLATE = {
    "module": "ACI",
    "form": {
        "carrierCode": "",
        "tripNumber": "",
        "portName": "",
        "portCode": "",
        "estArrivalDate": "",
        "estArrivalTime": "",
    },
    "trucks": copy.deepcopy(ACE_TEMPLATE["trucks"]),
    "drivers": copy.deepcopy(ACE_TEMPLATE["drivers"]),
    "cargos": [{
        "carrierCode": "",
        "parsNumber": "",
        "attachedToTrip": "",
        "shipper": dict(_PARTY),
        "consignee": dict(_PARTY),
        "commodities": copy.deepcopy(ACE_TEMPLATE["shipments"][0]["commodities"]),
    }],
    "trailers": [{
        "trailerNumber": "",
        "trailerType": "",
        "cargoControl": "",
        "commodity": "",
        "quantity": "",
        "weight": "",
        "licensePlates": copy.deepcopy(ACE_TEMPLATE["trailers"][0]["licensePlates"]),
    }],
}

TEMPLATES = {"ACE": ACE_TEMPLATE, "ACI": ACI_TEMPLATE}
MODULES = tuple(TEMPLATES)


def _to_text(value, default: str) -> str:
    if value is None or isinstance(value, (dict, list)):
        return default
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    text = str(value).strip()
    return text if text and text.lower() not in ("null", "none", "n/a") else default


def _conform(value, template):
    if isinstance(template, dict):
        source = value if isinstance(value, dict) else {}
        return {key: _conform(source.get(key), sub) for key, sub in template.items()}
    if isinstance(template, list):
        item_template = template[0]
        blank = _conform(None, item_template)
        items = value if isinstance(value, list) else [value] if isinstance(value, dict) else []
        rows = [_conform(item, item_template) for item in items if isinstance(item, dict)]
        rows = [row for row in rows if row != blank]  # drop rows with no real values
        return rows or [blank]
    return _to_text(value, template)


def conform_to_template(data, module: str) -> dict:
    """The LLM's reply reshaped to the module template (see module docstring)."""
    result = _conform(data, TEMPLATES[module])
    result["module"] = module
    return result


def blank_template(module: str) -> dict:
    return conform_to_template(None, module)
