"""All prompts sent to the LLM. Edit the wording here; no other file needs to change.

Never name an LLM vendor in these prompts.

The extraction prompt is assembled with plain string concatenation / str.replace, not
str.format, so the literal braces in the JSON templates do NOT need to be doubled.
"""

import json

from OCRAI.schemas import TEMPLATES

# Vision OCR (PRD 5.1): sent together with one image.
OCR_INSTRUCTION = (
    "Extract readable text from this image exactly as present. "
    "If the image has no readable text, reply with: NO_TEXT_FOUND"
)

# Marker the OCR model replies with when an image has no text.
NO_TEXT_MARKER = "NO_TEXT_FOUND"

# ---------------------------------------------------------------------------
# Structured extraction (ACE / ACI)
# ---------------------------------------------------------------------------

_INTRO = """You fill a {module} eManifest form from a transportation document (bill of lading,
manifest, rate confirmation, dispatch sheet, invoice, ...). Read the document and put every
value it contains into the matching field of the JSON template below.

## How the input is built
The DOCUMENT TEXT was extracted automatically, page by page. Besides the printed text it can
contain sections added by the extractor; they are part of the document:
- "=== FILLED-IN VALUES (label: value, same row) ===": values typed onto the page (a filled-in
  form). Each line is "<label printed on the same row> <typed value>".
- "=== PDF FORM FIELD VALUES ===": fillable form fields as "<internal field name>: <value>".
  Match the technical field names to the template fields by meaning.
- "=== EMBEDDED IMAGE OCR ===" and "=== PAGE RENDER OCR ===": text read from images or from
  the rendered page. It may repeat printed text and may contain small OCR errors.
Forms often print a label with an empty space next to it and store the value in one of these
sections. Look in ALL sections before leaving a field blank.
"""

_RULES = """
## Rules
1. Return ONLY one JSON object with EXACTLY the structure of the template: the same keys, the
   same nesting, the same order. Do not add, rename or remove keys.
2. Every value is a string. If a value is not in the document, use "" (empty string).
   Never use null, never guess, never invent values.
3. Lists ("trucks", "drivers", "trailers", "commodities", ...): one object per real item in the
   document (several trucks, drivers, trailers, commodities -> several objects). If the
   document has none, return the list with one object whose values are all "".
4. Dates: "YYYY-MM-DD". Times: 24-hour "HH:MM". Only convert when the date/time is
   unambiguous; otherwise copy it as written.
5. Countries: 2-letter codes ("US", "CA", "MX"). Provinces/states: 2-letter codes ("ON", "QC",
   "NY", "CA"). Only when certain; otherwise copy as written.
6. Quantity and weight: the number only ("24", "18500"); the unit goes in "quantityUnit"
   ("PCS", "PLT", "SKD", "CTN", ...) / "weightUnit" ("LB" or "KG"). Remove thousands separators.
7. People: split full names into "firstName" and "lastName". "driverRole" is "Driver" unless
   the document says co-driver ("Co-Driver").
8. Letterheads, logos and footers of the company that supplies the blank form (software
   vendor, form printer) are NOT data: never put them in any field.
9. Correct obvious OCR mistakes only when certain. Copy IDs, plates and numbers exactly
   (keep letters, digits and leading zeros).
"""

_FIELD_GUIDES = {
    "ACE": """
## Field guide (ACE = U.S. CBP truck eManifest, entering the United States)
- form.scac: the carrier's 4-letter SCAC code.
- form.tripNumber: the trip / manifest number (often starts with the SCAC).
- form.portName / form.portCode: U.S. port of entry / border crossing name (e.g.
  "Champlain, NY") and its 4-digit port code. Fill only what the document states; never
  look up or guess a code.
- form.estArrivalDate / estArrivalTime: estimated arrival (ETA) at the U.S. border.
- trucks: the power unit / tractor: unit number, VIN, type, license plate and its country and
  state/province.
- drivers: driver and co-driver: license number, citizenship, address.
- shipments: one per shipment / bill of lading. shipmentControlNumber is the SCN (PAPS number,
  usually the SCAC followed by digits; also "PRO" or "BOL" number when used as the SCN).
  shipper* / consignee*: name, street address, city and country of each party.
  commodities: one per line item: description, quantity + unit, weight + unit, country of
  origin, marks and numbers.
- trailers: trailer number, type (e.g. "Dry Van", "Reefer", "Flatbed"), the SCN of the shipment
  loaded on it (shipmentControl), its commodity, quantity, weight and license plates.""",
    "ACI": """
## Field guide (ACI = CBSA truck eManifest, entering Canada)
- form.carrierCode: the carrier's 4-character CBSA carrier code.
- form.tripNumber: the trip / conveyance reference number (often starts with the carrier code).
- form.portName / form.portCode: Canadian port of entry / border crossing name (e.g.
  "Lacolle, QC") and its 4-digit port code. Fill only what the document states; never
  look up or guess a code.
- form.estArrivalDate / estArrivalTime: estimated arrival (ETA) at the Canadian border.
- trucks: the power unit / tractor: unit number, VIN, type, license plate and its country and
  province/state.
- drivers: driver and co-driver: license number, citizenship, address.
- cargos: one per shipment / bill of lading. parsNumber is the PARS / cargo control number
  (CCN, usually the carrier code followed by digits). carrierCode: carrier code of that cargo.
  attachedToTrip: the trip number the cargo travels on, if stated.
  shipper / consignee: name, street address, city, province, country, postal code, contact
  person and phone of each party.
  commodities: one per line item: description, quantity + unit, weight + unit, country of
  origin, marks and numbers.
- trailers: trailer number, type (e.g. "Dry Van", "Reefer", "Flatbed"), the PARS / cargo
  control number of the cargo loaded on it (cargoControl), its commodity, quantity, weight and
  license plates.""",
}


def build_extraction_prompt(module: str, text: str) -> str:
    """The full extraction prompt for one module, with the document text filled in."""
    template = json.dumps(TEMPLATES[module], indent=2)
    return (
        _INTRO.replace("{module}", module)
        + _FIELD_GUIDES[module]
        + "\n"
        + _RULES
        + "\n## JSON template to fill\n"
        + template
        + "\n\nDOCUMENT TEXT:\n"
        + text
        + "\n"
    )
