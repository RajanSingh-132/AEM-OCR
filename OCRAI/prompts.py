"""All prompts sent to the LLM. Edit the wording here; no other file needs to change.

Never name an LLM vendor in these prompts.

EXTRACTION_PROMPT is filled with str.replace("{text}", ...), not str.format, so the
literal braces in its JSON examples do NOT need to be doubled.
"""

# Vision OCR (PRD 5.1): sent together with one image.
OCR_INSTRUCTION = (
    "Extract readable text from this image exactly as present. "
    "If the image has no readable text, reply with: NO_TEXT_FOUND"
)

# Marker the OCR model replies with when an image has no text.
NO_TEXT_MARKER = "NO_TEXT_FOUND"

# ---------------------------------------------------------------------------
# TEMPORARY PLACEHOLDER - replace the whole string below with PRD Appendix A,
# verbatim. Keep the single {text} placeholder where the document text goes.
# ---------------------------------------------------------------------------
EXTRACTION_PROMPT = """You extract order data from a logistics document.
Return ONLY one JSON object with exactly these top-level keys: "customerinfo", "shipment", "Revenue".
Use null for any value that is not in the document. Never invent values.

"shipment" is one object for a single line item, or an array of objects for 2+ line items.
Each shipment object has exactly these keys (every value a string or null):
commodity, pickup_location, pickup_date, pickup_time, pickup_refrence_no, distance,
delivery_location, delivery_date, delivery_time, delivery_refrence_no, ValueOfgoods,
Equipment, No.OfPackage, weight, temperature, dimention, pickupNote, DeliveryNotes,
Copmliancehandling
"commodity" is the description of the goods, never a package count or a weight.

"customerinfo" holds the customer's details and "Revenue" the charges, including
"fluecurrencyTypes": a list of fuel lines with keys fuelratemethod, fuel_rate_method_value,
fuel_total_value.

DOCUMENT TEXT:
{text}
"""
