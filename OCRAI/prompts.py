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

EXTRACTION_PROMPT = """You convert a business document into JSON that captures ALL of its data.

## How the input is built
The DOCUMENT TEXT below was extracted automatically, page by page. Besides the printed text,
it can contain sections added by the extractor. They are part of the document:
- "=== FILLED-IN VALUES (label: value, same row) ===": values typed onto the page (e.g. a
  filled-in form). Each line is "<label printed on the same row> <typed value>". If a line has
  no label, use the surrounding printed text and the value itself to decide what it is.
- "=== PDF FORM FIELD VALUES ===": fillable form fields as "<internal field name>: <value>".
  Field names are technical; match them to the printed labels by meaning.
- "=== EMBEDDED IMAGE OCR ===" and "=== PAGE RENDER OCR ===": text read from images or from
  the rendered page. It may repeat printed text and may contain small OCR errors.

## The most important rule: labels and values
Forms often print a label with an EMPTY space next to it ("Driver Name:" followed by nothing)
and store the value elsewhere. Before you set any field to null, look for its value in ALL
sections above. Use null ONLY when the value truly appears nowhere in the input.

## Output
Return ONLY one JSON object, no markdown, no explanations. Build the structure from the
document itself:
1. "document_type": what the document is (e.g. "Bill of Lading", "Rate Confirmation",
   "ACE eManifest"), as named in the document or inferred from its content.
2. Capture EVERY piece of data: reference and ID numbers, dates, times, parties, addresses,
   contacts, phone and fax numbers, emails, vehicles and plates, line items, quantities,
   weights, dimensions, charges, totals, currencies, instructions, notes, terms, signatures.
3. Keys: named after the document's own labels, in snake_case ("BOL #" -> "bol_number",
   "Truck License Plate" -> "truck_license_plate").
4. Nesting: group related fields the way the document groups them ("shipper", "consignee",
   "carrier", "driver", "pickup", "delivery", "charges"). Repeating sections (stops, items,
   charges) become arrays.
5. Tables: an array of objects, one object per row, keys from the column headers.
   Include ONLY rows that contain at least one value; skip blank rows. A table with no filled
   rows becomes an empty array [].
6. Letterheads, logos and footers of the company that printed or supplies the blank form
   (its name, phone, fax, website) go into a separate "form_provider" object. Never use them
   as the value of a field in the document body. For example, a "Company Name:" field gets
   the value written for it, not the name printed in the letterhead.
7. Values: copy exactly as written, keeping dates, units, currency symbols and leading zeros.
   Use JSON numbers only for plain amounts or counts without units. Each fact appears once:
   do not duplicate the same value under several keys, and merge a title that the text
   repeats ("MANIFESTMANIFEST") into one.
8. Never invent data or add anything that is not in the input. Correct obvious OCR mistakes
   only when certain.
9. Long free text (terms and conditions, legal notices) goes in full under a descriptive key.

DOCUMENT TEXT:
{text}
"""
