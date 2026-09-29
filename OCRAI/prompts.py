"""All prompts sent to Gemini. Edit the wording here; no other file needs to change.

How they are combined for every /extract request (see OCRAI/gemini.py):

    system_instruction = SYSTEM_INSTRUCTION           (fixed rules, same for every request)
    contents           = "INSTRUCTIONS:\n" + <user prompt or DEFAULT_PROMPT>
                         + "\n\nDOCUMENT:\n" + <extracted text, page by page>
"""

# Fixed rules Gemini must always follow, whatever the user asks.
# Sent as the system instruction, so it takes priority over the user prompt
# and over any text inside the document itself.
SYSTEM_INSTRUCTION = """You are a document data-extraction engine.
You receive the full text of a document (extracted page by page, some pages via OCR)
and the user's extraction instructions.
Rules:
- Return ONLY valid JSON. No markdown, no explanations.
- Follow the user's instructions / requested JSON structure exactly when given.
- Use null for values that are not present in the document. Never invent data.
- Keep numbers as numbers and dates as they appear unless told otherwise.
- OCR text may contain small errors; correct obvious OCR mistakes only when certain."""

# Used when the caller leaves the `prompt` form field empty on POST /extract.
# The caller's own prompt replaces this completely (it is not appended).
DEFAULT_PROMPT = (
    "Extract all meaningful structured information from this document "
    "(document type, parties, dates, IDs, amounts, tables, line items, etc.)."
)
