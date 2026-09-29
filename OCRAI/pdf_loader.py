"""PDF loading (PRD 5.2): text layer, embedded images, JPEG carving, page render, form fields."""

import io
import logging
import time

from pypdf import PdfReader

from OCRAI.ocr import ocr_image

try:
    import pypdfium2 as pdfium
except ImportError:  # optional native dependency: Tier 3 is disabled without it
    pdfium = None

log = logging.getLogger(__name__)

# Fillable PDFs emit many harmless warnings.
logging.getLogger("pypdf").setLevel(logging.ERROR)

MIN_REAL_WORDS = 30          # text layer this rich is kept alongside OCR / skips page render
MIN_RENDER_KEEP_WORDS = 10   # text layer this rich is kept alongside page-render OCR
MIN_CARVED_JPEG_BYTES = 10_000  # smaller JPEG chunks are icons, not photos
RENDER_SCALE = 2.08          # ~150 DPI
CHECKBOX_VALUES = {"", "Off", "Yes", "No"}  # compared without the leading "/"
CHECKBOX_SUFFIXES = ("check", "chk", "box", "flag")


def real_words(text: str) -> list[str]:
    return [word for word in text.split() if len(word) > 1]


def carve_jpegs(data: bytes) -> list[bytes]:
    """Find JPEGs directly in the raw PDF bytes (catches inline images that page.images misses)."""
    chunks = []
    pos = 0
    while (start := data.find(b"\xff\xd8\xff", pos)) != -1:
        end = data.find(b"\xff\xd9", start + 3)
        if end == -1:
            break
        chunk = data[start:end + 2]
        if len(chunk) >= MIN_CARVED_JPEG_BYTES:
            chunks.append(chunk)
        pos = end + 2
    return chunks


def form_field_lines(reader: PdfReader) -> list[str]:
    try:
        fields = reader.get_fields() or {}
    except Exception:
        log.exception("  Could not read PDF form fields")
        return []
    lines = []
    for name, field in fields.items():
        value = field.get("/V")
        if value is None:
            continue
        if hasattr(value, "get_object"):
            value = value.get_object()
        value = str(value).strip()
        if not value:
            continue
        lowered = name.lower()
        if lowered.endswith(CHECKBOX_SUFFIXES) and value.lstrip("/") in CHECKBOX_VALUES:
            continue
        if lowered.endswith("label") and value.endswith(":"):
            continue
        lines.append(f"{name}: {value}")
    return lines


class _PdfOcr:
    """Per-document state: JPEG carving and the page renderer run at most once."""

    def __init__(self, data: bytes, filename: str):
        self.data = data
        self.filename = filename
        self._carved_text: str | None = None
        self._carved_used = False
        self._renderer = None

    def carved_text(self) -> str:
        """OCR of carved JPEGs, computed once per document and handed out only once.

        The carved photos belong to the whole file, not to one page, so later pages
        get "" instead of a duplicate (they fall through to the page render).
        """
        if self._carved_used:
            return ""
        if self._carved_text is None:
            chunks = carve_jpegs(self.data)
            log.info("    Tier 2.5: carved %d JPEG(s) >= %d bytes from the raw file", len(chunks), MIN_CARVED_JPEG_BYTES)
            texts = [ocr_image(chunk, f"{self.filename}_carved{idx}.jpg") for idx, chunk in enumerate(chunks)]
            self._carved_text = "\n".join(text for text in texts if text)
        if self._carved_text:
            self._carved_used = True
        return self._carved_text

    def render_page(self, index: int) -> bytes:
        if self._renderer is None:
            self._renderer = pdfium.PdfDocument(self.data)
        page = self._renderer[index]
        try:
            image = page.render(scale=RENDER_SCALE).to_pil()
        finally:
            page.close()
        out = io.BytesIO()
        image.save(out, format="PNG")
        return out.getvalue()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()


def _embedded_image_text(page, index: int, filename: str) -> str:
    texts = []
    try:
        images = list(page.images)
    except Exception:
        log.exception("    Could not list images on page %d", index + 1)
        return ""
    for idx, image in enumerate(images):
        try:
            data = image.data
        except Exception:
            log.exception("    Could not decode image %d on page %d", idx, index + 1)
            continue
        text = ocr_image(data, image.name or f"{filename}_p{index}_img{idx}.png")
        if text:
            texts.append(text)
    return "\n".join(texts)


def load_pdf(data: bytes, filename: str) -> list[str]:
    reader = PdfReader(io.BytesIO(data))
    if reader.is_encrypted:
        reader.decrypt("")

    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            log.exception("  Could not read the text layer of a page")
            pages.append("")
    log.info("  PDF: %d page(s), Tier 1 text layer %d chars total", len(pages), sum(len(p) for p in pages))

    state = _PdfOcr(data, filename)
    try:
        for i, page in enumerate(reader.pages):
            start = time.perf_counter()
            text_layer = pages[i].strip()
            word_count = len(real_words(text_layer))

            ocr_text = _embedded_image_text(page, i, filename)
            tier = "2 (embedded images)"
            if not ocr_text:
                ocr_text = state.carved_text()
                tier = "2.5 (JPEG carving)"

            if ocr_text:
                if word_count >= MIN_REAL_WORDS:
                    pages[i] = text_layer + "\n\n=== EMBEDDED IMAGE OCR ===\n" + ocr_text
                    result = "text layer + OCR"
                elif len(ocr_text) >= len(text_layer):
                    pages[i] = ocr_text
                    result = "OCR only"
                else:
                    pages[i] = text_layer
                    result = "text layer only"
                log.info("  Page %d: %d words in text layer, Tier %s -> %s, %d chars, %.2fs",
                         i + 1, word_count, tier, result, len(pages[i]), time.perf_counter() - start)
                continue

            if pdfium is None or word_count >= MIN_REAL_WORDS:
                reason = "rich text layer" if word_count >= MIN_REAL_WORDS else "pypdfium2 not installed"
                log.info("  Page %d: %d words in text layer, no image text, Tier 3 skipped (%s), %.2fs",
                         i + 1, word_count, reason, time.perf_counter() - start)
                continue

            render_text = ocr_image(state.render_page(i), f"{filename}_p{i}_render.png")
            if render_text and real_words(render_text):
                if word_count >= MIN_RENDER_KEEP_WORDS:
                    pages[i] = text_layer + "\n\n=== PAGE RENDER OCR ===\n" + render_text
                elif len(render_text) > len(text_layer):
                    pages[i] = render_text
            log.info("  Page %d: %d words in text layer, Tier 3 (page render) -> %d chars, %.2fs",
                     i + 1, word_count, len(pages[i]), time.perf_counter() - start)
    finally:
        state.close()

    lines = form_field_lines(reader)
    if lines:
        block = "=== PDF FORM FIELD VALUES ===\n" + "\n".join(lines)
        if pages:
            pages[0] = (pages[0] + "\n\n" + block) if pages[0] else block
        else:
            pages = [block]
        log.info("  PDF form fields: %d value(s) added to page 1", len(lines))
    return pages
