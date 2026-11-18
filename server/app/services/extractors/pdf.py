"""PDF extraction with PyMuPDF: page-wise paragraphs and (where detectable) tables."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import pymupdf as fitz  # PyMuPDF

from pydantic import BaseModel

from app.core.exceptions import ExtractionError, ScannedDocumentError
from app.llm.llama_parse import OcrResult
from app.schemas.extraction import ExtractedUnit, ExtractionReport, Locator
from app.services.extractors import lap

MIN_TEXT_CHARS = 20  # a page with fewer characters than this has no usable text layer
MIN_IMAGE_COVERAGE = 0.5  # ...and counts as a picture-only page when images cover at least this share of it
SCANNED_PAGE_RATIO = 0.5  # a PDF is treated as scanned when at least this share of its pages is picture-only
# Table detection costs time in proportion to a page's vector drawings: a map or illustration with 10,000+
# paths can take 2 to 10 s, and on the real data room such pages never held a table worth the cost (one
# did). Above this count table detection is skipped; the page's text is still extracted as usual.
MAX_DRAWINGS_FOR_TABLES = 10_000


class PdfTextCheck(BaseModel):
    pages: int
    pages_without_text: list[int]  # 1-based numbers of picture-only pages (no text layer, mostly image)

    def is_scanned(self) -> bool:
        return self.pages > 0 and len(self.pages_without_text) / self.pages >= SCANNED_PAGE_RATIO


def _is_picture_only(page: "fitz.Page") -> bool:
    """Almost no text and mostly covered by images: a scanned page. A page with little text and
    no big image (a divider slide, a blank page) is not a scan."""
    if len(page.get_text().strip()) >= MIN_TEXT_CHARS:
        return False
    area = page.rect.width * page.rect.height
    if area <= 0:
        return False
    covered = 0.0
    for info in page.get_image_info():
        overlap = fitz.Rect(info["bbox"]) & page.rect
        covered += max(overlap.width, 0) * max(overlap.height, 0)
    return min(covered / area, 1.0) >= MIN_IMAGE_COVERAGE


def _text_check(doc: "fitz.Document") -> PdfTextCheck:
    return PdfTextCheck(pages=doc.page_count, pages_without_text=[p.number + 1 for p in doc if _is_picture_only(p)])


def check_pdf_text(path: Path) -> PdfTextCheck | None:
    """Which pages have no text layer. None if the file cannot be opened."""
    try:
        with fitz.open(path) as doc:
            return _text_check(doc)
    except Exception:
        return None


def count_pages(path: Path) -> int | None:
    try:
        with fitz.open(path) as doc:
            return doc.page_count
    except Exception:
        return None


def _table_rows(page: "fitz.Page", table) -> list[list[str | None]]:
    """Cell text read with the page's own text extraction clipped to each cell. PyMuPDF's ``table.extract()`` can join the
    words of a cell without spaces and move a decimal point into the next line ("92.4%" -> "924%" and "."), which would
    corrupt every number in the table. Falls back to ``extract()`` if the cells cannot be read."""
    try:
        rows = [[" ".join(page.get_text("text", clip=fitz.Rect(cell)).split()) if cell else "" for cell in row.cells]
                for row in table.rows]
        if any(any(c for c in row) for row in rows):
            return rows
    except Exception:  # best effort: keep the extractor's own result
        pass
    return table.extract()


HEADING_SIZE_RATIO = 1.15  # a short single-line block whose text is at least this much larger than the page's body text is a heading
MAX_HEADING_CHARS = 120


def _heading_block_numbers(page: "fitz.Page") -> set[int]:
    """Numbers of the text blocks of a page that are headings: one short line without a closing full stop, in a clearly larger size than the body text of the page
    (or bold and at least body size). A PDF without an outline has no other way to say where a section starts."""
    try:
        blocks = [b for b in page.get_text("dict")["blocks"] if b.get("type") == 0]
    except Exception:
        return set()
    sizes: dict[float, int] = {}
    for block in blocks:
        for line in block["lines"]:
            for span in line["spans"]:
                sizes[round(span["size"], 1)] = sizes.get(round(span["size"], 1), 0) + len(span["text"])
    if not sizes:
        return set()
    body = max(sizes, key=lambda k: sizes[k])
    found: set[int] = set()
    for block in blocks:
        lines = [ln for ln in block["lines"] if "".join(sp["text"] for sp in ln["spans"]).strip()]
        if len(lines) != 1:
            continue
        spans = [sp for sp in lines[0]["spans"] if sp["text"].strip()]
        text = " ".join(sp["text"].strip() for sp in spans)
        if not spans or len(text) > MAX_HEADING_CHARS or text.endswith("."):
            continue
        size = max(sp["size"] for sp in spans)
        bold = all(sp["flags"] & 16 for sp in spans)
        if size >= body * HEADING_SIZE_RATIO or (bold and size >= body):
            found.add(block["number"])
    return found


OcrFn = Callable[[Path, list[int]], OcrResult]  # (file, 1-based picture-only pages) -> the text of each; raises ExtractionError when it cannot


def _ocr_units(text: str, page_no: int, document_id: str, filename: str, heading: str | None) -> list[ExtractedUnit]:
    """The text OCR returned for one page, as units: blank-line separated blocks, a block of table lines (starting with "|") as a table."""
    units = []
    for block in (b.strip() for b in text.split("\n\n")):
        if not block:
            continue
        is_table = all(line.lstrip().startswith("|") for line in block.splitlines())
        units.append(ExtractedUnit(
            document_id=document_id, filename=filename, source_type="pdf", locator=Locator(page=page_no, heading=heading),
            content=block if is_table else " ".join(block.split()), content_type="table" if is_table else "paragraph", metadata={"ocr": True}))
    return units


def _rows_to_text(rows: list[list[str | None]]) -> str:
    return "\n".join(" | ".join((c or "").strip() for c in row) for row in rows)


def extract_pdf(path: Path, document_id: str, filename: str, report: ExtractionReport | None = None,
                ocr: OcrFn | None = None) -> list[ExtractedUnit]:
    """``report`` (optional) receives milliseconds per part (open, scan_check, tables, text, ocr) and notes about
    picture-only pages that were not captured and artwork pages where table detection was skipped. With ``ocr`` the picture-only pages are read by it
    and their text becomes units on the page it came from (the pages that have a text layer are still read here); without it a mostly scanned PDF is refused."""
    t = time.perf_counter()
    try:
        doc = fitz.open(path)
    except Exception as exc:
        raise ExtractionError(f"Cannot open PDF '{filename}': {exc}") from exc
    lap(report, "open", t)
    units: list[ExtractedUnit] = []
    artwork_pages: list[int] = []
    with doc:
        if doc.needs_pass:
            raise ExtractionError(f"PDF '{filename}' is encrypted")
        t = time.perf_counter()
        check = _text_check(doc)
        lap(report, "scan_check", t)
        ocr_text: dict[int, str] = {}
        if check.pages_without_text and ocr is not None:
            t = time.perf_counter()
            result = ocr(path, check.pages_without_text)
            lap(report, "ocr", t)
            ocr_text = result.pages
            empty = [p for p in check.pages_without_text if not ocr_text.get(p)]
            if report is not None:
                report.notes.append(f"OCR read {len(check.pages_without_text) - len(empty)} of {len(check.pages_without_text)} picture-only pages "
                                    f"({result.credits:g} credits)" + (f"; no text came back for pages {empty}" if empty else ""))
        elif check.is_scanned():
            raise ScannedDocumentError(
                f"'{filename}' is picture-only (no text layer) on {len(check.pages_without_text)} of {check.pages} pages, "
                "so it looks like a scanned document. It is left out because there is no OCR.")
        elif check.pages_without_text and report is not None:
            report.notes.append(f"Picture-only pages with no text layer were not captured (no OCR): {check.pages_without_text}")
        toc = [(title.strip(), page) for _level, title, page in doc.get_toc() if title and title.strip() and page and page > 0]
        heading: str | None = None  # the nearest heading above: from the PDF outline when it has one, otherwise found by font size
        for page in doc:
            page_no = page.number + 1
            if toc:
                heading = next((t for t, p in reversed(toc) if p <= page_no), heading)
            if page_no in ocr_text:
                units.extend(_ocr_units(ocr_text[page_no], page_no, document_id, filename, heading))
                continue
            heading_blocks = set() if toc else _heading_block_numbers(page)
            table_bboxes: list[fitz.Rect] = []
            t = time.perf_counter()
            tables = []
            if len(page.get_cdrawings()) > MAX_DRAWINGS_FOR_TABLES:
                artwork_pages.append(page_no)
            else:
                try:
                    tables = page.find_tables().tables
                except Exception:  # table detection is best-effort
                    tables = []
            lap(report, "tables", t)
            for t_idx, table in enumerate(tables, start=1):
                rows = _table_rows(page, table)
                text = _rows_to_text(rows)
                if not text.strip():
                    continue
                table_bboxes.append(fitz.Rect(table.bbox))
                units.append(ExtractedUnit(
                    document_id=document_id, filename=filename, source_type="pdf",
                    locator=Locator(page=page_no, table_index=t_idx, heading=heading),
                    content=text, content_type="table", metadata={"rows": rows},
                ))
            t = time.perf_counter()
            blocks = page.get_text("blocks")
            lap(report, "text", t)
            for block in blocks:
                x0, y0, x1, y1, text, _, block_type = block[:7]
                if block_type != 0 or not text.strip():
                    continue
                if any(fitz.Rect(x0, y0, x1, y1).intersects(b) for b in table_bboxes):
                    continue  # already captured as table content
                is_heading = block[5] in heading_blocks
                content = " ".join(text.split())
                if is_heading:
                    heading = content
                units.append(ExtractedUnit(
                    document_id=document_id, filename=filename, source_type="pdf",
                    locator=Locator(page=page_no, heading=heading),
                    content=content, content_type="heading" if is_heading else "paragraph",
                    metadata={"bbox": [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)]},
                ))
    if artwork_pages and report is not None:
        report.notes.append(f"Table detection skipped on {len(artwork_pages)} artwork-heavy pages (over {MAX_DRAWINGS_FOR_TABLES:,} "
                            f"vector paths); their text is extracted but any table there is not structured: {artwork_pages}")
    return units
