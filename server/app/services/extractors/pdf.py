"""PDF extraction with PyMuPDF: page-wise paragraphs and (where detectable) tables."""
from __future__ import annotations

import time
from pathlib import Path

import pymupdf as fitz  # PyMuPDF

from pydantic import BaseModel

from app.core.exceptions import ExtractionError, ScannedDocumentError
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


def _rows_to_text(rows: list[list[str | None]]) -> str:
    return "\n".join(" | ".join((c or "").strip() for c in row) for row in rows)


def extract_pdf(path: Path, document_id: str, filename: str, report: ExtractionReport | None = None) -> list[ExtractedUnit]:
    """``report`` (optional) receives milliseconds per part (open, scan_check, tables, text) and notes about
    picture-only pages that were not captured and artwork pages where table detection was skipped."""
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
        if check.is_scanned():
            raise ScannedDocumentError(
                f"'{filename}' is picture-only (no text layer) on {len(check.pages_without_text)} of {check.pages} pages, "
                "so it looks like a scanned document. It is left out because there is no OCR.")
        if check.pages_without_text and report is not None:
            report.notes.append(f"Picture-only pages with no text layer were not captured (no OCR): {check.pages_without_text}")
        for page in doc:
            page_no = page.number + 1
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
                rows = table.extract()
                text = _rows_to_text(rows)
                if not text.strip():
                    continue
                table_bboxes.append(fitz.Rect(table.bbox))
                units.append(ExtractedUnit(
                    document_id=document_id, filename=filename, source_type="pdf",
                    locator=Locator(page=page_no, table_index=t_idx),
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
                units.append(ExtractedUnit(
                    document_id=document_id, filename=filename, source_type="pdf",
                    locator=Locator(page=page_no),
                    content=" ".join(text.split()), content_type="paragraph",
                    metadata={"bbox": [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)]},
                ))
    if artwork_pages and report is not None:
        report.notes.append(f"Table detection skipped on {len(artwork_pages)} artwork-heavy pages (over {MAX_DRAWINGS_FOR_TABLES:,} "
                            f"vector paths); their text is extracted but any table there is not structured: {artwork_pages}")
    return units
