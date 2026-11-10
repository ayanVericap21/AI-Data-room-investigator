"""PDF extraction with PyMuPDF: page-wise paragraphs and (where detectable) tables."""
from __future__ import annotations

from pathlib import Path

import pymupdf as fitz  # PyMuPDF

from app.core.exceptions import ExtractionError
from app.schemas.extraction import ExtractedUnit, Locator


def count_pages(path: Path) -> int | None:
    try:
        with fitz.open(path) as doc:
            return doc.page_count
    except Exception:
        return None


def _rows_to_text(rows: list[list[str | None]]) -> str:
    return "\n".join(" | ".join((c or "").strip() for c in row) for row in rows)


def extract_pdf(path: Path, document_id: str, filename: str) -> list[ExtractedUnit]:
    try:
        doc = fitz.open(path)
    except Exception as exc:
        raise ExtractionError(f"Cannot open PDF '{filename}': {exc}") from exc
    units: list[ExtractedUnit] = []
    with doc:
        if doc.needs_pass:
            raise ExtractionError(f"PDF '{filename}' is encrypted")
        for page in doc:
            page_no = page.number + 1
            table_bboxes: list[fitz.Rect] = []
            try:
                tables = page.find_tables().tables
            except Exception:  # table detection is best-effort
                tables = []
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
            for block in page.get_text("blocks"):
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
    return units
