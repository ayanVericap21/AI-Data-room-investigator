"""DOCX extraction with python-docx, preserving body order, headings and tables."""
from __future__ import annotations

from pathlib import Path

import docx
from docx.table import Table
from docx.text.paragraph import Paragraph

from app.core.exceptions import ExtractionError
from app.schemas.extraction import ExtractedUnit, Locator


def _heading_level(paragraph: Paragraph) -> int | None:
    name = (paragraph.style.name if paragraph.style is not None else "") or ""
    if name == "Title":
        return 0
    if name.startswith("Heading "):
        try:
            return int(name.split()[1])
        except (IndexError, ValueError):
            return 1
    return None


def extract_docx(path: Path, document_id: str, filename: str) -> list[ExtractedUnit]:
    try:
        document = docx.Document(str(path))
    except Exception as exc:
        raise ExtractionError(f"Cannot open DOCX '{filename}': {exc}") from exc

    units: list[ExtractedUnit] = []
    heading: str | None = None
    para_no = 0
    table_no = 0
    # Iterate the body in document order so tables stay next to their surrounding text.
    for child in document.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            paragraph = Paragraph(child, document)
            text = paragraph.text.strip()
            if not text:
                continue
            para_no += 1
            level = _heading_level(paragraph)
            if level is not None:
                heading = text
            units.append(ExtractedUnit(
                document_id=document_id, filename=filename, source_type="docx",
                locator=Locator(paragraph_index=para_no, heading=heading),
                content=text, content_type="heading" if level is not None else "paragraph",
                metadata={"style": paragraph.style.name if paragraph.style is not None else None},
            ))
        elif tag == "tbl":
            table_no += 1
            rows = [[cell.text.strip() for cell in row.cells] for row in Table(child, document).rows]
            text = "\n".join(" | ".join(r) for r in rows)
            if not text.strip():
                continue
            units.append(ExtractedUnit(
                document_id=document_id, filename=filename, source_type="docx",
                locator=Locator(table_index=table_no, heading=heading),
                content=text, content_type="table", metadata={"rows": rows},
            ))
    return units
