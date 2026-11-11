"""Deterministic document outline built from stored chunks. No LLM.

Sections follow the document's own structure: sheets for XLSX, headings for DOCX, pages
for PDF. Consecutive chunks with the same section key are merged, so the outline is a
cheap map the agent can read before deciding what to open.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.exceptions import NotFoundError
from app.models import Document, DocumentChunk

SectionKind = Literal["sheet", "heading", "page", "body"]


class OutlineSection(BaseModel):
    title: str
    kind: SectionKind
    first_chunk_index: int
    last_chunk_index: int
    chunk_count: int
    page_start: int | None = None
    page_end: int | None = None
    sheet_name: str | None = None
    heading: str | None = None


class DocumentOutline(BaseModel):
    document_id: str
    filename: str
    file_type: str
    status: str
    chunk_count: int
    sections: list[OutlineSection]


def _section_key(chunk: DocumentChunk) -> tuple[SectionKind, object]:
    if chunk.sheet_name:
        return "sheet", chunk.sheet_name
    if chunk.heading:
        return "heading", chunk.heading
    if chunk.page_number is not None:
        return "page", chunk.page_number
    return "body", None


def build_outline(session: Session, document_id: str) -> DocumentOutline:
    document = session.get(Document, document_id)
    if document is None:
        raise NotFoundError(f"Document '{document_id}' not found")
    chunks = session.scalars(
        select(DocumentChunk).where(DocumentChunk.document_id == document_id).order_by(DocumentChunk.chunk_index)
    ).all()

    sections: list[OutlineSection] = []
    current_key: tuple[SectionKind, object] | None = None
    for chunk in chunks:
        key = _section_key(chunk)
        if key != current_key:
            kind, value = key
            title = {"sheet": f"Sheet: {value}", "heading": str(value), "page": f"Page {value}",
                     "body": "Start of document"}[kind]
            sections.append(OutlineSection(
                title=title, kind=kind, first_chunk_index=chunk.chunk_index, last_chunk_index=chunk.chunk_index,
                chunk_count=0, sheet_name=chunk.sheet_name if kind == "sheet" else None,
                heading=chunk.heading if kind == "heading" else None,
            ))
            current_key = key
        section = sections[-1]
        section.last_chunk_index = chunk.chunk_index
        section.chunk_count += 1
        if chunk.page_number is not None:
            section.page_start = chunk.page_number if section.page_start is None else min(section.page_start, chunk.page_number)
            section.page_end = chunk.page_number if section.page_end is None else max(section.page_end, chunk.page_number)

    return DocumentOutline(
        document_id=document.id, filename=document.filename, file_type=document.file_type,
        status=document.processing_status, chunk_count=len(chunks), sections=sections,
    )
