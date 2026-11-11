"""Read-only tools: list_documents, get_outline, get_chunk, read_section."""
from __future__ import annotations

import difflib

from sqlalchemy import func, select

from app.models import Document, DocumentChunk
from app.models.base import ProcessingStatus
from app.services.outline_service import DocumentOutline, build_outline
from app.services.retrieval_service import RetrievedChunk
from app.tools.base import ToolContext, ToolError
from app.tools.schemas import (ChunkView, Coverage, DocumentSummary, GetChunkInput, GetChunkOutput, GetOutlineInput,
                               ListDocumentsInput, ListDocumentsOutput, ReadSectionInput, ReadSectionOutput)

MAX_SECTION_CHARS = 12_000  # one read_section reply never exceeds this much source text


def locator_text(chunk: DocumentChunk | RetrievedChunk) -> str:
    """Readable source location; works for stored chunks and search hits alike."""
    if chunk.sheet_name:
        return f"sheet {chunk.sheet_name}" + (f", cell {chunk.cell_reference}" if chunk.cell_reference else "")
    parts = []
    if chunk.page_number is not None:
        parts.append(f"page {chunk.page_number}")
    if chunk.heading:
        parts.append(f"heading {chunk.heading}")
    return ", ".join(parts) or "document body"


def chunk_view(chunk: DocumentChunk, filename: str, score: float | None = None) -> ChunkView:
    return ChunkView(
        chunk_id=chunk.id, document_id=chunk.document_id, filename=filename, locator=locator_text(chunk),
        content=chunk.content, content_type=chunk.content_type, page_number=chunk.page_number,
        sheet_name=chunk.sheet_name, cell_reference=chunk.cell_reference, heading=chunk.heading, score=score,
    )


def list_documents(ctx: ToolContext, _: ListDocumentsInput) -> ListDocumentsOutput:
    rows = ctx.session.execute(
        select(Document, func.count(DocumentChunk.id))
        .outerjoin(DocumentChunk, DocumentChunk.document_id == Document.id)
        .where(Document.processing_status != ProcessingStatus.missing.value)
        .group_by(Document.id).order_by(Document.filename)
    ).all()
    return ListDocumentsOutput(documents=[
        DocumentSummary(document_id=d.id, filename=d.filename, file_type=d.file_type,
                        status=d.processing_status, chunk_count=count, opened=d.id in ctx.opened_documents)
        for d, count in rows
    ])


def get_outline(ctx: ToolContext, inp: GetOutlineInput) -> DocumentOutline:
    try:
        outline = build_outline(ctx.session, inp.document_id)
    except Exception as exc:  # NotFoundError and friends
        raise ToolError("not_found", str(exc)) from exc
    ctx.opened_documents.add(inp.document_id)
    return outline


def coverage(ctx: ToolContext) -> Coverage:
    """Readable documents (those with chunks) the agent has not yet opened."""
    rows = ctx.session.execute(
        select(Document.id, Document.filename).join(DocumentChunk, DocumentChunk.document_id == Document.id)
        .where(Document.processing_status != ProcessingStatus.missing.value)
        .group_by(Document.id, Document.filename).order_by(Document.filename)
    ).all()
    not_opened = [name for doc_id, name in rows if doc_id not in ctx.opened_documents]
    return Coverage(total=len(rows), opened=len(rows) - len(not_opened), not_opened=not_opened)


def get_chunk(ctx: ToolContext, inp: GetChunkInput) -> GetChunkOutput:
    row = ctx.session.execute(
        select(DocumentChunk, Document.filename).join(Document, Document.id == DocumentChunk.document_id)
        .where(DocumentChunk.id == inp.chunk_id)
    ).first()
    if row is None:
        raise ToolError("not_found", f"Chunk '{inp.chunk_id}' does not exist. Use ids returned by search or read_section.")
    chunk, filename = row
    ctx.opened_documents.add(chunk.document_id)
    before: list[ChunkView] = []
    after: list[ChunkView] = []
    if inp.neighbors:
        near = ctx.session.scalars(
            select(DocumentChunk).where(
                DocumentChunk.document_id == chunk.document_id,
                DocumentChunk.chunk_index.between(chunk.chunk_index - inp.neighbors, chunk.chunk_index + inp.neighbors),
                DocumentChunk.id != chunk.id,
            ).order_by(DocumentChunk.chunk_index)
        ).all()
        before = [chunk_view(c, filename) for c in near if c.chunk_index < chunk.chunk_index]
        after = [chunk_view(c, filename) for c in near if c.chunk_index > chunk.chunk_index]
    return GetChunkOutput(chunk=chunk_view(chunk, filename), before=before, after=after)


def read_section(ctx: ToolContext, inp: ReadSectionInput) -> ReadSectionOutput:
    document = ctx.session.get(Document, inp.document_id)
    if document is None:
        raise ToolError("not_found", f"Document '{inp.document_id}' does not exist. Use list_documents.")
    ctx.opened_documents.add(document.id)
    stmt = select(DocumentChunk).where(DocumentChunk.document_id == document.id)
    column, wanted = None, ""
    if inp.page_number is not None:
        stmt, selector = stmt.where(DocumentChunk.page_number == inp.page_number), f"page {inp.page_number}"
    elif inp.sheet_name is not None:
        # Case and surrounding spaces are ignored: real sheet names can end in a space ("Sheet "), which a model drops.
        wanted, column = inp.sheet_name.strip().lower(), DocumentChunk.sheet_name
        stmt, selector = stmt.where(func.lower(func.trim(column)) == wanted), f"sheet {inp.sheet_name.strip()}"
    else:
        wanted, column = inp.heading.strip().lower(), DocumentChunk.heading
        stmt, selector = stmt.where(func.lower(func.trim(column)) == wanted), f"heading {inp.heading.strip()}"
    chunks = ctx.session.scalars(stmt.order_by(DocumentChunk.chunk_index)).all()
    if not chunks:
        hint = ""
        if column is not None:
            names = [n.strip() for n in ctx.session.scalars(
                select(column).where(DocumentChunk.document_id == document.id, column.is_not(None)).distinct()) if n and n.strip()]
            by_lower = {n.lower(): n for n in names}
            close = [by_lower[m] for m in difflib.get_close_matches(wanted, list(by_lower), n=3, cutoff=0.4)]
            hint = f" Did you mean: {', '.join(repr(c) for c in close)}?" if close else ""
        raise ToolError("not_found", f"Nothing matches {selector} in {document.filename}.{hint} "
                                     "Call get_outline to see the valid pages, sheets and headings.")
    shown: list[DocumentChunk] = []
    used = 0
    for chunk in chunks[: inp.max_chunks]:
        if shown and used + len(chunk.content) > MAX_SECTION_CHARS:
            break
        shown.append(chunk)
        used += len(chunk.content)
    return ReadSectionOutput(
        document_id=document.id, filename=document.filename, selector=selector, total_chunks=len(chunks),
        truncated=len(shown) < len(chunks), chunks=[chunk_view(c, document.filename) for c in shown],
    )
