"""Embedding of document chunks (pgvector), in one place so processing and search use the same text and the same batches.

A ledger run does not read chunk vectors, so it can leave them out (``ProcessingService(embed_chunks=False)``); the agent's search does
read them, and embeds whatever is still missing the first time it searches (``RetrievalService``). A chunk is embedded from
``embedding_text`` (file name, heading, content), exactly as processing has always done.
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.llm.embeddings import EmbeddingProvider
from app.models import Document, DocumentChunk
from app.models.base import ProcessingStatus
from app.services.chunking import embedding_text


def embed_missing(session: Session, embedder: EmbeddingProvider, document: Document | None = None) -> int:
    """Embed every chunk that has no vector yet: those of ``document``, or of every document that is not missing. Progress is committed per
    batch, so a failure part-way keeps what was done and the next call embeds only the rest. Returns the number of chunks embedded."""
    stmt = (select(DocumentChunk, Document.filename).join(Document, Document.id == DocumentChunk.document_id)
            .where(DocumentChunk.embedding.is_(None)).order_by(Document.filename, DocumentChunk.chunk_index))
    stmt = stmt.where(DocumentChunk.document_id == document.id) if document is not None else stmt.where(Document.processing_status != ProcessingStatus.missing.value)
    rows = list(session.execute(stmt))
    batch = embedder.MAX_BATCH
    for start in range(0, len(rows), batch):
        part = rows[start:start + batch]
        vectors = embedder.embed([embedding_text(filename, chunk.heading, chunk.content) for chunk, filename in part], task="document")
        for (chunk, _filename), vector in zip(part, vectors):
            chunk.embedding = vector
        session.commit()  # persist progress per batch so a later failure keeps finished work
    return len(rows)
