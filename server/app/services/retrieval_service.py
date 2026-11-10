"""Semantic retrieval over pgvector (cosine distance, HNSW index).

PostgreSQL holds both the text and its vector, so a hit already carries its full source
metadata. Similarity is only a *retrieval* signal; it is never evidence of a contradiction.
"""
from __future__ import annotations

from pydantic import BaseModel
from sqlalchemy import Float, select
from sqlalchemy.orm import Session

from app.core.exceptions import AppError
from app.llm.embeddings import EmbeddingProvider
from app.models import Claim, Document, DocumentChunk
from app.models.base import ProcessingStatus


class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    score: float  # cosine similarity, 1.0 = identical direction
    content: str
    content_type: str
    page_number: int | None = None
    sheet_name: str | None = None
    cell_reference: str | None = None
    heading: str | None = None


class RetrievalService:
    def __init__(self, session: Session, embedder: EmbeddingProvider):
        self.session = session
        self.embedder = embedder

    def search(self, query: str, top_k: int = 5, document_ids: list[str] | None = None,
               exclude_chunk_ids: list[str] | None = None, exclude_document_ids: list[str] | None = None,
               min_score: float | None = None) -> list[RetrievedChunk]:
        if self.session.get_bind().dialect.name != "postgresql":
            raise AppError("Semantic search requires PostgreSQL with pgvector")
        if not query.strip():
            return []
        vector = self.embedder.embed([query], task="query")[0]
        distance = DocumentChunk.embedding.op("<=>", return_type=Float)(vector)
        stmt = (
            select(DocumentChunk, Document.filename, distance.label("distance"))
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(DocumentChunk.embedding.is_not(None),
                   Document.processing_status != ProcessingStatus.missing.value)
        )
        if document_ids:
            stmt = stmt.where(DocumentChunk.document_id.in_(document_ids))
        if exclude_document_ids:
            stmt = stmt.where(DocumentChunk.document_id.not_in(exclude_document_ids))
        if exclude_chunk_ids:
            stmt = stmt.where(DocumentChunk.id.not_in(exclude_chunk_ids))
        stmt = stmt.order_by(distance).limit(top_k)

        results = []
        for chunk, filename, dist in self.session.execute(stmt):
            score = 1.0 - float(dist)
            if min_score is not None and score < min_score:
                continue
            results.append(RetrievedChunk(
                chunk_id=chunk.id, document_id=chunk.document_id, filename=filename, score=score,
                content=chunk.content, content_type=chunk.content_type, page_number=chunk.page_number,
                sheet_name=chunk.sheet_name, cell_reference=chunk.cell_reference, heading=chunk.heading,
            ))
        return results

    def search_for_claim(self, claim: Claim, top_k: int = 5, exclude_own_document: bool = False) -> list[RetrievedChunk]:
        """Passages relevant to a claim, excluding the claim's own source chunk."""
        parts = [claim.subject, claim.metric, claim.period, claim.conditions, claim.claim_text]
        query = " ".join(p for p in parts if p)
        return self.search(
            query, top_k=top_k, exclude_chunk_ids=[claim.chunk_id],
            exclude_document_ids=[claim.document_id] if exclude_own_document else None,
        )
