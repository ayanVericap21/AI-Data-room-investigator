"""Process Documents stage: extract -> chunk -> persist chunks -> embed into pgvector.

Idempotent: a document whose status is ``processed`` and whose chunks all have embeddings
is skipped (its identity is its content hash, so unchanged files are never redone). If
embedding fails part-way, the saved chunks are kept and the next run embeds only the
missing ones.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.core.exceptions import AppError
from app.llm.embeddings import EmbeddingProvider
from app.llm.errors import LLMError
from app.models import Document, DocumentChunk
from app.models.base import ProcessingStatus
from app.services.chunking import chunk_units, embedding_text
from app.services.document_service import DocumentService
from app.services.extraction_service import extract_document

logger = logging.getLogger(__name__)
ProgressFn = Callable[[str, dict], None]


@dataclass
class ProcessResult:
    document_id: str
    filename: str
    status: str
    chunks: int = 0
    skipped: bool = False
    error: str | None = None


class ProcessingService:
    def __init__(self, session: Session, documents: DocumentService, embedder: EmbeddingProvider,
                 on_progress: ProgressFn | None = None):
        self.session = session
        self.documents = documents
        self.embedder = embedder
        self.on_progress = on_progress or (lambda _m, _d: None)

    def process_pending(self) -> list[ProcessResult]:
        results = []
        stmt = select(Document).where(Document.processing_status.in_(
            [ProcessingStatus.pending.value, ProcessingStatus.failed.value, ProcessingStatus.processing.value]
        )).order_by(Document.filename)
        for doc in list(self.session.scalars(stmt)):
            results.append(self.process_document(doc))
        done = sum(1 for r in results if r.status == ProcessingStatus.processed.value)
        self.on_progress(f"{done} of {len(results)} documents processed", {"processed": done, "total": len(results)})
        return results

    def _missing_embeddings(self, document_id: str) -> int:
        return self.session.scalar(
            select(func.count()).select_from(DocumentChunk)
            .where(DocumentChunk.document_id == document_id, DocumentChunk.embedding.is_(None))
        ) or 0

    def _chunk_count(self, document_id: str) -> int:
        return self.session.scalar(
            select(func.count()).select_from(DocumentChunk).where(DocumentChunk.document_id == document_id)
        ) or 0

    def process_document(self, doc: Document) -> ProcessResult:
        existing = self._chunk_count(doc.id)
        if (doc.processing_status == ProcessingStatus.processed.value
                and existing > 0 and self._missing_embeddings(doc.id) == 0):
            return ProcessResult(doc.id, doc.filename, doc.processing_status, existing, skipped=True)

        doc.processing_status = ProcessingStatus.processing.value
        self.session.commit()
        try:
            # Reuse already-saved chunks (embedding-only retry); otherwise extract + chunk afresh.
            if existing == 0:
                path = self.documents.resolve_path(doc)
                units = extract_document(path, doc.id, doc.filename)
                drafts = chunk_units(units)
                self.session.execute(delete(DocumentChunk).where(DocumentChunk.document_id == doc.id))
                for d in drafts:
                    self.session.add(DocumentChunk(
                        document_id=doc.id, chunk_index=d.chunk_index, content=d.content,
                        content_type=d.content_type, page_number=d.page_number, sheet_name=d.sheet_name,
                        cell_reference=d.cell_reference, heading=d.heading, metadata_=d.metadata,
                    ))
                self.session.commit()
            self._embed_missing(doc)
            doc.processing_status = ProcessingStatus.processed.value
            self.session.commit()
            count = self._chunk_count(doc.id)
            self.on_progress(f"Processed {doc.filename}", {"document_id": doc.id, "chunks": count})
            return ProcessResult(doc.id, doc.filename, doc.processing_status, count)
        except (AppError, LLMError, OSError) as exc:
            self.session.rollback()
            doc = self.session.get(Document, doc.id)
            doc.processing_status = ProcessingStatus.failed.value
            self.session.commit()
            logger.warning("Document processing failed", extra={"document": doc.filename, "error": str(exc)})
            self.on_progress(f"Failed to process {doc.filename}: {exc}", {"document_id": doc.id, "error": str(exc)})
            return ProcessResult(doc.id, doc.filename, doc.processing_status, error=str(exc))

    def _embed_missing(self, doc: Document) -> None:
        stmt = (select(DocumentChunk).where(DocumentChunk.document_id == doc.id, DocumentChunk.embedding.is_(None))
                .order_by(DocumentChunk.chunk_index))
        chunks = list(self.session.scalars(stmt))
        batch = self.embedder.MAX_BATCH
        for start in range(0, len(chunks), batch):
            part = chunks[start:start + batch]
            vectors = self.embedder.embed([embedding_text(doc.filename, c.heading, c.content) for c in part],
                                          task="document")
            for chunk, vector in zip(part, vectors):
                chunk.embedding = vector
            self.session.commit()  # persist progress per batch so a later failure keeps finished work
