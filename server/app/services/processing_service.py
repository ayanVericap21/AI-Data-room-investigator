"""Process Documents stage: extract -> chunk -> persist chunks -> embed into pgvector.

Idempotent: a document whose status is ``processed`` and whose chunks all have embeddings
is skipped (its identity is its content hash, so unchanged files are never redone). If
embedding fails part-way, the saved chunks are kept and the next run embeds only the
missing ones.

Parsing is CPU-bound and independent per file, so ``process_pending`` can parse all pending
documents at the same time in separate processes (``extract_workers``). Everything after that
(chunking, saving, embedding) still runs one document at a time, exactly as before.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.core.exceptions import AppError, ScannedDocumentError
from app.llm.embeddings import EmbeddingProvider
from app.llm.errors import LLMError
from app.models import Document, DocumentChunk
from app.models.base import ProcessingStatus
from app.schemas.extraction import ExtractedUnit, ExtractionReport
from app.services.chunking import chunk_units, embedding_text
from app.services.document_service import DocumentService
from app.services.extraction_service import extract_document

logger = logging.getLogger(__name__)
ProgressFn = Callable[[str, dict], None]


def _ms(started: float) -> int:
    return round((time.perf_counter() - started) * 1000)


@dataclass
class Extracted:
    """Outcome of parsing one file. Picklable, so it can come back from a worker process."""

    units: list[ExtractedUnit] = field(default_factory=list)
    report: ExtractionReport = field(default_factory=ExtractionReport)
    ms: int = 0
    error: Exception | None = None


def extract_one(path: str, document_id: str, filename: str) -> Extracted:
    """Parse one file. Never raises: an error is returned, and the caller re-raises it so it is handled
    exactly as if it had happened in-process. Module-level so a worker process can run it."""
    out = Extracted()
    started = time.perf_counter()
    try:
        out.units = extract_document(Path(path), document_id, filename, out.report)
    except Exception as exc:
        out.error = exc
    out.ms = _ms(started)
    return out


@dataclass
class ProcessResult:
    document_id: str
    filename: str
    status: str
    chunks: int = 0
    skipped: bool = False  # already processed earlier, nothing to do
    error: str | None = None
    note: str | None = None  # why a document was left out, or what was not captured
    # Milliseconds per part: extract, chunk, save, embed, plus extract.<part> detail (PDF: open, scan_check,
    # tables, text; XLSX: load_formulas, load_values, read_rows). Empty if the document was skipped as done.
    timings_ms: dict[str, int] = field(default_factory=dict)
    embed_calls: int = 0  # provider attempts spent embedding this document
    embed_failed_attempts: int = 0  # of those, how many failed and were retried


class ProcessingService:
    def __init__(self, session: Session, documents: DocumentService, embedder: EmbeddingProvider,
                 on_progress: ProgressFn | None = None, extract_workers: int = 1):
        self.session = session
        self.documents = documents
        self.embedder = embedder
        self.on_progress = on_progress or (lambda _m, _d: None)
        self.extract_workers = max(1, extract_workers)
        self.extract_phase_ms = 0  # wall time of the parallel parsing phase (0 when parsing one at a time)

    def process_pending(self) -> list[ProcessResult]:
        stmt = select(Document).where(Document.processing_status.in_(
            [ProcessingStatus.pending.value, ProcessingStatus.failed.value, ProcessingStatus.processing.value]
        )).order_by(Document.filename)
        docs = list(self.session.scalars(stmt))
        parsed = self._extract_in_parallel(docs)
        results = [self.process_document(doc, parsed.get(doc.id)) for doc in docs]
        done = sum(1 for r in results if r.status == ProcessingStatus.processed.value)
        self.on_progress(f"{done} of {len(results)} documents processed", {"processed": done, "total": len(results)})
        return results

    def _extract_in_parallel(self, docs: list[Document]) -> dict[str, Extracted]:
        """Parse every document that still needs it at the same time. Returns {} (and each document is then
        parsed in turn by process_document) with one worker, fewer than two documents, or if the pool fails."""
        self.extract_phase_ms = 0
        todo = [d for d in docs if self._chunk_count(d.id) == 0]
        if self.extract_workers <= 1 or len(todo) < 2:
            return {}
        workers = min(self.extract_workers, len(todo))
        self.on_progress(f"Parsing {len(todo)} documents in parallel ({workers} workers)", {"documents": len(todo), "workers": workers})
        started = time.perf_counter()
        try:
            paths = [str(self.documents.resolve_path(d)) for d in todo]
            with ProcessPoolExecutor(max_workers=workers) as pool:
                outcomes = list(pool.map(extract_one, paths, [d.id for d in todo], [d.filename for d in todo]))
        except Exception:
            logger.warning("Parallel parsing failed; parsing one document at a time", exc_info=True)
            return {}
        self.extract_phase_ms = _ms(started)
        return {d.id: o for d, o in zip(todo, outcomes)}

    def _missing_embeddings(self, document_id: str) -> int:
        return self.session.scalar(
            select(func.count()).select_from(DocumentChunk)
            .where(DocumentChunk.document_id == document_id, DocumentChunk.embedding.is_(None))
        ) or 0

    def _chunk_count(self, document_id: str) -> int:
        return self.session.scalar(
            select(func.count()).select_from(DocumentChunk).where(DocumentChunk.document_id == document_id)
        ) or 0

    def process_document(self, doc: Document, extracted: Extracted | None = None) -> ProcessResult:
        """``extracted`` is a parse already done in parallel; without it the file is parsed here."""
        existing = self._chunk_count(doc.id)
        if (doc.processing_status == ProcessingStatus.processed.value
                and existing > 0 and self._missing_embeddings(doc.id) == 0):
            return ProcessResult(doc.id, doc.filename, doc.processing_status, existing, skipped=True)

        doc.processing_status = ProcessingStatus.processing.value
        self.session.commit()
        note: str | None = None
        timings: dict[str, int] = {}
        embed_attempts_before = len(self.embedder.call_metrics)

        def embed_stats() -> tuple[int, int]:
            attempts = self.embedder.call_metrics[embed_attempts_before:]
            return len(attempts), sum(not m.ok for m in attempts)

        try:
            # Reuse already-saved chunks (embedding-only retry); otherwise extract + chunk afresh.
            if existing == 0:
                path = self.documents.resolve_path(doc)
                if extracted is None:
                    extracted = extract_one(str(path), doc.id, doc.filename)
                timings["extract"] = extracted.ms  # kept even when parsing failed or the PDF is skipped as scanned
                timings.update({f"extract.{k}": v for k, v in extracted.report.timings_ms.items()})
                if extracted.error is not None:
                    raise extracted.error
                note = " ".join(extracted.report.notes) or None
                t = time.perf_counter()
                drafts = chunk_units(extracted.units)
                timings["chunk"] = _ms(t)
                t = time.perf_counter()
                self.session.execute(delete(DocumentChunk).where(DocumentChunk.document_id == doc.id))
                for d in drafts:
                    self.session.add(DocumentChunk(
                        document_id=doc.id, chunk_index=d.chunk_index, content=d.content,
                        content_type=d.content_type, page_number=d.page_number, sheet_name=d.sheet_name,
                        cell_reference=d.cell_reference, heading=d.heading, metadata_=d.metadata,
                    ))
                self.session.commit()
                timings["save"] = _ms(t)
            t = time.perf_counter()
            try:
                self._embed_missing(doc)
            finally:
                timings["embed"] = _ms(t)
            doc.processing_status = ProcessingStatus.processed.value
            self.session.commit()
            count = self._chunk_count(doc.id)
            self.on_progress(f"Processed {doc.filename}", {"document_id": doc.id, "chunks": count})
            calls, failed = embed_stats()
            return ProcessResult(doc.id, doc.filename, doc.processing_status, count, note=note, timings_ms=timings,
                                 embed_calls=calls, embed_failed_attempts=failed)
        except ScannedDocumentError as exc:
            # Left out on purpose: not a failure, and never retried (skipped documents are not pending).
            self.session.rollback()
            doc = self.session.get(Document, doc.id)
            doc.processing_status = ProcessingStatus.skipped.value
            self.session.commit()
            logger.info("Scanned PDF left out", extra={"document": doc.filename})
            self.on_progress(f"Skipped {doc.filename}: scanned PDF", {"document_id": doc.id, "skipped": True})
            return ProcessResult(doc.id, doc.filename, doc.processing_status, note=exc.message, timings_ms=timings)
        except Exception as exc:  # expected failures and unexpected bugs alike: this document fails, the run goes on
            expected = isinstance(exc, (AppError, LLMError, OSError))
            error = str(exc) if expected else f"{exc.__class__.__name__}: {exc}"
            self.session.rollback()
            doc = self.session.get(Document, doc.id)
            doc.processing_status = ProcessingStatus.failed.value
            self.session.commit()
            if expected:
                logger.warning("Document processing failed", extra={"document": doc.filename, "error": error})
            else:
                logger.exception("Document processing crashed", extra={"document": doc.filename})
            self.on_progress(f"Failed to process {doc.filename}: {error}", {"document_id": doc.id, "error": error})
            calls, failed = embed_stats()
            return ProcessResult(doc.id, doc.filename, doc.processing_status, error=error, timings_ms=timings,
                                 embed_calls=calls, embed_failed_attempts=failed)

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
