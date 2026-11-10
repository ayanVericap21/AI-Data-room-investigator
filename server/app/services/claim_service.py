"""Extract Claims stage: LLM extraction validated by Pydantic plus deterministic checks.

The LLM proposes claims per batch of chunks. Python then rejects any claim that
- references a chunk that was not in the batch (no invented citations), or
- states a value that does not appear in the cited chunk text (no invented numbers).
Rejections are counted and logged, never silently kept.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.llm.base import LLMClient
from app.llm.errors import LLMInvalidOutputError, LLMTransientError
from app.llm.prompts import claim_extraction
from app.models import Claim, Document, DocumentChunk
from app.models.base import ProcessingStatus
from app.schemas.claim import ClaimExtractionResult, ClaimOut, ClaimView, SourceRef
from app.services.numbers import value_supported_by_text

logger = logging.getLogger(__name__)
ProgressFn = Callable[[str, dict], None]
EXCERPT_TYPES_SKIPPED = {"heading"}


@dataclass
class ExtractionSummary:
    documents_processed: int = 0
    documents_skipped: int = 0
    documents_failed: int = 0
    claims_created: int = 0
    claims_rejected: int = 0
    llm_calls: int = 0
    stopped: bool = False
    stop_reason: str | None = None
    rejections: list[str] = field(default_factory=list)


def location_label(chunk: DocumentChunk) -> str:
    if chunk.sheet_name:
        return f"sheet {chunk.sheet_name}, {chunk.cell_reference}"
    parts = []
    if chunk.page_number:
        parts.append(f"page {chunk.page_number}")
    if chunk.heading:
        parts.append(f"section '{chunk.heading}'")
    parts.append(chunk.content_type)
    return ", ".join(parts)


class ClaimService:
    def __init__(self, session: Session, llm: LLMClient, max_chars_per_call: int = 8000,
                 max_llm_calls: int | None = None, on_progress: ProgressFn | None = None,
                 should_stop: Callable[[], bool] | None = None):
        self.session = session
        self.llm = llm
        self.max_chars = max_chars_per_call
        self.max_llm_calls = max_llm_calls
        self.on_progress = on_progress or (lambda _m, _d: None)
        self.should_stop = should_stop or (lambda: False)

    # -- extraction ----------------------------------------------------------
    def extract_all(self, force: bool = False) -> ExtractionSummary:
        summary = ExtractionSummary()
        stmt = select(Document).where(Document.processing_status == ProcessingStatus.processed.value).order_by(Document.filename)
        for doc in list(self.session.scalars(stmt)):
            if summary.stopped:
                break
            self.extract_document(doc, summary, force)
        self.on_progress(f"{summary.claims_created} claims extracted",
                         {"claims": summary.claims_created, "rejected": summary.claims_rejected})
        return summary

    def _has_claims(self, document_id: str) -> bool:
        return (self.session.scalar(select(func.count()).select_from(Claim).where(Claim.document_id == document_id)) or 0) > 0

    def extract_document(self, doc: Document, summary: ExtractionSummary, force: bool = False) -> None:
        if self._has_claims(doc.id):
            if not force:
                summary.documents_skipped += 1
                return
            self.session.execute(delete(Claim).where(Claim.document_id == doc.id))
            self.session.commit()

        chunks = list(self.session.scalars(
            select(DocumentChunk).where(DocumentChunk.document_id == doc.id).order_by(DocumentChunk.chunk_index)))
        chunks = [c for c in chunks if c.content_type not in EXCERPT_TYPES_SKIPPED]
        created, failed = 0, False
        for batch in self._batches(chunks):
            reason = self._stop_reason(summary)
            if reason:
                summary.stopped, summary.stop_reason, failed = True, reason, True
                break
            try:
                summary.llm_calls += 1
                result = self.llm.generate_structured_output(
                    claim_extraction.build_prompt(doc.filename, [
                        {"chunk_id": c.id, "location": location_label(c), "content": c.content} for c in batch]),
                    ClaimExtractionResult, system=claim_extraction.SYSTEM)
            except (LLMTransientError, LLMInvalidOutputError) as exc:
                logger.warning("Claim extraction batch failed", extra={"document": doc.filename, "error": str(exc)})
                failed = True
                break
            created += self._persist(doc, batch, result, summary)

        if failed:
            # Keep extraction all-or-nothing per document so a rerun redoes it cleanly.
            self.session.rollback()
            self.session.execute(delete(Claim).where(Claim.document_id == doc.id))
            self.session.commit()
            summary.claims_created -= created
            summary.documents_failed += 1
            self.on_progress(f"Claim extraction incomplete for {doc.filename}", {"document_id": doc.id})
        else:
            self.session.commit()
            summary.documents_processed += 1
            self.on_progress(f"Extracted {created} claims from {doc.filename}", {"document_id": doc.id, "claims": created})

    def _stop_reason(self, summary: ExtractionSummary) -> str | None:
        if self.should_stop():
            return "stop requested"
        if self.max_llm_calls is not None and summary.llm_calls >= self.max_llm_calls:
            return "LLM call limit reached"
        return None

    def _batches(self, chunks: list[DocumentChunk]) -> list[list[DocumentChunk]]:
        batches, current, size = [], [], 0
        for c in chunks:
            if current and size + len(c.content) > self.max_chars:
                batches.append(current)
                current, size = [], 0
            current.append(c)
            size += len(c.content)
        if current:
            batches.append(current)
        return batches

    def _persist(self, doc: Document, batch: list[DocumentChunk], result: ClaimExtractionResult,
                 summary: ExtractionSummary) -> int:
        by_id = {c.id: c for c in batch}
        created = 0
        for extracted in result.claims:
            chunk = by_id.get(extracted.chunk_id)
            if chunk is None:
                self._reject(summary, f"unknown chunk_id '{extracted.chunk_id}'")
                continue
            if not value_supported_by_text(extracted.value, chunk.content):
                self._reject(summary, f"value {extracted.value!r} not found in cited chunk")
                continue
            self.session.add(Claim(
                document_id=doc.id, chunk_id=chunk.id, subject=extracted.subject, metric=extracted.metric,
                value=extracted.value, unit=extracted.unit, period=extracted.period,
                conditions=extracted.conditions, assumptions=extracted.assumptions or None,
                claim_text=extracted.claim_text, claim_category=extracted.claim_category,
                extraction_confidence=extracted.confidence,
            ))
            created += 1
        summary.claims_created += created
        return created

    @staticmethod
    def _reject(summary: ExtractionSummary, reason: str) -> None:
        summary.claims_rejected += 1
        summary.rejections.append(reason)
        logger.warning("Claim rejected", extra={"reason": reason})

    # -- reads ---------------------------------------------------------------
    def list_claims(self, document_id: str | None = None, limit: int = 50, offset: int = 0) -> tuple[list[ClaimOut], int]:
        base = select(Claim, Document.filename, DocumentChunk).join(Document, Document.id == Claim.document_id) \
            .join(DocumentChunk, DocumentChunk.id == Claim.chunk_id) \
            .where(Document.processing_status != ProcessingStatus.missing.value)
        count = select(func.count()).select_from(Claim).join(Document, Document.id == Claim.document_id) \
            .where(Document.processing_status != ProcessingStatus.missing.value)
        if document_id:
            base, count = base.where(Claim.document_id == document_id), count.where(Claim.document_id == document_id)
        total = self.session.scalar(count) or 0
        rows = self.session.execute(base.order_by(Document.filename, DocumentChunk.chunk_index, Claim.created_at)
                                    .limit(limit).offset(offset))
        items = []
        for claim, filename, chunk in rows:
            out = ClaimOut.model_validate({
                **{k: getattr(claim, k) for k in ClaimOut.model_fields if k != "source"},
                "source": SourceRef(document_id=claim.document_id, filename=filename, chunk_id=chunk.id,
                                    content_type=chunk.content_type, page_number=chunk.page_number,
                                    sheet_name=chunk.sheet_name, cell_reference=chunk.cell_reference,
                                    heading=chunk.heading),
            })
            items.append(out)
        return items, total

    def claim_views(self) -> list[ClaimView]:
        stmt = select(Claim).join(Document, Document.id == Claim.document_id) \
            .where(Document.processing_status != ProcessingStatus.missing.value)
        return [ClaimView(id=c.id, document_id=c.document_id, chunk_id=c.chunk_id, subject=c.subject,
                          metric=c.metric, value=c.value, unit=c.unit, period=c.period,
                          conditions=c.conditions, assumptions=c.assumptions or []) for c in self.session.scalars(stmt)]
