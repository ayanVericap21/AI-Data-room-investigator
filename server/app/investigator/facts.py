"""Typed fact extraction: each document is read once by LlamaExtract against a typed-fact schema.

A fact is kept only when its quote is found in a stored chunk of that document (the quote is replaced by the verbatim source text).
Observations and assertions with a value are also stored as claims, which is the input of the semantic comparison family (I);
constraints, events, references, definitions and conclusions never become claims, so a limit is never compared with a measurement
as if they were the same kind of thing.

Documents are read in parallel (``workers``); everything that touches the database happens on the calling thread, in document order.
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.investigator.models import FACT_TYPES, MODALITIES, VALUE_TYPES, Fact, FactBatch, TypedFact
from app.investigator.tables import TableStore, parse_when, to_num
from app.llm.llama_extract import LlamaExtractClient, LlamaExtractError
from app.models import Document, DocumentChunk
from app.schemas.claim import ClaimExtractionResult, ExtractedClaim
from app.services.claim_service import EXCERPT_TYPES_SKIPPED, ClaimService, ExtractionSummary
from app.services.numbers import value_supported_by_text
from app.tools.submit import find_source_span

logger = logging.getLogger(__name__)
ProgressFn = Callable[[str], None]

TYPE_HELP = (
    "observation: a measured, recorded or counted value stated in prose or in a summary or headline table (cover 33.3% at a site, 38 samples collected). "
    "constraint: a rule, limit, window, deadline, quota, required approval, required document or allowed list (operator <=, >=, =, between, in, must). "
    "event: something that happened or is scheduled, at a time and place, with its participants (a dive, a collection, a shipment, a receipt, a notification, an incident, a loss). "
    "assertion: a statement of fact in narrative, above all a quantified or universal one (all, no, none, every, only, fully, never) or a stated total, mean, count or percentage; "
    "for a computed figure fill aggregate_of with what it is computed from. "
    "reference: a pointer to another document, annex, schedule, form, approval, log, report, table or record, including one the text says is missing, pending or not provided; value is the item as named. "
    "definition: how a metric or term is defined or computed. "
    "conclusion: an inference, interpretation or causal claim (demonstrating recovery, attributable to, confirms).")

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "description": "Every fact the document states that could be checked against another part of the data room.",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": list(FACT_TYPES), "description": TYPE_HELP},
                    "subject": {"type": "string", "description": "What the fact is about: an entity, record, site, person, item or document. Include every identifying part (an entity and a date, a place and a date)"},
                    "attribute": {"type": "string", "description": "The property, rule, action or relation stated"},
                    "operator": {"type": "string", "description": "= for a stated value; <=, >=, <, >, between or in for a constraint"},
                    "value": {"type": "string", "description": "The value as written; for an event, a reference or a conclusion, the short thing itself"},
                    "unit": {"type": "string", "description": "Unit of the value, if stated"},
                    "scope": {"type": "string", "description": "What the fact applies to: per site, per day, the whole project, a period, a condition"},
                    "when": {"type": "string", "description": "The date or time the fact is about or applies from, as written"},
                    "modality": {"type": "string", "enum": list(MODALITIES), "description": "must for a rule or requirement, planned, actual, reported or estimated"},
                    "aggregate_of": {"type": "string", "description": "For a total, mean, count or percentage: what it is computed from"},
                    "value_type": {"type": "string", "enum": list(VALUE_TYPES), "description": "quantity, date, identifier, name, state or statement; for an observation or assertion with a value"},
                    "quote": {"type": "string", "description": "The sentence or line that states the fact, copied exactly from the document"},
                },
                "required": ["type", "subject", "attribute", "value", "quote"],
            },
        },
    },
}

SYSTEM_PROMPT = (
    "Read the document and return every fact it states that could be checked against another part of the data room, as typed facts. "
    "Types: " + TYPE_HELP + " "
    "Rules: copy quote exactly as written (one sentence or one table row), never shorten with an ellipsis, never calculate or infer. "
    "Do not return one fact per cell of a long table of records: those tables are loaded separately and queried. Do return rules, limits and schedules, "
    "events, notes, definitions, references, conclusions and every headline, summary or aggregate number, wherever they appear. "
    "Give each fact the subject that fully identifies what it is about (all identifying parts: an entity and its date, a place and its date). "
    "A rule that says what must, may not or may at most happen is a constraint, not an observation. A sentence that interprets results is a conclusion. "
    "Return an empty list only when the document states nothing checkable.")


@dataclass
class FactStats:
    documents: int = 0
    facts: int = 0
    rejected: int = 0
    claims: int = 0
    retried: int = 0
    credits: float = 0.0
    by_type: Counter = field(default_factory=Counter)
    failed: list[str] = field(default_factory=list)
    rejections: list[str] = field(default_factory=list)


def guess_value_type(value: Any, unit: str | None) -> str:
    text = str(value or "").strip()
    if to_num(text) is not None:
        return "quantity"
    if parse_when(text):
        return "date"
    if re.match(r"^[A-Za-z]{1,6}[-_]\d", text):
        return "identifier"
    return "statement"


def locate(quote: str, value: Any, subject: str, chunks: list[DocumentChunk]) -> tuple[DocumentChunk, str] | None:
    """The chunk that holds a fact and the verbatim text of its quote. The whole quote is tried first, then its two ends (a long sentence can run across
    chunks), then the value together with the subject."""
    quote = (quote or "").strip()
    if quote:
        for chunk in chunks:
            span = find_source_span(chunk.content or "", quote)
            if span:
                return chunk, span[0]
        if len(quote) > 160:
            for piece in (quote[:120], quote[-120:]):
                for chunk in chunks:
                    span = find_source_span(chunk.content or "", piece)
                    if span:
                        return chunk, span[0]
    if value not in (None, ""):
        holding = [c for c in chunks if value_supported_by_text(value, c.content or "")]
        wanted = " ".join((subject or "").split()).casefold()
        if wanted and len(holding) > 1:
            holding = [c for c in holding if wanted in " ".join((c.content or "").split()).casefold()] or holding
        if holding:
            chunk = holding[0]
            text = str(value)
            line = next((ln for ln in (chunk.content or "").split("\n") if text.lower() in ln.lower()), chunk.content or "")
            return chunk, line.strip()[:400] if len(line) > 400 else line.strip()
    return None


class FactExtractor:
    def __init__(self, session: Session, client: LlamaExtractClient | None, data_room_dir: Path, store: TableStore, llm: Any,
                 workers: int = 1, on_progress: ProgressFn | None = None, should_stop: Callable[[], bool] | None = None):
        self.session = session
        self.client = client
        self.root = data_room_dir
        self.store = store
        self.workers = max(1, workers)
        self.on_progress = on_progress or (lambda _m: None)
        self.should_stop = should_stop or (lambda: False)
        self._claims = ClaimService(session, llm)
        self.facts: list[Fact] = []
        self.summary = ExtractionSummary()
        self._n = 0

    # -- one document, on a worker thread: only the service call -------------------------------------------------------------------------
    def _read(self, doc: Document, retry: bool) -> tuple[FactBatch | None, float, str]:
        if self.should_stop():
            return None, 0.0, "stop requested"
        path = (self.root / doc.filename).resolve()
        credits = 0.0
        for attempt in (1, 2):
            try:
                job = self.client.extract(path, SCHEMA, system_prompt=SYSTEM_PROMPT)
            except LlamaExtractError as exc:
                return None, credits, str(exc)
            credits += job.credits
            if job.status != "COMPLETED" or not isinstance(job.result, dict):
                return None, credits, f"extraction job {job.status}: {job.error or 'no result'}"
            batch = FactBatch.model_validate(job.result)
            if batch.facts or not retry or attempt == 2:
                return batch, credits, ""
            self.on_progress(f"{doc.filename}: no facts returned although the text has values; asking once more")
        return None, credits, "unreachable"

    def run(self, documents: list[Document], coverage: dict | None = None) -> FactStats:
        stats = FactStats()
        chunks_by_doc: dict[str, list[DocumentChunk]] = {}
        for doc in documents:
            chunks_by_doc[doc.id] = [c for c in self.session.scalars(select(DocumentChunk).where(DocumentChunk.document_id == doc.id).order_by(DocumentChunk.chunk_index))
                                     if c.content_type not in EXCERPT_TYPES_SKIPPED]

        def needs_retry(doc: Document) -> bool:
            cov = (coverage or {}).get(doc.id)
            return bool(cov and cov.chunks_with_numbers and not cov.tables)

        if self.workers <= 1 or len(documents) <= 1:
            reads = ((doc, self._read(doc, needs_retry(doc))) for doc in documents)
            for doc, result in reads:
                self._handle(doc, chunks_by_doc[doc.id], result, stats, coverage)
        else:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                futures = [(doc, pool.submit(self._read, doc, needs_retry(doc))) for doc in documents]
                for doc, future in futures:
                    self._handle(doc, chunks_by_doc[doc.id], future.result(), stats, coverage)
        self.store.commit()
        return stats

    # -- on the calling thread: place, verify, store ---------------------------------------------------------------------------------------
    def _handle(self, doc: Document, chunks: list[DocumentChunk], result: tuple[FactBatch | None, float, str], stats: FactStats, coverage: dict | None) -> None:
        batch, credits, error = result
        stats.credits += credits
        if batch is None:
            stats.failed.append(f"{doc.filename}: {error}")
            self.on_progress(f"Fact extraction failed for {doc.filename}: {error[:140]}")
            return
        stats.documents += 1
        kept: list[Fact] = []
        claims: list[ExtractedClaim] = []
        for tf in batch.facts:
            kind = tf.kind
            if kind is None or not (tf.quote or tf.value not in (None, "")):
                continue
            placed = locate(tf.quote, tf.value, tf.subject, chunks)
            if placed is None:
                stats.rejected += 1
                stats.rejections.append(f"{doc.filename}: quote not found for {tf.subject!r} / {tf.attribute!r}: {(tf.quote or str(tf.value))[:80]!r}")
                continue
            chunk, quote = placed
            self._n += 1
            value = "" if tf.value is None else str(tf.value)
            fact = Fact(id=f"f{self._n:04d}", document_id=doc.id, chunk_id=chunk.id, type=kind, subject=tf.subject, attribute=tf.attribute, operator=tf.operator or "=",
                        value=value, unit=tf.unit or None, scope=tf.scope or None, when=tf.when or None, modality=tf.mode, aggregate_of=tf.aggregate_of or None,
                        value_type=tf.vtype, quote=quote)
            kept.append(fact)
            if kind in ("observation", "assertion") and value:
                vt = tf.vtype or guess_value_type(value, tf.unit)
                claims.append(ExtractedClaim(chunk_id=chunk.id, subject=tf.subject or None, metric=tf.attribute or None, value=value, unit=tf.unit or None,
                                             period=tf.when or None, conditions=tf.scope or None, claim_category=vt, claim_text=quote))
        created = self._claims._persist(doc, chunks, ClaimExtractionResult(claims=claims), self.summary) if claims else 0
        self.session.commit()
        self.facts.extend(kept)
        self.store.executemany(
            "INSERT OR REPLACE INTO _facts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [(f.id, f.document_id, doc.filename, f.chunk_id, f.type, f.subject, f.attribute, f.operator, f.value, f.unit, f.scope, f.when, parse_when(f.when),
              f.modality, f.aggregate_of, f.quote) for f in kept])
        stats.facts += len(kept)
        stats.claims += created
        stats.by_type.update(f.type for f in kept)
        if coverage and doc.id in coverage:
            coverage[doc.id].facts = len(kept)
            coverage[doc.id].claims = created
        self.on_progress(f"Extracted {len(kept)} facts ({created} as claims) from {doc.filename}: "
                         + ", ".join(f"{n} {t}" for t, n in Counter(f.type for f in kept).most_common()))
