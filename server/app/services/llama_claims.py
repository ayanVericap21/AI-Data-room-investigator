"""Claim extraction with LlamaExtract instead of the LLM-per-chunk extractor (chosen with a flag).

Each document is uploaded once and extracted against a generic claims schema. Every claim comes back with
the page and source text it was read from; here each claim is tied to one of the document's stored chunks
(by cited page and text) and then goes through the same checks as an LLM-extracted claim: the chunk must
exist and the value must appear in its text, or the claim is rejected and counted. Nothing unverified is stored.
"""
from __future__ import annotations

import logging
import re
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.llm.llama_extract import LlamaExtractClient, LlamaExtractError, LlamaJob
from app.models import Claim, Document, DocumentChunk
from app.schemas.claim import ClaimExtractionResult, ExtractedClaim
from app.services.claim_service import EXCERPT_TYPES_SKIPPED, ClaimService, ExtractionSummary
from app.services.doc_profiles import DocProfile
from app.services.missing_items import DocReference
from app.services.numbers import value_supported_by_text
from app.tools.submit import find_source_span

logger = logging.getLogger(__name__)
ProgressFn = Callable[[str, dict], None]

# The only kinds of value a claim can have; stored in the claim's existing claim_category field.
VALUE_TYPES = ("quantity", "date", "identifier", "name", "state", "statement")

# How each type is described to the model, and what code does with it (see value_rules.compare_values).
VALUE_TYPE_HELP = (
    "quantity: a numeric amount with or without a unit: a count, measure, currency amount, percentage, ratio, score, duration, range or bound "
    "(at least, up to, about), even when written inside brackets or quotation marks. Arithmetic on it is meaningful. Not a number used as a label (codes, "
    "phone numbers, version numbers). "
    "date: a point in time or a calendar or fiscal period (day, month, quarter, half, year, time of day), including a relative date anchored to an event. "
    "A year is a date when it says when something happened, not how many. A duration is a quantity. "
    "identifier: a string that labels exactly one thing and is compared character by character: a registration, certificate, account, order, version or phone number, "
    "a code or serial number, even if all digits. Arithmetic on it is meaningless and one changed character means a different thing. "
    "name: the proper name of a person, organisation, place, product or document title. A generic description (the company, the supplier) is not a name. "
    "state: a yes/no fact or a status from a small closed set (active, expired, pending), including quantified assertions (all, none, some, most). "
    "statement: any other assertion in words: a description, obligation, reason, list or policy.")

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "description": "Every stated value in the document that could be compared with another value, in this document or in another one.",
            "items": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string", "description": "What the value is about. For a table row, the row's identifying label (the entry in its ID or name column), never the table or sheet name. When the row is identified by several columns together (an entity and a date or period, a place and a date), the subject names all of them"},
                    "metric": {"type": "string", "description": "The quantity or attribute that has the value"},
                    "value": {"type": "string", "description": "The value exactly as written in the document"},
                    "unit": {"type": "string", "description": "Unit of the value, if stated"},
                    "period": {"type": "string", "description": "Period or date the value applies to, if stated"},
                    "conditions": {"type": "string", "description": "Scope or condition limiting the value, if stated"},
                    "claim_text": {"type": "string", "description": "The sentence or line that states it, copied from the document"},
                    "value_type": {"type": "string", "enum": list(VALUE_TYPES),
                                   "description": VALUE_TYPE_HELP},
                },
                "required": ["subject", "metric", "value", "claim_text", "value_type"],
            },
        },
        "references": {
            "type": "array",
            "description": "Every document, annex, schedule, exhibit, attachment, table, report, approval, certificate or record that the text says exists elsewhere, is attached or included, or is missing, not provided, not yet received, pending or outstanding. Not a section of this same document, and not a kind of document named in a general disclaimer.",
            "items": {
                "type": "object",
                "properties": {
                    "item": {"type": "string", "description": "The item as the document names it, for example the annex or schedule name"},
                    "quote": {"type": "string", "description": "The sentence that refers to it, copied from the document"},
                },
                "required": ["item", "quote"],
            },
        },
        "profile": {
            "type": "object",
            "description": "A short description of the whole document, so that its claims can be read in context.",
            "properties": {
                "main_entity": {"type": "string", "description": "The organisation, project, product, person or event the document is mainly about, by name"},
                "document_type": {"type": "string", "description": "What kind of document it is, for example permit, contract, financial model, lab report, presentation, register, minutes"},
                "purpose": {"type": "string", "description": "What the document is for, in one sentence"},
                "date": {"type": "string", "description": "The document's date or version, only if it states one"},
                "status": {"type": "string", "description": "Draft, final, signed or superseded, only if the document states it"},
                "scope": {"type": "string", "description": "What the document covers and what it does not, in one sentence"},
            },
        },
    },
}

SYSTEM_PROMPT = (
    "Extract every claim that states a value (a number, quantity, amount, date, duration, name, yes/no, status or condition) "
    "so it can be compared with other values, whether stated elsewhere in the same document or in other documents, including values in lists, tables and short fragments. If the same quantity "
    "is stated more than once, extract each mention. For a table, the subject is the row's identifying label (its ID or name), never the table or sheet name; when a row is identified by several columns together (an entity and a date or period, a place and a date), the subject names all of them, so that two rows differing in any identifying column never share a subject; the metric is the "
    "column heading. Copy values and sentences exactly as written; never calculate or infer, and never shorten or join passages with an ellipsis (...): copy the whole value. "
    "OUTPUT FORMAT (strict): every claim has non-empty string fields subject, metric, value and claim_text, and a value_type that is exactly one of "
    "quantity, date, identifier, name, state or statement. " + VALUE_TYPE_HELP + " No other value_type is allowed. Add unit, period and conditions "
    "only when the document states them; never return a claim without a value, and never invent other fields. "
    "Do not return a claim whose value is just the label that identifies its own row or item (the ID or name in the row's first column): that label is the subject, not a value. "
    "Also return references: each document, annex, schedule, exhibit, attachment, table, report, approval, certificate, record or other item that the text says exists "
    "elsewhere, is attached, is included, or is missing, not included, not provided, not available, available only on request, on file, provided separately, not yet received, pending, in process or outstanding, "
    "with item (its name as written) and quote (the sentence that refers to it, copied exactly). "
    "Do not list a section of the same document, and do not list a kind of document that a general disclaimer names (for example a statement that the document is not an audit report or a filing). "
    "Return an empty references list when there are none. "
    "Also return profile: a short description of the whole document (main_entity, document_type, purpose, date, status, scope), "
    "stating only what the document itself says; leave date and status empty when it gives none.")


class LlamaClaim(BaseModel):
    subject: str | None = None
    metric: str | None = None
    value: str | int | float | None = None
    unit: str | None = None
    period: str | None = None
    conditions: str | None = None
    claim_text: str = ""
    value_type: str | None = None

    @property
    def kind(self) -> str | None:
        """The value type when it is one of the allowed ones, else None (never a made-up label)."""
        kind = (self.value_type or "").strip().lower()
        return kind if kind in VALUE_TYPES else None


class LlamaReference(BaseModel):
    item: str = ""
    quote: str = ""


class LlamaProfile(BaseModel):
    """The extraction service returns null for a field it cannot fill, so every field may be missing."""
    main_entity: str | None = None
    document_type: str | None = None
    purpose: str | None = None
    date: str | None = None
    status: str | None = None
    scope: str | None = None

    def to_profile(self) -> DocProfile:
        return DocProfile(**{k: (v or "").strip() for k, v in self.model_dump().items()})


class LlamaClaims(BaseModel):
    claims: list[LlamaClaim] = []
    references: list[LlamaReference] = []
    profile: LlamaProfile | None = None  # what the document is about, in the same call (compared with the separate describing calls)


def _squash(text: str) -> str:
    return " ".join(text.split()).lower()


def _citation(metadata: dict[str, Any], index: int, field: str) -> tuple[int | None, str]:
    """(page, matching text) the service cites for field ``field`` of claim ``index``; (None, "") if none."""
    try:
        entry = metadata["field_metadata"]["document_metadata"]["claims"][index][field]["citation"][0]
    except (KeyError, IndexError, TypeError):
        return None, ""
    text = entry.get("matching_text") or ""
    return entry.get("page"), "" if re.fullmatch(r"claims\.\d+\.\w+", text) else text  # an internal path, not source text


def _is_own_label(claim: "LlamaClaim") -> bool:
    """The claim's value is the label of its own row or item (Tool_ID: MET-004 about MET-004): a label, not a value."""
    subject, value = _squash(claim.subject or ""), _squash(str(claim.value or ""))
    return bool(subject) and subject == value


def pick_chunk(chunks: list[DocumentChunk], value: Any, claim_text: str, page: int | None, cited_text: str,
               subject: str | None = None) -> DocumentChunk | None:
    """The chunk that states the value: on the cited page when the document has pages, containing the claim's subject (a table row's
    label) when several chunks hold the value, containing the cited or claim text when possible. None when no chunk contains the value."""
    holding = [c for c in chunks if value_supported_by_text(value, c.content)]
    if not holding:
        return None
    paged = [c for c in holding if page is not None and c.page_number == page]
    pool = paged or holding
    wanted_subject = _squash(subject or "")
    if wanted_subject and len(pool) > 1:
        pool = [c for c in pool if wanted_subject in _squash(c.content)] or pool
    for wanted in (_squash(claim_text), _squash(cited_text)):
        if wanted:
            for chunk in pool:
                if wanted in _squash(chunk.content):
                    return chunk
    return pool[0]


_ELLIPSIS = re.compile(r"\.{3,}|…")
MAX_EXPANDED_VALUE = 500  # characters: a value this long is a passage, not a value


def expand_ellipsis(value: Any, chunks: list[DocumentChunk]) -> tuple[DocumentChunk, str] | None:
    """A value the extractor shortened with an ellipsis ("NO sampling ... is permitted"): when every piece appears in one chunk, in order,
    return that chunk and the verbatim passage from the first piece to the last, so the stored value is real source text. None otherwise."""
    if not isinstance(value, str) or not _ELLIPSIS.search(value):
        return None
    pieces = [p.strip() for p in _ELLIPSIS.split(value) if len(p.strip()) >= 3]
    if len(pieces) < 2:
        return None
    for chunk in chunks:
        start, at = None, 0
        for piece in pieces:
            span = find_source_span(chunk.content[at:], piece)
            found = chunk.content.find(span[0], at) if span else -1
            if found < 0:
                start = None
                break
            start = found if start is None else start
            at = found + len(span[0])
        if start is not None and at - start <= MAX_EXPANDED_VALUE:
            return chunk, chunk.content[start:at]
    return None


def merge_jobs(jobs: list[LlamaJob]) -> LlamaJob:
    """The page-range jobs of one document as one job: claims and references in page order, each claim keeping its own citation (the pages the
    service cites are the document's own page numbers). If any range failed, that job stands for the document, which then fails as a whole."""
    if len(jobs) == 1:
        return jobs[0]
    failed = next((j for j in jobs if j.status != "COMPLETED" or not isinstance(j.result, dict)), None)
    if failed is not None:
        return failed
    claims: list = []
    references: list = []
    cited: list = []  # the citations of the claims, in the same order, so a claim's index finds its own
    for j in jobs:
        part = j.result.get("claims") or []
        claims += part
        references += j.result.get("references") or []
        own = (((j.metadata or {}).get("field_metadata") or {}).get("document_metadata") or {}).get("claims") or []
        cited += (list(own) + [{}] * len(part))[:len(part)]
    profile = next((j.result.get("profile") for j in jobs if j.result.get("profile")), None)  # the first range describes the document
    return LlamaJob(job_id=",".join(j.job_id for j in jobs), status="COMPLETED", result={"claims": claims, "references": references, "profile": profile},
                    metadata={"field_metadata": {"document_metadata": {"claims": cited}}}, credits=sum(j.credits for j in jobs),
                    pages_billed=sum(j.pages_billed for j in jobs), latency_ms=max(j.latency_ms for j in jobs))


class LlamaClaimExtractor:
    def __init__(self, session: Session, client: LlamaExtractClient, data_room_dir: Path,
                 on_progress: ProgressFn | None = None, should_stop: Callable[[], bool] | None = None, pages_per_job: int = 0):
        self.pages_per_job = pages_per_job  # a PDF longer than this is read in page ranges, one job each, started together (0 = one job per document)
        self.session = session
        self.client = client
        self.root = data_room_dir
        self.on_progress = on_progress or (lambda _m, _d: None)
        self.should_stop = should_stop or (lambda: False)
        self._claims = ClaimService(session)  # reused for the value check and storing
        self.references: list[DocReference] = []  # items the documents point to; checked for absence after the run's claims are judged
        self.profiles: dict[str, tuple[str, DocProfile]] = {}  # document id -> (file name, profile) as the extraction call described it
        self._pool: ThreadPoolExecutor | None = None
        self._jobs: dict[str, list[Future]] = {}  # document id -> its extraction jobs (one, or one per page range), started ahead by ``prefetch``

    def _page_ranges(self, doc: Document) -> list[str]:
        """The page ranges a PDF is read in ("1-4", "5-8", ...), or none when it is short enough for one job (or is not a PDF)."""
        if not self.pages_per_job or doc.file_type != "pdf":
            return []
        last = self.session.scalar(select(func.max(DocumentChunk.page_number)).where(DocumentChunk.document_id == doc.id)) or 0
        if last <= self.pages_per_job:
            return []
        return [f"{start}-{min(start + self.pages_per_job - 1, last)}" for start in range(1, last + 1, self.pages_per_job)]

    def prefetch(self, docs: list[Document], workers: int) -> None:
        """Start the extraction jobs of all the documents at once, a long PDF as several page-range jobs. The service does the reading, so the
        waiting overlaps; the results are still stored one document at a time by ``extract_document``, which finds its jobs here and waits for them."""
        if workers <= 1:
            return
        self._pool = ThreadPoolExecutor(max_workers=workers)
        for doc in docs:
            if self.should_stop():
                break
            path = (self.root / doc.filename).resolve()
            ranges = self._page_ranges(doc)
            if not ranges:
                self._jobs[doc.id] = [self._pool.submit(self.client.extract, path, SCHEMA, system_prompt=SYSTEM_PROMPT)]
                continue
            try:
                file_id = self.client.upload(path)  # uploaded once, read by every range
            except Exception as exc:  # reported when the document is stored
                failed: Future = Future()
                failed.set_exception(exc)
                self._jobs[doc.id] = [failed]
                continue
            self._jobs[doc.id] = [self._pool.submit(self.client.extract, path, SCHEMA, file_id=file_id, system_prompt=SYSTEM_PROMPT,
                                                    extra_configuration={"target_pages": pages}) for pages in ranges]

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool, self._jobs = None, {}

    def _job(self, doc: Document, path: Path) -> LlamaJob:
        futures = self._jobs.pop(doc.id, None)
        if futures is None:
            return self.client.extract(path, SCHEMA, system_prompt=SYSTEM_PROMPT)
        return merge_jobs([f.result() for f in futures])

    def extract_document(self, doc: Document, summary: ExtractionSummary, force: bool = False) -> None:
        if self.should_stop():
            summary.stopped, summary.stop_reason = True, "stop requested"
            summary.documents_failed += 1
            return
        path = (self.root / doc.filename).resolve()
        chunks = [c for c in self.session.scalars(
            select(DocumentChunk).where(DocumentChunk.document_id == doc.id).order_by(DocumentChunk.chunk_index))
            if c.content_type not in EXCERPT_TYPES_SKIPPED]
        try:
            summary.external_calls += 1
            job = self._job(doc, path)
        except LlamaExtractError as exc:
            self._fail(doc, summary, str(exc))
            if exc.code == 402:  # out of credits: the next documents would fail the same way
                summary.stopped, summary.stop_reason = True, "LlamaExtract is out of credits"
            return
        summary.external_credits += job.credits
        if job.status != "COMPLETED" or not isinstance(job.result, dict):
            self._fail(doc, summary, f"extraction job {job.status}: {job.error or 'no result'}")
            return

        parsed = LlamaClaims.model_validate(job.result)
        profile = parsed.profile.to_profile() if parsed.profile is not None else None
        if profile is not None and not profile.empty:
            self.profiles[doc.id] = (doc.filename, profile)
        extracted: list[ExtractedClaim] = []
        for index, claim in enumerate(parsed.claims):
            if claim.value in (None, ""):
                continue
            if _is_own_label(claim):
                summary.claims_skipped_as_labels += 1
                continue
            page, cited = _citation(job.metadata, index, "value")
            if page is None:
                page, cited = _citation(job.metadata, index, "claim_text")
            value = claim.value
            chunk = pick_chunk(chunks, value, claim.claim_text, page, cited, claim.subject)
            if chunk is None and (expanded := expand_ellipsis(value, chunks)) is not None:
                chunk, value = expanded  # the extractor shortened the value with an ellipsis; keep the verbatim passage instead
                summary.claims_ellipsis_expanded += 1
            if chunk is None:
                self._claims._reject(summary, f"value {claim.value!r} not found in any chunk of {doc.filename}")
                continue
            extracted.append(ExtractedClaim(
                chunk_id=chunk.id, subject=claim.subject, metric=claim.metric, value=value, unit=claim.unit,
                period=claim.period, conditions=claim.conditions, claim_category=claim.kind,
                claim_text=claim.claim_text or f"{claim.metric}: {claim.value}"))
        created = self._claims._persist(doc, chunks, ClaimExtractionResult(claims=extracted), summary)
        self.session.commit()
        self._keep_references(parsed, chunks, doc)
        with_numbers = sum(bool(re.search(r"\d", c.content or "")) for c in chunks)
        if created == 0 and with_numbers:
            # a document with numbers in its text and no claim at all is an extraction failure, not an empty document: say so loudly
            summary.documents_without_claims.append(doc.filename)
            summary.notes.append(f"{doc.filename}: LlamaExtract returned no claims, but {with_numbers} of its {len(chunks)} chunks contain numbers")
            self.on_progress(f"WARNING: {doc.filename} produced no claims although its text contains values; it is not treated as fully read", {"document_id": doc.id})
        summary.documents_processed += 1
        self.on_progress(f"Extracted {created} claims from {doc.filename} (LlamaExtract, {job.credits:g} credits)",
                         {"document_id": doc.id, "claims": created})

    def _keep_references(self, parsed: LlamaClaims, chunks: list[DocumentChunk], doc: Document) -> None:
        """Remember each reference with the chunk whose text holds its sentence (the exact source text); one that cannot be placed is dropped."""
        for ref in parsed.references:
            item, quote = ref.item.strip(), ref.quote.strip()
            if not item or len(quote) < 8:
                continue
            for chunk in chunks:
                span = find_source_span(chunk.content, quote)
                if span:
                    self.references.append(DocReference(doc.id, chunk.id, item, span[0]))
                    break

    def _fail(self, doc: Document, summary: ExtractionSummary, reason: str) -> None:
        self.session.rollback()
        self.session.execute(delete(Claim).where(Claim.document_id == doc.id))
        self.session.commit()
        summary.documents_failed += 1
        summary.notes.append(f"{doc.filename}: {reason}")
        logger.warning("LlamaExtract failed for a document", extra={"document": doc.filename, "reason": reason})
        self.on_progress(f"LlamaExtract failed for {doc.filename}: {reason[:120]}", {"document_id": doc.id})
