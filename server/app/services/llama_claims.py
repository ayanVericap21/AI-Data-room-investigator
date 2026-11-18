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
from app.services.claim_service import EXCERPT_TYPES_SKIPPED, ClaimService, ExtractionSummary, delete_claims
from app.services.claim_tags import make_tags
from app.services.doc_profiles import DocProfile
from app.services.extraction_cache import ExtractionCache
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
                    "subject": {"type": "string", "description": "The specific thing the value belongs to (the item, room, party, invoice, hall, document), not the project or event the whole document is about unless the value is a property of that event itself. For a table row, the row's identifying label (the entry in its ID or name column), never the table or sheet name. When the row is identified by several columns together (an entity and a date or period, a place and a date), the subject names all of them"},
                    "metric": {"type": "string", "description": "A short noun phrase naming the attribute that has the value, worded the same way every time the same attribute appears; it names what is compared (catering spend against budget)"},
                    "value": {"type": "string", "description": "The value exactly as written in the document. For an assertion about how many (all, none, some, not all) the quantifier as written. For a list of names or items in one sentence, the whole list as written"},
                    "unit": {"type": "string", "description": "Unit of the value, if stated"},
                    "period": {"type": "string", "description": "Period or date the value applies to, if stated. A relative period (day 2, week 3, within 30 days of handover, after completion) is written here as the document words it, together with its anchor, and is never a claim of its own"},
                    "conditions": {"type": "string", "description": "Scope or condition limiting the value, if stated"},
                    "claim_text": {"type": "string", "description": "The sentence or line that states it, copied from the document"},
                    "value_type": {"type": "string", "enum": list(VALUE_TYPES),
                                   "description": VALUE_TYPE_HELP},
                    "about": {"type": "string", "enum": ["content", "document"],
                              "description": "document when the claim is about the document itself (its title, type, number, version, revision, its own date, who wrote it); content when it is about what the document covers"},
                    "basis": {"type": "string", "enum": ["actual", "revised", "planned", "contracted", "forecast", "estimated", "limit"],
                              "description": "How the sentence words the value: actual (achieved, occurred, final), revised, planned, contracted, forecast, estimated, or limit (maximum, minimum, at least). Leave out when the sentence says nothing"},
                    "precision": {"type": "string", "enum": ["exact", "approximate", "vague"],
                                  "description": "approximate for about, around, over, at least; vague for words with no number (broadly in line, significant); otherwise exact"},
                    "quantifier": {"type": "string", "enum": ["all", "none", "some", "most", "not_all"],
                                   "description": "For an assertion about how many: all, none, some, most or not_all. The metric names what it covers (workshops recorded)"},
                    "anchor": {"type": "string", "description": "Required whenever period or value is relative to an event (day 2, week 3, within 30 days of handover, after completion): the event or date it is counted from, as the document names it, or the event the document is about when it says only day 2 or week 3"},
                    "complete": {"type": "boolean", "description": "For a list claim: true when the text says the list is complete (in full, all, only) or states how many there are"},
                    "is_rule": {"type": "boolean", "description": "true when the claim is a rule: a claim that states what is permitted, prohibited, required, limited, scheduled or suspended for a scope (a place, a kind of item, some people, a period), or the status of a scope during a period (closed, suspended, unavailable, expired). A limit or a quota is a rule. So is a universal or negative statement about a scope (none, no, never, all, always: \"no subcontractors are engaged\", \"all invoices were paid\"). Leave out (or false) for a claim that only reports something that was done, found, used, measured or stated as a fact"},
                },
                "required": ["subject", "metric", "value", "claim_text", "value_type"],
            },
        },
        "references": {
            "type": "array",
            "description": "Every document, annex, schedule, exhibit, attachment, table, report, approval, certificate or record that the text says exists elsewhere, is attached or included, or is missing, not provided, not yet received, pending or outstanding. Not a section of this same document, not a kind of document named in a general disclaimer, and nothing from a sentence that only says what the document is not or does not replace.",
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
                "main_entity": {"type": "string", "description": "The organisation, project, product, person or event the document is mainly about, by name (what it is about, not who wrote it)"},
                "issuer": {"type": "string", "description": "Who wrote or issued the document, by name, only if it states one"},
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
    "stating only what the document itself says; leave date and status empty when it gives none. "
    "SUBJECT AND METRIC: the subject is the specific thing the value belongs to (the item, room, party, invoice, hall, document), not the project or event the whole document is about, "
    "unless the value is a property of that event itself. The metric is a short noun phrase naming the attribute, worded the same way every time the same attribute appears, and it names "
    "what is compared (catering spend against budget, not just catering spend). A year that is part of a name (Forum 2025) is not a date: never return it as one. "
    "DOCUMENT DETAILS: a claim about the document itself (title, type, number, version, revision, its own date, who wrote it) has about = document and the document as its subject; every other claim has about = content. "
    "BASIS: set basis when the sentence words the value as actual (achieved, occurred, final), revised, planned, contracted, forecast, estimated or a limit (maximum, minimum, at least); leave it out otherwise. "
    "PRECISION: approximate for about, around, over, at least; vague for words with no number (broadly in line, significant); otherwise exact. "
    "QUANTIFIER: for an assertion about how many (all, none, some, most, not all) the value is the quantifier as written, quantifier repeats it (all, none, some, most or not_all), and the metric names what it covers (workshops recorded). "
    "FACTS INSIDE A CLAUSE: when a sentence states a fact inside a clause or qualifier (an external caterer, a subcontractor, an exception), return that fact as its own claim as well (external caterer used: yes). "
    "LISTS: a list of names or items in one sentence is ONE claim: the value is the whole list as written and the metric names what the items are; set complete to true when the text says the list is complete "
    "(in full, all, only) or states how many there are; a count stated in the sentence is a separate claim. "
    "RELATIVE DATES: a period or date counted from an event (day 2, week 3, within 30 days of handover, after completion) goes in period of the claim it belongs to, never as a claim of its own, and anchor always names the event it is counted from: the event the document names, or the event the document is about when it says only day 2 or week 3. "
    "PAGE NUMBERS: a page number or page count printed on a page (Page 4 of 6, 4 of 6) is layout, not a fact: return no claim for it. The document's title, its version or revision and its dates are still claims (DOCUMENT DETAILS). "
    "RULES: set is_rule to true on a claim that is a rule: a claim that states what is permitted, prohibited, required, limited, scheduled or suspended for a scope (a place, a kind of item, some people, a period), or the status of a scope during a period (closed, suspended, unavailable, expired). A limit or a quota is a rule. So is a universal or negative statement about a scope (none, no, never, all, always: \"no subcontractors are engaged\", \"all invoices were paid\"). A claim that only reports something that was done, found, used, measured or stated as a fact is not a rule: leave is_rule out. "
    "DISCLAIMERS: ignore a sentence that only says what the document is not or does not replace: return no claim and no reference for it.")


class LlamaClaim(BaseModel):
    subject: str | None = None
    metric: str | None = None
    value: str | int | float | None = None
    unit: str | None = None
    period: str | None = None
    conditions: str | None = None
    claim_text: str = ""
    value_type: str | None = None
    about: str | None = None
    basis: str | None = None
    precision: str | None = None
    quantifier: str | None = None
    anchor: str | None = None
    complete: bool | None = None
    is_rule: bool | None = None

    @property
    def tags(self) -> list[str]:
        return make_tags(self.about, self.basis, self.precision, self.quantifier, self.anchor, self.complete, self.is_rule)

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
    issuer: str | None = None  # who wrote it: kept in the purpose, so the main entity is what the document is about

    def to_profile(self) -> DocProfile:
        fields = {k: (v or "").strip() for k, v in self.model_dump().items() if k != "issuer"}
        if (self.issuer or "").strip():
            fields["purpose"] = f"{fields['purpose']} Issued by {self.issuer.strip()}.".strip()
        return DocProfile(**fields)


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


# a sentence that only says what the document is not, or does not replace: it states no value and names nothing the room should contain
DISCLAIMER = re.compile(r"\b(?:does|do|did)\s+not\s+(?:replace|supersede|constitute|amount\s+to|purport)\b|\b(?:is|are)\s+not\s+(?:an?|the)\s+(?:audit|audited|substitute|offer|filing)\b|"
                        r"\bnot\s+(?:intended|meant)\s+to\s+(?:replace|be)\b|\bno\s+substitute\s+for\b", re.IGNORECASE)


def _year_in_name(claim: "LlamaClaim") -> bool:
    """A "date" that is only the year written in the subject's name (event date 2025 about "Lakeside Health Forum 2025"): part of a name, not a date."""
    value = str(claim.value or "").strip()
    return claim.kind == "date" and bool(re.fullmatch(r"\d{4}", value)) and value in (claim.subject or "")


PAGE_WORDS = frozenset({"page", "pages", "number", "numbers", "count", "total", "document", "current"})  # what the label of a page-number claim is made of
PAGE_VALUE = re.compile(r"^(?:page\s*)?\d+(?:\s*(?:of|/)\s*\d+)?(?:\s*pages?)?$", re.IGNORECASE)  # "Page 6 of 6", "17 of 18", "22", "4 Page"


def _is_page_number(claim: "LlamaClaim") -> bool:
    """A page number or page count printed on the page ("page number: Page 6 of 6"), which is layout, not a fact. Its label is only page words
    and its value is only a page position. A bare number inside a full sentence ("The report has 24 pages.") is kept: that states something."""
    words = set(re.findall(r"[a-z]+", (claim.metric or "").lower()))
    if not words & {"page", "pages"} or not words <= PAGE_WORDS:
        return False
    value = " ".join(str(claim.value if claim.value is not None else "").split())
    if not PAGE_VALUE.match(value):
        return False
    text = (claim.claim_text or "").strip()
    return not (value.isdigit() and len(text.split()) >= 8 and "|" not in text)


_PAGE_FURNITURE = re.compile(r"\bpage\s+\d+\s+of\s+\d+\b", re.IGNORECASE)
_TITLE_WORDS = {"title", "heading", "header", "caption"}


def _is_own_label(claim: "LlamaClaim") -> bool:
    """The claim's value is the label of its own row or item (Tool_ID: MET-004 about MET-004): a label, not a value. A title or heading, a running title or
    footer ("... | Page 5 of 8") and a sentence that adds nothing to the value are labels. A sentence that says more than the value ("Taxpayer: Riverside Heritage
    Development LLC", "Owner of record: Ashby Mill Owner LLC") states a fact about the entity it names, so it is kept."""
    subject, value = _squash(claim.subject or ""), _squash(str(claim.value or ""))
    if not subject or subject != value:
        return False
    if set(re.findall(r"[a-z]+", (claim.metric or "").lower())) & _TITLE_WORDS:
        return True
    text = claim.claim_text or ""
    if _PAGE_FURNITURE.search(text):
        return True
    return not (set(re.findall(r"\w+", text.lower())) - set(re.findall(r"\w+", value)))


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
                    pages_billed=sum(j.pages_billed for j in jobs), latency_ms=max(j.latency_ms for j in jobs),
                    cached=all(j.cached for j in jobs), credits_saved=sum(j.credits_saved for j in jobs))


class LlamaClaimExtractor:
    def __init__(self, session: Session, client: LlamaExtractClient, data_room_dir: Path,
                 on_progress: ProgressFn | None = None, should_stop: Callable[[], bool] | None = None, pages_per_job: int = 0,
                 cache: ExtractionCache | None = None):
        self.pages_per_job = pages_per_job  # a PDF longer than this is read in page ranges, one job each, started together (0 = one job per document)
        self.cache = cache  # the stored responses of earlier jobs; None = every document is read by the service
        self._warned_cache = False
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
                self._jobs[doc.id] = [self._pool.submit(self._run, doc, path)]
                continue
            stored = {pages: self.cache.get(doc.file_hash, pages) for pages in ranges} if self.cache is not None else {}
            file_id = None
            if any(stored.get(pages) is None for pages in ranges):  # the file is uploaded only when some range has to be read
                try:
                    file_id = self.client.upload(path)  # uploaded once, read by every range
                except Exception as exc:  # reported when the document is stored
                    failed: Future = Future()
                    failed.set_exception(exc)
                    self._jobs[doc.id] = [failed]
                    continue
            futures: list[Future] = []
            for pages in ranges:
                if stored.get(pages) is not None:
                    done: Future = Future()
                    done.set_result(stored[pages])
                    futures.append(done)
                else:
                    futures.append(self._pool.submit(self._run, doc, path, file_id, pages))
            self._jobs[doc.id] = futures

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool, self._jobs = None, {}

    def _run(self, doc: Document, path: Path, file_id: str | None = None, pages: str | None = None) -> LlamaJob:
        """One extraction job: the stored response when this file, schema, prompt, tier and page range were read before, else the service
        (and the response is kept for the next run)."""
        if self.cache is not None:
            hit = self.cache.get(doc.file_hash, pages or "")
            if hit is not None:
                return hit
        kwargs: dict[str, Any] = {"system_prompt": SYSTEM_PROMPT}
        if file_id is not None:
            kwargs["file_id"] = file_id
        if pages:
            kwargs["extra_configuration"] = {"target_pages": pages}
        job = self.client.extract(path, SCHEMA, should_stop=self.should_stop, **kwargs)  # a stopped run cancels the job instead of leaving it to run and bill
        if self.cache is not None:
            self.cache.put(doc.file_hash, pages or "", job)
        return job

    def _job(self, doc: Document, path: Path) -> LlamaJob:
        futures = self._jobs.pop(doc.id, None)
        if futures is None:
            return self._run(doc, path)
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
            job = self._job(doc, path)
        except LlamaExtractError as exc:
            summary.external_calls += 1
            self._fail(doc, summary, str(exc))
            if exc.code == 402:  # out of credits: the next documents would fail the same way
                summary.stopped, summary.stop_reason = True, "LlamaExtract is out of credits"
            return
        summary.extraction_credits_saved += job.credits_saved  # the jobs (some or all) that came from the extraction cache
        if job.cached:  # every job of this document came from the extraction cache: nothing was uploaded and nothing was spent
            summary.documents_from_cache += 1
        else:
            summary.external_calls += 1
        if self.cache is not None and self.cache.error and not self._warned_cache:
            self._warned_cache = True
            self.on_progress(f"WARNING: the extraction cache is not working, documents are read by LlamaExtract and not stored ({self.cache.error})", {})
        summary.external_credits += job.credits
        if job.status != "COMPLETED" or not isinstance(job.result, dict):
            self._fail(doc, summary, f"extraction job {job.status}: {job.error or 'no result'}")
            return

        parsed = LlamaClaims.model_validate(job.result)
        profile = parsed.profile.to_profile() if parsed.profile is not None else None
        if profile is not None and not profile.empty:
            self.profiles[doc.id] = (doc.filename, profile)
        extracted: list[ExtractedClaim] = []
        seen: dict[tuple, ExtractedClaim] = {}  # a claim the extractor returned twice for this document (a sentence repeated on several pages): kept once
        for index, claim in enumerate(parsed.claims):
            if claim.value in (None, ""):
                continue
            if _is_own_label(claim):
                summary.claims_skipped_as_labels += 1
                continue
            if _is_page_number(claim):
                summary.claims_skipped_page_numbers += 1
                continue
            if _year_in_name(claim) or DISCLAIMER.search(claim.claim_text or ""):
                summary.claims_skipped_by_rule += 1
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
            kept = ExtractedClaim(
                chunk_id=chunk.id, subject=claim.subject, metric=claim.metric, value=value, unit=claim.unit,
                period=claim.period, conditions=claim.conditions, claim_category=claim.kind, assumptions=claim.tags,
                claim_text=claim.claim_text or f"{claim.metric}: {claim.value}")
            key = tuple(_squash(str(part or "")) for part in (kept.subject, kept.value, kept.unit, kept.period, kept.conditions, kept.claim_text))  # the metric is worded differently each time the sentence is read
            if key in seen:
                summary.claims_skipped_duplicates += 1
                first = seen[key]
                first.assumptions = list(dict.fromkeys([*(first.assumptions or []), *(kept.assumptions or [])]))  # a flag the repeat carries (a rule) is not lost
                continue
            seen[key] = kept
            extracted.append(kept)
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
        source = f"from the extraction cache, {job.credits_saved:g} credits not spent" if job.cached else f"LlamaExtract, {job.credits:g} credits"
        self.on_progress(f"Extracted {created} claims from {doc.filename} ({source})",
                         {"document_id": doc.id, "claims": created})

    def _keep_references(self, parsed: LlamaClaims, chunks: list[DocumentChunk], doc: Document) -> None:
        """Remember each reference with the chunk whose text holds its sentence (the exact source text); one that cannot be placed is dropped."""
        for ref in parsed.references:
            item, quote = ref.item.strip(), ref.quote.strip()
            if not item or len(quote) < 8 or DISCLAIMER.search(quote):
                continue
            for chunk in chunks:
                span = find_source_span(chunk.content, quote)
                if span:
                    self.references.append(DocReference(doc.id, chunk.id, item, span[0]))
                    break

    def _fail(self, doc: Document, summary: ExtractionSummary, reason: str) -> None:
        self.session.rollback()
        delete_claims(self.session, doc.id)
        self.session.commit()
        summary.documents_failed += 1
        summary.notes.append(f"{doc.filename}: {reason}")
        logger.warning("LlamaExtract failed for a document", extra={"document": doc.filename, "reason": reason})
        self.on_progress(f"LlamaExtract failed for {doc.filename}: {reason[:120]}", {"document_id": doc.id})
