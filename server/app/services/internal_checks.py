"""Checks inside one document: arithmetic and part-of-whole relations between its own claims, and relative dates with no anchor.

Grouping compares claims across documents. Some discrepancies are inside one document: an average times a number of periods that is not the total beside
it, a count larger than the capacity it is a part of, a "within 30 days of handover" in a room that never says when the handover was. Patterns in code
(``internal_patterns``) find the relations one sentence states ("1,840 out of 1,600", "average x periods = total", "rose 2% from 18% to 20%"). For the rest,
the model reads ONE document at a time and proposes relations between its claims ("across March and April" is two periods); code does the arithmetic and
decides whether each relation holds, and only a relation that fails becomes a finding. Quotes are taken from the document and verified like every other.
"""
from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Callable, Literal

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.llm.base import LLMClient
from app.llm.errors import LLMError
from app.models import Document, DocumentChunk
from app.models.base import ProcessingStatus
from app.services.claim_matching import normalize_unit
from app.services.claim_grouping import _line, load_claim_rows
from app.services.internal_patterns import find_relations
from app.services.severity import contradiction_severity, missing_severity
from app.services.value_rules import Num, parse_date, parse_numeric, unit_info
from app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)
CURRENCIES = frozenset({"gbp", "eur", "usd", "inr", "aud", "cad", "chf", "jpy", "cny", "sgd", "aed", "zar", "nzd", "sek", "nok", "dkk"})
TOLERANCE = 0.005  # a stated total may differ from the computed one by half a percent (rounding)

SYSTEM = """You find arithmetic and part-of-whole relations between claims of ONE document, which the document's own figures should satisfy.

Relations to look for:
- A total equals the sum of its parts, or an average (a rate per month, per site) times the number of periods or items the document states in words ("across March and April" is 2; "for 6 months" is 6).
- A count or amount of something that is part of a larger quantity cannot exceed it: installed out of the positions available, spent out of a budget, a share out of the whole.
- A before and an after figure that the document says changed by a stated amount.

Rules for your answer:
1. Use ONLY the claims shown. Refer to each claim by its ref (c1, c2, ...) exactly as shown. A relation uses claims of ONE document only.
2. Do NOT compute anything: give the relation, and code checks it. "terms" are the claims on the left side, each with "times" (a count of periods or items from the words, 1 when none); "against" is the claim the sum of the terms is compared with; "relation" is "equals" (the sum should equal it), "at_most" (the sum cannot exceed it) or "at_least".
3. Give only relations the document itself implies. Do not invent relations between unrelated figures, and do not repeat one relation.
4. label is a few words naming the relation.

OUTPUT FORMAT (strict). Return exactly one JSON object and nothing else: {"checks": [ {"label": "...", "terms": [{"ref": "c3", "times": 2}], "relation": "equals", "against": "c4"} ]}. If there is none, return {"checks": []}. Do not add other fields, comments or markdown."""


class Term(BaseModel):
    ref: str
    times: float = 1.0


class Relation(BaseModel):
    label: str = ""
    terms: list[Term] = Field(default_factory=list)
    relation: Literal["equals", "at_most", "at_least"] = "equals"
    against: str = ""


class Relations(BaseModel):
    checks: list[Relation] = Field(default_factory=list)


@dataclass
class InternalStats:
    claims: int = 0
    calls: int = 0
    failed: int = 0
    proposed: int = 0  # relations the model proposed
    checked: int = 0  # relations code could check (every value a plain number)
    unlike: int = 0  # relations that mix a rate, price or percentage with an amount: not checked
    held: int = 0
    broken: int = 0
    patterns: int = 0  # relations found by patterns in one sentence (part of a whole, average and total, a stated change) that the figures do not satisfy
    anchors_checked: int = 0  # relative dates ("30 days of handover")
    anchors_missing: int = 0
    saved: int = 0
    rejected: int = 0
    log: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _number(value: object) -> Num | None:
    return parse_numeric(value, ignore=CURRENCIES)


_SCALE_WORDS = {"thousand": Decimal(1_000), "million": Decimal(1_000_000), "billion": Decimal(1_000_000_000)}
_SCALE_IN_TEXT = re.compile(r"\b(thousand|million|billion)s?\b|(?:\$|usd|in)\s*'?000s?\b", re.IGNORECASE)


def _scale(view) -> Decimal:
    """The factor that takes the claim's number to whole units. Read from its unit ('$ in thousands'), and when the number has no currency sign or
    scale word of its own, from the claim's sentence ('... (in thousands)'). ``unit_info`` reads the scale word of the unit; this also reads "000s" and a scale stated only in the sentence."""
    factor = unit_info(view.unit, normalize_unit).factor
    if factor != 1:
        return factor
    for text, needs_bare in ((view.unit or "", False), (view.claim_text or "", True)):
        if needs_bare and re.search(r"[$€£]|[a-z]", str(view.value or ""), re.IGNORECASE):
            continue  # the value carries its own sign or scale word: the sentence does not rescale it
        m = _SCALE_IN_TEXT.search(text)
        if m:
            return _SCALE_WORDS[m.group(1).lower()] if m.group(1) else Decimal(1000)
    return Decimal(1)


def _is_fraction(view, number: Num) -> bool:
    """A percentage, a rate or a price per unit (0.88, 0.85, 20%): a number below one that is not zero. It is a multiplier, never an amount to add to one.
    Read with the scale of the claim's own unit only: a sentence that says "in thousands" somewhere does not turn a rate into an amount."""
    if "%" in str(view.value) or (view.unit or "").strip() == "%":
        return True
    factor = unit_info(view.unit, normalize_unit).factor
    low, high = number.low * factor, number.high * factor
    return low != 0 and abs(low) < 1 and abs(high) < 1


def clause_quote(content: str, value: object, fallback: str = "") -> str | None:
    """The clause of ``content`` that holds the value: from the delimiter before it to the one after (a full stop, comma or semicolon), exactly as written."""
    text = str(value or "").strip()
    at = content.find(text) if text else -1
    if at < 0:
        match = re.search(r"\d[\d,.]*", text)
        at = content.find(match.group(0)) if match else -1
        text = match.group(0) if match else text
    if at < 0:
        return fallback or None
    left = max((content.rfind(d, 0, at) for d in (". ", "; ", ", ", "\n")), default=-1)
    ends = [i for i in (content.find(d, at + len(text)) for d in (". ", "; ", ", ", "\n")) if i >= 0]
    right = min(ends) if ends else len(content)
    quote = content[left + 1 if left >= 0 else 0:right].strip(" ,;.\n")
    if left >= 0 and content[left] == ".":
        quote = content[left + 2:right].strip(" ,;.\n")
    return quote if len(quote) >= 8 else (fallback or None)


class InternalChecker:
    def __init__(self, session: Session, llm: LLMClient, registry: ToolRegistry, max_chars_per_call: int = 300_000,
                 should_stop: Callable[[], bool] | None = None, on_progress: Callable[[str], None] | None = None, workers: int = 1):
        self.session, self.llm, self.registry, self.max_chars = session, llm, registry, max_chars_per_call
        self.should_stop = should_stop or (lambda: False)
        self.on_progress = on_progress or (lambda _m: None)
        self.workers = max(1, workers)

    # -- the run ---------------------------------------------------------------------------------------------------------------------
    def run(self) -> InternalStats:
        rows = load_claim_rows(self.session)
        rows.sort(key=lambda r: (r[1], r[3]))
        stats = InternalStats(claims=len(rows))
        chunks = {c.id: c for c in self.session.scalars(select(DocumentChunk).where(DocumentChunk.id.in_({r[0].chunk_id for r in rows})))} if rows else {}
        self._seen: set[frozenset[str]] = set()
        self._pattern_figures: set[frozenset[tuple[str, str]]] = set()  # (chunk, figure) sets of the relations the patterns found: the model's version of one is not saved twice
        self._patterns(stats)
        for batch, answer in self._ask(self._batches(rows)):
            if answer is None:
                break
            stats.calls += 1
            if isinstance(answer, LLMError):
                stats.failed += 1
                stats.errors.append(str(answer))
                continue
            refs = {f"c{i}": r for i, r in enumerate(batch, start=1)}
            for check in answer.checks:
                stats.proposed += 1
                self._verify(check, refs, chunks, stats)
        self._anchors(rows, chunks, stats)
        return stats

    def _batches(self, rows):
        """One document to a call (a relation uses the claims of one document only): a document with fewer than two claims has no relation, and one too long
        for the character budget is read in parts."""
        by_doc: dict[str, list] = {}
        for r in rows:
            by_doc.setdefault(r[0].document_id, []).append(r)
        for doc_rows in by_doc.values():
            current: list = []
            size = 0
            for r in doc_rows:
                n = len(_line("c000", *r[:3])) + 1
                if current and size + n > self.max_chars:
                    yield current
                    current, size = [], 0
                current.append(r)
                size += n
            if len(current) >= 2:
                yield current

    def _patterns(self, stats: InternalStats) -> None:
        """Relations one sentence states, checked by patterns in code, over every chunk of every document that is still in the room."""
        stmt = select(DocumentChunk, Document.filename).join(Document, Document.id == DocumentChunk.document_id).where(Document.processing_status != ProcessingStatus.missing.value)
        for chunk, filename in list(self.session.execute(stmt)):
            for found in find_relations(chunk.content):
                if any(q not in chunk.content for q in found.quotes):
                    continue
                key = frozenset((chunk.id, n) for n in found.numbers)
                if key in self._pattern_figures:
                    continue
                self._pattern_figures.add(key)
                role = "contradicting" if found.classification == "contradiction" else "insufficient"
                result = self.registry.execute("submit_finding", {
                    "title": f"{found.title} ({filename})"[:500], "description": f"{filename}: {found.description}", "classification": found.classification,
                    "severity": found.severity, "confidence": 0.9, "evidence_status": "adequate",
                    "evidence": [{"chunk_id": chunk.id, "quoted_text": q, "role": role, "relevance_reason": found.title,
                                  "support_assessment": f"{filename} states: {q}"} for q in found.quotes]})
                if result.ok:
                    stats.patterns += 1
                    stats.saved += 1
                    stats.log.append(f"{filename}: {found.title}")
                else:
                    stats.rejected += 1
                    stats.log.append(f"{found.title}: {result.error.code}: {result.error.message[:140]}")

    def _ask(self, batches):
        batches = list(batches)

        def call(batch):
            if self.should_stop():
                return None
            lines, last = [], None
            for i, (v, f, w, _o) in enumerate(batch, start=1):
                if f != last:
                    lines.append(f"\n### Document: {f}")
                    last = f
                lines.append(_line(f"c{i}", v, f, w))
            try:
                return self.llm.generate_structured_output("Claims, document by document:\n" + "\n".join(lines), Relations, system=SYSTEM)
            except LLMError as exc:
                return exc

        if self.workers <= 1 or len(batches) <= 1:
            for batch in batches:
                answer = call(batch)
                yield batch, answer
                if answer is None:
                    return
            return
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = [(batch, pool.submit(call, batch)) for batch in batches]
            for batch, future in futures:
                yield batch, future.result()

    # -- arithmetic -------------------------------------------------------------------------------------------------------------------
    def _verify(self, check: Relation, refs: dict, chunks: dict, stats: InternalStats) -> None:
        terms = [(refs.get(t.ref.strip()), t.times) for t in check.terms]
        against = refs.get(check.against.strip())
        if not terms or against is None or any(r is None for r, _t in terms):
            return
        members = [r for r, _t in terms] + [against]
        if len({r[0].document_id for r in members}) != 1 or len({r[0].id for r in members}) != len(members):
            return  # inside one document only, and a claim is not compared with itself
        nums = [(_number(r[0].value), t) for r, t in terms]
        target = _number(against[0].value)
        if target is None or any(n is None or t <= 0 for n, t in nums) or target.low is None or target.high is None:
            return
        if target.kind == "range" or any(n.kind == "range" for n, _t in nums):
            return  # "1 to 3" is a span (photographs, pages, items), not an amount: adding spans is not arithmetic
        kinds = {_is_fraction(r[0], n) for (r, _t), (n, _t2) in zip(terms, nums)} | {_is_fraction(against[0], target)}
        if len(kinds) > 1:
            stats.unlike += 1  # a rate, price or percentage added to an amount (or a sum of percentages compared with an amount) is not arithmetic
            return
        nums = [(n.scaled(_scale(r[0])), t) for (r, _t), (n, t) in zip(terms, nums)]  # "4,505.6 ($ in thousands)" is 4,505,600 before it is compared
        target = target.scaled(_scale(against[0]))
        stats.checked += 1
        low = sum(n.low * type(n.low)(str(t)) for n, t in nums)
        high = sum(n.high * type(n.high)(str(t)) for n, t in nums)
        slack = abs(target.high) * type(target.high)(str(TOLERANCE))
        broken = {"equals": high < target.low - slack or low > target.high + slack,
                  "at_most": low > target.high + slack, "at_least": high < target.low - slack}[check.relation]
        key = frozenset(r[0].id for r in members)
        if frozenset((r[0].chunk_id, re.sub(r"[^\d.]", "", str(r[0].value))) for r in members) in self._pattern_figures:
            return  # a pattern found this relation already
        if not broken:
            stats.held += 1
            return
        stats.broken += 1
        if key in self._seen:
            return
        self._seen.add(key)
        self._save_arithmetic(check, terms, against, (low + high) / 2, chunks, stats)

    def _save_arithmetic(self, check: Relation, terms, against, computed, chunks: dict, stats: InternalStats) -> None:
        members = [r for r, _t in terms] + [against]
        quotes, seen = [], set()
        for r in members:
            chunk = chunks[r[0].chunk_id]
            q = clause_quote(chunk.content, r[0].value)
            if q and " ".join(q.split()) not in seen:
                seen.add(" ".join(q.split()))
                quotes.append((r, chunk, q))
        if len(quotes) < 2:
            stats.rejected += 1
            stats.log.append(f"{check.label}: broken, but the figures are not in two different passages")
            return
        sides = " + ".join(f"{r[0].value}" + (f" x {t:g}" if t != 1 else "") for r, t in terms)
        stated = against[0].value
        word = {"equals": "should equal", "at_most": "cannot exceed", "at_least": "cannot be less than"}[check.relation]
        description = (f"{against[1]}: {check.label}. The document's own figures do not add up: {sides} = {computed:,.0f}, which {word} the {stated} it states. "
                       f"Found by checking arithmetic in code.")
        claims_like = [r[0] for r in members]
        result = self.registry.execute("submit_finding", {
            "title": f"Internal inconsistency in {against[1]}: {check.label}"[:500], "description": description, "classification": "contradiction",
            "severity": contradiction_severity(claims_like), "confidence": 0.9, "evidence_status": "adequate",
            "evidence": [{"chunk_id": r[0].chunk_id, "quoted_text": q, "role": "contradicting", "relevance_reason": r[0].claim_text or check.label,
                          "support_assessment": f"{r[1]} states {r[0].value} ({r[2]})"} for r, _c, q in quotes]})
        if not result.ok:
            stats.rejected += 1
            stats.log.append(f"{check.label}: {result.error.code}: {result.error.message[:140]}")
            return
        stats.saved += 1
        stats.log.append(f"{against[1]}: {check.label} ({sides} = {computed:,.0f} against {stated})")

    # -- relative dates ---------------------------------------------------------------------------------------------------------------
    RELATIVE = re.compile(r"(?P<n>\d+)\s*(?P<unit>day|week|month|year)s?\s+(?:of|after|from|before)\s+(?:the\s+)?(?P<anchor>[A-Za-z][A-Za-z ]{2,30}?)"
                          r"(?=[.,;)]|\s+(?:and|or|but|which|if|following)\b|$)", re.IGNORECASE)
    _STOP = {"the", "of", "date", "a", "an", "each", "any", "its", "their", "this", "that"}

    @staticmethod
    def _stem(word: str) -> str:
        return word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word

    def _anchor_words(self, anchor: str) -> frozenset[str]:
        return frozenset(self._stem(w) for w in re.findall(r"[a-z0-9]+", anchor.lower()) if w not in self._STOP)

    def _anchors(self, rows, chunks: dict, stats: InternalStats) -> None:
        """A date written relative to an event ("within 30 days of handover") needs a claim somewhere that dates that event; with none, the anchor is missing."""
        dated = []
        for r in rows:
            v = r[0]
            if v.value_type == "date" or isinstance(parse_date(v.value), tuple):
                dated.append((v, frozenset(self._stem(w) for w in re.findall(r"[a-z0-9]+", f"{v.subject or ''} {v.metric or ''} {v.claim_text or ''}".lower()))))
        reported: set[tuple[str, frozenset[str]]] = set()
        for r in rows:
            v = r[0]
            m = self.RELATIVE.search(f"{v.value or ''} {v.claim_text or ''}")
            words = self._anchor_words(m.group("anchor")) if m else frozenset()
            if not m or not words:
                continue
            stats.anchors_checked += 1
            if any(d.id != v.id and words <= tokens for d, tokens in dated):
                continue
            if (v.document_id, words) in reported:
                continue
            reported.add((v.document_id, words))
            stats.anchors_missing += 1
            chunk = chunks[v.chunk_id]
            quote = clause_quote(chunk.content, m.group(0), fallback=v.claim_text or "")
            if not quote:
                continue
            result = self.registry.execute("submit_finding", {
                "title": f"Missing information: the date of '{m.group('anchor').strip()}' needed for '{m.group(0).strip()}'"[:500],
                "description": f"{r[1]} states a date relative to '{m.group('anchor').strip()}' ({m.group(0).strip()}), but no claim in the data room gives the date of that event, "
                               f"so the date cannot be worked out. Found by code: confirm that no document states it.",
                "classification": "missing_information", "severity": missing_severity(quote), "confidence": 0.5, "evidence_status": "unavailable",
                "unresolved_questions": [f"What is the date of {m.group('anchor').strip()}?"],
                "evidence": [{"chunk_id": v.chunk_id, "quoted_text": quote, "role": "insufficient", "relevance_reason": f"Relative date: {m.group(0).strip()}",
                              "support_assessment": f"{r[1]} gives a date relative to {m.group('anchor').strip()}, which no claim dates."}]})
            if result.ok:
                stats.saved += 1
                stats.log.append(f"{r[1]}: '{m.group(0).strip()}' has no dated '{m.group('anchor').strip()}'")
            else:
                stats.rejected += 1
