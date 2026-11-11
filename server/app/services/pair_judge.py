"""Judges candidate claim pairs with the LLM and saves findings whose quotes are verified in code.

Pairs whose claims agree exactly (relation hint "exact") are counted as corroboration and never sent to
the LLM. Every other pair is shown to the model with both claims and both source excerpts, in batches
sized by characters (no fixed counts). The model returns a verdict per pair; for a discrepancy it must
copy a quote from each excerpt. ``submit_finding`` then verifies each quote is an exact substring of its
chunk and rejects the finding otherwise, so a wrong or invented quote never reaches the report.
"""
from __future__ import annotations

import logging
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

from pydantic import BaseModel, Field
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.llm.base import LLMClient
from app.llm.errors import LLMError
from app.models import Claim, Document, DocumentChunk, Evidence, Finding
from app.models.base import Classification, Severity
from app.schemas.claim import CandidatePair
from app.tools.submit import find_source_span
from app.services.candidate_matching import TABLE_TYPES, normalize_unit, subjects_overlap, tokens
from app.services.numbers import value_supported_by_text
from app.services.value_rules import compare_values, comparison_note, normalise_text
from app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)
ProgressFn = Callable[[str, dict], None]

# Verdicts that become findings (with quotes). A pair the model calls not comparable is only counted.
REPORTED = {Classification.contradiction, Classification.contextual_difference, Classification.missing_information,
            Classification.consistent, Classification.insufficient_evidence}
NEEDS_QUOTES_TO_SAVE = {Classification.consistent, Classification.insufficient_evidence}  # without quotes: counted, not saved
ROLE = {Classification.contradiction: "contradicting", Classification.contextual_difference: "contextual",
        Classification.missing_information: "insufficient", Classification.consistent: "supporting",
        Classification.insufficient_evidence: "insufficient"}

SYSTEM = """You compare pairs of claims taken from documents of the same collection and decide whether each pair is a real discrepancy.

Rules:
1. Use ONLY the claims and excerpts shown. Never invent text, values or context.
2. Compare like with like: same subject, metric, unit, period and scope. Decide from what the two excerpts themselves state. Under each pair a "Code check" line says whether code found the two values to agree within their stated precision (scale words, units of one kind and rounding already allowed for), to differ (with the gap), or could not compare them. Trust it for the numbers; you decide what the difference means.
3. DECISION ORDER for classification (stop at the first step that applies):
   a. Different subject or metric, or explicit periods or scopes that neither overlap nor nest: same_attribute is false (rule 6). Shared words, digits or phrasing do not make claims comparable.
   b. consistent: the values agree under every reasonable reading (after units, scale, sign, format and spelling are normalised). If the readings split, it is not consistent.
   c. The values conflict under every reasonable reading. BASIS: two values of the same kind for the same item that differ only in basis (planned, budgeted, forecast, target, designed or estimated versus actual, installed, measured, reported or contracted) state the same attribute: set same_attribute true, name both bases in the description, and compare. When the values agree (equal, or within their precision), the claims are consistent whatever their bases. When they differ: different bases are a contextual_difference, the same basis is a contradiction. contextual_difference: the excerpts themselves state a distinguishing context (a different period, scope, condition, definition, version or as-of date) that fully accounts for the gap; name it in the description. contradiction: no such context is stated, or the stated context cannot explain the gap (for example a part larger than its whole). Two documents or editions giving different values for the same thing is a contradiction even if you can imagine a reason; never invent an explanation that is not in the excerpts ("may include tax" is not a quote). A version or an as-of date explains a gap only when an excerpt states the change (an event, the date it took effect, or that a later edition supersedes the earlier one); different document dates alone never explain a gap. A different label is a context difference only when an excerpt states what each figure includes or excludes.
   d. missing_information: an excerpt itself states that an item is not included, not attached, outstanding or not available, or a figure needed to decide (the base of a percentage, an anchor date, a fiscal year end) is not given. Name the item in the description. Whether the item exists elsewhere in the collection is checked by code, not by you.
   e. insufficient_evidence: anything else the excerpts cannot settle: a vague or hedged value, an unclear format, a currency without a stated rate, or readings that disagree about the outcome. Say what blocks the test.
4. For every verdict whose same_attribute is true (consistent and insufficient_evidence too), copy quote_a EXACTLY from excerpt A and quote_b EXACTLY from excerpt B: the shortest passage that shows the value together with the words or unit next to it (at least 8 characters, e.g. "185W/m² power supply", never the bare number).
5. quote_a must contain the value of claim A and quote_b the value of claim B, exactly as written in the excerpt. For a claim read from a table, quote that claim's own row including its label (its ID or name), never a neighbouring row that happens to show the same number.
6. same_attribute: true only if both claims state the same attribute of the same subject, so that their values could be compared. false if they measure different things (for example a height and a floor level, or a distance and a duration), even when their units or numbers look alike. The one exception is the BASIS rule in step 3: different bases of the same quantity of the same item are the same attribute. Add attribute_reasoning: one short sentence saying what each claim measures. When same_attribute is false the pair is not compared and the classification is ignored.
7. For every verdict whose same_attribute is true also fill in: title (one line saying what differs, e.g. the thing and the two values), description (both values, which document states each and where, and why they conflict or what context separates them), severity (see SEVERITY below), confidence (0 to 1) and materiality_reasoning (one sentence).
8. Never do arithmetic. State numbers as written. The gap in the Code check line is computed by code.
9. Return exactly one verdict per pair, using the pair_index shown (see the output format below). Both excerpts may come from the same chunk when two different sentences in it conflict; two claims read from one sentence are not a pair.

SEVERITY (choose it after the classification): consistent is informational. insufficient_evidence is low. contextual_difference is low, or medium when the explaining context appears in only one document or only in a note. missing_information is medium, or high when another figure or obligation depends on the missing item. contradiction is critical when the gap is large (5% or more, or any identifier, name, yes/no or date that cannot both be true) AND the claim is prominent (a title, summary, total, definition, obligation or signature block); high when it is large OR prominent; low when the gap is under 1%; otherwise medium. A basis difference is medium when the other items of the same kind agree between the two sources (the outlier is suspicious), otherwise low.

RULES BY TYPE OF VALUE
- Numbers: they agree when their stated precision allows it (10.2 million and 10,213,000; about 10,000 and 9,640; more than 500 and 612). Check scale words and table headings ("in thousands"), units (1.5 t and 1,500 kg agree) and sign (a bracketed table amount is negative). Compare currencies only when both are in the same currency; a different currency without a stated rate is insufficient_evidence. A percentage against an absolute figure needs a stated base, else missing_information. "Rose 2%" against 10% and 12% can mean percent or percentage points: if the readings disagree, insufficient_evidence. A part larger than its whole (a half-year above its full year, a total that excludes something above the total that includes it) is a contradiction whatever context is named. A balance is compared only at the same date, a flow only over the same period.
- Dates: a date covers the whole period it names (2024 is the full year; Q3 2024 contains 1 July 2024). Day and month order that the document does not settle is insufficient_evidence. A fiscal period equals a calendar one only when its year end is stated; otherwise missing_information naming it. A relative date ("30 days after completion") needs its anchor date. Years and months convert exactly; start plus term against an end date allows one day.
- Identifiers and phone numbers (certificate, order, account, serial numbers, codes): equal only when identical once spaces, hyphens, dots, slashes and case are ignored. Any other difference is a contradiction only when both claims label the same thing (the same certificate, order, account or item). Identifiers of two different things (two tools, two rows, two orders) do not conflict: same_attribute is false. Never judge them by similarity.
- Names: equal when case, punctuation, word order and legal-form words (Ltd, Limited, Inc, GmbH) are ignored, or when an alias is stated in the excerpts. Any other differing word is a contradiction when both name the same role.
- Yes/no and status: compare polarity and quantifier at the same date. None against a count above zero, all against none, or opposite polarity is a contradiction; "not all" agrees with "none"; a status that differs between two stated dates is a contextual_difference.
- Statements: they agree if one implies the other and contradict if both cannot be true. A list differs from a longer or shorter list only when one claims to be complete (only, all, exclusively). A hedge (generally, mostly, significantly) set against specific exceptions is insufficient_evidence.

OUTPUT FORMAT (strict). Return exactly one JSON object and nothing else: {"verdicts": [ ... ]}, with one verdict per pair. Every verdict has ALL of these fields:
- pair_index: the integer shown for the pair.
- same_attribute: true or false.
- attribute_reasoning: a string.
- classification: exactly one of contradiction, contextual_difference, consistent, insufficient_evidence, missing_information.
- severity: exactly one of critical, high, medium, low, informational.
- confidence: a number from 0 to 1.
- value_a and value_b: the value of claim A and of claim B exactly as shown for the pair (they let code check the verdict belongs to this pair).
- title, description, materiality_reasoning, quote_a, quote_b: strings. Use an empty string, never null, when there is nothing to put.
Never omit a field, never use a value outside the lists above, and never add other fields, comments or markdown."""


class PairVerdict(BaseModel):
    pair_index: int
    value_a: str = ""  # echoed by the model; checked against the pair
    value_b: str = ""
    same_attribute: bool = True  # false = the two claims measure different things and are not comparable
    attribute_reasoning: str = ""
    classification: Classification
    severity: Severity = Severity.informational
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    title: str = ""
    description: str = ""
    materiality_reasoning: str | None = None
    quote_a: str = ""
    quote_b: str = ""


class PairVerdictBatch(BaseModel):
    verdicts: list[PairVerdict] = Field(default_factory=list)


@dataclass
class PairRecord:
    """What happened to one candidate pair: both claims, how it was paired, and the outcome (for the run report)."""
    claim_a: str
    claim_b: str
    hint: str
    route: str
    outcome: str = ""


@dataclass
class SavedFinding:
    finding_id: str
    classification: str
    severity: str
    confidence: float
    claim_ids: tuple[str, str]
    title: str


MERGEABLE = {"contradiction": 3, "contextual_difference": 2, "insufficient_evidence": 1}  # discrepancy classes, strongest first
SEVERITY_ORDER = ["informational", "low", "medium", "high", "critical"]


@dataclass
class JudgeStats:
    pairs_total: int = 0
    pairs_exact: int = 0  # agreed exactly: corroboration, not sent to the LLM
    pairs_judged: int = 0
    llm_calls: int = 0
    batches_failed: int = 0
    verdict_counts: Counter = field(default_factory=Counter)
    findings_saved: int = 0
    findings_rejected: int = 0  # reported verdicts whose quotes failed verification (or were missing)
    rejections: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    corroborated_claims: set[str] = field(default_factory=set)  # claim ids with an exact match elsewhere
    paired_claims: set[str] = field(default_factory=set)  # claim ids that appear in any candidate pair
    findings_by_document: Counter = field(default_factory=Counter)
    finding_titles_by_document: dict[str, list[str]] = field(default_factory=dict)
    agreement_findings: int = 0  # `consistent` findings made from the same value stated in two documents
    duplicates_skipped: int = 0  # verdicts that restate a finding already saved (same two quotes in the same two chunks)
    seen: set = field(default_factory=set)
    records: dict[tuple[str, str], PairRecord] = field(default_factory=dict)  # one per candidate pair, in pairing order
    misaligned: int = 0  # verdicts whose echoed values did not match their pair (the pair is asked again, alone)
    echo_missing: int = 0  # verdicts that did not echo the values
    retries: int = 0  # single-pair calls for pairs with no usable verdict
    quotes_from_code: int = 0  # consistent / insufficient-evidence verdicts saved with the claims' own sentences because the model gave no quotes
    retry: list = field(default_factory=list)
    saved: list = field(default_factory=list)  # every finding saved from a verdict, with its claims
    merged: int = 0  # findings folded into another that describes the same discrepancy
    absorbed: int = 0  # insufficient-evidence findings dropped because a consistent finding settles the same fact
    grouped: int = 0  # consistent findings folded into one finding per pair of documents and attribute
    deleted: set = field(default_factory=set)  # ids of findings removed by the steps above
    merge_log: list = field(default_factory=list)
    echo_log: list = field(default_factory=list)  # what the model repeated against what the pair holds, for each mismatch

    def note_by_claims(self, claim_ids: tuple[str, str], suffix: str) -> None:
        record = self.records.get(claim_ids) or self.records.get((claim_ids[1], claim_ids[0]))
        if record is not None:
            record.outcome += suffix

    def note(self, p: CandidatePair, outcome: str) -> None:
        record = self.records.get((p.claim_a_id, p.claim_b_id))
        if record is not None:
            record.outcome = outcome


def _location(chunk: DocumentChunk) -> str:
    if chunk.sheet_name:
        return f"sheet {chunk.sheet_name}, {chunk.cell_reference}"
    return ", ".join(p for p in (f"page {chunk.page_number}" if chunk.page_number else "",
                                 f"section {chunk.heading}" if chunk.heading else "") if p) or chunk.content_type


def _shown(claim: Claim) -> str:
    """The claim's value with its unit, unless the value already carries it ("12,000 WSPM" with unit WSPM)."""
    value, unit = str(claim.value), (claim.unit or "").strip()
    return value if not unit or unit.lower() in value.lower() else f"{value} {unit}"


def _claim_line(c: Claim) -> str:
    attrs = {k: v for k, v in (("subject", c.subject), ("metric", c.metric), ("value", c.value), ("unit", c.unit),
                               ("period", c.period), ("conditions", c.conditions)) if v not in (None, "")}
    return f"{c.claim_text} | " + "; ".join(f"{k}={v}" for k, v in attrs.items())


def quote_for(claim: Claim, chunk: DocumentChunk) -> str | None:
    """An exact passage of the chunk that states the claim: its own sentence, else the line, else the words around its value."""
    text = chunk.content
    subject = " ".join((claim.subject or "").split()).casefold()
    if chunk.content_type in TABLE_TYPES and subject:
        # a table chunk holds several rows: quote the row labelled with the claim's subject, preferring the line that also holds its value
        rows = [line.strip() for line in text.splitlines() if len(line.strip()) >= 8 and subject in " ".join(line.split()).casefold()]
        value_text = str(claim.value or "").strip()
        own = [line for line in rows if value_text and value_text in line] or rows
        if own:
            return own[0][:400]
    if claim.claim_text:
        span = find_source_span(text, claim.claim_text)
        if span and len(span[0]) >= 8:
            return span[0][:400]
    value = str(claim.value or "").strip()
    if value:
        for line in text.splitlines():
            if value in line and len(line.strip()) >= 8:
                return line.strip()[:300]
        i = text.find(value)
        if i >= 0:
            around = text[max(0, i - 40): i + len(value) + 40].strip()
            return around if len(around) >= 8 else None
    return None


class PairJudge:
    def __init__(self, session: Session, llm: LLMClient, registry: ToolRegistry, max_chars_per_call: int = 12_000,
                 on_progress: ProgressFn | None = None, should_stop: Callable[[], bool] | None = None, workers: int = 1):
        self.session = session
        self.llm = llm
        self.workers = max(1, workers)  # model calls that run at the same time (1 = one after another)
        self.registry = registry
        self.max_chars = max_chars_per_call
        self.on_progress = on_progress or (lambda _m, _d: None)
        self.should_stop = should_stop or (lambda: False)

    def run(self, pairs: list[CandidatePair]) -> JudgeStats:
        stats = JudgeStats(pairs_total=len(pairs))
        claim_ids = {p.claim_a_id for p in pairs} | {p.claim_b_id for p in pairs}
        claims = {c.id: c for c in self.session.scalars(select(Claim).where(Claim.id.in_(claim_ids)))} if claim_ids else {}
        chunk_ids = {c.chunk_id for c in claims.values()}
        chunks = {c.id: c for c in self.session.scalars(select(DocumentChunk).where(DocumentChunk.id.in_(chunk_ids)))} if chunk_ids else {}
        names = {d.id: d.filename for d in self.session.scalars(select(Document))}

        to_judge: list[CandidatePair] = []
        for p in pairs:
            stats.paired_claims.update((p.claim_a_id, p.claim_b_id))
            stats.records[(p.claim_a_id, p.claim_b_id)] = PairRecord(
                self._describe(claims.get(p.claim_a_id), names), self._describe(claims.get(p.claim_b_id), names), p.relation_hint, p.matched_by)
            ca, cb = claims.get(p.claim_a_id), claims.get(p.claim_b_id)
            if p.relation_hint == "exact" and (ca is None or cb is None or subjects_overlap(ca.subject, cb.subject)):
                stats.pairs_exact += 1
                stats.corroborated_claims.update((p.claim_a_id, p.claim_b_id))
                self._save_agreement(p, claims, chunks, names, stats)
            else:  # equal values about different subjects (a design load and a consent record) are not an agreement until the judge says so
                to_judge.append(p)
        stats.pairs_judged = len(to_judge)

        batches = list(self._batches(to_judge, claims, chunks))
        answered = 0
        for batch, result in self._ask(batches, claims, chunks, names):
            if result is None:
                continue
            answered += 1
            stats.llm_calls += 1
            if isinstance(result, LLMError):
                stats.batches_failed += 1
                stats.errors.append(str(result)[:200])
                logger.warning("Pair judging batch failed", extra={"error": str(result)})
                continue
            for verdict in result.verdicts:
                self._handle(verdict, batch, claims, chunks, names, stats)
            self.on_progress(f"Judged {stats.pairs_judged and sum(stats.verdict_counts.values())} of {stats.pairs_judged} pairs",
                             {"judged": sum(stats.verdict_counts.values()), "total": stats.pairs_judged})
        if answered < len(batches):
            stats.errors.append("stop requested")
        if not self.should_stop():
            again = {(p.claim_a_id, p.claim_b_id): p for p in stats.retry}
            for p in to_judge:
                key = (p.claim_a_id, p.claim_b_id)
                if not stats.records[key].outcome and p.claim_a_id in claims and p.claim_b_id in claims:
                    again[key] = p
            singles = [[p] for p in again.values()]
            for batch, result in self._ask(singles, claims, chunks, names):
                if result is None:
                    continue
                stats.llm_calls += 1
                stats.retries += 1
                if isinstance(result, LLMError):
                    stats.errors.append(str(result)[:200])
                    continue
                for verdict in result.verdicts[:1]:
                    verdict.pair_index = 0
                    self._handle(verdict, batch, claims, chunks, names, stats, retrying=True)
        self._merge_findings(stats, pairs)
        self._absorb_insufficient(stats, pairs)
        self._group_consistent(stats, claims, names)
        for record in stats.records.values():
            if not record.outcome:
                record.outcome = "no verdict returned (batch failed, run stopped, or the model skipped the pair)"
        return stats

    def _ask(self, batches: list[list[CandidatePair]], claims, chunks, names):
        """The model's answer for each batch, in batch order (an LLMError instead of an answer when the call failed). With several workers
        the calls run at the same time; only the model calls do: prompts are built and answers handled one at a time, here and by the caller,
        because the database session is not shared between threads."""
        prompts = [self._prompt(batch, claims, chunks, names) for batch in batches]

        def call(prompt: str):
            if self.should_stop():  # a stop request also skips the calls that are still waiting for a worker
                return None
            try:
                return self.llm.generate_structured_output(prompt, PairVerdictBatch, system=SYSTEM)
            except LLMError as exc:
                return exc

        if self.workers <= 1 or len(prompts) <= 1:
            for batch, prompt in zip(batches, prompts):
                answer = call(prompt)
                if answer is None:
                    return
                yield batch, answer
            return
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = [(batch, pool.submit(call, prompt)) for batch, prompt in zip(batches, prompts)]
            for batch, future in futures:
                yield batch, future.result()

    @staticmethod
    def _components(edges: list[tuple[str, str]]):
        parent: dict[str, str] = {}

        def find(x: str) -> str:
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for x, y in edges:
            parent[find(x)] = find(y)
        return find

    def _remove_finding(self, finding_id: str, stats: JudgeStats) -> None:
        self.session.execute(delete(Evidence).where(Evidence.finding_id == finding_id))
        gone = self.session.get(Finding, finding_id)
        if gone is not None:
            self.session.delete(gone)
        stats.deleted.add(finding_id)
        stats.findings_saved -= 1

    def _absorb_insufficient(self, stats: JudgeStats, pairs: list[CandidatePair]) -> None:
        """When a consistent finding settles a fact (and nothing says it conflicts), an insufficient-evidence finding about the same fact is noise."""
        alive = [s for s in stats.saved if s.finding_id not in stats.deleted]
        if not any(s.classification == "insufficient_evidence" for s in alive):
            return
        edges = [s.claim_ids for s in alive] + [(p.claim_a_id, p.claim_b_id) for p in pairs if p.relation_hint == "exact"]
        find = self._components(edges)
        groups: dict[str, list[SavedFinding]] = defaultdict(list)
        for s in alive:
            groups[find(s.claim_ids[0])].append(s)
        for group in groups.values():
            classes = {s.classification for s in group}
            if "consistent" in classes and not classes & {"contradiction", "contextual_difference"}:
                for s in group:
                    if s.classification == "insufficient_evidence":
                        self._remove_finding(s.finding_id, stats)
                        stats.absorbed += 1
                        stats.merge_log.append(f"insufficient-evidence finding dropped because a consistent finding settles the same fact: {s.title[:90]}")
        self.session.commit()

    def _group_consistent(self, stats: JudgeStats, claims: dict[str, Claim], names: dict[str, str]) -> None:
        """One consistent finding per pair of documents and attribute (\"Tool_Category agrees for 12 of 12 tools\"), not one per row."""
        groups: dict[tuple, list[SavedFinding]] = defaultdict(list)
        for s in stats.saved:
            if s.classification != "consistent" or s.finding_id in stats.deleted:
                continue
            a, b = claims.get(s.claim_ids[0]), claims.get(s.claim_ids[1])
            if a is None or b is None:
                continue
            label = lambda c: " ".join(sorted(tokens(c.metric))) or "-"
            groups[(frozenset((a.document_id, b.document_id)), frozenset((label(a), label(b))))].append(s)
        for (docs, _metrics), group in groups.items():
            if len(group) < 2:
                continue
            keep, others = group[0], group[1:]
            main = self.session.get(Finding, keep.finding_id)
            if main is None:
                continue
            first = claims[keep.claim_ids[0]]
            doc_names = sorted(names.get(d, "?") for d in docs)
            have = {(e.chunk_id, " ".join(e.quoted_text.split())) for e in self.session.scalars(select(Evidence).where(Evidence.finding_id == main.id))}
            kept_rows = len(have)
            values: list[str] = []
            for s in group:
                v = str(claims[s.claim_ids[0]].value)
                if v not in values:
                    values.append(v)
            for other in others:
                for e in list(self.session.scalars(select(Evidence).where(Evidence.finding_id == other.finding_id))):
                    key = (e.chunk_id, " ".join(e.quoted_text.split()))
                    if key not in have and kept_rows < 16:  # the report stays readable: the first 8 items are shown as evidence
                        have.add(key)
                        kept_rows += 1
                        self.session.add(Evidence(finding_id=main.id, chunk_id=e.chunk_id, evidence_role=e.evidence_role, quoted_text=e.quoted_text,
                                                  relevance_reason=e.relevance_reason, support_assessment=e.support_assessment))
                self._remove_finding(other.finding_id, stats)
            what = first.metric or "value"
            main.title = f"Consistent: {what} agrees in {len(group)} places"[:300]
            main.description = (f"{doc_names[0]} and {doc_names[-1]} state the same value for {what} in {len(group)} places "
                                f"(values: {'; '.join(values[:8])}{'; ...' if len(values) > 8 else ''}).")
            stats.grouped += len(others)
            stats.merge_log.append(f"{len(group)} consistent findings between {doc_names[0]} and {doc_names[-1]} about {what} became one")
        self.session.commit()

    def _merge_findings(self, stats: JudgeStats, pairs: list[CandidatePair]) -> None:
        """One discrepancy, one finding. Findings of the discrepancy classes whose claims are connected (they share a claim, or two of their
        claims state the same value) describe the same discrepancy: the strongest class is kept, the others' quotes are added to it as evidence,
        and the others are removed."""
        members = [s for s in stats.saved if s.classification in MERGEABLE]
        if len(members) < 2:
            return
        parent: dict[str, str] = {}

        def find(x: str) -> str:
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: str, y: str) -> None:
            parent[find(x)] = find(y)

        for s in members:
            union(*s.claim_ids)
        for p in pairs:
            if p.relation_hint == "exact":  # the same value stated twice is one fact
                union(p.claim_a_id, p.claim_b_id)
        groups: dict[str, list[SavedFinding]] = defaultdict(list)
        for s in members:
            groups[find(s.claim_ids[0])].append(s)
        for group in groups.values():
            if len(group) < 2:
                continue
            group.sort(key=lambda s: (-MERGEABLE[s.classification], -s.confidence))
            keep, others = group[0], group[1:]
            main = self.session.get(Finding, keep.finding_id)
            if main is None:
                continue
            role = ROLE[Classification(keep.classification)]
            have = {(e.chunk_id, " ".join(e.quoted_text.split())) for e in self.session.scalars(select(Evidence).where(Evidence.finding_id == main.id))}
            severity = max([keep.severity] + [o.severity for o in others], key=SEVERITY_ORDER.index)
            for other in others:
                for e in list(self.session.scalars(select(Evidence).where(Evidence.finding_id == other.finding_id))):
                    key = (e.chunk_id, " ".join(e.quoted_text.split()))
                    if key not in have:
                        have.add(key)
                        self.session.add(Evidence(finding_id=main.id, chunk_id=e.chunk_id, evidence_role=e.evidence_role, quoted_text=e.quoted_text,
                                                  relevance_reason=e.relevance_reason, support_assessment=e.support_assessment))
                self.session.execute(delete(Evidence).where(Evidence.finding_id == other.finding_id))
                gone = self.session.get(Finding, other.finding_id)
                if gone is not None:
                    self.session.delete(gone)
                stats.deleted.add(other.finding_id)
            main.severity = severity
            main.description = (main.description or "") + " Also reported separately and merged here: " + "; ".join(f"{o.classification.replace('_', ' ')} ({o.title[:80]})" for o in others) + "."
            stats.merged += len(others)
            stats.findings_saved -= len(others)
            stats.merge_log.append(f"{len(others) + 1} findings about one discrepancy became one {keep.classification}: {keep.title[:90]} (merged: {', '.join(o.classification for o in others)})")
            for s in group:
                stats.note_by_claims(s.claim_ids, f" [merged into: {keep.title[:60]}]")
        self.session.commit()

    @staticmethod
    def _describe(claim: Claim | None, names: dict[str, str]) -> str:
        if claim is None:
            return "(claim not found)"
        return f"{(names.get(claim.document_id) or '?')[:2]} | {claim.subject or '-'} | {claim.metric or '-'} | {claim.value}" + (f" {claim.unit}" if claim.unit else "")

    @staticmethod
    def _row_quote(claim: Claim, chunk: DocumentChunk, quote: str) -> tuple[str, str | None]:
        """A claim read from a table row must be quoted from that row. A quote from another row of the same chunk is refused; a quote from
        the right row that leaves out the row label is widened to the whole row, so the stored evidence names the row. (quote, error)."""
        subject = " ".join((claim.subject or "").split()).casefold()
        if chunk.content_type not in TABLE_TYPES or not subject:
            return quote, None
        content = chunk.content
        if subject not in " ".join(content.split()).casefold():
            return quote, None  # the subject is not a label in this chunk, so the row cannot be told
        span = find_source_span(content, quote)
        start = content.find(span[0]) if span else -1
        if start < 0:
            return quote, None
        end, position, lines = start + len(span[0]), 0, []
        for line in content.split("\n"):
            low, high = position, position + len(line)
            position = high + 1
            if low < end and high > start:
                lines.append(line)
        holding = [line for line in lines if subject in " ".join(line.split()).casefold()]
        if not holding:
            return quote, f"the quote is from a different row than the claim about '{claim.subject}'"
        if subject in " ".join(span[0].split()).casefold():
            return quote, None
        return holding[0].strip()[:400], None

    # -- batching and prompt -------------------------------------------------
    def _size(self, p: CandidatePair, claims: dict[str, Claim], chunks: dict[str, DocumentChunk]) -> int:
        a, b = claims[p.claim_a_id], claims[p.claim_b_id]
        return len(chunks[a.chunk_id].content) + len(chunks[b.chunk_id].content) + len(a.claim_text) + len(b.claim_text) + 200

    def _batches(self, pairs, claims, chunks):
        current, size = [], 0
        for p in pairs:
            if p.claim_a_id not in claims or p.claim_b_id not in claims:
                continue
            n = self._size(p, claims, chunks)
            if current and size + n > self.max_chars:
                yield current
                current, size = [], 0
            current.append(p)
            size += n
        if current:
            yield current

    @staticmethod
    def _prompt(batch, claims, chunks, names) -> str:
        blocks = []
        for i, p in enumerate(batch):
            a, b = claims[p.claim_a_id], claims[p.claim_b_id]
            ca, cb = chunks[a.chunk_id], chunks[b.chunk_id]
            note = comparison_note(a.value, a.unit, b.value, b.unit, a.claim_category, b.claim_category, normalize_unit)
            blocks.append(
                f"### Pair {i}\n{note}\nClaim A ({names.get(a.document_id)}, {_location(ca)}): {_claim_line(a)}\nExcerpt A:\n{ca.content}\n\n"
                f"Claim B ({names.get(b.document_id)}, {_location(cb)}): {_claim_line(b)}\nExcerpt B:\n{cb.content}")
        return "Decide each pair below.\n\n" + "\n\n".join(blocks)

    def _save_agreement(self, p: CandidatePair, claims, chunks, names, stats: JudgeStats) -> None:
        """Two documents state the same value: a `consistent` finding with a verified quote from each (no LLM call)."""
        a, b = claims.get(p.claim_a_id), claims.get(p.claim_b_id)
        if a is None or b is None or a.document_id == b.document_id:
            stats.note(p, "agreement inside one document: counted, not saved")
            return
        ca, cb = chunks.get(a.chunk_id), chunks.get(b.chunk_id)
        if ca is None or cb is None:
            stats.note(p, "agreement not saved: a claim's chunk was not found")
            return
        quote_a, quote_b = quote_for(a, ca), quote_for(b, cb)
        if not quote_a or not quote_b:
            stats.note(p, "agreement not saved: no exact quote found for a claim")
            return
        key = frozenset(((a.chunk_id, " ".join(quote_a.split())), (b.chunk_id, " ".join(quote_b.split()))))
        if key in stats.seen:
            stats.duplicates_skipped += 1
            stats.note(p, "agreement: same quotes as a finding already saved")
            return
        stats.seen.add(key)
        what = a.metric or b.metric or a.subject or "value"
        shown = _shown(a)
        title = f"Consistent: {what}"
        result = self.registry.execute("submit_finding", {
            "title": title[:300], "classification": "consistent", "severity": "informational", "confidence": 0.9,
            "evidence_status": "adequate",
            "description": f"{names.get(a.document_id)} ({_location(ca)}) and {names.get(b.document_id)} ({_location(cb)}) "
                           f"state the same value for {what}: {shown}.",
            "evidence": [
                {"chunk_id": a.chunk_id, "quoted_text": quote_a, "role": "supporting", "relevance_reason": a.claim_text or what,
                 "support_assessment": f"{names.get(a.document_id)} states: {_claim_line(a)}"[:600]},
                {"chunk_id": b.chunk_id, "quoted_text": quote_b, "role": "supporting", "relevance_reason": b.claim_text or what,
                 "support_assessment": f"{names.get(b.document_id)} states: {_claim_line(b)}"[:600]},
            ]})
        if not result.ok:
            stats.findings_rejected += 1
            stats.rejections.append(f"consistent '{title[:60]}': {result.error.code}: {result.error.message[:160]}")
            stats.note(p, f"agreement rejected: {result.error.code}")
            return
        stats.findings_saved += 1
        stats.agreement_findings += 1
        stats.saved.append(SavedFinding(result.output.finding_id, "consistent", "informational", 0.9, (a.id, b.id), title))
        stats.note(p, "agreement saved as a consistent finding")
        for doc_id in {a.document_id, b.document_id}:
            stats.findings_by_document[doc_id] += 1
            stats.finding_titles_by_document.setdefault(doc_id, []).append(title)

    # -- verdict handling ----------------------------------------------------
    @staticmethod
    def _echo_matches(v: PairVerdict, a: Claim, b: Claim) -> bool:
        """The values the model repeated belong to this pair's claims (in either order)."""
        def same(echo: str, claim: Claim) -> bool:
            if not echo.strip():
                return True
            return compare_values(echo, claim.unit, claim.value, claim.unit, None, None, normalize_unit) == "equal" or \
                normalise_text(echo) == normalise_text(claim.value)
        return (same(v.value_a, a) and same(v.value_b, b)) or (same(v.value_a, b) and same(v.value_b, a))

    def _handle(self, v: PairVerdict, batch, claims, chunks, names, stats: JudgeStats, retrying: bool = False) -> None:
        if not 0 <= v.pair_index < len(batch):
            stats.verdict_counts[v.classification.value] += 1
            stats.rejections.append(f"verdict for unknown pair_index {v.pair_index}")
            return
        p = batch[v.pair_index]
        a, b = claims[p.claim_a_id], claims[p.claim_b_id]
        if not (v.value_a.strip() or v.value_b.strip()):
            stats.echo_missing += 1
        elif not self._echo_matches(v, a, b):
            stats.misaligned += 1
            stats.echo_log.append(f"{'retry' if retrying else 'first call'}: pair holds {a.value!r} / {b.value!r}; model repeated {v.value_a!r} / {v.value_b!r}; it said: {v.attribute_reasoning[:100]}")
            if retrying:
                stats.note(p, "judge: verdict misaligned with its pair twice, discarded")
            else:
                stats.retry.append(p)
                stats.note(p, "judge: verdict misaligned with its pair (echoed values differ), asked again")
            return
        if not v.same_attribute:  # different attributes: not a comparison at all, so no classification applies
            stats.verdict_counts["not_comparable"] += 1
            stats.note(p, "judge: not comparable. " + (v.attribute_reasoning.strip() or "(no reason given)"))
            return
        stats.verdict_counts[v.classification.value] += 1
        if v.classification not in REPORTED:
            stats.note(p, f"judge: {v.classification.value} (not a reported class)")
            return
        role = ROLE[v.classification]
        quote_a, quote_b = v.quote_a, v.quote_b
        if v.classification in NEEDS_QUOTES_TO_SAVE and not (quote_a.strip() and quote_b.strip()):
            # the model gave no quotes: use each claim's own sentence from its chunk, so the class is still reported with verified evidence,
            # but only when the two claims are about the same thing (unrelated claims would give a finding about nothing)
            if not subjects_overlap(a.subject, b.subject):
                stats.note(p, f"judge: {v.classification.value}, counted, not saved (no quotes from the model, and the claims' subjects do not match)")
                return
            quote_a, quote_b = quote_for(a, chunks[a.chunk_id]) or "", quote_for(b, chunks[b.chunk_id]) or ""
            if not (quote_a and quote_b):
                stats.note(p, f"judge: {v.classification.value}, counted, not saved (no quotes, and no sentence of the claims could be quoted)")
                return
            stats.quotes_from_code += 1
        straight = value_supported_by_text(a.value, quote_a) and value_supported_by_text(b.value, quote_b)
        swapped = value_supported_by_text(a.value, quote_b) and value_supported_by_text(b.value, quote_a)
        if not straight and swapped:  # the model gave the quotes in the opposite order: attach each to its own claim
            quote_a, quote_b = quote_b, quote_a
        elif not straight:
            stats.findings_rejected += 1
            stats.rejections.append(f"{v.classification.value} pair {v.pair_index}: the quotes do not show the claims' values "
                                    f"({a.value!r} and {b.value!r})")
            stats.note(p, f"judge: {v.classification.value}, REJECTED: the quotes do not show the claims' values")
            return
        quote_a, error_a = self._row_quote(a, chunks[a.chunk_id], quote_a)
        quote_b, error_b = self._row_quote(b, chunks[b.chunk_id], quote_b)
        if error_a or error_b:
            stats.findings_rejected += 1
            stats.rejections.append(f"{v.classification.value} pair {v.pair_index}: {error_a or error_b}")
            stats.note(p, f"judge: {v.classification.value}, REJECTED: {error_a or error_b}")
            return
        if v.classification == Classification.consistent:  # an agreement is the same finding whichever claims it came from
            key = frozenset(((a.chunk_id, " ".join(quote_a.split())), (b.chunk_id, " ".join(quote_b.split()))))
        else:  # two different facts can be stated by the same two passages (a start date and a volume in one sentence and one row)
            key = frozenset(((a.id, a.chunk_id, " ".join(quote_a.split())), (b.id, b.chunk_id, " ".join(quote_b.split()))))
        if key in stats.seen:
            stats.duplicates_skipped += 1
            stats.note(p, f"judge: {v.classification.value}, same claims and quotes as a finding already saved")
            return
        stats.seen.add(key)
        title = v.title.strip() or f"{v.classification.value.replace('_', ' ')}: {a.metric or a.subject or 'claims'}"
        description = v.description.strip() or (
            f"{names.get(a.document_id)} states: {a.claim_text} ({_location(chunks[a.chunk_id])}). "
            f"{names.get(b.document_id)} states: {b.claim_text} ({_location(chunks[b.chunk_id])}).")
        result = self.registry.execute("submit_finding", {
            "title": title[:300], "description": description,
            "classification": v.classification.value, "severity": v.severity.value, "confidence": v.confidence,
            "evidence_status": "adequate", "materiality_reasoning": v.materiality_reasoning,
            "evidence": [
                {"chunk_id": a.chunk_id, "quoted_text": quote_a, "role": role, "relevance_reason": a.claim_text,
                 "support_assessment": f"{names.get(a.document_id)} states: {_claim_line(a)}"[:600]},
                {"chunk_id": b.chunk_id, "quoted_text": quote_b, "role": role, "relevance_reason": b.claim_text,
                 "support_assessment": f"{names.get(b.document_id)} states: {_claim_line(b)}"[:600]},
            ]})
        if not result.ok:
            stats.findings_rejected += 1
            stats.rejections.append(f"{v.classification.value} '{title[:60]}': {result.error.code}: {result.error.message[:160]}")
            stats.note(p, f"judge: {v.classification.value}, REJECTED: {result.error.code}")
            return
        stats.findings_saved += 1
        stats.saved.append(SavedFinding(result.output.finding_id, v.classification.value, v.severity.value, v.confidence, (a.id, b.id), title))
        stats.note(p, f"judge: {v.classification.value} ({v.severity.value}), saved: {title[:90]}")
        for doc_id in {a.document_id, b.document_id}:
            stats.findings_by_document[doc_id] += 1
            stats.finding_titles_by_document.setdefault(doc_id, []).append(title)
