"""Judges candidate claim pairs with the LLM and saves findings whose quotes are verified in code.

Pairs whose claims agree exactly (relation hint "exact") are counted as corroboration and never sent to
the LLM. Every other pair is shown to the model with both claims and both source excerpts, in batches
sized by characters (no fixed counts). The model returns a verdict per pair; for a discrepancy it must
copy a quote from each excerpt. ``submit_finding`` then verifies each quote is an exact substring of its
chunk and rejects the finding otherwise, so a wrong or invented quote never reaches the report.
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.llm.base import LLMClient
from app.llm.errors import LLMError
from app.models import Claim, Document, DocumentChunk
from app.models.base import Classification, Severity
from app.schemas.claim import CandidatePair
from app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)
ProgressFn = Callable[[str, dict], None]

# Verdicts that become findings. consistent / insufficient_evidence are counted but not reported as findings.
REPORTED = {Classification.contradiction, Classification.contextual_difference, Classification.missing_information}
ROLE = {Classification.contradiction: "contradicting", Classification.contextual_difference: "contextual",
        Classification.missing_information: "insufficient"}

SYSTEM = """You compare pairs of claims taken from documents of the same collection and decide whether each pair is a real discrepancy.

Rules:
1. Use ONLY the claims and excerpts shown. Never invent text, values or context.
2. Compare like with like: same subject, metric, unit, period and scope. A difference explained by a different period, unit, scope, condition or definition is a contextual_difference, not a contradiction.
3. classification is one of: contradiction (they cannot both be true in the same context), contextual_difference (they differ but context explains it), consistent (they agree), insufficient_evidence (the excerpts do not support a conclusion), missing_information (something needed to compare is absent).
4. For contradiction, contextual_difference and missing_information, copy quote_a EXACTLY from excerpt A and quote_b EXACTLY from excerpt B: the shortest passage that shows the value. For other classifications leave both quotes empty.
5. Never do arithmetic. State numbers as written.
6. Return exactly one verdict per pair, using the pair_index shown."""


class PairVerdict(BaseModel):
    pair_index: int
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


def _location(chunk: DocumentChunk) -> str:
    if chunk.sheet_name:
        return f"sheet {chunk.sheet_name}, {chunk.cell_reference}"
    return ", ".join(p for p in (f"page {chunk.page_number}" if chunk.page_number else "",
                                 f"section {chunk.heading}" if chunk.heading else "") if p) or chunk.content_type


def _claim_line(c: Claim) -> str:
    attrs = {k: v for k, v in (("subject", c.subject), ("metric", c.metric), ("value", c.value), ("unit", c.unit),
                               ("period", c.period), ("conditions", c.conditions)) if v not in (None, "")}
    return f"{c.claim_text} | " + "; ".join(f"{k}={v}" for k, v in attrs.items())


class PairJudge:
    def __init__(self, session: Session, llm: LLMClient, registry: ToolRegistry, max_chars_per_call: int = 12_000,
                 on_progress: ProgressFn | None = None, should_stop: Callable[[], bool] | None = None):
        self.session = session
        self.llm = llm
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
            if p.relation_hint == "exact":
                stats.pairs_exact += 1
                stats.corroborated_claims.update((p.claim_a_id, p.claim_b_id))
            else:
                to_judge.append(p)
        stats.pairs_judged = len(to_judge)

        for batch in self._batches(to_judge, claims, chunks):
            if self.should_stop():
                stats.errors.append("stop requested")
                break
            prompt = self._prompt(batch, claims, chunks, names)
            try:
                stats.llm_calls += 1
                result = self.llm.generate_structured_output(prompt, PairVerdictBatch, system=SYSTEM)
            except LLMError as exc:
                stats.batches_failed += 1
                stats.errors.append(str(exc)[:200])
                logger.warning("Pair judging batch failed", extra={"error": str(exc)})
                continue
            for verdict in result.verdicts:
                self._handle(verdict, batch, claims, chunks, names, stats)
            self.on_progress(f"Judged {stats.pairs_judged and sum(stats.verdict_counts.values())} of {stats.pairs_judged} pairs",
                             {"judged": sum(stats.verdict_counts.values()), "total": stats.pairs_judged})
        return stats

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
            blocks.append(
                f"### Pair {i}\nClaim A ({names.get(a.document_id)}, {_location(ca)}): {_claim_line(a)}\nExcerpt A:\n{ca.content}\n\n"
                f"Claim B ({names.get(b.document_id)}, {_location(cb)}): {_claim_line(b)}\nExcerpt B:\n{cb.content}")
        return "Decide each pair below.\n\n" + "\n\n".join(blocks)

    # -- verdict handling ----------------------------------------------------
    def _handle(self, v: PairVerdict, batch, claims, chunks, names, stats: JudgeStats) -> None:
        stats.verdict_counts[v.classification.value] += 1
        if not 0 <= v.pair_index < len(batch):
            stats.rejections.append(f"verdict for unknown pair_index {v.pair_index}")
            return
        if v.classification not in REPORTED:
            return
        p = batch[v.pair_index]
        a, b = claims[p.claim_a_id], claims[p.claim_b_id]
        role = ROLE[v.classification]
        title = v.title.strip() or f"{v.classification.value.replace('_', ' ')}: {a.metric or a.subject or 'claims'}"
        result = self.registry.execute("submit_finding", {
            "title": title[:300], "description": v.description.strip() or title,
            "classification": v.classification.value, "severity": v.severity.value, "confidence": v.confidence,
            "evidence_status": "adequate", "materiality_reasoning": v.materiality_reasoning,
            "evidence": [
                {"chunk_id": a.chunk_id, "quoted_text": v.quote_a, "role": role, "relevance_reason": a.claim_text,
                 "support_assessment": f"{names.get(a.document_id)} states: {_claim_line(a)}"[:600]},
                {"chunk_id": b.chunk_id, "quoted_text": v.quote_b, "role": role, "relevance_reason": b.claim_text,
                 "support_assessment": f"{names.get(b.document_id)} states: {_claim_line(b)}"[:600]},
            ]})
        if not result.ok:
            stats.findings_rejected += 1
            stats.rejections.append(f"{v.classification.value} '{title[:60]}': {result.error.code}: {result.error.message[:160]}")
            return
        stats.findings_saved += 1
        for doc_id in {a.document_id, b.document_id}:
            stats.findings_by_document[doc_id] += 1
            stats.finding_titles_by_document.setdefault(doc_id, []).append(title)
