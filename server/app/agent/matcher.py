"""Matches the facts the agent recorded while reading, one LLM call per batch of facts.

The model sees every fact (label, value, unit, qualifier, source) and returns groups of facts that
conflict or differ. It never copies quotes: each fact already carries a quote verified against its
chunk when it was recorded, so the finding's evidence is those stored quotes, and ``submit_finding``
checks them again. Batches are sized by characters; facts in different batches are not compared
with each other, which is reported in the stats.
"""
from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from app.llm.base import LLMClient
from app.llm.errors import LLMError
from app.models.base import Classification, Severity
from app.tools.registry import ToolRegistry
from app.tools.schemas import RecordedFact

logger = logging.getLogger(__name__)

REPORTED = {Classification.contradiction, Classification.contextual_difference}

SYSTEM = """You compare facts that were recorded while reading the documents of one collection, and find the ones that disagree.

Rules:
1. Use ONLY the facts shown. Never invent values or context.
2. Compare like with like: same subject, quantity, unit, period and scope. Facts about different things are not related, even if the numbers differ.
3. Report a group only when two or more facts are about the same thing. classification is contradiction (they cannot both be true in the same context) or contextual_difference (they differ but a stated scope, period, condition or definition explains it).
4. A fact may be in the same document as another: a document can contradict itself.
5. Facts that agree are not reported. Do not report a fact on its own.
6. fact_ids lists the ids of the facts in the group, exactly as shown (e.g. fact_12).
7. Never do arithmetic. State numbers as written."""


class FactGroup(BaseModel):
    fact_ids: list[str] = Field(min_length=2)
    classification: Classification
    severity: Severity = Severity.medium
    confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    title: str = Field(max_length=300)
    description: str
    materiality_reasoning: str | None = None


class FactGroups(BaseModel):
    groups: list[FactGroup] = Field(default_factory=list)


class MatchStats(BaseModel):
    facts: int = 0
    batches: int = 0
    llm_calls: int = 0
    groups_returned: int = 0
    findings_saved: int = 0
    findings_rejected: int = 0
    rejections: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


def _fact_line(f: RecordedFact) -> str:
    parts = [f"{f.fact_id} | {f.filename}, {f.locator} | {f.label}: {f.value}"]
    if f.unit:
        parts.append(f.unit)
    if f.qualifier:
        parts.append(f"(applies to: {f.qualifier})")
    return " ".join(parts) + f' | quote: "{f.quoted_text}"'


class FactMatcher:
    def __init__(self, llm: LLMClient, registry: ToolRegistry, max_chars_per_call: int = 60_000):
        self.llm = llm
        self.registry = registry
        self.max_chars = max_chars_per_call

    def _batches(self, facts: list[RecordedFact]):
        current: list[RecordedFact] = []
        size = 0
        for f in facts:
            n = len(_fact_line(f)) + 1
            if current and size + n > self.max_chars:
                yield current
                current, size = [], 0
            current.append(f)
            size += n
        if current:
            yield current

    def run(self) -> MatchStats:
        facts = list(self.registry.context.facts)
        stats = MatchStats(facts=len(facts))
        for batch in self._batches(facts):
            stats.batches += 1
            prompt = "Facts recorded from the documents:\n\n" + "\n".join(_fact_line(f) for f in batch)
            try:
                stats.llm_calls += 1
                groups = self.llm.generate_structured_output(prompt, FactGroups, system=SYSTEM)
            except LLMError as exc:
                stats.errors.append(str(exc)[:200])
                logger.warning("Fact matching batch failed", extra={"error": str(exc)})
                continue
            by_id = {f.fact_id: f for f in batch}
            for group in groups.groups:
                stats.groups_returned += 1
                self._save(group, by_id, stats)
        return stats

    def _save(self, group: FactGroup, by_id: dict[str, RecordedFact], stats: MatchStats) -> None:
        if group.classification not in REPORTED:
            return
        members = [by_id[i] for i in dict.fromkeys(group.fact_ids) if i in by_id]
        if len({m.chunk_id for m in members}) < 2:
            stats.findings_rejected += 1
            stats.rejections.append(f"'{group.title[:60]}': fewer than two known facts from different chunks")
            return
        role = "contradicting" if group.classification == Classification.contradiction else "contextual"
        result = self.registry.execute("submit_finding", {
            "title": group.title, "description": group.description, "classification": group.classification.value,
            "severity": group.severity.value, "confidence": group.confidence, "evidence_status": "adequate",
            "materiality_reasoning": group.materiality_reasoning,
            "evidence": [{
                "chunk_id": m.chunk_id, "quoted_text": m.quoted_text, "role": role,
                "relevance_reason": f"{m.label}: {m.value}" + (f" {m.unit}" if m.unit else ""),
                "support_assessment": f"{m.filename}, {m.locator}" + (f", applies to {m.qualifier}" if m.qualifier else ""),
            } for m in members][:12]})
        if not result.ok:
            stats.findings_rejected += 1
            stats.rejections.append(f"'{group.title[:60]}': {result.error.code}: {result.error.message[:160]}")
            return
        stats.findings_saved += 1
