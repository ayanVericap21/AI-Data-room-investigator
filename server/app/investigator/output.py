"""Consolidation of verified candidates by root cause, and saving them as findings.

Consolidation asks the model which candidates are symptoms of one underlying cause (it sees titles, entities and evidence places, and a shared entity alone
is not a reason to group), and merges each group into one finding with the symptoms linked. Saving maps the new taxonomy onto the existing classification
column (no schema change): the taxonomy label is the start of the title, and the status (definitive or needs investigation), the check family, the
computation and the verifier's reasons are written into the description.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.investigator.models import FAMILIES, SEVERITY_ORDER, TAXONOMY, Candidate, ConsolidationPlan
from app.models import Evidence, Finding
from app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

GROUP_SYSTEM = (
    "You consolidate findings from a data room investigation. Each numbered candidate is one finding. Group together candidates that are symptoms of the same underlying cause, "
    "for example several records that all follow from one lost instrument, or one document error repeated in two places. Members of a group must share the cause, not merely the same "
    "site, person, file or identifier: two different problems about one site stay separate. A group has at least two members. For each group give a title for the combined finding and "
    "a description of how the symptoms relate. Candidates that belong to no group are not listed.")


@dataclass
class ConsolidationStats:
    before: int = 0
    after: int = 0
    exact_duplicates: int = 0
    groups: int = 0
    calls: int = 0
    log: list[str] = field(default_factory=list)


def _rank(c: Candidate) -> tuple:
    return (c.source != "pairing", SEVERITY_ORDER.index(c.severity) if c.severity in SEVERITY_ORDER else 0, c.confidence)


def _merge(primary: Candidate, others: list[Candidate], title: str = "", description: str = "") -> Candidate:
    seen = {(e.chunk_id, " ".join(e.quote.split())) for e in primary.evidence}
    for o in others:
        for e in o.evidence:
            key = (e.chunk_id, " ".join(e.quote.split()))
            if key not in seen and len(primary.evidence) < 12:
                seen.add(key)
                primary.evidence.append(e)
        primary.linked.append(o.title)
        primary.linked += o.linked
        primary.entities = list(dict.fromkeys(primary.entities + o.entities))
        if o.finding_id and o.finding_id != primary.finding_id:
            primary.absorbed.append(o.finding_id)
        primary.absorbed += o.absorbed
        if SEVERITY_ORDER.index(o.severity) > SEVERITY_ORDER.index(primary.severity):
            primary.severity = o.severity
        if o.status == "needs_investigation":
            primary.status = "needs_investigation"
        primary.evidence_needed = list(dict.fromkeys(primary.evidence_needed + o.evidence_needed))
    if title.strip():
        primary.title = title.strip()[:260]
    if description.strip():
        primary.description = description.strip() + "\n\nPrimary symptom: " + primary.description
    return primary


class Consolidator:
    def __init__(self, llm: Any, batch_size: int = 60, on_progress: Callable[[str], None] | None = None):
        self.llm = llm
        self.batch_size = batch_size
        self.on_progress = on_progress or (lambda _m: None)

    def run(self, candidates: list[Candidate]) -> tuple[list[Candidate], ConsolidationStats]:
        stats = ConsolidationStats(before=len(candidates))
        by_key: dict[tuple, Candidate] = {}
        kept: list[Candidate] = []
        for c in sorted(candidates, key=_rank, reverse=True):
            key = (c.type, c.key())
            if c.evidence and key in by_key:
                stats.exact_duplicates += 1
                _merge(by_key[key], [c])
                continue
            by_key[key] = c
            kept.append(c)
        ordered = sorted(kept, key=lambda c: ((c.entities[0].lower() if c.entities else "~"), c.family))
        merged_away: set[int] = set()
        for start in range(0, len(ordered), self.batch_size):
            batch = ordered[start:start + self.batch_size]
            if len(batch) < 2:
                continue
            prompt = "Candidates:\n" + "\n".join(
                f"{n}. [{c.type.replace('_', ' ')}, family {c.family}] {c.title} | entities: {', '.join(c.entities[:5]) or '-'} | evidence: "
                + "; ".join(sorted({f'{e.filename} {e.location}' for e in c.evidence}))[:200] + f" | {c.description[:180]}" for n, c in enumerate(batch, start=1))
            try:
                plan = self.llm.generate_structured_output(prompt, ConsolidationPlan, system=GROUP_SYSTEM)
            except Exception as exc:
                stats.log.append(f"grouping call failed ({type(exc).__name__}); candidates left separate")
                continue
            stats.calls += 1
            used: set[int] = set()
            for g in plan.groups:
                members = [m for m in dict.fromkeys(g.members) if 1 <= m <= len(batch) and m not in used]
                if len(members) < 2:
                    continue
                used.update(members)
                cands = [batch[m - 1] for m in members]
                primary = max(cands, key=_rank)
                others = [c for c in cands if c is not primary]
                _merge(primary, others, g.title, g.description)
                merged_away.update(id(o) for o in others)
                stats.groups += 1
                stats.log.append(f"{len(cands)} findings became one: {primary.title[:100]} (merged: {'; '.join(o.title[:60] for o in others)})")
        final = [c for c in kept if id(c) not in merged_away]
        stats.after = len(final)
        return final, stats


@dataclass
class SavedItem:
    finding_id: str
    candidate: Candidate
    classification: str


class Saver:
    def __init__(self, session: Session, registry: ToolRegistry):
        self.session = session
        self.registry = registry
        self.rejected: list[str] = []

    @staticmethod
    def classify(c: Candidate) -> tuple[str, str]:
        label, two, one = TAXONOMY.get(c.type, TAXONOMY["direct_contradiction"])
        passages = len({(e.chunk_id, " ".join(e.quote.split())) for e in c.evidence})
        return label, (two if passages >= 2 else one)

    @staticmethod
    def description(c: Candidate) -> str:
        label = TAXONOMY.get(c.type, TAXONOMY["direct_contradiction"])[0]
        head = f"Status: {'definitive' if c.status == 'definitive' else 'needs investigation'}. Type: {label}. Check family {c.family}: {FAMILIES.get(c.family, '')}."
        parts = [head, c.description.strip()]
        if c.computed.strip():
            parts.append("Computed: " + c.computed.strip())
        if c.linked:
            parts.append("Related symptoms merged here: " + "; ".join(dict.fromkeys(c.linked)) + ".")
        if c.verifier_notes.strip():
            parts.append("Verification: " + c.verifier_notes.strip())
        return "\n\n".join(parts)

    def save(self, candidates: list[Candidate]) -> list[SavedItem]:
        saved: list[SavedItem] = []
        for c in candidates:
            label, classification = self.classify(c)
            title = c.title if c.title.lower().startswith(f"[{label}") else f"[{label}] {c.title}"
            if c.finding_id:  # a family I finding already in the database: relabel it
                f = self.session.get(Finding, c.finding_id)
                if f is not None:
                    f.title, f.description = title[:300], self.description(c)
                    f.severity = c.severity
                    f.unresolved_questions = list(c.evidence_needed)
                    self.session.commit()
                    saved.append(SavedItem(f.id, c, f.classification))
                    continue
            evidence_status = "unavailable" if c.type == "missing_evidence" else "adequate" if c.status == "definitive" else "partial"
            result = self.registry.execute("submit_finding", {
                "title": title[:300], "description": self.description(c), "classification": classification, "severity": c.severity, "confidence": c.confidence,
                "evidence_status": evidence_status, "materiality_reasoning": c.verifier_notes[:500] or None,
                "unresolved_questions": list(c.evidence_needed) if (c.status != "definitive" or c.type in ("missing_evidence", "reconciliation_gap")) else [],
                "evidence": [{"chunk_id": e.chunk_id, "quoted_text": e.quote, "role": e.role if e.role in ("supporting", "contradicting", "contextual", "insufficient") else "supporting",
                              "relevance_reason": (e.shows or c.title)[:500], "support_assessment": f"{e.filename}, {e.location}: {e.shows or 'see the quote'}"[:600]} for e in c.evidence[:12]],
            })
            if not result.ok:
                self.rejected.append(f"{title[:70]}: {result.error.code}: {result.error.message[:140]}")
                continue
            c.finding_id = result.output.finding_id
            saved.append(SavedItem(result.output.finding_id, c, classification))
        for c in candidates:  # findings that were merged into another are removed
            for fid in c.absorbed:
                self.remove(fid)
        return saved

    def remove(self, finding_id: str) -> None:
        self.session.execute(delete(Evidence).where(Evidence.finding_id == finding_id))
        gone = self.session.get(Finding, finding_id)
        if gone is not None:
            self.session.delete(gone)
        self.session.commit()
