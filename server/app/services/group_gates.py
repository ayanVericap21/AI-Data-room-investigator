"""Gates that remove candidates which cannot be a discrepancy, before the judge sees them.

Entity gate: two claims of documents about unrelated subjects (two companies, two projects) are not comparable, whatever their labels say.
One model call reads the document cards (see ``doc_profiles``) and groups the documents by the subject they are about; a candidate across two
unrelated groups is dropped. The gate is deliberately slow to separate: unsure documents stay together, a document without a profile is
compared with everything, and if the call fails nothing is dropped.

List-member guard: two name or statement claims on one page of one document are items of a list or sibling entries (themes, initiatives, stages,
team members), not rival values of one attribute, so such a candidate is dropped.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.llm.errors import LLMError
from app.models import Claim, DocumentChunk
from app.schemas.claim import Candidate
from app.services.doc_profiles import DocProfile, card

logger = logging.getLogger(__name__)

GATE_SYSTEM = (
    "You decide which documents of one collection are about the same subject, so that their claims may be compared. Put documents about the same subject "
    "(the same project, case, organisation, product, transaction or event) in one group, however different their types are (a permit, a dive log and a lab "
    "report about one expedition are one group). Put a document in a separate group only when it is clearly about a different, unrelated subject (another "
    "organisation, another project). When unsure, keep documents together. Every document is in exactly one group. In related, list the groups "
    "whose subjects are connected (a customer and its supplier, a company and its subsidiary, a regulator and the body it regulates, a project and its "
    "funder), so that their documents may still be compared. OUTPUT FORMAT (strict): one JSON object "
    '{"groups": [{"docs": ["d1", "d2"], "entity": "what the group is about"}], "related": [{"a": 1, "b": 2}]}, groups numbered from 1 in the order listed.')

NAME_OR_STATEMENT = {"name", "statement"}


class EntityGroup(BaseModel):
    docs: list[str] = Field(default_factory=list)
    entity: str = ""


class EntityLink(BaseModel):
    a: int = 0
    b: int = 0


class EntityGroups(BaseModel):
    groups: list[EntityGroup] = Field(default_factory=list)
    related: list[EntityLink] = Field(default_factory=list)


@dataclass
class EntityGate:
    applied: bool = False
    note: str = ""
    calls: int = 0
    groups: list[tuple[str, list[str]]] = field(default_factory=list)  # (what the group is about, file names)
    related: list[tuple[int, int]] = field(default_factory=list)
    doc_group: dict[str, int] = field(default_factory=dict)  # document id -> group number

    def allows(self, doc_a: str, doc_b: str) -> bool:
        ga, gb = self.doc_group.get(doc_a), self.doc_group.get(doc_b)
        if ga is None or gb is None or ga == gb:
            return True
        return (ga, gb) in self.related or (gb, ga) in self.related


def decide_entity_groups(llm: Any, entries: dict[str, tuple[str, DocProfile]]) -> EntityGate:
    """Group the documents that have a profile by the subject they are about (one call). Fewer than two profiled documents: no gate."""
    gate = EntityGate()
    if len(entries) < 2:
        gate.note = "fewer than two documents have a profile; nothing to separate"
        return gate
    refs = {f"d{i}": doc_id for i, doc_id in enumerate(sorted(entries, key=lambda d: entries[d][0]), start=1)}
    listing = "\n".join(f"{ref}: " + card(entries[doc_id][0], entries[doc_id][1])[2:] for ref, doc_id in refs.items())
    gate.calls = 1
    try:
        result = llm.generate_structured_output("Documents:\n" + listing, EntityGroups, system=GATE_SYSTEM)
    except LLMError as exc:
        gate.note = f"the grouping call failed ({str(exc)}); no candidate was dropped"
        return gate
    for number, group in enumerate(result.groups, start=1):
        members = [refs[r.strip()] for r in group.docs if r.strip() in refs and refs[r.strip()] not in gate.doc_group]
        if not members:
            continue
        for doc_id in members:
            gate.doc_group[doc_id] = number
        gate.groups.append((group.entity.strip() or "(not named)", [entries[d][0] for d in members]))
    kept_numbers = {n for n in gate.doc_group.values()}
    gate.related = [(link.a, link.b) for link in result.related if link.a in kept_numbers and link.b in kept_numbers and link.a != link.b]
    gate.applied = True
    unlisted = [entries[d][0] for d in entries if d not in gate.doc_group]
    gate.note = f"{len(gate.groups)} group(s), {len(gate.related)} related link(s)" + (f"; not listed by the model, compared with everything: {', '.join(unlisted)}" if unlisted else "")
    return gate


def list_members(a: Claim, b: Claim, chunk_a: DocumentChunk, chunk_b: DocumentChunk) -> bool:
    """Two name or statement claims on one page of one document: items of a list, not rival values of one attribute."""
    return (a.document_id == b.document_id and chunk_a.page_number is not None and chunk_a.page_number == chunk_b.page_number
            and a.claim_category in NAME_OR_STATEMENT and b.claim_category in NAME_OR_STATEMENT)


@dataclass
class GateResult:
    candidates: list[Candidate]
    entity_dropped: int = 0
    list_dropped: int = 0
    log: list[str] = field(default_factory=list)


def apply_gates(session: Session, candidates: list[Candidate], gate: EntityGate | None, list_guard: bool = True, log_limit: int = 40) -> GateResult:
    """The candidates that pass the entity gate and the list-member guard, with how many each gate removed."""
    ids = {p.claim_id for p in candidates} | {p.candidate_id for p in candidates}
    claims = {c.id: c for c in session.scalars(select(Claim).where(Claim.id.in_(ids)))} if ids else {}
    chunk_ids = {c.chunk_id for c in claims.values()}
    chunks = {c.id: c for c in session.scalars(select(DocumentChunk).where(DocumentChunk.id.in_(chunk_ids)))} if chunk_ids else {}
    result = GateResult(candidates=[])
    for p in candidates:
        a, b = claims.get(p.claim_id), claims.get(p.candidate_id)
        if a is None or b is None:
            result.candidates.append(p)
            continue
        if gate is not None and gate.applied and not gate.allows(a.document_id, b.document_id):
            result.entity_dropped += 1
            if len(result.log) < log_limit:
                result.log.append(f"entity gate: {a.subject!r} / {b.subject!r} ({a.metric!r}, {b.metric!r})")
            continue
        ca, cb = chunks.get(a.chunk_id), chunks.get(b.chunk_id)
        if list_guard and p.matched_by != "rule" and ca is not None and cb is not None and list_members(a, b, ca, cb):
            result.list_dropped += 1
            if len(result.log) < log_limit:
                result.log.append(f"list members: {a.value!r} / {b.value!r} ({a.metric!r}) on page {ca.page_number}")
            continue
        result.candidates.append(p)
    return result
