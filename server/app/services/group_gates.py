"""Gates that remove claims which cannot be a discrepancy from the clusters, before the judge sees them.

Entity gate: claims of documents about unrelated subjects (two companies, two projects) are not comparable, whatever their labels say. One model call
reads the document cards (see ``doc_profiles``) and groups the documents by the subject they are about. The entity group that holds most of the documents
stays; a document of another group is dropped with its claims right after the call (see ``EntityGate.dropped``). Without a majority nothing is dropped, and
a cluster whose claims come from unrelated groups loses the claims of the groups that do not belong with the rest. The gate is deliberately slow to
separate: unsure documents stay together, a document without a profile is compared with everything, and if the call fails nothing is dropped.

List-member guard: name or statement claims on one page of one document are items of a list or sibling entries (themes, initiatives, stages, team members),
not rival values of one attribute, so they are not kept together in one cluster.

Fragment guard: a claim whose text value is only a piece of another claim's value in the same cluster ("not" next to "have not been sold, assigned or
pledged to any other person") is an extraction cut, not a rival value: it is left out of the cluster, since the judge would call the pair a contradiction.
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
import re
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.llm.errors import LLMError
from app.models import Claim, DocumentChunk
from app.schemas.claim import Cluster
from app.services.clusters import connected_clusters
from app.services.doc_profiles import DocProfile, card
from app.services.numbers import parse_number
from app.services.value_rules import parse_date, parse_numeric

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


class RelatedGroups(BaseModel):
    a: int = 0
    b: int = 0


class EntityGroups(BaseModel):
    groups: list[EntityGroup] = Field(default_factory=list)
    related: list[RelatedGroups] = Field(default_factory=list)


@dataclass
class EntityGate:
    applied: bool = False
    note: str = ""
    calls: int = 0
    groups: list[tuple[str, list[str]]] = field(default_factory=list)  # (what the group is about, file names)
    related: list[tuple[int, int]] = field(default_factory=list)
    doc_group: dict[str, int] = field(default_factory=dict)  # document id -> group number
    names: dict[int, str] = field(default_factory=dict)  # group number -> what the group is about
    main: int | None = None  # the group that holds most of the documents (more than half), if there is one
    dropped: list[tuple[str, str, str]] = field(default_factory=list)  # (document id, file name, what it is about): documents of an unrelated entity

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
        gate.note = f"the grouping call failed ({str(exc)}); no claim was dropped"
        return gate
    for number, group in enumerate(result.groups, start=1):
        members = [refs[r.strip()] for r in group.docs if r.strip() in refs and refs[r.strip()] not in gate.doc_group]
        if not members:
            continue
        for doc_id in members:
            gate.doc_group[doc_id] = number
        gate.groups.append((group.entity.strip() or "(not named)", [entries[d][0] for d in members]))
        gate.names[number] = gate.groups[-1][0]
    kept_numbers = {n for n in gate.doc_group.values()}
    gate.related = [(rel.a, rel.b) for rel in result.related if rel.a in kept_numbers and rel.b in kept_numbers and rel.a != rel.b]
    gate.applied = True
    sizes = Counter(gate.doc_group.values())
    gate.main = next((n for n, size in sizes.items() if size * 2 > len(entries)), None)  # the entity that holds most of the data room stays
    if gate.main is not None and len(sizes) > 1:
        keep = {gate.main} | {b if a == gate.main else a for a, b in gate.related if gate.main in (a, b)}  # groups the model marked as related to it stay too
        gate.dropped = [(doc_id, entries[doc_id][0], gate.names[n]) for doc_id, n in gate.doc_group.items() if n not in keep]
    unlisted = [entries[d][0] for d in entries if d not in gate.doc_group]
    gate.note = f"{len(gate.groups)} group(s), {len(gate.related)} related group(s)" + (f"; not listed by the model, compared with everything: {', '.join(unlisted)}" if unlisted else "")
    if len(sizes) > 1 and gate.main is None:
        gate.note += "; no group holds most of the documents, so no document was dropped"
    elif gate.dropped:
        gate.note += f"; {len(gate.dropped)} document(s) about another entity than the rest are dropped: {', '.join(name for _, name, _ in gate.dropped)}"
    return gate


def list_members(a: Claim, b: Claim, chunk_a: DocumentChunk, chunk_b: DocumentChunk) -> bool:
    """Two name or statement claims on one page of one document: items of a list, not rival values of one attribute."""
    return (a.document_id == b.document_id and chunk_a.page_number is not None and chunk_a.page_number == chunk_b.page_number
            and a.claim_category in NAME_OR_STATEMENT and b.claim_category in NAME_OR_STATEMENT)


@dataclass
class GateResult:
    clusters: list[Cluster]
    entity_dropped: int = 0  # claims taken out of a cluster because their document is about an unrelated entity
    fragment_dropped: int = 0  # claims left out of a cluster because their value is only a piece of another claim's value
    list_dropped: int = 0  # claims left out of a cluster because they are list members of the cluster's other claims
    clusters_dropped: int = 0  # clusters that fell below two claims
    log: list[str] = field(default_factory=list)


def _anchor_document(cluster: Cluster, claims: dict[str, Claim], gate: EntityGate) -> str | None:
    """The document the other claims of a cluster are measured against: the rule's own document for a rule cluster, else a document of the entity group
    most of the cluster's claims belong to."""
    if cluster.rule_id and cluster.rule_id in claims:
        return claims[cluster.rule_id].document_id
    docs = [claims[i].document_id for i in cluster.claim_ids if i in claims and gate.doc_group.get(claims[i].document_id) is not None]
    if not docs:
        return None
    main = Counter(gate.doc_group[d] for d in docs).most_common(1)[0][0]
    return next(d for d in docs if gate.doc_group[d] == main)


MAX_FRAGMENT_WORDS = 3  # a fragment is at most this many words
FRAGMENT_EXTRA_WORDS = 2  # and the value it is a piece of has at least this many more


def _words(value: object) -> list[str]:
    return re.findall(r"[^\W_]+", str(value).lower())


def is_fragment_of(a: Claim, b: Claim) -> bool:
    """``a``'s value is a few words that appear, in order and side by side, inside the longer text value of ``b``. A value with a digit in it, a number or a date is never a fragment."""
    if a.value is None or b.value is None or isinstance(a.value, (int, float)) or isinstance(b.value, (int, float)):
        return False
    if re.search(r"\d", str(a.value)) or parse_number(a.value) is not None or parse_numeric(a.value) is not None or parse_date(a.value) is not None:
        return False  # a figure or a date, written in any form ("March 28"), is a value of its own
    wa, wb = _words(a.value), _words(b.value)
    if not wa or len(wa) > MAX_FRAGMENT_WORDS or len(wb) < len(wa) + FRAGMENT_EXTRA_WORDS:
        return False
    return any(wb[i:i + len(wa)] == wa for i in range(len(wb) - len(wa) + 1))


def apply_gates(session: Session, clusters: list[Cluster], gate: EntityGate | None, list_guard: bool = True, log_limit: int = 40) -> GateResult:
    """The clusters that pass the entity gate and the list-member guard: each loses the claims the gates remove, and one that falls below two claims is
    dropped. The result counts what each gate removed."""
    ids = {i for c in clusters for i in c.claim_ids}
    claims = {c.id: c for c in session.scalars(select(Claim).where(Claim.id.in_(ids)))} if ids else {}
    chunk_ids = {c.chunk_id for c in claims.values()}
    chunks = {c.id: c for c in session.scalars(select(DocumentChunk).where(DocumentChunk.id.in_(chunk_ids)))} if chunk_ids else {}
    result = GateResult(clusters=[])

    def note(text: str) -> None:
        if len(result.log) < log_limit:
            result.log.append(text)

    for cluster in clusters:
        members = list(cluster.claim_ids)
        if gate is not None and gate.applied:
            anchor = _anchor_document(cluster, claims, gate)
            kept = [i for i in members if i not in claims or anchor is None or gate.allows(anchor, claims[i].document_id)]
            for i in members:
                if i not in kept:
                    result.entity_dropped += 1
                    note(f"entity gate: {claims[i].subject!r} ({claims[i].metric!r}) of a document about another entity left out of its cluster")
            members = kept
        if len(members) < 2:
            result.clusters_dropped += 1
            continue
        fragments = [i for i in members if i in claims and any(j != i and j in claims and is_fragment_of(claims[i], claims[j]) for j in members)]
        if fragments:
            for i in fragments:
                result.fragment_dropped += 1
                note(f"fragment: {claims[i].value!r} ({claims[i].metric!r}) is only a piece of another claim's value in its cluster, left out")
            members = [i for i in members if i not in fragments]
            if len(members) < 2:
                result.clusters_dropped += 1
                continue
        pieces = [members]
        if list_guard and cluster.rule_id is None:
            # claims that are list members of each other do not go together: the cluster is what the other claims still join
            together = [(a, b, 1.0) for n, a in enumerate(members) for b in members[n + 1:]
                        if not (a in claims and b in claims and claims[a].chunk_id in chunks and claims[b].chunk_id in chunks
                                and list_members(claims[a], claims[b], chunks[claims[a].chunk_id], chunks[claims[b].chunk_id]))]
            if len(together) < len(members) * (len(members) - 1) // 2:
                pieces = [c.claim_ids for c in connected_clusters(together, cluster.matched_by)]
                inside = {i for piece in pieces for i in piece}
                for i in members:
                    if i not in inside:
                        result.list_dropped += 1
                        if i in claims:
                            note(f"list members: {claims[i].value!r} ({claims[i].metric!r}) left out of its cluster, on page {chunks[claims[i].chunk_id].page_number}")
        if not pieces:
            result.clusters_dropped += 1
        for piece in pieces:
            if len(piece) < 2:
                result.clusters_dropped += 1
            else:
                result.clusters.append(cluster.model_copy(update={"claim_ids": piece, "rule_id": cluster.rule_id if cluster.rule_id in piece else None,
                                                                  "via": {i: cluster.via[i] for i in piece if i in cluster.via}}))
    return result
