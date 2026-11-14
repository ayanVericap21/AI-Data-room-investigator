"""Candidate claims shown as groups: a fact, the claims that may state it, and for each claim the claims it may be compared with."""
from __future__ import annotations

from app.schemas.claim import Candidate, CandidateLink, ClaimGroup, ClaimOut, GroupedClaim


def group_candidates(candidates: list[Candidate], claims: dict[str, ClaimOut]) -> list[ClaimGroup]:
    """Claims that are candidates of one another, directly or through other claims, form one group, in order of first appearance. A candidate
    whose claim is not in ``claims`` (its document is gone) is left out."""
    candidates = [c for c in candidates if c.claim_id in claims and c.candidate_id in claims]
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for p in candidates:
        parent[find(p.claim_id)] = find(p.candidate_id)
    members: dict[str, list[str]] = {}
    links: dict[str, list[CandidateLink]] = {}
    for p in candidates:
        for claim, other in ((p.claim_id, p.candidate_id), (p.candidate_id, p.claim_id)):
            group = members.setdefault(find(claim), [])
            if claim not in group:
                group.append(claim)
            links.setdefault(claim, []).append(CandidateLink(claim=claims[other], relation_hint=p.relation_hint, topic_score=p.topic_score, matched_by=p.matched_by))
    return [ClaimGroup(claims=[GroupedClaim(claim=claims[c], candidates=links[c]) for c in ids]) for ids in members.values()]
