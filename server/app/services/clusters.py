"""Clusters of claims: what every route (the grouping model, label and meaning matching, record ids, rules) hands to the judge.

A route may compare claims with each other inside its own code to decide what goes together; what it returns is always a cluster, the claims that go
together under that route.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from app.schemas.claim import Cluster


class _Sets:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def join(self, ids: Iterable[str]) -> None:
        ids = list(ids)
        for other in ids[1:]:
            self.parent[self.find(other)] = self.find(ids[0])
        if ids:
            self.find(ids[0])


def merge_overlapping(id_lists: list[list[str]]) -> list[list[str]]:
    """Lists of claim ids that share a claim are one set; the sets keep the order in which their claims first appear."""
    sets = _Sets()
    for ids in id_lists:
        sets.join(ids)
    out: dict[str, list[str]] = defaultdict(list)
    for ids in id_lists:
        for claim_id in ids:
            members = out[sets.find(claim_id)]
            if claim_id not in members:
                members.append(claim_id)
    return list(out.values())


def clusters_from_matches(matches: Iterable[tuple[str, str, float]], how: str, topic: str = "") -> list[Cluster]:
    """What a route found when it compared claims with each other, as (claim, claim, score): each claim with the claims it matched is one cluster, the best
    score of its matches. A cluster that another one contains is dropped. Clusters that share a claim overlap: the judge's group forming joins them.
    Used by the routes that match by labels or by meaning, where a claim matches only its nearest ones and the matches must not chain into one large set."""
    near: dict[str, dict[str, float]] = defaultdict(dict)
    for a, b, score in matches:
        near[a][b] = max(score, near[a].get(b, 0.0))
        near[b][a] = max(score, near[b].get(a, 0.0))
    stars = {claim_id: frozenset([claim_id, *near[claim_id]]) for claim_id in sorted(near)}
    holders: dict[str, list[str]] = defaultdict(list)
    for claim_id, members in stars.items():
        for member in members:
            holders[member].append(claim_id)
    out: list[Cluster] = []
    for claim_id, members in stars.items():
        # a cluster that another one contains (or equals, and comes first) adds nothing
        if any(other != claim_id and members <= stars[other] and (members < stars[other] or other < claim_id) for other in holders[claim_id]):
            continue
        out.append(Cluster(claim_ids=[claim_id, *sorted(near[claim_id])], topic=topic, matched_by=how, topic_score=round(max(near[claim_id].values()), 4)))
    return out


def connected_clusters(matches: Iterable[tuple[str, str, float]], how: str, topic: str = "") -> list[Cluster]:
    """Claims that match each other, directly or through other claims, are one cluster (a set of claims that all state one thing, such as the claims of one
    grouping-model group or of one record id): the best score of its matches. The size is not limited here: the judge divides a set that is too large."""
    matches = list(matches)
    sets = _Sets()
    for a, b, _score in matches:
        sets.join([a, b])
    members: dict[str, list[str]] = defaultdict(list)
    best: dict[str, float] = {}
    for a, b, score in sorted(matches, key=lambda m: (m[0], m[1])):
        root = sets.find(a)
        for claim_id in (a, b):
            if claim_id not in members[root]:
                members[root].append(claim_id)
        best[root] = max(best.get(root, 0.0), score)
    return [Cluster(claim_ids=ids, topic=topic, matched_by=how, topic_score=round(best[root], 4)) for root, ids in members.items()]
