"""Clusters of claims shown as groups: a fact and the claims that may state it."""
from __future__ import annotations

from app.schemas.claim import ClaimGroup, ClaimOut, Cluster
from app.services.clusters import merge_overlapping


def group_clusters(clusters: list[Cluster], claims: dict[str, ClaimOut]) -> list[ClaimGroup]:
    """Clusters that share a claim, directly or through other clusters, form one group, in order of first appearance. A claim that is not in ``claims``
    (its document is gone) is left out, and a group left with fewer than two claims is dropped."""
    shown = [[i for i in c.claim_ids if i in claims] for c in clusters]
    first: dict[str, Cluster] = {}
    for cluster, ids in zip(clusters, shown):
        for claim_id in ids:
            first.setdefault(claim_id, cluster)
    groups = []
    for ids in merge_overlapping([ids for ids in shown if len(ids) >= 2]):
        cluster = first[ids[0]]
        groups.append(ClaimGroup(claims=[claims[i] for i in ids], topic=cluster.topic, matched_by=cluster.matched_by))
    return groups
