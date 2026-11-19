"""Splitting a set of claims that is too large to be one fact into smaller clusters by how alike the claims are.

A cluster the grouping model made (or a set of claims the code chained together) can hold dozens of claims. The judge reads about fifteen at a time. A larger set
is divided by meaning into clusters: claims are joined, the most alike first, into clusters that never exceed the cap, and a cluster stops growing when the claims
that remain are not alike enough. Only clusters come out of it; a claim that is alike to nothing stays alone and is left out. The similarity is the cosine of the claims' embeddings (subject, metric and sentence), the same vectors the matching already uses.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

SPLIT_FLOOR = 0.5  # two clusters are joined only while their claims are this alike on average
COHERENCE_FLOOR = 0.4  # a cluster of the grouping model whose claims are less alike than this on average is divided even when it is small
MAX_AVERAGE_LINKAGE = 1200  # above this many claims, joining by the average over all the claims of two clusters is too slow, and the strongest single match is used


def split_by_similarity(vectors: np.ndarray, cap: int, floor: float = SPLIT_FLOOR) -> list[list[int]]:
    """Clusters of row indexes, none larger than ``cap``: the two clusters whose claims are on average the most alike are joined, again and again, until the
    the most alike two clusters are below ``floor`` or none of them fit under the cap together. Claims that are alike to nothing stay alone."""
    n = len(vectors)
    if n == 0:
        return []
    similarity = vectors @ vectors.T
    if n == 1:
        return [[0]]
    if n > MAX_AVERAGE_LINKAGE:
        return _strongest_match(similarity, cap, floor)
    total = similarity.astype(np.float64).copy()  # the sum of the similarities between the claims of cluster i and of cluster j
    np.fill_diagonal(total, 0.0)
    size = np.ones(n)
    members: list[list[int]] = [[i] for i in range(n)]
    alive = np.ones(n, dtype=bool)
    while True:
        average = total / np.outer(size, size)
        allowed = alive[:, None] & alive[None, :] & ((size[:, None] + size[None, :]) <= cap)
        np.fill_diagonal(allowed, False)
        average = np.where(allowed, average, -np.inf)
        best = int(np.argmax(average))
        i, j = divmod(best, n)
        if average[i, j] < floor:
            break
        total[i, :] += total[j, :]
        total[:, i] = total[i, :]
        total[i, i] = 0.0
        total[j, :] = 0.0
        total[:, j] = 0.0
        size[i] += size[j]
        members[i] += members[j]
        alive[j] = False
    return [sorted(members[i]) for i in range(n) if alive[i]]


def _strongest_match(similarity: np.ndarray, cap: int, floor: float) -> list[list[int]]:
    """The same for a very large set: the strongest matches first, joining two clusters when together they fit under the cap."""
    n = len(similarity)
    parent = list(range(n))
    size = [1] * n

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    rows, cols = np.where(np.triu(similarity >= floor, k=1))
    for k in np.argsort(-similarity[rows, cols], kind="stable"):
        a, b = find(int(rows[k])), find(int(cols[k]))
        if a != b and size[a] + size[b] <= cap:
            parent[b] = a
            size[a] += size[b]
    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)
    return list(clusters.values())


class ClaimSimilarity:
    """How alike claims are, from their embeddings, computed when first asked for and kept for the rest of the run."""

    def __init__(self, embedder: Any, claims: Mapping[str, Any]):
        self._embedder, self._claims = embedder, claims
        self._vectors: dict[str, np.ndarray] = {}

    def _load(self, ids: list[str]) -> None:
        missing = [i for i in dict.fromkeys(ids) if i not in self._vectors]
        if not missing:
            return
        texts = [f"{self._claims[i].subject or ''}: {self._claims[i].metric or ''}: {self._claims[i].claim_text or self._claims[i].value or ''}" for i in missing]
        vectors = np.asarray(self._embedder.embed(texts, task="document"), dtype=np.float64)
        norms = np.linalg.norm(vectors, axis=1)
        vectors = vectors / np.where(norms == 0.0, 1.0, norms)[:, None]
        self._vectors.update(zip(missing, vectors))

    def matrix(self, ids: list[str]) -> np.ndarray:
        self._load(ids)
        return np.stack([self._vectors[i] for i in ids])

    def coherence(self, ids: list[str]) -> float:
        """The average similarity between each claim and each of the other claims."""
        if len(ids) < 2:
            return 1.0
        vectors = self.matrix(ids)
        total = vectors @ vectors.T
        n = len(ids)
        return float((total.sum() - np.trace(total)) / (n * (n - 1)))

    def split(self, ids: list[str], cap: int, floor: float = SPLIT_FLOOR) -> list[list[str]]:
        """``ids`` divided into clusters of at most ``cap`` claims, the most alike together (see ``split_by_similarity``)."""
        return [[ids[i] for i in part] for part in split_by_similarity(self.matrix(ids), cap, floor)]
