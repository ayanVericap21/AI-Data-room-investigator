"""Key grouping: clusters made by a lookup on the claims' labels, not by asking a model which claims belong together.

Every claim is labelled on its own (``claim_labels``: subject, attribute, value type, basis, rule or record); the attribute names are merged into one
vocabulary in a single call. A claim's key is (canonical attribute, value type): money is only grouped with money, counts with counts, dates with dates.
The claims that share a key are one group. So the result depends on the claims and the labels only, never on how claims were batched.

- A claim with other attributes (``other_attributes``) is in the group of each key it fits: the clusters overlap, which keeps a pair from being missed
  because one claim has to choose. The judge treats each cluster on its own and merges findings that repeat across documents.
- A group larger than ``max_group_claims`` is divided by subject (subjects in alphabetical order, one subject kept whole), never by document order; a
  subject larger than the cap is cut in the order of its values, so equal values stay next to each other.
- The code's gates still apply inside a group: rows of one table and claims about different records do not go together (``ClaimGrouper._together``).
- A rule (flagged by the extraction or labelled "rule") joins the records of its key. A cluster with exactly one rule is a rule cluster.
"""
from __future__ import annotations

import logging
from collections import Counter
from itertools import combinations
from typing import Callable

from sqlalchemy.orm import Session

from app.llm.base import LLMClient
from app.llm.embeddings import EmbeddingProvider
from app.schemas.claim import ClaimView, Cluster
from app.services.claim_grouping import ClaimGrouper, GroupingStats, _value_key, ordered_rows
from app.services.claim_labels import (LABEL_INSTRUCTIONS, VOCABULARY_INSTRUCTIONS, ClaimLabel, ClaimLabeler, LabelCache, LabelStats, digest, merge_vocabulary)
from app.services.claim_tags import is_rule
from app.services.clusters import connected_clusters

logger = logging.getLogger(__name__)
DEFAULT_MAX_GROUP_CLAIMS = 15  # the judge's own cap (claim_group_judge.MAX_GROUP_CLAIMS): a larger set is not one fact


class KeyGrouper:
    def __init__(self, session: Session, llm: LLMClient, session_factory: Callable[[], Session], max_group_claims: int = DEFAULT_MAX_GROUP_CLAIMS,
                 batch_size: int = 40, workers: int = 4, refresh: bool = False, should_stop: Callable[[], bool] | None = None,
                 on_progress: Callable[[str], None] | None = None, embedder: EmbeddingProvider | None = None):
        self.session = session
        self.embedder = embedder  # the code's own matching of the claims no key cluster took uses it (nearest neighbours by meaning)
        self.llm = llm
        self.cap = max(2, max_group_claims)
        self.batch_size = batch_size
        self.workers = workers
        self.should_stop = should_stop or (lambda: False)
        self.on_progress = on_progress or (lambda _m: None)
        self.label_cache = LabelCache(session_factory, digest(LABEL_INSTRUCTIONS), refresh=refresh)
        self.vocabulary_cache = LabelCache(session_factory, digest(VOCABULARY_INSTRUCTIONS), refresh=refresh)
        self.labels: dict[str, ClaimLabel] = {}  # claim id -> its label with the canonical attribute (kept for the report and for scoring)
        self.label_stats = LabelStats()

    def run(self) -> tuple[list[Cluster], GroupingStats]:
        views = [row[0] for row in ordered_rows(self.session)]
        stats = GroupingStats(claims=len(views))
        labels, label_stats = ClaimLabeler(self.llm, self.label_cache, self.batch_size, self.workers, self.should_stop, self.on_progress).label(views)
        counts = Counter(name for label in labels.values() for name in [label.attribute, *label.other_attributes] if name)
        self.on_progress(f"merging {len(counts)} attribute names")
        mapping = {} if self.should_stop() else merge_vocabulary(self.llm, self.vocabulary_cache, dict(counts), label_stats)
        self.labels = {i: label.model_copy(update={"attribute": mapping.get(label.attribute, label.attribute),
                                                   "other_attributes": [o for o in dict.fromkeys(mapping.get(x, x) for x in label.other_attributes) if o != mapping.get(label.attribute, label.attribute)]})
                       for i, label in labels.items()}
        label_stats.attributes = len(counts)
        label_stats.attributes_merged = len({mapping.get(n, n) for n in counts})
        self.label_stats = label_stats
        stats.errors += label_stats.errors + ([self.label_cache.error] if self.label_cache.error else [])
        stats.llm_calls = label_stats.calls + (0 if label_stats.vocabulary_cached or len(counts) < 2 else 1)
        stats.batches = label_stats.calls
        stats.labelled_cached, stats.labelled_asked, stats.labelled_fallback = label_stats.cached, label_stats.asked, label_stats.fallback
        stats.attributes, stats.attributes_merged = label_stats.attributes, label_stats.attributes_merged

        rules_among = {v.id for v in views if is_rule(v.assumptions) or self.labels[v.id].kind == "rule"}
        by_key: dict[tuple[str, str], list[ClaimView]] = {}
        unkeyed = 0
        for v in views:
            label = self.labels[v.id]
            names = [n for n in dict.fromkeys([label.attribute, *label.other_attributes]) if n]
            unkeyed += not names
            for name in names:
                by_key.setdefault((name, label.value_type), []).append(v)
        stats.key_groups = sum(len(m) >= 2 for m in by_key.values())

        found: list[Cluster] = []
        seen: set[frozenset[str]] = set()
        for (attribute, value_type), members in sorted(by_key.items()):
            if len(members) < 2:
                continue
            for piece in self._pieces(members):
                matches = [(a.id, b.id, 1.0) for a, b in combinations(piece, 2) if ClaimGrouper._together(a, b)]
                for cluster in connected_clusters(matches, "key", f"{attribute} ({value_type})"):
                    ids = frozenset(cluster.claim_ids)
                    if ids in seen:
                        continue
                    seen.add(ids)
                    found.append(self._with_rule(cluster, rules_among))
        # the claims no key cluster took (no label, or a key nothing else shares) go through the code's own matching of the other grouping stage: windows,
        # negations, record ids, shared words, rare label words, nearest neighbours. Its clusters only add to the key clusters; a claim a key cluster took is left alone.
        taken = {i for c in found for i in c.claim_ids}
        key_clusters = len(found)

        def add(cluster: Cluster, via: str | None = None) -> bool:
            ids = [i for i in cluster.claim_ids if i not in taken]
            if len(ids) < 2 or (cluster.rule_id is not None and cluster.rule_id not in ids):
                return False
            found.append(cluster.model_copy(update={"claim_ids": ids, "via": {i: via or cluster.via.get(i, cluster.matched_by) for i in ids}}))
            taken.update(ids)
            return True

        if not self.should_stop():
            self.on_progress(f"code matching of the {len(views) - len(taken)} claims no key cluster took")
            ClaimGrouper(self.session, self.llm, embedder=self.embedder, should_stop=self.should_stop).code_matching(views, taken, add, stats)
        stats.clusters = len(found)
        stats.rule_clusters = sum(c.rule_id is not None for c in found)
        stats.model_clusters = key_clusters - sum(c.rule_id is not None for c in found[:key_clusters])  # key clusters without a rule (the field name is shared with the LLM grouping)
        stats.claims_by_code = sum(len(c.claim_ids) for c in found[key_clusters:])
        in_cluster = {i for c in found for i in c.claim_ids}
        stats.claims_grouped = len(in_cluster)
        stats.claims_left = len(views) - len(in_cluster)
        stats.claims_by_model = sum(len(c.claim_ids) for c in found[:key_clusters] if c.rule_id is None)
        stats.claims_by_rule = sum(len(c.claim_ids) for c in found if c.rule_id is not None)
        stats.largest_group = max((len(c.claim_ids) for c in found), default=0)
        stats.groups = len(found)
        stats.distinct_values = 0
        logger.info("Key grouping done", extra={"claims": len(views), "keys": len(by_key), "clusters": len(found), "unkeyed": unkeyed})
        return found, stats

    def _pieces(self, members: list[ClaimView]) -> list[list[ClaimView]]:
        """A group as the clusters of at most ``cap`` claims it is judged in: divided by subject (alphabetical, one subject kept whole while it fits),
        and a subject larger than the cap by its values."""
        if len(members) <= self.cap:
            return [members]
        buckets: dict[str, list[ClaimView]] = {}
        for v in members:
            buckets.setdefault(self.labels[v.id].subject, []).append(v)
        pieces: list[list[ClaimView]] = []
        current: list[ClaimView] = []
        for subject in sorted(buckets):
            bucket = sorted(buckets[subject], key=lambda v: (repr(_value_key(v)), v.id))
            while len(bucket) > self.cap:
                pieces.append(bucket[:self.cap])
                bucket = bucket[self.cap:]
            if current and len(current) + len(bucket) > self.cap:
                pieces.append(current)
                current = []
            current += bucket
        if current:
            pieces.append(current)
        return pieces

    @staticmethod
    def _with_rule(cluster: Cluster, rules_among: set[str]) -> Cluster:
        """A cluster with exactly one rule (and at least one record) is a rule cluster: the rule first, as the judge expects."""
        rules = [i for i in cluster.claim_ids if i in rules_among]
        ids = cluster.claim_ids
        if len(rules) == 1 and len(ids) >= 2:
            ids = [rules[0], *[i for i in ids if i != rules[0]]]
            return cluster.model_copy(update={"claim_ids": ids, "rule_id": rules[0], "via": {i: "key" for i in ids}})
        return cluster.model_copy(update={"via": {i: "key" for i in ids}})
