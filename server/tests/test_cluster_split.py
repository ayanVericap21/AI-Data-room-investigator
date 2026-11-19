"""Sets of claims that are too large to be one fact are divided by how alike the claims are, not cut at random."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from app.services.claim_group_judge import ClaimGroupJudge
from app.services.cluster_split import ClaimSimilarity, split_by_similarity
from app.schemas.claim import Cluster

TOPICS = 8


class TopicEmbedder:
    """A vector for each text: the topic named in it ("topic3") as a direction, a little noise for each claim."""

    def embed(self, texts, task="document"):
        out = []
        for text in texts:
            v = np.zeros(TOPICS + 1)
            v[int(text.split("topic")[1][0])] = 1.0
            v[-1] = 0.05 * (hash(text) % 7)  # claims of one topic are alike but not identical
            out.append(v)
        return out


def cluster_of(members, how="llm"):
    return [Cluster(claim_ids=[c.id for c in members], matched_by=how)]


def claims(topics: list[int], per_topic: int):
    return [SimpleNamespace(id=f"t{t}-{i}", subject=f"topic{t}", metric="m", claim_text=f"topic{t} claim {i}", value=i) for t in topics for i in range(per_topic)]


def test_the_most_alike_claims_are_joined_first_and_no_cluster_exceeds_the_cap():
    vectors = np.array([[1.0, 0.0]] * 10 + [[0.0, 1.0]] * 10 + [[0.7071, 0.7071]])
    parts = split_by_similarity(vectors, cap=15)
    assert all(len(p) <= 15 for p in parts)
    assert any(set(range(10)) <= set(p) for p in parts) and any(set(range(10, 20)) <= set(p) for p in parts)
    one_topic = split_by_similarity(np.array([[1.0, 0.0]] * 40), cap=15)
    assert max(len(p) for p in one_topic) <= 15 and sorted(i for p in one_topic for i in p) == list(range(40))


def test_claims_that_are_alike_to_nothing_stay_alone():
    parts = split_by_similarity(np.eye(4), cap=15)
    assert sorted(len(p) for p in parts) == [1, 1, 1, 1]


def test_a_large_cluster_of_the_model_is_divided_by_meaning_into_clusters():
    members = claims([0, 1, 2, 3], 10)  # 40 claims, four topics, one cluster of the model
    clusters = cluster_of(members)
    similarity = ClaimSimilarity(TopicEmbedder(), {c.id: c for c in members})
    tally = {"split": 0}
    groups = ClaimGroupJudge._form_groups(clusters, similarity=similarity, tally=tally)
    assert sorted(len(g.claim_ids) for g in groups) == [10, 10, 10, 10] and tally["split"] == 1
    assert all(len({i.split("-")[0] for i in g.claim_ids}) == 1 for g in groups)  # each cluster is one topic
    cut = ClaimGroupJudge._form_groups(clusters)  # without the embedder the set is cut in the order of its claims into clusters of nearly equal size
    assert sorted(len(g.claim_ids) for g in cut) == [12, 14, 14] and all(len(g.claim_ids) <= 15 for g in cut)


def test_a_cluster_over_the_cap_with_two_topics_is_divided_into_one_cluster_for_each_topic():
    members = claims([0, 1], 3)
    near = SimpleNamespace(id="t0-9", subject="topic0", metric="m", claim_text="topic0 claim 9", value=9)
    members.append(near)
    similarity = ClaimSimilarity(TopicEmbedder(), {c.id: c for c in members})
    groups = ClaimGroupJudge._form_groups(cluster_of(members), similarity=similarity, cap=4)  # 7 claims: more than the cap, two topics
    topics = sorted(sorted(i.split("-")[0] for i in g.claim_ids) for g in groups)
    assert ["t0", "t0", "t0", "t0"] in topics and ["t1", "t1", "t1"] in topics and len(groups) == 2


def test_a_cluster_whose_claims_are_not_alike_is_divided_even_when_it_is_small_and_an_alike_one_is_kept_whole():
    scattered = claims([0, 1, 2], 2)  # three topics, two claims each: on average the claims are far from alike
    similarity = ClaimSimilarity(TopicEmbedder(), {c.id: c for c in scattered})
    groups = ClaimGroupJudge._form_groups(cluster_of(scattered), similarity=similarity)
    assert sorted(len(g.claim_ids) for g in groups) == [2, 2, 2]
    alike = claims([2], 5)
    kept = ClaimGroupJudge._form_groups(cluster_of(alike), similarity=ClaimSimilarity(TopicEmbedder(), {c.id: c for c in alike}))
    assert [len(g.claim_ids) for g in kept] == [5]
