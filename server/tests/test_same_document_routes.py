"""Clusters inside one document: numbers and dates in different passages of one document are compared (a document can contradict itself)."""
from __future__ import annotations

from app.schemas.claim import ClaimView
from app.services.blocking import inside_document_ok, shared_label_clusters
from app.services.claim_grouping import ClaimGrouper


def claim(n, doc, chunk, subject, metric, value, kind="quantity", text=""):
    return ClaimView(id=f"c{n}", document_id=doc, chunk_id=chunk, subject=subject, metric=metric, value=value, claim_text=text or f"{subject} {metric} {value}", value_type=kind)


class SameKeywordEmbedder:
    """Texts about liability get one direction, so their similarity is 1; everything else is orthogonal."""

    def embed(self, texts, task="document"):
        return [[1.0 if "liab" in t.lower() or "indemn" in t.lower() else 0.0, 0.0 if "liab" in t.lower() or "indemn" in t.lower() else 1.0] for t in texts]


def neighbors(views):
    return ClaimGrouper(None, None, embedder=SameKeywordEmbedder())._neighbor_clusters(views, set())


def test_two_numbers_in_two_passages_of_one_document_may_go_together_but_one_passage_and_statements_may_not():
    a = claim(1, "d1", "ch1", "Guaranty", "maximum indemnity", "$1,500,000")
    b = claim(2, "d1", "ch2", "Guarantor", "maximum total liability", "$1,000,000")
    same_passage = claim(3, "d1", "ch1", "Guarantor", "maximum liability cap", "$900,000")
    statement_a = claim(4, "d1", "ch3", "Guaranty", "indemnity terms", "shall remain in effect", kind="statement")
    statement_b = claim(5, "d1", "ch4", "Guarantor", "liability terms", "shall remain in force", kind="statement")
    assert inside_document_ok(a, b) and not inside_document_ok(a, same_passage) and not inside_document_ok(statement_a, statement_b)
    assert inside_document_ok(a, claim(6, "d2", "ch9", "x", "y", "z", kind="statement"))  # different documents: always


def test_the_nearest_neighbour_route_puts_two_figures_of_one_document_that_say_the_same_thing_in_different_words_together():
    a = claim(1, "d1", "ch1", "Credit recapture guaranty", "maximum indemnity", "$1,500,000")
    b = claim(2, "d1", "ch2", "Guarantor", "maximum total liability for recapture", "$1,000,000")
    found = neighbors([a, b])
    assert [set(c.claim_ids) for c in found] == [{"c1", "c2"}] and found[0].matched_by == "embedding"


def test_the_neighbour_route_does_not_put_statements_of_one_document_together():
    a = claim(1, "d1", "ch1", "Guaranty", "indemnity terms", "remains in effect", kind="statement")
    b = claim(2, "d1", "ch2", "Guarantor", "liability terms", "remains in force", kind="statement")
    assert neighbors([a, b]) == []


def test_an_in_document_neighbour_does_not_take_the_place_of_a_cross_document_one():
    a = claim(1, "d1", "ch1", "Guaranty", "maximum liability", "$1,500,000")
    own = claim(2, "d1", "ch2", "Guaranty", "maximum liability cap", "$1,400,000")
    other = claim(3, "d2", "ch3", "Guaranty", "maximum liability", "$1,000,000")
    found = neighbors([a, own, other])
    assert [set(c.claim_ids) for c in found] == [{"c1", "c2", "c3"}]  # both go with c1: the two are ranked apart


def test_the_rare_label_route_puts_a_rare_metric_stated_in_two_passages_of_one_document_together():
    views = [claim(1, "d1", "ch1", "Fund", "escrow reserve", "$50,000"), claim(2, "d1", "ch2", "Escrow account", "escrow reserve", "$75,000"),
             claim(3, "d2", "ch3", "Site", "headcount", "48"), claim(4, "d3", "ch4", "Plant", "weather", "dry", kind="state")]
    found = {frozenset(c.claim_ids) for c in shared_label_clusters(views)}
    assert found == {frozenset({"c1", "c2"})}


def test_the_rare_label_route_still_ignores_two_claims_of_one_passage():
    views = [claim(1, "d1", "ch1", "Fund", "escrow reserve", "$50,000"), claim(2, "d1", "ch1", "Escrow account", "escrow reserve", "$75,000")]
    assert shared_label_clusters(views) == []
