"""The claim groups tab shows claims that are candidates of one another, each with its own candidates, and every claim comes with its source."""
from __future__ import annotations

from datetime import datetime

from app.schemas.claim import AttributeComparison, Candidate, ClaimOut, SourceRef
from app.services.claim_groups import group_candidates


def link(a, b, hint="conflicting", how="lexical", score=0.9):
    return Candidate(claim_id=a, candidate_id=b, topic_score=score, relation_hint=hint, matched_by=how, comparison=AttributeComparison(
        subject_match=True, metric_match=True, unit_match=True, period_match=True, value_relation="different", conditions_differ=False, assumptions_differ=False))


def claim(id):
    return ClaimOut(id=id, subject="plant", metric="capacity", value=1, unit=None, period=None, conditions=None, assumptions=None, claim_text=f"text of {id}",
                    claim_category=None, extraction_confidence=None, created_at=datetime(2026, 1, 1),
                    source=SourceRef(document_id="d", filename=f"{id}.pdf", chunk_id=f"chunk-{id}", content_type=None, page_number=1, sheet_name=None,
                                     cell_reference=None, heading=None))


def claims_of(*ids):
    return {i: claim(i) for i in ids}


def test_claims_that_are_candidates_of_each_other_form_one_group_with_a_list_per_claim_and_each_comes_with_its_source():
    groups = group_candidates([link("c1", "c2"), link("c1", "c3", "contextual", "embedding", 0.8), link("x1", "x2")], claims_of("c1", "c2", "c3", "x1", "x2"))
    assert len(groups) == 2
    first = {gc.claim.id: [(k.claim.id, k.relation_hint, k.matched_by) for k in gc.candidates] for gc in groups[0].claims}
    assert list(first) == ["c1", "c2", "c3"]
    assert first["c1"] == [("c2", "conflicting", "lexical"), ("c3", "contextual", "embedding")]
    assert first["c2"] == [("c1", "conflicting", "lexical")] and first["c3"] == [("c1", "contextual", "embedding")]
    assert groups[0].claims[0].candidates[0].claim.source.filename == "c2.pdf" and groups[0].claims[0].candidates[0].claim.claim_text == "text of c2"
    assert [gc.claim.id for gc in groups[1].claims] == ["x1", "x2"]


def test_a_chain_of_candidates_is_one_group_and_no_candidates_is_no_group():
    assert len(group_candidates([link("a", "b"), link("b", "c"), link("c", "d")], claims_of("a", "b", "c", "d"))) == 1
    assert group_candidates([], {}) == []


def test_a_candidate_whose_claim_is_gone_is_left_out():
    groups = group_candidates([link("a", "b"), link("a", "gone")], claims_of("a", "b"))
    assert [gc.claim.id for gc in groups[0].claims] == ["a", "b"] and len(groups[0].claims[0].candidates) == 1
