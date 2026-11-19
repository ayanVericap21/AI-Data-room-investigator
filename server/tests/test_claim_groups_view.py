"""The claim groups tab shows the claims of a group with the route that found them, and every claim comes with its source."""
from __future__ import annotations

from datetime import datetime

from app.schemas.claim import ClaimOut, Cluster, SourceRef
from app.services.claim_groups import group_clusters


def cluster(ids, how="lexical", topic=""):
    return Cluster(claim_ids=ids, matched_by=how, topic=topic)


def claim(id):
    return ClaimOut(id=id, subject="plant", metric="capacity", value=1, unit=None, period=None, conditions=None, assumptions=None, claim_text=f"text of {id}",
                    claim_category=None, extraction_confidence=None, created_at=datetime(2026, 1, 1),
                    source=SourceRef(document_id="d", filename=f"{id}.pdf", chunk_id=f"chunk-{id}", content_type=None, page_number=1, sheet_name=None,
                                     cell_reference=None, heading=None))


def claims_of(*ids):
    return {i: claim(i) for i in ids}


def test_clusters_that_share_a_claim_form_one_group_and_each_claim_comes_with_its_source():
    groups = group_clusters([cluster(["c1", "c2"], topic="capacity"), cluster(["c1", "c3"], "embedding"), cluster(["x1", "x2"])], claims_of("c1", "c2", "c3", "x1", "x2"))
    assert len(groups) == 2
    assert [c.id for c in groups[0].claims] == ["c1", "c2", "c3"] and groups[0].topic == "capacity" and groups[0].matched_by == "lexical"
    assert groups[0].claims[1].source.filename == "c2.pdf" and groups[0].claims[1].claim_text == "text of c2"
    assert [c.id for c in groups[1].claims] == ["x1", "x2"]


def test_a_chain_of_overlapping_clusters_is_one_group_and_no_clusters_is_no_group():
    assert len(group_clusters([cluster(["a", "b"]), cluster(["b", "c"]), cluster(["c", "d"])], claims_of("a", "b", "c", "d"))) == 1
    assert group_clusters([], {}) == []


def test_a_claim_that_is_gone_is_left_out_and_a_group_left_with_one_claim_is_dropped():
    groups = group_clusters([cluster(["a", "b", "gone"]), cluster(["x", "gone"])], claims_of("a", "b", "x"))
    assert [[c.id for c in g.claims] for g in groups] == [["a", "b"]]
