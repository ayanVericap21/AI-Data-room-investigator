"""The claim type is shown to the grouping model as information only: it never keeps two claims from being grouped."""
from app.services.candidate_matching import generate_candidates
from app.services.claim_grouping import SYSTEM, ClaimGrouper, _line
from tests.test_claims import view
from tests.test_neighbors_and_pdf_tables import KeywordEmbedder


def typed(id, kind, doc, **kw):
    v = view(id, doc=doc, metric="capacity", value=kw.pop("value", 1), claim_text=f"capacity {id}", **kw)
    v.value_type = kind
    return v


def test_claims_of_different_types_are_still_grouped_by_every_route():
    a, b, c = typed("a", "numerical", "d1", value=10), typed("b", "textual", "d2", value="ten"), typed("c", "numerical", "d3", value=12)
    assert {frozenset((p.claim_id, p.candidate_id)) for p in generate_candidates([a, b, c])} == {
        frozenset(("a", "b")), frozenset(("a", "c")), frozenset(("b", "c"))}
    found: dict = {}
    ClaimGrouper._add_candidate(found, a, b)
    ClaimGrouper._add_candidate(found, a, c)
    assert set(found) == {("a", "b"), ("a", "c")}
    near: dict = {}
    ClaimGrouper(None, None, embedder=KeywordEmbedder())._add_neighbors([a, b, c], near)
    assert ("a", "b") in near and ("a", "c") in near


