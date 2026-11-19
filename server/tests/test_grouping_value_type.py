"""The claim type is shown to the grouping model as information only: it never keeps two claims from being grouped."""
from app.services.claim_matching import match_clusters
from app.services.claim_grouping import SYSTEM, ClaimGrouper, _line
from tests.test_claims import view
from tests.test_neighbors_and_pdf_tables import KeywordEmbedder


def typed(id, kind, doc, **kw):
    v = view(id, doc=doc, metric="capacity", value=kw.pop("value", 1), claim_text=f"capacity {id}", **kw)
    v.value_type = kind
    return v


def test_claims_of_different_types_are_still_grouped_by_every_route():
    a, b, c = typed("a", "numerical", "d1", value=10), typed("b", "textual", "d2", value="ten"), typed("c", "numerical", "d3", value=12)
    assert {frozenset(c.claim_ids) for c in match_clusters([a, b, c])} == {frozenset(("a", "b", "c"))}
    assert ClaimGrouper._together(a, b) and ClaimGrouper._together(a, c)
    near = ClaimGrouper(None, None, embedder=KeywordEmbedder())._neighbor_clusters([a, b, c], set())
    assert any({"a", "b"} <= set(cl.claim_ids) for cl in near) and any({"a", "c"} <= set(cl.claim_ids) for cl in near)


