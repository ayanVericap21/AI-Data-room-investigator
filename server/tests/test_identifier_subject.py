"""Identifiers of two different things never conflict: they are never put in one cluster, by any route."""
from app.services.claim_matching import identifiers_about_one_subject, match_clusters
from app.services.claim_grouping import ClaimGrouper
from tests.test_claims import view
from tests.test_neighbors_and_pdf_tables import KeywordEmbedder


def ident(id, doc, subject, value):
    v = view(id, doc=doc, subject=subject, metric="Tool_ID capacity", value=value, claim_text=f"capacity {id}", unit=None, period=None)
    v.value_type = "identifier"
    return v


def test_two_tools_ids_are_not_grouped_but_one_certificates_two_numbers_are():
    tool_a, tool_b = ident("a", "d1", "MET-004", "MET-004"), ident("b", "d2", "MET-003", "MET-003")
    assert not identifiers_about_one_subject(tool_a, tool_b)
    cert_a, cert_b = ident("c", "d1", "ISO certificate", "4471-A"), ident("d", "d2", "ISO certificate", "4417-A")
    assert identifiers_about_one_subject(cert_a, cert_b)
    assert identifiers_about_one_subject(ident("e", "d1", None, "X1"), tool_b)  # a claim with no subject is not filtered
    q1, q2 = view("f", doc="d1", subject="Plant one", value=5), view("g", doc="d2", subject="Plant two", value=6)  # claims that are not identifiers: not this rule's business
    assert identifiers_about_one_subject(q1, q2)


def test_every_grouping_route_drops_identifiers_of_different_things():
    tool_a, tool_b = ident("a", "d1", "MET-004", "MET-004"), ident("b", "d2", "MET-003", "MET-003")
    cert_a, cert_b = ident("c", "d1", "ISO certificate", "4471-A"), ident("d", "d2", "ISO certificate", "4417-A")
    assert {frozenset(c.claim_ids) for c in match_clusters([tool_a, tool_b, cert_a, cert_b], min_subject_overlap=None)} == {frozenset(("c", "d"))}
    assert not ClaimGrouper._together(tool_a, tool_b) and ClaimGrouper._together(cert_a, cert_b)
    near = ClaimGrouper(None, None, embedder=KeywordEmbedder())._neighbor_clusters([tool_a, tool_b, cert_a, cert_b], set())
    assert not any({"a", "b"} <= set(c.claim_ids) for c in near)


