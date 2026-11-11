"""Identifiers of two different things never conflict: such a pair is not made, by any route."""
from app.services.candidate_matching import generate_candidate_pairs, identifiers_about_one_subject
from app.services.llm_pairing import SYSTEM as GROUPING, LLMPairer
from app.services.pair_judge import SYSTEM as JUDGE
from tests.test_claims import view
from tests.test_neighbors_and_pdf_tables import KeywordEmbedder


def ident(id, doc, subject, value):
    v = view(id, doc=doc, subject=subject, metric="Tool_ID capacity", value=value, claim_text=f"capacity {id}", unit=None, period=None)
    v.value_type = "identifier"
    return v


def test_two_tools_ids_are_not_paired_but_one_certificates_two_numbers_are():
    tool_a, tool_b = ident("a", "d1", "MET-004", "MET-004"), ident("b", "d2", "MET-003", "MET-003")
    assert not identifiers_about_one_subject(tool_a, tool_b)
    cert_a, cert_b = ident("c", "d1", "ISO certificate", "4471-A"), ident("d", "d2", "ISO certificate", "4417-A")
    assert identifiers_about_one_subject(cert_a, cert_b)
    assert identifiers_about_one_subject(ident("e", "d1", None, "X1"), tool_b)  # a claim with no subject is not filtered
    q1, q2 = view("f", doc="d1", subject="Plant one", value=5), view("g", doc="d2", subject="Plant two", value=6)  # claims that are not identifiers: not this rule's business
    assert identifiers_about_one_subject(q1, q2)


def test_every_pairing_route_drops_identifiers_of_different_things():
    tool_a, tool_b = ident("a", "d1", "MET-004", "MET-004"), ident("b", "d2", "MET-003", "MET-003")
    cert_a, cert_b = ident("c", "d1", "ISO certificate", "4471-A"), ident("d", "d2", "ISO certificate", "4417-A")
    assert {frozenset((p.claim_a_id, p.claim_b_id)) for p in generate_candidate_pairs([tool_a, tool_b, cert_a, cert_b], min_subject_overlap=None)} == {frozenset(("c", "d"))}
    pairs: dict = {}
    LLMPairer._add_pair(pairs, tool_a, tool_b)
    LLMPairer._add_pair(pairs, cert_a, cert_b)
    assert set(pairs) == {("c", "d")}
    near: dict = {}
    LLMPairer(None, None, embedder=KeywordEmbedder())._add_neighbors([tool_a, tool_b, cert_a, cert_b], near)
    assert ("a", "b") not in near


def test_the_judge_prompt_limits_identifier_contradictions_to_one_thing():
    assert "only when both claims label the same thing" in JUDGE and "two different things" in JUDGE
