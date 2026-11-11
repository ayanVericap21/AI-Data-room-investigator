"""Embedding route of candidate generation, and safety of forced re-extraction."""
from app.llm.mock import MockLLMClient
from app.models import Claim, Finding, FindingClaim, Investigation
from app.services.candidate_matching import generate_candidate_pairs
from app.services.claim_service import ClaimService
from tests.test_claims import claim, doc_with_chunks, view  # noqa: F401  (fixtures/helpers)


def test_embedding_route_finds_synonyms_but_is_gated():
    cap = view("cap", subject="plant", metric="production capacity")
    sup = view("sup", doc="s", subject="supplier", metric="delivery volume", value=25000)
    rent = view("rent", subject="office", metric="monthly rent", value=5000, unit="USD", period="monthly")
    lease = view("lease", subject="office lease", metric="lease end year", value=2031, unit=None, period=None)
    assert generate_candidate_pairs([cap, sup]) == []  # lexical alone misses the synonym
    vec = {"cap": [1.0, 0.0], "sup": [0.9, 0.1], "rent": [0.95, 0.05], "lease": [0.9, 0.1]}
    pairs = generate_candidate_pairs([cap, sup, rent, lease], vectors=vec)
    keys = {frozenset((p.claim_a_id, p.claim_b_id)) for p in pairs}
    assert frozenset(("cap", "sup")) in keys
    assert all(p.matched_by == "embedding" for p in pairs)
    assert not any({"rent", "cap"} <= k for k in keys)  # unit mismatch gate (USD vs tonne)
    assert not any("lease" in k for k in keys)  # unknown-unit gate
    assert generate_candidate_pairs([cap, sup], vectors={"cap": [1.0, 0.0], "sup": [0.0, 1.0]}) == []


def test_embedding_route_respects_period_conflict():
    a = view("a", subject="x", metric="alpha", period="annual")
    b = view("b", subject="y", metric="beta", period="monthly")
    assert generate_candidate_pairs([a, b], vectors={"a": [1.0, 0.0], "b": [1.0, 0.0]}) == []


def test_lexical_pairs_keep_lexical_label():
    assert generate_candidate_pairs([view("a"), view("b")])[0].matched_by == "lexical"


def test_force_never_deletes_claims_cited_by_findings(session, doc_with_chunks):
    _, chunks = doc_with_chunks
    ClaimService(session, MockLLMClient([{"claims": [claim(chunks[0].id)]}])).extract_all()
    c = session.query(Claim).one()
    inv = Investigation(name="t")
    session.add(inv)
    session.flush()
    f = Finding(investigation_id=inv.id, title="t", description="d", classification="contradiction",
                severity="low", confidence=0.5, evidence_status="partial", verification_status="unverified")
    session.add(f)
    session.flush()
    session.add(FindingClaim(finding_id=f.id, claim_id=c.id, relationship_type="supports"))
    session.commit()

    summary = ClaimService(session, MockLLMClient([])).extract_all(force=True)
    assert summary.documents_skipped == 1 and "referenced by findings" in summary.notes[0]
    assert session.query(Claim).count() == 1 and summary.llm_calls == 0


def test_read_paths_do_not_need_an_llm(session, doc_with_chunks):
    assert ClaimService(session).list_claims() == ([], 0)
    assert ClaimService(session).candidate_pairs() == []


def test_same_metric_about_a_different_subject_is_not_paired():
    a = view("a", subject="Light Industrial Units", metric="Total Area", value=396, unit="sq.m")
    b = view("b", doc="d2", subject="Standard Warehouses", metric="Total Area", value=787, unit="sq.m")
    c = view("c", doc="d3", subject="Light Industrial Unit", metric="Total Area", value=400, unit="sq.m")
    keys = {frozenset((p.claim_a_id, p.claim_b_id)) for p in generate_candidate_pairs([a, b, c])}
    assert keys == {frozenset(("a", "c"))}  # plural/singular of the same subject still pairs
    assert generate_candidate_pairs([a, b], min_subject_overlap=None)  # the check can be turned off
    vec = {"a": [1.0, 0.0], "b": [1.0, 0.0]}
    assert generate_candidate_pairs([a, b], vectors=vec) == []  # embeddings do not bring it back
    assert generate_candidate_pairs([a, view("n", doc="d4", subject=None, metric="Total Area", unit="sq.m")])  # no subject: kept


def test_two_claims_with_the_same_statement_are_corroboration_not_something_to_judge():
    a = view("a", claim_text="Unit Sizes from 395m2", conditions="x")
    b = view("b", doc="d2", claim_text="unit sizes from 395m2.", conditions="different wording of the condition")
    c = view("c", doc="d3", claim_text="Unit sizes from 400m2", value=400)
    hints = {frozenset((p.claim_a_id, p.claim_b_id)): p.relation_hint for p in generate_candidate_pairs([a, b, c])}
    assert hints[frozenset(("a", "b"))] == "exact"  # same text: never sent to the LLM, counted as agreement
    assert hints[frozenset(("a", "c"))] != "exact"
