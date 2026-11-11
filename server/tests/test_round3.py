"""One discrepancy, one finding; label-only claims skipped; declared absences asked for; no finding about unrelated claims; no repeated unit."""
from sqlalchemy import select

from app.llm.mock import MockLLMClient
from app.models import Claim, Evidence, Finding
from app.schemas.claim import CandidatePair
from app.services.llama_claims import SCHEMA, SYSTEM_PROMPT, LlamaClaim, _is_own_label
from app.services.pair_judge import PairJudge, _shown
from app.tools import build_registry
from tests.test_pair_judge import comparison, make_claims, pair
from tests.test_tools import add_doc, room  # noqa: F401  (fixtures and helper)


# ---- 5 and 2: extraction -------------------------------------------------------
def test_a_claim_whose_value_is_only_its_own_row_label_is_not_a_claim():
    assert _is_own_label(LlamaClaim(subject="MET-004", metric="Tool_ID", value="MET-004"))
    assert _is_own_label(LlamaClaim(subject=" met-004 ", metric="Tool_ID", value="MET-004"))
    assert not _is_own_label(LlamaClaim(subject="MET-004", metric="Installed_Cost", value="26"))
    assert not _is_own_label(LlamaClaim(subject=None, metric="x", value="MET-004"))
    assert "that label is the subject, not a value" in SYSTEM_PROMPT


def test_extraction_asks_for_declared_absences_and_not_for_disclaimer_document_classes():
    for phrase in ("not provided", "not yet received", "pending", "outstanding"):
        assert phrase in SYSTEM_PROMPT and phrase in SCHEMA["properties"]["references"]["description"] or phrase in SYSTEM_PROMPT
    assert "general disclaimer" in SYSTEM_PROMPT and "general disclaimer" in SCHEMA["properties"]["references"]["description"]


# ---- 4: unit text -------------------------------------------------------------------
def test_a_unit_already_in_the_value_is_not_repeated():
    assert _shown(Claim(value="12,000 WSPM", unit="WSPM")) == "12,000 WSPM"
    assert _shown(Claim(value="24000", unit="WSPM")) == "24000 WSPM"
    assert _shown(Claim(value="96.8%", unit="%")) == "96.8%"
    assert _shown(Claim(value="Karnataka", unit=None)) == "Karnataka"


# ---- 3: no finding about unrelated claims -------------------------------------------
def test_insufficient_evidence_from_code_quotes_needs_claims_about_the_same_thing(session, room):  # noqa: F811
    a, b = make_claims(session, room)
    a.subject, b.subject = "CMP-006 tool", "Paid-up equity capital"
    session.commit()
    p = pair(a, b)
    verdict = dict(pair_index=0, classification="insufficient_evidence", severity="low", confidence=0.5, title="x", description="d", quote_a="", quote_b="")
    stats = PairJudge(session, MockLLMClient([{"verdicts": [verdict]}]), build_registry(session, None, room.investigation.id)).run([p])
    assert stats.findings_saved == 0 and "subjects do not match" in stats.records[(p.claim_a_id, p.claim_b_id)].outcome
    assert session.scalars(select(Finding)).all() == []


# ---- 1: one discrepancy, one finding --------------------------------------------------
def test_findings_about_one_discrepancy_become_one_finding_with_all_the_evidence(session, room):  # noqa: F811
    a, b = make_claims(session, room)  # a: 120,000 in the deck; b: 100,000 in the capex file
    third, rows = add_doc(session, "third.pdf", "pdf", [dict(content="Plant capacity is 120,000 tonnes in total.", page_number=1)])
    a.value, b.value = "120,000", "100,000"
    a.claim_text, b.claim_text = "Annual production capacity is 120,000 tonnes per year", "Capacity | 100,000 tonnes"
    c = Claim(document_id=third.id, chunk_id=rows[0].id, subject="plant", metric="capacity", value="120,000", unit="tonnes", claim_text="Plant capacity is 120,000 tonnes in total")
    session.add(c)
    session.commit()
    quote = {a.id: "capacity is 120,000 tonnes per year", b.id: "Capacity | 100,000 tonnes", c.id: "capacity is 120,000 tonnes in total"}

    def verdict(p, index, classification, severity, confidence):
        return dict(pair_index=index, classification=classification, severity=severity, confidence=confidence, title=f"{classification} capacity",
                    description="d", quote_a=quote[p.claim_a_id], quote_b=quote[p.claim_b_id])

    p1, p2 = pair(a, b), pair(c, b)
    agreement = CandidatePair(claim_a_id=a.id, claim_b_id=c.id, topic_score=1.0, relation_hint="exact", comparison=comparison(value_relation="equal"), matched_by="llm")
    llm = MockLLMClient([{"verdicts": [verdict(p1, 0, "contradiction", "high", 0.9), verdict(p2, 1, "contextual_difference", "low", 0.8)]}])
    stats = PairJudge(session, llm, build_registry(session, None, room.investigation.id)).run([p1, p2, agreement])
    findings = session.scalars(select(Finding)).all()
    assert sorted(f.classification for f in findings) == ["consistent", "contradiction"]  # the contextual one was folded into the contradiction
    main = next(f for f in findings if f.classification == "contradiction")
    quotes = {e.quoted_text for e in session.scalars(select(Evidence).where(Evidence.finding_id == main.id))}
    assert quotes == set(quote.values()) and main.severity == "high" and "Also reported separately and merged here" in main.description
    assert stats.merged == 1 and stats.findings_saved == 2 and "merged into" in stats.records[(p2.claim_a_id, p2.claim_b_id)].outcome
    assert not session.scalars(select(Evidence).where(~Evidence.finding_id.in_([f.id for f in findings]))).all()  # nothing left behind


def test_unrelated_discrepancies_are_not_merged(session, room):  # noqa: F811
    a, b = make_claims(session, room)
    other_doc, rows = add_doc(session, "other.pdf", "pdf", [dict(content="Rent is 5,000 per month.", page_number=1), dict(content="Rent in the lease is 6,000 per month.", page_number=2)])
    c = Claim(document_id=other_doc.id, chunk_id=rows[0].id, subject="rent", metric="rent", value="5,000", claim_text="Rent is 5,000 per month")
    d = Claim(document_id=other_doc.id, chunk_id=rows[1].id, subject="rent", metric="rent", value="6,000", claim_text="Rent in the lease is 6,000 per month")
    a.value, b.value = "120,000", "100,000"
    session.add_all([c, d])
    session.commit()
    quote = {a.id: "capacity is 120,000 tonnes per year", b.id: "Capacity | 100,000 tonnes", c.id: "Rent is 5,000 per month", d.id: "Rent in the lease is 6,000 per month"}
    p1, p2 = pair(a, b), pair(c, d)

    def verdict(p, i):
        return dict(pair_index=i, classification="contradiction", severity="high", confidence=0.9, title=f"t{i}", description="d", quote_a=quote[p.claim_a_id], quote_b=quote[p.claim_b_id])

    stats = PairJudge(session, MockLLMClient([{"verdicts": [verdict(p1, 0), verdict(p2, 1)]}]), build_registry(session, None, room.investigation.id)).run([p1, p2])
    assert stats.merged == 0 and len(session.scalars(select(Finding)).all()) == 2
