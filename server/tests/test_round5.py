"""Different attributes are not an insufficient-evidence finding; findings are merged only when their differing claims carry the same label."""
from sqlalchemy import select

from app.llm.mock import MockLLMClient
from app.models import Claim, Evidence, Finding
from app.schemas.claim import CandidatePair
from app.services.pair_judge import PairJudge
from app.tools import build_registry
from tests.test_pair_judge import comparison, make_claims, pair
from tests.test_tools import add_doc, room  # noqa: F401  (fixtures and helper)


def judge(session, room, llm):  # noqa: F811
    return PairJudge(session, llm, build_registry(session, None, room.investigation.id))


def verdict(index, cls, quote_a, quote_b, **extra):
    return dict(pair_index=index, classification=cls, **{"severity": "low", "confidence": 0.6, "title": cls, "description": "d", "quote_a": quote_a, "quote_b": quote_b, **extra})


# ---- 1: two attributes of one item are not comparable, whatever the judge says --------------------------------
def test_insufficient_evidence_about_two_different_attributes_is_dropped(session, room):  # noqa: F811
    a, b = make_claims(session, room)
    a.subject = b.subject = "LTH-001"
    a.metric, b.metric = "Tool_Category", "Procurement_Status"
    a.value, b.value = "Lithography", "Installed"
    session.commit()
    p = pair(a, b)
    stats = judge(session, room, MockLLMClient([{"verdicts": [verdict(0, "insufficient_evidence", "", "")]}])).run([p])
    assert session.scalars(select(Finding)).all() == []
    assert stats.verdict_counts["not_comparable"] == 1 and "share no word" in stats.records[(p.claim_a_id, p.claim_b_id)].outcome


def test_insufficient_evidence_about_one_attribute_is_still_reported(session, room):  # noqa: F811
    a, b = make_claims(session, room)  # both: plant, capacity
    a.value, b.value = "120,000", "100,000"
    session.commit()
    v = verdict(0, "insufficient_evidence", "capacity is 120,000 tonnes per year", "Capacity | 100,000 tonnes")
    stats = judge(session, room, MockLLMClient([{"verdicts": [v]}])).run([pair(a, b)])
    assert [f.classification for f in session.scalars(select(Finding))] == ["insufficient_evidence"] and stats.verdict_counts["not_comparable"] == 0


def test_the_attribute_may_be_the_row_label_of_the_other_claim(session, room):  # noqa: F811
    a, b = make_claims(session, room)
    a.subject, a.metric = "Designed nameplate capacity", "Model_Value"
    b.subject, b.metric = "SiliconForge", "nameplate capacity"
    a.value, b.value = "120,000", "100,000"
    session.commit()
    v = verdict(0, "insufficient_evidence", "capacity is 120,000 tonnes per year", "Capacity | 100,000 tonnes")
    judge(session, room, MockLLMClient([{"verdicts": [v]}])).run([pair(a, b)])
    assert len(session.scalars(select(Finding)).all()) == 1


# ---- 2: findings sharing a claim are merged only when the claims that differ carry the same label -------------
def _capacity_case(session, room, label_c):  # noqa: F811
    a, b = make_claims(session, room)
    third, rows = add_doc(session, "third.pdf", "pdf", [dict(content="Plant capacity is 120,000 tonnes in total.", page_number=1)])
    a.value, b.value = "120,000", "100,000"
    c = Claim(document_id=third.id, chunk_id=rows[0].id, subject=label_c[0], metric=label_c[1], value="120,000", unit="tonnes", claim_text="Plant capacity is 120,000 tonnes in total")
    session.add(c)
    session.commit()
    quote = {a.id: "capacity is 120,000 tonnes per year", b.id: "Capacity | 100,000 tonnes", c.id: "capacity is 120,000 tonnes in total"}
    p1, p2 = pair(a, b), pair(c, b)
    verdicts = [verdict(i, cls, quote[p.claim_a_id], quote[p.claim_b_id], severity="high")
                for i, (p, cls) in enumerate(((p1, "contradiction"), (p2, "contextual_difference")))]
    exact = CandidatePair(claim_a_id=a.id, claim_b_id=c.id, topic_score=1.0, relation_hint="exact", comparison=comparison(value_relation="equal"), matched_by="llm")
    stats = judge(session, room, MockLLMClient([{"verdicts": verdicts}])).run([p1, p2, exact])
    return stats, quote


def test_findings_about_a_different_labelled_claim_are_not_merged(session, room):  # noqa: F811
    stats, quote = _capacity_case(session, room, ("plant", "current installed nameplate"))  # a is "plant capacity"; c is a different label
    assert stats.merged == 0
    assert sorted(f.classification for f in session.scalars(select(Finding))) == ["consistent", "contextual_difference", "contradiction"]
    contradiction = session.scalars(select(Finding).where(Finding.classification == "contradiction")).one()
    quotes = {e.quoted_text for e in session.scalars(select(Evidence).where(Evidence.finding_id == contradiction.id))}
    assert quote and len(quotes) == 2  # the installed-capacity quote was not added as evidence


def test_findings_about_the_same_labelled_claim_are_still_merged(session, room):  # noqa: F811
    stats, _ = _capacity_case(session, room, ("plant", "capacity"))
    assert stats.merged == 1
