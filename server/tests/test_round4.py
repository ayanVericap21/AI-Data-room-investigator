"""Duplicates by claims and quotes, evidence keeps its role, agreements about different subjects are judged, missing items per sentence,
insufficient evidence absorbed by a consistent finding, consistent findings grouped by document pair and attribute."""
from sqlalchemy import select

from app.llm.mock import MockLLMClient
from app.models import Claim, Evidence, Finding
from app.schemas.claim import CandidatePair
from app.services.missing_items import DocReference, MissingItemChecker
from app.services.pair_judge import PairJudge
from app.tools import build_registry
from tests.test_pair_judge import comparison, make_claims, pair
from tests.test_tools import add_doc, room  # noqa: F401  (fixtures and helper)


def judge(session, room, llm):  # noqa: F811
    return PairJudge(session, llm, build_registry(session, None, room.investigation.id))


def exact(a, b):
    return CandidatePair(claim_a_id=a.id, claim_b_id=b.id, topic_score=1.0, relation_hint="exact", comparison=comparison(value_relation="equal"), matched_by="llm")


# ---- 1: a start date and a volume stated by the same two passages are two findings ---------------
def test_two_facts_stated_by_the_same_two_passages_are_both_findings(session, room):  # noqa: F811
    sentence = "Forecast is 8,000 wafers per month, commencing 1 April 2027."
    row = "Northstar | Volume: 10000 | Start: 2027-04-01"
    d1, c1 = add_doc(session, "agreement.docx", "docx", [dict(content=sentence, heading="Forecast")])
    d2, c2 = add_doc(session, "plan.xlsx", "xlsx", [dict(content=row, sheet_name="Plan", cell_reference="A2:C2", content_type="table_row")])
    a1 = Claim(document_id=d1.id, chunk_id=c1[0].id, subject="Northstar", metric="start", value="1 April 2027", claim_text=sentence)
    a2 = Claim(document_id=d1.id, chunk_id=c1[0].id, subject="Northstar", metric="volume", value="8,000", claim_text=sentence)
    b1 = Claim(document_id=d2.id, chunk_id=c2[0].id, subject="Northstar", metric="start", value="2027-04-01", claim_text=row)
    b2 = Claim(document_id=d2.id, chunk_id=c2[0].id, subject="Northstar", metric="volume", value="10000", claim_text=row)
    session.add_all([a1, a2, b1, b2])
    session.commit()
    p2 = pair(a2, b2)
    quotes = {a2.id: sentence, b2.id: row}
    verdict = dict(pair_index=0, classification="contextual_difference", severity="medium", confidence=0.9, title="Volume differs", description="d",
                   quote_a=quotes[p2.claim_a_id], quote_b=quotes[p2.claim_b_id])
    stats = judge(session, room, MockLLMClient([{"verdicts": [verdict]}])).run([exact(a1, b1), p2])
    assert sorted(f.classification for f in session.scalars(select(Finding))) == ["consistent", "contextual_difference"]
    assert stats.duplicates_skipped == 0


# ---- 2: merged evidence keeps its own role ---------------------------------------------------------
def test_merged_evidence_keeps_the_role_it_had(session, room):  # noqa: F811
    a, b = make_claims(session, room)
    third, rows = add_doc(session, "third.pdf", "pdf", [dict(content="Plant capacity is 120,000 tonnes in total.", page_number=1)])
    a.value, b.value = "120,000", "100,000"
    c = Claim(document_id=third.id, chunk_id=rows[0].id, subject="plant", metric="capacity", value="120,000", unit="tonnes", claim_text="Plant capacity is 120,000 tonnes in total")
    session.add(c)
    session.commit()
    quote = {a.id: "capacity is 120,000 tonnes per year", b.id: "Capacity | 100,000 tonnes", c.id: "capacity is 120,000 tonnes in total"}
    p1, p2 = pair(a, b), pair(c, b)

    def verdict(p, i, cls, conf):
        return dict(pair_index=i, classification=cls, severity="high", confidence=conf, title=cls, description="d", quote_a=quote[p.claim_a_id], quote_b=quote[p.claim_b_id])

    judge(session, room, MockLLMClient([{"verdicts": [verdict(p1, 0, "contradiction", 0.9), verdict(p2, 1, "contextual_difference", 0.8)]}])).run([p1, p2, exact(a, c)])
    main = session.scalars(select(Finding).where(Finding.classification == "contradiction")).one()
    roles = {e.quoted_text: e.evidence_role for e in session.scalars(select(Evidence).where(Evidence.finding_id == main.id))}
    assert roles[quote[a.id]] == "contradicting" and roles[quote[c.id]] == "contextual"  # the added quote is context, not a contradiction


# ---- 3: equal values about different subjects are not an agreement until the judge says so --------------
def test_equal_values_about_different_subjects_go_to_the_judge(session, room):  # noqa: F811
    a, b = make_claims(session, room)
    a.subject, b.subject = "Design connected electrical load", "final Consent for Establishment amendment"
    session.commit()
    verdict = dict(pair_index=0, same_attribute=False, attribute_reasoning="A is a design load, B is a consent record", classification="consistent",
                   severity="informational", confidence=0.5, title="", description="", quote_a="", quote_b="")
    stats = judge(session, room, MockLLMClient([{"verdicts": [verdict]}])).run([exact(a, b)])
    assert stats.pairs_exact == 0 and stats.pairs_judged == 1 and stats.verdict_counts["not_comparable"] == 1
    assert session.scalars(select(Finding)).all() == []


# ---- 4: missing items named by one sentence are one finding ------------------------------------------------
def test_items_named_by_one_sentence_are_one_missing_information_finding(session, room):  # noqa: F811
    sentence = "Annual rent is INR 1.2 crore."
    refs = [DocReference(room.lease.id, room.rent_1, item, sentence) for item in ("MES exports", "uptime logs", "acceptance records")]
    stats = MissingItemChecker(session, build_registry(session, None, room.investigation.id)).run(refs)
    finding = session.scalars(select(Finding)).one()
    assert stats.saved == 1 and stats.missing_items == ["MES exports", "uptime logs", "acceptance records"]
    assert all(item in finding.title for item in ("MES exports", "uptime logs", "acceptance records"))
    assert len(finding.unresolved_questions) == 3


# ---- 5: a consistent finding absorbs an insufficient-evidence one about the same fact ---------------------------
def test_a_consistent_finding_absorbs_an_insufficient_one_about_the_same_fact(session, room):  # noqa: F811
    a, b = make_claims(session, room)
    third, rows = add_doc(session, "third.pdf", "pdf", [dict(content="Plant capacity is 120,000 tonnes in total.", page_number=1)])
    a.value, b.value = "120,000", "100,000"
    c = Claim(document_id=third.id, chunk_id=rows[0].id, subject="plant", metric="capacity", value="120,000", unit="tonnes", claim_text="Plant capacity is 120,000 tonnes in total")
    session.add(c)
    session.commit()
    quote = {a.id: "capacity is 120,000 tonnes per year", b.id: "Capacity | 100,000 tonnes", c.id: "capacity is 120,000 tonnes in total"}
    p1, p2 = pair(a, b), pair(a, c)

    def verdict(p, i, cls):
        return dict(pair_index=i, classification=cls, severity="low", confidence=0.6, title=cls, description="d", quote_a=quote[p.claim_a_id], quote_b=quote[p.claim_b_id])

    stats = judge(session, room, MockLLMClient([{"verdicts": [verdict(p1, 0, "consistent"), verdict(p2, 1, "insufficient_evidence")]}])).run([p1, p2])
    assert [f.classification for f in session.scalars(select(Finding))] == ["consistent"] and stats.absorbed == 1


# ---- 6: consistent findings are grouped by pair of documents and attribute -------------------------------------
def test_consistent_findings_between_two_documents_about_one_attribute_become_one(session, room):  # noqa: F811
    d1, c1 = add_doc(session, "left.pdf", "pdf", [dict(content=f"Category of T{i} is Alpha{i}.", page_number=i + 1) for i in range(3)])
    d2, c2 = add_doc(session, "right.pdf", "pdf", [dict(content=f"The category of T{i} is Alpha{i} here.", page_number=i + 1) for i in range(3)])
    pairs = []
    for i in range(3):
        a = Claim(document_id=d1.id, chunk_id=c1[i].id, subject=f"T{i}", metric="category", value=f"Alpha{i}", claim_text=f"Category of T{i} is Alpha{i}")
        b = Claim(document_id=d2.id, chunk_id=c2[i].id, subject=f"T{i}", metric="category", value=f"Alpha{i}", claim_text=f"The category of T{i} is Alpha{i} here")
        session.add_all([a, b])
        session.commit()
        pairs.append(exact(a, b))
    stats = judge(session, room, MockLLMClient([])).run(pairs)
    finding = session.scalars(select(Finding)).one()
    assert finding.classification == "consistent" and "agrees in 3 places" in finding.title and stats.grouped == 2 and stats.findings_saved == 1
    assert len(session.scalars(select(Evidence).where(Evidence.finding_id == finding.id)).all()) == 6
