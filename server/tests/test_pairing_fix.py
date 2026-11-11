"""Rows of one table are never compared with each other; agreements and every verdict category become findings."""
from __future__ import annotations

from sqlalchemy import select

from app.llm.mock import MockLLMClient
from app.models import Claim, Evidence, Finding
from app.schemas.claim import CandidatePair
from app.services.candidate_matching import compare_attributes, generate_candidate_pairs, relation_hint, table_key
from app.services.llm_pairing import LLMPairer
from app.services.numbers import value_supported_by_text
from tests.test_claims import view
from tests.test_pair_judge import comparison, judge, make_claims, pair, verdict_for  # noqa: F401
from tests.test_tools import add_doc, room  # noqa: F401  (fixture)


# ---- rows of one table -------------------------------------------------------
def test_claims_from_one_table_are_never_paired_but_other_places_still_are():
    sheet = table_key("d1", "table_row", "Capex", None)
    rows = [view("a", subject="Tool_Capex", metric="Budgeted cost", value=185, table_key=sheet),
            view("b", subject="Tool_Capex", metric="Budgeted cost", value=34, table_key=sheet),
            view("c", doc="d2", subject="Tool_Capex", metric="Budgeted cost", value=190, table_key=table_key("d2", "table_row", "Capex", None))]
    keys = {frozenset((p.claim_a_id, p.claim_b_id)) for p in generate_candidate_pairs(rows)}
    assert keys == {frozenset(("a", "c")), frozenset(("b", "c"))}  # a and b are rows of the same sheet
    assert table_key("d1", "paragraph", None, 3) is None and table_key("d1", "table", None, 3) == "d1||3"
    embeddings = {"a": [1.0, 0.0], "b": [1.0, 0.0]}
    assert generate_candidate_pairs(rows[:2], vectors=embeddings, min_subject_overlap=None) == []


def test_the_llm_grouping_never_pairs_rows_of_one_table(session, room):  # noqa: F811
    sheet, _ = add_doc(session, "sheet.xlsx", "xlsx", [dict(content="| LTH-001 | 185 |", sheet_name="Capex", content_type="table_row"),
                                                       dict(content="| ETCH-014 | 42 |", sheet_name="Capex", content_type="table_row")])
    from app.models import DocumentChunk
    chunks = [c.id for c in session.scalars(select(DocumentChunk).where(DocumentChunk.document_id == sheet.id).order_by(DocumentChunk.chunk_index))]
    for chunk_id, value in zip(chunks, ("185", "42")):
        session.add(Claim(document_id=sheet.id, chunk_id=chunk_id, subject="Capex", metric="cost", value=value, claim_text=f"cost {value}"))
    session.add(Claim(document_id=room.deck.id, chunk_id=room.deck_p1, subject="Capex", metric="cost", value="190", claim_text="cost 190"))
    session.commit()
    llm = MockLLMClient([{"groups": [{"refs": ["c1", "c2", "c3"], "topic": "cost"}]}])
    pairs, stats = LLMPairer(session, llm).run()
    assert len(pairs) == 2 and stats.pairs_all_combinations == 3  # only each sheet row against the deck claim


# ---- same sentence is not the same value -----------------------------------
def test_two_claims_read_from_one_sentence_are_not_agreement_when_their_values_differ():
    a = view("a", claim_text="Paid-up capital: INR 620 crore. Spent: INR 2,480 crore", value=620)
    b = view("b", doc="d2", claim_text="Paid-up capital: INR 620 crore. Spent: INR 2,480 crore", value=2480)
    assert generate_candidate_pairs([a, b])[0].relation_hint != "exact"
    c = view("c", doc="d3", claim_text="Paid-up capital: INR 620 crore. Spent: INR 2,480 crore", value=620)
    assert {p.relation_hint for p in generate_candidate_pairs([a, c])} == {"exact"}


# ---- the value check ignores units and separators --------------------------
def test_a_value_with_units_is_shown_by_the_same_numbers_and_words_in_any_formatting():
    assert value_supported_by_text("24,000 WSPM", "capacity of 24000 WSPM per month")
    assert value_supported_by_text("INR 2,480 crore", "Cumulative expenditure: INR 2480 crore")
    assert not value_supported_by_text("28 October 2026", "completed on 5 November 2026")
    assert not value_supported_by_text("24,000 WSPM", "capacity of 24000 tonnes")


# ---- all categories become findings ------------------------------------------
def test_consistent_and_insufficient_evidence_verdicts_are_saved_when_they_carry_quotes(session, room):  # noqa: F811
    a, b = make_claims(session, room)
    p = pair(a, b)
    consistent = verdict_for(p, a, classification="consistent", severity="informational", title="Capacity agrees")
    stats = judge(session, room, MockLLMClient([consistent])).run([p])
    saved = session.scalars(select(Finding)).one()
    assert saved.classification == "consistent" and stats.findings_saved == 1
    roles = {e.evidence_role for e in session.scalars(select(Evidence).where(Evidence.finding_id == saved.id))}
    assert roles == {"supporting"}


def test_an_insufficient_evidence_verdict_without_quotes_is_only_counted(session, room):  # noqa: F811
    a, b = make_claims(session, room)
    p = pair(a, b)
    verdict = verdict_for(p, a, classification="insufficient_evidence", quote_a="", quote_b="")
    stats = judge(session, room, MockLLMClient([verdict])).run([p])
    assert stats.verdict_counts == {"insufficient_evidence": 1} and stats.findings_saved == 0 and stats.findings_rejected == 0


def test_the_same_value_in_two_documents_becomes_a_consistent_finding_without_an_llm_call(session, room):  # noqa: F811
    other, _ = add_doc(session, "other.pdf", "pdf", [dict(content="Annual production capacity is 120,000 tonnes per year.", page_number=1)])
    from app.models import DocumentChunk
    other_chunk = session.scalars(select(DocumentChunk).where(DocumentChunk.document_id == other.id)).one().id
    a = Claim(document_id=room.deck.id, chunk_id=room.deck_p1, subject="plant", metric="capacity", value="120,000", unit="tonnes",
              claim_text="Annual production capacity is 120,000 tonnes per year")
    b = Claim(document_id=other.id, chunk_id=other_chunk, subject="plant", metric="capacity", value="120,000", unit="tonnes",
              claim_text="Annual production capacity is 120,000 tonnes per year")
    same_doc = Claim(document_id=room.deck.id, chunk_id=room.deck_p2, subject="plant", metric="capacity", value="120,000", claim_text="x" * 10)
    session.add_all([a, b, same_doc])
    session.commit()
    pairs = [CandidatePair(claim_a_id=a.id, claim_b_id=b.id, topic_score=1.0, relation_hint="exact", comparison=comparison(value_relation="equal"),
                           matched_by="llm"),
             CandidatePair(claim_a_id=a.id, claim_b_id=same_doc.id, topic_score=1.0, relation_hint="exact",
                           comparison=comparison(value_relation="equal"), matched_by="llm")]
    stats = judge(session, room, MockLLMClient([])).run(pairs)  # any LLM call would fail the test
    finding = session.scalars(select(Finding)).one()  # the pair inside one document is not an agreement between documents
    assert (finding.classification, finding.severity, stats.agreement_findings, stats.llm_calls) == ("consistent", "informational", 1, 0)
    quotes = {e.quoted_text for e in session.scalars(select(Evidence).where(Evidence.finding_id == finding.id))}
    assert quotes == {"Annual production capacity is 120,000 tonnes per year"}
    assert "other.pdf" in finding.description and "deck.pdf" in finding.description


def test_the_judge_prompt_asks_for_quotes_on_every_verdict_that_is_comparable():
    from app.services.pair_judge import SYSTEM

    assert "consistent and insufficient_evidence too" in SYSTEM and "leave both quotes empty" not in SYSTEM
