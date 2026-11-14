"""Rows of one table are never compared with each other; agreements and every verdict category become findings."""
from __future__ import annotations

from sqlalchemy import select

from app.llm.mock import MockLLMClient
from app.models import Claim, Evidence, Finding
from app.schemas.claim import Candidate
from app.services.candidate_matching import compare_attributes, generate_candidates, relation_hint, table_key
from app.services.claim_grouping import ClaimGrouper
from app.services.numbers import value_supported_by_text
from tests.test_claims import view
from tests.test_claim_group_judge import comparison, make_claims  # noqa: F401
from tests.test_tools import add_doc, room  # noqa: F401  (fixture)


# ---- rows of one table -------------------------------------------------------
def test_claims_from_one_table_are_never_grouped_but_other_places_still_are():
    sheet = table_key("d1", "table_row", "Capex", None)
    rows = [view("a", subject="Tool_Capex", metric="Budgeted cost", value=185, table_key=sheet),
            view("b", subject="Tool_Capex", metric="Budgeted cost", value=34, table_key=sheet),
            view("c", doc="d2", subject="Tool_Capex", metric="Budgeted cost", value=190, table_key=table_key("d2", "table_row", "Capex", None))]
    keys = {frozenset((p.claim_id, p.candidate_id)) for p in generate_candidates(rows)}
    assert keys == {frozenset(("a", "c")), frozenset(("b", "c"))}  # a and b are rows of the same sheet
    assert table_key("d1", "paragraph", None, 3) is None and table_key("d1", "table", None, 3) == "d1||3"
    embeddings = {"a": [1.0, 0.0], "b": [1.0, 0.0]}
    assert generate_candidates(rows[:2], vectors=embeddings, min_subject_overlap=None) == []


def test_the_llm_grouping_never_candidates_rows_of_one_table(session, room):  # noqa: F811
    sheet, _ = add_doc(session, "sheet.xlsx", "xlsx", [dict(content="| LTH-001 | 185 |", sheet_name="Capex", content_type="table_row"),
                                                       dict(content="| ETCH-014 | 42 |", sheet_name="Capex", content_type="table_row")])
    from app.models import DocumentChunk
    chunks = [c.id for c in session.scalars(select(DocumentChunk).where(DocumentChunk.document_id == sheet.id).order_by(DocumentChunk.chunk_index))]
    for chunk_id, value in zip(chunks, ("185", "42")):
        session.add(Claim(document_id=sheet.id, chunk_id=chunk_id, subject="Capex", metric="cost", value=value, claim_text=f"cost {value}"))
    session.add(Claim(document_id=room.deck.id, chunk_id=room.deck_p1, subject="Capex", metric="cost", value="190", claim_text="cost 190"))
    session.commit()
    llm = MockLLMClient([{"groups": [{"refs": ["c1", "c2", "c3"], "topic": "cost"}]}])
    found, stats = ClaimGrouper(session, llm).run()
    assert len(found) == 2 and stats.candidates == 2  # only each sheet row against the deck claim


# ---- same sentence is not the same value -----------------------------------
def test_two_claims_read_from_one_sentence_are_not_agreement_when_their_values_differ():
    a = view("a", claim_text="Paid-up capital: INR 620 crore. Spent: INR 2,480 crore", value=620)
    b = view("b", doc="d2", claim_text="Paid-up capital: INR 620 crore. Spent: INR 2,480 crore", value=2480)
    assert generate_candidates([a, b])[0].relation_hint != "exact"
    c = view("c", doc="d3", claim_text="Paid-up capital: INR 620 crore. Spent: INR 2,480 crore", value=620)
    assert {p.relation_hint for p in generate_candidates([a, c])} == {"exact"}


# ---- the value check ignores units and separators --------------------------
def test_a_value_with_units_is_shown_by_the_same_numbers_and_words_in_any_formatting():
    assert value_supported_by_text("24,000 WSPM", "capacity of 24000 WSPM per month")
    assert value_supported_by_text("INR 2,480 crore", "Cumulative expenditure: INR 2480 crore")
    assert not value_supported_by_text("28 October 2026", "completed on 5 November 2026")
    assert not value_supported_by_text("24,000 WSPM", "capacity of 24000 tonnes")


# ---- all categories become findings ------------------------------------------






