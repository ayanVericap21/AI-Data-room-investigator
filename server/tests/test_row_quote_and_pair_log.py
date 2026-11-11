"""A claim read from a table row is quoted from that row; every candidate pair ends with a recorded outcome."""
from sqlalchemy import select

from app.llm.mock import MockLLMClient
from app.models import Claim, Evidence, Finding
from app.schemas.claim import CandidatePair
from app.services.pair_judge import PairJudge
from app.tools import build_registry
from tests.test_pair_judge import comparison, pair
from tests.test_tools import add_doc, room  # noqa: F401  (fixtures and helper)

ROWS = ("[Installed_Tools!A8:F8] Tool_ID: CLEAN-011; Installed_Cost_INR_Crore: 19\n"
        "[Installed_Tools!A9:F9] Tool_ID: TEST-021; Installed_Cost_INR_Crore: 24")
OTHER = "[Tool_Capex_Summary!A8:D8] Tool_ID: CLEAN-011; Budgeted_Cost_INR_Crore: 19\n[Tool_Capex_Summary!A9:D9] Tool_ID: TEST-021; Budgeted_Cost_INR_Crore: 24"


def table_claims(session):
    reg, rows_a = add_doc(session, "tool_register.xlsx", "xlsx", [dict(content=ROWS, sheet_name="Installed_Tools", cell_reference="A8:F9", content_type="table_row")])
    cap, rows_b = add_doc(session, "capex_summary.xlsx", "xlsx", [dict(content=OTHER, sheet_name="Tool_Capex_Summary", cell_reference="A8:D9", content_type="table_row")])
    a = Claim(document_id=reg.id, chunk_id=rows_a[0].id, subject="CLEAN-011", metric="Installed_Cost_INR_Crore", value="19", claim_text="Tool_ID: CLEAN-011; Installed_Cost_INR_Crore: 19")
    b = Claim(document_id=cap.id, chunk_id=rows_b[0].id, subject="CLEAN-011", metric="Budgeted_Cost_INR_Crore", value="29", claim_text="x" * 10)
    session.add_all([a, b])
    session.commit()
    return a, b


def run(session, room, a, b, verdict):  # noqa: F811
    p = pair(a, b)
    llm = MockLLMClient([{"verdicts": [dict(pair_index=0, classification="contradiction", severity="high", confidence=0.9, title="Cost differs",
                                            description="d", **verdict(p, a))]}])
    judge = PairJudge(session, llm, build_registry(session, None, room.investigation.id))
    return judge.run([p]), p


def quotes(p, a, qa, qb):
    return dict(quote_a=qa, quote_b=qb) if p.claim_a_id == a.id else dict(quote_a=qb, quote_b=qa)


def test_a_quote_from_another_row_of_the_same_chunk_is_refused(session, room):  # noqa: F811
    a, b = table_claims(session)
    b.value = "24"  # the model quoted the neighbouring row (TEST-021, 24) for the claim about CLEAN-011
    session.commit()
    stats, p = run(session, room, a, b, lambda p, a: quotes(p, a, "Installed_Cost_INR_Crore: 19", "Budgeted_Cost_INR_Crore: 24"))
    assert stats.findings_saved == 0 and stats.findings_rejected == 1 and "different row" in stats.rejections[0]
    assert "REJECTED: the quote is from a different row" in stats.records[(p.claim_a_id, p.claim_b_id)].outcome


def test_a_quote_from_the_right_row_without_its_label_is_widened_to_the_whole_row(session, room):  # noqa: F811
    a, b = table_claims(session)
    b.value = "19"
    session.commit()
    stats, _ = run(session, room, a, b, lambda p, a: quotes(p, a, "Installed_Cost_INR_Crore: 19", "Budgeted_Cost_INR_Crore: 19"))
    assert stats.findings_saved == 1
    stored = {e.quoted_text for e in session.scalars(select(Evidence))}
    assert stored == {"[Installed_Tools!A8:F8] Tool_ID: CLEAN-011; Installed_Cost_INR_Crore: 19",
                      "[Tool_Capex_Summary!A8:D8] Tool_ID: CLEAN-011; Budgeted_Cost_INR_Crore: 19"}  # each names its row
    assert session.scalars(select(Finding)).one().classification == "contradiction"


def test_every_pair_gets_an_outcome_agreements_judged_pairs_and_pairs_the_model_skipped(session, room):  # noqa: F811
    a, b = table_claims(session)
    other = Claim(document_id=room.deck.id, chunk_id=room.deck_p1, subject="CLEAN-011", metric="capacity", value=120000, claim_text="Annual production capacity is 120,000 tonnes per year")
    session.add(other)
    session.commit()
    exact = CandidatePair(claim_a_id=a.id, claim_b_id=other.id, topic_score=1.0, relation_hint="exact", comparison=comparison(value_relation="equal"), matched_by="llm")
    skipped = pair(b, other)
    llm = MockLLMClient([{"verdicts": []}, {"verdicts": []}])  # no verdict, and none on the single-pair retry either
    stats = PairJudge(session, llm, build_registry(session, None, room.investigation.id)).run([exact, skipped])
    outcomes = [r.outcome for r in stats.records.values()]
    assert len(outcomes) == 2 and all(outcomes) and stats.retries == 1
    assert any(o.startswith("agreement") for o in outcomes) and any(o.startswith("no verdict returned") for o in outcomes)
    first = next(iter(stats.records.values()))
    assert first.claim_a.count("|") == 3 and first.route == "llm" and first.hint == "exact"
