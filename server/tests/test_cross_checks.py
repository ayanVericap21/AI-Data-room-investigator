"""Relations between figures of different documents: the model proposes them, code does the arithmetic and the date comparison."""
from __future__ import annotations

from sqlalchemy import select

from app.llm.errors import LLMTransientError
from app.llm.mock import MockLLMClient
from app.models import Claim, Evidence, Finding
from app.services.cross_checks import SYSTEM, CrossChecker, CrossRelations
from app.tools import build_registry
from tests.test_tools import add_doc, room  # noqa: F401  (fixtures and helper)


def make(session, name, text, **claim):
    doc, chunks = add_doc(session, name, "pdf", [dict(content=text, page_number=1)])
    base = dict(subject="s", metric="m", unit=None, claim_text=text, claim_category="quantity")
    base.update(claim)
    c = Claim(document_id=doc.id, chunk_id=chunks[0].id, **base)
    session.add(c)
    session.commit()
    return c


def credit_room(session, credit="$5,120,000"):
    """c1 the costs (in thousands of dollars), c2 the rate, c3 the credit: refs follow the file names."""
    make(session, "a_model.pdf", "Qualified rehabilitation expenses: 24,600 ($ in thousands).", subject="Qualified rehabilitation expenses", metric="amount", value="24,600", unit="$ in thousands")
    make(session, "b_rate.pdf", "Federal historic tax credit rate: 20%.", subject="Federal historic tax credit", metric="rate", value="20%")
    make(session, "c_credit.pdf", f"Federal historic tax credit: {credit}.", subject="Federal historic tax credit", metric="amount", value=credit)


def run(session, room, answers, **kw):  # noqa: F811
    llm = MockLLMClient(answers)
    checker = CrossChecker(session, llm, build_registry(session, None, room.investigation.id), **kw)
    return checker.run(), llm


CREDIT = {"label": "Federal credit equals the credit rate times the qualified costs", "kind": "amount", "terms": [{"ref": "c1", "times": 1, "rate": "c2"}],
          "relation": "equals", "against": "c3", "reason": "The model states the federal credit rate as 20%."}


def test_a_credit_that_is_not_the_rate_times_the_costs_is_a_finding_with_the_arithmetic_done_in_code(session, room):  # noqa: F811
    credit_room(session)
    stats, llm = run(session, room, [{"checks": [CREDIT]}])
    finding = session.scalars(select(Finding)).one()
    assert (stats.proposed, stats.checked, stats.broken, stats.saved) == (1, 1, 1, 1)
    assert finding.classification == "contradiction" and finding.severity in ("high", "critical") and "4,920,000" in finding.description  # 20% of 24,600 thousands
    assert "5,120,000" in finding.description and "3.9% of the stated figure" in finding.description
    assert len(session.scalars(select(Evidence)).all()) == 3  # the costs, the rate and the credit, each quoted from its own document


def test_a_credit_that_matches_the_rate_times_the_costs_holds_and_saves_nothing(session, room):  # noqa: F811
    credit_room(session, credit="$4,920,000")
    stats, _ = run(session, room, [{"checks": [CREDIT]}])
    assert (stats.checked, stats.held, stats.broken, stats.saved) == (1, 1, 0, 0) and session.scalars(select(Finding)).all() == []


def test_a_figure_in_thousands_equals_the_same_figure_in_dollars(session, room):  # noqa: F811
    make(session, "a_model.pdf", "QRE: 24,600 ($ in thousands).", value="24,600", unit="$ in thousands")
    make(session, "b_cert.pdf", "QRE certified: $24,600,000.", value="$24,600,000", unit="USD")
    same = {"label": "QRE in the model equals the certified QRE", "kind": "amount", "terms": [{"ref": "c1"}], "relation": "equals", "against": "c2"}
    stats, _ = run(session, room, [{"checks": [same]}])
    assert (stats.held, stats.broken) == (1, 0)


def test_a_sum_of_parts_stated_in_other_documents_must_match_the_total(session, room):  # noqa: F811
    make(session, "a_part.pdf", "Hard costs: $22,540,000.", value="$22,540,000", unit="USD")
    make(session, "b_part.pdf", "Soft costs: $2,770,000.", value="$2,770,000", unit="USD")
    make(session, "c_total.pdf", "Total qualified costs: $25,310,000.", subject="Total qualified costs", value="$25,310,000", unit="USD")
    total = {"label": "Hard and soft costs add up to the total", "kind": "amount", "terms": [{"ref": "c1"}, {"ref": "c2"}], "relation": "equals", "against": "c3"}
    assert run(session, room, [{"checks": [total]}])[0].held == 1
    wrong = {**total, "label": "Hard and soft costs add up to the total (wrong total)", "against": "c3"}
    make(session, "d_total.pdf", "Total qualified costs: $24,600,000.", subject="Total qualified costs", value="$24,600,000", unit="USD")
    wrong["against"] = "c4"
    stats, _ = run(session, room, [{"checks": [wrong]}])
    assert stats.broken == 1 and stats.saved == 1 and "25,310,000" in session.scalars(select(Finding)).one().description


def test_a_payment_that_arrives_after_the_loan_it_repays_matures_is_a_finding(session, room):  # noqa: F811
    make(session, "a_budget.pdf", "Equity bridge loan matures 3/31/2026.", subject="Equity bridge loan", metric="maturity date", value="3/31/2026", claim_category="date")
    make(session, "b_model.pdf", "Installment 3 received June 30, 2026.", subject="Installment 3", metric="receipt date", value="June 30, 2026", claim_category="date")
    order = {"label": "The installment that repays the bridge loan arrives before it matures", "kind": "order", "earlier": "c2", "later": "c1", "reason": "The loan is repaid from installment 3."}
    stats, _ = run(session, room, [{"checks": [order]}])
    finding = session.scalars(select(Finding)).one()
    assert (stats.checked, stats.broken, stats.saved) == (1, 1, 1) and finding.severity == "high"
    assert "June 30, 2026" in finding.description and "3/31/2026" in finding.description


def test_dates_in_the_right_order_hold(session, room):  # noqa: F811
    make(session, "a_one.pdf", "Approval August 22, 2023.", value="August 22, 2023", claim_category="date")
    make(session, "b_two.pdf", "Work started October 2, 2023.", value="October 2, 2023", claim_category="date")
    stats, _ = run(session, room, [{"checks": [{"label": "Approval comes before the work", "kind": "order", "earlier": "c1", "later": "c2"}]}])
    assert (stats.held, stats.broken, stats.saved) == (1, 0, 0)


def test_a_relation_code_cannot_check_is_counted_and_never_saved(session, room):  # noqa: F811
    make(session, "a_one.pdf", "Contract sum $21,900,000.", value="$21,900,000", unit="USD")
    make(session, "b_two.pdf", "Fee within 30 days of handover.", value="within 30 days of handover", claim_category="date")
    make(session, "c_three.pdf", "Fee schedule: see exhibit.", value="see exhibit")
    checks = [
        {"label": "an unknown ref", "kind": "amount", "terms": [{"ref": "c99"}], "against": "c1"},
        {"label": "a claim against itself", "kind": "amount", "terms": [{"ref": "c1"}], "against": "c1"},
        {"label": "a value that is not a number", "kind": "amount", "terms": [{"ref": "c3"}], "against": "c1"},
        {"label": "a relative date", "kind": "order", "earlier": "c2", "later": "c1"},
        {"label": "a rate that is not a claim", "kind": "amount", "terms": [{"ref": "c1", "rate": "c42"}], "against": "c1"},
    ]
    stats, _ = run(session, room, [{"checks": checks}])
    assert (stats.proposed, stats.unusable, stats.checked, stats.saved) == (5, 5, 0, 0)


def test_only_figures_and_dates_are_shown_to_the_model(session, room):  # noqa: F811
    credit_room(session)
    make(session, "d_text.pdf", "The Managing Member shall deliver audited statements.", value="annual", claim_category="statement")
    _stats, llm = run(session, room, [{"checks": []}])
    assert "Managing Member" not in llm.prompts[0] and llm.prompts[0].count("Claim") + llm.prompts[0].count("| value:") >= 3
    assert "### Document: a_model.pdf" in llm.prompts[0] and "d_text.pdf" not in llm.prompts[0]


def test_a_relation_whose_passages_are_already_in_a_finding_is_not_saved_twice(session, room):  # noqa: F811
    credit_room(session)
    first, _ = run(session, room, [{"checks": [CREDIT]}])
    second, _ = run(session, room, [{"checks": [CREDIT]}])
    assert first.saved == 1 and second.saved == 0 and second.already_reported == 1 and len(session.scalars(select(Finding)).all()) == 1


def test_a_failed_call_is_counted_and_the_run_goes_on(session, room):  # noqa: F811
    credit_room(session)
    llm = MockLLMClient([LLMTransientError("503")] * 3, max_attempts=3)
    stats = CrossChecker(session, llm, build_registry(session, None, room.investigation.id)).run()
    assert (stats.calls, stats.failed, stats.saved) == (1, 1, 0) and stats.errors


def test_the_schema_goes_to_gemini_as_a_response_schema_and_the_prompt_stays_short():
    from app.llm.google import gemini_schema

    schema = gemini_schema(CrossRelations.model_json_schema())
    check = schema["properties"]["checks"]["items"]
    assert check["properties"]["kind"]["enum"] == ["amount", "order"] and check["properties"]["relation"]["enum"] == ["equals", "at_most", "at_least"]
    assert len(SYSTEM) < 10_000


def test_a_memo_row_is_not_one_of_the_parts_of_the_total_beside_it(session, room):  # noqa: F811
    make(session, "a_loan.pdf", "Construction loan: $18,500,000.", value="$18,500,000", unit="USD")
    make(session, "b_bridge.pdf", "Equity bridge loan $1,700,000, matures 3/31/2026 | (memo)", subject="Equity bridge loan", value="$1,700,000", unit="USD")
    make(session, "c_total.pdf", "Total sources: $18,500,000.", subject="Total sources", value="$18,500,000", unit="USD")
    sources = {"label": "Total sources sum", "kind": "amount", "terms": [{"ref": "c1"}, {"ref": "c2"}], "relation": "equals", "against": "c3"}
    stats, _ = run(session, room, [{"checks": [sources]}])
    assert (stats.unusable, stats.checked, stats.saved) == (1, 0, 0) and session.scalars(select(Finding)).all() == []


def test_a_span_is_not_an_amount_in_a_cross_document_relation(session, room):  # noqa: F811
    make(session, "a_photos.pdf", "Photographs 1 to 3.", value="1 to 3")
    make(session, "b_photos.pdf", "Photographs 4 to 6.", value="4 to 6")
    make(session, "c_photos.pdf", "Photographs 1 to 14 in total.", value="1 to 14")
    spans = {"label": "photograph spans add up to the total range", "kind": "amount", "terms": [{"ref": "c1"}, {"ref": "c2"}], "relation": "equals", "against": "c3"}
    stats, _ = run(session, room, [{"checks": [spans]}])
    assert (stats.unusable, stats.saved) == (1, 0)


def test_a_model_proposed_sum_is_never_critical_however_large_the_gap(session, room):  # noqa: F811
    make(session, "a_total.pdf", "Total qualified costs: $1,000,000.", subject="Total qualified costs", value="$1,000,000", unit="USD")
    make(session, "b_total.pdf", "Total qualified costs certified: $2,000,000.", subject="Total qualified costs", value="$2,000,000", unit="USD")
    same = {"label": "the two totals agree", "kind": "amount", "terms": [{"ref": "c1"}], "relation": "equals", "against": "c2"}
    stats, _ = run(session, room, [{"checks": [same]}])
    assert stats.saved == 1 and session.scalars(select(Finding)).one().severity == "high"  # a 50% gap on a total: high, not critical
