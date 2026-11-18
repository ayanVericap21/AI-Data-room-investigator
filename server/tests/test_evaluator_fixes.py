"""Rules decided in code rather than left to the judge: severity, internal arithmetic, relative-date anchors, basis, currency, percentage base, duplicates."""
from __future__ import annotations

from sqlalchemy import select

from app.llm.mock import MockLLMClient
from app.models import Claim, Finding
from app.services.aliases import defined_names, resolve
from app.services.internal_checks import InternalChecker, clause_quote
from app.services.severity import contradiction_severity, missing_severity
from app.tools import build_registry
from tests.test_claim_group_judge import answer, candidate, judge, verdict
from tests.test_tools import add_doc, room  # noqa: F401  (fixtures and helper)


def claim_in(session, filename, text, value, subject="Harbor Point racking", metric="amount", category="quantity", **kw):
    doc, chunks = add_doc(session, filename, "pdf", [dict(content=text, page_number=1)])
    c = Claim(document_id=doc.id, chunk_id=chunks[0].id, subject=subject, metric=metric, value=value, claim_text=text, claim_category=category, **kw)
    session.add(c)
    session.commit()
    return c


# ---- severity by rule -------------------------------------------------------------------------------------------------------------------
def test_severity_follows_the_gap_and_how_prominent_the_claim_is(session, room):  # noqa: F811
    big = [claim_in(session, "a.pdf", "The contract value is GBP 2.5 million.", "GBP 2.5 million", metric="contract value"),
           claim_in(session, "b.pdf", "The contract value is GBP 2.8 million.", "GBP 2.8 million", metric="contract value")]
    small = [claim_in(session, "c.pdf", "Paid 1,000 units.", "1,000", metric="units"), claim_in(session, "d.pdf", "Paid 1,004 units.", "1,004", metric="units")]
    assert contradiction_severity(big) == "critical"  # a gap over 5% in a contract value
    assert contradiction_severity(small) == "low"  # a gap under 1% in an ordinary figure
    assert missing_severity("The Supplier shall not use subcontractors without prior written consent") == "high" and missing_severity("See the appendix") == "medium"


# ---- names the documents define ---------------------------------------------------------------------------------------------------------
def test_a_defined_role_and_its_short_form_resolve_to_the_full_name():
    names = defined_names(['This agreement is made on 3 February 2025 between Calder Foods Limited ("the Customer") and Northwind Logistics Ltd ("the Supplier" or "NWL").'])
    assert names == {"customer": "Calder Foods Limited", "supplier": "Northwind Logistics Ltd", "nwl": "Northwind Logistics Ltd"}
    assert resolve("the Supplier liability and NWL crews", names) == "Northwind Logistics Ltd liability and Northwind Logistics Ltd crews"


# ---- arithmetic inside a document -------------------------------------------------------------------------------------------------------
SPEND = "Average monthly spend is GBP 410,000 across March and April, giving total spend to date of GBP 900,000."


def test_a_clause_is_cut_at_the_delimiters_around_a_value():
    assert clause_quote(SPEND, "GBP 410,000") == "Average monthly spend is GBP 410,000 across March and April"
    assert clause_quote(SPEND, "GBP 900,000") == "giving total spend to date of GBP 900,000"


RUNNING = "Spend runs at GBP 410,000 each month across March and April, against a ceiling of GBP 900,000."  # no "average" or "total": only the model's relation can check it


def run_checks(session, room, relation):  # noqa: F811
    doc, chunks = add_doc(session, "board.pdf", "pdf", [dict(content=RUNNING, page_number=1)])
    for value, metric in (("GBP 410,000", "average monthly spend"), ("GBP 900,000", "total spend")):
        session.add(Claim(document_id=doc.id, chunk_id=chunks[0].id, subject="Harbor Point", metric=metric, value=value, claim_text=RUNNING, claim_category="quantity"))
    session.commit()
    return InternalChecker(session, MockLLMClient([{"checks": [relation]}]), build_registry(session, None, room.investigation.id)).run()


def test_an_average_times_its_periods_that_is_not_the_stated_total_is_a_finding(session, room):  # noqa: F811
    stats = run_checks(session, room, {"label": "monthly spend times periods", "terms": [{"ref": "c1", "times": 2}], "relation": "equals", "against": "c2"})
    finding = session.scalars(select(Finding)).one()
    assert (stats.proposed, stats.checked, stats.broken, stats.saved) == (1, 1, 1, 1) and finding.classification == "contradiction" and "820,000" in finding.description


def test_a_relation_that_holds_saves_nothing(session, room):  # noqa: F811
    stats = run_checks(session, room, {"label": "ok", "terms": [{"ref": "c1", "times": 2.0}], "relation": "at_most", "against": "c2"})  # 820,000 cannot exceed 900,000: true
    assert (stats.checked, stats.held, stats.saved) == (1, 1, 0) and session.scalar(select(Finding)) is None


# ---- a relative date needs its anchor ---------------------------------------------------------------------------------------------------
def test_a_date_relative_to_an_event_that_no_claim_dates_is_missing_information(session, room):  # noqa: F811
    claim_in(session, "mail.pdf", "The retention release is due within 30 days of handover.", "within 30 days of handover", metric="retention release", category="date")
    stats = InternalChecker(session, MockLLMClient([{"checks": []}]), build_registry(session, None, room.investigation.id)).run()
    finding = session.scalars(select(Finding)).one()
    assert (stats.anchors_checked, stats.anchors_missing, stats.saved) == (1, 1, 1) and finding.classification == "missing_information"


def test_a_relative_date_whose_event_is_dated_elsewhere_is_fine(session, room):  # noqa: F811
    claim_in(session, "mail.pdf", "The retention release is due within 30 days of handover.", "within 30 days of handover", metric="retention release", category="date")
    claim_in(session, "cert.pdf", "Handover took place on 8 July 2025.", "8 July 2025", subject="handover", metric="handover date", category="date")
    stats = InternalChecker(session, MockLLMClient([{"checks": []}]), build_registry(session, None, room.investigation.id)).run()
    assert (stats.anchors_checked, stats.anchors_missing, stats.saved) == (1, 0, 0)


# ---- the judge's code rules -------------------------------------------------------------------------------------------------------------
def test_a_contracted_and_an_actual_figure_are_a_contextual_difference_decided_without_the_model(session, room):  # noqa: F811
    plan = claim_in(session, "msa.pdf", "Works commence on 1 March 2025 for a term of 6 months.", "1 March 2025", metric="start date", category="date")
    actual = claim_in(session, "progress.pdf", "Works commenced on 3 March 2025 according to the site log.", "3 March 2025", metric="start date", category="date")
    llm = MockLLMClient([])
    stats = judge(session, room, llm).run([candidate(plan, actual)])
    finding = session.scalars(select(Finding)).one()
    assert llm.prompts == [] and stats.groups_basis == 1 and finding.classification == "contextual_difference" and finding.severity == "low"


def test_a_percentage_set_against_a_count_with_no_stated_base_is_missing_information(session, room):  # noqa: F811
    pct = claim_in(session, "close.pdf", "Defects at handover: 3% of installed bays.", "3%", metric="defects", category="quantity")
    cnt = claim_in(session, "msa.pdf", "Defects at handover shall not exceed 40 bays.", "40 bays", metric="defects limit", category="quantity")
    quotes = {"3%": "Defects at handover: 3% of installed bays.", "40 bays": "Defects at handover shall not exceed 40 bays."}
    stats = judge(session, room, MockLLMClient([answer(verdict([pct, cnt], cls="consistent", quote=lambda v: quotes[v]))])).run([candidate(pct, cnt)])
    finding = session.scalars(select(Finding)).one()
    assert stats.base_missing == 1 and finding.classification == "missing_information" and "not stated" in finding.description and "within" not in finding.description


def test_two_claims_for_two_periods_are_judged_not_dropped_as_a_series(session, room):  # noqa: F811
    a = claim_in(session, "a.pdf", "Balance at 31 May 2025: GBP 384,000.", "GBP 384,000", metric="balance", period="31 May 2025")
    b = claim_in(session, "b.pdf", "Balance at 15 September 2025: nil.", "nil", metric="balance", period="15 September 2025")
    llm = MockLLMClient([{"verdicts": []}] * 20)
    stats = judge(session, room, llm).run([candidate(a, b)])
    assert stats.groups_series == 0 and llm.prompts


def test_the_same_values_from_the_same_passages_are_one_finding_and_a_conflicting_verdict_is_flagged(session, room):  # noqa: F811
    a = claim_in(session, "a.pdf", "The cap is 150% of the contract value.", "150%", metric="liability cap")
    b = claim_in(session, "b.pdf", "The cap remains at 120% of the contract value.", "120%", metric="liability cap")
    q = {"150%": "The cap is 150% of the contract value.", "120%": "The cap remains at 120% of the contract value."}
    verdicts = [verdict([a, b], quote=lambda v: q[v]), verdict([a, b], quote=lambda v: q[v]), verdict([a, b], cls="contextual_difference", quote=lambda v: q[v])]
    stats = judge(session, room, MockLLMClient([answer(*verdicts)])).run([candidate(a, b)])
    assert stats.findings_saved == 1 and stats.duplicates_skipped == 1 and len(stats.conflicts) == 1


def test_a_span_of_items_is_not_an_amount_so_adding_spans_is_never_a_finding(session, room):  # noqa: F811
    text = "Photographs 1 to 3, 4 to 6 and 7 are on the sheets; the application has photographs 1 to 14."
    doc, chunks = add_doc(session, "photos.pdf", "pdf", [dict(content=text, page_number=1)])
    for value in ("1 to 3", "4 to 6", "1 to 14"):
        session.add(Claim(document_id=doc.id, chunk_id=chunks[0].id, subject="photographs", metric=f"photographs {value}", value=value, claim_text=text, claim_category="quantity"))
    session.commit()
    relation = {"label": "photograph spans equal the total range", "terms": [{"ref": "c1"}, {"ref": "c2"}], "relation": "equals", "against": "c3"}
    stats = InternalChecker(session, MockLLMClient([{"checks": [relation]}]), build_registry(session, None, room.investigation.id)).run()
    assert (stats.proposed, stats.checked, stats.saved) == (1, 0, 0) and session.scalar(select(Finding)) is None
