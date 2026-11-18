"""The model's arithmetic relations are checked by code: a scale stated in the unit is applied, and a rate, price or percentage is never added to an amount."""
from __future__ import annotations

from sqlalchemy import select

from app.llm.mock import MockLLMClient
from app.models import Claim, Finding
from app.services.internal_checks import InternalChecker
from app.tools import build_registry
from tests.test_tools import add_doc, room  # noqa: F401  (fixtures and helper)


def run(session, room, rows, relation):  # noqa: F811
    """rows: (value, unit, metric) for c1, c2, ... in one document; one relation proposed by the model."""
    text = " ".join(f"The {metric} is {value}{' (' + unit + ')' if unit else ''}." for value, unit, metric in rows)  # each figure in a sentence of its own
    doc, chunks = add_doc(session, "model.pdf", "pdf", [dict(content=text, page_number=1)])
    for value, unit, metric in rows:
        session.add(Claim(document_id=doc.id, chunk_id=chunks[0].id, subject="Model", metric=metric, value=value, unit=unit, claim_text=text, claim_category="quantity"))
    session.commit()
    return InternalChecker(session, MockLLMClient([{"checks": [relation]}]), build_registry(session, None, room.investigation.id)).run()


def test_a_figure_in_thousands_is_compared_in_whole_units(session, room):  # noqa: F811
    # 5,120,000 x 0.88 = 4,505,600, and "4,505.6 ($ in thousands)" is 4,505,600: the relation holds
    rows = [("$5,120,000", None, "federal credit"), ("4,505.6", "$ in thousands", "federal investor equity")]
    stats = run(session, room, rows, {"label": "equity", "terms": [{"ref": "c1", "times": 0.88}], "relation": "equals", "against": "c2"})
    assert (stats.checked, stats.held, stats.broken, stats.saved) == (1, 1, 0, 0) and session.scalar(select(Finding)) is None


def test_a_wrong_figure_in_thousands_is_still_a_finding_in_whole_units(session, room):  # noqa: F811
    # 24,600 ($ in thousands) x 0.2 = 4,920,000, not the $5,120,000 stated
    rows = [("24,600", "$ in thousands", "qualified expenses"), ("$5,120,000", None, "federal credit")]
    stats = run(session, room, rows, {"label": "federal credit", "terms": [{"ref": "c1", "times": 0.2}], "relation": "equals", "against": "c2"})
    finding = session.scalars(select(Finding)).one()
    assert (stats.checked, stats.broken, stats.saved) == (1, 1, 1) and "4,920,000" in finding.description


def test_a_price_added_to_an_amount_is_not_arithmetic(session, room):  # noqa: F811
    rows = [("$4,920,000", None, "projected credits"), ("$0.88", None, "price per credit"), ("$4,329,600", None, "capital contribution")]
    stats = run(session, room, rows, {"label": "capital contribution", "terms": [{"ref": "c1"}, {"ref": "c2"}], "relation": "equals", "against": "c3"})
    assert (stats.proposed, stats.unlike, stats.checked, stats.saved) == (1, 1, 0, 0) and session.scalar(select(Finding)) is None


def test_a_rate_added_to_a_product_is_not_arithmetic(session, room):  # noqa: F811
    rows = [("24,600", "$ in thousands", "qualified expenses"), ("0.85", None, "credit price"), ("5,227.5", "$ in thousands", "credit proceeds")]
    stats = run(session, room, rows, {"label": "proceeds", "terms": [{"ref": "c1", "times": 0.25}, {"ref": "c2"}], "relation": "equals", "against": "c3"})
    assert (stats.unlike, stats.checked, stats.saved) == (1, 0, 0)


def test_percentages_summed_against_an_amount_are_not_arithmetic(session, room):  # noqa: F811
    rows = [("20%", None, "installment 1"), ("30%", None, "installment 2"), ("40%", None, "installment 3"), ("$4,329,600", None, "capital contribution")]
    refs = [{"ref": f"c{i}"} for i in (1, 2, 3)]
    stats = run(session, room, rows, {"label": "capital contribution", "terms": refs, "relation": "equals", "against": "c4"})
    assert (stats.unlike, stats.checked, stats.saved) == (1, 0, 0)


def test_percentages_that_do_not_sum_to_the_stated_percentage_are_still_a_finding(session, room):  # noqa: F811
    rows = [("20%", None, "installment 1"), ("30%", None, "installment 2"), ("40%", None, "installment 3"), ("100%", None, "total")]
    refs = [{"ref": f"c{i}"} for i in (1, 2, 3)]
    stats = run(session, room, rows, {"label": "installment shares", "terms": refs, "relation": "equals", "against": "c4"})
    assert (stats.unlike, stats.checked, stats.broken) == (0, 1, 1)
