"""A group of three or more claims gets one verdict. When it is called consistent or not comparable but its claims state different values, the pairs that differ are judged alone."""
from __future__ import annotations

from sqlalchemy import select

from app.llm.mock import MockLLMClient
from app.models import Finding
from tests.test_claim_group_judge import answer, candidates_of, judge, short, verdict, warehouse_claims
from tests.test_tools import room  # noqa: F401  (fixture)


def test_a_cluster_called_consistent_with_one_claim_that_differs_has_that_pair_judged_alone(session, room):  # noqa: F811
    claims = warehouse_claims(session, [500, 500, 500, 450])  # three agree, one differs
    pair = [claims[0], claims[3]]
    llm = MockLLMClient([answer(short(claims)), answer(verdict(pair, title="Capacity differs"))])
    stats = judge(session, room, llm, save_consistent=False).run(candidates_of(claims))
    finding = session.scalars(select(Finding)).one()
    assert finding.title == "Capacity differs" and stats.pair_rechecks == 1 and len(llm.prompts) == 2
    assert "### Group 1: 2 claims" in llm.prompts[1]  # the second call holds the pair


def test_a_cluster_called_not_comparable_with_three_different_values_has_each_pair_judged(session, room):  # noqa: F811
    claims = warehouse_claims(session, [500, 450, 400])
    first = answer(short(claims, same_attribute=False, reasoning="different limits"))
    pairs = [[claims[0], claims[1]], [claims[0], claims[2]], [claims[1], claims[2]]]
    second = answer(*[verdict(p, start=1 + 2 * n, group=n + 1, title=f"Pair {n}") for n, p in enumerate(pairs)])
    llm = MockLLMClient([first, second])
    stats = judge(session, room, llm, save_consistent=False).run(candidates_of(claims))
    assert stats.pair_rechecks == 3 and stats.groups_judged == 4 and session.scalar(select(Finding).where(Finding.title == "Pair 1")) is not None


def test_a_cluster_whose_claims_all_agree_is_not_checked_again(session, room):  # noqa: F811
    claims = warehouse_claims(session, [500, 500, 500])
    llm = MockLLMClient([])
    stats = judge(session, room, llm, save_consistent=False).run(candidates_of(claims))
    assert stats.pair_rechecks == 0 and llm.prompts == []  # code decided: every claim states one value


def test_a_pair_group_is_never_checked_again(session, room):  # noqa: F811
    claims = warehouse_claims(session, [500, 450])
    llm = MockLLMClient([answer(short(claims))])
    stats = judge(session, room, llm, save_consistent=False).run(candidates_of(claims))
    assert stats.pair_rechecks == 0 and len(llm.prompts) == 1
