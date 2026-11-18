"""Grouping by the LLM: sorted claims in batches, groups expanded to candidates, nothing invented."""
from __future__ import annotations

from app.llm.mock import MockLLMClient
from app.models import Claim
from app.services.claim_grouping import ClaimGrouper
from tests.test_claim_group_judge import make_claims, room  # noqa: F401  (fixtures/helpers)


def grouper(session, llm, **kw):
    return ClaimGrouper(session, llm, **kw)


def refs_in(prompt: str) -> dict[str, str]:
    """ref -> text of its line, as shown to the model."""
    return {line.split(" | ")[0]: line for line in prompt.splitlines() if line.startswith("c") and " | " in line}


def test_a_group_becomes_a_candidate_with_the_llm_route_and_both_claims_are_shown(session, room):
    a, b = make_claims(session, room)
    llm = MockLLMClient([{"groups": [{"refs": ["c1", "c2"], "topic": "capacity"}]}])
    candidates, stats = grouper(session, llm).run()
    assert len(candidates) == 1 and candidates[0].matched_by == "llm" and {candidates[0].claim_id, candidates[0].candidate_id} == {a.id, b.id}
    assert (stats.claims, stats.batches, stats.llm_calls, stats.groups, stats.claims_grouped, stats.candidates) == (2, 1, 1, 1, 2, 1)
    shown = refs_in(llm.prompts[0])
    assert set(shown) == {"c1", "c2"} and any("deck.pdf" in v for v in shown.values()) and any("capex.xlsx" in v for v in shown.values())


def test_claims_are_shown_in_document_order(session, room):
    make_claims(session, room)
    session.add(Claim(document_id=room.lease.id, chunk_id=room.rent_1, subject="lease", metric="annual rent", value=1, unit="crore",
                      claim_text="Annual rent is 1.2 crore"))
    session.commit()
    llm = MockLLMClient([{"groups": []}])
    grouper(session, llm).run()
    shown = list(refs_in(llm.prompts[0]).values())
    files = [line.split(" | ")[1].split(",")[0] for line in shown]
    assert files == sorted(files) and len(files) == 3  # by file name (then by position in the file), not by label


def test_groups_with_unknown_or_repeated_refs_never_invent_candidates(session, room):
    make_claims(session, room)
    llm = MockLLMClient([{"groups": [{"refs": ["c1", "c9"]}, {"refs": ["c1", "c2"]}, {"refs": ["c2", "c1"]}]}])
    candidates, stats = grouper(session, llm).run()
    assert len(candidates) == 1 and stats.unknown_refs == 1 and stats.groups == 1  # c9 ignored; a claim is in one group only


def test_batches_are_cut_by_size_and_a_failed_batch_does_not_stop_the_rest(session, room):
    make_claims(session, room)
    llm = MockLLMClient([{"groups": []}, {"groups": []}])
    candidates, stats = grouper(session, llm, max_chars_per_call=1).run()  # every claim in its own batch
    assert stats.batches == 2 and stats.llm_calls == 2 and stats.groups == 0  # nothing came from the model (the code's word matching may still add a candidate)


def test_identical_statements_in_a_group_are_marked_exact(session, room):
    a, b = make_claims(session, room)
    b.claim_text, b.value = a.claim_text, a.value  # the same statement and the same value
    session.commit()
    llm = MockLLMClient([{"groups": [{"refs": ["c1", "c2"]}]}])
    candidates, _ = grouper(session, llm).run()
    assert candidates[0].relation_hint == "exact"


def add_claim(session, room, value, chunk, document, unit="m", **kw):
    claim = Claim(document_id=document.id, chunk_id=chunk, subject="warehouse", metric="clear height", value=value, unit=unit,
                  claim_text=kw.pop("claim_text", f"{value}m clear height"), **kw)
    session.add(claim)
    session.commit()
    return claim




def test_the_same_number_in_another_unit_or_period_is_a_different_value(session, room):
    add_claim(session, room, 100, room.deck_p1, room.deck, period="annual", claim_text="100m a year")
    add_claim(session, room, 100, room.capex_current, room.capex, period="monthly", claim_text="100m a month")
    add_claim(session, room, 100, room.rent_1, room.lease, unit="kg", period="annual", claim_text="100kg a year")
    llm = MockLLMClient([{"groups": [{"refs": ["c1", "c2", "c3"]}]}])
    candidates, stats = grouper(session, llm).run()
    assert stats.distinct_values == 3 and len(candidates) == 3
    assert all(p.relation_hint != "exact" for p in candidates)  # nothing is agreement: periods or units differ




def test_a_group_with_a_single_ref_is_ignored_and_does_not_fail_the_batch(session, room):
    make_claims(session, room)
    llm = MockLLMClient([{"groups": [{"refs": ["c1"], "topic": "alone"}, {"refs": ["c1", "c2"], "topic": "capacity"}]}])
    candidates, stats = grouper(session, llm).run()
    assert stats.batches_failed == 0 and stats.groups == 1 and len(candidates) == 1  # the group of one is skipped, the valid group stays




def test_claims_that_share_no_subject_or_metric_words_with_another_claim_are_not_sent_to_the_llm(session, room):
    make_claims(session, room)  # two capacity claims: partners of each other
    session.add(Claim(document_id=room.lease.id, chunk_id=room.rent_1, subject="lease", metric="annual rent", value=1, unit="crore", claim_text="Annual rent is 1.2 crore"))
    session.commit()
    llm = MockLLMClient([{"groups": []}])
    _, stats = ClaimGrouper(session, llm, prefilter=True).run()
    assert stats.claims == 3 and stats.claims_filtered == 1 and len(refs_in(llm.prompts[0])) == 2
    llm_all = MockLLMClient([{"groups": []}])
    _, unfiltered = ClaimGrouper(session, llm_all).run()
    assert unfiltered.claims_filtered == 0 and len(refs_in(llm_all.prompts[0])) == 3


# ---- the groups of earlier calls are carried forward ----------------------------------------------------------------------------------------
def llm_pairs(candidates, claims):
    names = {c.id: n for n, c in enumerate(claims, start=1)}
    return {frozenset((names[c.claim_id], names[c.candidate_id])) for c in candidates if c.matched_by == "llm"}


def test_a_later_call_is_shown_the_ungrouped_claims_of_the_earlier_one_and_can_group_with_them(session, room):  # noqa: F811
    from tests.test_claim_group_judge import warehouse_claims
    claims = warehouse_claims(session, [1, 2])
    llm = MockLLMClient([{"groups": []}, {"groups": [{"refs": ["c1", "u1"], "topic": "storage"}]}])
    candidates, stats = ClaimGrouper(session, llm, max_chars_per_call=1, carry_groups=True).run()  # one claim to a call
    assert "GROUPS OF EARLIER CALLS" not in llm.prompts[0] and "GROUPS OF EARLIER CALLS" in llm.prompts[1] and "\nu1 | doc0.pdf" in llm.prompts[1]
    assert llm_pairs(candidates, claims) == {frozenset((1, 2))} and stats.carried and stats.batches == 2 and stats.groups == 1


def test_a_claim_can_join_a_group_an_earlier_call_formed(session, room):  # noqa: F811
    from tests.test_claim_group_judge import warehouse_claims
    claims = warehouse_claims(session, [1, 2, 3])
    llm = MockLLMClient([{"groups": []}, {"groups": [{"refs": ["c1", "u1"], "topic": "storage"}]}, {"groups": [], "joins": [{"group": "g1", "refs": ["c1", "c7"]}]}])
    candidates, stats = ClaimGrouper(session, llm, max_chars_per_call=1, carry_groups=True).run()
    assert "\ng1 | doc1.pdf" in llm.prompts[2] and "group topic: storage | 2 claims" in llm.prompts[2]  # the group is shown by one example claim, its topic and its size
    assert llm_pairs(candidates, claims) == {frozenset((1, 2)), frozenset((1, 3)), frozenset((2, 3))}
    assert stats.joined == 1 and stats.claims_grouped == 3 and stats.largest_group == 3 and stats.unknown_refs == 1  # c7 does not exist


def test_without_the_flag_batches_are_independent_and_nothing_is_carried(session, room):  # noqa: F811
    from tests.test_claim_group_judge import warehouse_claims
    claims = warehouse_claims(session, [1, 2])
    llm = MockLLMClient([{"groups": []}, {"groups": []}])
    candidates, stats = ClaimGrouper(session, llm, max_chars_per_call=1).run()
    assert not any("GROUPS OF EARLIER CALLS" in prompt for prompt in llm.prompts) and not stats.carried and llm_pairs(candidates, claims) == set()


def test_a_failed_call_leaves_its_claims_ungrouped_for_the_next_one(session, room):  # noqa: F811
    from app.llm.errors import LLMPermanentError
    from tests.test_claim_group_judge import warehouse_claims
    claims = warehouse_claims(session, [1, 2])
    llm = MockLLMClient([LLMPermanentError("down"), {"groups": [{"refs": ["c1", "u1"], "topic": "storage"}]}])
    candidates, stats = ClaimGrouper(session, llm, max_chars_per_call=1, carry_groups=True).run()
    assert stats.batches_failed == 1 and "\nu1 | doc0.pdf" in llm.prompts[1] and llm_pairs(candidates, claims) == {frozenset((1, 2))}
