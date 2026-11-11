"""Pairing by the LLM: sorted claims in batches, groups expanded to pairs, nothing invented."""
from __future__ import annotations

from app.llm.mock import MockLLMClient
from app.models import Claim
from app.services.llm_pairing import LLMPairer
from tests.test_pair_judge import make_claims, room  # noqa: F401  (fixtures/helpers)


def pairer(session, llm, **kw):
    return LLMPairer(session, llm, **kw)


def refs_in(prompt: str) -> dict[str, str]:
    """ref -> text of its line, as shown to the model."""
    return {line.split(" | ")[0]: line for line in prompt.splitlines() if line.startswith("c") and " | " in line}


def test_a_group_becomes_a_pair_with_the_llm_route_and_both_claims_are_shown(session, room):
    a, b = make_claims(session, room)
    llm = MockLLMClient([{"groups": [{"refs": ["c1", "c2"], "topic": "capacity"}]}])
    pairs, stats = pairer(session, llm).run()
    assert len(pairs) == 1 and pairs[0].matched_by == "llm" and {pairs[0].claim_a_id, pairs[0].claim_b_id} == {a.id, b.id}
    assert (stats.claims, stats.batches, stats.llm_calls, stats.groups, stats.claims_grouped, stats.pairs) == (2, 1, 1, 1, 2, 1)
    shown = refs_in(llm.prompts[0])
    assert set(shown) == {"c1", "c2"} and any("deck.pdf" in v for v in shown.values()) and any("capex.xlsx" in v for v in shown.values())


def test_claims_are_sorted_by_label_before_they_are_shown(session, room):
    make_claims(session, room)
    session.add(Claim(document_id=room.lease.id, chunk_id=room.rent_1, subject="lease", metric="annual rent", value=1, unit="crore",
                      claim_text="Annual rent is 1.2 crore"))
    session.commit()
    llm = MockLLMClient([{"groups": []}])
    pairer(session, llm).run()
    shown = list(refs_in(llm.prompts[0]).values())
    metrics = [next(p for p in line.split(" | ") if p.startswith("metric: ")) for line in shown]
    # sorted by the label words the matching code uses (stopwords such as "annual" ignored): capacity < rent
    assert metrics == ["metric: capacity", "metric: capacity", "metric: annual rent"]


def test_groups_with_unknown_or_repeated_refs_never_invent_pairs(session, room):
    make_claims(session, room)
    llm = MockLLMClient([{"groups": [{"refs": ["c1", "c9"]}, {"refs": ["c1", "c2"]}, {"refs": ["c2", "c1"]}]}])
    pairs, stats = pairer(session, llm).run()
    assert len(pairs) == 1 and stats.unknown_refs == 1 and stats.groups == 1  # c9 ignored; a claim is in one group only


def test_batches_are_cut_by_size_and_a_failed_batch_does_not_stop_the_rest(session, room):
    make_claims(session, room)
    llm = MockLLMClient([{"groups": []}, {"groups": []}])
    pairs, stats = pairer(session, llm, max_chars_per_call=1).run()  # every claim in its own batch
    assert stats.batches == 2 and stats.llm_calls == 2 and pairs == []


def test_identical_statements_in_a_group_are_marked_exact(session, room):
    a, b = make_claims(session, room)
    b.claim_text, b.value = a.claim_text, a.value  # the same statement and the same value
    session.commit()
    llm = MockLLMClient([{"groups": [{"refs": ["c1", "c2"]}]}])
    pairs, _ = pairer(session, llm).run()
    assert pairs[0].relation_hint == "exact"


def add_claim(session, room, value, chunk, document, unit="m", **kw):
    claim = Claim(document_id=document.id, chunk_id=chunk, subject="warehouse", metric="clear height", value=value, unit=unit,
                  claim_text=kw.pop("claim_text", f"{value}m clear height"), **kw)
    session.add(claim)
    session.commit()
    return claim


def test_a_group_is_compared_by_distinct_value_not_claim_by_claim(session, room):
    # four claims, two distinct values: 8.5 twice and 9.5 twice
    c = [add_claim(session, room, 8.5, room.deck_p1, room.deck), add_claim(session, room, 8.5, room.deck_p2, room.deck),
         add_claim(session, room, 9.5, room.capex_current, room.capex), add_claim(session, room, "9.5", room.capex_expansion, room.capex)]
    llm = MockLLMClient([{"groups": [{"refs": ["c1", "c2", "c3", "c4"], "topic": "clear height"}]}])
    pairs, stats = pairer(session, llm).run()
    assert stats.pairs_all_combinations == 6 and stats.largest_group == 4 and stats.distinct_values == 2
    judged = [p for p in pairs if p.relation_hint != "exact"]
    assert len(judged) == 1  # one claim of each distinct value, compared once
    agreement = [p for p in pairs if p.relation_hint == "exact"]
    assert len(agreement) == 2 and len(pairs) == 3  # same-value claims are linked as agreement, never judged
    judged_ids = {judged[0].claim_a_id, judged[0].claim_b_id}
    assert len(judged_ids & {c[0].id, c[1].id}) == 1 and len(judged_ids & {c[2].id, c[3].id}) == 1


def test_the_same_number_in_another_unit_or_period_is_a_different_value(session, room):
    add_claim(session, room, 100, room.deck_p1, room.deck, period="annual", claim_text="100m a year")
    add_claim(session, room, 100, room.capex_current, room.capex, period="monthly", claim_text="100m a month")
    add_claim(session, room, 100, room.rent_1, room.lease, unit="kg", period="annual", claim_text="100kg a year")
    llm = MockLLMClient([{"groups": [{"refs": ["c1", "c2", "c3"]}]}])
    pairs, stats = pairer(session, llm).run()
    assert stats.distinct_values == 3 and len(pairs) == 3
    assert all(p.relation_hint != "exact" for p in pairs)  # nothing is agreement: periods or units differ


def test_the_same_range_written_two_ways_is_one_value(session, room):
    add_claim(session, room, "395-7,916", room.deck_p1, room.deck, claim_text="Units from 395 to 7,916 m2")
    add_claim(session, room, "395 to 7,916", room.capex_current, room.capex, claim_text="Unit sizes 395m2 up to 7,916m2")
    add_claim(session, room, "395-8,000", room.rent_1, room.lease, claim_text="Units from 395 to 8,000 m2")
    llm = MockLLMClient([{"groups": [{"refs": ["c1", "c2", "c3"]}]}])
    pairs, stats = pairer(session, llm).run()
    assert stats.distinct_values == 2  # the first two are the same range
    assert sorted(p.relation_hint == "exact" for p in pairs) == [False, True]  # one agreement link, one pair to judge


def test_a_group_with_a_single_ref_is_ignored_and_does_not_fail_the_batch(session, room):
    make_claims(session, room)
    llm = MockLLMClient([{"groups": [{"refs": ["c1"], "topic": "alone"}, {"refs": ["c1", "c2"], "topic": "capacity"}]}])
    pairs, stats = pairer(session, llm).run()
    assert stats.batches_failed == 0 and stats.groups == 1 and len(pairs) == 1  # the group of one is skipped, the valid group stays


def test_the_prompts_spell_out_the_exact_output_shape():
    from app.services.llama_claims import SYSTEM_PROMPT
    from app.services.llm_pairing import SYSTEM as GROUPING
    from app.services.pair_judge import SYSTEM as JUDGE

    assert "OUTPUT FORMAT (strict)" in GROUPING and "AT LEAST TWO refs" in GROUPING and '{"groups": []}' in GROUPING
    assert "OUTPUT FORMAT (strict)" in JUDGE and "never null" in JUDGE
    for name in ("pair_index", "same_attribute", "classification", "severity", "confidence", "quote_a", "quote_b"):
        assert name in JUDGE.split("OUTPUT FORMAT (strict)")[1]
    assert "OUTPUT FORMAT (strict)" in SYSTEM_PROMPT and "subject, metric, value and claim_text" in SYSTEM_PROMPT
