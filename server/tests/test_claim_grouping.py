"""Grouping by the LLM: sorted claims in batches, each group one cluster, nothing invented."""
from __future__ import annotations

from app.llm.mock import MockLLMClient
from app.models import Claim
from app.services.claim_grouping import ClaimGrouper
from tests.test_claim_group_judge import make_claims, room  # noqa: F401  (fixtures/helpers)


def grouper(session, llm, **kw):
    return ClaimGrouper(session, llm, **kw)


def by_model(clusters):
    """The clusters that are the grouping model's own groups."""
    return [c for c in clusters if c.matched_by == "llm"]


def refs_in(prompt: str) -> dict[str, str]:
    """ref -> text of its line, as shown to the model."""
    return {line.split(" | ")[0]: line for line in prompt.splitlines() if line.startswith("c") and " | " in line}


def test_a_group_becomes_a_cluster_with_the_llm_route_and_both_claims_are_shown(session, room):
    a, b = make_claims(session, room)
    llm = MockLLMClient([{"groups": [{"refs": ["c1", "c2"], "topic": "capacity"}]}])
    clusters, stats = grouper(session, llm).run()
    assert len(by_model(clusters)) == 1 and set(by_model(clusters)[0].claim_ids) == {a.id, b.id} and by_model(clusters)[0].topic == "capacity"
    assert (stats.claims, stats.batches, stats.llm_calls, stats.groups, stats.claims_grouped) == (2, 1, 1, 1, 2) and stats.clusters >= 1
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


def test_groups_with_unknown_or_repeated_refs_never_invent_clusters(session, room):
    make_claims(session, room)
    llm = MockLLMClient([{"groups": [{"refs": ["c1", "c9"]}, {"refs": ["c1", "c2"]}, {"refs": ["c2", "c1"]}]}])
    clusters, stats = grouper(session, llm).run()
    assert len(by_model(clusters)) == 1 and stats.unknown_refs == 1 and stats.groups == 1  # c9 ignored; a claim is in one group only


def test_batches_are_cut_by_size_and_a_failed_batch_does_not_stop_the_rest(session, room):
    make_claims(session, room)
    llm = MockLLMClient([{"groups": []}, {"groups": []}])
    clusters, stats = grouper(session, llm, max_chars_per_call=1).run()  # every claim in its own batch
    assert stats.batches == 2 and stats.llm_calls == 2 and stats.groups == 0 and by_model(clusters) == []  # nothing came from the model (the code's word matching may still add a cluster)


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
    clusters, stats = grouper(session, llm).run()
    assert stats.distinct_values == 3 and len(by_model(clusters)) == 1 and len(by_model(clusters)[0].claim_ids) == 3  # nothing is agreement: periods or units differ




def test_a_group_with_a_single_ref_is_ignored_and_does_not_fail_the_batch(session, room):
    make_claims(session, room)
    llm = MockLLMClient([{"groups": [{"refs": ["c1"], "topic": "alone"}, {"refs": ["c1", "c2"], "topic": "capacity"}]}])
    clusters, stats = grouper(session, llm).run()
    assert stats.batches_failed == 0 and stats.groups == 1 and len(by_model(clusters)) == 1  # the group of one is skipped, the valid group stays




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


# ---- the batches are asked independently, at the same time ----------------------------------------------------------------------------------
def test_batches_are_asked_independently_and_in_parallel_without_the_groups_of_the_others(session, room):  # noqa: F811
    from tests.test_claim_group_judge import warehouse_claims
    warehouse_claims(session, [1, 2, 3])
    llm = MockLLMClient([{"groups": []}] * 3)
    clusters, stats = ClaimGrouper(session, llm, max_chars_per_call=1, workers=3).run()  # one claim to a call, three calls at the same time
    assert stats.batches == 3 and stats.llm_calls == 3 and len(llm.prompts) == 3
    assert not any("EARLIER" in prompt or "\nu1 |" in prompt or "\ng1 |" in prompt for prompt in llm.prompts)  # no call is shown what another one found
    assert all(refs_in(prompt).keys() == {"c1"} for prompt in llm.prompts)  # each call numbers only its own claims


def test_a_failed_batch_is_asked_again_as_two_independent_halves(session, room, monkeypatch):  # noqa: F811
    from app.llm.errors import LLMTransientError
    from tests.test_claim_group_judge import warehouse_claims
    monkeypatch.setattr("app.services.claim_grouping.MIN_SPLIT_CLAIMS", 1)
    claims = warehouse_claims(session, [1, 2, 3, 4])
    llm = MockLLMClient([LLMTransientError("503 high demand")] * 3 + [{"groups": [{"refs": ["c1", "c2"], "topic": "storage"}]}, {"groups": [{"refs": ["c1", "c2"], "topic": "storage"}]}])
    clusters, stats = ClaimGrouper(session, llm, retry_pause_seconds=0, sleep=lambda _s: None).run()  # one batch, which fails once
    assert stats.batches_split == 1 and stats.batches == 2 and stats.batches_failed == 0 and stats.groups == 2
    assert set(refs_in(llm.prompts[0])) == {"c1", "c2", "c3", "c4"} and set(refs_in(llm.prompts[-2])) == {"c1", "c2"} and set(refs_in(llm.prompts[-1])) == {"c1", "c2"}  # the whole batch, then each half numbered from c1
    assert not any("EARLIER" in prompt for prompt in llm.prompts)
    assert len({frozenset(c.claim_ids) for c in by_model(clusters)}) == 2  # each half formed its own cluster
    assert {i for c in by_model(clusters) for i in c.claim_ids} == {c.id for c in claims}


# ---- one stage: the model's groups, the rules, and the code's matching of the claims left ---------------------------------------------------------
def flagged(session, values):
    from tests.test_claim_group_judge import warehouse_claims
    claims = warehouse_claims(session, values)
    claims[0].assumptions = ["@rule: yes"]
    session.commit()
    return claims


def ids_of(clusters):
    return [i for c in clusters for i in c.claim_ids]


def test_flagged_claims_are_shown_in_a_rules_list_and_a_rule_group_is_a_rule_cluster(session, room):  # noqa: F811
    rule, rec_a, rec_b = flagged(session, ["no storage allowed", 500, 450])
    llm = MockLLMClient([{"groups": [{"refs": ["r1", "c1", "c2", "c1", "r9", "c9"], "topic": "storage ban"}]}])
    clusters, stats = grouper(session, llm).run()
    prompt = llm.prompts[0]
    assert "RULES (claims the extraction flagged as rules)" in prompt and "\nr1 | doc0.pdf" in prompt and "RULES list" in prompt
    assert "\nc1 | doc1.pdf" in prompt and "\nc2 | doc2.pdf" in prompt and "\nc3 |" not in prompt  # the rule is not also one of the claims
    assert [(c.rule_id, set(c.claim_ids), c.matched_by) for c in clusters if c.rule_id] == [(rule.id, {rule.id, rec_a.id, rec_b.id}, "rule")]
    assert stats.rules_shown == 1 and stats.rule_groups == 1 and stats.rule_clusters == 1 and stats.unknown_refs == 2 and stats.claims == 3
    assert clusters[0].claim_ids[0] == rule.id and set(clusters[0].via.values()) == {"rule"}


def test_every_call_is_shown_every_rule_and_a_rule_of_two_calls_is_one_cluster_with_the_records_of_both(session, room):  # noqa: F811
    rule, rec_a, rec_b = flagged(session, ["no storage allowed", 500, 450])
    llm = MockLLMClient([{"groups": [{"refs": ["r1", "c1"], "topic": "storage ban"}]}, {"groups": [{"refs": ["r1", "c1"], "topic": "storage ban"}]}])
    clusters, stats = grouper(session, llm, max_chars_per_call=1).run()  # one record to a call
    assert stats.batches == 2 and all("\nr1 | doc0.pdf" in prompt for prompt in llm.prompts)
    assert [(c.rule_id, c.claim_ids) for c in clusters] == [(rule.id, [rule.id, rec_a.id, rec_b.id])] and stats.rule_groups == 2 and stats.rule_clusters == 1


def test_without_flagged_claims_the_prompt_has_no_rules_list_and_a_group_of_a_rule_ref_is_unknown(session, room):  # noqa: F811
    from app.services.claim_grouping import SYSTEM, SYSTEM_WITH_RULES
    from tests.test_claim_group_judge import warehouse_claims
    warehouse_claims(session, ["no storage allowed", 500, 450])
    llm = MockLLMClient([{"groups": [{"refs": ["r1", "c1"], "topic": "x"}]}])
    clusters, stats = grouper(session, llm).run()
    assert "RULES (claims the extraction flagged" not in llm.prompts[0] and SYSTEM in llm.prompts[0] and SYSTEM_WITH_RULES not in llm.prompts[0]
    assert stats.rules_shown == 0 and stats.unknown_refs == 1 and not [c for c in clusters if "rule" in c.via.values()]  # the code's own negation route may still make a rule cluster, the model made none


def test_two_rules_of_one_attribute_in_one_group_are_a_cluster_of_claims_not_a_rule_cluster(session, room):  # noqa: F811
    from tests.test_claim_group_judge import warehouse_claims
    caps = warehouse_claims(session, [1500, 1000, 700])[:2]  # the third claim is not a rule: a call needs a claim to group
    for claim in caps:
        claim.assumptions = ["@rule: yes"]
    session.commit()
    llm = MockLLMClient([{"groups": [{"refs": ["r1", "r2"], "topic": "guaranty cap"}]}])
    clusters, stats = grouper(session, llm).run()
    assert stats.rules_shown == 2 and [(c.rule_id, set(c.claim_ids), c.matched_by) for c in by_model(clusters)] == [(None, {c.id for c in caps}, "llm")]


def test_clusters_are_disjoint_and_a_claim_the_model_grouped_is_never_in_a_code_cluster(session, room):  # noqa: F811
    from tests.test_claim_group_judge import warehouse_claims
    claims = warehouse_claims(session, [1, 2, 3, 4])  # all four share subject and metric words: the code's matching would put them together
    llm = MockLLMClient([{"groups": [{"refs": ["c1", "c2"], "topic": "storage"}]}])
    clusters, stats = grouper(session, llm).run()
    all_ids = ids_of(clusters)
    assert len(all_ids) == len(set(all_ids))  # no claim in two clusters
    model = by_model(clusters)
    assert [set(c.claim_ids) for c in model] == [{claims[0].id, claims[1].id}]
    code = [c for c in clusters if c not in model]
    assert code and all(set(c.claim_ids) <= {claims[2].id, claims[3].id} for c in code)  # only the claims the model left alone
    assert stats.claims_left == 2 and stats.claims_by_model == 2 and stats.model_clusters == 1


def test_the_codes_matching_clusters_the_claims_the_model_left_alone_with_their_route(session, room):  # noqa: F811
    from tests.test_claim_group_judge import warehouse_claims
    claims = warehouse_claims(session, [1, 2, 3])
    llm = MockLLMClient([{"groups": []}])
    clusters, stats = grouper(session, llm).run()
    assert len(clusters) == 1 and set(clusters[0].claim_ids) == {c.id for c in claims} and set(clusters[0].via.values()) <= {"lexical", "label", "embedding", "identifier"}
    assert stats.code_clusters == 1 and stats.claims_by_code == 3 and stats.claims_by_model == 0 and stats.groups == 0
