"""Key grouping: each claim labelled on its own, attribute names merged once, claims grouped by a lookup on (attribute, value type)."""
from __future__ import annotations

import re

from app.llm.errors import LLMTransientError
from app.llm.mock import MockLLMClient
from app.models import Claim, ClaimLabelCacheEntry
from app.services.claim_grouping import ordered_rows
from app.services.claim_labels import ClaimLabeler, LabelAnswer, LabelCache, claim_key, merge_vocabulary, normalize_name
from app.services.key_grouping import KeyGrouper
from tests.test_claim_group_judge import make_claims, room, warehouse_claims  # noqa: F401  (fixtures/helpers)

DEFAULT = dict(subject="warehouse", attribute="storage capacity", other_attributes=[], value_type="count", basis="", kind="record")


def labeller(by_text: dict[str, dict] | None = None, merge: dict | None = None):
    """A scripted model: labels each claim line by the first key of by_text found in it (else DEFAULT); merges names as merge says."""
    by_text = by_text or {}

    def answer(prompt: str):
        if "Attribute names:" in prompt:
            return {"groups": [{"canonical": c, "members": m} for c, m in (merge or {}).items()]}
        out = []
        for line in prompt.splitlines():
            m = re.match(r"^(c\d+) \| ", line)
            if m:
                label = next((v for k, v in by_text.items() if k in line), {})
                out.append({"ref": m.group(1), **DEFAULT, **label})
        return {"labels": out}
    return answer


def grouper(session, session_factory, llm, **kw):
    return KeyGrouper(session, llm, session_factory, **kw)


def test_claims_with_the_same_key_are_one_cluster_and_the_vocabulary_merge_joins_different_wordings(session, session_factory, room):  # noqa: F811
    a, b = warehouse_claims(session, [100, 120])
    llm = MockLLMClient([labeller({"120": {"attribute": "space for goods"}}, merge={"storage capacity": ["storage capacity", "space for goods"]})] * 2)
    clusters, stats = grouper(session, session_factory, llm).run()
    assert len(clusters) == 1 and set(clusters[0].claim_ids) == {a.id, b.id} and clusters[0].matched_by == "key" and clusters[0].via == {a.id: "key", b.id: "key"}
    assert (stats.claims, stats.attributes, stats.attributes_merged, stats.clusters, stats.claims_left) == (2, 2, 1, 1, 0)


def test_claims_of_different_value_types_are_never_in_a_key_cluster_even_with_one_attribute(session, session_factory, room):  # noqa: F811
    warehouse_claims(session, [100, 120])
    llm = MockLLMClient([labeller({"120": {"value_type": "money"}})])
    clusters, stats = grouper(session, session_factory, llm).run()
    assert not any(c.matched_by == "key" for c in clusters)  # the code's own matching may still join them by their words; the key never does


def test_a_claim_with_two_attributes_is_in_both_groups(session, session_factory, room):  # noqa: F811
    a, b, c = warehouse_claims(session, [100, 120, 130])
    labels = {"100": {"attribute": "storage capacity"}, "120": {"attribute": "storage capacity", "other_attributes": ["floor area"]},
              "130": {"attribute": "floor area"}}
    clusters, _ = grouper(session, session_factory, MockLLMClient([labeller(labels), labeller(labels)])).run()
    assert sorted(sorted(c.claim_ids) for c in clusters) == sorted([sorted([a.id, b.id]), sorted([b.id, c.id])])


def test_the_result_does_not_depend_on_how_the_claims_are_batched(session, session_factory, room):  # noqa: F811
    claims = warehouse_claims(session, [100, 120, 130, 140])
    labels = {"100": {"attribute": "a"}, "120": {"attribute": "a"}, "130": {"attribute": "b"}, "140": {"attribute": "b"}}
    runs = []
    for size in (1, 2, 40):
        with session_factory() as s:
            s.query(ClaimLabelCacheEntry).delete()
            s.commit()
        clusters, _ = grouper(session, session_factory, MockLLMClient([labeller(labels)] * 8), batch_size=size).run()
        runs.append(sorted(sorted(c.claim_ids) for c in clusters))
    assert runs[0] == runs[1] == runs[2] and len(runs[0]) == 2 and {i for r in runs[0] for i in r} == {c.id for c in claims}


def test_a_second_run_reads_every_label_and_the_merge_from_the_cache_and_calls_nothing(session, session_factory, room):  # noqa: F811
    warehouse_claims(session, [100, 120])
    labels = {"120": {"attribute": "space for goods"}}
    merge = {"storage capacity": ["storage capacity", "space for goods"]}
    first, s1 = grouper(session, session_factory, MockLLMClient([labeller(labels, merge)] * 2)).run()
    silent = MockLLMClient([])  # a call would raise: no scripted response
    second, s2 = grouper(session, session_factory, silent).run()
    assert silent.prompts == [] and [sorted(c.claim_ids) for c in first] == [sorted(c.claim_ids) for c in second]
    assert (s1.labelled_asked, s1.labelled_cached, s2.labelled_asked, s2.labelled_cached) == (2, 0, 0, 2)


def test_refresh_ignores_the_cache(session, session_factory, room):  # noqa: F811
    warehouse_claims(session, [100, 120])
    grouper(session, session_factory, MockLLMClient([labeller()])).run()
    llm = MockLLMClient([labeller()])
    grouper(session, session_factory, llm, refresh=True).run()
    assert len(llm.prompts) >= 1


def test_a_group_over_the_cap_is_divided_by_subject(session, session_factory, room):  # noqa: F811
    claims = warehouse_claims(session, list(range(10, 18)))
    subjects = {str(v): {"subject": "north" if v < 14 else "south"} for v in range(10, 18)}
    clusters, stats = grouper(session, session_factory, MockLLMClient([labeller(subjects)]), max_group_claims=4).run()
    assert sorted(len(c.claim_ids) for c in clusters) == [4, 4] and stats.largest_group == 4
    north = {c.id for c in claims if int(c.value) < 14}
    assert {frozenset(c.claim_ids) for c in clusters} == {frozenset(north), frozenset(c.id for c in claims) - north}


def test_exactly_one_rule_among_records_makes_a_rule_cluster_with_the_rule_first(session, session_factory, room):  # noqa: F811
    a, b, c = warehouse_claims(session, [100, 120, 130])
    clusters, stats = grouper(session, session_factory, MockLLMClient([labeller({"130": {"kind": "rule"}})])).run()
    assert len(clusters) == 1 and clusters[0].rule_id == c.id and clusters[0].claim_ids[0] == c.id and stats.rule_clusters == 1


def test_two_rules_make_a_plain_cluster(session, session_factory, room):  # noqa: F811
    warehouse_claims(session, [100, 120, 130])
    clusters, _ = grouper(session, session_factory, MockLLMClient([labeller({"120": {"kind": "rule"}, "130": {"kind": "rule"}})])).run()
    assert len(clusters) == 1 and clusters[0].rule_id is None


def test_a_failed_labelling_call_falls_back_to_the_extracted_metric_and_is_reported(session, session_factory, room):  # noqa: F811
    a, b = make_claims(session, room)  # both: subject plant, metric capacity
    llm = MockLLMClient([LLMTransientError("busy")] * 6, max_attempts=1)
    clusters, stats = grouper(session, session_factory, llm).run()
    assert stats.labelled_fallback == 2 and any("busy" in e for e in stats.errors)
    assert len(clusters) == 1 and set(clusters[0].claim_ids) == {a.id, b.id}  # the metric alone still puts them together
    with session_factory() as s:
        assert s.query(ClaimLabelCacheEntry).count() == 0  # a fallback is never cached


def test_the_vocabulary_merge_ignores_unknown_names_and_a_name_in_two_groups(session_factory):
    llm = MockLLMClient([{"groups": [{"canonical": "unit count", "members": ["apartments", "units", "invented"]},
                                     {"canonical": "other", "members": ["units", "rooms"]},
                                     {"canonical": "alone", "members": ["rooms"]}]}])
    mapping = merge_vocabulary(llm, LabelCache(session_factory, "p"), {"apartments": 3, "units": 2, "rooms": 1})
    assert mapping == {"apartments": "unit count", "units": "unit count"}


def test_the_label_key_depends_on_the_claim_alone_and_names_are_normalised(session, room):  # noqa: F811
    a, b = make_claims(session, room)
    views = {r[0].id: r[0] for r in ordered_rows(session)}
    assert claim_key(views[a.id]) != claim_key(views[b.id]) and claim_key(views[a.id]) == claim_key(views[a.id].model_copy(update={"id": "other"}))
    assert normalize_name("  Total  Developer-Fee (USD) ") == "total developer fee usd"


def test_claims_that_repeat_one_sentence_are_labelled_once(session, session_factory, room):  # noqa: F811
    a, b = make_claims(session, room)
    session.add(Claim(document_id=room.deck.id, chunk_id=room.deck_p2, subject=a.subject, metric=a.metric, value=a.value, unit=a.unit, claim_text=a.claim_text))
    session.commit()
    _labels, stats = ClaimLabeler(MockLLMClient([labeller()]), LabelCache(session_factory, "p")).label([r[0] for r in ordered_rows(session)])
    assert stats.claims == 3 and stats.distinct == 2 and stats.asked == 2


def test_an_off_list_value_type_or_kind_is_coerced_and_does_not_invalidate_the_other_labels(session, session_factory, room):  # noqa: F811
    a, b = make_claims(session, room)
    llm = MockLLMClient([labeller({"120,000": {"value_type": "Currency", "kind": "requirement"}, "100,000": {"value_type": " COUNT "}})])
    labels, stats = ClaimLabeler(llm, LabelCache(session_factory, "p")).label([r[0] for r in ordered_rows(session)])
    assert len(llm.prompts) == 1 and stats.fallback == 0  # one call, no retry, every claim labelled by the model
    assert (labels[a.id].value_type, labels[a.id].kind) == ("text", "record") and labels[b.id].value_type == "count"


def test_the_invalid_output_warning_says_why(caplog):
    llm = MockLLMClient(["not json", {"labels": []}])
    with caplog.at_level("WARNING"):
        llm.generate_structured_output("p", LabelAnswer)
    assert any("Invalid structured output (LabelAnswer, attempt 1 of 3)" in r.getMessage() and "Expecting value" in r.getMessage() for r in caplog.records)


def test_every_label_field_is_required_in_the_response_schema():
    from app.llm.google import gemini_schema
    item = gemini_schema(LabelAnswer.model_json_schema())["properties"]["labels"]["items"]
    assert set(item["required"]) == set(item["properties"])  # an optional field comes back empty (all 1,184 claims of the 21 Nov run)


def test_a_refs_only_answer_is_invalid_and_is_reported_instead_of_being_taken_for_labels(session, session_factory, room):  # noqa: F811
    make_claims(session, room)
    views = [r[0] for r in ordered_rows(session)]
    refs_only = {"labels": [{"ref": "c1"}, {"ref": "c2"}]}  # what the model returned on 21 Nov for every claim: it must fail validation, not pass as an empty label
    llm = MockLLMClient([refs_only, refs_only, refs_only], max_attempts=3)
    labels, stats = ClaimLabeler(llm, LabelCache(session_factory, "p")).label(views)
    assert len(llm.prompts) == 3 and stats.fallback == 2 and any("did not match LabelAnswer" in e for e in stats.errors)


def test_two_unusable_answers_in_a_row_are_reported_and_what_was_labelled_is_kept(session, session_factory, room):  # noqa: F811
    a, b = make_claims(session, room)
    views = [r[0] for r in ordered_rows(session)]
    half = lambda p: {"labels": [{"ref": "c1", **DEFAULT}, {"ref": "c2", **DEFAULT, "attribute": ""}]}  # one of two claims labelled: exactly half is enough
    none = lambda p: {"labels": [{"ref": "c1", **DEFAULT, "attribute": ""}, {"ref": "c2", **DEFAULT, "attribute": ""}]}
    llm = MockLLMClient([none, none])
    _labels, stats = ClaimLabeler(llm, LabelCache(session_factory, "p")).label(views)
    assert stats.fallback == 2 and any("only 0 of 2" in e for e in stats.errors) and len(llm.prompts) == 2
    llm2 = MockLLMClient([half])
    _labels, stats2 = ClaimLabeler(llm2, LabelCache(session_factory, "p2")).label(views)
    assert stats2.fallback == 1 and stats2.errors == [] and len(llm2.prompts) == 1


def test_claims_no_key_cluster_took_go_through_the_codes_own_matching(session, session_factory, room):  # noqa: F811
    a, b = warehouse_claims(session, [100, 120])  # the same subject and metric words, so the code's word matching joins them
    labels = {"100": {"attribute": "alpha"}, "120": {"attribute": "beta"}}  # the labels give them different keys: no key cluster
    clusters, stats = grouper(session, session_factory, MockLLMClient([labeller(labels)] * 2)).run()
    assert len(clusters) == 1 and set(clusters[0].claim_ids) == {a.id, b.id} and clusters[0].matched_by != "key"
    assert (stats.code_clusters, stats.claims_by_code, stats.claims_left, stats.model_clusters) == (1, 2, 0, 0)


def test_the_codes_own_matching_never_changes_a_key_cluster(session, session_factory, room):  # noqa: F811
    a, b, c = warehouse_claims(session, [100, 120, 130])
    labels = {"130": {"attribute": "beta"}}  # a and b share a key; c does not
    clusters, stats = grouper(session, session_factory, MockLLMClient([labeller(labels)] * 2)).run()
    by_key = [x for x in clusters if x.matched_by == "key"]
    assert len(by_key) == 1 and set(by_key[0].claim_ids) == {a.id, b.id}
    assert all(c.id not in x.claim_ids or x.matched_by != "key" for x in clusters)
