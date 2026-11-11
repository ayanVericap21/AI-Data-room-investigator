"""PairJudge: exact pairs are corroboration, other pairs go to the LLM, and findings need exact quotes."""
from __future__ import annotations

from sqlalchemy import func, select

from app.llm.errors import LLMPermanentError
from app.llm.mock import MockLLMClient
from app.models import Claim, Evidence, Finding
from app.schemas.claim import AttributeComparison, CandidatePair
from app.services.pair_judge import PairJudge
from app.tools import build_registry
from tests.test_tools import room, add_doc  # noqa: F401  (fixtures and helper)


def comparison(**kw):
    base = dict(subject_match=True, metric_match=True, unit_match=True, period_match=True, value_relation="different",
                conditions_differ=False, assumptions_differ=False)
    base.update(kw)
    return AttributeComparison(**base)


def pair(a, b, hint="conflicting"):
    first, second = (a, b) if a.id <= b.id else (b, a)
    return CandidatePair(claim_a_id=first.id, claim_b_id=second.id, topic_score=0.9, relation_hint=hint,
                         comparison=comparison(), matched_by="lexical")


def make_claims(session, room):
    a = Claim(document_id=room.deck.id, chunk_id=room.deck_p1, subject="plant", metric="capacity", value=120000, unit="tonnes",
              claim_text="Capacity is 120,000 tonnes per year")
    b = Claim(document_id=room.capex.id, chunk_id=room.capex_current, subject="plant", metric="capacity", value=100000, unit="tonnes",
              claim_text="Capacity 100,000 tonnes")
    session.add_all([a, b])
    session.commit()
    return a, b


def judge(session, room, llm):
    registry = build_registry(session, None, room.investigation.id)
    return PairJudge(session, llm, registry)


def verdict_for(p, deck_claim, **kw):
    """A contradiction verdict whose quotes follow the pair's claim order (which depends on random ids)."""
    deck_quote, capex_quote = "capacity is 120,000 tonnes per year", "Capacity | 100,000 tonnes"
    a_is_deck = p.claim_a_id == deck_claim.id
    v = dict(pair_index=0, classification="contradiction", severity="high", confidence=0.9, title="Capacity differs",
             description="One says 120,000, the other 100,000.",
             quote_a=deck_quote if a_is_deck else capex_quote, quote_b=capex_quote if a_is_deck else deck_quote)
    v.update(kw)
    return {"verdicts": [v]}


def test_a_conflicting_pair_becomes_a_finding_with_verified_quotes(session, room):
    a, b = make_claims(session, room)
    p = pair(a, b)
    llm = MockLLMClient([verdict_for(p, a)])
    stats = judge(session, room, llm).run([p])
    assert stats.pairs_judged == 1 and stats.llm_calls == 1 and stats.findings_saved == 1 and stats.findings_rejected == 0
    finding = session.scalars(select(Finding)).one()
    assert finding.classification == "contradiction" and finding.verification_status == "verified"
    assert len(session.scalars(select(Evidence).where(Evidence.finding_id == finding.id)).all()) == 2
    assert stats.findings_by_document[room.deck.id] == 1 and stats.findings_by_document[room.capex.id] == 1
    assert "Claim A (" in llm.prompts[0] and "Excerpt B:" in llm.prompts[0]  # both claims and both excerpts were shown


def test_an_invented_quote_is_rejected_and_nothing_is_saved(session, room):
    a, b = make_claims(session, room)
    p = pair(a, b)
    llm = MockLLMClient([verdict_for(p, a, quote_b="Total capacity | 100,000 tonnes here", quote_a="capacity is 120,000 tonnes per year")])         if p.claim_a_id == a.id else MockLLMClient([verdict_for(p, a, quote_a="Total capacity | 100,000 tonnes here", quote_b="capacity is 120,000 tonnes per year")])
    stats = judge(session, room, llm).run([p])
    assert stats.findings_saved == 0 and stats.findings_rejected == 1 and "validation_failed" in stats.rejections[0]
    assert session.scalar(select(func.count()).select_from(Finding)) == 0


def test_exact_agreements_count_as_corroboration_and_are_not_sent_to_the_llm(session, room):
    a, b = make_claims(session, room)
    llm = MockLLMClient([])  # any LLM call would fail the test
    stats = judge(session, room, llm).run([pair(a, b, hint="exact")])
    assert stats.pairs_exact == 1 and stats.pairs_judged == 0 and stats.llm_calls == 0
    assert stats.corroborated_claims == {a.id, b.id}


def test_consistent_and_insufficient_verdicts_are_counted_but_not_reported(session, room):
    a, b = make_claims(session, room)
    p = pair(a, b)
    llm = MockLLMClient([verdict_for(p, a, classification="consistent", quote_a="", quote_b="")])
    stats = judge(session, room, llm).run([p])
    assert stats.verdict_counts["consistent"] == 1 and stats.findings_saved == 0 and stats.findings_rejected == 0


def test_a_failed_batch_is_recorded_and_the_run_continues(session, room):
    a, b = make_claims(session, room)
    llm = MockLLMClient([LLMPermanentError("401 bad key"), LLMPermanentError("401 bad key")])  # the batch, then the single-pair retry
    stats = judge(session, room, llm).run([pair(a, b)])
    assert stats.batches_failed == 1 and stats.retries == 1 and "401" in stats.errors[0] and stats.findings_saved == 0


def test_batches_are_sized_by_characters_not_by_a_fixed_count(session, room):
    a, b = make_claims(session, room)
    llm = MockLLMClient([{"verdicts": []}, {"verdicts": []}, {"verdicts": []}])  # two batches, then one retry of the pair that got no verdict
    registry = build_registry(session, None, room.investigation.id)
    stats = PairJudge(session, llm, registry, max_chars_per_call=250).run([pair(a, b), pair(a, b)])
    assert stats.llm_calls - stats.retries == 2  # each pair alone exceeds half the budget, so they cannot share a call


def test_a_quote_that_does_not_show_the_claims_value_is_rejected(session, room):
    a, b = make_claims(session, room)
    p = pair(a, b)
    # both quotes are real text of their chunks, but the capex quote does not contain the capex claim's value
    wrong = ("Annual production capacity is 120,000 tonnes per year at the Pune plant.", "Annual production capacity is 120,000 tonnes per year")
    llm = MockLLMClient([verdict_for(p, a, quote_a=wrong[0], quote_b=wrong[1])])
    stats = judge(session, room, llm).run([p])
    assert stats.findings_saved == 0 and stats.findings_rejected == 1 and "do not show" in stats.rejections[0]


def test_the_same_discrepancy_reported_twice_is_saved_once(session, room):
    a, b = make_claims(session, room)
    p = pair(a, b)
    one = verdict_for(p, a)["verdicts"][0]
    llm = MockLLMClient([{"verdicts": [one, {**one, "pair_index": 1}]}])
    stats = judge(session, room, llm).run([p, pair(a, b)])
    assert stats.findings_saved == 1 and stats.duplicates_skipped == 1
    assert session.scalar(select(func.count()).select_from(Finding)) == 1


def test_a_missing_title_and_description_fall_back_to_what_both_documents_state(session, room):
    a, b = make_claims(session, room)
    p = pair(a, b)
    stats = judge(session, room, MockLLMClient([verdict_for(p, a, title="", description="")])).run([p])
    finding = session.scalars(select(Finding)).one()
    assert stats.findings_saved == 1
    assert "Capacity is 120,000 tonnes per year" in finding.description and "Capacity 100,000 tonnes" in finding.description


def test_quotes_given_in_the_opposite_order_are_attached_to_their_own_claims(session, room):
    a, b = make_claims(session, room)
    p = pair(a, b)
    v = verdict_for(p, a)["verdicts"][0]
    v["quote_a"], v["quote_b"] = v["quote_b"], v["quote_a"]  # the model swapped them
    stats = judge(session, room, MockLLMClient([{"verdicts": [v]}])).run([p])
    assert stats.findings_saved == 1 and stats.findings_rejected == 0
    rows = {e.chunk_id: e.quoted_text for e in session.scalars(select(Evidence))}
    assert rows == {room.deck_p1: "capacity is 120,000 tonnes per year", room.capex_current: "Capacity | 100,000 tonnes"}


def test_a_pair_about_different_attributes_is_counted_as_not_comparable_and_never_becomes_a_finding(session, room):
    a, b = make_claims(session, room)
    p = pair(a, b)
    llm = MockLLMClient([verdict_for(p, a, same_attribute=False, attribute_reasoning="A is a capacity, B is a height")])
    stats = judge(session, room, llm).run([p])
    assert stats.verdict_counts == {"not_comparable": 1} and stats.findings_saved == 0 and stats.findings_rejected == 0
    assert session.scalar(select(func.count()).select_from(Finding)) == 0


def test_same_attribute_defaults_to_true_and_the_prompt_asks_for_it():
    from app.services.pair_judge import SYSTEM, PairVerdict

    assert PairVerdict(pair_index=0, classification="consistent").same_attribute is True
    assert "same_attribute" in SYSTEM
