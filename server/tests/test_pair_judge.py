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


def verdict_for(a_id, **kw):
    v = dict(pair_index=0, classification="contradiction", severity="high", confidence=0.9, title="Capacity differs",
             description="One says 120,000, the other 100,000.", quote_a="capacity is 120,000 tonnes per year",
             quote_b="Capacity | 100,000 tonnes")
    v.update(kw)
    return {"verdicts": [v]}


def test_a_conflicting_pair_becomes_a_finding_with_verified_quotes(session, room):
    a, b = make_claims(session, room)
    p = pair(a, b)
    llm = MockLLMClient([verdict_for(p.claim_a_id)])
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
    llm = MockLLMClient([verdict_for(p.claim_a_id, quote_b="Capacity | 110,000 tonnes")])
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
    llm = MockLLMClient([verdict_for(p.claim_a_id, classification="consistent", quote_a="", quote_b="")])
    stats = judge(session, room, llm).run([p])
    assert stats.verdict_counts["consistent"] == 1 and stats.findings_saved == 0 and stats.findings_rejected == 0


def test_a_failed_batch_is_recorded_and_the_run_continues(session, room):
    a, b = make_claims(session, room)
    llm = MockLLMClient([LLMPermanentError("401 bad key")])
    stats = judge(session, room, llm).run([pair(a, b)])
    assert stats.batches_failed == 1 and "401" in stats.errors[0] and stats.findings_saved == 0


def test_batches_are_sized_by_characters_not_by_a_fixed_count(session, room):
    a, b = make_claims(session, room)
    llm = MockLLMClient([{"verdicts": []}, {"verdicts": []}])
    registry = build_registry(session, None, room.investigation.id)
    stats = PairJudge(session, llm, registry, max_chars_per_call=250).run([pair(a, b), pair(a, b)])
    assert stats.llm_calls == 2  # each pair alone exceeds half the budget, so they cannot share a call
