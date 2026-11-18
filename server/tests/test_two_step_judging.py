"""Two-step judging: a smaller model screens every group, a bigger model decides the groups the screen calls findings; routing over a chain of models."""
from __future__ import annotations

import pytest
from pydantic import BaseModel
from sqlalchemy import select

from app.llm.errors import LLMPermanentError, LLMQuotaError, LLMTransientError
from app.llm.mock import MockLLMClient
from app.llm.router import RoutedLLMClient
from app.models import Finding
from tests.test_claim_group_judge import answer, candidates_of, judge, short, verdict, warehouse_claims
from tests.test_tools import room  # noqa: F401  (fixture)


def run(session, room, screen, strong, claims):  # noqa: F811
    return judge(session, room, screen, strong_llm=strong, save_consistent=False).run(candidates_of(claims))


def test_a_group_the_screen_calls_consistent_never_reaches_the_bigger_model(session, room):  # noqa: F811
    claims = warehouse_claims(session, [500, 500])
    strong = MockLLMClient([])
    stats = run(session, room, MockLLMClient([answer(short(claims))]), strong, claims)
    assert strong.prompts == [] and stats.escalated == 0


def test_the_bigger_model_decides_a_candidate_and_its_verdict_is_the_one_saved(session, room):  # noqa: F811
    claims = warehouse_claims(session, [500, 450])
    screen = MockLLMClient([answer(verdict(claims, title="Screen title"))])
    strong = MockLLMClient([answer(verdict(claims, title="Bigger model title"))])
    stats = run(session, room, screen, strong, claims)
    finding = session.scalars(select(Finding)).one()
    assert finding.title == "Bigger model title" and (stats.escalated, stats.strong_calls, stats.strong_confirmed, stats.strong_overridden) == (1, 1, 1, 0)


def test_a_candidate_the_bigger_model_finds_consistent_saves_nothing(session, room):  # noqa: F811
    claims = warehouse_claims(session, [500, 450])
    stats = run(session, room, MockLLMClient([answer(verdict(claims))]), MockLLMClient([answer(short(claims))]), claims)
    assert session.scalar(select(Finding)) is None and (stats.escalated, stats.strong_confirmed, stats.strong_overridden) == (1, 0, 1)


def test_when_the_bigger_model_fails_the_screens_verdict_is_kept(session, room):  # noqa: F811
    claims = warehouse_claims(session, [500, 450])
    strong = MockLLMClient([LLMTransientError("busy", status=503)], max_attempts=1)
    stats = run(session, room, MockLLMClient([answer(verdict(claims, title="Screen title"))]), strong, claims)
    assert session.scalars(select(Finding)).one().title == "Screen title"
    assert (stats.strong_failed, stats.strong_kept_screen, stats.findings_saved) == (1, 1, 1)


def test_only_the_candidate_groups_go_to_the_bigger_model(session, room):  # noqa: F811
    first, second = warehouse_claims(session, [500, 450], prefix="a"), warehouse_claims(session, [300, 300], prefix="b")
    screen = MockLLMClient([answer(verdict(first), short(second, group=2))])
    strong = MockLLMClient([answer(verdict(first, title="Decided"))])
    stats = judge(session, room, screen, strong_llm=strong, save_consistent=False).run(candidates_of(first) + candidates_of(second))
    assert stats.escalated == 1 and len(strong.prompts) == 1 and "### Group 2" not in strong.prompts[0]  # the consistent group stayed with the screen
    assert session.scalars(select(Finding)).one().title == "Decided"


class Pong(BaseModel):
    ok: bool = True


OK = {"ok": True}


def chain(*scripts):
    clients = []
    for n, script in enumerate(scripts):
        client = MockLLMClient(script, max_attempts=1)
        client.model = f"model-{n}"
        clients.append(client)
    return RoutedLLMClient(clients)


def test_a_call_goes_to_the_first_model_of_the_chain():
    routed = chain([OK], [OK])
    assert routed.generate_structured_output("p", Pong).ok and routed.routed_calls == [1, 0] and routed.fell_back == 0


def test_a_model_out_of_quota_is_left_out_for_the_rest_of_the_run():
    routed = chain([LLMQuotaError("daily quota used up")], [OK, OK])
    routed.generate_structured_output("p", Pong)
    routed.generate_structured_output("p", Pong)
    assert routed.routed_calls == [0, 2] and 0 in routed.unavailable and routed.fell_back == 2
    assert len(routed.clients[0].prompts) == 1  # asked once; the second call did not try it


def test_a_model_that_fails_one_call_is_tried_again_on_the_next():
    routed = chain([LLMTransientError("busy", status=503), OK], [OK])
    routed.generate_structured_output("p", Pong)  # model-0 fails, model-1 answers
    routed.generate_structured_output("p", Pong)  # model-0 is not left out for one failure
    assert routed.routed_calls == [1, 1] and routed.unavailable == {}


def test_when_every_model_is_out_of_quota_the_chain_says_so():
    routed = chain([LLMQuotaError("a")], [LLMQuotaError("b")])
    with pytest.raises(LLMQuotaError) as err:
        routed.generate_structured_output("p", Pong)
    assert "every model of the chain is out" in str(err.value) and routed.quota_exhausted


def test_a_bad_request_is_not_tried_on_other_models_and_a_retired_model_is_skipped():
    routed = chain([LLMPermanentError("Google API 400: bad request")], [OK])
    with pytest.raises(LLMPermanentError):
        routed.generate_structured_output("p", Pong)
    routed = chain([LLMPermanentError("Google API 404: models/x is not found")], [OK])
    assert routed.generate_structured_output("p", Pong).ok and 0 in routed.unavailable


def test_the_chain_shares_one_report_with_the_run_client_and_names_the_model():
    main = MockLLMClient([])
    routed = chain([OK], [OK])
    routed.share_with(main)
    routed.generate_structured_output("p", Pong)
    assert len(main.call_metrics) == 1 and main.call_metrics[0].model == "model-0"


def test_the_report_records_the_screens_and_the_bigger_models_verdict_for_every_candidate(session, room):  # noqa: F811
    claims = warehouse_claims(session, [500, 450])
    stats = run(session, room, MockLLMClient([answer(verdict(claims, title="Screen title"))]), MockLLMClient([answer(short(claims, reasoning="same milestone"))]), claims)
    (entry,) = stats.strong_log
    assert entry["outcome"].startswith("overridden") and "Screen title" in entry["screen"] and "consistent" in entry["bigger"] and len(entry["claims"]) == 2
    assert entry["claims"][0].startswith("doc|") or "| warehouse |" in entry["claims"][0]


def test_the_log_says_why_the_screens_verdict_was_kept(session, room):  # noqa: F811
    claims = warehouse_claims(session, [500, 450])
    strong = MockLLMClient([LLMTransientError("busy now", status=503)], max_attempts=1)
    stats = run(session, room, MockLLMClient([answer(verdict(claims, title="Screen title"))]), strong, claims)
    (entry,) = stats.strong_log
    assert "screen's verdict kept" in entry["outcome"] and "busy now" in entry["outcome"] and entry["bigger"] == "(none)"
