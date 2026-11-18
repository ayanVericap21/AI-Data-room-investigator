"""Several Google keys: a call that fails on one key is tried on the next; when every key fails the client's usual backoff applies; a key whose daily quota is
used up or that Google refuses is left out for the rest of the run; a bad request is not tried on other keys."""
from __future__ import annotations

import pytest
from pydantic import BaseModel

from app.core.config import Settings
from app.llm import google
from app.llm.errors import LLMPermanentError, LLMQuotaError, LLMTransientError
from app.llm.google import GoogleLLMClient

OK = {"candidates": [{"content": {"parts": [{"text": '{"ok": true}'}]}}], "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 3, "totalTokenCount": 8}}


class Pong(BaseModel):
    ok: bool = True


def scripted(monkeypatch, plan):
    """plan: key -> list of results in order (a dict is an answer, an exception is raised); returns the list of keys used."""
    used = []

    def post(url, key, body):
        used.append(key)
        item = plan[key].pop(0) if len(plan[key]) > 1 else plan[key][0]
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(google, "_post", post)
    return used


def client(**kw):
    kw.setdefault("sleep", lambda _s: None)
    return GoogleLLMClient("m", ["k1", "k2", "k3"], **kw)


def test_a_call_that_fails_on_one_key_is_answered_by_the_next_without_waiting(monkeypatch):
    waits = []
    used = scripted(monkeypatch, {"k1": [LLMTransientError("busy", status=503)], "k2": [OK], "k3": [OK]})
    c = client(sleep=waits.append)
    assert c.generate_structured_output("p", Pong).ok is True
    assert used == ["k1", "k2"] and waits == [] and c.retry_count == 0
    assert "key 1: 0 calls answered, 1 attempts failed" in c.key_report() and "key 2: 1 calls answered" in c.key_report()


def test_when_every_key_fails_the_usual_backoff_applies_and_the_next_try_goes_through_the_keys_again(monkeypatch):
    waits = []
    busy = LLMTransientError("busy", status=503)
    used = scripted(monkeypatch, {"k1": [busy, busy, OK], "k2": [busy, busy, OK], "k3": [busy, busy, OK]})
    c = client(sleep=waits.append, max_attempts=3)
    assert c.generate_structured_output("p", Pong).ok is True
    assert len(used) == 7 and waits == [1.0, 2.0] and c.retry_count == 2  # 3 + 3 failed tries, then one answer; the waits are the client's own 1 s and 2 s


def test_when_every_key_keeps_failing_the_error_is_raised_after_the_attempts_like_a_single_key(monkeypatch):
    scripted(monkeypatch, {k: [LLMTransientError("busy", status=503)] for k in ("k1", "k2", "k3")})
    c = client(max_attempts=2)
    with pytest.raises(LLMTransientError):
        c.generate_structured_output("p", Pong)


def test_the_shortest_wait_asked_by_any_busy_key_sets_the_backoff(monkeypatch):
    waits = []
    scripted(monkeypatch, {"k1": [LLMTransientError("limit", status=429, retry_after=27.0), OK], "k2": [LLMTransientError("busy", status=503, retry_after=2.0), OK],
                           "k3": [LLMTransientError("limit", status=429, retry_after=40.0), OK]})
    c = client(sleep=waits.append)
    c.generate_structured_output("p", Pong)
    assert waits == [2.0]


def test_a_key_whose_daily_quota_is_used_up_is_left_out_and_the_others_carry_on(monkeypatch):
    used = scripted(monkeypatch, {"k1": [LLMQuotaError("daily quota")], "k2": [OK], "k3": [OK]})
    c = client()
    c.generate_structured_output("p", Pong)
    c.generate_structured_output("p", Pong)
    c.generate_structured_output("p", Pong)
    assert used.count("k1") == 1 and "left out (daily quota" in c.key_report() and c.quota_exhausted == ""  # the run is not stopped


def test_a_key_google_refuses_is_left_out(monkeypatch):
    used = scripted(monkeypatch, {"k1": [LLMPermanentError("Google API 400: API key not valid. Please pass a valid API key. API_KEY_INVALID")], "k2": [OK], "k3": [OK]})
    c = client()
    for _ in range(3):
        c.generate_structured_output("p", Pong)
    assert used.count("k1") == 1


def test_a_bad_request_is_not_tried_on_the_other_keys(monkeypatch):
    used = scripted(monkeypatch, {k: [LLMPermanentError("Google API 400: Invalid JSON payload")] for k in ("k1", "k2", "k3")})
    with pytest.raises(LLMPermanentError):
        client().generate_structured_output("p", Pong)
    assert used == ["k1"]


def test_when_every_key_is_out_the_client_reports_the_quota_and_stops_asking(monkeypatch):
    used = scripted(monkeypatch, {k: [LLMQuotaError("daily quota")] for k in ("k1", "k2", "k3")})
    c = client()
    with pytest.raises(LLMQuotaError):
        c.generate_structured_output("p", Pong)
    assert "every API key is out" in c.quota_exhausted and len(used) == 3
    with pytest.raises(LLMQuotaError):
        c.generate_structured_output("p", Pong)
    assert len(used) == 3  # no key is asked again


def test_keys_that_can_start_at_once_take_turns(monkeypatch):
    used = scripted(monkeypatch, {"k1": [OK], "k2": [OK], "k3": [OK]})
    c = client()
    for _ in range(6):
        c.generate_structured_output("p", Pong)
    assert sorted(used) == ["k1", "k1", "k2", "k2", "k3", "k3"]


def test_the_keys_come_from_the_numbered_settings_or_the_single_key():
    assert Settings(llm_api_key_1="a", llm_api_key_2=" b ", llm_api_key_3="a", llm_api_key="z").google_api_keys() == ["a", "b"]
    assert Settings(llm_api_key="z").google_api_keys() == ["z"] and Settings().google_api_keys() == []
