"""Google: a daily-quota 429 is never retried and ends the run's work; a per-minute 429 is retried; calls are paced to the requests-per-minute limit."""
from __future__ import annotations

import json

import httpx
import pytest
from pydantic import BaseModel

from app.llm import google
from app.llm.errors import LLMQuotaError, LLMTransientError
from app.llm.google import GoogleLLMClient, _check
from app.llm.groq import RateLimiter

DAILY = {"error": {"code": 429, "message": "You exceeded your current quota.", "status": "RESOURCE_EXHAUSTED", "details": [
    {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [
        {"quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests",
         "quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier", "quotaDimensions": {"model": "gemini-3.5-flash-lite"}, "quotaValue": "500"}]},
    {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "27s"}]}}
PER_MINUTE = {"error": {"code": 429, "message": "Quota exceeded", "status": "RESOURCE_EXHAUSTED", "details": [
    {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [
        {"quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests",
         "quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier", "quotaValue": "15"}]},
    {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "27s"}]}}
OK = {"candidates": [{"content": {"parts": [{"text": '{"ok": true}'}]}}], "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 3, "totalTokenCount": 8}}


class Pong(BaseModel):
    ok: bool = True


def reply(status, body):
    return httpx.Response(status, content=json.dumps(body).encode(), headers={"content-type": "application/json"})


def test_a_daily_quota_429_is_a_quota_error_and_a_per_minute_429_is_transient():
    with pytest.raises(LLMQuotaError) as daily:
        _check(reply(429, DAILY))
    assert "GenerateRequestsPerDayPerProjectPerModel-FreeTier" in str(daily.value) and daily.value.status == 429
    with pytest.raises(LLMTransientError) as minute:
        _check(reply(429, PER_MINUTE))
    assert minute.value.status == 429 and minute.value.retry_after == 27.0


def test_a_daily_quota_is_not_retried_and_is_remembered_by_the_client(monkeypatch):
    calls = []

    def post(url, key, body):
        calls.append(1)
        return _check(reply(429, DAILY))

    monkeypatch.setattr(google, "_post", post)
    client = GoogleLLMClient("m", "k", max_attempts=5, sleep=lambda s: pytest.fail("a daily quota must not wait"))
    with pytest.raises(LLMQuotaError):
        client.generate_structured_output("p", Pong)
    assert len(calls) == 1 and "daily quota" in client.quota_exhausted and client.call_metrics[0].status == 429


def test_a_per_minute_429_is_retried_and_the_client_stays_usable(monkeypatch):
    answers = [lambda: _check(reply(429, PER_MINUTE)), lambda: OK]
    monkeypatch.setattr(google, "_post", lambda url, key, body: answers.pop(0)())
    waits = []
    client = GoogleLLMClient("m", "k", max_attempts=3, sleep=waits.append)
    assert client.generate_structured_output("p", Pong).ok is True
    assert waits == [27.0] and client.quota_exhausted == ""


def test_calls_are_spaced_to_the_requests_per_minute_limit_and_a_429_pauses_every_thread(monkeypatch):
    now = [0.0]
    slept = []

    def sleep(seconds):
        slept.append(round(seconds, 3))
        now[0] += seconds

    monkeypatch.setattr(google, "_post", lambda url, key, body: OK)
    client = GoogleLLMClient("m", "k", requests_per_minute=15)
    client._limiter = RateLimiter(15, clock=lambda: now[0], sleep=sleep)  # 15 a minute is one start every 4 seconds
    for _ in range(3):
        client.generate_structured_output("p", Pong)
    assert slept == [4.0, 4.0]  # the first call starts at once

    monkeypatch.setattr(google, "_post", lambda url, key, body: (_ for _ in ()).throw(LLMTransientError("busy", retry_after=30.0, status=429)))
    client.max_attempts = 1
    with pytest.raises(LLMTransientError):
        client.generate_structured_output("p", Pong)
    assert client._limiter.wait_for() >= 29.0  # the next call, from any thread, waits for the provider's delay


def test_no_pacing_when_the_limit_is_zero():
    assert GoogleLLMClient("m", "k", requests_per_minute=0)._limiter is None
