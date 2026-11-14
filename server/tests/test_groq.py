"""The Groq adapter: request shape, error handling, request and token limits, and the chain of fallback models, with no network."""
from __future__ import annotations

import httpx
import pytest

from app.core.config import Settings
from app.llm import groq
from app.llm.errors import LLMConfigError, LLMPermanentError, LLMTransientError
from app.llm.factory import get_llm_client
from app.llm.groq import GroqLLMClient, RateLimiter, TokenLimiter, estimate_tokens


class Clock:
    """A clock whose sleep moves time forward, so waits are measured and cost nothing."""

    def __init__(self):
        self.now, self.slept = 1000.0, []

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(round(seconds, 2))
        self.now += seconds


def reply(status=200, content='{"ok": true}', usage=None, headers=None, text=None):
    body = {"choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": usage or {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120,
                               "completion_tokens_details": {"reasoning_tokens": 7}, "prompt_tokens_details": {"cached_tokens": 5}}}
    return httpx.Response(status, json=body, headers=headers or {}) if text is None else httpx.Response(status, text=text, headers=headers or {})


def chain(monkeypatch, responses=(), fallbacks=("m2", "m3"), **kwargs):
    """A client over models m1 (preferred), m2, m3 whose HTTP calls return the scripted responses in order."""
    clock = Clock()
    client = GroqLLMClient("m1", "k-secret", fallback_models=fallbacks, clock=clock.time, wait=clock.sleep, sleep=clock.sleep, **kwargs)
    client.sent = []
    queue = list(responses)

    def fake_post(url, json=None, headers=None, timeout=None):
        client.sent.append((url, json, headers))
        return queue.pop(0)

    monkeypatch.setattr(groq.httpx, "post", fake_post)
    return client, clock


def models_used(client):
    return [body["model"] for _url, body, _h in client.sent]


# ---- the limits --------------------------------------------------------------------------------------------------------------------------
def test_calls_are_spaced_to_the_request_limit():
    clock = Clock()
    limiter = RateLimiter(30, clock.time, clock.sleep)  # 2 seconds apart
    for _ in range(4):
        limiter.acquire()
    assert clock.slept == [2.0, 2.0, 2.0] and round(limiter.waited) == 6  # one caller after another: each starts 2 seconds after the last
    limiter.pause(10)
    assert limiter.wait_for() > 9
    limiter.acquire()
    assert clock.slept[-1] >= 10


def test_the_token_window_makes_a_caller_wait_until_the_minute_has_room():
    clock = Clock()
    limiter = TokenLimiter(8000, clock.time, clock.sleep)
    first = limiter.reserve(6000)
    clock.now += 10
    assert limiter.wait_for(1500) == 0.0 and round(limiter.wait_for(3000)) == 50
    limiter.reserve(1500)  # fits beside the first
    assert clock.slept == []
    limiter.reserve(3000)  # does not fit: waits until the first entry is a minute old
    assert clock.slept and round(sum(clock.slept)) == 50
    limiter.settle(first, 100)  # a settled entry counts its real tokens
    limiter.reserve(6000)
    with pytest.raises(LLMPermanentError, match="cannot fit in the limit of 8000 tokens per minute"):
        limiter.reserve(8001)


def test_a_provider_pause_blocks_the_token_window_for_that_long():
    clock = Clock()
    limiter = TokenLimiter(8000, clock.time, clock.sleep)
    limiter.pause(20)
    limiter.reserve(100)
    assert round(sum(clock.slept)) == 20


def test_a_token_estimate_counts_three_characters_to_a_token_until_the_run_has_measured_its_own(monkeypatch):
    assert estimate_tokens("x" * 300, "y" * 30, False) == 110 and estimate_tokens("", None, True) == 30 and estimate_tokens("x" * 400, None, False, 4.0) == 100
    c, _ = chain(monkeypatch, [reply(usage={"prompt_tokens": 2500, "completion_tokens": 5, "total_tokens": 2505}), reply()])
    assert c.chars_per_token() == 3.0
    c._raw_generate("x" * 10_000, None, False)  # 10,000 characters were 2,500 tokens: 4 characters to a token
    assert c.chars_per_token() == 4.0
    c._raw_generate("y" * 12_000, None, False)  # now estimated at 3,000 tokens (plus the answer reserve), not 4,000
    assert c.slots["m1"].tokens._entries[-1][1] == 120.0  # settled to the real count of the second call
    c._chars_sent, c._prompt_tokens_seen = 1000, 100_000  # absurd ratios are clamped
    assert c.chars_per_token() == 2.5


# ---- one call ----------------------------------------------------------------------------------------------------------------------------
def test_a_successful_call_sends_the_expected_request_and_reads_the_usage(monkeypatch):
    c, _ = chain(monkeypatch, [reply(), reply()])
    raw = c._raw_generate("hello", "be brief", json_mode=True)
    url, body, headers = c.sent[0]
    assert url.endswith("/chat/completions") and headers["Authorization"] == "Bearer k-secret"
    assert body["model"] == "m1" and body["temperature"] == 0 and body["response_format"] == {"type": "json_object"}
    assert body["messages"] == [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hello"}]
    assert raw.text == '{"ok": true}' and (raw.usage.prompt_tokens, raw.usage.completion_tokens, raw.usage.thinking_tokens, raw.usage.cached_tokens, raw.usage.total_tokens) == (100, 20, 7, 5, 120)
    c._raw_generate("hello", None, json_mode=False)
    assert "response_format" not in c.sent[1][1] and c.sent[1][1]["messages"] == [{"role": "user", "content": "hello"}]
    assert c.model_usage == {"m1": 2}


def test_reasoning_written_into_the_answer_is_removed_and_the_real_token_count_replaces_the_estimate(monkeypatch):
    c, _ = chain(monkeypatch, [reply(content='<think>let me see\n{"no": 1}</think>\n{"yes": 1}', usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})])
    assert c._raw_generate("x" * 3000, None, json_mode=True).text == '{"yes": 1}'  # estimated 1,030 + 1,500 reserved, really 15
    assert [e[1] for e in c.slots["m1"].tokens._entries] == [15.0]


@pytest.mark.parametrize("status,error", [(500, LLMTransientError), (503, LLMTransientError), (401, LLMPermanentError), (413, LLMPermanentError), (400, LLMPermanentError)])
def test_error_statuses_are_transient_or_permanent(monkeypatch, status, error):
    c, _ = chain(monkeypatch, [reply(status, text="boom")])
    with pytest.raises(error) as caught:
        c._raw_generate("hi", None, False)
    if status == 413:
        assert "less text per call" in str(caught.value)


def test_an_empty_answer_and_a_network_error_are_retried(monkeypatch):
    c, _ = chain(monkeypatch, [reply(content=""), reply(content='{"x": 1}')], max_attempts=3)
    assert c.generate_text("hi") == '{"x": 1}' and c.retry_count == 1

    def broken(*a, **k):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(groq.httpx, "post", broken)
    c2 = GroqLLMClient("m1", "k")
    with pytest.raises(LLMTransientError, match="network error"):
        c2._raw_generate("hi", None, False)


def test_a_request_too_large_for_the_token_window_is_refused_before_it_is_sent(monkeypatch):
    c, _ = chain(monkeypatch, [])
    with pytest.raises(LLMPermanentError, match="send less text per call"):
        c._raw_generate("x" * 30_000, None, True)  # about 10,000 tokens
    assert c.sent == []


# ---- the chain of models ---------------------------------------------------------------------------------------------------------------
def test_the_preferred_model_is_used_while_it_has_room_and_a_busy_one_hands_the_call_over_without_waiting(monkeypatch):
    c, clock = chain(monkeypatch, [reply(), reply(), reply()])
    c._raw_generate("hi", None, False)  # all free: the preferred model
    c.slots["m1"].tokens.reserve(7000)  # m1's minute is now full
    c._raw_generate("hi", None, False)  # m2 starts at once
    c._raw_generate("hi", None, False)
    assert models_used(c) == ["m1", "m2", "m2"] and max(clock.slept, default=0) <= 2.0  # nobody waited for m1's minute; the 2 second gap is not a reason to switch


def test_calls_stay_on_the_preferred_model_while_it_has_room(monkeypatch):
    c, clock = chain(monkeypatch, [reply(), reply(), reply()])
    for _ in range(3):
        c._raw_generate("hi", None, False)
    assert models_used(c) == ["m1", "m1", "m1"] and clock.slept == [2.0, 2.0]  # only the request spacing, never a switch


def test_a_rate_limited_call_is_retried_on_another_model_at_once(monkeypatch):
    c, clock = chain(monkeypatch, [reply(429, text="tokens per minute", headers={"retry-after": "30"}), reply()], max_attempts=3)
    assert c.generate_text("hi") == '{"ok": true}'
    assert models_used(c) == ["m1", "m2"] and c.retry_count == 1 and "m1 rate limited, paused 30s; the call goes to another model" in c.events
    assert max(clock.slept, default=0) < 30  # nobody waited for m1's pause
    assert c.slots["m1"].rate_limited == 1 and c.model_usage == {"m2": 1}
    # m1 is paused: while m2 and m3 are free the next calls do not go to m1
    c.sent.clear()


def test_a_model_with_no_other_to_try_waits_for_the_delay_groq_names(monkeypatch):
    c, clock = chain(monkeypatch, [reply(429, text="x", headers={"retry-after": "12"}), reply()], fallbacks=(), max_attempts=3)
    c.generate_text("hi")
    assert max(clock.slept) >= 11 and models_used(c) == ["m1", "m1"]


def test_a_model_that_hit_its_daily_quota_is_left_out_for_the_rest_of_the_run(monkeypatch):
    c, _ = chain(monkeypatch, [reply(429, text="tokens per day", headers={"retry-after": "7200"}), reply(), reply()], max_attempts=3)
    c.generate_text("hi")
    c.generate_text("hi")
    assert models_used(c) == ["m1", "m2", "m2"] and "quota reached" in c.slots["m1"].unavailable and any("left out for the rest of the run" in e for e in c.events)


def test_when_every_model_is_out_of_quota_the_error_is_permanent_and_names_them(monkeypatch):
    c, _ = chain(monkeypatch, [reply(429, text="d", headers={"retry-after": "7200"}), reply(429, text="d", headers={"retry-after": "7200"})], fallbacks=("m2",), max_attempts=4)
    with pytest.raises(LLMPermanentError, match="every model of the chain has used its quota"):
        c.generate_text("hi")
    with pytest.raises(LLMPermanentError, match="every model of the chain has used its quota"):
        c._raw_generate("hi", None, False)  # nothing is sent once every model is out


def test_a_model_the_key_cannot_use_is_left_out(monkeypatch):
    c, _ = chain(monkeypatch, [reply(404, text='{"error": {"code": "model_not_found"}}'), reply()], max_attempts=3)
    assert c.generate_text("hi") == '{"ok": true}' and c.slots["m1"].unavailable == "not available to this key" and models_used(c) == ["m1", "m2"]


def test_the_factory_builds_the_chain_from_the_settings():
    c = get_llm_client(Settings(llm_api_key="k", llm_provider="groq", llm_model="openai/gpt-oss-120b"))
    assert isinstance(c, GroqLLMClient) and list(c.slots) == ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]
    assert c.model == "openai/gpt-oss-120b" and c.slots["openai/gpt-oss-20b"].requests.interval == 2.0 and c.slots["qwen/qwen3.8-27b"].tokens.per_minute == 8000
    assert list(get_llm_client(Settings(llm_api_key="k", llm_provider="groq", llm_model="openai/gpt-oss-120b", llm_fallback_models=["openai/gpt-oss-120b"])).slots) == ["openai/gpt-oss-120b"]  # no duplicates
    with pytest.raises(LLMConfigError, match="LLM_API_KEY is not set"):
        get_llm_client(Settings(llm_provider="groq", llm_model="openai/gpt-oss-120b"))
    with pytest.raises(LLMConfigError, match="Third-party APIs are disabled"):
        get_llm_client(Settings(llm_api_key="k", llm_provider="groq", llm_model="openai/gpt-oss-120b", allow_third_party_apis=False))


def test_invalid_json_from_the_model_is_retried_on_another_model(monkeypatch):
    bad = reply(400, text='{"error":{"message":"Failed to validate JSON.","type":"invalid_request_error","code":"json_validate_failed"}}')
    c, _ = chain(monkeypatch, [bad, reply()], max_attempts=3)
    assert c.generate_text("hi") == '{"ok": true}'
    assert models_used(c) == ["m1", "m2"] and c.retry_count == 1 and any("returned invalid JSON" in e for e in c.events)


def test_the_model_that_gave_invalid_json_is_skipped_once_not_left_out(monkeypatch):
    bad = reply(400, text="json_validate_failed")
    c, _ = chain(monkeypatch, [bad, reply(), reply()], max_attempts=3)
    c.generate_text("hi")
    c.generate_text("hi")
    assert models_used(c) == ["m1", "m2", "m1"] and not c.slots["m1"].unavailable


def test_invalid_json_with_one_model_is_asked_again_on_it(monkeypatch):
    c, _ = chain(monkeypatch, [reply(400, text="json_validate_failed"), reply()], fallbacks=(), max_attempts=3)
    assert c.generate_text("hi") == '{"ok": true}' and models_used(c) == ["m1", "m1"]
