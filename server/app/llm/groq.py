"""Groq adapter (OpenAI-compatible chat completions) over plain HTTPS via httpx, with rate limits and fallback models.

Groq limits each model separately: requests per minute, tokens per minute (8,000 on the key this project was set up with, the limit that binds) and
requests and tokens per day. The client keeps a request limiter and a one-minute token window for every model it may use. Every call picks, among the
models that still have quota, the first in the preferred order that can start within ``patience`` seconds (else the one that can start soonest), reserves its tokens
there (the prompt, estimated from its length, plus a reserve for the answer, replaced by the real count once the answer is back) and takes a request
slot. So parallel workers never trip a limit, and a model that is busy or rate limited hands the call to the next one instead of making it wait.
The preferred model keeps the call unless it would have to wait longer than ``patience`` seconds (the short gap between two requests is not a reason to
switch), so one run uses one model for as long as that model has room.

When Groq answers 429 anyway, the model is paused for the delay it names and the call is retried on another model at once. A 429 that asks for a very
long wait is a daily quota: that model is left out for the rest of the run. A request that cannot fit in the token window even when it is empty is
refused at once: it must be sent in smaller batches. Only chat models belong in the chain: the prompt-guard models are text classifiers and reject chat
requests.
"""
from __future__ import annotations

import re
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

import httpx

from app.llm.base import LLMClient, RawResponse, TokenUsage
from app.llm.errors import LLMPermanentError, LLMTransientError

BASE_URL = "https://api.groq.com/openai/v1"
TIMEOUT = httpx.Timeout(120.0, connect=10.0)  # a reasoning model can take a while to answer
LONG_WAIT_SECONDS = 300.0  # a 429 that asks for more than this is a quota that will not clear within a run
THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)  # some models write their reasoning into the answer


class RateLimiter:
    """At most ``per_minute`` call starts a minute, shared by every thread: each caller reserves the next free slot and waits for it."""

    def __init__(self, per_minute: int, clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep):
        self.interval = 60.0 / max(1, per_minute)
        self._clock, self._sleep = clock, sleep
        self._lock = threading.Lock()
        self._next = 0.0
        self.waited = 0.0  # seconds callers spent waiting for a slot (for the run report)

    def wait_for(self) -> float:
        """Seconds until a call could start."""
        with self._lock:
            return max(0.0, self._next - self._clock())

    def acquire(self) -> None:
        with self._lock:
            now = self._clock()
            start = max(now, self._next)
            self._next = start + self.interval
            wait = start - now
            self.waited += max(0.0, wait)
        if wait > 0:
            self._sleep(wait)

    def pause(self, seconds: float) -> None:
        """No call starts for the next ``seconds`` (the provider asked everyone to back off)."""
        with self._lock:
            self._next = max(self._next, self._clock() + seconds)


class TokenLimiter:
    """At most ``per_minute`` tokens in any 60 seconds, shared by every thread. A caller reserves an estimate, waits until it fits, and settles the
    entry with the real count afterwards."""

    WINDOW = 60.0

    def __init__(self, per_minute: int, clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep):
        self.per_minute = max(1, per_minute)
        self._clock, self._sleep = clock, sleep
        self._lock = threading.Lock()
        self._entries: list[list[float]] = []  # [time, tokens]
        self.waited = 0.0

    def check(self, tokens: int) -> None:
        if tokens > self.per_minute:
            raise LLMPermanentError(f"a request of about {tokens} tokens cannot fit in the limit of {self.per_minute} tokens per minute; send less text per call")

    def wait_for(self, tokens: int) -> float:
        """Seconds until ``tokens`` would fit in the window (0 when they fit now)."""
        self.check(tokens)
        with self._lock:
            now = self._clock()
            live = sorted((e for e in self._entries if e[0] > now - self.WINDOW), key=lambda e: e[0])
            used = sum(e[1] for e in live)
            if used + tokens <= self.per_minute:
                return 0.0
            for e in live:  # entries leave the window oldest first
                used -= e[1]
                if used + tokens <= self.per_minute:
                    return max(0.0, e[0] + self.WINDOW - now)
            return 0.0

    def reserve(self, tokens: int) -> list[float]:
        self.check(tokens)
        while True:
            with self._lock:
                now = self._clock()
                self._entries = [e for e in self._entries if e[0] > now - self.WINDOW]
                if sum(e[1] for e in self._entries) + tokens <= self.per_minute:
                    entry = [now, float(tokens)]
                    self._entries.append(entry)
                    return entry
                wait = max(self._entries[0][0] + self.WINDOW - now, 0.05)
                self.waited += wait
            self._sleep(wait)

    def settle(self, entry: list[float], actual: int) -> None:
        with self._lock:
            entry[1] = float(actual)

    def pause(self, seconds: float) -> None:
        """Treat the window as full for the next ``seconds`` (the provider said the token limit was reached)."""
        with self._lock:
            now = self._clock()
            self._entries.append([now - self.WINDOW + max(seconds, 0.0), float(self.per_minute)])


def estimate_tokens(prompt: str, system: str | None, json_mode: bool, chars_per_token: float = 3.0) -> int:
    """An estimate of the prompt's tokens (the JSON schema the base class appends is already part of the prompt). Until the client has seen real
    counts it assumes 3 characters to a token, the cautious figure a repetitive test text measured."""
    return int(-(-(len(prompt) + len(system or "")) // chars_per_token)) + (30 if json_mode else 0)


def retry_delay_seconds(response: httpx.Response) -> float | None:
    """How long Groq asks us to wait (the Retry-After header, in seconds), or None."""
    header = response.headers.get("retry-after")
    try:
        return float(header) if header is not None else None
    except ValueError:
        return None


@dataclass
class ModelSlot:
    """The limits and the state of one model of the chain."""

    model: str
    requests: RateLimiter
    tokens: TokenLimiter
    unavailable: str = ""  # why the model is left out for the rest of the run (a daily quota, a retired model)
    calls: int = 0
    total_tokens: int = 0
    rate_limited: int = 0
    avoid_next: bool = False  # the model's last answer was invalid JSON: the next call goes to another model when one is free


class GroqLLMClient(LLMClient):
    def __init__(self, model: str, api_key: str, fallback_models: tuple[str, ...] | list[str] = (), requests_per_minute: int = 30,
                 tokens_per_minute: int = 8000, answer_tokens: int = 1500, clock: Callable[[], float] = time.monotonic,
                 wait: Callable[[float], None] = time.sleep, patience: float = 5.0, **kwargs):
        super().__init__(**kwargs)
        self.patience = patience  # seconds a model may make a call wait before the call goes to the next model
        self.model = model  # the preferred model
        self._api_key = api_key
        self.slots = {m: ModelSlot(m, RateLimiter(requests_per_minute, clock, wait), TokenLimiter(tokens_per_minute, clock, wait))
                      for m in dict.fromkeys([model, *fallback_models])}
        self.answer_tokens = answer_tokens  # tokens reserved for the answer (reasoning and JSON) until the real count is known
        self.events: list[str] = []  # what made the chain switch models, for the run report
        self._chars_sent = 0  # characters and prompt tokens of the answered calls: the real characters-per-token of this run's text
        self._prompt_tokens_seen = 0

    def chars_per_token(self) -> float:
        """What the answered calls measured (documents and instructions run about 3 to 4 characters to a token); 3.0 until 2,000 tokens were seen."""
        if self._prompt_tokens_seen < 2000:
            return 3.0
        return min(5.0, max(2.5, self._chars_sent / self._prompt_tokens_seen))

    # -- choosing the model --------------------------------------------------------------------------------------------------------------
    def _choose(self, tokens: int) -> ModelSlot:
        """The first model in the preferred order that can start within ``patience`` seconds; when none can, the one that can start soonest."""
        live = [s for s in self.slots.values() if not s.unavailable]
        if not live:
            raise LLMPermanentError("every model of the chain has used its quota: " + "; ".join(f"{s.model}: {s.unavailable}" for s in self.slots.values()))
        fresh = [s for s in live if not s.avoid_next]
        for s in live:
            s.avoid_next = False  # a model is skipped once, for the call that follows its invalid answer
        live = fresh or live
        for s in live:
            s.tokens.check(tokens)
        need = {s.model: max(s.tokens.wait_for(tokens), s.requests.wait_for()) for s in live}
        quick = [s for s in live if need[s.model] <= self.patience]
        return quick[0] if quick else min(live, key=lambda s: need[s.model])

    def _others_left(self, slot: ModelSlot) -> bool:
        return any(not s.unavailable and s is not slot for s in self.slots.values())

    def _note(self, text: str) -> None:
        with self._lock:
            self.events.append(text)

    # -- one call -------------------------------------------------------------------------------------------------------------------------
    def _raw_generate(self, prompt: str, system: str | None, json_mode: bool) -> RawResponse:
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        estimate = estimate_tokens(prompt, system, json_mode, self.chars_per_token()) + self.answer_tokens
        slot = self._choose(estimate)
        reservation = slot.tokens.reserve(estimate)  # may wait for the minute's tokens
        slot.requests.acquire()  # every attempt, retries included, takes a slot
        body: dict = {"model": slot.model, "messages": messages, "temperature": 0}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        try:
            response = httpx.post(f"{BASE_URL}/chat/completions", json=body, headers={"Authorization": f"Bearer {self._api_key}"}, timeout=TIMEOUT)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            slot.tokens.settle(reservation, 0)
            raise LLMTransientError(f"Groq API network error: {exc}") from exc
        if response.status_code == 429:
            delay = retry_delay_seconds(response)
            slot.rate_limited += 1
            if delay is not None and delay > LONG_WAIT_SECONDS:
                slot.unavailable = f"quota reached (resets in {delay / 3600:.1f} h)"
                self._note(f"{slot.model} left out for the rest of the run: {slot.unavailable}")
                if not self._others_left(slot):
                    raise LLMPermanentError(f"Groq API 429: every model of the chain has used its quota; last: {slot.model} ({slot.unavailable}): {response.text[:160]}")
                raise LLMTransientError(f"Groq API 429 on {slot.model}: daily quota; trying another model", retry_after=0.0, status=429)
            pause = delay if delay is not None else slot.requests.interval
            slot.requests.pause(pause)
            slot.tokens.pause(pause)
            self._note(f"{slot.model} rate limited, paused {pause:.0f}s" + ("; the call goes to another model" if self._others_left(slot) else ""))
            raise LLMTransientError(f"Groq API 429 on {slot.model}: {response.text[:160]}", retry_after=0.0 if self._others_left(slot) else delay, status=429)
        if response.status_code >= 500:
            raise LLMTransientError(f"Groq API {response.status_code}: {response.text[:200]}", retry_after=retry_delay_seconds(response), status=response.status_code)
        if response.status_code == 413:
            raise LLMPermanentError(f"Groq API 413: the request is larger than the model's token limit; send less text per call: {response.text[:200]}")
        if response.status_code == 404 or (response.status_code == 400 and ("model_not_found" in response.text or "decommissioned" in response.text)):
            slot.unavailable = "not available to this key"
            self._note(f"{slot.model} left out for the rest of the run: {slot.unavailable}")
            if not self._others_left(slot):
                raise LLMPermanentError(f"Groq API {response.status_code}: no model of the chain is available: {response.text[:200]}")
            raise LLMTransientError(f"Groq API {response.status_code} on {slot.model}: not available; trying another model", retry_after=0.0, status=response.status_code)
        if response.status_code == 400 and "json_validate_failed" in response.text:
            # the model wrote an answer that is not valid JSON (the request itself is fine): ask again, on another model when one is free
            slot.tokens.settle(reservation, int(reservation[1]))
            slot.avoid_next = True
            self._note(f"{slot.model} returned invalid JSON; the call is asked again" + (" on another model" if self._others_left(slot) else ""))
            raise LLMTransientError(f"Groq API 400 on {slot.model}: invalid JSON in the answer (json_validate_failed)", retry_after=0.0, status=400)
        if response.status_code >= 400:
            raise LLMPermanentError(f"Groq API {response.status_code}: {response.text[:300]}")
        data = response.json()
        try:
            choice = data["choices"][0]
            text = THINK_BLOCK.sub("", choice["message"]["content"] or "").strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMPermanentError("Groq API returned no usable content") from exc
        usage = data.get("usage") or {}
        used = int(usage.get("total_tokens") or 0)
        slot.tokens.settle(reservation, used or int(reservation[1]))
        with self._lock:
            slot.calls += 1
            slot.total_tokens += used
            self._chars_sent += len(prompt) + len(system or "")
            self._prompt_tokens_seen += int(usage.get("prompt_tokens") or 0)
        if not text:
            raise LLMTransientError(f"Groq API returned an empty answer from {slot.model} (finish reason: {choice.get('finish_reason')})")
        return RawResponse(text=text, usage=TokenUsage(
            prompt_tokens=usage.get("prompt_tokens", 0), completion_tokens=usage.get("completion_tokens", 0),
            thinking_tokens=(usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0),
            cached_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0), total_tokens=used))

    @property
    def model_usage(self) -> Counter:
        return Counter({s.model: s.calls for s in self.slots.values() if s.calls})
