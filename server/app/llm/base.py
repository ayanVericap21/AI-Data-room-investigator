"""Provider-independent LLM interface used by the LangGraph workflow and services.

Adapters implement only the two ``_raw_*`` hooks; validation, bounded retries with
backoff, and token accounting live here so every provider behaves identically.
"""
from __future__ import annotations

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from typing import Callable, TypeVar

from pydantic import BaseModel, ValidationError

from app.llm.errors import LLMError, LLMInvalidOutputError, LLMTransientError
from app.llm.metrics import CallMetric, LatencySummary, summarize

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    thinking_tokens: int = 0  # hidden reasoning tokens (billed, and a main source of latency)
    total_tokens: int = 0
    calls: int = 0

    def add(self, other: "TokenUsage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.thinking_tokens += other.thinking_tokens
        self.total_tokens += other.total_tokens
        self.calls += other.calls


class RawResponse(BaseModel):
    text: str
    usage: TokenUsage = TokenUsage()


def extract_json(text: str) -> object:
    """Parse model text as JSON, tolerating ```json fences."""
    stripped = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, re.DOTALL)
    if fence:
        stripped = fence.group(1)
    return json.loads(stripped)


class LLMClient(ABC):
    def __init__(self, max_attempts: int = 3, base_delay: float = 1.0, sleep: Callable[[float], None] = time.sleep):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self._sleep = sleep
        self._usage = TokenUsage()
        self.retry_count = 0  # total retries performed (for run metadata)
        self.call_metrics: list[CallMetric] = []  # one entry per provider attempt

    # -- provider hooks ------------------------------------------------------
    @abstractmethod
    def _raw_generate(self, prompt: str, system: str | None, json_mode: bool) -> RawResponse:
        """One provider call. Raise LLMTransientError / LLMPermanentError."""

    # -- public API ----------------------------------------------------------
    def get_token_usage(self) -> TokenUsage:
        return self._usage.model_copy()

    def latency_summary(self) -> LatencySummary:
        return summarize(self.call_metrics)

    def generate_text(self, prompt: str, system: str | None = None) -> str:
        return self._call(prompt, system, json_mode=False).text

    def generate_structured_output(self, prompt: str, schema: type[T], system: str | None = None) -> T:
        """Return a validated ``schema`` instance. The schema is shown to the model in
        the prompt (provider-neutral); validation is always done by Pydantic here."""
        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        full_prompt = (
            f"{prompt}\n\nRespond with ONLY a JSON object that conforms to this JSON Schema "
            f"(no prose, no markdown):\n{schema_json}"
        )
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            response = self._call(full_prompt, system, json_mode=True)
            try:
                return schema.model_validate(extract_json(response.text))
            except (ValueError, ValidationError) as exc:  # JSONDecodeError is a ValueError
                last_error = exc
                self.retry_count += 1
                logger.warning("Invalid structured output", extra={"attempt": attempt, "schema": schema.__name__})
                full_prompt += f"\n\nYour previous reply was invalid ({str(exc)[:300]}). Return valid JSON only."
        raise LLMInvalidOutputError(
            f"Model output did not match {schema.__name__} after {self.max_attempts} attempts: {last_error}"
        )

    # -- internals -----------------------------------------------------------
    def _call(self, prompt: str, system: str | None, json_mode: bool) -> RawResponse:
        """Provider call with bounded exponential backoff for transient errors only."""
        for attempt in range(1, self.max_attempts + 1):
            started = time.perf_counter()
            try:
                response = self._raw_generate(prompt, system, json_mode)
            except LLMTransientError as exc:
                self._record(started, None, exc)
                if attempt == self.max_attempts:
                    raise
                self.retry_count += 1
                delay = exc.retry_after if exc.retry_after is not None else self.base_delay * (2 ** (attempt - 1))
                logger.warning("Transient LLM error, retrying", extra={"attempt": attempt, "delay": delay, "error": str(exc)})
                self._sleep(min(delay, 60))
                continue
            except LLMError as exc:  # permanent: record the attempt, then let it propagate
                self._record(started, None, exc)
                raise
            self._record(started, response.usage, None)
            usage = response.usage.model_copy()
            usage.calls = 1
            self._usage.add(usage)
            return response
        raise AssertionError("unreachable")

    def _record(self, started: float, usage: TokenUsage | None, error: Exception | None) -> None:
        self.call_metrics.append(CallMetric(
            kind="llm", latency_ms=round((time.perf_counter() - started) * 1000), ok=error is None,
            error=str(error)[:200] if error else None,
            prompt_tokens=usage.prompt_tokens if usage else 0, completion_tokens=usage.completion_tokens if usage else 0,
            thinking_tokens=usage.thinking_tokens if usage else 0, total_tokens=usage.total_tokens if usage else 0,
        ))
