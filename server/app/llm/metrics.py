"""Per-call timing for LLM and embedding providers.

Every provider attempt is recorded, including failed and retried ones, so retry cost is
visible in latency numbers rather than hidden.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class CallMetric(BaseModel):
    kind: Literal["llm", "embedding", "tool"]
    latency_ms: int
    ok: bool = True
    error: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    thinking_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0  # input tokens served from the provider's prompt cache
    prompt_chars: int = 0  # size of what was sent (system + prompt, or the embedded texts)
    response_chars: int = 0
    items: int = 0  # texts in an embedding batch
    started_at: float = 0.0  # epoch seconds when the attempt began


class LatencySummary(BaseModel):
    calls: int = 0
    failed_calls: int = 0
    total_ms: int = 0
    avg_ms: int = 0
    max_ms: int = 0
    p95_ms: int = 0


def summarize(metrics: list[CallMetric]) -> LatencySummary:
    """Summary over successful attempts' latency, with failed attempts counted separately
    (their time is still included in total_ms, because it was really spent waiting)."""
    if not metrics:
        return LatencySummary()
    ok = sorted(m.latency_ms for m in metrics if m.ok)
    total = sum(m.latency_ms for m in metrics)
    p95 = ok[min(len(ok) - 1, int(0.95 * len(ok)))] if ok else 0
    return LatencySummary(
        calls=len(metrics), failed_calls=sum(1 for m in metrics if not m.ok), total_ms=total,
        avg_ms=round(sum(ok) / len(ok)) if ok else 0, max_ms=max((m.latency_ms for m in metrics), default=0), p95_ms=p95,
    )
