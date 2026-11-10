import time

from app.llm.embeddings import MockEmbeddingProvider
from app.llm.errors import LLMPermanentError, LLMTransientError
from app.llm.metrics import CallMetric, summarize
from app.llm.mock import MockLLMClient


def test_every_attempt_is_timed_including_failures_and_retries():
    llm = MockLLMClient([LLMTransientError("503"), "ok"])
    llm.generate_text("a reasonably long prompt so the mock token estimate is non-zero")
    assert [m.ok for m in llm.call_metrics] == [False, True]
    assert llm.call_metrics[0].error and llm.call_metrics[1].total_tokens > 0
    s = llm.latency_summary()
    assert s.calls == 2 and s.failed_calls == 1 and s.total_ms >= s.max_ms


def test_permanent_failure_is_recorded_before_raising():
    llm = MockLLMClient([LLMPermanentError("401")])
    try:
        llm.generate_text("q")
    except LLMPermanentError:
        pass
    assert len(llm.call_metrics) == 1 and not llm.call_metrics[0].ok


def test_latency_reflects_real_elapsed_time():
    class Slow(MockLLMClient):
        def _raw_generate(self, prompt, system, json_mode):
            time.sleep(0.05)
            return super()._raw_generate(prompt, system, json_mode)

    llm = Slow(["x"])
    llm.generate_text("q")
    assert llm.call_metrics[0].latency_ms >= 45


def test_embedding_calls_are_timed():
    emb = MockEmbeddingProvider(dimensions=8)
    emb.embed(["a", "b"])
    assert len(emb.call_metrics) == 1 and emb.call_metrics[0].kind == "embedding"
    assert emb.latency_summary().calls == 1


def test_summary_math_and_empty():
    assert summarize([]).calls == 0
    ms = [CallMetric(kind="llm", latency_ms=v) for v in (100, 200, 300, 400)]
    ms.append(CallMetric(kind="llm", latency_ms=1000, ok=False))
    s = summarize(ms)
    assert (s.calls, s.failed_calls, s.total_ms, s.avg_ms, s.max_ms) == (5, 1, 2000, 250, 1000)


def test_thinking_tokens_accumulate():
    from app.llm.base import TokenUsage
    u = TokenUsage()
    u.add(TokenUsage(thinking_tokens=200, calls=1))
    u.add(TokenUsage(thinking_tokens=50, calls=1))
    assert u.thinking_tokens == 250
