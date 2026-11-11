import pytest
from pydantic import BaseModel

from app.core.config import Settings
from app.llm.embeddings import MockEmbeddingProvider
from app.llm.errors import (LLMConfigError, LLMInvalidOutputError, LLMPermanentError,
                            LLMTransientError)
from app.llm.factory import get_embedding_provider, get_llm_client
from app.llm.mock import MockLLMClient


class Out(BaseModel):
    answer: int


def test_structured_output_valid_and_fenced_json():
    llm = MockLLMClient(['```json\n{"answer": 5}\n```'])
    assert llm.generate_structured_output("q", Out).answer == 5
    assert llm.get_token_usage().calls == 1 and llm.get_token_usage().total_tokens > 0
    assert "JSON Schema" in llm.prompts[0]  # schema is shown to the model


def test_invalid_output_retried_then_succeeds():
    llm = MockLLMClient(["not json", {"answer": "x"}, {"answer": 2}])
    assert llm.generate_structured_output("q", Out).answer == 2
    assert "previous reply was invalid" in llm.prompts[1]


def test_invalid_output_is_bounded():
    llm = MockLLMClient(["bad"] * 10, max_attempts=3)
    with pytest.raises(LLMInvalidOutputError):
        llm.generate_structured_output("q", Out)
    assert len(llm.prompts) == 3


def test_transient_errors_retry_with_backoff_then_succeed():
    delays = []
    llm = MockLLMClient([LLMTransientError("429"), LLMTransientError("503"), "ok"], sleep=delays.append, base_delay=1)
    assert llm.generate_text("q") == "ok"
    assert delays == [1, 2] and llm.retry_count == 2


def test_transient_errors_exhaust_budget():
    llm = MockLLMClient([LLMTransientError("429")] * 5, max_attempts=3)
    with pytest.raises(LLMTransientError):
        llm.generate_text("q")
    assert len(llm.prompts) == 3


def test_permanent_error_not_retried():
    llm = MockLLMClient([LLMPermanentError("401"), "never used"])
    with pytest.raises(LLMPermanentError):
        llm.generate_text("q")
    assert len(llm.prompts) == 1


def test_missing_api_key_is_config_error_not_fallback():
    with pytest.raises(LLMConfigError, match="LLM_API_KEY"):
        get_llm_client(Settings(llm_provider="google", llm_model="m", llm_api_key=""))
    with pytest.raises(LLMConfigError, match="LLM_MODEL"):
        get_llm_client(Settings(llm_provider="google", llm_model="", llm_api_key="k"))
    with pytest.raises(LLMConfigError, match="Unsupported"):
        get_llm_client(Settings(llm_provider="nope"))
    with pytest.raises(LLMConfigError, match="EMBEDDING_MODEL"):
        get_embedding_provider(Settings(llm_provider="google", llm_api_key="k", embedding_model=""))


def test_embedding_provider_defaults_to_llm_provider_and_dimensions():
    emb = get_embedding_provider(Settings(llm_provider="mock", embedding_dimensions=64))
    assert len(emb.embed(["hello"])[0]) == 64


def test_mock_embeddings_similar_texts_are_closer():
    emb = MockEmbeddingProvider(dimensions=256)
    a, b, c = emb.embed(["annual production capacity", "production capacity per year", "office lease terms"])
    dot = lambda x, y: sum(p * q for p, q in zip(x, y))
    assert dot(a, b) > dot(a, c)


def test_embedding_retries_transient_failures():
    calls = []

    class Flaky(MockEmbeddingProvider):
        def _embed_batch(self, texts, task):
            calls.append(1)
            if len(calls) < 3:
                raise LLMTransientError("429")
            return super()._embed_batch(texts, task)

    emb = Flaky(max_attempts=3, sleep=lambda _s: None)
    assert len(emb.embed(["x"])) == 1 and len(calls) == 3


# ---- Google rate-limit handling ----------------------------------------------
def test_google_429_wait_time_is_read_from_the_json_body_or_header():
    import httpx

    from app.llm.google import _check, retry_delay_seconds

    body = {"error": {"code": 429, "details": [
        {"@type": "type.googleapis.com/google.rpc.QuotaFailure"},
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "27s"}]}}
    assert retry_delay_seconds(httpx.Response(429, json=body)) == 27.0
    assert retry_delay_seconds(httpx.Response(429, headers={"retry-after": "5"}, json=body)) == 5.0  # header wins
    assert retry_delay_seconds(httpx.Response(429, text="not json")) is None
    assert retry_delay_seconds(httpx.Response(429, json={"error": {"details": "odd"}})) is None
    with pytest.raises(LLMTransientError) as info:
        _check(httpx.Response(429, json=body))
    assert info.value.retry_after == 27.0
