"""The Ollama provider: schema-constrained output, token accounting, and error mapping. The Ollama server is faked."""
from __future__ import annotations

from types import SimpleNamespace

import httpx
import ollama
import pytest
from pydantic import BaseModel

from app.llm.errors import LLMInvalidOutputError, LLMPermanentError, LLMTransientError
from app.llm.ollama import OllamaLLMClient


class Pong(BaseModel):
    answer: str


class FakeOllama:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(message=SimpleNamespace(content=reply), prompt_eval_count=30, eval_count=7)


def client(replies, **kw):
    return OllamaLLMClient("m", client=FakeOllama(replies), max_attempts=2, base_delay=0, sleep=lambda _s: None, **kw)


def test_the_reply_is_constrained_to_the_schema_and_validated_and_tokens_are_counted():
    llm = client(['{"answer": "ok"}'], context_tokens=4096)
    assert llm.generate_structured_output("say ok", Pong, system="be brief").answer == "ok"
    call = llm._client.calls[0]
    assert call["format"] == Pong.model_json_schema() and call["options"] == {"temperature": 0, "num_ctx": 4096}
    assert call["think"] is False and call["messages"][0] == {"role": "system", "content": "be brief"}
    usage = llm.get_token_usage()
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens, usage.calls) == (30, 7, 37, 1)
    assert llm.call_metrics[0].ok and llm._schema is None  # the schema is only attached for the one call


def test_plain_text_calls_have_no_format():
    llm = client(["hello"])
    assert llm.generate_text("hi") == "hello" and llm._client.calls[0]["format"] is None


def test_invalid_json_is_retried_then_reported():
    assert client(["not json", '{"answer": "ok"}']).generate_structured_output("q", Pong).answer == "ok"
    with pytest.raises(LLMInvalidOutputError):
        client(["nope", "still nope"]).generate_structured_output("q", Pong)


def test_a_model_without_thinking_support_is_retried_without_it():
    llm = client([ollama.ResponseError("model does not support thinking", 400), '{"answer": "ok"}'])
    assert llm.generate_structured_output("q", Pong).answer == "ok"
    assert "think" in llm._client.calls[0] and "think" not in llm._client.calls[1]


def test_errors_are_mapped_so_a_missing_model_or_dead_server_fails_fast_and_a_busy_one_is_retried():
    with pytest.raises(LLMPermanentError, match="ollama pull"):
        client([ollama.ResponseError("model not found", 404)]).generate_text("q")
    with pytest.raises(LLMPermanentError, match="Cannot reach Ollama"):
        client([ConnectionError("refused")]).generate_text("q")
    with pytest.raises(LLMPermanentError, match="rejected"):
        client([ollama.ResponseError("bad request", 400)]).generate_text("q")
    busy = client([ollama.ResponseError("overloaded", 503), "recovered"])
    assert busy.generate_text("q") == "recovered" and busy.retry_count == 1
    with pytest.raises(LLMTransientError):
        client([httpx.ReadTimeout("slow"), httpx.ReadTimeout("slow")]).generate_text("q")
