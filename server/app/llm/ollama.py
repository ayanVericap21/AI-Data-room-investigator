"""LLM client for a model served by Ollama on this machine (no API key, no quota, nothing leaves the computer).

Structured output uses Ollama's schema-constrained ``format``: the model can only produce JSON that fits the
Pydantic model, and the base class still validates it and retries a bounded number of times. Everything else
(retries, token accounting, call metrics) is inherited, so this behaves like every other provider.
"""
from __future__ import annotations

from typing import Any, TypeVar

import httpx
import ollama
from pydantic import BaseModel

from app.llm.base import LLMClient, RawResponse, TokenUsage
from app.llm.errors import LLMPermanentError, LLMTransientError

T = TypeVar("T", bound=BaseModel)


class OllamaLLMClient(LLMClient):
    def __init__(self, model: str, base_url: str = "http://localhost:11434", context_tokens: int = 16384,
                 timeout: float = 900.0, think: bool = False, client: Any | None = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.model = model
        self.base_url = base_url
        self.context_tokens = context_tokens  # Ollama's default window is small; every request sets this one
        self.think = think  # reasoning models: False asks for no hidden reasoning (faster; ignored by models without it)
        self._client = client or ollama.Client(host=base_url, timeout=timeout)
        self._schema: dict[str, Any] | None = None

    def generate_structured_output(self, prompt: str, schema: type[T], system: str | None = None) -> T:
        self._schema = schema.model_json_schema()  # constrains the reply to this schema (see _raw_generate)
        try:
            return super().generate_structured_output(prompt, schema, system)
        finally:
            self._schema = None

    def _raw_generate(self, prompt: str, system: str | None, json_mode: bool) -> RawResponse:
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        options = {"temperature": 0, "num_ctx": self.context_tokens}
        fmt: Any = (self._schema or "json") if json_mode else None
        try:
            try:
                response = self._client.chat(model=self.model, messages=messages, format=fmt, options=options, think=self.think)
            except ollama.ResponseError as exc:
                if "think" not in str(exc).lower():
                    raise
                response = self._client.chat(model=self.model, messages=messages, format=fmt, options=options)  # no thinking mode
        except ollama.ResponseError as exc:
            status = exc.status_code
            if status == 404:
                raise LLMPermanentError(f"Ollama has no model '{self.model}' (download it first: ollama pull {self.model})") from exc
            if status is not None and status >= 500:
                raise LLMTransientError(f"Ollama error {status}: {str(exc)[:200]}") from exc
            raise LLMPermanentError(f"Ollama rejected the request ({status}): {str(exc)[:200]}") from exc
        except (httpx.TimeoutException, TimeoutError) as exc:
            raise LLMTransientError(f"Ollama did not answer in time: {exc.__class__.__name__}") from exc
        except (ConnectionError, httpx.ConnectError) as exc:
            raise LLMPermanentError(f"Cannot reach Ollama at {self.base_url}: is the Ollama app running?") from exc

        text = response.message.content or ""
        prompt_tokens, completion_tokens = response.prompt_eval_count or 0, response.eval_count or 0
        return RawResponse(text=text, usage=TokenUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                                                       total_tokens=prompt_tokens + completion_tokens))
