"""Google Gemini adapter (generateContent + batchEmbedContents) over plain HTTPS via httpx."""
from __future__ import annotations

import httpx

from app.llm.base import LLMClient, RawResponse, TokenUsage
from app.llm.embeddings import EmbeddingProvider, l2_normalize
from app.llm.errors import LLMPermanentError, LLMTransientError

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
TIMEOUT = httpx.Timeout(60.0, connect=10.0)


def _check(response: httpx.Response) -> dict:
    if response.status_code == 429 or response.status_code >= 500:
        retry_after = response.headers.get("retry-after")
        raise LLMTransientError(
            f"Google API {response.status_code}: {response.text[:200]}",
            retry_after=float(retry_after) if retry_after and retry_after.isdigit() else None,
        )
    if response.status_code >= 400:
        raise LLMPermanentError(f"Google API {response.status_code}: {response.text[:300]}")
    return response.json()


def _post(url: str, api_key: str, body: dict) -> dict:
    try:
        response = httpx.post(url, json=body, headers={"x-goog-api-key": api_key}, timeout=TIMEOUT)
    except (httpx.TimeoutException, httpx.TransportError) as exc:
        raise LLMTransientError(f"Google API network error: {exc}") from exc
    return _check(response)


class GoogleLLMClient(LLMClient):
    def __init__(self, model: str, api_key: str, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self._api_key = api_key

    def _raw_generate(self, prompt: str, system: str | None, json_mode: bool) -> RawResponse:
        body: dict = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        generation_config: dict = {"temperature": 0}
        if json_mode:
            generation_config["responseMimeType"] = "application/json"
        body["generationConfig"] = generation_config
        data = _post(f"{BASE_URL}/models/{self.model}:generateContent", self._api_key, body)
        try:
            parts = data["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts)
        except (KeyError, IndexError, TypeError) as exc:
            reason = (data.get("promptFeedback") or {}).get("blockReason") or "no candidates"
            raise LLMPermanentError(f"Google API returned no usable content ({reason})") from exc
        meta = data.get("usageMetadata", {})
        usage = TokenUsage(
            prompt_tokens=meta.get("promptTokenCount", 0),
            completion_tokens=meta.get("candidatesTokenCount", 0),
            thinking_tokens=meta.get("thoughtsTokenCount", 0),
            total_tokens=meta.get("totalTokenCount", 0),
        )
        return RawResponse(text=text, usage=usage)


class GoogleEmbeddingProvider(EmbeddingProvider):
    MAX_BATCH = 100

    def __init__(self, model: str, api_key: str, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self._api_key = api_key

    def _embed_batch(self, texts: list[str], task: str) -> list[list[float]]:
        task_type = "RETRIEVAL_QUERY" if task == "query" else "RETRIEVAL_DOCUMENT"
        body = {
            "requests": [
                {"model": f"models/{self.model}", "content": {"parts": [{"text": t}]}, "taskType": task_type,
                 "outputDimensionality": self.dimensions}
                for t in texts
            ]
        }
        data = _post(f"{BASE_URL}/models/{self.model}:batchEmbedContents", self._api_key, body)
        # Truncated (Matryoshka) vectors are not unit length, so normalise them ourselves.
        vectors = [l2_normalize(e["values"]) for e in data.get("embeddings", [])]
        if len(vectors) != len(texts):
            raise LLMPermanentError(f"Embedding API returned {len(vectors)} vectors for {len(texts)} texts")
        return vectors
