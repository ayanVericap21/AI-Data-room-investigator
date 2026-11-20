"""Google Gemini adapter (generateContent + batchEmbedContents) over plain HTTPS via httpx."""
from __future__ import annotations

import re

import httpx

from app.llm.base import LLMClient, RawResponse, TokenUsage
from app.llm.embeddings import EmbeddingProvider, l2_normalize
from app.llm.errors import LLMError, LLMPermanentError, LLMQuotaError, LLMTransientError
from app.llm.groq import RateLimiter

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
TIMEOUT = httpx.Timeout(180.0, connect=10.0)  # a call with tens of thousands of answer tokens takes a minute or more


def retry_delay_seconds(response: httpx.Response) -> float | None:
    """How long Google asks us to wait: the Retry-After header, or the RetryInfo.retryDelay
    (e.g. "27s") that Google puts in the JSON error body of a 429."""
    header = response.headers.get("retry-after")
    if header and header.isdigit():
        return float(header)
    try:
        for detail in response.json().get("error", {}).get("details", []):
            delay = detail.get("retryDelay")
            if isinstance(delay, str) and delay.endswith("s"):
                return float(delay[:-1])
    except (ValueError, AttributeError, TypeError):
        pass
    return None


DAILY_QUOTA = re.compile(r'"quotaId"\s*:\s*"([^"]*PerDay[^"]*)"')  # e.g. GenerateRequestsPerDayPerProjectPerModel-FreeTier in the details of a 429


def _check(response: httpx.Response) -> dict:
    if response.status_code == 429:
        daily = DAILY_QUOTA.search(response.text)
        if daily:  # a limit per day does not clear within a run: waiting and asking again only uses up more of it
            raise LLMQuotaError(f"Google API 429: the daily quota ({daily.group(1)}) is used up: {response.text[:400]}")
    if response.status_code == 429 or response.status_code >= 500:
        raise LLMTransientError(
            f"Google API {response.status_code}: {response.text}", retry_after=retry_delay_seconds(response), status=response.status_code,
        )
    if response.status_code >= 400:
        raise LLMPermanentError(f"Google API {response.status_code}: {response.text}")
    return response.json()


def _post(url: str, api_key: str, body: dict) -> dict:
    try:
        response = httpx.post(url, json=body, headers={"x-goog-api-key": api_key}, timeout=TIMEOUT)
    except (httpx.TimeoutException, httpx.TransportError) as exc:
        raise LLMTransientError(f"Google API network error: {exc}") from exc
    return _check(response)


class UnsupportedSchema(Exception):
    """The schema uses something Gemini's response schema cannot express."""


_GEMINI_TYPES = {"string": "STRING", "number": "NUMBER", "integer": "INTEGER", "boolean": "BOOLEAN", "array": "ARRAY", "object": "OBJECT"}
_COPIED = ("format", "description", "minimum", "maximum", "minItems", "maxItems")
_DROPPED = {"title", "default", "examples", "$defs"}  # carry no constraint Gemini enforces
_HANDLED = {"type", "properties", "required", "items", "anyOf", "$ref", "enum", "additionalProperties"}


def gemini_schema(json_schema: dict) -> dict | None:
    """A Pydantic JSON schema as Gemini's ``responseSchema`` (OpenAPI subset, upper-case types, no $ref, a ``propertyOrdering`` so the
    model writes the fields in the order the model class declares them), or None when it holds something that subset cannot express
    (a free-form object, a union of several types, a constant, a pattern): the caller then pastes the schema into the prompt."""
    defs = json_schema.get("$defs", {})

    def convert(node: dict, depth: int = 0) -> dict:
        if depth > 12:
            raise UnsupportedSchema("recursive schema")
        if "$ref" in node:
            return convert(defs[node["$ref"].split("/")[-1]], depth + 1)
        unknown = set(node) - _DROPPED - _HANDLED - set(_COPIED)
        if unknown:
            raise UnsupportedSchema(f"unsupported keyword {sorted(unknown)}")
        if "anyOf" in node:  # an optional field: the one real type and null
            options = [o for o in node["anyOf"] if o.get("type") != "null"]
            if len(options) != 1 or len(node["anyOf"]) != 2:
                raise UnsupportedSchema("a union of several types")
            return {**convert(options[0], depth + 1), "nullable": True}
        kind = node.get("type")
        if kind not in _GEMINI_TYPES:
            raise UnsupportedSchema(f"type {kind!r}")
        out: dict = {"type": _GEMINI_TYPES[kind], **{k: node[k] for k in _COPIED if k in node}}
        if "enum" in node:
            if kind != "string" or not all(isinstance(v, str) for v in node["enum"]):
                raise UnsupportedSchema("an enum that is not made of strings")
            out["enum"] = list(node["enum"])
        if kind == "object":
            properties = node.get("properties")
            if not properties or node.get("additionalProperties") not in (None, False):
                raise UnsupportedSchema("a free-form object")
            out["properties"] = {name: convert(sub, depth + 1) for name, sub in properties.items()}
            out["propertyOrdering"] = list(properties)
            if node.get("required"):
                out["required"] = list(node["required"])
        elif kind == "array":
            if "items" not in node:
                raise UnsupportedSchema("an array without items")
            out["items"] = convert(node["items"], depth + 1)
        return out

    try:
        return convert(json_schema)
    except (UnsupportedSchema, KeyError):
        return None


def _key_refused(exc: LLMPermanentError) -> bool:
    """Google refuses this key (invalid, revoked, no access): the next key may still work, so only this one is left out."""
    text = str(exc)
    return text.startswith(("Google API 401", "Google API 403")) or "API_KEY_INVALID" in text or "API key not valid" in text


class _KeySlot:
    """One API key with its own request pacing (the limits are per key) and its own record for the run."""

    def __init__(self, key: str, limiter: RateLimiter | None):
        self.key, self.limiter = key, limiter
        self.out = ""  # why this key is left out for the rest of the run (its daily quota is used up, or Google refuses it)
        self.calls = 0  # calls this key answered
        self.failed = 0  # attempts that failed on this key


class GoogleLLMClient(LLMClient):
    def __init__(self, model: str, api_key: str | list[str], requests_per_minute: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        keys = [api_key] if isinstance(api_key, str) else list(api_key)
        # the provider's requests-per-minute limit (15 on the free tier) is per key and shared by every thread; 0 = no pacing (a paid key). Paced calls do not hit a 429 that costs a retry wait.
        self._slots = [_KeySlot(k, RateLimiter(requests_per_minute) if requests_per_minute > 0 else None) for k in keys]
        self._turn = 0  # calls started so far: the keys take turns

    @property
    def _limiter(self) -> RateLimiter | None:
        return self._slots[0].limiter

    @_limiter.setter
    def _limiter(self, value: RateLimiter | None) -> None:
        self._slots[0].limiter = value

    @property
    def key_count(self) -> int:
        return len(self._slots)

    def key_report(self) -> str:
        return "; ".join(f"key {i}: {s.calls} calls answered, {s.failed} attempts failed" + (f", left out ({s.out[:90]})" if s.out else "") for i, s in enumerate(self._slots, start=1))

    def _order(self) -> list[_KeySlot]:
        """The keys still in use, the one that can start a call soonest first; keys that can start at once take turns."""
        with self._lock:
            turn = self._turn
            self._turn += 1
        count = len(self._slots)
        live = [(i, s) for i, s in enumerate(self._slots) if not s.out]
        live.sort(key=lambda p: (round(p[1].limiter.wait_for(), 1) if p[1].limiter is not None else 0.0, (p[0] - turn) % count))
        return [s for _i, s in live]

    def native_schema(self, json_schema: dict) -> dict | None:
        return gemini_schema(json_schema)  # sent as generationConfig.responseSchema instead of being pasted into every prompt

    def _post_with_keys(self, body: dict) -> dict:
        """One generateContent call: on the first key that can start it, and on the next key when that fails (a quota, a busy service, a timeout, a key Google
        refuses). A key whose daily quota is used up, or that Google refuses, is left out for the rest of the run. When every key failed, the error of the last
        busy one is raised (the client's own retries then start again over the keys still in use), or a quota error when every key is out."""
        live = self._order()
        if not live:
            raise LLMQuotaError("every API key is out for the rest of the run: " + "; ".join(f"key {i}: {s.out[:120]}" for i, s in enumerate(self._slots, start=1)))
        busy: LLMTransientError | None = None
        last: LLMError | None = None
        for slot in live:
            if slot.limiter is not None:
                slot.limiter.acquire()
            try:
                data = _post(f"{BASE_URL}/models/{self.model}:generateContent", slot.key, body)
            except LLMQuotaError as exc:
                slot.out, slot.failed, last = str(exc)[:300], slot.failed + 1, exc
                continue
            except LLMTransientError as exc:
                slot.failed += 1
                if slot.limiter is not None and exc.status == 429 and exc.retry_after:
                    slot.limiter.pause(exc.retry_after)  # the provider asked this key for a wait: every thread backs off it, not only this one
                last = exc
                if busy is None or (exc.retry_after or 0.0) < (busy.retry_after or 0.0):
                    busy = exc  # of the keys that were busy, the one that asks for the shortest wait sets the client's backoff
                continue
            except LLMPermanentError as exc:
                if not _key_refused(exc):
                    raise  # a bad request fails the same way on every key
                slot.out, slot.failed, last = str(exc)[:300], slot.failed + 1, exc
                continue
            slot.calls += 1
            return data
        if busy is not None:
            raise busy
        if all(s.out for s in self._slots):
            raise LLMQuotaError("every API key is out for the rest of the run: " + "; ".join(f"key {i}: {s.out[:120]}" for i, s in enumerate(self._slots, start=1)))
        assert last is not None
        raise last

    def _raw_generate(self, prompt: str, system: str | None, json_mode: bool) -> RawResponse:
        body: dict = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        generation_config: dict = {"temperature": 0}
        if json_mode:
            generation_config["responseMimeType"] = "application/json"
            response_schema = getattr(self._tls, "response_schema", None)
            if response_schema:
                generation_config["responseSchema"] = response_schema
            max_output = getattr(self._tls, "max_output_tokens", None)
            if max_output:
                generation_config["maxOutputTokens"] = int(max_output)
        body["generationConfig"] = generation_config
        data = self._post_with_keys(body)
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
            cached_tokens=meta.get("cachedContentTokenCount", 0),
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
