"""Thin LLM layer: structured output validated by Pydantic, prompt caching, retries, usage + cost accounting."""
from __future__ import annotations

import threading
import time
from typing import Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from .config import Settings

T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    """The LLM call failed after retries (API error, refusal, or output that never validated)."""


class Block(BaseModel):
    """A piece of the user turn. `cache=True` puts a prompt-cache breakpoint at the END of this block, so everything
    up to and including it (system prompt + earlier blocks) can be reused by later requests that start identically."""

    text: str
    cache: bool = False


class Usage(BaseModel):
    calls: int = 0
    input_tokens: int = 0          # uncached input only
    output_tokens: int = 0
    cache_write_tokens: int = 0    # billed at cache_write_multiplier x input price
    cache_read_tokens: int = 0     # billed at cache_read_multiplier x input price
    latency_s: float = 0.0
    by_purpose: dict[str, int] = {}

    def cost_usd(self, s: Settings) -> float:
        per_in = s.price_in_per_mtok / 1e6
        return round(self.input_tokens * per_in
                     + self.cache_write_tokens * per_in * s.cache_write_multiplier
                     + self.cache_read_tokens * per_in * s.cache_read_multiplier
                     + self.output_tokens * s.price_out_per_mtok / 1e6, 4)

    @property
    def cache_hit_ratio(self) -> float | None:
        """Share of all input tokens that were served from cache."""
        total = self.input_tokens + self.cache_write_tokens + self.cache_read_tokens
        return round(self.cache_read_tokens / total, 3) if total else None


class LLM(Protocol):
    usage: Usage

    def structured(self, *, system: str, user: str, schema: type[T], purpose: str,
                   prefix: list[Block] | None = None) -> T: ...


class UsageTracker:
    def __init__(self):
        self.usage = Usage()
        self._lock = threading.Lock()

    def record(self, purpose: str, in_tok: int, out_tok: int, latency: float, cache_write: int = 0, cache_read: int = 0):
        with self._lock:
            u = self.usage
            u.calls += 1
            u.input_tokens += in_tok
            u.output_tokens += out_tok
            u.cache_write_tokens += cache_write
            u.cache_read_tokens += cache_read
            u.latency_s = round(u.latency_s + latency, 3)
            u.by_purpose[purpose] = u.by_purpose.get(purpose, 0) + 1


def build_content(user: str, prefix: list[Block] | None, retry_note: str = "") -> str | list[dict]:
    """Plain string when there is no prefix; otherwise text blocks: [prefix blocks..., user suffix].
    Volatile content belongs in `user` (after the last breakpoint) so it never invalidates the cached prefix."""
    suffix = user + (f"\n\n{retry_note}" if retry_note else "")
    if not prefix:
        return suffix
    blocks = [{"type": "text", "text": b.text, **({"cache_control": {"type": "ephemeral"}} if b.cache else {})}
              for b in prefix if b.text.strip()]
    blocks.append({"type": "text", "text": suffix})
    return blocks


class GeminiLLM(UsageTracker):
    """Google Gemini via `google-genai`. Structured output = JSON mode with the Pydantic JSON schema, validated by us
    (same validate-and-retry path as the Anthropic adapter). Caching: Gemini caches repeated prompt prefixes
    implicitly - the prompts are already laid out stable-first - so `Block.cache` markers are only concatenated;
    cached tokens are read from the usage metadata and reported as cache reads."""

    _RETRY_CODES = {429, 500, 502, 503, 504}

    def __init__(self, settings: Settings):
        super().__init__()
        if not settings.gemini_api_key:
            raise LLMError("GEMINI_API_KEY is not set")
        if not settings.model or settings.model.startswith("claude"):
            raise LLMError("DRI_MODEL must name a Gemini model when DRI_LLM_PROVIDER=gemini "
                           "(see https://ai.google.dev/gemini-api/docs/models)")
        from google import genai

        self._genai = genai
        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._model = settings.model
        self._retries = settings.llm_retries
        self._sleep = time.sleep

    def structured(self, *, system: str, user: str, schema: type[T], purpose: str,
                   prefix: list[Block] | None = None) -> T:
        from google.genai import errors, types

        cfg = types.GenerateContentConfig(system_instruction=system, response_mime_type="application/json",
                                          response_json_schema=schema.model_json_schema(), max_output_tokens=32768)
        parts = [b.text for b in (prefix or []) if b.text.strip()]
        note, last_err = "", ""
        for _ in range(self._retries + 1):
            contents = "\n\n".join([*parts, user + (f"\n\n{note}" if note else "")])
            t0, resp = time.time(), None
            for attempt in range(4):  # transient errors (rate limit / overload): back off and retry
                try:
                    resp = self._client.models.generate_content(model=self._model, contents=contents, config=cfg)
                    break
                except errors.APIError as e:
                    msg = str(e).lower()
                    if getattr(e, "code", None) == 429 and ("per day" in msg or "perday" in msg):
                        # A daily quota does not clear in seconds: retrying only wastes time. Say how to fix it.
                        raise LLMError(f"{purpose}: daily quota exhausted for model '{self._model}'. Wait for the daily reset, "
                                       "set DRI_MODEL to another model (quotas are per model), or enable billing.") from e
                    if getattr(e, "code", None) in self._RETRY_CODES and attempt < 3:
                        self._sleep(min(2 ** attempt * 2, 20))
                        continue
                    raise LLMError(f"{purpose}: API error: {e}") from e
            um = resp.usage_metadata
            cached = (getattr(um, "cached_content_token_count", 0) or 0) if um else 0
            prompt = (getattr(um, "prompt_token_count", 0) or 0) if um else 0
            out = ((getattr(um, "candidates_token_count", 0) or 0) + (getattr(um, "thoughts_token_count", 0) or 0)) if um else 0
            self.record(purpose, max(prompt - cached, 0), out, time.time() - t0, 0, cached)
            fr = resp.candidates[0].finish_reason if getattr(resp, "candidates", None) else None
            fr = str(getattr(fr, "name", fr or "STOP")).split(".")[-1]
            if fr == "MAX_TOKENS":
                raise LLMError(f"{purpose}: output hit the token limit before completing")
            if fr not in {"STOP", "FINISH_REASON_UNSPECIFIED"}:
                raise LLMError(f"{purpose}: the model declined or was blocked (finish reason {fr})")
            try:
                return schema.model_validate_json(resp.text or "")
            except ValidationError as e:
                last_err = str(e)[:800]
                note = f"Your previous answer was invalid ({last_err}). Return a valid answer."
        raise LLMError(f"{purpose}: output failed validation after retries: {last_err}")


def create_llm(settings: Settings) -> "LLM":
    """Build the configured LLM client. Raises LLMError when it is not usable (missing key / model)."""
    if settings.llm_provider == "gemini":
        return GeminiLLM(settings)
    if settings.llm_provider == "anthropic":
        return AnthropicLLM(settings)
    raise LLMError(f"unknown DRI_LLM_PROVIDER '{settings.llm_provider}' (use anthropic or gemini)")


class AnthropicLLM(UsageTracker):
    """Structured output via `messages.parse` (output_config.format). Forced tool_choice is not used: it is rejected
    (HTTP 400) by the current Claude models."""

    def __init__(self, settings: Settings):
        super().__init__()
        if not settings.anthropic_api_key:
            raise LLMError("ANTHROPIC_API_KEY is not set")
        import anthropic

        self._anthropic = anthropic
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key, max_retries=3)
        self._model = settings.model
        self._retries = settings.llm_retries

    def structured(self, *, system: str, user: str, schema: type[T], purpose: str,
                   prefix: list[Block] | None = None) -> T:
        note, last_err = "", ""
        for _ in range(self._retries + 1):
            t0 = time.time()
            try:
                resp = self._client.messages.parse(
                    model=self._model, max_tokens=16000, system=system, output_format=schema,
                    messages=[{"role": "user", "content": build_content(user, prefix, note)}])
            except self._anthropic.APIError as e:
                raise LLMError(f"{purpose}: API error: {e}") from e
            except ValidationError as e:
                last_err = str(e)[:800]
                note = f"Your previous answer was invalid ({last_err}). Return a valid answer."
                continue
            u = resp.usage
            self.record(purpose, u.input_tokens, u.output_tokens, time.time() - t0,
                        getattr(u, "cache_creation_input_tokens", 0) or 0, getattr(u, "cache_read_input_tokens", 0) or 0)
            if resp.stop_reason == "refusal":
                raise LLMError(f"{purpose}: the model declined this request (refusal)")
            if resp.stop_reason == "max_tokens":
                raise LLMError(f"{purpose}: output hit max_tokens before completing")
            if resp.parsed_output is not None:
                return resp.parsed_output
            last_err = "no structured output returned"
            note = f"Your previous answer was invalid ({last_err}). Return a valid answer."
        raise LLMError(f"{purpose}: output failed validation after retries: {last_err}")
