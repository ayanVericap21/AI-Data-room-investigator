"""GeminiLLM + provider selection, with the google-genai client faked (no network, no credentials)."""
import json
from types import SimpleNamespace

import pytest
from google import genai
from google.genai import errors

from app.agent.schemas import StepOut, SupportOut
from app.config import Settings, get_settings
from app.llm import AnthropicLLM, Block, GeminiLLM, LLMError, create_llm

# Any non-empty value satisfies the "is configured" check; the client is faked and nothing is sent anywhere.
PLACEHOLDER_NOT_A_CREDENTIAL = "-".join(["placeholder", "for", "tests", "only"])
MODEL = "gemini-test-model"


def _resp(obj=None, text=None, finish="STOP", prompt=1000, cached=0, cand=50, thoughts=0):
    return SimpleNamespace(
        text=text if text is not None else (json.dumps(obj) if obj is not None else None),
        candidates=[SimpleNamespace(finish_reason=finish)],
        usage_metadata=SimpleNamespace(prompt_token_count=prompt, cached_content_token_count=cached,
                                       candidates_token_count=cand, thoughts_token_count=thoughts))


class FakeModels:
    def __init__(self, results):
        self.results, self.requests = list(results), []

    def generate_content(self, **kw):
        self.requests.append(kw)
        r = self.results.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def make_llm(monkeypatch, results, **kw):
    models = FakeModels(results)
    monkeypatch.setattr(genai, "Client", lambda **k: SimpleNamespace(models=models))
    llm = GeminiLLM(Settings(llm_provider="gemini", model=MODEL, gemini_api_key=PLACEHOLDER_NOT_A_CREDENTIAL, **kw))
    llm._sleep = lambda s: None  # no real backoff in tests
    return llm, models


def test_request_shape_and_prefix_concatenation(monkeypatch):
    llm, models = make_llm(monkeypatch, [_resp({"supported": True, "issues": []})])
    out = llm.structured(system="SYS", user="suffix", schema=SupportOut, purpose="support_check",
                         prefix=[Block(text="overview", cache=True), Block(text="  "), Block(text="evidence", cache=True)])
    req = models.requests[0]
    assert out == SupportOut(supported=True, issues=[])
    assert req["model"] == MODEL and req["contents"] == "overview\n\nevidence\n\nsuffix"  # stable prefix first, volatile last
    cfg = req["config"]
    assert cfg.system_instruction == "SYS" and cfg.response_mime_type == "application/json"
    assert cfg.response_json_schema == SupportOut.model_json_schema()


def test_usage_maps_cached_and_thinking_tokens(monkeypatch):
    llm, _ = make_llm(monkeypatch, [_resp({"supported": True}, prompt=1000, cached=800, cand=50, thoughts=30)])
    llm.structured(system="s", user="q", schema=SupportOut, purpose="p")
    u = llm.usage
    assert (u.input_tokens, u.cache_read_tokens, u.cache_write_tokens, u.output_tokens) == (200, 800, 0, 80)
    assert u.cache_hit_ratio == pytest.approx(0.8)


def test_invalid_json_is_retried_then_succeeds(monkeypatch):
    llm, models = make_llm(monkeypatch, [_resp(text="not json at all"), _resp({"supported": False, "issues": ["x"]})])
    assert llm.structured(system="s", user="q", schema=SupportOut, purpose="p").supported is False
    assert len(models.requests) == 2 and "invalid" in models.requests[1]["contents"]


def test_gives_up_after_retries(monkeypatch):
    llm, _ = make_llm(monkeypatch, [_resp(text="{}")] * 3)
    with pytest.raises(LLMError, match="failed validation"):
        llm.structured(system="s", user="q", schema=SupportOut, purpose="p")


@pytest.mark.parametrize("finish,match", [("MAX_TOKENS", "token limit"), ("SAFETY", "blocked"), ("PROHIBITED_CONTENT", "blocked")])
def test_truncation_and_blocks_are_errors_without_retry(monkeypatch, finish, match):
    llm, models = make_llm(monkeypatch, [_resp(text="", finish=finish)])
    with pytest.raises(LLMError, match=match):
        llm.structured(system="s", user="q", schema=SupportOut, purpose="p")
    assert len(models.requests) == 1


def test_rate_limit_is_retried_with_backoff(monkeypatch):
    err = errors.ClientError(429, {"error": {"message": "quota"}}, None)
    llm, models = make_llm(monkeypatch, [err, err, _resp({"supported": True})])
    sleeps = []
    llm._sleep = sleeps.append
    assert llm.structured(system="s", user="q", schema=SupportOut, purpose="p").supported
    assert len(models.requests) == 3 and sleeps == [2, 4]


def test_non_transient_api_error_is_not_retried(monkeypatch):
    llm, models = make_llm(monkeypatch, [errors.ClientError(400, {"error": {"message": "bad schema"}}, None)])
    with pytest.raises(LLMError, match="API error"):
        llm.structured(system="s", user="q", schema=SupportOut, purpose="p")
    assert len(models.requests) == 1


def test_daily_quota_fails_fast_with_a_helpful_message(monkeypatch):
    err = errors.ClientError(429, {"error": {"message": "Quota exceeded for metric: generate_content_free_tier_requests, "
                                                        "limit: 20, model: m, description: Generate content requests per day"}}, None)
    llm, models = make_llm(monkeypatch, [err, _resp({"supported": True})])
    sleeps = []
    llm._sleep = sleeps.append
    with pytest.raises(LLMError, match="daily quota exhausted.*DRI_MODEL"):
        llm.structured(system="s", user="q", schema=SupportOut, purpose="p")
    assert len(models.requests) == 1 and sleeps == []      # no pointless retries


def test_persistent_overload_gives_up(monkeypatch):
    llm, models = make_llm(monkeypatch, [errors.ServerError(503, {"error": {"message": "overloaded"}}, None)] * 4)
    with pytest.raises(LLMError, match="API error"):
        llm.structured(system="s", user="q", schema=SupportOut, purpose="p")
    assert len(models.requests) == 4


def test_real_agent_schema_round_trips(monkeypatch):
    payload = {"belief": "contradiction", "reasoning": "r", "action": "search", "query": "tonnes"}
    llm, _ = make_llm(monkeypatch, [_resp(payload)])
    step = llm.structured(system="s", user="q", schema=StepOut, purpose="investigate_step")
    assert step.action == "search" and step.belief.value == "contradiction" and step.conclusion is None


# ---------- configuration / provider selection ----------
def test_missing_key_and_model_are_clean_errors():
    with pytest.raises(LLMError, match="GEMINI_API_KEY"):
        GeminiLLM(Settings(llm_provider="gemini", model=MODEL))
    with pytest.raises(LLMError, match="DRI_MODEL"):
        GeminiLLM(Settings(llm_provider="gemini", model="", gemini_api_key=PLACEHOLDER_NOT_A_CREDENTIAL))
    with pytest.raises(LLMError, match="DRI_MODEL"):  # a Claude model name is not valid for Gemini
        GeminiLLM(Settings(llm_provider="gemini", gemini_api_key=PLACEHOLDER_NOT_A_CREDENTIAL))


def test_create_llm_dispatch(monkeypatch):
    make_llm(monkeypatch, [])  # patches genai.Client
    assert isinstance(create_llm(Settings(llm_provider="gemini", model=MODEL, gemini_api_key=PLACEHOLDER_NOT_A_CREDENTIAL)), GeminiLLM)
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        create_llm(Settings())  # default provider is anthropic
    with pytest.raises(LLMError, match="unknown"):
        create_llm(Settings(llm_provider="nope"))


def test_get_settings_gemini_env(monkeypatch):
    for k in ["DRI_MODEL", "DRI_PRICE_IN", "DRI_PRICE_OUT", "GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY"]:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("DRI_LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GOOGLE_API_KEY", PLACEHOLDER_NOT_A_CREDENTIAL)  # fallback env name
    s = get_settings()
    assert s.llm_provider == "gemini" and s.gemini_api_key == PLACEHOLDER_NOT_A_CREDENTIAL
    assert s.model == "" and (s.price_in_per_mtok, s.price_out_per_mtok) == (0.0, 0.0)   # no stale model, unpriced
    monkeypatch.setenv("DRI_MODEL", MODEL)
    monkeypatch.setenv("DRI_PRICE_IN", "0.3")
    s = get_settings()
    assert s.model == MODEL and s.price_in_per_mtok == 0.3 and s.price_out_per_mtok == 0.0
    monkeypatch.delenv("DRI_LLM_PROVIDER")
    monkeypatch.delenv("DRI_MODEL")
    monkeypatch.delenv("DRI_PRICE_IN")
    s = get_settings()
    assert s.model == "claude-sonnet-5-5" and s.price_in_per_mtok == 2.0  # anthropic defaults untouched
    assert AnthropicLLM  # still available
