"""AnthropicLLM request shape, prompt-cache accounting and cost - with the SDK client faked (no network, no credentials)."""
from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from app.agent import store
from app.agent.schemas import CitationOut, SupportOut
from app.config import Settings
from app.llm import AnthropicLLM, Block, LLMError, Usage, build_content
from app.models import Classification as C, DocSummary
from tests.test_agent import base_script, conclusion, room, run, step  # noqa: F401

# Any non-empty value satisfies AnthropicLLM's "is configured" check; the SDK client is faked, nothing is sent anywhere.
PLACEHOLDER_NOT_A_CREDENTIAL = "-".join(["placeholder", "for", "tests", "only"])


def _resp(parsed, stop="end_turn", inp=100, out=20, write=0, read=0):
    return SimpleNamespace(parsed_output=parsed, stop_reason=stop,
                           usage=SimpleNamespace(input_tokens=inp, output_tokens=out,
                                                 cache_creation_input_tokens=write, cache_read_input_tokens=read))


class FakeClient:
    def __init__(self, results):
        self.results, self.requests = list(results), []
        self.messages = self

    def parse(self, **kw):
        self.requests.append(kw)
        r = self.results.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def make_llm(monkeypatch, results, **settings_kw):
    fake = FakeClient(results)
    monkeypatch.setattr(anthropic, "Anthropic", lambda **kw: fake)
    return AnthropicLLM(Settings(anthropic_api_key=PLACEHOLDER_NOT_A_CREDENTIAL, **settings_kw)), fake


# ---------- content blocks ----------
def test_build_content_plain_and_blocks():
    assert build_content("hello", None) == "hello"
    blocks = build_content("suffix", [Block(text="stable", cache=True), Block(text="  "), Block(text="mid")], "RETRY")
    assert blocks[0] == {"type": "text", "text": "stable", "cache_control": {"type": "ephemeral"}}
    assert blocks[1] == {"type": "text", "text": "mid"}                       # blank block dropped, no marker
    assert blocks[-1] == {"type": "text", "text": "suffix\n\nRETRY"}           # volatile suffix is last and uncached
    assert sum("cache_control" in b for b in blocks) == 1


# ---------- request shape ----------
def test_structured_uses_parse_not_forced_tool_choice(monkeypatch):
    llm, fake = make_llm(monkeypatch, [_resp(SupportOut(supported=True))])
    out = llm.structured(system="sys", user="q", schema=SupportOut, purpose="support_check")
    req = fake.requests[0]
    assert out.supported is True
    assert req["output_format"] is SupportOut and "tool_choice" not in req and "tools" not in req
    assert req["model"] == "claude-sonnet-5-5" and req["system"] == "sys"
    assert req["messages"] == [{"role": "user", "content": "q"}]
    assert "temperature" not in req and "thinking" not in req and "budget_tokens" not in str(req)


def test_prefix_blocks_carry_cache_control(monkeypatch):
    llm, fake = make_llm(monkeypatch, [_resp(SupportOut(supported=True))])
    llm.structured(system="s", user="Steps used: 1/5.", schema=SupportOut, purpose="p",
                   prefix=[Block(text="overview", cache=True), Block(text="evidence", cache=True)])
    content = fake.requests[0]["messages"][0]["content"]
    assert [("cache_control" in b) for b in content] == [True, True, False]
    assert content[-1]["text"] == "Steps used: 1/5."


# ---------- usage + cost ----------
def test_usage_and_cost_with_cache_tokens(monkeypatch):
    llm, _ = make_llm(monkeypatch, [_resp(SupportOut(supported=True), inp=1_000_000, out=1_000_000, write=1_000_000, read=1_000_000)])
    llm.structured(system="s", user="q", schema=SupportOut, purpose="p")
    u = llm.usage
    assert (u.calls, u.cache_write_tokens, u.cache_read_tokens) == (1, 1_000_000, 1_000_000)
    s = Settings()  # $2 in / $10 out, write 1.25x, read 0.1x
    assert u.cost_usd(s) == pytest.approx(2.0 + 2.5 + 0.2 + 10.0)
    assert u.cache_hit_ratio == pytest.approx(1 / 3, abs=1e-3)
    assert Usage().cache_hit_ratio is None


def test_cache_read_discount_is_configurable():
    u = Usage(cache_read_tokens=1_000_000)
    assert u.cost_usd(Settings(cache_read_multiplier=0.05)) == pytest.approx(0.1)


# ---------- errors / retries ----------
def test_retries_when_no_parsed_output_then_succeeds(monkeypatch):
    llm, fake = make_llm(monkeypatch, [_resp(None), _resp(SupportOut(supported=True))])
    assert llm.structured(system="s", user="q", schema=SupportOut, purpose="p").supported
    assert len(fake.requests) == 2 and "invalid" in fake.requests[1]["messages"][0]["content"]
    assert llm.usage.calls == 2


def test_gives_up_after_retries(monkeypatch):
    llm, _ = make_llm(monkeypatch, [_resp(None)] * 3)
    with pytest.raises(LLMError, match="failed validation"):
        llm.structured(system="s", user="q", schema=SupportOut, purpose="p")


def test_refusal_and_truncation_are_errors_without_retry(monkeypatch):
    llm, fake = make_llm(monkeypatch, [_resp(None, stop="refusal")])
    with pytest.raises(LLMError, match="refusal"):
        llm.structured(system="s", user="q", schema=SupportOut, purpose="p")
    assert len(fake.requests) == 1
    llm, _ = make_llm(monkeypatch, [_resp(None, stop="max_tokens")])
    with pytest.raises(LLMError, match="max_tokens"):
        llm.structured(system="s", user="q", schema=SupportOut, purpose="p")


def test_api_error_becomes_llm_error(monkeypatch):
    err = anthropic.APIError("boom", httpx2.Request("POST", "https://example.invalid"), body=None)
    llm, _ = make_llm(monkeypatch, [err])
    with pytest.raises(LLMError, match="API error"):
        llm.structured(system="s", user="q", schema=SupportOut, purpose="p")


def test_missing_key_is_a_clean_error():
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        AnthropicLLM(Settings())


# ---------- caching layout inside the agent ----------
def _overview_docs(db):
    db.execute("UPDATE documents SET summary=?, summary_status='done'",
               (DocSummary(overview="Supplier quotation.", key_topics=["guarantee"], key_figures=["summary-only wording about guaranteed output"]).model_dump_json(),))


def test_agent_prompt_layout_for_caching(db, settings, room):
    _overview_docs(db)
    steps = [step("search", query="tonnes"), step("conclude", concl=conclusion(db))]
    inv_id, llm = run(db, settings, room, base_script(db, steps))
    inv_calls = [c for c in llm.calls if c[0] == "investigate_step"]
    assert len(inv_calls) == 2
    for _, prefix, suffix in inv_calls:
        assert prefix[0].cache and prefix[0].text.startswith("Data room overview")   # shared across all hypotheses/steps
        assert "NOT evidence" in prefix[0].text and "Supplier quotation." in prefix[0].text
        assert prefix[-1].cache and sum(b.cache for b in prefix) == 2 <= 4         # overview + growing evidence prefix
        assert suffix.startswith("Steps used:")                                      # only the volatile bit is uncached
    first, second = inv_calls[0][1], inv_calls[1][1]
    assert [b.text for b in second[: len(first)]][:-1] == [b.text for b in first][:-1]  # append-only: byte-identical prefix
    assert len(second) > len(first)
    assert "Supplier quotation." in next(u for p, u in llm.prompts if p == "hypothesize")  # summaries inform hypotheses
    assert store.get(db, inv_id).usage["cache_hit_ratio"] == 0.0  # recorded: fake LLM reports no cache tokens


def test_agent_without_summaries_has_no_overview_block(db, settings, room):
    inv_id, llm = run(db, settings, room, base_script(db, [step("conclude", concl=conclusion(db))]))
    _, prefix, _ = next(c for c in llm.calls if c[0] == "investigate_step")
    assert not any("Data room overview" in b.text for b in prefix) and prefix[-1].cache
    assert store.get(db, inv_id).status == "completed"


def test_citation_copied_from_summary_fails_verification(db, settings, room):
    """Summaries are orientation only: a quote that exists only in a summary must not verify."""
    _overview_docs(db)
    bad = conclusion(db, cites=[CitationOut(chunk_id=1, quote="Supplier quotation."),
                                CitationOut(chunk_id=2, quote="summary-only wording about guaranteed output")])
    inv_id, _ = run(db, settings, room, base_script(db, [step("conclude", concl=bad)]))
    f = store.findings(db, inv_id)[0]
    assert f.classification == C.INSUFFICIENT_EVIDENCE and not any(c.verified for c in f.citations)
