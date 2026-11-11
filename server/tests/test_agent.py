"""Investigation agent loop: bounded, recoverable, and persisted incrementally."""
from __future__ import annotations

from sqlalchemy import select

from app.agent import InvestigationAgent
from app.agent import run_investigation as _run_investigation
from app.llm.errors import LLMPermanentError
from app.llm.mock import MockLLMClient
from app.models import Finding, Investigation, InvestigationEvent
from app.tools import ToolBudget, build_registry
from tests.test_tools import finding_args, room, registry  # noqa: F401  (fixtures)


def run_agent(*args, **kwargs):
    kwargs.setdefault("max_finish_refusals", 0)  # short scripted runs do not open every document
    return _run_investigation(*args, **kwargs)


def call(tool, arguments=None, why="checking"):
    return {"action": "call_tool", "tool": tool, "arguments": arguments or {}, "rationale": why}


def finish(summary="done"):
    return {"action": "finish", "rationale": "covered the focus", "summary": summary}


def test_full_run_lists_reads_submits_and_finishes(session, room):
    llm = MockLLMClient([
        call("list_documents"),
        call("get_chunk", {"chunk_id": room.deck_p1}),
        call("calculate", {"expression": "120000 - 100000"}),
        call("submit_finding", finding_args(room)),
        finish("Found one capacity discrepancy."),
    ])
    result = run_agent(session, llm, None, name="capacity review", focus="capacity", max_steps=10)

    assert result.termination == "finished" and result.summary == "Found one capacity discrepancy."
    assert [s.tool for s in result.steps] == ["list_documents", "get_chunk", "calculate", "submit_finding", None]
    assert all(s.ok for s in result.steps)
    assert len(result.finding_ids) == 1 and session.get(Finding, result.finding_ids[0]).verification_status == "verified"
    assert result.token_usage.calls == 5 and result.llm_latency.calls == 5 and result.tool_latency.calls == 4

    investigation = session.get(Investigation, result.investigation_id)
    assert investigation.status == "completed" and investigation.completed_at and investigation.duration_ms is not None
    assert investigation.total_tokens == result.token_usage.total_tokens
    assert investigation.execution_config["focus"] == "capacity"
    steps = session.scalars(select(InvestigationEvent).where(
        InvestigationEvent.investigation_id == investigation.id, InvestigationEvent.event_type == "agent_step")).all()
    assert len(steps) == 5  # one trace event per step, plus the finding_saved event elsewhere


def test_agent_recovers_from_a_bad_tool_call_and_a_rejected_finding(session, room):
    bad_finding = finding_args(room)
    bad_finding["evidence"][0]["quoted_text"] = "capacity is 999 tonnes per year"
    llm = MockLLMClient([
        call("not_a_tool"),
        call("submit_finding", bad_finding),
        call("submit_finding", finding_args(room)),
        finish(),
    ])
    result = run_agent(session, llm, None, name="recovery")
    assert [s.error_code for s in result.steps[:3]] == ["unknown_tool", "validation_failed", None]
    assert result.termination == "finished" and len(result.finding_ids) == 1
    assert len(session.scalars(select(Finding)).all()) == 1  # the rejected attempt left nothing behind
    # the error text reached the model on the next turn
    assert "not in chunk" in llm.prompts[2]


def test_step_cap_ends_the_run_as_partial(session, room):
    llm = MockLLMClient([call("list_documents")] * 3)
    result = run_agent(session, llm, None, name="capped", max_steps=3)
    assert result.termination == "max_steps" and len(result.steps) == 3
    assert session.get(Investigation, result.investigation_id).status == "partially_completed"


def test_tool_budget_ends_the_run(session, room):
    llm = MockLLMClient([call("list_documents")] * 5)
    result = run_agent(session, llm, None, name="budget", max_steps=10, budget=ToolBudget(max_tool_calls=2))
    assert result.termination == "tool_budget" and result.steps[-1].error_code == "budget_exhausted"
    assert len(llm.prompts) == 3


def test_stop_flag_is_checked_before_every_step(session, room):
    flags = iter([False, True])
    llm = MockLLMClient([call("list_documents"), call("list_documents")])
    result = run_agent(session, llm, None, name="stoppable", should_stop=lambda: next(flags))
    assert result.termination == "stopped" and len(result.steps) == 1
    investigation = session.get(Investigation, result.investigation_id)
    assert investigation.status == "stopped" and investigation.stopped_at is not None


def test_provider_failure_ends_cleanly_and_keeps_saved_findings(session, room):
    llm = MockLLMClient([call("submit_finding", finding_args(room)), LLMPermanentError("401 invalid key")])
    result = run_agent(session, llm, None, name="provider down", max_steps=5)
    assert result.termination == "llm_error" and "401" in result.error
    assert len(result.finding_ids) == 1  # saved before the failure, still there
    investigation = session.get(Investigation, result.investigation_id)
    assert investigation.status == "failed" and "401" in investigation.error_message


def test_invalid_action_output_is_retried_by_the_llm_layer(session, room):
    llm = MockLLMClient(["this is not json", call("list_documents"), finish()])
    result = run_agent(session, llm, None, name="garbled")
    assert result.termination == "finished" and len(result.steps) == 2
    assert llm.retry_count == 1


def test_a_call_action_without_a_tool_name_is_invalid_output(session, room):
    llm = MockLLMClient([{"action": "call_tool", "rationale": "forgot the tool"}, finish()])
    result = run_agent(session, llm, None, name="no tool")
    assert result.termination == "finished" and llm.retry_count == 1


def test_older_results_are_shortened_and_recent_ones_stay_full(session, room):
    llm = MockLLMClient([call("get_chunk", {"chunk_id": room.deck_p1}),
                         call("get_chunk", {"chunk_id": room.deck_p2}),
                         call("get_chunk", {"chunk_id": room.capex_current}),
                         call("get_chunk", {"chunk_id": room.rent_1}),
                         finish()])
    run_agent(session, llm, None, name="compaction", max_steps=6)
    final_prompt = llm.prompts[-1]
    assert "(shortened)" in final_prompt                       # step 1 shortened
    assert "Annual production capacity is 120,000" not in final_prompt  # its full text is gone
    assert "EBITDA margin for 2024" in final_prompt            # steps 2-4 still full
    assert "Annual rent is INR 1.2 crore" in final_prompt


def test_oversized_results_are_truncated_in_the_prompt(session, room):
    registry = build_registry(session, None, room.investigation.id)
    llm = MockLLMClient([call("list_documents"), finish()])
    InvestigationAgent(llm, registry, max_steps=3, max_result_chars=40, max_finish_refusals=0).run()
    assert "[truncated" in llm.prompts[-1]


def test_prompt_shows_the_tools_and_the_remaining_steps(session, room):
    llm = MockLLMClient([finish()])
    run_agent(session, llm, None, name="prompt check", focus="Check rent figures", max_steps=7)
    prompt = llm.prompts[0]
    for tool in ("list_documents", "get_outline", "search", "get_chunk", "read_section", "calculate", "submit_finding"):
        assert f"- {tool}:" in prompt
    assert "Check rent figures" in prompt and "7 steps left" in prompt


def test_unexpected_exception_marks_the_investigation_failed(session, room):
    class Broken(MockLLMClient):
        def generate_structured_output(self, *args, **kwargs):
            raise RuntimeError("bug")

    try:
        run_agent(session, Broken(), None, name="bug")
    except RuntimeError:
        pass
    investigation = session.scalars(select(Investigation).where(Investigation.name == "bug")).one()
    assert investigation.status == "failed" and "RuntimeError" in investigation.error_message


def test_every_step_records_its_llm_attempts_context_size_and_tool_call(session, room):
    llm = MockLLMClient([call("list_documents"), call("get_chunk", {"chunk_id": room.deck_p1}), finish()])
    result = run_agent(session, llm, None, name="metrics", max_steps=5)
    assert all(len(s.llm_attempts) == 1 and s.llm_attempts[0].prompt_chars > 0 for s in result.steps)
    assert result.steps[1].llm_attempts[0].prompt_chars > result.steps[0].llm_attempts[0].prompt_chars  # history grows
    assert [s.tool_call_index for s in result.steps[:2]] == [1, 2] and result.steps[2].tool_call_index is None
    assert len(result.llm_calls) == 3 and [c.tool for c in result.tool_calls] == ["list_documents", "get_chunk"]
    assert set(result.tool_latency_by_tool) == {"list_documents", "get_chunk"} and result.llm_calls[0].started_at > 0


def test_a_failed_turn_is_recorded_with_its_failed_attempts(session, room):
    llm = MockLLMClient([LLMPermanentError("401 invalid key")])
    result = run_agent(session, llm, None, name="failed turn", max_steps=3)
    step = result.steps[-1]
    assert step.action == "llm_error" and not step.ok and step.llm_attempts and not step.llm_attempts[0].ok


# ---- fix: "finish" sent as a tool is treated as finishing ----------------------
def test_finish_sent_as_a_tool_name_means_finish(session, room):
    llm = MockLLMClient([{"action": "call_tool", "tool": "finish", "arguments": {"summary": "all checked"}, "rationale": "done"}])
    result = run_agent(session, llm, None, name="finish as tool")
    assert result.termination == "finished" and result.summary == "all checked"
    assert [s.error_code for s in result.steps] == [None] and result.steps[0].action == "finish"  # no unknown_tool failure


# ---- fix: the agent must open every document before it may finish --------------
def test_finishing_is_refused_until_every_document_is_opened(session, room):
    llm = MockLLMClient([finish(), call("get_outline", {"document_id": room.deck.id}),
                         call("get_outline", {"document_id": room.capex.id}),
                         call("read_section", {"document_id": room.lease.id, "heading": "Rent"}), finish("covered all")])
    result = _run_investigation(session, llm, None, name="coverage gate", max_steps=10)
    assert [s.action for s in result.steps] == ["finish_refused", "call_tool", "call_tool", "call_tool", "finish"]
    assert result.termination == "finished" and result.finish_refusals == 1
    assert (result.documents_total, result.documents_opened, result.documents_not_opened) == (3, 3, [])
    assert "You cannot finish yet: 3 of 3 documents have not been opened" in llm.prompts[1]  # the model is told why
    assert "capex.xlsx" in llm.prompts[1] and "lease.docx" in llm.prompts[1]


def test_refusing_to_finish_is_bounded_so_a_stubborn_model_still_ends(session, room):
    llm = MockLLMClient([finish(), finish(), finish("giving up on the rest")])
    result = _run_investigation(session, llm, None, name="bounded gate", max_steps=10)
    assert result.termination == "finished" and result.finish_refusals == 2 and result.summary == "giving up on the rest"
    assert sorted(result.documents_not_opened) == ["capex.xlsx", "deck.pdf", "lease.docx"]  # the report can show what was skipped


def test_every_prompt_shows_which_documents_are_still_unopened(session, room):
    llm = MockLLMClient([call("get_outline", {"document_id": room.deck.id}), finish()])
    run_agent(session, llm, None, name="coverage line", max_steps=5)
    assert "Documents opened: 0 of 3. Not yet opened: capex.xlsx, deck.pdf, lease.docx." in llm.prompts[0]
    assert "Documents opened: 1 of 3. Not yet opened: capex.xlsx, lease.docx." in llm.prompts[1]


def test_the_prompt_tells_the_model_finish_is_an_action_and_documents_must_be_covered():
    from app.agent.prompts import SYSTEM

    assert "finish is an action, not a tool" in SYSTEM and "Cover every document" in SYSTEM


# ---- no limits by default: a run lasts as long as the data room needs -------------
def test_with_no_step_limit_the_run_continues_until_the_model_finishes(session, room):
    llm = MockLLMClient([call("list_documents")] * 70 + [finish("covered everything")])
    result = run_agent(session, llm, None, name="unlimited", max_steps=None)
    assert result.termination == "finished" and len(result.steps) == 71 and result.summary == "covered everything"
    assert "There is no step limit." in llm.prompts[0] and "steps left" not in llm.prompts[0]


def test_tool_and_finding_limits_default_to_none_so_nothing_cuts_a_large_data_room_short(session, room):
    budget = ToolBudget()
    assert budget.max_tool_calls is None and budget.max_findings is None
    registry = build_registry(session, None, room.investigation.id, budget)
    assert all(registry.execute("list_documents").ok for _ in range(100))  # well past the old limit of 60
    for i in range(25):  # and past the old limit of 20 findings
        assert registry.execute("submit_finding", finding_args(room, title=f"finding {i}")).ok


def test_explicit_limits_still_work_when_asked_for(session, room):
    llm = MockLLMClient([call("list_documents")] * 5)
    result = run_agent(session, llm, None, name="limited", max_steps=2)
    assert result.termination == "max_steps" and len(result.steps) == 2
    assert "You have 2 steps left including this one." in llm.prompts[0]
