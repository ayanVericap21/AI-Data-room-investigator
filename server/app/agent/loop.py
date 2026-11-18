"""Layer 3: a bounded investigation agent loop.

Each turn the model returns one validated action (call a tool, or finish). The loop speaks
a plain JSON protocol through ``generate_structured_output``, so it works with any provider
and needs no native function-calling. Code owns every limit: step cap, tool budget, stop
flag. A provider failure ends the run cleanly with whatever was already saved. The loop is a
plain callable, so a LangGraph node can wrap it later without changes.
"""
from __future__ import annotations

import itertools
import json
import logging
import time
from dataclasses import dataclass
from typing import Callable

from app.agent.prompts import SYSTEM, build_prompt
from app.agent.schemas import AgentAction, AgentRunResult, AgentStepRecord, Termination
from app.llm.base import LLMClient
from app.llm.errors import LLMError
from app.tools.base import ToolErrorInfo, ToolResult
from app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class _Turn:
    step: int
    action: AgentAction
    result: ToolResult


class InvestigationAgent:
    def __init__(self, llm: LLMClient, registry: ToolRegistry, max_steps: int | None = 15, full_results: int = 3,
                 max_result_chars: int = 12_000, should_stop: Callable[[], bool] | None = None,
                 on_step: Callable[[AgentStepRecord], None] | None = None, max_finish_refusals: int = 2):
        self.llm = llm
        self.registry = registry
        self.max_steps = max_steps  # None = no step limit; the run ends when the model finishes or is stopped
        self.full_results = full_results  # how many of the latest tool results are shown in full
        self.max_result_chars = max_result_chars
        self.should_stop = should_stop
        self.on_step = on_step
        self.max_finish_refusals = max_finish_refusals  # times finishing may be refused while documents are unopened

    def _history(self, turns: list[_Turn]) -> tuple[list[str], dict]:
        """Older results are shortened to their one-line summary so the prompt stays bounded; long recent ones are cut. Returns the rendered
        history and what was cut (stored with every step, so the prompt the model saw can be reconstructed)."""
        rendered = []
        notes: dict = {"shortened_steps": [], "truncated": []}
        for i, turn in enumerate(turns):
            recent = i >= len(turns) - self.full_results
            args = json.dumps(turn.action.arguments, separators=(",", ":"), default=str)
            if recent:
                body = turn.result.to_text(self.max_result_chars)
                full_length = len(turn.result.to_text())
                if full_length > self.max_result_chars:
                    notes["truncated"].append({"step": turn.step, "chars_cut": full_length - self.max_result_chars})
            else:
                body = f"(shortened) {turn.result.summary}"
                notes["shortened_steps"].append(turn.step)
            rendered.append(f"Step {turn.step}: you called {turn.action.tool or turn.action.action} {args}\n"
                            f"Why: {turn.action.rationale}\nResult: {body}")
        return rendered, notes

    def run(self, focus: str | None = None) -> AgentRunResult:
        started = time.perf_counter()
        tools = self.registry.describe()
        turns: list[_Turn] = []
        steps: list[AgentStepRecord] = []
        termination: Termination = "max_steps"
        summary: str | None = None
        error: str | None = None
        refusals = 0

        def emit(record: AgentStepRecord) -> None:
            steps.append(record)
            if self.on_step:
                self.on_step(record)

        for n in itertools.count(1):
            if self.max_steps is not None and n > self.max_steps:
                break
            if self.should_stop and self.should_stop():
                termination = "stopped"
                break
            cov = self.registry.coverage()
            coverage_line = f"Documents opened: {cov.opened} of {cov.total}. Not yet opened: {', '.join(cov.not_opened) or 'none'}."
            if cov.unreadable:
                coverage_line += f" Not readable (no passages: processing failed or was skipped, so they cannot be opened): {', '.join(cov.unreadable)}."
            history, notes = self._history(turns)
            prompt = build_prompt(focus, tools, history, None if self.max_steps is None else self.max_steps - n + 1, coverage_line)
            attempts_before = len(self.llm.call_metrics)
            try:
                action = self.llm.generate_structured_output(prompt, AgentAction, system=SYSTEM)
            except LLMError as exc:
                termination, error = "llm_error", str(exc)[:300]
                logger.warning("Agent stopped on LLM error", extra={"step": n, "error": error})
                emit(AgentStepRecord(step=n, action="llm_error", ok=False, error_code="llm_error", result_summary=error, error_message=str(exc),
                                     llm_attempts=self.llm.call_metrics[attempts_before:], prompt=prompt, prompt_notes=notes))
                break
            attempts = self.llm.call_metrics[attempts_before:]
            action_json = action.model_dump(mode="json")

            if action.action == "finish" and cov.not_opened and refusals < self.max_finish_refusals:
                refusals += 1
                message = (f"You cannot finish yet: {len(cov.not_opened)} of {cov.total} documents have not been opened "
                           f"({', '.join(cov.not_opened)}). Open each with get_outline, then read what matters, and finish once "
                           "you have looked at all of them.")
                turns.append(_Turn(n, action, ToolResult(tool="finish", ok=False, summary=message[:200],
                                                         error=ToolErrorInfo(code="documents_not_opened", message=message))))
                emit(AgentStepRecord(step=n, action="finish_refused", ok=False, error_code="documents_not_opened", rationale=action.rationale,
                                     result_summary=f"{len(cov.not_opened)} documents unopened", error_message=message, llm_attempts=attempts,
                                     prompt=prompt, action_json=action_json, prompt_notes=notes))
                continue

            if action.action == "finish":
                emit(AgentStepRecord(step=n, action="finish", rationale=action.rationale, llm_attempts=attempts, prompt=prompt, action_json=action_json,
                                     prompt_notes=notes))
                summary, termination = action.summary, "finished"
                break

            result = self.registry.execute(action.tool or "", action.arguments)
            turns.append(_Turn(n, action, result))
            emit(AgentStepRecord(
                step=n, action="call_tool", tool=action.tool, arguments=action.arguments, rationale=action.rationale,
                ok=result.ok, error_code=result.error.code if result.error else None,
                result_summary=result.summary, tool_latency_ms=result.latency_ms,
                tool_call_index=self.registry.context.calls[-1].index, llm_attempts=attempts,
                error_message=result.error.message if result.error else None, prompt=prompt, action_json=action_json,
                output=result.output.model_dump(mode="json") if result.output is not None else None, prompt_notes=notes,
            ))
            if result.error and result.error.code == "budget_exhausted":
                termination = "tool_budget"
                break

        final = self.registry.coverage()
        return AgentRunResult(
            termination=termination, summary=summary, error=error, steps=steps,
            finding_ids=list(self.registry.context.finding_ids), token_usage=self.llm.get_token_usage(),
            llm_latency=self.llm.latency_summary(), tool_latency=self.registry.latency_summary(),
            duration_ms=round((time.perf_counter() - started) * 1000), llm_retries=self.llm.retry_count,
            llm_calls=list(self.llm.call_metrics), tool_calls=list(self.registry.context.calls),
            tool_latency_by_tool=self.registry.latency_by_tool(), documents_total=final.total,
            documents_opened=final.opened, documents_not_opened=final.not_opened, finish_refusals=refusals,
            documents_unreadable=final.unreadable,
        )
