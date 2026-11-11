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

from app.agent.matcher import FactMatcher
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
                 on_step: Callable[[AgentStepRecord], None] | None = None, max_finish_refusals: int = 2,
                 match_facts: bool = True):
        self.llm = llm
        self.registry = registry
        self.max_steps = max_steps  # None = no step limit; the run ends when the model finishes or is stopped
        self.full_results = full_results  # how many of the latest tool results are shown in full
        self.max_result_chars = max_result_chars
        self.should_stop = should_stop
        self.on_step = on_step
        self.match_facts = match_facts  # after the agent finishes, one LLM step compares the recorded facts
        self.max_finish_refusals = max_finish_refusals  # times finishing may be refused while documents are unopened

    @staticmethod
    def _unread_text(cov) -> str:
        """Which chunks are still unread, per document, for the prompt and the refusal message."""
        parts = []
        for d in cov.not_fully_read:
            shown = ", ".join(d.unread_sections[:8]) + (f" and {len(d.unread_sections) - 8} more" if len(d.unread_sections) > 8 else "")
            parts.append(f"{d.filename} (read {d.chunks_read} of {d.chunks} chunks; unread: {shown})")
        return "Still to read: " + "; ".join(parts) + "." if parts else ""

    def _coverage_line(self, cov) -> str:
        if not cov.not_fully_read:
            return f"Documents fully read: {cov.fully_read} of {cov.total}. Nothing is left to read."
        return f"Documents fully read: {cov.fully_read} of {cov.total}. {self._unread_text(cov)}"

    def _history(self, turns: list[_Turn]) -> list[str]:
        """Older results are shortened to their one-line summary so the prompt stays bounded."""
        rendered = []
        for i, turn in enumerate(turns):
            recent = i >= len(turns) - self.full_results
            args = json.dumps(turn.action.arguments, separators=(",", ":"), default=str)
            body = turn.result.to_text(self.max_result_chars) if recent else f"(shortened) {turn.result.summary}"
            rendered.append(f"Step {turn.step}: you called {turn.action.tool or turn.action.action} {args}\n"
                            f"Why: {turn.action.rationale}\nResult: {body}")
        return rendered

    def run(self, focus: str | None = None) -> AgentRunResult:
        started = time.perf_counter()
        tools = self.registry.describe()
        turns: list[_Turn] = []
        steps: list[AgentStepRecord] = []
        termination: Termination = "max_steps"
        summary: str | None = None
        error: str | None = None
        refusals = 0
        stalled = 0
        reads_at_refusal = -1

        for n in itertools.count(1):
            if self.max_steps is not None and n > self.max_steps:
                break
            if self.should_stop and self.should_stop():
                termination = "stopped"
                break
            cov = self.registry.coverage()
            coverage_line = self._coverage_line(cov)
            prompt = build_prompt(focus, tools, self._history(turns), None if self.max_steps is None else self.max_steps - n + 1, coverage_line)
            attempts_before = len(self.llm.call_metrics)
            try:
                action = self.llm.generate_structured_output(prompt, AgentAction, system=SYSTEM)
            except LLMError as exc:
                termination, error = "llm_error", str(exc)[:300]
                logger.warning("Agent stopped on LLM error", extra={"step": n, "error": error})
                record = AgentStepRecord(step=n, action="llm_error", ok=False, error_code="llm_error",
                                         result_summary=error, llm_attempts=self.llm.call_metrics[attempts_before:])
                steps.append(record)
                if self.on_step:
                    self.on_step(record)
                break
            attempts = self.llm.call_metrics[attempts_before:]

            if action.action == "finish" and cov.not_fully_read:
                reads_now = sum(len(v) for v in self.registry.context.chunks_read.values())
                # Patience: refuse while the agent keeps reading; allow finishing only if it insists
                # max_finish_refusals times in a row without reading anything new (so it cannot loop forever).
                stalled = stalled + 1 if reads_now == reads_at_refusal else 0
                if stalled < self.max_finish_refusals:
                    refusals += 1
                    reads_at_refusal = reads_now
                    message = (f"You cannot finish yet: {len(cov.not_fully_read)} of {cov.total} documents are not fully read. "
                               f"{self._unread_text(cov)} Read the unread sections (read_section), then finish.")
                    turns.append(_Turn(n, action, ToolResult(tool="finish", ok=False, summary=message[:200],
                                                             error=ToolErrorInfo(code="chunks_unread", message=message))))
                    record = AgentStepRecord(step=n, action="finish_refused", ok=False, error_code="chunks_unread",
                                             rationale=action.rationale,
                                             result_summary=f"{len(cov.not_fully_read)} documents not fully read",
                                             llm_attempts=attempts)
                    steps.append(record)
                    if self.on_step:
                        self.on_step(record)
                    continue

            if action.action == "finish":
                record = AgentStepRecord(step=n, action="finish", rationale=action.rationale, llm_attempts=attempts)
                steps.append(record)
                if self.on_step:
                    self.on_step(record)
                summary, termination = action.summary, "finished"
                break

            result = self.registry.execute(action.tool or "", action.arguments)
            turns.append(_Turn(n, action, result))
            record = AgentStepRecord(
                step=n, action="call_tool", tool=action.tool, arguments=action.arguments, rationale=action.rationale,
                ok=result.ok, error_code=result.error.code if result.error else None,
                result_summary=result.summary, tool_latency_ms=result.latency_ms,
                tool_call_index=self.registry.context.calls[-1].index, llm_attempts=attempts,
            )
            steps.append(record)
            if self.on_step:
                self.on_step(record)
            if result.error and result.error.code == "budget_exhausted":
                termination = "tool_budget"
                break

        matching = None
        if self.match_facts and self.registry.context.facts and termination in ("finished", "max_steps", "tool_budget"):
            matching = FactMatcher(self.llm, self.registry).run()
        final = self.registry.coverage()
        return AgentRunResult(
            termination=termination, summary=summary, error=error, steps=steps,
            finding_ids=list(self.registry.context.finding_ids), token_usage=self.llm.get_token_usage(),
            llm_latency=self.llm.latency_summary(), tool_latency=self.registry.latency_summary(),
            duration_ms=round((time.perf_counter() - started) * 1000), llm_retries=self.llm.retry_count,
            llm_calls=list(self.llm.call_metrics), tool_calls=list(self.registry.context.calls),
            tool_latency_by_tool=self.registry.latency_by_tool(), documents_total=final.total,
            documents_fully_read=final.fully_read,
            documents_not_fully_read=[f"{d.filename} (read {d.chunks_read} of {d.chunks} chunks)" for d in final.not_fully_read],
            finish_refusals=refusals, matching=matching, facts=list(self.registry.context.facts),
        )
