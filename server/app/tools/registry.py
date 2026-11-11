"""The tool registry: describes tools to the agent and executes calls safely.

``execute`` never raises. Bad arguments, unknown tools, exhausted budgets and tool errors all
come back as a structured result so the agent can correct itself. Every call is counted
against the budget, logged, and timed.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from pydantic import BaseModel, ValidationError
from sqlalchemy.orm import Session

from app.llm.metrics import CallMetric, LatencySummary, summarize
from app.tools.base import (SearchBackend, ToolBudget, ToolCallRecord, ToolContext, ToolDescription, ToolError,
                            ToolErrorInfo, ToolResult, ToolSpec)
from app.tools.calculate import calculate
from app.tools.facts import record_fact
from app.tools.reading import coverage, get_chunk, get_outline, list_documents, read_section
from app.tools.schemas import (CalculateInput, Coverage, GetChunkInput, GetOutlineInput, ListDocumentsInput, ReadSectionInput,
                               RecordFactInput, SearchInput, SubmitFindingInput)
from app.tools.search import search
from app.tools.submit import submit_finding

logger = logging.getLogger(__name__)

TOOLS: list[ToolSpec] = [
    ToolSpec(
        "list_documents",
        "List every document in the data room with its id, type, processing status and number of chunks. Start here.",
        ListDocumentsInput, list_documents, lambda o: f"{len(o.documents)} documents",
    ),
    ToolSpec(
        "get_outline",
        "Show the structure of one document: its pages, sheets or headings with the number of chunks in each. "
        "Use it to decide what to read.",
        GetOutlineInput, get_outline, lambda o: f"{o.filename}: {len(o.sections)} sections, {o.chunk_count} chunks",
    ),
    ToolSpec(
        "search",
        "Semantic search over all stored passages. Returns the closest passages with their exact text, source and "
        "location. Scores are similarity only and never prove a contradiction.",
        SearchInput, search,
        lambda o: f"{len(o.hits)} hits" + (f", best {o.hits[0].score} ({o.hits[0].filename}, {o.hits[0].locator})" if o.hits else ""),
    ),
    ToolSpec(
        "get_chunk",
        "Fetch one passage by id, optionally with neighbouring passages for context.",
        GetChunkInput, get_chunk, lambda o: f"{o.chunk.filename}, {o.chunk.locator}",
    ),
    ToolSpec(
        "read_section",
        "Read the passages of one page (PDF), sheet (XLSX) or heading (DOCX) of a document, in order. "
        "Give exactly one of page_number, sheet_name or heading.",
        ReadSectionInput, read_section, lambda o: f"{o.filename} {o.selector}: {len(o.chunks)} of {o.total_chunks} chunks",
    ),
    ToolSpec(
        "calculate",
        "Exact arithmetic on plain numbers (+ - * / % ** and parentheses). Always use this for differences, "
        "ratios and percentages instead of computing them yourself.",
        CalculateInput, calculate, lambda o: f"{o.expression} = {o.result}",
    ),
    ToolSpec(
        "record_fact",
        "Save facts you have just read into your running memory: what it is about (label), the stated value, its unit, "
        "what it applies to (qualifier), and the exact quote with its chunk id. Quotes are verified; if any is not "
        "found exactly, none of the batch is saved. Record every value, date, name, quantity, condition and "
        "commitment as you read, so they can be compared across documents afterwards.",
        RecordFactInput, record_fact, lambda o: f"recorded {len(o.recorded)} facts, {o.total_facts} in memory",
    ),
    ToolSpec(
        "submit_finding",
        "Save a finding. Every evidence quote is verified against the stored text and the finding is rejected "
        "if any quote is not found exactly. Comparing classifications (contradiction, contextual_difference, "
        "consistent) need evidence from at least two chunks. For missing_information or insufficient_evidence, "
        "cite what exists and describe in the description which searches found nothing.",
        SubmitFindingInput, submit_finding, lambda o: f"saved {o.finding_id} with {len(o.evidence)} verified quotes",
    ),
]


class ToolRegistry:
    def __init__(self, context: ToolContext, tools: list[ToolSpec] | None = None):
        self.context = context
        self._tools = {t.name: t for t in (tools or TOOLS)}

    # -- description ---------------------------------------------------------
    def describe(self) -> list[ToolDescription]:
        return [
            ToolDescription(name=t.name, description=t.description, input_schema=t.input_model.model_json_schema())
            for t in self._tools.values()
        ]

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    # -- execution -----------------------------------------------------------
    def execute(self, name: str, arguments: dict[str, Any] | None = None) -> ToolResult:
        ctx = self.context
        started = time.perf_counter()
        arguments = arguments or {}

        spec = self._tools.get(name)
        if spec is None:
            return self._finish(name, arguments, started, ToolError(
                "unknown_tool", f"There is no tool '{name}'. Available tools: {', '.join(self._tools)}."))
        if ctx.budget.max_tool_calls is not None and ctx.tool_calls_used >= ctx.budget.max_tool_calls:
            return self._finish(name, arguments, started, ToolError(
                "budget_exhausted", f"The tool budget of {ctx.budget.max_tool_calls} calls is used up. "
                                    "Submit any remaining findings you can verify, then finish."))
        ctx.tool_calls_used += 1

        try:
            args = spec.input_model.model_validate(arguments)
        except ValidationError as exc:
            detail = "; ".join(f"{'.'.join(str(p) for p in e['loc']) or 'arguments'}: {e['msg']}" for e in exc.errors())
            return self._finish(name, arguments, started, ToolError("invalid_arguments", detail))
        try:
            output = spec.handler(ctx, args)
        except ToolError as exc:
            return self._finish(name, arguments, started, exc)
        except Exception:  # a tool bug must not crash the run
            logger.exception("Tool failed unexpectedly", extra={"tool": name})
            return self._finish(name, arguments, started, ToolError("internal_error", "The tool failed unexpectedly."))
        return self._finish(name, arguments, started, None, output=output, spec=spec)

    def _finish(self, name: str, arguments: dict[str, Any], started: float, error: ToolError | None,
                output: BaseModel | None = None, spec: ToolSpec | None = None) -> ToolResult:
        latency = round((time.perf_counter() - started) * 1000)
        summary = spec.summarize(output) if spec and output is not None else (f"{error.code}: {error.message}"[:200] if error else "")
        ctx = self.context
        ctx.calls.append(ToolCallRecord(
            index=len(ctx.calls) + 1, tool=name, arguments=arguments, ok=error is None,
            error_code=error.code if error else None, latency_ms=latency, result_summary=summary,
        ))
        ctx.metrics.append(CallMetric(kind="tool", latency_ms=latency, ok=error is None,
                                      error=error.code if error else None))
        return ToolResult(
            tool=name, ok=error is None, output=output, latency_ms=latency, summary=summary,
            error=ToolErrorInfo(code=error.code, message=error.message) if error else None,
        )

    def coverage(self) -> Coverage:
        return coverage(self.context)

    # -- telemetry -----------------------------------------------------------
    def latency_summary(self) -> LatencySummary:
        return summarize(self.context.metrics)

    def latency_by_tool(self) -> dict[str, LatencySummary]:
        by_tool: dict[str, list[CallMetric]] = {}
        for record, metric in zip(self.context.calls, self.context.metrics):
            by_tool.setdefault(record.tool, []).append(metric)
        return {name: summarize(items) for name, items in by_tool.items()}


def build_registry(session: Session, retrieval: SearchBackend | None, investigation_id: str | None,
                   budget: ToolBudget | None = None) -> ToolRegistry:
    return ToolRegistry(ToolContext(session=session, retrieval=retrieval, investigation_id=investigation_id,
                                    budget=budget or ToolBudget()))
