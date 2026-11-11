"""Models for the agent loop: the action the model returns each turn, and the run result."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from app.llm.base import TokenUsage
from app.llm.metrics import CallMetric, LatencySummary
from app.tools.base import ToolCallRecord

Termination = Literal["finished", "max_steps", "tool_budget", "stopped", "llm_error"]


class AgentAction(BaseModel):
    """One turn from the model: call exactly one tool, or finish.

    ``rationale`` is a short operational note on why this action (shown in the trace). It is
    not a place for step-by-step reasoning."""

    action: Literal["call_tool", "finish"]
    tool: str | None = Field(None, description="Tool name, when action is call_tool.")
    arguments: dict[str, Any] = Field(default_factory=dict, description="Arguments for the tool.")
    rationale: str = Field(max_length=500, description="One or two sentences: why this action.")
    summary: str | None = Field(None, description="When finishing: what you examined and found.")

    @model_validator(mode="before")
    @classmethod
    def _finish_sent_as_a_tool(cls, data: Any) -> Any:
        """Models sometimes answer action=call_tool with tool="finish". That means finish, not an unknown tool."""
        if isinstance(data, dict) and data.get("action") == "call_tool" and str(data.get("tool") or "").strip().lower() in ("finish", "done"):
            args = data.get("arguments")
            summary = data.get("summary") or (args.get("summary") if isinstance(args, dict) else None)
            return {**data, "action": "finish", "tool": None, "arguments": {}, "summary": summary}
        return data

    @model_validator(mode="after")
    def _tool_required(self) -> "AgentAction":
        if self.action == "call_tool" and not self.tool:
            raise ValueError("tool is required when action is call_tool")
        return self


class AgentStepRecord(BaseModel):
    step: int
    action: str  # call_tool | finish | finish_refused | llm_error
    tool: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""
    ok: bool = True
    error_code: str | None = None
    result_summary: str = ""
    tool_latency_ms: int = 0
    tool_call_index: int | None = None  # index in the tool call log
    llm_attempts: list[CallMetric] = Field(default_factory=list)  # every provider attempt made for this turn


class AgentRunResult(BaseModel):
    investigation_id: str | None = None
    termination: Termination
    summary: str | None = None
    error: str | None = None
    steps: list[AgentStepRecord]
    finding_ids: list[str]
    token_usage: TokenUsage
    llm_latency: LatencySummary
    tool_latency: LatencySummary
    duration_ms: int
    llm_retries: int = 0
    llm_calls: list[CallMetric] = Field(default_factory=list)  # every attempt, in order
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    tool_latency_by_tool: dict[str, LatencySummary] = Field(default_factory=dict)
    documents_total: int = 0  # readable documents in the data room
    documents_opened: int = 0  # opened by the agent (outline, read_section or get_chunk)
    documents_not_opened: list[str] = Field(default_factory=list)
    finish_refusals: int = 0  # times finishing was refused because documents were still unopened
