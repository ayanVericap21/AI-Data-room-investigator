"""Shared types for the tool layer: context, budget, specs, results.

The tool layer is the only way the investigation agent touches the data. It is plain,
typed Python (no MCP dependency); an MCP server can later wrap ``ToolRegistry`` without
changing any tool. Everything the agent must not be trusted with is enforced here, in
code: call budgets, the call log, and exact-quote verification in ``submit_finding``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Protocol

from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny
from sqlalchemy.orm import Session

from app.llm.metrics import CallMetric
from app.services.retrieval_service import RetrievedChunk

if TYPE_CHECKING:
    from app.tools.schemas import RecordedFact


class ToolInput(BaseModel):
    """Base for tool arguments. Unknown arguments are rejected so a misnamed argument is
    reported back to the agent instead of being silently ignored."""

    model_config = ConfigDict(extra="forbid")


class ToolError(Exception):
    """Raised by a tool handler; becomes a structured error result the agent can act on."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class SearchBackend(Protocol):
    def search(self, query: str, top_k: int = 5, document_ids: list[str] | None = None,
               exclude_chunk_ids: list[str] | None = None, exclude_document_ids: list[str] | None = None,
               min_score: float | None = None) -> list[RetrievedChunk]: ...


class ToolBudget(BaseModel):
    max_tool_calls: int | None = Field(None, ge=1)  # None = no limit; every call counts, including failed ones
    max_findings: int | None = Field(None, ge=1)  # None = no limit


class ToolCallRecord(BaseModel):
    """One entry of the call log. Operational facts only, never model reasoning."""

    index: int
    tool: str
    arguments: dict[str, Any]
    ok: bool
    error_code: str | None = None
    latency_ms: int
    result_summary: str = ""


class ToolErrorInfo(BaseModel):
    code: str
    message: str


class ToolResult(BaseModel):
    tool: str
    ok: bool
    output: SerializeAsAny[BaseModel] | None = None
    error: ToolErrorInfo | None = None
    latency_ms: int = 0
    summary: str = ""

    def to_text(self, max_chars: int | None = None) -> str:
        """JSON shown to the agent, optionally truncated to bound prompt size."""
        if self.ok and self.output is not None:
            text = self.output.model_dump_json(exclude_none=True)
        else:
            text = self.model_dump_json(include={"ok", "error"})
        if max_chars is not None and len(text) > max_chars:
            text = text[:max_chars] + f'... [truncated {len(text) - max_chars} chars; call a narrower tool]'
        return text


@dataclass
class ToolContext:
    """Everything a tool may use. One context per investigation run."""

    session: Session
    retrieval: SearchBackend | None = None
    investigation_id: str | None = None
    budget: ToolBudget = field(default_factory=ToolBudget)
    calls: list[ToolCallRecord] = field(default_factory=list)
    metrics: list[CallMetric] = field(default_factory=list)
    tool_calls_used: int = 0
    finding_ids: list[str] = field(default_factory=list)
    recent_queries: list[str] = field(default_factory=list)  # searches since the last saved finding
    # chunk ids whose text a tool has shown to the agent, by document (search hits, read_section, get_chunk).
    # An outline shows no chunk text, so it does not count as reading.
    chunks_read: dict[str, set[str]] = field(default_factory=dict)
    # the agent's running memory: verified facts it has recorded this run (in memory only, not stored)
    facts: list[RecordedFact] = field(default_factory=list)

    def mark_read(self, document_id: str, chunk_ids) -> None:
        self.chunks_read.setdefault(document_id, set()).update(chunk_ids)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_model: type[ToolInput]
    handler: Callable[[ToolContext, Any], BaseModel]
    summarize: Callable[[BaseModel], str]


class ToolDescription(BaseModel):
    """What the agent (and docs) see about a tool."""

    name: str
    description: str
    input_schema: dict[str, Any]
