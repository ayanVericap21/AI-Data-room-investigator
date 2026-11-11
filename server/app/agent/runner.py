"""Runs one investigation end to end: creates the investigation record, runs the agent, writes
a trace event after every step (committed immediately, so a UI polling the table sees live
progress), and records the outcome. Uses only the existing tables."""
from __future__ import annotations

import logging
from typing import Callable

from sqlalchemy.orm import Session

from app.agent.loop import InvestigationAgent
from app.agent.schemas import AgentRunResult, AgentStepRecord
from app.llm.base import LLMClient
from app.models import Investigation, InvestigationEvent
from app.models.base import Approach, InvestigationStatus, utcnow
from app.tools.base import SearchBackend, ToolBudget
from app.tools.registry import build_registry

logger = logging.getLogger(__name__)

_STATUS = {
    "finished": InvestigationStatus.completed,
    "max_steps": InvestigationStatus.partially_completed,
    "tool_budget": InvestigationStatus.partially_completed,
    "stopped": InvestigationStatus.stopped,
    "llm_error": InvestigationStatus.failed,
}


def run_investigation(session: Session, llm: LLMClient, retrieval: SearchBackend | None, *, name: str,
                      focus: str | None = None, max_steps: int | None = 15, budget: ToolBudget | None = None,
                      should_stop: Callable[[], bool] | None = None,
                      on_progress: Callable[[AgentStepRecord], None] | None = None,
                      max_finish_refusals: int = 2) -> AgentRunResult:
    budget = budget or ToolBudget()
    investigation = Investigation(
        name=name, status=InvestigationStatus.running.value, approach=Approach.improved.value, started_at=utcnow(),
        execution_config={"focus": focus, "max_steps": max_steps, "budget": budget.model_dump(), "agent": "tool_loop"},
    )
    session.add(investigation)
    session.commit()
    investigation_id = investigation.id

    def on_step(record: AgentStepRecord) -> None:
        label = record.tool if record.action == "call_tool" else record.action
        session.add(InvestigationEvent(
            investigation_id=investigation_id, event_type="agent_step", node_name="investigate",
            message=f"Step {record.step}: {label}. {record.rationale}".strip(),
            status="success" if record.ok else "warning",
            details={"tool": record.tool, "arguments": record.arguments, "ok": record.ok, "error_code": record.error_code,
                     "result": record.result_summary, "latency_ms": record.tool_latency_ms},
        ))
        session.commit()
        if on_progress:
            on_progress(record)

    registry = build_registry(session, retrieval, investigation_id, budget)
    agent = InvestigationAgent(llm, registry, max_steps=max_steps, should_stop=should_stop, on_step=on_step,
                               max_finish_refusals=max_finish_refusals)
    try:
        result = agent.run(focus)
    except Exception as exc:  # unexpected bug: record it, then re-raise
        session.rollback()
        investigation = session.get(Investigation, investigation_id)
        investigation.status = InvestigationStatus.failed.value
        investigation.error_message = f"{exc.__class__.__name__}: {exc}"[:500]
        investigation.completed_at = utcnow()
        session.commit()
        raise

    investigation = session.get(Investigation, investigation_id)
    investigation.status = _STATUS[result.termination].value
    investigation.error_message = result.error
    investigation.total_tokens = result.token_usage.total_tokens
    investigation.duration_ms = result.duration_ms
    now = utcnow()
    if result.termination == "stopped":
        investigation.stopped_at = now
    else:
        investigation.completed_at = now
    session.commit()
    result.investigation_id = investigation_id
    return result
