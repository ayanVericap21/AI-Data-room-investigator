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
                      max_finish_refusals: int = 2, search_min_score: float | None = None,
                      prices_per_mtok: tuple[float, float] | None = None) -> AgentRunResult:
    """``prices_per_mtok``: (input, output) price per million tokens. Without it the cost is not calculated and the run says so."""
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
                     "result": record.result_summary, "latency_ms": record.tool_latency_ms,
                     # everything needed to reconstruct the step: what the model was shown, what it answered, what the tool returned in full
                     "prompt": record.prompt, "action": record.action_json, "output": record.output, "error_message": record.error_message,
                     "llm_attempts": [a.model_dump() for a in record.llm_attempts], "prompt_notes": record.prompt_notes},
        ))
        if record.tool == "submit_finding" and not record.ok:  # a rejected finding keeps its full payload and the reasons
            session.add(InvestigationEvent(
                investigation_id=investigation_id, event_type="finding_rejected", node_name="submit_finding",
                message=f"Step {record.step}: finding rejected ({record.error_code})", status="warning",
                details={"submitted": record.arguments, "error_code": record.error_code, "error_message": record.error_message},
            ))
        session.commit()
        if on_progress:
            on_progress(record)

    registry = build_registry(session, retrieval, investigation_id, budget, search_min_score, agent_checks=True)
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
    status = _STATUS[result.termination]
    gaps = bool(result.documents_not_opened or result.documents_unreadable)
    if result.termination == "finished" and gaps:  # finishing is accepted after the refusal limit, but the run is not complete
        status = InvestigationStatus.partially_completed
    investigation.status = status.value
    investigation.error_message = result.error
    investigation.total_tokens = result.token_usage.total_tokens
    investigation.duration_ms = result.duration_ms
    cost = None
    if prices_per_mtok is not None:
        cost = (result.token_usage.prompt_tokens * prices_per_mtok[0] + result.token_usage.completion_tokens * prices_per_mtok[1]) / 1_000_000
        investigation.estimated_cost = cost
    investigation.execution_config = {**(investigation.execution_config or {}), "coverage": {
        "documents_total": result.documents_total, "documents_opened": result.documents_opened, "not_opened": result.documents_not_opened,
        "unreadable": result.documents_unreadable, "finish_refusals": result.finish_refusals},
        "cost": {"estimated": cost, "note": None if cost is not None else "not calculated: no token prices configured"}}
    now = utcnow()
    if result.termination == "stopped":
        investigation.stopped_at = now
    else:
        investigation.completed_at = now
    session.commit()
    result.investigation_id = investigation_id
    return result
