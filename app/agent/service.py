"""Investigation orchestration: start, run (guarded against concurrent runs), user follow-ups, resume."""
from __future__ import annotations

import logging
import threading
from typing import Callable

from ..config import Settings
from ..db import Database
from ..documents import get_document, list_documents
from ..llm import LLM, LLMError
from ..mcp_server.client import MCPToolClient, ToolClient
from ..models import DocStatus
from . import store
from .runner import InvestigationRunner

log = logging.getLogger(__name__)
_ACTIVE: set[int] = set()
_LOCK = threading.Lock()


class InvestigationError(Exception):
    """Invalid request (documents not processed, nothing to investigate, already running ...)."""


def start_investigation(db: Database, document_ids: list[int] | None) -> store.Investigation:
    docs = [get_document(db, i) for i in document_ids] if document_ids else [d for d in list_documents(db) if d.status == DocStatus.PROCESSED]
    if not docs:
        raise InvestigationError("no processed documents to investigate; upload and ingest documents first")
    bad = [d.id for d in docs if d.status != DocStatus.PROCESSED]
    if bad:
        raise InvestigationError(f"documents not processed yet: {bad}")
    inv = store.create(db, [d.id for d in docs])
    store.log_event(db, inv.id, "created", f"Investigation created over {len(docs)} documents", {"document_ids": inv.document_ids})
    return inv


def run_investigation(db: Database, settings: Settings, llm_factory: Callable[[], LLM],
                      tool_client_factory: Callable[[list[int]], ToolClient] | None, inv_id: int) -> None:
    """Run (or resume) an investigation. Never raises."""
    with _LOCK:
        if inv_id in _ACTIVE:
            return
        _ACTIVE.add(inv_id)
    tools = None
    try:
        inv = store.get(db, inv_id)
        try:
            llm = llm_factory()
        except LLMError as e:
            store.update(db, inv_id, status="failed", error=str(e))
            store.log_event(db, inv_id, "error", f"Cannot start: {e}")
            return
        tools = tool_client_factory(inv.document_ids) if tool_client_factory else MCPToolClient(db.url, db.schema, inv.document_ids)
        InvestigationRunner(db, settings, llm, tools, inv_id).run()
    except Exception as e:
        log.exception("run_investigation failed")
        store.update(db, inv_id, status="failed", error=f"{type(e).__name__}: {e}")
        store.log_event(db, inv_id, "error", f"Investigation failed to run: {e}")
    finally:
        if tools:
            tools.close()
        with _LOCK:
            _ACTIVE.discard(inv_id)


def is_active(inv_id: int) -> bool:
    with _LOCK:
        return inv_id in _ACTIVE


def request_followup(db: Database, inv_id: int, *, hypothesis_id: int | None = None, finding_id: int | None = None,
                     note: str | None = None, question: str | None = None) -> int:
    """User intervention. Either re-open an existing hypothesis (optionally with a clarification the agent must
    take into account) or add a brand-new question. Returns the hypothesis id that will be (re)investigated."""
    inv = store.get(db, inv_id)
    if is_active(inv_id) or inv.status == "running":
        raise InvestigationError("investigation is running; wait for it to finish")
    if finding_id is not None and hypothesis_id is None:
        f = next((x for x in store.findings(db, inv_id) if x.id == finding_id), None)
        if not f:
            raise InvestigationError(f"finding {finding_id} not found")
        hypothesis_id = f.hypothesis_id
    if hypothesis_id is not None:
        h = store.get_hypothesis(db, hypothesis_id)
        if not h or h.investigation_id != inv_id:
            raise InvestigationError(f"hypothesis {hypothesis_id} not found in this investigation")
        if note:
            store.add_clarification(db, hypothesis_id, note)
        store.supersede_findings_for(db, hypothesis_id)  # history keeps the old finding; UI shows the new one
        store.set_hyp_status(db, hypothesis_id, "pending")
        store.log_event(db, inv_id, "user_feedback", note or "re-investigation requested",
                        {"hypothesis_id": hypothesis_id, "finding_id": finding_id})
    elif question and question.strip():
        hypothesis_id = store.add_hypothesis(db, inv_id, [], question.strip(), "requested by user")
        store.log_event(db, inv_id, "user_request", question.strip(), {"hypothesis_id": hypothesis_id})
    else:
        raise InvestigationError("provide a finding/hypothesis to re-investigate or a new question")
    store.update(db, inv_id, status="queued", phase="investigate")
    return hypothesis_id


def recover_interrupted(db: Database) -> int:
    """On startup: runs that were 'running' when the process died become 'interrupted' (resumable)."""
    rows = db.query("SELECT id FROM investigations WHERE status='running'")
    for r in rows:
        store.update(db, r["id"], status="interrupted")
        store.log_event(db, r["id"], "interrupted", "Process restarted while running; resume to continue")
    return len(rows)


def prepare_resume(db: Database, inv_id: int) -> None:
    inv = store.get(db, inv_id)
    if is_active(inv_id):
        raise InvestigationError("investigation is already running")
    if inv.status not in {"interrupted", "failed", "partial", "queued"}:
        raise InvestigationError(f"cannot resume an investigation in status '{inv.status}'")
    if inv.phase == "done":
        # re-open failed hypotheses only
        for h in store.hypotheses(db, inv_id):
            if h.status == "failed":
                store.set_hyp_status(db, h.id, "pending")
        store.update(db, inv_id, phase="investigate")
    store.update(db, inv_id, status="queued")
    store.log_event(db, inv_id, "resume", "Resume requested")
