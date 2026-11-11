"""Investigation routes: start, progress, hypotheses, findings, history, feedback, resume."""
from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel

from ..agent import service, store
from ..documents import DocumentNotFound
from ..models import Finding


class StartRequest(BaseModel):
    document_ids: list[int] | None = None  # default: all processed documents


class FeedbackRequest(BaseModel):
    finding_id: int | None = None
    hypothesis_id: int | None = None
    note: str | None = None      # clarification the agent must consider
    question: str | None = None  # brand-new question to investigate


class Progress(BaseModel):
    investigation: store.Investigation
    hypotheses_total: int
    hypotheses_by_status: dict[str, int]
    claims: int
    findings: int
    running: bool


def _schedule(request: Request, background: BackgroundTasks, inv_id: int):
    ctx = request.app.state.ctx
    args = (ctx.db, ctx.settings, ctx.llm_factory, ctx.tool_client_factory, inv_id)
    if request.app.state.run_inline:
        service.run_investigation(*args)
    else:
        background.add_task(service.run_investigation, *args)


def build_router() -> APIRouter:
    r = APIRouter(prefix="/api/v1/investigations")

    def db_of(request: Request):
        return request.app.state.ctx.db

    def guard(fn, *a, **k):
        try:
            return fn(*a, **k)
        except store.InvestigationNotFound as e:
            raise HTTPException(404, str(e))
        except DocumentNotFound as e:
            raise HTTPException(404, str(e))
        except service.InvestigationError as e:
            raise HTTPException(409, str(e))

    @r.post("", status_code=202)
    def start(req: StartRequest, request: Request, background: BackgroundTasks) -> store.Investigation:
        inv = guard(service.start_investigation, db_of(request), req.document_ids)
        _schedule(request, background, inv.id)
        return store.get(db_of(request), inv.id)

    @r.get("")
    def list_all(request: Request) -> list[store.Investigation]:
        return store.list_all(db_of(request))

    @r.get("/{inv_id}")
    def progress(inv_id: int, request: Request) -> Progress:
        db = db_of(request)
        inv = guard(store.get, db, inv_id)
        hyps = store.hypotheses(db, inv_id)
        by: dict[str, int] = {}
        for h in hyps:
            by[h.status] = by.get(h.status, 0) + 1
        return Progress(investigation=inv, hypotheses_total=len(hyps), hypotheses_by_status=by,
                        claims=len(store.claims(db, inv_id)), findings=len(store.findings(db, inv_id)),
                        running=service.is_active(inv_id))

    @r.get("/{inv_id}/claims")
    def claims(inv_id: int, request: Request):
        guard(store.get, db_of(request), inv_id)
        return store.claims(db_of(request), inv_id)

    @r.get("/{inv_id}/hypotheses")
    def hypotheses(inv_id: int, request: Request) -> list[store.Hypothesis]:
        guard(store.get, db_of(request), inv_id)
        return store.hypotheses(db_of(request), inv_id)

    @r.get("/{inv_id}/findings")
    def findings(inv_id: int, request: Request, include_superseded: bool = False) -> list[Finding]:
        guard(store.get, db_of(request), inv_id)
        return store.findings(db_of(request), inv_id, include_superseded)

    @r.get("/{inv_id}/events")
    def events(inv_id: int, request: Request, after_id: int = 0) -> list[store.Event]:
        guard(store.get, db_of(request), inv_id)
        return store.events(db_of(request), inv_id, after_id)

    @r.post("/{inv_id}/feedback", status_code=202)
    def feedback(inv_id: int, req: FeedbackRequest, request: Request, background: BackgroundTasks):
        db = db_of(request)
        hid = guard(service.request_followup, db, inv_id, hypothesis_id=req.hypothesis_id, finding_id=req.finding_id,
                    note=req.note, question=req.question)
        _schedule(request, background, inv_id)
        return {"hypothesis_id": hid}

    @r.post("/{inv_id}/resume", status_code=202)
    def resume(inv_id: int, request: Request, background: BackgroundTasks):
        guard(service.prepare_resume, db_of(request), inv_id)
        _schedule(request, background, inv_id)
        return {"status": "queued"}

    return r
