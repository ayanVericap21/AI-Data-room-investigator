"""HTTP API: documents (upload triggers automatic preprocessing), ingest jobs, MCP server status.
Investigation routes live in app/api/investigations.py."""
from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Callable

from fastapi import APIRouter, BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .. import jobs as jobs_svc
from ..config import Settings, get_settings
from ..db import Database
from ..llm import create_llm
from ..documents import DocumentNotFound, InvalidUpload, get_document, list_documents, save_upload
from ..mcp_server.client import MCPToolClient
from ..models import Chunk, DocStatus, Document, IngestJob, utcnow
from ..tools import DataRoomTools

log = logging.getLogger(__name__)


class UploadResult(BaseModel):
    filename: str
    document: Document | None = None
    duplicate: bool = False
    error: str | None = None
    job_id: int | None = None  # the automatic preprocessing job this document was queued in (None if nothing to do)


class McpTool(BaseModel):
    name: str
    description: str


class McpStatus(BaseModel):
    ready: bool                      # a real MCP handshake (spawn server, list tools) succeeded
    transport: str = "stdio"
    tools: list[McpTool] = Field(default_factory=list)
    documents_served: int = 0        # processed documents the server exposes
    documents_with_summary: int = 0
    error: str | None = None
    checked_at: str


class JobRequest(BaseModel):
    document_ids: list[int] = Field(min_length=1)
    idempotency_key: str | None = None


class AppContext:
    def __init__(self, settings: Settings, llm_factory: Callable | None = None, tool_client_factory: Callable | None = None):
        self.settings = settings
        self.db = Database(settings.require_database_url(), settings.db_schema)
        # Default LLM: Anthropic when a key is configured. The factory raises LLMError otherwise, which the ingest job
        # treats as "summaries skipped" and an investigation treats as a clean failure.
        self.llm_factory = llm_factory or (lambda: create_llm(settings))
        self.tool_client_factory = tool_client_factory
        jobs_svc.recover_stuck_jobs(self.db)
        self.mcp_cache: tuple[float, list[McpTool], str | None] | None = None  # (expires_at, tools, error)

    def start_pending_preprocessing(self, run_inline: bool) -> None:
        """At boot: anything uploaded but never preprocessed (or interrupted half-way) is processed now."""
        jobs_svc.recover_stuck_documents(self.db)
        job = jobs_svc.submit_for_upload(self.db, [d.id for d in list_documents(self.db)])
        if job:
            run = lambda: jobs_svc.run_job(self.db, self.settings, job.id, self.llm_factory)  # noqa: E731
            run() if run_inline else threading.Thread(target=run, daemon=True, name="preprocess-pending").start()

    def mcp_status(self) -> McpStatus:
        """Handshake with the MCP server (cached briefly: it spawns a process) + live counts of what it serves."""
        now = time.time()
        if not self.mcp_cache or self.mcp_cache[0] < now:
            client = None
            try:
                client = (self.tool_client_factory([]) if self.tool_client_factory
                          else MCPToolClient(self.db.url, self.db.schema, None))
                tools = [McpTool(**t) for t in client.list_tools()]
                self.mcp_cache = (now + 60, tools, None)
            except Exception as e:  # server failed to start / handshake failed: report, don't crash
                log.warning("MCP status check failed: %s", e)
                self.mcp_cache = (now + 10, [], f"{type(e).__name__}: {e}"[:300])
            finally:
                if client:
                    client.close()
        _, tools, error = self.mcp_cache
        docs = DataRoomTools(self.db).list_documents()["documents"]
        return McpStatus(ready=error is None and bool(tools), tools=tools, documents_served=len(docs),
                         documents_with_summary=sum(1 for d in docs if d["summary"]), error=error, checked_at=utcnow())


def create_app(settings: Settings | None = None, llm_factory: Callable | None = None,
               tool_client_factory: Callable | None = None, run_inline: bool = False) -> FastAPI:
    """run_inline=True runs background work synchronously (tests)."""
    ctx = AppContext(settings or get_settings(), llm_factory, tool_client_factory)
    app = FastAPI(title="Data Room Investigator")
    app.add_middleware(CORSMiddleware, allow_origins=ctx.settings.cors_origins, allow_methods=["GET", "POST"],
                       allow_headers=["*"])  # only the configured UI origins; the API itself has no auth - keep it on localhost
    app.state.ctx = ctx
    app.state.run_inline = run_inline
    r = APIRouter(prefix="/api/v1")

    @r.get("/health")
    def health():
        return {"status": "ok"}

    @r.post("/documents", status_code=201)
    async def upload(background: BackgroundTasks, files: list[UploadFile] = File(...)) -> list[UploadResult]:
        """Store the files, then preprocess them automatically (parse -> chunk -> index -> summarise) in the background."""
        out = []
        for f in files:
            try:
                doc, dup = save_upload(ctx.db, ctx.settings, f.filename or "", await f.read())
                out.append(UploadResult(filename=f.filename or "", document=doc, duplicate=dup))
            except InvalidUpload as e:  # per-file errors don't reject the whole batch
                out.append(UploadResult(filename=f.filename or "", error=str(e)))
        job = jobs_svc.submit_for_upload(ctx.db, [o.document.id for o in out if o.document])
        if job:
            for o in out:
                if o.document and o.document.id in job.document_ids:
                    o.job_id = job.id
            if app.state.run_inline:
                jobs_svc.run_job(ctx.db, ctx.settings, job.id, ctx.llm_factory)
            else:
                background.add_task(jobs_svc.run_job, ctx.db, ctx.settings, job.id, ctx.llm_factory)
        return out

    @r.get("/mcp")
    def mcp() -> McpStatus:
        return ctx.mcp_status()

    @r.get("/documents")
    def documents() -> list[Document]:
        return list_documents(ctx.db)

    @r.get("/documents/{doc_id}")
    def document(doc_id: int) -> Document:
        try:
            return get_document(ctx.db, doc_id)
        except DocumentNotFound as e:
            raise HTTPException(404, str(e))

    @r.get("/documents/{doc_id}/chunks")
    def chunks(doc_id: int) -> list[Chunk]:
        try:
            get_document(ctx.db, doc_id)
        except DocumentNotFound as e:
            raise HTTPException(404, str(e))
        return [Chunk(**{k: x[k] for k in x.keys()}) for x in
                ctx.db.query("SELECT * FROM chunks WHERE document_id=? ORDER BY ordinal", (doc_id,))]

    @r.post("/jobs", status_code=202)
    def submit(req: JobRequest, background: BackgroundTasks) -> IngestJob:
        key = req.idempotency_key or f"auto-{uuid.uuid4()}"
        try:
            job, created = jobs_svc.submit_job(ctx.db, req.document_ids, key)
        except DocumentNotFound as e:
            raise HTTPException(404, str(e))
        except (jobs_svc.JobConflict, jobs_svc.NothingToProcess) as e:
            raise HTTPException(409, str(e))
        except ValueError as e:
            raise HTTPException(422, str(e))
        if created:
            if app.state.run_inline:
                return jobs_svc.run_job(ctx.db, ctx.settings, job.id, ctx.llm_factory)
            background.add_task(jobs_svc.run_job, ctx.db, ctx.settings, job.id, ctx.llm_factory)
        return job

    @r.get("/jobs")
    def jobs() -> list[IngestJob]:
        return jobs_svc.list_jobs(ctx.db)

    @r.get("/jobs/{job_id}")
    def job(job_id: int) -> IngestJob:
        try:
            return jobs_svc.get_job(ctx.db, job_id)
        except jobs_svc.JobNotFound as e:
            raise HTTPException(404, str(e))

    from .investigations import build_router
    from ..agent import service as agent_service

    agent_service.recover_interrupted(ctx.db)
    ctx.start_pending_preprocessing(run_inline)
    app.include_router(r)
    app.include_router(build_router())
    return app


def app_factory() -> FastAPI:  # uvicorn --factory app.api.main:app_factory
    return create_app()
