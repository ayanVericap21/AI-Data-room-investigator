"""document_ingest job lifecycle: submit (validate + idempotency) -> run (isolated per document) -> status.

Mirrors the ai-service intake order: idempotency -> existence -> duplicate-processing -> queue.
"""
from __future__ import annotations

import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from .config import Settings
from .db import Database, dumps, loads
from .documents import DocumentNotFound, get_document
from .ingest.pipeline import ingest_document
from .ingest.summarize import mark_skipped, summarize_document
from .llm import LLM, LLMError
from .models import DocResult, DocStatus, IngestJob, JobStatus, utcnow

log = logging.getLogger(__name__)

# Serialises ingest runs (analogue of the per-deal Redis lock): one job at a time per process.
_RUN_LOCK = threading.Lock()


class JobConflict(Exception):
    """Same idempotency key with different documents, or documents already being processed."""


class NothingToProcess(Exception):
    """All requested documents are already processed."""


class JobNotFound(LookupError):
    pass


def _row_to_job(r) -> IngestJob:
    return IngestJob(
        id=r["id"], status=JobStatus(r["status"]), document_ids=loads(r["document_ids"], []),
        idempotency_key=r["idempotency_key"], results=[DocResult(**x) for x in loads(r["results"], [])],
        usage=loads(r["usage"], {}), error=r["error"], created_at=r["created_at"], started_at=r["started_at"], finished_at=r["finished_at"],
    )


def get_job(db: Database, job_id: int) -> IngestJob:
    r = db.one("SELECT * FROM ingest_jobs WHERE id=?", (job_id,))
    if not r:
        raise JobNotFound(f"job {job_id} not found")
    return _row_to_job(r)


def list_jobs(db: Database) -> list[IngestJob]:
    return [_row_to_job(r) for r in db.query("SELECT * FROM ingest_jobs ORDER BY id DESC")]


def submit_job(db: Database, document_ids: list[int], idempotency_key: str) -> tuple[IngestJob, bool]:
    """Returns (job, created). created=False means an idempotent replay of an earlier submit."""
    ids = sorted(set(document_ids))
    if not ids:
        raise ValueError("document_ids must not be empty")
    if not idempotency_key.strip():
        raise ValueError("idempotency_key must not be empty")

    # 1. idempotency
    existing = db.one("SELECT * FROM ingest_jobs WHERE idempotency_key=?", (idempotency_key,))
    if existing:
        job = _row_to_job(existing)
        if sorted(job.document_ids) != ids:
            raise JobConflict("idempotency key already used with different document_ids")
        return job, False

    # 2. every document must exist
    docs = []
    for i in ids:
        docs.append(get_document(db, i))  # raises DocumentNotFound

    # 3. documents already in a live job -> conflict
    live = [_row_to_job(r) for r in db.query("SELECT * FROM ingest_jobs WHERE status IN ('queued','running')")]
    busy = {d for j in live for d in j.document_ids} & set(ids)
    if busy:
        raise JobConflict(f"documents already being processed: {sorted(busy)}")

    # 4. skip documents already processed (Node-side dedupe analogue)
    todo = [d.id for d in docs if d.status != DocStatus.PROCESSED]
    if not todo:
        raise NothingToProcess("all requested documents are already processed")

    job_id = db.execute(
        "INSERT INTO ingest_jobs(status, document_ids, idempotency_key, created_at) VALUES(?,?,?,?)",
        (JobStatus.QUEUED.value, dumps(todo), idempotency_key, utcnow()),
    )
    return get_job(db, job_id), True


def run_job(db: Database, settings: Settings, job_id: int, llm_factory: Callable[[], LLM] | None = None) -> IngestJob:
    """Run a queued job to completion. Never raises: failures end up on the job row.

    Per document: parse -> chunk -> index, then (if an LLM is available) summarise. The summary is best-effort and
    never fails the document or the job: without an LLM, or if summarising fails, the document is still `processed`
    and searchable, with summary_status `skipped` / `failed`.
    """
    with _RUN_LOCK:
        job = get_job(db, job_id)
        if job.status != JobStatus.QUEUED:
            return job  # already run / running elsewhere
        db.execute("UPDATE ingest_jobs SET status=?, started_at=? WHERE id=?", (JobStatus.RUNNING.value, utcnow(), job_id))
        try:
            llm, skip_reason = None, "no LLM configured"
            if llm_factory:
                try:
                    llm = llm_factory()
                except LLMError as e:
                    skip_reason = str(e)

            def process(doc_id: int) -> DocResult:
                res = ingest_document(db, settings, doc_id)
                if res.status == DocStatus.PROCESSED:
                    if llm:
                        res.summary_status = summarize_document(db, settings, llm, doc_id)
                    else:
                        mark_skipped(db, [doc_id], skip_reason)
                        res.summary_status = "skipped"
                return res

            with ThreadPoolExecutor(max_workers=settings.ingest_workers) as pool:
                results = list(pool.map(process, job.document_ids))
            failed = [r for r in results if r.status == DocStatus.FAILED]
            status = JobStatus.FAILED if failed else JobStatus.COMPLETED  # fails if ANY document failed
            error = f"{len(failed)} of {len(results)} documents failed" if failed else None
            usage = llm.usage.model_dump() if llm else {}
            if llm:
                usage["cost_usd"] = llm.usage.cost_usd(settings)
            db.execute(
                "UPDATE ingest_jobs SET status=?, results=?, usage=?, error=?, finished_at=? WHERE id=?",
                (status.value, dumps([r.model_dump(mode="json") for r in results]), dumps(usage), error, utcnow(), job_id),
            )
        except Exception as e:
            log.exception("job %s crashed", job_id)
            db.execute(
                "UPDATE ingest_jobs SET status=?, error=?, finished_at=? WHERE id=?",
                (JobStatus.FAILED.value, f"{type(e).__name__}: {e}", utcnow(), job_id),
            )
        return get_job(db, job_id)


def _live_document_ids(db: Database) -> set[int]:
    return {d for r in db.query("SELECT document_ids FROM ingest_jobs WHERE status IN ('queued','running')")
            for d in loads(r["document_ids"], [])}


def submit_for_upload(db: Database, doc_ids: list[int]) -> IngestJob | None:
    """Automatic preprocessing: queue ONE ingest job for the given documents that still need work (new, previously
    failed, or interrupted) and are not already in a live job. Returns None when there is nothing to do."""
    live = _live_document_ids(db)
    todo = []
    for i in dict.fromkeys(doc_ids):
        doc = get_document(db, i)
        if i not in live and doc.status in (DocStatus.UPLOADED, DocStatus.FAILED):
            todo.append(i)
    if not todo:
        return None
    job, _ = submit_job(db, todo, f"auto-{uuid.uuid4()}")
    return job


def recover_stuck_documents(db: Database) -> int:
    """At start-up (no job can be running): documents interrupted mid-chunking or mid-summary, or whose summary failed
    (e.g. quota, wrong model), go back to `uploaded` so they are processed again. Returns how many were reset."""
    with db.conn() as c:
        cur = c.execute("UPDATE documents SET status='uploaded' WHERE status='processing' "
                        "OR (status='processed' AND summary_status IN ('pending', 'failed'))")
        return cur.rowcount


def recover_stuck_jobs(db: Database) -> int:
    """After a crash/restart, jobs left 'running' will never finish; mark them failed so they can be resubmitted."""
    with db.conn() as c:
        cur = c.execute(
            "UPDATE ingest_jobs SET status='failed', error='interrupted by restart', finished_at=? WHERE status='running'",
            (utcnow(),),
        )
        return cur.rowcount
