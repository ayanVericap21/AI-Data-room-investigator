"""Minimal in-process background jobs for long-running pipeline steps (document
processing, claim extraction).

One job per name may run at a time (duplicate starts get a 409). State lives in memory, so
it is lost on restart; the durable results are the rows the job writes to PostgreSQL.
The investigation worker (Stage 3) persists its state in PostgreSQL instead.
"""
from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Callable

from pydantic import BaseModel

from app.core.exceptions import ConflictError

logger = logging.getLogger(__name__)


class JobState(BaseModel):
    name: str
    state: str = "idle"  # idle | running | completed | failed
    message: str = ""
    started_at: datetime | None = None
    finished_at: datetime | None = None
    result: dict | None = None
    error: str | None = None


ProgressFn = Callable[[str, dict], None]


class JobManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, JobState] = {}
        self._threads: dict[str, threading.Thread] = {}

    def get(self, name: str) -> JobState:
        with self._lock:
            return self._jobs.get(name, JobState(name=name)).model_copy()

    def all(self, names: list[str]) -> list[JobState]:
        return [self.get(n) for n in names]

    def start(self, name: str, fn: Callable[[ProgressFn], dict]) -> JobState:
        with self._lock:
            current = self._jobs.get(name)
            if current is not None and current.state == "running":
                raise ConflictError(f"Job '{name}' is already running")
            state = JobState(name=name, state="running", message="Started", started_at=datetime.now(timezone.utc))
            self._jobs[name] = state
            thread = threading.Thread(target=self._run, args=(name, fn), name=f"job-{name}", daemon=True)
            self._threads[name] = thread
        thread.start()
        return state.model_copy()

    def wait(self, name: str, timeout: float = 30) -> None:
        """Block until the job's thread finishes (used by tests)."""
        thread = self._threads.get(name)
        if thread is not None:
            thread.join(timeout)

    def _progress(self, name: str) -> ProgressFn:
        def update(message: str, _details: dict | None = None) -> None:
            with self._lock:
                job = self._jobs[name]
                job.message = message
        return update

    def _run(self, name: str, fn: Callable[[ProgressFn], dict]) -> None:
        try:
            result = fn(self._progress(name))
            with self._lock:
                job = self._jobs[name]
                job.state, job.result, job.finished_at = "completed", result, datetime.now(timezone.utc)
        except Exception as exc:  # job boundary: record and surface, never crash the server
            logger.exception("Background job failed", extra={"job": name})
            with self._lock:
                job = self._jobs[name]
                job.state, job.error, job.finished_at = "failed", str(exc), datetime.now(timezone.utc)
                job.message = f"Failed: {exc}"


_manager = JobManager()


def get_job_manager() -> JobManager:
    return _manager
