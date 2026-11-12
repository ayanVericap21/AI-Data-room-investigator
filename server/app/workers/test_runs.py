"""Starts, stops and follows one test run: ``scripts/investigator_run.py`` (or ``scripts/ledger_run.py`` with DATAROOM_PIPELINE=classic) run as a child process.

The run uses the script's own defaults, so whatever logic and configuration the script holds is what the API
runs; the API only chooses which documents are read. One run at a time. The child writes its output to a
log file, and ``read_log`` returns the new lines since a byte offset (what the ``/api/updates`` endpoint
serves). Stopping first asks the script to stop at its next safe point (a stop file), so it still writes its
results file with what was done so far; only if it has not exited after a grace period is the process tree killed.
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from pydantic import BaseModel

from app.core.config import SERVER_DIR
from app.core.exceptions import ConflictError

logger = logging.getLogger(__name__)

# The pipeline the page runs: the investigator (pipeline v2) by default on this branch; DATAROOM_PIPELINE=classic runs the fact-ledger script instead.
SCRIPTS = {"investigator": SERVER_DIR / "scripts" / "investigator_run.py", "classic": SERVER_DIR / "scripts" / "ledger_run.py"}
SCRIPT = SCRIPTS["investigator"]
RESULTS_LINE = re.compile(r"\[run\] results file=(.+)")


class RunStatus(BaseModel):
    run_id: str | None = None
    state: str = "idle"  # idle | running | stopping | finished | stopped | failed
    documents: list[str] = []
    started_at: datetime | None = None
    finished_at: datetime | None = None
    exit_code: int | None = None
    results_file: str | None = None  # the file the script writes when it ends, however it ends


CommandFactory = Callable[[list[str], Path], list[str]]


def default_command(documents: list[str], stop_file: Path) -> list[str]:
    script = SCRIPTS.get(os.environ.get("DATAROOM_PIPELINE", "investigator").strip().lower(), SCRIPT)
    command = [sys.executable, "-u", str(script), "--config", "ui-run", "--stop-file", str(stop_file)]
    for document in documents:
        command += ["--only", document]
    return command


class TestRunManager:
    __test__ = False  # not a pytest class

    def __init__(self, command_factory: CommandFactory = default_command, log_dir: Path | None = None,
                 stop_grace_seconds: float = 90.0):
        self._command_factory = command_factory
        self._dir = log_dir or Path(tempfile.gettempdir()) / "dataroom_runs"
        self._grace = stop_grace_seconds
        self._lock = threading.Lock()
        self._status = RunStatus()
        self._process: subprocess.Popen | None = None
        self._log_path: Path | None = None
        self._stop_path: Path | None = None
        self._stop_requested = False

    # -- state ---------------------------------------------------------------
    def status(self) -> RunStatus:
        with self._lock:
            return self._status.model_copy()

    def _active(self) -> bool:
        return self._status.state in ("running", "stopping")

    # -- start / stop ----------------------------------------------------------
    def start(self, documents: list[str], judge_workers: int = 1) -> RunStatus:
        with self._lock:
            if self._active():
                raise ConflictError("A test run is already in progress.")
            self._dir.mkdir(parents=True, exist_ok=True)
            run_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
            self._log_path, self._stop_path = self._dir / f"{run_id}.log", self._dir / f"{run_id}.stop"
            self._stop_requested = False
            env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
            flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
            log = open(self._log_path, "wb")
            try:
                self._process = subprocess.Popen(
                    self._command_factory(documents, self._stop_path) + (['--judge-workers', str(judge_workers)] if judge_workers != 1 else []),
                    cwd=str(SERVER_DIR), env=env, stdout=log,
                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, creationflags=flags)
            finally:
                log.close()  # the child holds its own handle
            self._status = RunStatus(run_id=run_id, state="running", documents=list(documents),
                                     started_at=datetime.now(timezone.utc))
            threading.Thread(target=self._watch, args=(self._process, run_id), daemon=True).start()
            return self._status.model_copy()

    def stop(self) -> RunStatus:
        with self._lock:
            if not self._active() or self._process is None or self._stop_path is None:
                return self._status.model_copy()
            self._stop_requested = True
            self._status.state = "stopping"
            self._stop_path.write_text("stop", encoding="utf-8")
            process = self._process
            timer = threading.Timer(self._grace, self._kill, args=(process,))
            timer.daemon = True
            timer.start()
            return self._status.model_copy()

    @staticmethod
    def _kill(process: subprocess.Popen) -> None:
        if process.poll() is not None:
            return
        logger.warning("Test run did not stop in time; killing its process tree", extra={"pid": process.pid})
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True)
        else:
            process.kill()

    def _watch(self, process: subprocess.Popen, run_id: str) -> None:
        code = process.wait()
        with self._lock:
            if self._status.run_id != run_id:
                return
            self._status.exit_code = code
            self._status.finished_at = datetime.now(timezone.utc)
            self._status.state = "stopped" if self._stop_requested else "finished" if code == 0 else "failed"
            self._status.results_file = self._find_results_file()

    def _find_results_file(self) -> str | None:
        if self._log_path is None or not self._log_path.exists():
            return None
        found = RESULTS_LINE.findall(self._log_path.read_text(encoding="utf-8", errors="replace"))
        return found[-1].strip() if found else None

    # -- log -------------------------------------------------------------------
    def read_log(self, offset: int = 0) -> tuple[list[str], int]:
        """New complete lines after byte ``offset`` and the offset to ask for next time. A last line that is
        still being written is held back until it is complete (or the run has ended)."""
        with self._lock:
            path, ended = self._log_path, not self._active()
        if path is None or not path.exists():
            return [], 0
        size = path.stat().st_size
        offset = max(0, min(offset, size))
        with open(path, "rb") as fh:
            fh.seek(offset)
            data = fh.read()
        if not data:
            return [], offset
        if not ended and not data.endswith(b"\n"):
            cut = data.rfind(b"\n") + 1
            data = data[:cut]
        text = data.decode("utf-8", errors="replace")
        return text.splitlines(), offset + len(data)


_manager = TestRunManager()


def get_test_run_manager() -> TestRunManager:
    return _manager
