"""The test-run endpoints and the run manager. The real script is replaced by a tiny fake that prints
lines, waits, honours the stop file and prints a results-file line, so no LLM or database is involved."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import create_app
from app.workers.test_runs import TestRunManager, get_test_run_manager

FAKE = """
import os, sys, time
stop, docs = sys.argv[1], sys.argv[2:]
print("[run] results file=/tmp/TEST-RUN-fake.txt", flush=True)
print("reading " + ", ".join(docs), flush=True)
mode = os.environ.get("FAKE_MODE", "finish")
for i in range(200):
    if os.path.exists(stop):
        print("[run] STOP requested", flush=True)
        sys.exit(0)
    if mode == "finish" and i == 3:
        print("done", flush=True)
        sys.exit(0)
    if mode == "fail" and i == 1:
        sys.exit(3)
    time.sleep(0.05)
"""


@pytest.fixture
def manager(tmp_path: Path, monkeypatch):
    script = tmp_path / "fake_run.py"
    script.write_text(FAKE, encoding="utf-8")
    return TestRunManager(lambda docs, stop: [sys.executable, "-u", str(script), str(stop), *docs],
                          log_dir=tmp_path / "logs", stop_grace_seconds=5)


@pytest.fixture
def client(manager, settings, data_room):
    (data_room / "sub").mkdir()
    for name in ["a.pdf", "sub/b.docx", "sub/model.xlsm", "notes.txt", "~$lock.docx", ".hidden.pdf"]:
        (data_room / name).write_bytes(b"x")
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_test_run_manager] = lambda: manager
    return TestClient(app)


def wait_until(manager, predicate, timeout=15):
    end = time.time() + timeout
    while time.time() < end:
        if predicate(manager.status()):
            return manager.status()
        time.sleep(0.05)
    raise AssertionError(f"timed out; status {manager.status()}")


def test_files_are_listed_as_relative_paths_and_xlsm_is_unchecked_by_default(client):
    files = client.get("/api/test-run/files").json()["files"]
    assert {f["path"]: f["selected_by_default"] for f in files} == {"a.pdf": True, "sub/b.docx": True, "sub/model.xlsm": False}


def test_start_runs_the_chosen_documents_and_updates_serve_the_log_in_pieces(client, manager):
    assert client.get("/api/updates").json()["run"]["state"] == "idle"
    started = client.post("/api/test-run/start", json={"documents": ["sub/b.docx", "a.pdf"]})
    assert started.status_code == 202 and started.json()["state"] == "running"
    wait_until(manager, lambda s: s.state == "finished")
    first = client.get("/api/updates?offset=0").json()
    assert first["run"]["state"] == "finished" and first["run"]["exit_code"] == 0
    assert first["run"]["results_file"] == "/tmp/TEST-RUN-fake.txt"
    assert first["lines"][1] == "reading a.pdf, sub/b.docx" and first["lines"][-1] == "done"
    again = client.get(f"/api/updates?offset={first['next_offset']}").json()
    assert again["lines"] == [] and again["next_offset"] == first["next_offset"]  # nothing new: nothing repeated


def test_only_documents_of_the_data_room_can_be_started_and_at_least_one_is_needed(client):
    assert client.post("/api/test-run/start", json={"documents": ["../secret.pdf"]}).status_code == 422
    assert client.post("/api/test-run/start", json={"documents": ["notes.txt"]}).status_code == 422
    assert client.post("/api/test-run/start", json={"documents": []}).status_code == 422


def test_a_second_start_while_running_is_refused(client, manager, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "wait")
    assert client.post("/api/test-run/start", json={"documents": ["a.pdf"]}).status_code == 202
    assert client.post("/api/test-run/start", json={"documents": ["a.pdf"]}).status_code == 409
    client.post("/api/test-run/stop")
    wait_until(manager, lambda s: s.state == "stopped")


def test_stop_asks_the_run_to_stop_and_it_still_reports_its_results_file(client, manager, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "wait")
    client.post("/api/test-run/start", json={"documents": ["a.pdf"]})
    assert client.post("/api/test-run/stop").json()["state"] == "stopping"
    done = wait_until(manager, lambda s: s.state == "stopped")
    assert done.results_file == "/tmp/TEST-RUN-fake.txt" and done.exit_code == 0
    assert "[run] STOP requested" in client.get("/api/updates").json()["lines"]


def test_a_run_that_exits_with_an_error_is_marked_failed(client, manager, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "fail")
    client.post("/api/test-run/start", json={"documents": ["a.pdf"]})
    assert wait_until(manager, lambda s: s.state == "failed").exit_code == 3


def test_stop_does_nothing_when_nothing_is_running_and_a_new_run_can_follow(client, manager):
    assert client.post("/api/test-run/stop").json()["state"] == "idle"
    client.post("/api/test-run/start", json={"documents": ["a.pdf"]})
    wait_until(manager, lambda s: s.state == "finished")
    second = client.post("/api/test-run/start", json={"documents": ["a.pdf"]})
    assert second.status_code == 202
    wait_until(manager, lambda s: s.state == "finished")


def test_a_process_that_ignores_the_stop_request_is_killed_after_the_grace_period(tmp_path):
    script = tmp_path / "stubborn.py"
    script.write_text("import time\nprint('start', flush=True)\ntime.sleep(60)\n", encoding="utf-8")
    manager = TestRunManager(lambda docs, stop: [sys.executable, "-u", str(script)], log_dir=tmp_path / "logs", stop_grace_seconds=0.5)
    manager.start(["a.pdf"])
    manager.stop()
    assert wait_until(manager, lambda s: s.state == "stopped", timeout=20).exit_code != 0


def test_the_judge_workers_option_reaches_the_script_only_when_it_is_not_the_default(client, manager):
    client.post("/api/test-run/start", json={"documents": ["a.pdf"], "judge_workers": 4})
    wait_until(manager, lambda s: s.state == "finished")
    lines, _ = manager.read_log(0)
    assert any("--judge-workers, 4" in line for line in lines)
    client.post("/api/test-run/start", json={"documents": ["a.pdf"]})
    wait_until(manager, lambda s: s.state == "finished")
    assert not any("--judge-workers" in line for line in manager.read_log(0)[0])
    assert client.post("/api/test-run/start", json={"documents": ["a.pdf"], "judge_workers": 0}).status_code == 422  # at least one
