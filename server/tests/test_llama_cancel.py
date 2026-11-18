"""A stopped run cancels the LlamaExtract job it is waiting for and starts no new one."""
from __future__ import annotations

import httpx
import pytest

from app.llm.llama_extract import LlamaExtractClient, LlamaExtractError


def client_with(handler):
    return LlamaExtractClient("secret-key", project_id="proj-1", http=httpx.Client(base_url="https://x", transport=httpx.MockTransport(handler)), poll_seconds=0)


def test_a_stop_while_a_job_runs_cancels_it_and_ends_the_wait():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path, request.url.params.get("project_id")))
        if request.method == "POST" and request.url.path.endswith("/cancel"):
            return httpx.Response(200, json={"id": "ext-1", "status": "CANCELLED"})
        return httpx.Response(200, json={"id": "ext-1", "status": "RUNNING"})

    polls = []

    def should_stop():
        polls.append(1)
        return len(polls) > 2  # two polls pass, then the run is stopped

    with pytest.raises(LlamaExtractError) as err:
        client_with(handler).wait("ext-1", should_stop)
    assert "cancelled" in str(err.value)
    assert seen[-1] == ("POST", "/api/v2/extract/ext-1/cancel", "proj-1")
    assert [m for m, _p, _q in seen].count("GET") == 2


def test_a_stopped_run_starts_no_job_and_uploads_nothing(tmp_path):
    seen = []
    client = client_with(lambda request: seen.append(request.url.path) or httpx.Response(200, json={"id": "x"}))
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF")
    with pytest.raises(LlamaExtractError):
        client.extract(pdf, {"type": "object"}, should_stop=lambda: True)
    assert seen == []


def test_a_cancel_the_service_refuses_is_not_an_error():
    client = client_with(lambda request: httpx.Response(409, text="job already completed"))
    assert client.cancel("ext-1") is False


def test_without_a_stop_the_wait_is_unchanged():
    answers = iter(["RUNNING", "COMPLETED"])
    client = client_with(lambda request: httpx.Response(200, json={"id": "ext-1", "status": next(answers)}))
    assert client.wait("ext-1", lambda: False)["status"] == "COMPLETED"
