"""Upload -> automatic preprocessing -> MCP status: no manual 'process' step anywhere."""
import pytest
from fastapi.testclient import TestClient

from app import jobs
from app.api.main import create_app
from app.documents import save_upload
from app.mcp_server.client import LocalToolClient
from app.models import DocSummary, JobStatus
from app.tools import TOOL_NAMES, DataRoomTools
from tests.test_agent import FakeLLM
from tests.conftest import make_pdf


def _client(settings, **kw):
    return TestClient(create_app(settings, run_inline=True, **kw))


def _up(c, *files):
    return c.post("/api/v1/documents", files=[("files", f) for f in files]).json()


# ---------- upload triggers preprocessing ----------
def test_batch_is_one_job_and_failures_are_isolated(settings):
    c = _client(settings)
    out = _up(c, ("good.txt", b"The supplier guarantees only 25,000 tonnes per year.", "text/plain"),
              ("broken.pdf", b"this is not a pdf", "application/pdf"), ("nope.exe", b"zzz", "application/x"))
    assert out[0]["job_id"] == out[1]["job_id"] and out[2]["job_id"] is None and out[2]["error"]
    docs = {d["filename"]: d for d in c.get("/api/v1/documents").json()}
    assert docs["good.txt"]["status"] == "processed" and docs["good.txt"]["chunk_count"] >= 1
    assert docs["broken.pdf"]["status"] == "failed" and docs["broken.pdf"]["failure_code"]


def test_summaries_are_created_during_preprocessing(settings):
    llm = FakeLLM({"summarize": [DocSummary(overview="A supplier quotation.", key_topics=["guarantee"])]})
    c = _client(settings, llm_factory=lambda: llm)
    out = _up(c, ("q.txt", b"The supplier guarantees only 25,000 tonnes per year.", "text/plain"))
    doc = c.get(f"/api/v1/documents/{out[0]['document']['id']}").json()
    assert doc["summary_status"] == "done" and doc["summary"]["overview"] == "A supplier quotation."
    assert c.get(f"/api/v1/jobs/{out[0]['job_id']}").json()["usage"]["calls"] == 1


def test_reuploading_processed_files_does_nothing_but_failed_ones_are_retried(settings):
    c = _client(settings)
    first = _up(c, ("a.txt", b"some capacity text here", "text/plain"), ("bad.pdf", b"not a pdf", "application/pdf"))
    again = _up(c, ("a.txt", b"some capacity text here", "text/plain"), ("bad.pdf", b"not a pdf", "application/pdf"))
    assert again[0]["duplicate"] and again[0]["job_id"] is None            # already processed: nothing to do
    assert again[1]["duplicate"] and again[1]["job_id"] not in (None, first[1]["job_id"])  # failed: tried again


def test_folder_paths_survive_preprocessing(settings):
    c = _client(settings)
    _up(c, ("room/2024/report.txt", b"Report 2024: capacity 25,000 tonnes.", "text/plain"),
        ("room/2025/report.txt", b"Report 2025: capacity 30,000 tonnes.", "text/plain"))
    docs = c.get("/api/v1/documents").json()
    assert {d["filename"] for d in docs} == {"room/2024/report.txt", "room/2025/report.txt"}
    assert all(d["status"] == "processed" for d in docs)


# ---------- start-up recovery: nothing stays 'uploaded' ----------
def test_pending_documents_are_processed_when_the_app_starts(settings, db):
    stuck_new, _ = save_upload(db, settings, "never_processed.txt", b"uploaded but nobody processed me")
    stuck_mid, _ = save_upload(db, settings, "mid_chunking.txt", b"interrupted while chunking this one")
    db.execute("UPDATE documents SET status='processing' WHERE id=?", (stuck_mid.id,))
    stuck_sum, _ = save_upload(db, settings, "mid_summary.txt", b"chunked fine, summary never finished")
    jobs.run_job(db, settings, jobs.submit_job(db, [stuck_sum.id], "x")[0].id)      # processed, summary 'skipped'
    db.execute("UPDATE documents SET summary_status='pending' WHERE id=?", (stuck_sum.id,))
    c = _client(settings)                                                             # app start = recovery + processing
    docs = {d["filename"]: d for d in c.get("/api/v1/documents").json()}
    assert {d["status"] for d in docs.values()} == {"processed"}
    assert docs["mid_summary.txt"]["summary_status"] != "pending"
    assert all(d["chunk_count"] >= 1 for d in docs.values())


def test_failed_summaries_are_retried_at_start_up(settings, db):
    d, _ = save_upload(db, settings, "q.txt", b"The supplier guarantees only 25,000 tonnes per year.")
    jobs.run_job(db, settings, jobs.submit_job(db, [d.id], "x")[0].id, lambda: FakeLLM({}))   # summarising fails
    assert db.one("SELECT summary_status FROM documents WHERE id=?", (d.id,))["summary_status"] == "failed"
    llm = FakeLLM({"summarize": [DocSummary(overview="Now it works.")]})                      # e.g. quota is back / model fixed
    c = _client(settings, llm_factory=lambda: llm)
    doc = c.get(f"/api/v1/documents/{d.id}").json()
    assert doc["summary_status"] == "done" and doc["summary"]["overview"] == "Now it works."


def test_submit_for_upload_skips_documents_already_in_a_live_job(db, settings):
    a, _ = save_upload(db, settings, "a.txt", b"first document content")
    b, _ = save_upload(db, settings, "b.txt", b"second document content")
    j1, _ = jobs.submit_job(db, [a.id], "live")           # queued, not run yet
    j2 = jobs.submit_for_upload(db, [a.id, b.id])
    assert j2.document_ids == [b.id]
    assert jobs.submit_for_upload(db, [a.id, b.id]) is None
    assert jobs.get_job(db, j1.id).status == JobStatus.QUEUED


# ---------- MCP server status ----------
def test_mcp_status_reports_tools_and_what_it_serves(settings, db):
    llm = FakeLLM({"summarize": [DocSummary(overview="s")]})
    c = _client(settings, llm_factory=lambda: llm,
                tool_client_factory=lambda ids: LocalToolClient(DataRoomTools(db, ids)))
    _up(c, ("a.txt", b"first document content", "text/plain"), ("b.txt", b"second document content", "text/plain"))
    m = c.get("/api/v1/mcp").json()
    assert m["ready"] and m["transport"] == "stdio" and m["error"] is None
    assert [t["name"] for t in m["tools"]] == TOOL_NAMES and all(t["description"] for t in m["tools"])
    assert m["documents_served"] == 2 and m["documents_with_summary"] == 2


def test_mcp_status_failure_is_reported_not_raised(settings):
    def broken(ids):
        raise RuntimeError("server failed to start")

    c = _client(settings, tool_client_factory=broken)
    m = c.get("/api/v1/mcp")
    assert m.status_code == 200 and m.json()["ready"] is False and "failed to start" in m.json()["error"]


def test_mcp_status_with_the_real_server(settings):
    """No injected client: spawns the actual MCP server over stdio and lists its tools."""
    c = _client(settings)
    _up(c, ("a.txt", b"first document content", "text/plain"))
    m = c.get("/api/v1/mcp").json()
    assert m["ready"] and [t["name"] for t in m["tools"]] == TOOL_NAMES and m["documents_served"] == 1
    assert c.get("/api/v1/mcp").json()["ready"]            # served from the short cache
