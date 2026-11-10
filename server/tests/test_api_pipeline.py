"""API-level tests for processing, claims, candidates and jobs (SQLite + mocks)."""
import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_embedder, get_llm, get_session_factory
from app.core.config import get_settings
from app.core.database import get_session
from app.llm.embeddings import MockEmbeddingProvider
from app.llm.mock import MockLLMClient
from app.main import create_app
from app.workers.jobs import JobManager, get_job_manager
from tests.conftest import make_pdf


@pytest.fixture
def ctx(session_factory, settings, data_room):
    app = create_app()
    jobs = JobManager()
    llm = MockLLMClient()

    def _session():
        with session_factory() as s:
            yield s

    app.dependency_overrides[get_session] = _session
    app.dependency_overrides[get_session_factory] = lambda: session_factory
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_embedder] = lambda: MockEmbeddingProvider(dimensions=32)
    app.dependency_overrides[get_llm] = lambda: llm
    app.dependency_overrides[get_job_manager] = lambda: jobs
    return TestClient(app), jobs, llm, data_room


def test_process_then_extract_then_list_claims(ctx, session_factory):
    client, jobs, llm, room = ctx
    make_pdf(room / "a.pdf", ["Annual production capacity is 30000 tonnes."])
    client.post("/api/documents/scan")
    assert client.post("/api/documents/process").status_code == 202
    jobs.wait("process_documents")
    job = {j["name"]: j for j in client.get("/api/jobs").json()}["process_documents"]
    assert job["state"] == "completed", job
    assert client.get("/api/documents").json()["items"][0]["processing_status"] == "processed"

    assert client.get("/api/claims").json()["total"] == 0
    from app.models import DocumentChunk
    with session_factory() as s:
        cid = s.query(DocumentChunk).first().id
    llm.responses.append({"claims": [{"chunk_id": cid, "subject": "Production", "metric": "Annual capacity",
                                      "value": 30000, "unit": "tonnes", "period": "annual",
                                      "claim_text": "Capacity is 30000 tonnes."}]})
    assert client.post("/api/claims/extract").status_code == 202
    jobs.wait("extract_claims")
    ext = {j["name"]: j for j in client.get("/api/jobs").json()}["extract_claims"]
    assert ext["state"] == "completed" and ext["result"]["claims_created"] == 1
    claims = client.get("/api/claims").json()
    assert claims["total"] == 1 and claims["items"][0]["source"]["page_number"] == 1
    chunk = client.get(f"/api/chunks/{cid}").json()
    assert "30000 tonnes" in chunk["content"] and chunk["filename"] == "a.pdf"
    assert client.get("/api/claims/candidates").json() == {"items": [], "total": 0}
    assert client.get("/api/chunks/chk_nope").status_code == 404


def test_duplicate_job_start_conflicts(ctx):
    client, jobs, *_ = ctx
    import threading
    gate = threading.Event()
    jobs.start("process_documents", lambda p: gate.wait(5) or {})
    r = client.post("/api/documents/process")
    assert r.status_code == 409 and r.json()["error"] == "conflict"
    gate.set()
    jobs.wait("process_documents")


def test_missing_llm_key_gives_clear_api_error(session_factory, settings):
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: settings  # default settings: google, no key
    app.dependency_overrides[get_session] = lambda: iter([None])
    r = TestClient(app).post("/api/claims/extract")
    assert r.status_code == 503 and r.json()["error"] == "llm_not_configured"
    assert "LLM_API_KEY" in r.json()["message"]


def test_failed_job_is_reported(ctx):
    client, jobs, *_ = ctx
    jobs.start("extract_claims", lambda p: (_ for _ in ()).throw(RuntimeError("boom")))
    jobs.wait("extract_claims")
    job = {j["name"]: j for j in client.get("/api/jobs").json()}["extract_claims"]
    assert job["state"] == "failed" and "boom" in job["error"]
