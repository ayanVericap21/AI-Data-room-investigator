import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.mcp_server.client import LocalToolClient, MCPToolClient, ToolError
from app.tools import DataRoomTools


@pytest.fixture
def client(settings):
    return TestClient(create_app(settings, run_inline=True))


def _upload(client, files):
    return client.post("/api/v1/documents", files=[("files", f) for f in files])


def test_upload_preprocesses_automatically(client):
    r = _upload(client, [("a.txt", b"Capacity is 30,000 tonnes per year.", "text/plain")])
    assert r.status_code == 201
    item = r.json()[0]
    doc = item["document"]
    assert item["job_id"] is not None                                  # no separate "process" step
    assert client.get(f"/api/v1/jobs/{item['job_id']}").json()["status"] == "completed"
    assert client.get(f"/api/v1/documents/{doc['id']}").json()["status"] == "processed"
    chunks = client.get(f"/api/v1/documents/{doc['id']}/chunks").json()
    assert chunks and "30,000" in chunks[0]["text"]


def test_upload_batch_reports_per_file_errors(client):
    r = _upload(client, [("ok.txt", b"fine content here", "text/plain"), ("bad.exe", b"zzz", "application/x")])
    body = r.json()
    assert body[0]["document"] and body[1]["error"]


def test_api_error_codes(client):
    assert client.post("/api/v1/jobs", json={"document_ids": [99]}).status_code == 404
    assert client.post("/api/v1/jobs", json={"document_ids": []}).status_code == 422
    assert client.get("/api/v1/jobs/99").status_code == 404
    assert client.get("/api/v1/documents/99").status_code == 404
    assert client.get("/api/v1/documents/99/chunks").status_code == 404
    doc = _upload(client, [("a.txt", b"hello there friend", "text/plain")]).json()[0]["document"]
    # already preprocessed automatically by the upload -> a manual job has nothing to do
    assert client.post("/api/v1/jobs", json={"document_ids": [doc["id"]], "idempotency_key": "k2"}).status_code == 409


def test_local_tool_client_rejects_unknown_tool(db):
    c = LocalToolClient(DataRoomTools(db))
    with pytest.raises(ToolError):
        c.call("rm_rf", {})
    with pytest.raises(ToolError):
        c.call("search_chunks", {"nope": 1})


def test_mcp_server_end_to_end(client, settings):
    """Spawns the real MCP server over stdio and calls its tools against the ingested data room."""
    doc = _upload(client, [("s.txt", b"The supplier guarantees only 25,000 tonnes per year.", "text/plain")]).json()[0]["document"]
    mc = MCPToolClient(settings.database_url, settings.db_schema, [doc["id"]])
    try:
        from app.tools import TOOL_NAMES
        listed = mc.list_tools()                                   # real protocol round-trip
        assert [t["name"] for t in listed] == TOOL_NAMES and all(t["description"] for t in listed)
        assert mc.call("list_documents", {})["documents"][0]["filename"] == "s.txt"
        hit = mc.call("search_chunks", {"query": "supplier tonnes"})["results"][0]
        assert hit["document_id"] == doc["id"]
        assert mc.call("verify_quote", {"chunk_id": hit["chunk_id"], "quote": "guarantees only 25,000 tonnes"})["found"]
        with pytest.raises(ToolError):
            mc.call("nonexistent", {})
    finally:
        mc.close()
