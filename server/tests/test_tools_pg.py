"""The search tool and a full agent run against real PostgreSQL + pgvector (skipped if unreachable).

Uses the throwaway-database fixtures from test_retrieval_pg and the mock embedding provider,
so no network or API key is needed."""
from __future__ import annotations

from sqlalchemy import select

from app.agent import run_investigation
from app.llm.embeddings import MockEmbeddingProvider
from app.llm.mock import MockLLMClient
from app.models import Document, DocumentChunk, Evidence, Finding
from app.services.processing_service import ProcessingService
from app.services.retrieval_service import RetrievalService
from app.tools import build_registry
from tests.test_retrieval_pg import pg_session, pg_url, setup_room  # noqa: F401  (fixtures and helper)


def processed_room(pg_session, data_room):
    svc = setup_room(pg_session, data_room)
    ProcessingService(pg_session, svc, MockEmbeddingProvider()).process_pending()
    return RetrievalService(pg_session, MockEmbeddingProvider())


def test_search_tool_returns_real_hits_with_exact_locators(pg_session, data_room):
    retrieval = processed_room(pg_session, data_room)
    registry = build_registry(pg_session, retrieval, None)
    result = registry.execute("search", {"query": "annual production capacity tonnes", "top_k": 3})
    assert result.ok and result.output.hits
    top = result.output.hits[0]
    assert "30000 tonnes" in top.content and top.chunk_id and top.locator
    assert [h.score for h in result.output.hits] == sorted((h.score for h in result.output.hits), reverse=True)

    only_docx = pg_session.scalar(select(Document.id).where(Document.file_type == "docx"))
    scoped = registry.execute("search", {"query": "capacity", "document_ids": [only_docx]}).output
    assert {h.document_id for h in scoped.hits} == {only_docx}


def test_agent_searches_then_saves_a_finding_with_verified_quotes(pg_session, data_room):
    retrieval = processed_room(pg_session, data_room)
    pdf_chunk = pg_session.scalar(select(DocumentChunk).join(Document).where(Document.file_type == "pdf"))
    docx_chunk = pg_session.scalar(
        select(DocumentChunk).join(Document).where(Document.file_type == "docx", DocumentChunk.content.like("%30000 tonnes%")))
    finding = dict(
        title="Capacity statements agree", description="Both documents give 30000 tonnes.", classification="consistent",
        severity="informational", confidence=0.9, evidence_status="adequate",
        evidence=[
            dict(chunk_id=pdf_chunk.id, quoted_text="production capacity is 30000 tonnes", role="supporting",
                 relevance_reason="Capacity statement", support_assessment="States 30000 tonnes"),
            dict(chunk_id=docx_chunk.id, quoted_text="production capacity is estimated at 30000 tonnes", role="supporting",
                 relevance_reason="Capacity statement", support_assessment="States 30000 tonnes"),
        ],
    )
    llm = MockLLMClient([
        {"action": "call_tool", "tool": "search", "arguments": {"query": "annual production capacity"}, "rationale": "find capacity"},
        {"action": "call_tool", "tool": "submit_finding", "arguments": finding, "rationale": "both agree"},
        {"action": "finish", "rationale": "done", "summary": "One consistent statement."},
    ])
    result = run_investigation(pg_session, llm, retrieval, name="pg run", max_steps=5, max_finish_refusals=0)

    assert result.termination == "finished" and len(result.finding_ids) == 1
    assert [s.tool for s in result.steps[:2]] == ["search", "submit_finding"] and all(s.ok for s in result.steps)
    assert "hits" in result.steps[0].result_summary
    stored = pg_session.scalars(select(Evidence).where(Evidence.finding_id == result.finding_ids[0])).all()
    assert {e.chunk_id for e in stored} == {pdf_chunk.id, docx_chunk.id}
    assert all(e.quoted_text in pg_session.get(DocumentChunk, e.chunk_id).content for e in stored)
    assert pg_session.get(Finding, result.finding_ids[0]).verification_status == "verified"


# ---- per-document latency of each part (embedding only runs on PostgreSQL) ----
def test_processing_records_time_and_embedding_calls_per_document(pg_session, data_room):
    svc = setup_room(pg_session, data_room)
    results = ProcessingService(pg_session, svc, MockEmbeddingProvider()).process_pending()
    for r in results:
        assert {"extract", "chunk", "save", "embed"} <= set(r.timings_ms) and r.embed_calls >= 1 and r.embed_failed_attempts == 0
    pdf = next(r for r in results if r.filename.endswith(".pdf"))
    assert {"extract.open", "extract.tables", "extract.text"} <= set(pdf.timings_ms)


def test_a_failing_embedding_still_reports_its_time_and_failed_attempts(pg_session, data_room):
    from app.llm.errors import LLMTransientError

    class Flaky(MockEmbeddingProvider):
        def _embed_batch(self, texts, task):
            raise LLMTransientError("rate limited")

    svc = setup_room(pg_session, data_room)
    result = ProcessingService(pg_session, svc, Flaky(max_attempts=2, sleep=lambda _s: None)).process_pending()[0]
    assert result.status == "failed" and "embed" in result.timings_ms and "save" in result.timings_ms
    assert result.embed_calls == 2 and result.embed_failed_attempts == 2
