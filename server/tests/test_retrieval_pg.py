"""Integration tests against real PostgreSQL + pgvector (skipped if unreachable).

Each test module run creates a throwaway database, applies the real Alembic migrations,
and drops it afterwards, so the dev database is never touched.
"""
from __future__ import annotations

import uuid

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, func, select, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker

from app.core.config import Settings, get_settings
from app.llm.embeddings import MockEmbeddingProvider
from app.llm.errors import LLMTransientError
from app.models import Document, DocumentChunk
from app.services.document_service import DocumentService
from app.services.processing_service import ProcessingService
from app.services.retrieval_service import RetrievalService
from tests.conftest import SERVER_DIR, make_docx, make_pdf


@pytest.fixture(scope="module")
def pg_url():
    admin_url = make_url(get_settings().database_url)
    name = f"data_room_test_{uuid.uuid4().hex[:8]}"
    try:
        admin = create_engine(admin_url.set(database="postgres"), isolation_level="AUTOCOMMIT")
        with admin.connect() as c:
            c.execute(text(f'CREATE DATABASE "{name}"'))
    except Exception as exc:
        pytest.skip(f"PostgreSQL not reachable: {exc.__class__.__name__}")
    url = admin_url.set(database=name).render_as_string(hide_password=False)
    cfg = Config(str(SERVER_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(SERVER_DIR / "alembic"))
    cfg.set_main_option("sqlalchemy.url", url)
    command.upgrade(cfg, "head")
    yield url
    with admin.connect() as c:
        c.execute(text(f'DROP DATABASE "{name}" WITH (FORCE)'))
    admin.dispose()


@pytest.fixture
def pg_session(pg_url):
    engine = create_engine(pg_url)
    with sessionmaker(bind=engine, expire_on_commit=False)() as s:
        yield s
        s.rollback()
        for table in ("evidence", "finding_claims", "findings", "claims", "document_chunks", "documents"):
            s.execute(text(f"DELETE FROM {table}"))
        s.commit()
    engine.dispose()


def setup_room(pg_session, data_room):
    make_pdf(data_room / "capacity.pdf", ["Annual production capacity is 30000 tonnes at the plant."])
    make_docx(data_room / "ops.docx")
    settings = Settings(data_room_path=data_room)
    svc = DocumentService(pg_session, settings)
    svc.scan()
    return svc


def test_process_embeds_chunks_and_search_returns_source_metadata(pg_session, data_room):
    svc = setup_room(pg_session, data_room)
    results = ProcessingService(pg_session, svc, MockEmbeddingProvider()).process_pending()
    assert {r.status for r in results} == {"processed"}
    missing = pg_session.scalar(select(func.count()).select_from(DocumentChunk).where(DocumentChunk.embedding.is_(None)))
    assert missing == 0

    hits = RetrievalService(pg_session, MockEmbeddingProvider()).search("annual production capacity tonnes", top_k=3)
    assert hits[0].filename == "capacity.pdf" and hits[0].page_number == 1
    assert hits[0].score > hits[-1].score or len(hits) == 1
    assert "30000 tonnes" in hits[0].content


def test_document_filter_and_exclusion(pg_session, data_room):
    svc = setup_room(pg_session, data_room)
    ProcessingService(pg_session, svc, MockEmbeddingProvider()).process_pending()
    retrieval = RetrievalService(pg_session, MockEmbeddingProvider())
    docx_id = pg_session.scalar(select(Document.id).where(Document.file_type == "docx"))
    only = retrieval.search("production capacity", top_k=10, document_ids=[docx_id])
    assert only and {h.document_id for h in only} == {docx_id}
    rest = retrieval.search("production capacity", top_k=10, exclude_document_ids=[docx_id])
    assert docx_id not in {h.document_id for h in rest}


def test_reprocessing_is_idempotent(pg_session, data_room):
    svc = setup_room(pg_session, data_room)
    ProcessingService(pg_session, svc, MockEmbeddingProvider()).process_pending()
    ids_before = set(pg_session.scalars(select(DocumentChunk.id)))
    again = ProcessingService(pg_session, svc, MockEmbeddingProvider()).process_pending()
    assert again == []  # nothing pending
    doc = pg_session.scalars(select(Document)).first()
    assert ProcessingService(pg_session, svc, MockEmbeddingProvider()).process_document(doc).skipped
    assert set(pg_session.scalars(select(DocumentChunk.id))) == ids_before


def test_embedding_failure_keeps_chunks_then_retry_embeds_only_missing(pg_session, data_room):
    svc = setup_room(pg_session, data_room)

    class Flaky(MockEmbeddingProvider):
        def _embed_batch(self, texts, task):
            raise LLMTransientError("rate limited")

    failing = Flaky(max_attempts=2, sleep=lambda _s: None)
    results = ProcessingService(pg_session, svc, failing).process_pending()
    assert {r.status for r in results} == {"failed"}
    chunk_ids = set(pg_session.scalars(select(DocumentChunk.id)))
    assert chunk_ids  # text was kept

    retry = ProcessingService(pg_session, svc, MockEmbeddingProvider()).process_pending()
    assert {r.status for r in retry} == {"processed"}
    assert set(pg_session.scalars(select(DocumentChunk.id))) == chunk_ids  # same rows, not re-created


def test_corrupt_file_marks_failed_without_breaking_others(pg_session, data_room):
    (data_room / "bad.pdf").write_bytes(b"garbage")
    svc = setup_room(pg_session, data_room)
    results = {r.filename: r for r in ProcessingService(pg_session, svc, MockEmbeddingProvider()).process_pending()}
    assert results["bad.pdf"].status == "failed" and results["bad.pdf"].error
    assert results["capacity.pdf"].status == "processed"


def test_search_on_empty_query_returns_nothing(pg_session):
    assert RetrievalService(pg_session, MockEmbeddingProvider()).search("   ") == []
