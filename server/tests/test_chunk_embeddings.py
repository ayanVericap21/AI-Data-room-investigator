"""A ledger run may save chunks without vectors; whatever reads them (the agent's search) embeds the missing ones first, so nothing is left incomplete.

Runs on PostgreSQL (the throwaway database of test_retrieval_pg): there an empty vector is a real NULL; SQLite stores a JSON null the query cannot see."""
import pytest

from app.core.config import Settings
from app.llm.embeddings import MockEmbeddingProvider
from app.llm.errors import LLMTransientError
from app.models import Document, DocumentChunk
from app.models.base import ProcessingStatus
from app.services.chunk_embeddings import embed_missing
from app.services.chunking import embedding_text
from app.services.document_service import DocumentService
from app.services.processing_service import ProcessingService
from app.services.retrieval_service import RetrievalService
from tests.conftest import make_pdf
from tests.test_retrieval_pg import pg_session, pg_url  # noqa: F401  (fixtures)


class CountingEmbedder(MockEmbeddingProvider):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.texts: list[str] = []

    def _embed_batch(self, texts, task):
        self.texts += texts
        return super()._embed_batch(texts, task)


def room_with_two_documents(session, data_room):
    make_pdf(data_room / "a.pdf", ["alpha text about warehouse capacity"])
    make_pdf(data_room / "b.pdf", ["beta text about the lease term"])
    svc = DocumentService(session, Settings(data_room_path=data_room))
    svc.scan()
    return svc


def vectors(pg_session):
    return {c.document_id: c.embedding for c in pg_session.query(DocumentChunk).all()}


def test_a_run_that_skips_chunk_vectors_saves_the_chunks_and_marks_the_documents_processed(pg_session, data_room):
    svc = room_with_two_documents(pg_session, data_room)
    embedder = CountingEmbedder()
    results = ProcessingService(pg_session, svc, embedder, embed_chunks=False).process_pending()
    assert [r.status for r in results] == [ProcessingStatus.processed.value] * 2 and all(r.chunks == 1 for r in results)
    assert embedder.texts == [] and all(c.embedding is None for c in pg_session.query(DocumentChunk).all())  # nothing was embedded


def test_processing_again_without_vectors_does_not_parse_or_embed_anything(pg_session, data_room):
    svc = room_with_two_documents(pg_session, data_room)
    ProcessingService(pg_session, svc, CountingEmbedder(), embed_chunks=False).process_pending()
    embedder = CountingEmbedder()
    again = ProcessingService(pg_session, svc, embedder, embed_chunks=False)
    assert all(again.process_document(d).skipped for d in pg_session.query(Document).all()) and embedder.texts == []


def test_embedding_what_is_missing_gives_every_chunk_the_same_vector_processing_would_have_given(pg_session, data_room):
    svc = room_with_two_documents(pg_session, data_room)
    ProcessingService(pg_session, svc, MockEmbeddingProvider(), embed_chunks=False).process_pending()
    embedder = CountingEmbedder()
    assert embed_missing(pg_session, embedder) == 2 and embed_missing(pg_session, embedder) == 0  # the second call has nothing left
    expected = sorted(embedding_text(d.filename, c.heading, c.content) for c in pg_session.query(DocumentChunk).all() for d in [pg_session.get(Document, c.document_id)])
    assert sorted(embedder.texts) == expected
    late = vectors(pg_session)
    pg_session.query(DocumentChunk).update({DocumentChunk.embedding: None})
    pg_session.commit()
    ProcessingService(pg_session, svc, MockEmbeddingProvider()).process_document(pg_session.query(Document).filter_by(filename="a.pdf").one())
    ProcessingService(pg_session, svc, MockEmbeddingProvider()).process_document(pg_session.query(Document).filter_by(filename="b.pdf").one())
    assert vectors(pg_session) == late  # embedded as part of processing: identical vectors


def test_the_default_still_embeds_the_chunks_while_processing(pg_session, data_room):
    svc = room_with_two_documents(pg_session, data_room)
    embedder = CountingEmbedder()
    ProcessingService(pg_session, svc, embedder).process_pending()
    assert len(embedder.texts) == 2 and all(c.embedding is not None for c in pg_session.query(DocumentChunk).all())


def test_a_document_saved_without_vectors_is_completed_when_processing_runs_the_usual_way(pg_session, data_room):
    svc = room_with_two_documents(pg_session, data_room)
    ProcessingService(pg_session, svc, MockEmbeddingProvider(), embed_chunks=False).process_pending()
    embedder = CountingEmbedder()
    results = [ProcessingService(pg_session, svc, embedder).process_document(d) for d in pg_session.query(Document).all()]
    assert not any(r.skipped for r in results) and len(embedder.texts) == 2 and all(c.embedding is not None for c in pg_session.query(DocumentChunk).all())


def test_the_agents_search_embeds_the_missing_chunks_once_before_it_searches(pg_session, data_room):
    svc = room_with_two_documents(pg_session, data_room)
    ProcessingService(pg_session, svc, MockEmbeddingProvider(), embed_chunks=False).process_pending()
    embedder = CountingEmbedder()
    retrieval = RetrievalService(pg_session, embedder)
    assert retrieval.ensure_embedded() == 2 and retrieval.ensure_embedded() == 0 and len(embedder.texts) == 2  # embedded once; later searches pay nothing
    assert all(c.embedding is not None for c in pg_session.query(DocumentChunk).all())


def test_a_failed_embedding_is_tried_again_by_the_next_search(pg_session, data_room):
    svc = room_with_two_documents(pg_session, data_room)
    ProcessingService(pg_session, svc, MockEmbeddingProvider(), embed_chunks=False).process_pending()

    class Flaky(CountingEmbedder):
        failing = True

        def _embed_batch(self, texts, task):
            if self.failing:
                raise LLMTransientError("rate limited")
            return super()._embed_batch(texts, task)

    embedder = Flaky(max_attempts=1)
    retrieval = RetrievalService(pg_session, embedder)
    with pytest.raises(LLMTransientError):
        retrieval.ensure_embedded()
    embedder.failing = False
    assert retrieval.ensure_embedded() == 2  # the flag was not set by the failure
