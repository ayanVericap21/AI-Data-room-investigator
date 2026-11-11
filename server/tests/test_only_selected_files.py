"""A run on a selection of files must not scan, parse or change any other file."""
from app.models import Document, DocumentChunk
from app.models.base import ProcessingStatus
from app.llm.embeddings import MockEmbeddingProvider
from app.services.document_service import DocumentService
from app.services.processing_service import ProcessingService

from tests.conftest import make_pdf


def test_scan_and_process_leave_unselected_files_untouched(session, settings, data_room):
    make_pdf(data_room / "a.pdf", ["alpha text"])
    make_pdf(data_room / "b.pdf", ["beta text"])
    svc = DocumentService(session, settings)

    first = svc.scan(only={"a.pdf"})
    assert (first.discovered, first.added) == (1, 1)  # b.pdf was not even read
    assert [d.filename for d in session.query(Document).all()] == ["a.pdf"]

    svc.scan()  # a full scan registers both
    ProcessingService(session, svc, MockEmbeddingProvider()).process_pending(only={"a.pdf"})
    status = {d.filename: d.processing_status for d in session.query(Document).all()}
    assert status == {"a.pdf": ProcessingStatus.processed.value, "b.pdf": ProcessingStatus.pending.value}
    b = session.query(Document).filter_by(filename="b.pdf").one()
    assert session.query(DocumentChunk).filter_by(document_id=b.id).count() == 0


def test_a_selected_scan_does_not_mark_unselected_documents_missing(session, settings, data_room):
    make_pdf(data_room / "a.pdf", ["alpha"])
    make_pdf(data_room / "b.pdf", ["beta"])
    svc = DocumentService(session, settings)
    svc.scan()
    result = svc.scan(only={"a.pdf"})
    assert result.missing == 0
    assert {d.processing_status for d in session.query(Document).all()} == {ProcessingStatus.pending.value}
