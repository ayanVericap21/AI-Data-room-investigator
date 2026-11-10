import hashlib
import os

from app.models import Document
from app.models.base import ProcessingStatus
from app.services.document_service import DocumentService, hash_file
from tests.conftest import make_docx, make_pdf


def test_hash_file_is_sha256(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"hello")
    assert hash_file(f) == hashlib.sha256(b"hello").hexdigest()


def test_scan_adds_and_is_idempotent(session, settings, data_room):
    make_pdf(data_room / "a.pdf", ["one"])
    (data_room / "sub").mkdir()
    make_docx(data_room / "sub" / "b.docx")
    (data_room / "notes.txt").write_text("ignored")

    svc = DocumentService(session, settings)
    first = svc.scan()
    assert (first.discovered, first.added) == (2, 2)
    assert first.unsupported == ["notes.txt"]
    second = svc.scan()
    assert (second.added, second.updated, second.unchanged) == (0, 0, 2)

    names = sorted(d.filename for d in svc.list_documents())
    assert names == ["a.pdf", "sub/b.docx"]  # relative, POSIX, never absolute


def test_rename_keeps_identity(session, settings, data_room):
    make_pdf(data_room / "a.pdf", ["one"])
    svc = DocumentService(session, settings)
    svc.scan()
    original = svc.list_documents()[0].id
    os.rename(data_room / "a.pdf", data_room / "renamed.pdf")
    result = svc.scan()
    assert result.updated == 1
    doc = svc.list_documents()[0]
    assert doc.id == original and doc.filename == "renamed.pdf"


def test_duplicate_content_is_skipped(session, settings, data_room):
    make_pdf(data_room / "a.pdf", ["same"])
    (data_room / "copy.pdf").write_bytes((data_room / "a.pdf").read_bytes())
    result = DocumentService(session, settings).scan()
    assert result.added == 1 and result.skipped_duplicates == ["copy.pdf"]


def test_removed_file_marked_missing_and_hidden(session, settings, data_room):
    make_pdf(data_room / "a.pdf", ["one"])
    svc = DocumentService(session, settings)
    svc.scan()
    (data_room / "a.pdf").unlink()
    assert svc.scan().missing == 1
    assert svc.list_documents() == []
    row = session.query(Document).one()
    assert row.processing_status == ProcessingStatus.missing.value


def test_changed_content_creates_new_document(session, settings, data_room):
    make_pdf(data_room / "a.pdf", ["v1"])
    svc = DocumentService(session, settings)
    svc.scan()
    make_pdf(data_room / "a.pdf", ["v2 different"])
    result = svc.scan()
    assert result.added == 1 and result.missing == 1
    assert [d.filename for d in svc.list_documents()] == ["a.pdf"]


def test_empty_data_room_dir_missing(session, tmp_path):
    from app.core.config import Settings

    svc = DocumentService(session, Settings(data_room_path=tmp_path / "nope"))
    assert svc.scan().discovered == 0
