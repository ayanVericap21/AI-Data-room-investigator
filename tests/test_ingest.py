import pytest

from app.documents import InvalidUpload, safe_filename, save_upload
from app.ingest.chunker import chunk_pages
from app.ingest.parsers import PageText, ParseError, parse_file
from app.ingest.pipeline import ingest_document
from app.models import DocStatus
from tests.conftest import make_pdf


# ---------- chunker ----------
def test_chunk_offsets_roundtrip():
    text = "First para about capacity.\n\nSecond para about price.\n\n" + ("long sentence " * 100)
    pages = [PageText(page=1, text=text), PageText(page=2, text="Page two text here.\n\nMore.")]
    chunks = chunk_pages(pages, max_chars=200)
    assert chunks
    by_page = {p.page: p.text for p in pages}
    for c in chunks:
        assert by_page[c.page][c.char_start:c.char_end] == c.text  # evidence references stay valid
        assert len(c.text) <= 200


def test_chunk_merges_small_paragraphs():
    chunks = chunk_pages([PageText(page=1, text="a b c d e f g\n\nh i j k l m n")], max_chars=500)
    assert len(chunks) == 1


def test_chunk_empty_and_invalid():
    assert chunk_pages([PageText(page=1, text="   \n\n  ")]) == []
    with pytest.raises(ValueError):
        chunk_pages([], max_chars=5)


# ---------- parsers ----------
def test_parse_pdf_pages(tmp_path):
    p = tmp_path / "a.pdf"
    p.write_bytes(make_pdf(["Capacity is 30000 tonnes", "Price is 5 USD"]))
    pages = parse_file(p)
    assert [x.page for x in pages] == [1, 2]
    assert "30000" in pages[0].text and "Price" in pages[1].text


def test_parse_corrupt_pdf(tmp_path):
    p = tmp_path / "bad.pdf"
    p.write_bytes(b"%PDF-1.4 this is not a pdf")
    with pytest.raises(ParseError) as e:
        parse_file(p)
    assert e.value.code in {"PARSE_ERROR", "EMPTY_TEXT"}


def test_parse_blank_pdf_is_empty_text(tmp_path):
    p = tmp_path / "scan.pdf"
    p.write_bytes(make_pdf([""]))
    with pytest.raises(ParseError) as e:
        parse_file(p)
    assert e.value.code == "EMPTY_TEXT"


def test_parse_csv_keeps_headers(tmp_path):
    p = tmp_path / "m.csv"
    p.write_text("year,revenue\n2024,100\n2025,120\n")
    text = parse_file(p)[0].text
    assert "year=2024; revenue=100" in text


# ---------- upload / storage ----------
def test_save_upload_stores_locally(settings, db):
    doc, dup = save_upload(db, settings, "report.txt", b"hello world data")
    assert not dup and doc.status == DocStatus.UPLOADED
    from pathlib import Path
    assert Path(doc.stored_path).read_bytes() == b"hello world data"
    assert Path(doc.stored_path).parent == settings.uploads_dir


def test_duplicate_content_not_stored_twice(settings, db):
    a, _ = save_upload(db, settings, "a.txt", b"same content")
    b, dup = save_upload(db, settings, "b.txt", b"same content")
    assert dup and a.id == b.id


@pytest.mark.parametrize("name,data", [("x.exe", b"abc"), ("e.txt", b""), ("", b"abc")])
def test_invalid_uploads(settings, db, name, data):
    with pytest.raises(InvalidUpload):
        save_upload(db, settings, name, data)


def test_upload_too_large(settings, db):
    settings.max_upload_bytes = 10
    with pytest.raises(InvalidUpload):
        save_upload(db, settings, "a.txt", b"x" * 11)


def test_path_traversal_filename_sanitised():
    assert safe_filename("../../etc/passwd.txt") == "passwd.txt"
    assert safe_filename("..\\..\\win.txt") == "win.txt"


# ---------- pipeline ----------
def test_ingest_document_success(settings, db):
    doc, _ = save_upload(db, settings, "a.txt", b"Capacity is 30,000 tonnes per year.\n\nPrice is fixed.")
    res = ingest_document(db, settings, doc.id)
    assert res.status == DocStatus.PROCESSED and res.chunk_count >= 1
    assert len(db.search_chunks(["capacity"], 10)) == res.chunk_count == 1  # search index is populated


def test_reingest_replaces_chunks(settings, db):
    doc, _ = save_upload(db, settings, "a.txt", b"some text for the document body")
    ingest_document(db, settings, doc.id)
    ingest_document(db, settings, doc.id)
    assert db.one("SELECT COUNT(*) n FROM chunks")["n"] == 1
    assert len(db.search_chunks(["document"], 10)) == 1  # no stale index rows left behind


def test_ingest_failure_is_recorded_not_raised(settings, db):
    doc, _ = save_upload(db, settings, "bad.pdf", b"not really a pdf")
    res = ingest_document(db, settings, doc.id)
    assert res.status == DocStatus.FAILED and res.failure_code
    assert db.one("SELECT status FROM documents WHERE id=?", (doc.id,))["status"] == "failed"


def test_ingest_missing_file_is_internal_error(settings, db):
    import os
    doc, _ = save_upload(db, settings, "a.txt", b"some text for the document")
    os.remove(doc.stored_path)
    settings.io_retries = 0
    res = ingest_document(db, settings, doc.id)
    assert res.status == DocStatus.FAILED


def test_ingest_unknown_document(settings, db):
    assert ingest_document(db, settings, 999).failure_code == "NOT_FOUND"
