"""Ingest one document: parse -> chunk -> persist chunks + FTS index. Failures are recorded, never raised."""
from __future__ import annotations

import logging
import time
from pathlib import Path

from ..config import Settings
from ..db import Database
from ..models import DocResult, DocStatus
from .chunker import chunk_pages
from .parsers import ParseError, parse_file

log = logging.getLogger(__name__)


def _parse_with_retry(path: Path, retries: int):
    """Retry only transient I/O errors. ParseError is deterministic and propagates immediately."""
    for attempt in range(retries + 1):
        try:
            return parse_file(path)
        except OSError:
            if attempt == retries:
                raise
            time.sleep(0.2 * (attempt + 1))


def _set_status(db: Database, doc_id: int, status: DocStatus, code=None, error=None, pages=0, chunks=0):
    db.execute(
        "UPDATE documents SET status=?, failure_code=?, error=?, page_count=?, chunk_count=? WHERE id=?",
        (status.value, code, error, pages, chunks, doc_id),
    )


def ingest_document(db: Database, settings: Settings, doc_id: int) -> DocResult:
    row = db.one("SELECT * FROM documents WHERE id=?", (doc_id,))
    if not row:
        return DocResult(document_id=doc_id, status=DocStatus.FAILED, failure_code="NOT_FOUND", error="no such document")
    _set_status(db, doc_id, DocStatus.PROCESSING)
    try:
        pages = _parse_with_retry(Path(row["stored_path"]), settings.io_retries)
        raw = chunk_pages(pages, settings.chunk_max_chars)
        db.replace_chunks(doc_id, raw)  # atomic, re-ingest safe, keeps the search index in sync
        db.execute("UPDATE documents SET summary=NULL, summary_status='pending', summary_error=NULL WHERE id=?", (doc_id,))
        _set_status(db, doc_id, DocStatus.PROCESSED, pages=len(pages), chunks=len(raw))
        return DocResult(document_id=doc_id, status=DocStatus.PROCESSED, chunk_count=len(raw))
    except ParseError as e:
        _set_status(db, doc_id, DocStatus.FAILED, code=e.code, error=str(e))
        return DocResult(document_id=doc_id, status=DocStatus.FAILED, failure_code=e.code, error=str(e))
    except Exception as e:  # isolate: one bad document must not take down the job
        log.exception("ingest failed for document %s", doc_id)
        _set_status(db, doc_id, DocStatus.FAILED, code="INTERNAL_ERROR", error=f"{type(e).__name__}: {e}")
        return DocResult(document_id=doc_id, status=DocStatus.FAILED, failure_code="INTERNAL_ERROR", error=str(e))
