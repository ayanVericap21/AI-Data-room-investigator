"""Document storage: validate an upload, write it to local disk, register it."""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

from .config import Settings
from .db import Database
from .models import DocStatus, Document, utcnow

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".csv"}


class InvalidUpload(ValueError):
    pass


class DocumentNotFound(LookupError):
    pass


def safe_filename(name: str) -> str:
    base = os.path.basename((name or "").replace("\\", "/")).strip()
    base = re.sub(r"[^\w.\- ]", "_", base).strip(". ")
    if not base:
        raise InvalidUpload("empty filename")
    return base[:150]


def safe_relpath(name: str) -> str:
    """Sanitised relative path for a document, e.g. 'legal/contracts/lease.pdf' (folder uploads keep their structure).

    Backslashes become '/', empty / '.' / '..' segments are dropped (no traversal), every segment is reduced to safe
    characters, and the result is length-capped. A bare filename is returned unchanged (sanitised)."""
    segs = []
    for seg in (name or "").replace("\\", "/").split("/"):
        seg = re.sub(r"[^\w.\- ]", "_", seg).strip(". ")
        if seg:
            segs.append(seg[:100])
    if not segs:
        raise InvalidUpload("empty filename")
    return "/".join(segs[-8:])[-200:].lstrip("/")  # at most 8 levels deep


def _row_to_document(r) -> Document:
    return Document(**{k: r[k] for k in r.keys()})


def get_document(db: Database, doc_id: int) -> Document:
    r = db.one("SELECT * FROM documents WHERE id=?", (doc_id,))
    if not r:
        raise DocumentNotFound(f"document {doc_id} not found")
    return _row_to_document(r)


def list_documents(db: Database) -> list[Document]:
    return [_row_to_document(r) for r in db.query("SELECT * FROM documents ORDER BY id")]


def save_upload(db: Database, settings: Settings, filename: str, data: bytes) -> tuple[Document, bool]:
    """Store bytes under data/uploads and register the document. Returns (document, was_duplicate).

    Identical content (same sha256) is not stored twice - the existing record is returned.
    """
    display = safe_relpath(filename)          # what users/citations see; keeps folder context
    name = safe_filename(display)             # flat name used on disk
    ext = Path(name).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise InvalidUpload(f"unsupported file type '{ext}'; supported: {sorted(SUPPORTED_EXTENSIONS)}")
    if not data:
        raise InvalidUpload("file is empty")
    if len(data) > settings.max_upload_bytes:
        raise InvalidUpload(f"file exceeds {settings.max_upload_bytes} bytes")

    sha = hashlib.sha256(data).hexdigest()
    existing = db.one("SELECT * FROM documents WHERE sha256=?", (sha,))
    if existing:
        return _row_to_document(existing), True

    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    final = settings.uploads_dir / f"{sha[:12]}_{name}"
    tmp = final.with_suffix(final.suffix + ".part")
    tmp.write_bytes(data)
    os.replace(tmp, final)  # atomic: never leave a half-written file under the final name

    doc_id = db.execute(
        "INSERT INTO documents(filename, stored_path, sha256, size_bytes, status, created_at) VALUES(?,?,?,?,?,?)",
        (display, str(final), sha, len(data), DocStatus.UPLOADED.value, utcnow()),
    )
    return get_document(db, doc_id), False
