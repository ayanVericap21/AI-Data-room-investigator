"""Scans the predefined data room and keeps the documents table in sync.

Deterministic only (no LLM). Identity is the SHA-256 of file content.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.exceptions import NotFoundError
from app.models import Document
from app.models.base import ProcessingStatus
from app.repositories.document_repository import DocumentRepository
from app.schemas.document import DocumentOut
from app.services.extraction_service import file_type_for, unit_count

logger = logging.getLogger(__name__)


def hash_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class ScanResult:
    discovered: int = 0
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    missing: int = 0
    skipped_duplicates: list[str] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)


class DocumentService:
    def __init__(self, session: Session, settings: Settings):
        self.session = session
        self.root = settings.data_room_dir
        self.repo = DocumentRepository(session)

    # -- path handling -------------------------------------------------------
    def resolve_path(self, document: Document) -> Path:
        """Map a document's relative filename to a file inside the data room only."""
        path = (self.root / document.filename).resolve()
        if not path.is_relative_to(self.root):
            raise NotFoundError("Document path is outside the data room")
        return path

    def _iter_files(self) -> list[Path]:
        if not self.root.is_dir():
            return []
        return sorted(p for p in self.root.rglob("*") if p.is_file() and not p.name.startswith((".", "~$")))

    # -- scanning ------------------------------------------------------------
    def scan(self, only: set[str] | None = None) -> ScanResult:
        """``only`` (relative paths) limits the scan to those files: any other file is not read, hashed or changed,
        and no other document is marked missing."""
        result = ScanResult()
        seen_hashes: set[str] = set()
        for path in self._iter_files():
            rel = path.relative_to(self.root).as_posix()
            if only is not None and rel not in only:
                continue
            file_type = file_type_for(path)
            if file_type is None:
                if path.name != ".gitkeep":
                    result.unsupported.append(rel)
                continue
            result.discovered += 1
            try:
                file_hash = hash_file(path)
                stat = path.stat()
            except OSError as exc:
                logger.warning("Cannot read file", extra={"file": rel, "error": str(exc)})
                result.unsupported.append(rel)
                continue
            if file_hash in seen_hashes:
                result.skipped_duplicates.append(rel)  # identical content already seen this scan
                continue
            seen_hashes.add(file_hash)
            modified = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)

            existing = self.repo.get_by_hash(file_hash)
            if existing is None:
                self.repo.add(Document(
                    filename=rel, file_hash=file_hash, file_type=file_type,
                    file_size=stat.st_size, last_modified_at=modified,
                    processing_status=ProcessingStatus.pending.value,
                ))
                result.added += 1
            else:
                changed = (
                    existing.filename != rel
                    or existing.processing_status == ProcessingStatus.missing.value
                    or existing.last_modified_at.replace(tzinfo=timezone.utc) != modified
                )
                if changed:
                    existing.filename = rel
                    existing.last_modified_at = modified
                    existing.file_size = stat.st_size
                    if existing.processing_status == ProcessingStatus.missing.value:
                        existing.processing_status = ProcessingStatus.pending.value
                    result.updated += 1
                else:
                    result.unchanged += 1
            self.session.flush()

        # Rows whose content is no longer present in the data room.
        for doc in self.repo.list_all():
            if only is not None and doc.filename not in only:
                continue
            if doc.file_hash not in seen_hashes and doc.processing_status != ProcessingStatus.missing.value:
                doc.processing_status = ProcessingStatus.missing.value
                result.missing += 1
        self.session.commit()
        logger.info("Data room scan complete", extra={"scan": result.__dict__})
        return result

    # -- reads ---------------------------------------------------------------
    def to_out(self, document: Document) -> DocumentOut:
        out = DocumentOut.model_validate(document)
        try:
            path = self.resolve_path(document)
            if path.is_file():
                out.unit_count = unit_count(path, document.file_type)
        except NotFoundError:
            pass
        return out

    def list_documents(self) -> list[DocumentOut]:
        return [self.to_out(d) for d in self.repo.list_present()]

    def get_document(self, document_id: str) -> DocumentOut:
        document = self.repo.get(document_id)
        if document is None:
            raise NotFoundError(f"Document '{document_id}' not found")
        return self.to_out(document)
