"""Persistence for documents."""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Document
from app.models.base import ProcessingStatus


class DocumentRepository:
    def __init__(self, session: Session):
        self.session = session

    def get(self, document_id: str) -> Document | None:
        return self.session.get(Document, document_id)

    def get_by_hash(self, file_hash: str) -> Document | None:
        return self.session.scalar(select(Document).where(Document.file_hash == file_hash))

    def list_present(self) -> list[Document]:
        stmt = (
            select(Document)
            .where(Document.processing_status != ProcessingStatus.missing.value)
            .order_by(Document.filename)
        )
        return list(self.session.scalars(stmt))

    def list_all(self) -> list[Document]:
        return list(self.session.scalars(select(Document)))

    def add(self, document: Document) -> Document:
        self.session.add(document)
        return document
