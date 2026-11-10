"""documents and document_chunks tables."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import EMBEDDING_DIM, Base, JsonType, ProcessingStatus, new_id, tz_datetime, utcnow
from app.models.vector import Embedding


class Document(Base):
    """One file in the predefined data room (server/data_room/).

    Written by: Scan Documents node (create/update), Process Documents node
    (updates ``processing_status``).
    Used by: every later stage; chunks and claims reference it, and the UI lists it.
    Identity is the SHA-256 of the file *content*, so renaming a file keeps its id
    and unchanged files are never reprocessed (idempotent indexing).
    """

    __tablename__ = "documents"
    __table_args__ = (UniqueConstraint("file_hash", name="uq_documents_file_hash"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: new_id("doc"))
    # Path relative to the data room root, POSIX separators. Never an absolute path.
    filename: Mapped[str] = mapped_column(String(512), index=True)
    # SHA-256 hex digest of content: stable identity + change detection.
    file_hash: Mapped[str] = mapped_column(String(64))
    file_type: Mapped[str] = mapped_column(String(8))  # pdf | docx | xlsx
    file_size: Mapped[int] = mapped_column(Integer)
    processing_status: Mapped[str] = mapped_column(String(16), default=ProcessingStatus.pending.value)
    last_modified_at: Mapped[datetime] = mapped_column(tz_datetime())
    created_at: Mapped[datetime] = mapped_column(tz_datetime(), default=utcnow)

    chunks: Mapped[list["DocumentChunk"]] = relationship(back_populates="document", cascade="all, delete-orphan")


class DocumentChunk(Base):
    """A passage of extracted text with its exact source location.

    Written by: Process Documents node (text, locator, then the embedding vector).
    Text and vector live in the same row, so they can never drift apart.
    Used by: claim extraction (input), evidence retrieval (results), and as the
    anchor for every ``evidence`` row, so any citation can be re-verified verbatim.
    """

    __tablename__ = "document_chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_chunks_doc_index"),
        Index("ix_chunks_document", "document_id"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: new_id("chk"))
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    chunk_index: Mapped[int] = mapped_column(Integer)  # reading order within the document
    content: Mapped[str] = mapped_column(Text)
    content_type: Mapped[str] = mapped_column(String(32))  # paragraph | heading | table | table_row | ...
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)  # PDF
    sheet_name: Mapped[str | None] = mapped_column(String(255), nullable=True)  # XLSX
    cell_reference: Mapped[str | None] = mapped_column(String(64), nullable=True)  # XLSX cell/range
    heading: Mapped[str | None] = mapped_column(Text, nullable=True)  # nearest heading (DOCX/PDF)
    # Flexible, source-specific extras: DOCX paragraph/table ids, XLSX cell formulas and
    # cached values, table cells, etc.
    metadata_: Mapped[dict] = mapped_column("metadata", JsonType, default=dict)
    # pgvector embedding of the chunk text (document-side). NULL until embedded, which
    # is how indexing stays idempotent. Searched with cosine distance (HNSW index).
    embedding: Mapped[list[float] | None] = mapped_column(Embedding(EMBEDDING_DIM), nullable=True)
    created_at: Mapped[datetime] = mapped_column(tz_datetime(), default=utcnow)

    document: Mapped[Document] = relationship(back_populates="chunks")
