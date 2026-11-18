"""extraction_cache table."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import Float, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, JsonType, new_id, tz_datetime, utcnow


class ExtractionCacheEntry(Base):
    """The raw response of one LlamaExtract job, kept so a run on an unchanged file, schema and prompt does not pay for it again.

    Written by: the claim extractor, after a job completes. Used by: the claim extractor, before it starts a job.
    The key is (file_sha256, schema_hash, prompt_hash, tier, pages): ``tier`` and ``pages`` are part of it because a different tier, or
    a different page range of the same file, is a different response. ``pages`` is '' for a whole document and e.g. '5-8' for a range job.
    The response is kept raw (not the claims made from it), so the checks that run after it (value gate, page-number filter) still run on every run.
    Not tied to ``documents`` or ``claims``: clearing results does not touch it.
    """

    __tablename__ = "extraction_cache"
    __table_args__ = (UniqueConstraint("file_sha256", "schema_hash", "prompt_hash", "tier", "pages", name="uq_extraction_cache_key"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: new_id("exc"))
    file_sha256: Mapped[str] = mapped_column(String(64))  # the file's content hash (documents.file_hash)
    schema_hash: Mapped[str] = mapped_column(String(64))  # SHA-256 of the extraction schema sent
    prompt_hash: Mapped[str] = mapped_column(String(64))  # SHA-256 of the system prompt sent
    tier: Mapped[str] = mapped_column(String(32))
    pages: Mapped[str] = mapped_column(String(32), default="")
    result: Mapped[object] = mapped_column(JsonType)  # extract_result: claims, references, profile
    extract_metadata: Mapped[object | None] = mapped_column(JsonType, nullable=True)  # the citations of each field
    credits: Mapped[float] = mapped_column(Float, default=0.0)  # what the job cost when it ran: the saving of every later hit
    pages_billed: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(tz_datetime(), default=utcnow)
