"""claims table."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import Float, ForeignKey, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, JsonType, new_id, tz_datetime, utcnow


class Claim(Base):
    """A structured business claim extracted from one chunk.

    Written by: Extract Claims node (LLM, validated by Pydantic). Unknown attributes
    are NULL, never invented.
    Used by: candidate pair generation (matches on subject/metric/unit/period),
    discrepancy detection, and ``finding_claims`` / ``evidence`` for traceability.
    """

    __tablename__ = "claims"
    __table_args__ = (Index("ix_claims_document", "document_id"), Index("ix_claims_metric", "metric"))

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: new_id("clm"))
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    chunk_id: Mapped[str] = mapped_column(ForeignKey("document_chunks.id", ondelete="CASCADE"))
    subject: Mapped[str | None] = mapped_column(Text, nullable=True)
    metric: Mapped[str | None] = mapped_column(Text, nullable=True)
    # JSON so one column holds numbers, strings or ranges without losing the original form.
    value: Mapped[object | None] = mapped_column(JsonType, nullable=True)
    unit: Mapped[str | None] = mapped_column(String(64), nullable=True)
    period: Mapped[str | None] = mapped_column(String(128), nullable=True)
    conditions: Mapped[str | None] = mapped_column(Text, nullable=True)
    assumptions: Mapped[list | None] = mapped_column(JsonType, nullable=True)
    claim_text: Mapped[str] = mapped_column(Text)  # the statement, close to the source wording
    claim_category: Mapped[str | None] = mapped_column(String(64), nullable=True)
    extraction_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(tz_datetime(), default=utcnow)
