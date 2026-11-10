"""findings, finding_claims and evidence tables."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import Float, ForeignKey, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, JsonType, new_id, tz_datetime, utcnow


class Finding(Base):
    """Final, validated result of investigating one candidate.

    Written by: Final Validation and Persistence node, incrementally as each finding
    passes validation (so the UI shows it before the run ends).
    Only findings with ``verification_status == 'verified'`` may be shown as
    established contradictions; others stay explicitly qualified.
    ``evidence_status`` (adequate/partial/inadequate/unavailable) is separate from
    ``classification``: it says how well the evidence supports the conclusion.
    """

    __tablename__ = "findings"
    __table_args__ = (Index("ix_findings_investigation", "investigation_id"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: new_id("fnd"))
    investigation_id: Mapped[str] = mapped_column(ForeignKey("investigations.id", ondelete="CASCADE"))
    title: Mapped[str] = mapped_column(String(512))
    description: Mapped[str] = mapped_column(Text)
    classification: Mapped[str] = mapped_column(String(32))
    severity: Mapped[str] = mapped_column(String(16))
    confidence: Mapped[float] = mapped_column(Float)
    materiality_reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)
    business_impact: Mapped[str | None] = mapped_column(Text, nullable=True)
    evidence_status: Mapped[str] = mapped_column(String(16))
    verification_status: Mapped[str] = mapped_column(String(16))  # verified | unverified
    unresolved_questions: Mapped[list] = mapped_column(JsonType, default=list)
    created_at: Mapped[datetime] = mapped_column(tz_datetime(), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(tz_datetime(), default=utcnow, onupdate=utcnow)


class FindingClaim(Base):
    """Many-to-many link: which claims a finding is about, and how each relates.

    Written by: Persist node. Used by: finding detail view ("original claims").
    """

    __tablename__ = "finding_claims"

    finding_id: Mapped[str] = mapped_column(ForeignKey("findings.id", ondelete="CASCADE"), primary_key=True)
    claim_id: Mapped[str] = mapped_column(ForeignKey("claims.id", ondelete="CASCADE"), primary_key=True)
    relationship_type: Mapped[str] = mapped_column(String(16))  # supports|contradicts|contextualizes|derived_from


class Evidence(Base):
    """A verified source passage backing a finding.

    Written by: Evidence Verification and Persist nodes. ``chunk_id`` is mandatory so
    every quote traces to stored source text; ``quoted_text`` must be an exact
    substring of that chunk (checked in Python, never trusted from the LLM).
    Calculation evidence stores its inputs/formula/result in the quote/reason fields.
    """

    __tablename__ = "evidence"
    __table_args__ = (Index("ix_evidence_finding", "finding_id"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: new_id("evd"))
    finding_id: Mapped[str] = mapped_column(ForeignKey("findings.id", ondelete="CASCADE"))
    chunk_id: Mapped[str] = mapped_column(ForeignKey("document_chunks.id"))
    claim_id: Mapped[str | None] = mapped_column(ForeignKey("claims.id"), nullable=True)
    evidence_role: Mapped[str] = mapped_column(String(16))  # supporting|contradicting|contextual|insufficient
    quoted_text: Mapped[str] = mapped_column(Text)
    relevance_reason: Mapped[str] = mapped_column(Text)
    support_assessment: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(tz_datetime(), default=utcnow)
