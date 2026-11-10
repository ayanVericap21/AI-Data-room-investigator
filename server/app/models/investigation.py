"""investigations and investigation_events tables."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import BigInteger, ForeignKey, Index, Numeric, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Approach, Base, InvestigationStatus, JsonType, new_id, tz_datetime, utcnow


class Investigation(Base):
    """One execution of an investigation approach over the data room.

    Written by: the API on start/stop/continue, and the worker/graph as it runs
    (status, tokens, cost, duration, errors). A stop request is recorded by moving
    status to ``stopping``; the graph checks it at safe boundaries.
    Used by: the frontend (status polling), findings and events (parent), evaluation.
    ``execution_config`` snapshots limits (max iterations, LLM calls, duration) and,
    for follow-up runs, the parent investigation and user question.
    """

    __tablename__ = "investigations"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: new_id("inv"))
    name: Mapped[str] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(24), default=InvestigationStatus.pending.value, index=True)
    approach: Mapped[str] = mapped_column(String(16), default=Approach.improved.value)
    started_at: Mapped[datetime | None] = mapped_column(tz_datetime(), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(tz_datetime(), nullable=True)
    stopped_at: Mapped[datetime | None] = mapped_column(tz_datetime(), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    execution_config: Mapped[dict] = mapped_column(JsonType, default=dict)
    total_tokens: Mapped[int] = mapped_column(BigInteger, default=0)
    estimated_cost: Mapped[float] = mapped_column(Numeric(12, 6), default=0)
    duration_ms: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    created_at: Mapped[datetime] = mapped_column(tz_datetime(), default=utcnow)


class InvestigationEvent(Base):
    """Append-only execution trace row shown in the frontend timeline.

    Written by: every graph node (and the API for user interventions), in its own
    committed transaction so polling sees progress immediately.
    Contains operational summaries and concise decision explanations only - never
    raw model reasoning.
    """

    __tablename__ = "investigation_events"
    __table_args__ = (Index("ix_events_investigation_created", "investigation_id", "created_at"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: new_id("evt"))
    investigation_id: Mapped[str] = mapped_column(ForeignKey("investigations.id", ondelete="CASCADE"))
    event_type: Mapped[str] = mapped_column(String(64))
    node_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    message: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(24))  # info | success | warning | error
    details: Mapped[dict] = mapped_column(JsonType, default=dict)
    created_at: Mapped[datetime] = mapped_column(tz_datetime(), default=utcnow)
