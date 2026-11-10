"""Declarative base, shared column types and enums for all tables.

Pipeline overview (which stage writes which table):
  Scan Documents      -> documents
  Process Documents   -> document_chunks (+ vectors in ChromaDB, keyed by chunk id)
  Extract Claims      -> claims
  Detect/Plan/Verify  -> investigation_events (trace), evidence (retrieved passages)
  Classify/Persist    -> findings, finding_claims, evidence
All execution of one run hangs off ``investigations``.
"""
from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase

# JSONB on PostgreSQL; plain JSON elsewhere (lets unit tests run on SQLite).
JsonType = JSON().with_variant(JSONB(), "postgresql")


# Width of document_chunks.embedding. pgvector cannot index `vector` columns wider than
# 2000 dims, so providers are asked for this size (see EMBEDDING_DIMENSIONS).
EMBEDDING_DIM = 1536


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def tz_datetime() -> DateTime:
    return DateTime(timezone=True)


class Base(DeclarativeBase):
    pass


class _StrEnum(str, enum.Enum):
    def __str__(self) -> str:
        return self.value


class ProcessingStatus(_StrEnum):
    pending = "pending"
    processing = "processing"
    processed = "processed"
    failed = "failed"
    missing = "missing"  # row exists but the file is no longer in the data room


class InvestigationStatus(_StrEnum):
    pending = "pending"
    running = "running"
    stopping = "stopping"
    completed = "completed"
    stopped = "stopped"
    failed = "failed"
    partially_completed = "partially_completed"


class Approach(_StrEnum):
    baseline = "baseline"
    improved = "improved"


class Classification(_StrEnum):
    contradiction = "contradiction"
    contextual_difference = "contextual_difference"
    missing_information = "missing_information"
    insufficient_evidence = "insufficient_evidence"
    consistent = "consistent"


class Severity(_StrEnum):
    critical = "critical"
    high = "high"
    medium = "medium"
    low = "low"
    informational = "informational"


class EvidenceStatus(_StrEnum):
    adequate = "adequate"
    partial = "partial"
    inadequate = "inadequate"
    unavailable = "unavailable"


class RelationshipType(_StrEnum):
    supports = "supports"
    contradicts = "contradicts"
    contextualizes = "contextualizes"
    derived_from = "derived_from"


class EvidenceRole(_StrEnum):
    supporting = "supporting"
    contradicting = "contradicting"
    contextual = "contextual"
    insufficient = "insufficient"
