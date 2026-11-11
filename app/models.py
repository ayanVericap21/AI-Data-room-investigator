"""Pydantic models for everything that crosses a function boundary."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, field_validator


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class DocStatus(str, Enum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Classification(str, Enum):
    CONTRADICTION = "contradiction"
    CONTEXTUAL_DIFFERENCE = "contextual_difference"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    CONSISTENT = "consistent"


class DocSummary(BaseModel):
    """LLM-written orientation aid for a document. NOT evidence: citations must quote the source passages."""

    overview: str = Field(description="3-6 sentences: what the document is, its purpose and main subject matter")
    key_topics: list[str] = Field(default_factory=list, description="Short topic labels")
    key_figures: list[str] = Field(default_factory=list, description="Important numbers/dates/commitments, each with what it refers to")


class Document(BaseModel):
    id: int
    filename: str
    stored_path: str
    sha256: str
    size_bytes: int
    status: DocStatus
    failure_code: str | None = None
    error: str | None = None
    page_count: int = 0
    chunk_count: int = 0
    summary: DocSummary | None = None
    summary_status: str = "pending"  # pending | done | failed | skipped
    summary_error: str | None = None
    created_at: str

    @field_validator("summary", mode="before")
    @classmethod
    def _parse_summary(cls, v):
        return DocSummary.model_validate_json(v) if isinstance(v, str) and v else (None if not v else v)


class Chunk(BaseModel):
    """A passage with enough location info to find it again: page + char offsets into that page's text."""

    id: int
    document_id: int
    ordinal: int
    page: int
    char_start: int
    char_end: int
    text: str


class DocResult(BaseModel):
    document_id: int
    status: DocStatus
    failure_code: str | None = None
    error: str | None = None
    chunk_count: int = 0
    summary_status: str = "pending"


class IngestJob(BaseModel):
    id: int
    status: JobStatus
    document_ids: list[int]
    idempotency_key: str
    results: list[DocResult] = Field(default_factory=list)
    usage: dict = Field(default_factory=dict)  # LLM usage of the summarisation step
    error: str | None = None
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None


class Claim(BaseModel):
    id: int
    investigation_id: int
    document_id: int
    chunk_id: int
    subject: str
    value: str
    unit: str | None = None
    conditions: str | None = None
    quote: str


class Citation(BaseModel):
    chunk_id: int
    quote: str
    document_id: int | None = None
    filename: str | None = None
    page: int | None = None
    verified: bool = False
    note: str | None = None


class Finding(BaseModel):
    id: int | None = None
    investigation_id: int | None = None
    hypothesis_id: int | None = None
    classification: Classification
    title: str
    explanation: str
    citations: list[Citation] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    # established: verified and confident enough to present as a finding; unresolved: shown as open question
    status: str = "unresolved"
    verification_notes: list[str] = Field(default_factory=list)
    superseded: bool = False
