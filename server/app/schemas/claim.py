"""Pydantic schemas for claims: LLM output, API output, and candidate matching."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ExtractedClaim(BaseModel):
    """One claim as returned by the LLM. Unknown attributes must be null, never guessed."""

    chunk_id: str
    subject: str | None = None
    metric: str | None = None
    value: int | float | str | None = None
    unit: str | None = None
    period: str | None = None
    conditions: str | None = None
    assumptions: list[str] = Field(default_factory=list)
    claim_text: str = Field(min_length=1)
    claim_category: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("subject", "metric", "unit", "period", "conditions", "claim_category", mode="before")
    @classmethod
    def _blank_to_none(cls, v):
        if isinstance(v, str) and not v.strip():
            return None
        return v.strip() if isinstance(v, str) else v


class ClaimExtractionResult(BaseModel):
    claims: list[ExtractedClaim] = Field(default_factory=list)


class SourceRef(BaseModel):
    document_id: str
    filename: str
    chunk_id: str
    content_type: str | None = None
    page_number: int | None = None
    sheet_name: str | None = None
    cell_reference: str | None = None
    heading: str | None = None


class ClaimOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    subject: str | None
    metric: str | None
    value: int | float | str | None
    unit: str | None
    period: str | None
    conditions: str | None
    assumptions: list[str] | None
    claim_text: str
    claim_category: str | None
    extraction_confidence: float | None
    created_at: datetime
    source: SourceRef


class ClaimListOut(BaseModel):
    items: list[ClaimOut]
    total: int


class ChunkOut(BaseModel):
    id: str
    document_id: str
    filename: str
    content: str
    content_type: str
    page_number: int | None
    sheet_name: str | None
    cell_reference: str | None
    heading: str | None


# ---- candidate matching -------------------------------------------------------
RelationHint = Literal["exact", "partial", "conflicting", "contextual"]


class ClaimView(BaseModel):
    """The subset of a claim used for matching (decoupled from the ORM row)."""

    id: str
    document_id: str
    chunk_id: str
    subject: str | None = None
    metric: str | None = None
    value: int | float | str | None = None
    unit: str | None = None
    period: str | None = None
    conditions: str | None = None
    assumptions: list[str] = Field(default_factory=list)


class AttributeComparison(BaseModel):
    subject_match: bool | None
    metric_match: bool | None
    unit_match: bool | None  # None = at least one side unknown
    period_match: bool | None
    value_relation: Literal["equal", "different", "unknown"]
    conditions_differ: bool
    assumptions_differ: bool


class CandidatePair(BaseModel):
    """A *candidate* comparison. relation_hint is a deterministic hint, not a finding."""

    claim_a_id: str
    claim_b_id: str
    topic_score: float
    relation_hint: RelationHint
    comparison: AttributeComparison


class CandidateListOut(BaseModel):
    items: list[CandidatePair]
    total: int
