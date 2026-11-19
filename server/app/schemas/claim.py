"""Pydantic schemas for claims: LLM output, API output, and claim matching."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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


# ---- claim matching -----------------------------------------------------------


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
    claim_text: str | None = None
    value_type: str | None = None  # quantity, date, identifier, name, state or statement (the claim's claim_category); None when unknown
    table_key: str | None = None  # set when the claim was read from a table; claims with the same key are rows of one table


ROUTES = Literal["lexical", "label", "embedding", "llm", "identifier", "rule"]


class Cluster(BaseModel):
    """Claims that may state one fact: the grouping model's cluster, a rule and the records that may fall under it, or claims the code's matching put together
    (labels, meaning, a record id) that the model left alone. A claim is in one cluster only. Whether the claims state one value or differ is decided later, by the judge."""

    claim_ids: list[str]
    topic: str = ""
    matched_by: ROUTES = "lexical"
    topic_score: float = 1.0
    rule_id: str | None = None  # a rule cluster: the claim that is the rule (the first of ``claim_ids``)
    via: dict[str, str] = Field(default_factory=dict)  # claim id -> what put it in the cluster: llm, rule, lexical, label, embedding, identifier, negation or window

    @model_validator(mode="after")
    def _distinct_and_at_least_two(self) -> "Cluster":
        self.claim_ids = list(dict.fromkeys(self.claim_ids))
        if len(self.claim_ids) < 2:
            raise ValueError("a cluster holds at least two claims")
        return self


class ClaimGroup(BaseModel):
    """A fact and the claims that may state it."""

    claims: list[ClaimOut]
    topic: str = ""
    matched_by: ROUTES = "lexical"


class ClaimGroupListOut(BaseModel):
    items: list[ClaimGroup]
    total: int  # groups
