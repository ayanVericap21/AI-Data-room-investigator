"""Structured outputs requested from the LLM. Everything is validated before the application uses it."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ..models import Classification


class ClaimOut(BaseModel):
    chunk_id: int
    subject: str = Field(description="What the claim is about, e.g. 'annual production capacity'")
    value: str = Field(description="The stated value/commitment, e.g. '30,000'")
    unit: str | None = None
    conditions: str | None = Field(None, description="Scope, period, assumptions or conditions attached to the claim")
    quote: str = Field(description="Verbatim text copied from the passage that states the claim")


class ClaimsOut(BaseModel):
    claims: list[ClaimOut]


class HypothesisOut(BaseModel):
    claim_ids: list[int] = Field(description="Ids of the claims to compare (2 or more)")
    question: str = Field(description="The specific question to investigate")
    rationale: str


class HypothesesOut(BaseModel):
    hypotheses: list[HypothesisOut]


class CitationOut(BaseModel):
    chunk_id: int
    quote: str = Field(description="Verbatim text copied from that passage")


class ConclusionOut(BaseModel):
    classification: Classification
    title: str
    explanation: str = Field(description="Why this classification; refer to the cited quotes")
    citations: list[CitationOut]
    uncertainties: list[str] = []
    assumptions: list[str] = []
    confidence: float = Field(ge=0, le=1)


class StepOut(BaseModel):
    belief: Classification = Field(description="Your current best classification of the hypothesis")
    reasoning: str = Field(description="What you learned and why you chose the next action")
    action: Literal["search", "get_neighbors", "conclude"]
    query: str | None = Field(None, description="For search: keywords")
    chunk_id: int | None = Field(None, description="For get_neighbors")
    conclusion: ConclusionOut | None = Field(None, description="Required when action is conclude")


class SupportOut(BaseModel):
    supported: bool
    issues: list[str] = []


class BaselineOut(BaseModel):
    findings: list[ConclusionOut]
