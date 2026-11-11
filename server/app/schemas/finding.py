"""Read models for the findings a run saved: grouped by category, each with its verified evidence."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel

from app.schemas.claim import SourceRef


class EvidenceOut(BaseModel):
    role: str  # supporting | contradicting | contextual | insufficient
    quoted_text: str  # exact text of the source chunk
    relevance_reason: str
    support_assessment: str
    source: SourceRef  # document, chunk and location (page, sheet and cell, or heading)


class FindingOut(BaseModel):
    id: str
    title: str
    description: str
    classification: str
    severity: str
    confidence: float
    evidence_status: str
    verification_status: str
    materiality_reasoning: str | None = None
    business_impact: str | None = None
    unresolved_questions: list[str] = []
    evidence: list[EvidenceOut]


class InvestigationOut(BaseModel):
    id: str
    name: str
    status: str
    started_at: datetime | None = None
    completed_at: datetime | None = None
    finding_count: int = 0


class FindingsOut(BaseModel):
    investigation: InvestigationOut | None  # None when nothing has been run yet
    counts: dict[str, int]  # findings per category (classification)
    findings: list[FindingOut]  # contradictions first, then by severity within a category
