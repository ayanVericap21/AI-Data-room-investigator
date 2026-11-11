"""Input and output models for every tool. Pydantic only; nothing crosses a tool boundary as
a bare dict. The JSON schemas of the input models are what the agent is shown."""
from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from app.models.base import Classification, EvidenceRole, EvidenceStatus, Severity
from app.services.outline_service import DocumentOutline
from app.tools.base import ToolInput


# ---- shared views -------------------------------------------------------------
class ChunkView(BaseModel):
    """A stored passage with its exact source location. ``content`` is the text any quote
    must be copied from."""

    chunk_id: str
    document_id: str
    filename: str
    locator: str  # human-readable: "page 3", "sheet Revenue, cell B2", "heading Operations"
    content: str
    content_type: str
    page_number: int | None = None
    sheet_name: str | None = None
    cell_reference: str | None = None
    heading: str | None = None
    score: float | None = None  # cosine similarity, search results only


# ---- list_documents -----------------------------------------------------------
class ListDocumentsInput(ToolInput):
    pass


class DocumentSummary(BaseModel):
    document_id: str
    filename: str
    file_type: str
    status: str
    chunk_count: int
    opened: bool = False  # has the agent opened it (outline, read_section or get_chunk) yet


class ListDocumentsOutput(BaseModel):
    documents: list[DocumentSummary]


class Coverage(BaseModel):
    """How many readable documents the agent has opened. Search hits alone do not count as opened."""

    total: int
    opened: int
    not_opened: list[str]  # file names


# ---- get_outline --------------------------------------------------------------
class GetOutlineInput(ToolInput):
    document_id: str = Field(description="Id from list_documents.")


# ---- search -------------------------------------------------------------------
class SearchInput(ToolInput):
    query: str = Field(min_length=1, max_length=500, description="What to look for, in natural language.")
    top_k: int = Field(5, ge=1, le=10, description="How many passages to return.")
    document_ids: list[str] | None = Field(None, description="Only search these documents.")
    exclude_document_ids: list[str] | None = Field(None, description="Skip these documents.")


class SearchOutput(BaseModel):
    query: str
    hits: list[ChunkView]


# ---- get_chunk ----------------------------------------------------------------
class GetChunkInput(ToolInput):
    chunk_id: str
    neighbors: int = Field(0, ge=0, le=3, description="Also return this many chunks before and after, in reading order.")


class GetChunkOutput(BaseModel):
    chunk: ChunkView
    before: list[ChunkView] = []
    after: list[ChunkView] = []


# ---- read_section -------------------------------------------------------------
class ReadSectionInput(ToolInput):
    document_id: str
    page_number: int | None = Field(None, ge=1, description="PDF page.")
    sheet_name: str | None = Field(None, description="XLSX sheet name.")
    heading: str | None = Field(None, description="DOCX heading text, as shown by get_outline.")
    max_chunks: int = Field(15, ge=1, le=40)

    @model_validator(mode="after")
    def _exactly_one_selector(self) -> "ReadSectionInput":
        chosen = [v for v in (self.page_number, self.sheet_name, self.heading) if v is not None]
        if len(chosen) != 1:
            raise ValueError("give exactly one of page_number, sheet_name or heading")
        return self


class ReadSectionOutput(BaseModel):
    document_id: str
    filename: str
    selector: str
    total_chunks: int
    truncated: bool
    chunks: list[ChunkView]


# ---- calculate ----------------------------------------------------------------
class CalculateInput(ToolInput):
    expression: str = Field(min_length=1, max_length=200,
                            description="Arithmetic on plain numbers: + - * / % ** and parentheses. No names, no commas.")
    label: str | None = Field(None, max_length=200, description="What is being computed, for the trace.")


class CalculateOutput(BaseModel):
    expression: str
    result: str  # exact decimal arithmetic, shown as text


# ---- submit_finding -----------------------------------------------------------
class EvidenceInput(BaseModel):
    chunk_id: str = Field(description="Chunk the quote comes from.")
    quoted_text: str = Field(min_length=8, description="Text copied exactly from that chunk's content.")
    role: EvidenceRole
    relevance_reason: str = Field(min_length=1)
    support_assessment: str = Field(min_length=1, description="What this quote does and does not show.")


class SubmitFindingInput(ToolInput):
    title: str = Field(min_length=1, max_length=300)
    description: str = Field(min_length=1)
    classification: Classification
    severity: Severity
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_status: EvidenceStatus
    materiality_reasoning: str | None = None
    business_impact: str | None = None
    unresolved_questions: list[str] = Field(default_factory=list)
    evidence: list[EvidenceInput] = Field(min_length=1, max_length=12)


class StoredEvidence(BaseModel):
    evidence_id: str
    chunk_id: str
    filename: str
    locator: str
    quoted_text: str  # the exact text as stored in the source chunk


class SubmitFindingOutput(BaseModel):
    finding_id: str
    verification_status: str
    evidence: list[StoredEvidence]
    notes: list[str] = []


__all__ = [n for n in dir() if not n.startswith("_")] + ["DocumentOutline"]
