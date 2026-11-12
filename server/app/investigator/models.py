"""Schemas shared by the investigator pipeline: typed facts, the case model, worker plans and outcomes, verifier verdicts.

Every structured value that crosses a function boundary is a Pydantic model (or a dataclass for purely internal records).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field, field_validator

# ---- typed facts -------------------------------------------------------------------------------------------------------------------
FACT_TYPES = ("observation", "constraint", "event", "assertion", "reference", "definition", "conclusion")
MODALITIES = ("must", "planned", "actual", "reported", "estimated")
VALUE_TYPES = ("quantity", "date", "identifier", "name", "state", "statement")


class TypedFact(BaseModel):
    """One fact as the extractor returns it. ``quote`` is the sentence or line that states it, copied from the document."""

    type: str = ""
    subject: str = ""
    attribute: str = ""
    operator: str = "="
    value: str | int | float | None = None
    unit: str | None = None
    scope: str | None = None
    when: str | None = None
    modality: str | None = None
    aggregate_of: str | None = None
    value_type: str | None = None
    quote: str = ""

    @field_validator("subject", "attribute", "operator", "quote", mode="before")
    @classmethod
    def _text(cls, v):
        return "" if v is None else str(v).strip()

    @property
    def kind(self) -> str | None:
        kind = (self.type or "").strip().lower()
        return kind if kind in FACT_TYPES else None

    @property
    def mode(self) -> str | None:
        mode = (self.modality or "").strip().lower()
        return mode if mode in MODALITIES else None

    @property
    def vtype(self) -> str | None:
        kind = (self.value_type or "").strip().lower()
        return kind if kind in VALUE_TYPES else None


class FactBatch(BaseModel):
    facts: list[TypedFact] = Field(default_factory=list)


@dataclass
class Fact:
    """A fact that was tied to a stored chunk and verified against its text."""

    id: str
    document_id: str
    chunk_id: str
    type: str
    subject: str
    attribute: str
    operator: str
    value: str
    unit: str | None
    scope: str | None
    when: str | None
    modality: str | None
    aggregate_of: str | None
    value_type: str | None
    quote: str  # exact text of the chunk (verbatim span)

    def line(self) -> str:
        parts = [f"{self.id} [{self.type}]", f"{self.subject or '-'} | {self.attribute or '-'} {self.operator} {self.value}" + (f" {self.unit}" if self.unit else "")]
        if self.scope:
            parts.append(f"scope: {self.scope}")
        if self.when:
            parts.append(f"when: {self.when}")
        if self.modality:
            parts.append(f"modality: {self.modality}")
        return " | ".join(parts)


# ---- the case model -----------------------------------------------------------------------------------------------------------------
DOC_ROLES = ("authority", "primary_record", "derived_analysis", "narrative", "event_register", "other")


class DocRole(BaseModel):
    document: str = ""
    role: str = "other"
    summary: str = ""

    @field_validator("role", mode="before")
    @classmethod
    def _role(cls, v):
        v = str(v or "other").strip().lower().replace(" ", "_")
        return v if v in DOC_ROLES else "other"


class IdPattern(BaseModel):
    """A kind of identifier the room uses (sample, site, person code ...), with how to find and normalise it."""

    entity_type: str = ""
    regex: str = ""
    description: str = ""
    case_fold: bool = True  # compare without case
    strip_separators: bool = True  # compare without - _ . / and spaces
    pad_last_number_to: int = 0  # zero-pad the last run of digits to this width (CF-IR06-3 becomes CF-IR06-03); 0 = leave


class AliasGroup(BaseModel):
    canonical: str = ""
    entity_type: str = ""
    aliases: list[str] = Field(default_factory=list)


class Definition(BaseModel):
    metric: str = ""
    definition: str = ""
    document: str = ""


class UnitEquivalence(BaseModel):
    unit: str = ""
    equals: str = ""


class TimeConvention(BaseModel):
    document: str = ""
    convention: str = ""


class CaseModel(BaseModel):
    """What the room is about and how its documents relate; stored as an editable JSON file."""

    case_summary: str = ""
    roles: list[DocRole] = Field(default_factory=list)
    id_patterns: list[IdPattern] = Field(default_factory=list)
    aliases: list[AliasGroup] = Field(default_factory=list)
    definitions: list[Definition] = Field(default_factory=list)
    units: list[UnitEquivalence] = Field(default_factory=list)
    time_conventions: list[TimeConvention] = Field(default_factory=list)


class AliasProposal(BaseModel):
    aliases: list[AliasGroup] = Field(default_factory=list)


# ---- finding taxonomy ---------------------------------------------------------------------------------------------------------------
# taxonomy key -> (label shown in the title, database classification when there are two or more passages, with fewer)
TAXONOMY: dict[str, tuple[str, str, str]] = {
    "direct_contradiction": ("direct contradiction", "contradiction", "insufficient_evidence"),
    "contextual_difference": ("contextual difference", "contextual_difference", "insufficient_evidence"),
    "temporal_inconsistency": ("temporal inconsistency", "contradiction", "insufficient_evidence"),
    "constraint_violation": ("constraint violation", "contradiction", "insufficient_evidence"),
    "reconciliation_gap": ("reconciliation gap", "missing_information", "missing_information"),
    "missing_evidence": ("missing evidence", "missing_information", "missing_information"),
    "unsupported_conclusion": ("unsupported conclusion", "insufficient_evidence", "insufficient_evidence"),
    "data_integrity_issue": ("data integrity issue", "contradiction", "insufficient_evidence"),
}
SEVERITIES = ("critical", "high", "medium", "low", "informational")
SEVERITY_ORDER = list(reversed(SEVERITIES))  # weakest first, so max() picks the most severe

FAMILIES = {
    "A": "constraint compliance", "B": "set and chain reconciliation", "C": "temporal consistency", "D": "recomputation",
    "E": "universal and negative claims", "F": "data integrity", "G": "reference resolution", "H": "conclusion audit",
    "I": "semantic value comparison",
}


# ---- workers --------------------------------------------------------------------------------------------------------------------------
class ToolCall(BaseModel):
    """One tool call a worker asks for. Only the fields of the chosen tool are used."""

    task_id: str = ""
    tool: str = ""  # sql | search | read | find_entity | timeline | get_facts
    sql: str | None = None
    query: str | None = None
    document: str | None = None
    location: str | None = None
    entity: str | None = None
    window_start: str | None = None
    window_end: str | None = None


class WorkerPlan(BaseModel):
    note: str = ""
    calls: list[ToolCall] = Field(default_factory=list)


class EvidenceRef(BaseModel):
    """Where a candidate's evidence is. ``kind`` row: a row of a loaded table (table + ref). ``kind`` quote: text copied from a document."""

    kind: str = "quote"  # row | quote
    table: str | None = None
    ref: str | None = None
    document: str | None = None
    quote: str | None = None
    shows: str = ""  # what this evidence shows
    role: str = "supporting"  # supporting | contradicting | contextual | insufficient


class CandidateOut(BaseModel):
    type: str = "direct_contradiction"  # a TAXONOMY key
    title: str = ""
    description: str = ""
    severity: str = "medium"
    confidence: float = 0.5
    status: str = "definitive"  # definitive | needs_investigation
    evidence_needed: list[str] = Field(default_factory=list)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    computed: str = ""  # the query or arithmetic behind the finding and its result
    entities: list[str] = Field(default_factory=list)
    root_cause: str = ""

    @field_validator("confidence", mode="before")
    @classmethod
    def _conf(cls, v):
        try:
            return max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            return 0.5


class TaskResult(BaseModel):
    task_id: str = ""
    status: str = "passed"  # finding | passed | untestable | needs_more
    reason: str = ""
    candidates: list[CandidateOut] = Field(default_factory=list)


class WorkerOutcome(BaseModel):
    results: list[TaskResult] = Field(default_factory=list)


# ---- verifier -------------------------------------------------------------------------------------------------------------------------
VERDICTS = ("confirmed", "contextual_difference", "unsupported", "missing_evidence", "rejected")


class QuoteCheck(BaseModel):
    index: int = 0
    supports: bool = True
    note: str = ""


class VerifierVerdict(BaseModel):
    candidate: int = 0  # the number shown for the candidate
    verdict: str = "rejected"
    reasons: str = ""
    explanation_found: str = ""
    resolving_evidence: str = ""
    severity: str | None = None
    quotes: list[QuoteCheck] = Field(default_factory=list)

    @field_validator("verdict", mode="before")
    @classmethod
    def _verdict(cls, v):
        v = str(v or "rejected").strip().lower().replace(" ", "_")
        return v if v in VERDICTS else "rejected"


class VerifierBatch(BaseModel):
    verdicts: list[VerifierVerdict] = Field(default_factory=list)


# ---- consolidation ----------------------------------------------------------------------------------------------------------------------
class RootCauseGroup(BaseModel):
    members: list[int] = Field(default_factory=list)
    title: str = ""
    description: str = ""


class ConsolidationPlan(BaseModel):
    groups: list[RootCauseGroup] = Field(default_factory=list)


class ReferenceKind(BaseModel):
    number: int = 0
    kind: str = "unclear"  # expected_artifact | external_standard | internal_part | unclear
    reason: str = ""


class ReferenceKinds(BaseModel):
    items: list[ReferenceKind] = Field(default_factory=list)


# ---- internal records -------------------------------------------------------------------------------------------------------------------
@dataclass
class ResolvedEvidence:
    chunk_id: str
    document_id: str
    filename: str
    location: str
    quote: str  # exact text of the chunk
    role: str
    shows: str


@dataclass
class Candidate:
    """A finding that has not yet survived the verifier."""

    family: str
    type: str
    title: str
    description: str
    severity: str
    confidence: float
    status: str
    evidence_needed: list[str]
    evidence: list[ResolvedEvidence]
    computed: str = ""
    entities: list[str] = field(default_factory=list)
    root_cause: str = ""
    task_id: str = ""
    source: str = "worker"  # worker | code | pairing
    finding_id: str | None = None  # set when a finding was already saved for it (family I)
    verdict: str = ""
    verifier_notes: str = ""
    linked: list[str] = field(default_factory=list)  # titles of symptoms merged into this one
    absorbed: list[str] = field(default_factory=list)  # ids of saved findings (family I) merged into this candidate, to be removed

    def key(self) -> frozenset:
        return frozenset((e.chunk_id, " ".join(e.quote.split())) for e in self.evidence)


@dataclass
class TaskRecord:
    """One line of the audit log: what was asked, and how it ended."""

    task_id: str
    family: str
    title: str
    status: str = "pending"  # finding | passed | untestable | error | skipped
    reason: str = ""
    calls: int = 0
    candidates: int = 0


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    return obj
