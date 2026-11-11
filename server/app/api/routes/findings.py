"""Read the findings a run saved (the discrepancies), grouped by category with their evidence."""
from collections import Counter

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.dependencies import get_session
from app.core.exceptions import NotFoundError
from app.models import Document, DocumentChunk, Evidence, Finding, Investigation
from app.schemas.claim import SourceRef
from app.schemas.finding import EvidenceOut, FindingOut, FindingsOut, InvestigationOut
from app.schemas.run_metrics import RunMetrics
from app.services.run_metrics import find_metrics

router = APIRouter(prefix="/api", tags=["findings"])

# contradictions first; the other categories follow in a fixed order, unknown ones last
CATEGORY_ORDER = ["contradiction", "contextual_difference", "missing_information", "insufficient_evidence", "consistent"]
SEVERITY_ORDER = ["critical", "high", "medium", "low", "informational"]


def _rank(order: list[str], value: str) -> int:
    return order.index(value) if value in order else len(order)


def _investigation_out(session: Session, inv: Investigation) -> InvestigationOut:
    count = session.scalar(select(func.count()).select_from(Finding).where(Finding.investigation_id == inv.id)) or 0
    return InvestigationOut(id=inv.id, name=inv.name, status=inv.status, started_at=inv.started_at,
                            completed_at=inv.completed_at, finding_count=count)


@router.get("/investigations", response_model=list[InvestigationOut])
def list_investigations(session: Session = Depends(get_session)) -> list[InvestigationOut]:
    """Runs that saved findings or are in progress, newest first."""
    rows = session.scalars(select(Investigation).order_by(Investigation.started_at.desc().nulls_last(), Investigation.id).limit(30))
    return [_investigation_out(session, inv) for inv in rows]


@router.get("/findings", response_model=FindingsOut)
def list_findings(investigation_id: str | None = None, session: Session = Depends(get_session)) -> FindingsOut:
    """Findings of one run (default: the newest), grouped by category, each with its verified quotes."""
    if investigation_id:
        inv = session.get(Investigation, investigation_id)
        if inv is None:
            raise NotFoundError("Investigation not found")
    else:
        inv = session.scalars(select(Investigation).order_by(Investigation.started_at.desc().nulls_last(), Investigation.id).limit(1)).first()
    if inv is None:
        return FindingsOut(investigation=None, counts={}, findings=[])

    findings = list(session.scalars(select(Finding).where(Finding.investigation_id == inv.id)))
    evidence_rows = session.execute(
        select(Evidence, DocumentChunk, Document.filename)
        .join(DocumentChunk, DocumentChunk.id == Evidence.chunk_id).join(Document, Document.id == DocumentChunk.document_id)
        .where(Evidence.finding_id.in_([f.id for f in findings] or [""]))
        .order_by(Evidence.created_at, Evidence.id)).all()
    by_finding: dict[str, list[EvidenceOut]] = {}
    for ev, chunk, filename in evidence_rows:
        by_finding.setdefault(ev.finding_id, []).append(EvidenceOut(
            role=ev.evidence_role, quoted_text=ev.quoted_text, relevance_reason=ev.relevance_reason,
            support_assessment=ev.support_assessment,
            source=SourceRef(document_id=chunk.document_id, filename=filename, chunk_id=chunk.id, content_type=chunk.content_type,
                             page_number=chunk.page_number, sheet_name=chunk.sheet_name, cell_reference=chunk.cell_reference,
                             heading=chunk.heading)))

    findings.sort(key=lambda f: (_rank(CATEGORY_ORDER, f.classification), _rank(SEVERITY_ORDER, f.severity), -f.confidence, f.title))
    out = [FindingOut(
        id=f.id, title=f.title, description=f.description, classification=f.classification, severity=f.severity,
        confidence=f.confidence, evidence_status=f.evidence_status, verification_status=f.verification_status,
        materiality_reasoning=f.materiality_reasoning, business_impact=f.business_impact,
        unresolved_questions=list(f.unresolved_questions or []), evidence=by_finding.get(f.id, [])) for f in findings]
    counts = dict(Counter(f.classification for f in findings))
    return FindingsOut(investigation=_investigation_out(session, inv),
                       counts=dict(sorted(counts.items(), key=lambda kv: _rank(CATEGORY_ORDER, kv[0]))), findings=out)


@router.get("/investigations/{investigation_id}/metrics", response_model=RunMetrics)
def investigation_metrics(investigation_id: str, session: Session = Depends(get_session)) -> RunMetrics:
    """Time, calls, tokens and latencies of a run, as in its results file (runs from before this existed have none)."""
    if session.get(Investigation, investigation_id) is None:
        raise NotFoundError("Investigation not found")
    metrics = find_metrics(investigation_id)
    if metrics is None:
        raise NotFoundError("No metrics were recorded for this run")
    return metrics
