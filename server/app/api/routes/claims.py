import time
from dataclasses import asdict

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session, sessionmaker

from app.api.dependencies import (JobManager, Settings, get_job_manager, get_llm, get_optional_embedder, get_session,
                                  get_session_factory, get_settings)
from app.core.exceptions import NotFoundError
from app.llm.base import LLMClient
from app.llm.embeddings import EmbeddingProvider
from app.models import Document, DocumentChunk
from app.schemas.claim import CandidateListOut, ChunkOut, ClaimListOut
from app.schemas.job import JobOut
from app.services.claim_service import ClaimService

router = APIRouter(prefix="/api", tags=["claims"])
EXTRACT_JOB = "extract_claims"


@router.get("/claims", response_model=ClaimListOut)
def list_claims(document_id: str | None = None, limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0),
                session: Session = Depends(get_session)) -> ClaimListOut:
    items, total = ClaimService(session).list_claims(document_id, limit, offset)
    return ClaimListOut(items=items, total=total)


@router.post("/claims/extract", response_model=JobOut, status_code=202)
def extract_claims(
    force: bool = False,
    jobs: JobManager = Depends(get_job_manager),
    factory: sessionmaker = Depends(get_session_factory),
    settings: Settings = Depends(get_settings),
    llm: LLMClient = Depends(get_llm),
) -> JobOut:
    """Extract claims from processed documents in the background (skips documents that already have claims)."""

    def run(progress):
        started = time.perf_counter()
        with factory() as session:
            summary = ClaimService(session, llm, max_llm_calls=settings.max_llm_calls, on_progress=progress) \
                .extract_all(force=force)
            return {**asdict(summary), "token_usage": llm.get_token_usage().model_dump(),
                    "llm_latency": llm.latency_summary().model_dump(),
                    "duration_ms": round((time.perf_counter() - started) * 1000)}

    return JobOut.model_validate(jobs.start(EXTRACT_JOB, run), from_attributes=True)


@router.get("/claims/candidates", response_model=CandidateListOut)
def candidate_pairs(min_topic_score: float = Query(0.5, ge=0, le=1), max_partners_per_claim: int = Query(5, ge=1, le=50),
                    min_embedding_similarity: float = Query(0.80, ge=0, le=1), use_embeddings: bool = True,
                    session: Session = Depends(get_session),
                    embedder: EmbeddingProvider | None = Depends(get_optional_embedder)) -> CandidateListOut:
    """Deterministic candidate comparisons between extracted claims (not findings). Embeddings
    add a synonym route when configured and ``use_embeddings`` is true."""
    pairs = ClaimService(session).candidate_pairs(
        embedder if use_embeddings else None, min_topic_score=min_topic_score,
        max_partners_per_claim=max_partners_per_claim, min_embedding_similarity=min_embedding_similarity)
    return CandidateListOut(items=pairs, total=len(pairs))


@router.get("/chunks/{chunk_id}", response_model=ChunkOut)
def get_chunk(chunk_id: str, session: Session = Depends(get_session)) -> ChunkOut:
    chunk = session.get(DocumentChunk, chunk_id)
    if chunk is None:
        raise NotFoundError(f"Chunk '{chunk_id}' not found")
    doc = session.get(Document, chunk.document_id)
    return ChunkOut(id=chunk.id, document_id=chunk.document_id, filename=doc.filename, content=chunk.content,
                    content_type=chunk.content_type, page_number=chunk.page_number, sheet_name=chunk.sheet_name,
                    cell_reference=chunk.cell_reference, heading=chunk.heading)
