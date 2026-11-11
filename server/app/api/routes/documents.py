import time
from dataclasses import asdict

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session, sessionmaker

from app.api.dependencies import (JobManager, Settings, get_embedder, get_job_manager, get_session,
                                  get_session_factory, get_settings)
from app.llm.embeddings import EmbeddingProvider
from app.schemas.document import DocumentListOut, DocumentOut, ScanResultOut
from app.schemas.job import JobOut
from app.services.document_service import DocumentService
from app.services.processing_service import ProcessingService

router = APIRouter(prefix="/api/documents", tags=["documents"])
PROCESS_JOB = "process_documents"


def _service(session: Session = Depends(get_session), settings: Settings = Depends(get_settings)) -> DocumentService:
    return DocumentService(session, settings)


@router.get("", response_model=DocumentListOut)
def list_documents(service: DocumentService = Depends(_service)) -> DocumentListOut:
    items = service.list_documents()
    return DocumentListOut(items=items, total=len(items))


@router.post("/scan", response_model=ScanResultOut)
def scan_documents(service: DocumentService = Depends(_service)) -> ScanResultOut:
    result = service.scan()
    docs = service.list_documents()
    return ScanResultOut(**result.__dict__, documents=docs)


@router.post("/process", response_model=JobOut, status_code=202)
def process_documents(
    jobs: JobManager = Depends(get_job_manager),
    factory: sessionmaker = Depends(get_session_factory),
    settings: Settings = Depends(get_settings),
    embedder: EmbeddingProvider = Depends(get_embedder),
) -> JobOut:
    """Extract, chunk and embed every pending document in the background."""

    def run(progress):
        started = time.perf_counter()
        with factory() as session:
            processing = ProcessingService(session, DocumentService(session, settings), embedder, progress,
                                           extract_workers=settings.extract_workers)
            results = processing.process_pending()
            return {"results": [asdict(r) for r in results],
                    "extract_phase_ms": processing.extract_phase_ms,
                    "embedding_latency": embedder.latency_summary().model_dump(),
                    "duration_ms": round((time.perf_counter() - started) * 1000)}

    return JobOut.model_validate(jobs.start(PROCESS_JOB, run), from_attributes=True)


@router.get("/{document_id}", response_model=DocumentOut)
def get_document(document_id: str, service: DocumentService = Depends(_service)) -> DocumentOut:
    return service.get_document(document_id)
