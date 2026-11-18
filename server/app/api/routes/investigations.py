"""Start, stop and follow an investigation (the tool-using agent) from the API. Runs in the background like the other jobs; the findings it saves are read through
/api/findings. The agent runs on the app's own database, so ids carry over between runs."""
import threading
from datetime import datetime

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import sessionmaker

from app.agent import run_investigation
from app.api.dependencies import JobManager, Settings, get_job_manager, get_llm, get_optional_embedder, get_session_factory, get_settings
from app.llm.base import LLMClient
from app.llm.embeddings import EmbeddingProvider
from app.schemas.job import JobOut
from app.services.retrieval_service import RetrievalService

router = APIRouter(prefix="/api/investigations", tags=["investigations"])
INVESTIGATION_JOB = "investigation"
_stop = threading.Event()


class StartInvestigationIn(BaseModel):
    focus: str | None = Field(None, description="What to look at; empty reviews the whole data room.")
    max_steps: int | None = Field(None, ge=1, description="Step limit; empty means no limit.")


def _out(jobs: JobManager) -> JobOut:
    return JobOut.model_validate(jobs.get(INVESTIGATION_JOB), from_attributes=True)


@router.post("/start", response_model=JobOut, status_code=202)
def start(body: StartInvestigationIn, jobs: JobManager = Depends(get_job_manager), factory: sessionmaker = Depends(get_session_factory),
          settings: Settings = Depends(get_settings), llm: LLMClient = Depends(get_llm),
          embedder: EmbeddingProvider | None = Depends(get_optional_embedder)) -> JobOut:
    _stop.clear()

    def run(progress):
        prices = ((settings.llm_input_price_per_mtok, settings.llm_output_price_per_mtok)
                  if settings.llm_input_price_per_mtok is not None and settings.llm_output_price_per_mtok is not None else None)
        with factory() as session:
            result = run_investigation(
                session, llm, RetrievalService(session, embedder) if embedder is not None else None,
                name=f"investigation {datetime.now():%Y-%m-%d %H:%M}", focus=body.focus, max_steps=body.max_steps, should_stop=_stop.is_set,
                on_progress=lambda rec: progress(f"Step {rec.step}: {rec.tool or rec.action}", {}),
                search_min_score=settings.search_min_score, prices_per_mtok=prices)
            return {"investigation_id": result.investigation_id, "termination": result.termination, "findings": len(result.finding_ids),
                    "documents_not_opened": result.documents_not_opened, "documents_unreadable": result.documents_unreadable,
                    "total_tokens": result.token_usage.total_tokens, "duration_ms": result.duration_ms}

    return JobOut.model_validate(jobs.start(INVESTIGATION_JOB, run), from_attributes=True)


@router.post("/stop", response_model=JobOut)
def stop(jobs: JobManager = Depends(get_job_manager)) -> JobOut:
    """Ask the running investigation to stop at its next step; what it already saved is kept."""
    _stop.set()
    return _out(jobs)


@router.get("/status", response_model=JobOut)
def status(jobs: JobManager = Depends(get_job_manager)) -> JobOut:
    return _out(jobs)
