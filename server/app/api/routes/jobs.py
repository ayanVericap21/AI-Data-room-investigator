from fastapi import APIRouter, Depends

from app.api.dependencies import JobManager, get_job_manager
from app.api.routes.claims import EXTRACT_JOB
from app.api.routes.documents import PROCESS_JOB
from app.schemas.job import JobOut

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


@router.get("", response_model=list[JobOut])
def list_jobs(jobs: JobManager = Depends(get_job_manager)) -> list[JobOut]:
    return [JobOut.model_validate(j, from_attributes=True) for j in jobs.all([PROCESS_JOB, EXTRACT_JOB])]
