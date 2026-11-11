"""Start and stop a test run and follow its log. The run itself uses the script's current configuration."""
from fastapi import APIRouter, Depends, Query

from app.api.dependencies import Settings, get_settings
from app.core.exceptions import AppError
from app.schemas.test_run import DataRoomFileOut, DataRoomFilesOut, StartRunIn, UpdatesOut
from app.services.extraction_service import SUPPORTED_EXTENSIONS
from app.workers.test_runs import RunStatus, TestRunManager, get_test_run_manager

router = APIRouter(prefix="/api", tags=["test-run"])


class InvalidDocumentsError(AppError):
    status_code = 422
    code = "invalid_documents"


def data_room_files(settings: Settings) -> list[str]:
    """Supported files of the data room, as paths relative to it (same rules as the document scan)."""
    root = settings.data_room_dir
    if not root.is_dir():
        return []
    return sorted(
        p.relative_to(root).as_posix() for p in root.rglob("*")
        if p.is_file() and not p.name.startswith((".", "~$")) and p.suffix.lower() in SUPPORTED_EXTENSIONS)


@router.get("/test-run/files", response_model=DataRoomFilesOut)
def list_files(settings: Settings = Depends(get_settings)) -> DataRoomFilesOut:
    return DataRoomFilesOut(files=[DataRoomFileOut(path=p, selected_by_default=not p.lower().endswith(".xlsm"))
                                   for p in data_room_files(settings)])


@router.post("/test-run/start", response_model=RunStatus, status_code=202)
def start_run(body: StartRunIn, settings: Settings = Depends(get_settings),
              manager: TestRunManager = Depends(get_test_run_manager)) -> RunStatus:
    known = set(data_room_files(settings))
    unknown = [d for d in body.documents if d not in known]
    if unknown:
        raise InvalidDocumentsError(f"Not documents of the data room: {', '.join(unknown[:5])}")
    return manager.start(sorted(set(body.documents)), judge_workers=body.judge_workers)


@router.post("/test-run/stop", response_model=RunStatus)
def stop_run(manager: TestRunManager = Depends(get_test_run_manager)) -> RunStatus:
    return manager.stop()


@router.get("/updates", response_model=UpdatesOut)
def updates(offset: int = Query(0, ge=0), manager: TestRunManager = Depends(get_test_run_manager)) -> UpdatesOut:
    lines, next_offset = manager.read_log(offset)
    return UpdatesOut(run=manager.status(), lines=lines, next_offset=next_offset)
