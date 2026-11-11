from pydantic import BaseModel, Field

from app.workers.test_runs import RunStatus


class DataRoomFileOut(BaseModel):
    path: str  # relative to the data room, with forward slashes
    selected_by_default: bool


class DataRoomFilesOut(BaseModel):
    files: list[DataRoomFileOut]


class StartRunIn(BaseModel):
    documents: list[str] = Field(min_length=1, description="Paths from /api/test-run/files to read in this run.")
    judge_workers: int = Field(1, ge=1, le=16, description="Pair-judging model calls made at the same time (1 = one after another).")


class UpdatesOut(BaseModel):
    """New log lines since ``offset``; ask again with ``next_offset`` (and start from 0 when ``run`` changes)."""

    run: RunStatus
    lines: list[str]
    next_offset: int
