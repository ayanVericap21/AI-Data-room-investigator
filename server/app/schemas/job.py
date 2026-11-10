from datetime import datetime

from pydantic import BaseModel, ConfigDict


class JobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    name: str
    state: str  # idle | running | completed | failed
    message: str
    started_at: datetime | None = None
    finished_at: datetime | None = None
    result: dict | None = None
    error: str | None = None
