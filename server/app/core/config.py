"""Application settings, loaded from environment variables (and an optional .env file).

Settings are a plain Pydantic model populated from ``os.environ`` so no extra
dependency is needed. Secrets (LLM_API_KEY) are only ever read here, server-side.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field

SERVER_DIR = Path(__file__).resolve().parents[2]


def _load_dotenv() -> None:
    """Minimal .env loader: real environment variables always win."""
    for candidate in (SERVER_DIR / ".env", SERVER_DIR.parent / ".env"):
        if not candidate.is_file():
            continue
        for raw in candidate.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
        return


class Settings(BaseModel):
    app_env: str = "development"
    app_port: int = 8000
    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/data_room"

    llm_provider: str = "google"
    llm_model: str = ""
    llm_api_key: str = Field(default="", repr=False)

    embedding_provider: str = ""
    embedding_model: str = ""
    # Must match the vector(N) column on document_chunks.embedding (see migration 0002).
    embedding_dimensions: int = 1536

    max_investigation_iterations: int = 3
    max_llm_calls: int = 30
    max_investigation_duration_seconds: int = 900
    investigation_poll_interval: int = 10

    data_room_path: Path = Path("./data_room")

    @property
    def data_room_dir(self) -> Path:
        """Absolute data room directory; relative values resolve against server/."""
        p = self.data_room_path
        return (p if p.is_absolute() else SERVER_DIR / p).resolve()


def load_settings() -> Settings:
    _load_dotenv()
    values = {}
    for name in Settings.model_fields:
        env = os.environ.get(name.upper())
        if env not in (None, ""):
            values[name] = env
    return Settings(**values)


@lru_cache
def get_settings() -> Settings:
    return load_settings()
