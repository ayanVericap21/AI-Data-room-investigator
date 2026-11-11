"""Application settings. Keys, the database URL and paths come from environment variables (and an optional .env file);
model names, providers and their options are fixed in code (see CODE_DEFINED).

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

    llm_provider: str = "llamaextract"  # llamaextract | google | ollama (a model on this machine) | mock
    llm_model: str = "cost_effective"  # for llamaextract this is the tier; for the others a model name
    llm_base_url: str = "http://localhost:11434"  # where Ollama listens
    llm_context_tokens: int = Field(default=16384, ge=2048)  # context window requested from Ollama for every call
    # Set to false to refuse every hosted API (Google, LlamaExtract) and keep a run entirely on this machine.
    allow_third_party_apis: bool = True
    llm_api_key: str = Field(default="", repr=False)
    llama_cloud_api_key: str = Field(default="", repr=False)  # LlamaExtract (only used with --extractor llamaextract)
    llama_cloud_project_id: str = ""  # optional

    embedding_provider: str = "local"  # runs on this machine
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Must match the vector(N) column on document_chunks.embedding (see migration 0002).
    embedding_dimensions: int = 384

    max_investigation_iterations: int = 3
    max_llm_calls: int = 30
    max_investigation_duration_seconds: int = 900
    investigation_poll_interval: int = 10

    data_room_path: Path = Path("./data_room")
    # Documents parsed at the same time during processing (separate processes; 1 = one at a time).
    extract_workers: int = Field(default=4, ge=1)

    @property
    def data_room_dir(self) -> Path:
        """Absolute data room directory; relative values resolve against server/."""
        p = self.data_room_path
        return (p if p.is_absolute() else SERVER_DIR / p).resolve()


# Chosen in code, never from the environment, so a stale .env line cannot change them. Everything else (keys, the
# database URL, paths) is still read from the environment.
CODE_DEFINED = {"llm_provider", "llm_model", "llm_base_url", "llm_context_tokens", "allow_third_party_apis",
                "embedding_provider", "embedding_model", "embedding_dimensions"}


def load_settings() -> Settings:
    _load_dotenv()
    values = {}
    for name in Settings.model_fields:
        if name in CODE_DEFINED:
            continue
        env = os.environ.get(name.upper())
        if env not in (None, ""):
            values[name] = env
    return Settings(**values)


@lru_cache
def get_settings() -> Settings:
    return load_settings()
