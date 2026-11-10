"""FastAPI dependencies. Providers are built per request so configuration errors (e.g. a
missing API key) surface immediately as a clear API error, and tests can override them."""
from __future__ import annotations

from fastapi import Depends
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings, get_settings
from app.core.database import get_session, get_sessionmaker
from app.llm.base import LLMClient
from app.llm.embeddings import EmbeddingProvider
from app.llm.factory import get_embedding_provider, get_llm_client
from app.workers.jobs import JobManager, get_job_manager

__all__ = [
    "Settings", "get_settings", "get_session", "get_session_factory", "get_llm", "get_embedder",
    "get_job_manager", "JobManager", "Session",
]


def get_session_factory() -> sessionmaker[Session]:
    return get_sessionmaker()


def get_llm(settings: Settings = Depends(get_settings)) -> LLMClient:
    return get_llm_client(settings)


def get_embedder(settings: Settings = Depends(get_settings)) -> EmbeddingProvider:
    return get_embedding_provider(settings)
