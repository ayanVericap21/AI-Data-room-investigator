"""Builds the configured LLM and embedding providers. No model names are hardcoded here:
everything comes from settings (environment variables)."""
from __future__ import annotations

from app.core.config import Settings
from app.llm.base import LLMClient
from app.llm.embeddings import EmbeddingProvider, MockEmbeddingProvider
from app.llm.errors import LLMConfigError
from app.llm.google import GoogleEmbeddingProvider, GoogleLLMClient
from app.llm.mock import MockLLMClient


def get_llm_client(settings: Settings) -> LLMClient:
    provider = settings.llm_provider.strip().lower()
    if provider == "mock":
        return MockLLMClient()
    if provider == "google":
        if not settings.llm_api_key:
            raise LLMConfigError("LLM_API_KEY is not set")
        if not settings.llm_model:
            raise LLMConfigError("LLM_MODEL is not set")
        return GoogleLLMClient(model=settings.llm_model, api_key=settings.llm_api_key)
    raise LLMConfigError(f"Unsupported LLM_PROVIDER '{settings.llm_provider}'")


def get_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """EMBEDDING_PROVIDER defaults to LLM_PROVIDER when blank; the API key is shared
    with the LLM only when both use the same provider."""
    provider = (settings.embedding_provider or settings.llm_provider).strip().lower()
    if provider == "mock":
        return MockEmbeddingProvider(dimensions=settings.embedding_dimensions)
    if provider == "google":
        if not settings.llm_api_key:
            raise LLMConfigError("LLM_API_KEY is not set (required for Google embeddings)")
        if not settings.embedding_model:
            raise LLMConfigError("EMBEDDING_MODEL is not set")
        return GoogleEmbeddingProvider(
            model=settings.embedding_model, api_key=settings.llm_api_key, dimensions=settings.embedding_dimensions
        )
    raise LLMConfigError(f"Unsupported EMBEDDING_PROVIDER '{provider}'")
