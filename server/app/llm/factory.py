"""Builds the configured LLM and embedding providers. No model names are hardcoded here:
everything comes from settings (environment variables)."""
from __future__ import annotations

from app.core.config import Settings
from app.llm.base import LLMClient
from app.llm.embeddings import EmbeddingProvider, MockEmbeddingProvider
from app.llm.errors import LLMConfigError
from app.llm.google import GoogleEmbeddingProvider, GoogleLLMClient
from app.llm.llama_extract import LlamaExtractClient
from app.llm.llama_llm import LlamaExtractLLMClient
from app.llm.local import DEFAULT_LOCAL_MODEL, LocalEmbeddingProvider
from app.llm.mock import MockLLMClient
from app.llm.ollama import OllamaLLMClient

THIRD_PARTY_OFF = ("Third-party APIs are disabled (ALLOW_THIRD_PARTY_APIS is not true). "
                   "Use LLM_PROVIDER=ollama with a local model, or set ALLOW_THIRD_PARTY_APIS=true to use {what}.")


def get_llm_client(settings: Settings) -> LLMClient:
    provider = settings.llm_provider.strip().lower()
    if provider == "mock":
        return MockLLMClient()
    if provider == "llamaextract":  # every call goes through LlamaExtract (LlamaIndex Cloud)
        if not settings.allow_third_party_apis:
            raise LLMConfigError(THIRD_PARTY_OFF.format(what="LlamaExtract"))
        if not settings.llama_cloud_api_key:
            raise LLMConfigError("LLAMA_CLOUD_API_KEY is not set")
        return LlamaExtractLLMClient(LlamaExtractClient(settings.llama_cloud_api_key, project_id=settings.llama_cloud_project_id,
                                                          tier=settings.llm_model or "cost_effective"))
    if provider == "ollama":
        if not settings.llm_model:
            raise LLMConfigError("LLM_MODEL is not set (the name of a model Ollama has, e.g. from `ollama list`)")
        return OllamaLLMClient(model=settings.llm_model, base_url=settings.llm_base_url, context_tokens=settings.llm_context_tokens)
    if provider == "google":
        if not settings.allow_third_party_apis:
            raise LLMConfigError(THIRD_PARTY_OFF.format(what="Google"))
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
    if provider == "local":  # runs on this machine, no API key and no quota
        return LocalEmbeddingProvider(model=settings.embedding_model or DEFAULT_LOCAL_MODEL,
                                      dimensions=settings.embedding_dimensions)
    if provider == "google":
        if not settings.allow_third_party_apis:
            raise LLMConfigError(THIRD_PARTY_OFF.format(what="Google embeddings"))
        if not settings.llm_api_key:
            raise LLMConfigError("LLM_API_KEY is not set (required for Google embeddings)")
        if not settings.embedding_model:
            raise LLMConfigError("EMBEDDING_MODEL is not set")
        return GoogleEmbeddingProvider(
            model=settings.embedding_model, api_key=settings.llm_api_key, dimensions=settings.embedding_dimensions
        )
    raise LLMConfigError(f"Unsupported EMBEDDING_PROVIDER '{provider}'")
