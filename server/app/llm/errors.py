"""Provider-independent LLM/embedding errors. Only transient errors are retried."""
from __future__ import annotations


class LLMError(Exception):
    """Base class for all LLM/embedding failures."""


class LLMConfigError(LLMError):
    """Missing/invalid configuration (e.g. no API key). Never retried."""


class LLMTransientError(LLMError):
    """Rate limit, timeout, 5xx, network blip. Retried with bounded backoff."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class LLMPermanentError(LLMError):
    """Auth failure, bad request, unknown model... Never retried; never silently
    replaced by another model."""


class LLMInvalidOutputError(LLMError):
    """Model output was not valid JSON / did not match the requested Pydantic schema."""
