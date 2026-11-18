"""Provider-independent LLM/embedding errors. Only transient errors are retried."""
from __future__ import annotations


class LLMError(Exception):
    """Base class for all LLM/embedding failures."""


class LLMConfigError(LLMError):
    """Missing/invalid configuration (e.g. no API key). Never retried."""


class LLMTransientError(LLMError):
    """Rate limit, timeout, 5xx, network blip. Retried with bounded backoff."""

    def __init__(self, message: str, retry_after: float | None = None, status: int | None = None):
        super().__init__(message)
        self.retry_after = retry_after
        self.status = status  # the provider's HTTP status (429, 503...); None for a network error, a timeout or an empty answer

    @property
    def status_label(self) -> str:
        return f"HTTP {self.status}" if self.status is not None else "no HTTP status (network error, timeout or empty answer)"


class LLMPermanentError(LLMError):
    """Auth failure, bad request, unknown model... Never retried; never silently
    replaced by another model."""


class LLMQuotaError(LLMPermanentError):
    """A usage quota that does not clear within a run (a daily request limit) is used up. Retrying only burns waits (and probably more of the quota), so it is never retried."""

    status = 429  # the provider's HTTP status, recorded for the report like a transient error's


class LLMInvalidOutputError(LLMError):
    """Model output was not valid JSON / did not match the requested Pydantic schema."""
