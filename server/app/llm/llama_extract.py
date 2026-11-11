"""Client for LlamaExtract (LlamaIndex Cloud) over plain HTTP: upload a file, run an extraction job with a
schema, poll until it ends, return the result and its per-field metadata (citations).

The API key is only read from settings (environment) and sent as a bearer header; it is never logged or put
in an error message. Documents are uploaded to LlamaIndex's cloud, so this client is only used when asked for.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

BASE_URL = "https://api.cloud.llamaindex.ai"
DONE = {"COMPLETED", "FAILED", "CANCELLED"}


class LlamaExtractError(Exception):
    """A failed request or job. ``code`` is the HTTP status when there was one (402 = out of credits)."""

    def __init__(self, message: str, code: int | None = None):
        super().__init__(message)
        self.code = code


class LlamaJob(BaseModel):
    job_id: str
    status: str
    result: dict[str, Any] | list[Any] | None = None  # extract_result: data matching the schema
    metadata: dict[str, Any] = Field(default_factory=dict)  # extract_metadata: field citations and confidences
    credits: float = 0.0
    pages_billed: int = 0
    latency_ms: int = 0
    error: str | None = None


class LlamaExtractClient:
    def __init__(self, api_key: str, project_id: str = "", tier: str = "cost_effective", timeout: float = 60.0,
                 poll_seconds: float = 3.0, max_wait_seconds: float = 600.0, http: httpx.Client | None = None):
        if not api_key:
            raise LlamaExtractError("LLAMA_CLOUD_API_KEY is not set")
        self.tier = tier
        self.project_id = project_id
        self.poll_seconds = poll_seconds
        self.max_wait = max_wait_seconds
        self._http = http or httpx.Client(base_url=BASE_URL, timeout=timeout)
        self._headers = {"Authorization": f"Bearer {api_key}"}

    # -- low level -------------------------------------------------------------
    def _params(self, **extra: Any) -> dict[str, Any]:
        params = dict(extra)
        if self.project_id:
            params["project_id"] = self.project_id
        return params

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = self._http.request(method, path, headers=self._headers, **kwargs)
        except httpx.HTTPError as exc:
            raise LlamaExtractError(f"Could not reach LlamaExtract: {exc.__class__.__name__}") from exc
        if response.status_code >= 400:
            detail = response.text[:300].replace("\n", " ")
            hint = " (out of credits)" if response.status_code == 402 else ""
            raise LlamaExtractError(f"LlamaExtract {method} {path} -> {response.status_code}{hint}: {detail}", response.status_code)
        return response.json()

    # -- the three steps -------------------------------------------------------
    def upload(self, path: Path) -> str:
        with open(path, "rb") as fh:
            data = self._request("POST", "/api/v1/beta/files", params=self._params(), files={"file": (path.name, fh)},
                                 data={"purpose": "extract"})
        file_id = data.get("id")
        if not file_id:
            raise LlamaExtractError("The upload response had no file id")
        return file_id

    def create_job(self, file_id: str, schema: dict[str, Any], system_prompt: str | None = None,
                   extra_configuration: dict[str, Any] | None = None) -> str:
        configuration: dict[str, Any] = {"tier": self.tier, "data_schema": schema, "cite_sources": True,
                                         **(extra_configuration or {})}
        if system_prompt:
            configuration["system_prompt"] = system_prompt
        data = self._request("POST", "/api/v2/extract", params=self._params(), json={"file_input": file_id, "configuration": configuration})
        job_id = data.get("id")
        if not job_id:
            raise LlamaExtractError("The extract response had no job id")
        return job_id

    def wait(self, job_id: str) -> dict[str, Any]:
        """Poll until the job ends; returns the final job JSON with metadata and usage."""
        started = time.monotonic()
        while True:
            job = self._request("GET", f"/api/v2/extract/{job_id}", params=self._params(expand=["extract_metadata", "usage"]))
            if job.get("status") in DONE:
                return job
            if time.monotonic() - started > self.max_wait:
                raise LlamaExtractError(f"Extraction job {job_id} did not finish within {self.max_wait:.0f}s")
            time.sleep(self.poll_seconds)

    def extract(self, path: Path, schema: dict[str, Any], **kwargs: Any) -> LlamaJob:
        started = time.monotonic()
        job = self.wait(self.create_job(self.upload(path), schema, **kwargs))
        usage = job.get("usage") or {}
        billed = (job.get("metadata") or {}).get("usage") or {}
        return LlamaJob(
            job_id=job.get("id", ""), status=job.get("status", ""), result=job.get("extract_result"),
            metadata=job.get("extract_metadata") or {}, credits=float(usage.get("credits") or 0),
            pages_billed=int(billed.get("num_pages_billed") or 0), latency_ms=round((time.monotonic() - started) * 1000),
            error=job.get("error_message"))
