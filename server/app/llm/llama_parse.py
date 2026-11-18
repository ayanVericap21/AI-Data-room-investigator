"""OCR for picture-only PDF pages through LlamaParse (LlamaIndex Cloud) over plain HTTP: upload the file, parse only the named pages, poll until the
job ends, return the text of each page. Same client, key handling and error type as LlamaExtract (the key is never logged).

Billed per page (cost_effective tier: 3 credits per page); only the pages named are sent for parsing, but the whole file is uploaded.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.llm.llama_extract import DONE, LlamaExtractClient, LlamaExtractError

logger = logging.getLogger(__name__)


class OcrConfig(BaseModel):
    """What a parsing worker needs to run OCR. Picklable, so it can be handed to a worker process; the key stays out of repr and logs."""

    api_key: str = Field(repr=False)
    project_id: str = ""
    tier: str = "cost_effective"


class OcrResult(BaseModel):
    pages: dict[int, str]  # 1-based page number in the file -> its text ("" when the page could not be read)
    credits: float = 0.0
    latency_ms: int = 0
    failed_pages: list[int] = Field(default_factory=list)


def page_ranges(pages: list[int]) -> str:
    """[1, 2, 3, 5, 7, 8] -> "1-3,5,7-8" (the format of LlamaParse's ``target_pages``)."""
    ordered = sorted(set(pages))
    parts: list[str] = []
    start = prev = ordered[0]
    for p in ordered[1:] + [None]:  # type: ignore[list-item]
        if p is not None and p == prev + 1:
            prev = p
            continue
        parts.append(str(start) if start == prev else f"{start}-{prev}")
        if p is not None:
            start = prev = p
    return ",".join(parts)


class LlamaParseClient(LlamaExtractClient):
    def parse_pages(self, path: Path, pages: list[int]) -> OcrResult:
        """OCR ``pages`` (1-based) of the file. Raises LlamaExtractError when the job fails or does not finish."""
        started = time.monotonic()
        wanted = sorted(set(pages))
        file_id = self.upload(path, purpose="parse")
        created = self._request("POST", "/api/v2/parse", params=self._params(), json={
            "file_id": file_id, "tier": self.tier, "version": "latest", "page_ranges": {"target_pages": page_ranges(wanted)}})
        job_id = created.get("id")
        if not job_id:
            raise LlamaExtractError("The parse response had no job id")
        data = self._wait_parse(job_id)
        job = data.get("job") or {}
        if job.get("status") != "COMPLETED":
            raise LlamaExtractError(f"Parse job {job_id} ended as {job.get('status')}: {job.get('error_message') or 'no reason given'}")
        returned = (data.get("markdown") or {}).get("pages") or []
        return OcrResult(pages=self._map_pages(wanted, returned), credits=float((job.get("usage") or {}).get("credits") or 0),
                         latency_ms=round((time.monotonic() - started) * 1000),
                         failed_pages=[w for w, r in zip(wanted, returned) if not r.get("success", True)] if len(returned) == len(wanted) else [])

    def _wait_parse(self, job_id: str) -> dict[str, Any]:
        started = time.monotonic()
        while True:
            data = self._request("GET", f"/api/v2/parse/{job_id}", params=self._params(expand=["markdown", "usage"]))
            if (data.get("job") or data).get("status") in DONE:
                return data
            if time.monotonic() - started > self.max_wait:
                raise LlamaExtractError(f"Parse job {job_id} did not finish within {self.max_wait:.0f}s")
            time.sleep(self.poll_seconds)

    @staticmethod
    def _map_pages(wanted: list[int], returned: list[dict[str, Any]]) -> dict[int, str]:
        """Page numbers of the result are matched to the requested pages. When the job returns one entry per requested page they are taken in
        order (right whether the service numbers them by the file or within the selection); otherwise only entries that carry a requested
        page number are used."""
        text = lambda r: (r.get("markdown") or "").strip() if r.get("success", True) else ""  # noqa: E731
        if len(returned) == len(wanted):
            return {w: text(r) for w, r in zip(wanted, returned)}
        by_number = {r.get("page_number"): text(r) for r in returned}
        found = {w: by_number[w] for w in wanted if w in by_number}
        if not found:
            raise LlamaExtractError(f"The parse result has {len(returned)} pages for the {len(wanted)} requested, and none of them is numbered as requested")
        return {w: found.get(w, "") for w in wanted}
