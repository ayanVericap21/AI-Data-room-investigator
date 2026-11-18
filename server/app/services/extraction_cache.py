"""The LlamaExtract response of a job, kept by (file hash, schema hash, prompt hash, tier, pages).

A run on a file whose content, extraction schema and extraction prompt are unchanged reads the stored response instead of starting a
job, so it costs no credits and takes no waiting. The response is stored raw: the checks after it (the value gate, the page-number
filter, chunk matching) run again on every run, so a change to them takes effect without paying for a new extraction. A different
tier, a different page range, an edited prompt or schema is a different key, so a stale response is never used.
Each call opens its own short session, because the extraction jobs run in threads.
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Callable

from sqlalchemy import delete, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.llm.llama_extract import LlamaJob
from app.models import ExtractionCacheEntry

logger = logging.getLogger(__name__)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def schema_digest(schema: dict[str, Any]) -> str:
    return digest(json.dumps(schema, sort_keys=True, separators=(",", ":"), ensure_ascii=False))


class ExtractionCache:
    def __init__(self, session_factory: Callable[[], Session], schema: dict[str, Any], prompt: str, tier: str, refresh: bool = False):
        self._factory = session_factory
        self.schema_hash, self.prompt_hash, self.tier = schema_digest(schema), digest(prompt), tier
        self.refresh = refresh  # never read, always write: replaces what is stored with a fresh extraction
        self.error: str | None = None  # the first database error; the run goes on without the cache and says so

    def _key(self, file_sha256: str, pages: str):
        return (ExtractionCacheEntry.file_sha256 == file_sha256, ExtractionCacheEntry.schema_hash == self.schema_hash,
                ExtractionCacheEntry.prompt_hash == self.prompt_hash, ExtractionCacheEntry.tier == self.tier, ExtractionCacheEntry.pages == pages)

    def _failed(self, what: str, exc: Exception) -> None:
        self.error = self.error or f"{what}: {exc.__class__.__name__}: {str(exc)[:200]}"
        logger.warning("Extraction cache unavailable", extra={"error": self.error})

    def get(self, file_sha256: str, pages: str = "") -> LlamaJob | None:
        """The stored job for this file and page range, or None (not stored, refresh requested, or the table cannot be read)."""
        if self.refresh:
            return None
        try:
            with self._factory() as session:
                row = session.scalar(select(ExtractionCacheEntry).where(*self._key(file_sha256, pages)))
                if row is None:
                    return None
                return LlamaJob(job_id=f"cache:{row.id}", status="COMPLETED", result=row.result, metadata=row.extract_metadata or {}, credits=0.0,
                                pages_billed=0, latency_ms=0, cached=True, credits_saved=float(row.credits or 0.0))
        except SQLAlchemyError as exc:
            self._failed("reading the extraction cache", exc)
            return None

    def put(self, file_sha256: str, pages: str, job: LlamaJob) -> None:
        """Keep a completed job. A failed or empty one is never stored."""
        if job.cached or job.status != "COMPLETED" or not isinstance(job.result, dict):
            return
        try:
            with self._factory() as session:
                session.execute(delete(ExtractionCacheEntry).where(*self._key(file_sha256, pages)))
                session.add(ExtractionCacheEntry(file_sha256=file_sha256, schema_hash=self.schema_hash, prompt_hash=self.prompt_hash, tier=self.tier, pages=pages,
                                                 result=job.result, extract_metadata=job.metadata or None, credits=job.credits, pages_billed=job.pages_billed))
                session.commit()
        except SQLAlchemyError as exc:
            self._failed("writing the extraction cache", exc)
