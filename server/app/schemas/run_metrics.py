"""The numbers of one test run (time, calls, tokens, latencies, counts), as written next to its results file."""
from __future__ import annotations

from pydantic import BaseModel


class PhaseMetrics(BaseModel):
    phase: str
    attempts: int
    failed: int
    prompt_tokens: int
    completion_tokens: int
    thinking_tokens: int
    avg_context_tokens: int
    max_context_tokens: int
    avg_ms: int
    p95_ms: int
    total_s: float


class StageTiming(BaseModel):
    stage: str
    seconds: float
    percent: float  # of the wall time


class LLMTotals(BaseModel):
    attempts: int = 0
    failed: int = 0
    retries: int = 0
    total_s: float = 0.0
    avg_ms: int = 0
    p95_ms: int = 0
    max_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    thinking_tokens: int = 0
    cached_tokens: int = 0
    total_tokens: int = 0
    credits: float = 0.0  # LlamaExtract credits used by grouping and judging (0 for other providers)
    min_context_tokens: int = 0
    avg_context_tokens: int = 0
    max_context_tokens: int = 0


class RunCounts(BaseModel):
    documents: int = 0
    documents_fully_read: int = 0
    chunks: int = 0
    claims_recorded: int = 0
    claims_rejected: int = 0
    candidate_pairs: int = 0
    extractor_credits: float = 0.0  # LlamaExtract credits used (0 for the LLM extractor)
    pairs_without_clustering: int = 0  # LLM pairing: every claim of a group with every other
    pairs_exact: int = 0
    pairs_judged: int = 0
    verdicts: dict[str, int] = {}
    findings_saved: int = 0
    findings_rejected: int = 0
    embedding_calls: int = 0
    embedding_texts: int = 0
    coverage_calls: int = 0
    coverage_claims_added: int = 0


class RunSettings(BaseModel):
    extractor: str = "llm"  # llm | llamaextract
    pairing: str
    vocabulary: str
    coverage_check: str
    batch_chunks: int | None = None
    batch_chars: int
    llm_model: str
    embedding_model: str


class RunMetrics(BaseModel):
    investigation_id: str | None = None
    config_name: str
    state: str  # completed | stopped | failed
    started_at: str
    finished_at: str
    wall_s: float
    results_file: str
    settings: RunSettings
    counts: RunCounts
    llm: LLMTotals
    phases: list[PhaseMetrics]
    stages: list[StageTiming]
