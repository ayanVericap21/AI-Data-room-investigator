"""Runtime settings. Everything comes from environment variables (DRI_*); secrets are never literals."""
from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel


class Settings(BaseModel):
    data_dir: Path = Path("data")          # uploaded files live in data_dir/uploads
    database_url: str | None = None        # DRI_DATABASE_URL: postgresql://... (docker compose) - required at runtime
    db_schema: str | None = None           # DRI_DB_SCHEMA: optional Postgres schema (isolation); default search_path
    # Browser origins allowed to call the API (the Streamlit UI uploads folders straight from the browser).
    cors_origins: list[str] = ["http://localhost:8501", "http://127.0.0.1:8501"]
    llm_provider: str = "anthropic"        # DRI_LLM_PROVIDER: anthropic | gemini
    model: str = "claude-sonnet-5-5"       # DRI_MODEL (required for gemini: no default, model names change)
    anthropic_api_key: str | None = None   # from ANTHROPIC_API_KEY only
    gemini_api_key: str | None = None      # from GEMINI_API_KEY (or GOOGLE_API_KEY) only

    # ingest
    chunk_max_chars: int = 1000
    ingest_workers: int = 4
    max_upload_bytes: int = 25 * 1024 * 1024
    io_retries: int = 2
    summary_batch_chars: int = 24_000      # passages per summarisation call; longer documents are map-reduced

    # investigation limits
    max_steps_per_hypothesis: int = 5
    max_hypotheses: int = 15
    max_claims_in_prompt: int = 300
    chars_per_extraction_batch: int = 12_000
    llm_retries: int = 2
    min_established_confidence: float = 0.5

    # baseline
    baseline_max_chars: int = 400_000

    # Approximate USD per 1M tokens (list prices for claude-sonnet-5-5 as of 2026-10; verify before relying on them,
    # and override with DRI_PRICE_IN / DRI_PRICE_OUT when you change DRI_MODEL).
    price_in_per_mtok: float = 2.0
    price_out_per_mtok: float = 10.0
    # Prompt-cache billing relative to the input price: 5-minute writes cost 1.25x, reads 0.1x (0.05x on some models).
    cache_write_multiplier: float = 1.25
    cache_read_multiplier: float = 0.1

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    def require_database_url(self) -> str:
        if not self.database_url:
            raise RuntimeError(
                "DRI_DATABASE_URL is not set. Start Postgres with `docker compose up -d db` and set "
                "DRI_DATABASE_URL=postgresql://<user>:<password>@localhost:5432/<db> (see README, Configuration).")
        return self.database_url


def get_settings() -> Settings:
    env = os.environ
    kwargs: dict = {}
    mapping = {
        "DRI_DATA_DIR": ("data_dir", Path),
        "DRI_DATABASE_URL": ("database_url", str),
        "DRI_DB_SCHEMA": ("db_schema", str),
        "DRI_MODEL": ("model", str),
        "DRI_PRICE_IN": ("price_in_per_mtok", float),
        "DRI_PRICE_OUT": ("price_out_per_mtok", float),
        "DRI_CACHE_READ_MULT": ("cache_read_multiplier", float),
        "DRI_MAX_STEPS": ("max_steps_per_hypothesis", int),
        "DRI_MAX_HYPOTHESES": ("max_hypotheses", int),
    }
    mapping["DRI_LLM_PROVIDER"] = ("llm_provider", str)
    if env.get("DRI_CORS_ORIGINS"):
        kwargs["cors_origins"] = [o.strip() for o in env["DRI_CORS_ORIGINS"].split(",") if o.strip()]
    for var, (field, cast) in mapping.items():
        if env.get(var):
            kwargs[field] = cast(env[var])
    if env.get("ANTHROPIC_API_KEY"):
        kwargs["anthropic_api_key"] = env["ANTHROPIC_API_KEY"]
    if env.get("GEMINI_API_KEY") or env.get("GOOGLE_API_KEY"):
        kwargs["gemini_api_key"] = env.get("GEMINI_API_KEY") or env["GOOGLE_API_KEY"]
    if kwargs.get("llm_provider", "anthropic") == "gemini":
        if not env.get("DRI_MODEL"):
            kwargs["model"] = ""  # forces a clear error instead of a stale default model name
        # The Anthropic list prices above do not apply. Without DRI_PRICE_IN/OUT, cost is reported as 0 (unpriced).
        kwargs.setdefault("price_in_per_mtok", 0.0)
        kwargs.setdefault("price_out_per_mtok", 0.0)
    return Settings(**kwargs)
