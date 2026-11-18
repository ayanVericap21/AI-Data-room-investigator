"""Application settings. Keys, the database URL and paths come from environment variables (and an optional .env file);
model names, providers and their options are fixed in code (see CODE_DEFINED).

Settings are a plain Pydantic model populated from ``os.environ`` so no extra
dependency is needed. Secrets (LLM_API_KEY) are only ever read here, server-side.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field

SERVER_DIR = Path(__file__).resolve().parents[2]


def _load_dotenv() -> None:
    """Minimal .env loader: real environment variables always win."""
    for candidate in (SERVER_DIR / ".env", SERVER_DIR.parent / ".env"):
        if not candidate.is_file():
            continue
        for raw in candidate.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
        return


class Settings(BaseModel):
    app_env: str = "development"
    app_port: int = 8000
    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/data_room"

    llm_provider: str = "google"  # groq | google | llamaextract | ollama (a model on this machine) | mock
    llm_model: str = "gemini-3.5-flash-lite"  # for llamaextract this is the tier; for the others a model name
    llm_requests_per_minute: int = Field(default=30, ge=1)  # the provider's request limit; Groq calls are spaced to stay under it
    google_requests_per_minute: int = Field(default=15, ge=0)  # Gemini request limit (15 on the free tier); calls are spaced to stay under it, 0 = no pacing (a paid key)
    # Two-step judging on Gemini: the smaller models screen every group, the bigger models decide the groups the screen calls findings. Each tier is an ordered chain:
    # a call goes to the first model that can take it, and to the next when one is out of quota or fails. The free tier gives Flash Lite models 15 requests a minute and
    # 500 a day, Flash models 5 a minute and 20 a day (failed attempts count), so the bigger tier makes few attempts and paces itself harder.
    judge_two_step: bool = False  # switched off for now (was True): every group is judged by the one settings model; set True to screen with the smaller models again
    llm_small_models: list[str] = Field(default_factory=lambda: ["gemini-3.1-flash-lite", "gemini-3.5-flash-lite"])
    llm_big_models: list[str] = Field(default_factory=lambda: ["gemini-3.6-flash", "gemini-3.8-flash"])
    google_big_requests_per_minute: int = Field(default=5, ge=0)
    google_big_max_attempts: int = Field(default=2, ge=1)
    llm_tokens_per_minute: int = Field(default=8000, ge=1)  # the provider's token limit (what Groq's response headers report for this key)
    # Groq models tried when the preferred one is rate limited or out of quota (each model has its own limits). Chat models only: the prompt-guard
    # models are classifiers and reject chat requests.
    llm_fallback_models: list[str] = Field(default_factory=lambda: ["openai/gpt-oss-20b", "qwen/qwen3.8-27b"])
    llm_base_url: str = "http://localhost:11434"  # where Ollama listens
    llm_context_tokens: int = Field(default=16384, ge=2048)  # context window requested from Ollama for every call
    # Set to false to refuse every hosted API (Google, LlamaExtract) and keep a run entirely on this machine.
    allow_third_party_apis: bool = True
    llm_api_key: str = Field(default="", repr=False)
    # Google keys from several accounts or projects, used in turn: a call that fails on one key is tried on the next. LLM_API_KEY is used when none of these is set.
    llm_api_key_1: str = Field(default="", repr=False)
    llm_api_key_2: str = Field(default="", repr=False)
    llm_api_key_3: str = Field(default="", repr=False)

    def google_api_keys(self) -> list[str]:
        """The Google keys to use, in order, without repeats and without blanks."""
        keys = [k.strip() for k in (self.llm_api_key_1, self.llm_api_key_2, self.llm_api_key_3) if k and k.strip()]
        return list(dict.fromkeys(keys)) or ([self.llm_api_key.strip()] if self.llm_api_key.strip() else [])

    llama_cloud_api_key: str = Field(default="", repr=False)  # LlamaExtract (only used with --extractor llamaextract)
    llama_cloud_project_id: str = ""  # optional
    # OCR of picture-only PDF pages through LlamaParse (uses the LlamaCloud key above; billed per page: cost_effective = 3 credits). Off when the key is empty or
    # allow_third_party_apis is false; set OCR_ENABLED=false to turn it off.
    ocr_enabled: bool = True
    ocr_tier: str = "cost_effective"

    embedding_provider: str = "local"  # runs on this machine
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Must match the vector(N) column on document_chunks.embedding (see migration 0002).
    embedding_dimensions: int = 384

    # Price per million tokens of the model in use, to calculate a run's cost (left empty: the cost is not calculated and the run says so).
    llm_input_price_per_mtok: float | None = None
    llm_output_price_per_mtok: float | None = None
    # Minimum similarity of a search hit for the investigation agent (empty: none).
    search_min_score: float | None = None

    max_investigation_iterations: int = 3
    max_llm_calls: int = 30
    max_investigation_duration_seconds: int = 900
    investigation_poll_interval: int = 10

    data_room_path: Path = Path("./data_room")
    # Documents parsed at the same time during processing (separate processes; 1 = one at a time).
    extract_workers: int = Field(default=4, ge=1)

    @property
    def ocr_config(self):
        """What a parsing worker needs to run OCR (an ``OcrConfig``), or None when OCR is off or cannot run (no key, third-party APIs not allowed)."""
        if not (self.ocr_enabled and self.allow_third_party_apis and self.llama_cloud_api_key):
            return None
        from app.llm.llama_parse import OcrConfig
        return OcrConfig(api_key=self.llama_cloud_api_key, project_id=self.llama_cloud_project_id, tier=self.ocr_tier)

    @property
    def data_room_dir(self) -> Path:
        """Absolute data room directory; relative values resolve against server/."""
        p = self.data_room_path
        return (p if p.is_absolute() else SERVER_DIR / p).resolve()


# Chosen in code, never from the environment, so a stale .env line cannot change them. Everything else (keys, the
# database URL, paths) is still read from the environment.
CODE_DEFINED = {"llm_provider", "llm_model", "llm_requests_per_minute", "judge_two_step", "llm_small_models", "llm_big_models", "google_big_requests_per_minute", "google_big_max_attempts", "google_requests_per_minute", "llm_tokens_per_minute", "llm_fallback_models", "llm_base_url", "llm_context_tokens", "allow_third_party_apis",
                "embedding_provider", "embedding_model", "embedding_dimensions"}


def load_settings() -> Settings:
    _load_dotenv()
    values = {}
    for name in Settings.model_fields:
        if name in CODE_DEFINED:
            continue
        env = os.environ.get(name.upper())
        if env not in (None, ""):
            values[name] = env
    return Settings(**values)


@lru_cache
def get_settings() -> Settings:
    return load_settings()
