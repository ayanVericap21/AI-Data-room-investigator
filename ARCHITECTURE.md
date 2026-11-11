# Architecture

## Components
| Component | Where | Role |
|---|---|---|
| API server (FastAPI) | `app/api/` | upload, ingest jobs, investigations, findings, events, feedback, resume |
| Document store | `app/documents.py` | validates + writes uploads to `data/uploads/` (atomic, sha256 de-dup, sanitised names) |
| Ingest job | `app/jobs.py`, `app/ingest/` | `document_ingest` lifecycle modelled on the ai-service flow (below) |
| Data-room tools | `app/tools.py` | `list_documents` (+summaries), `get_document_summary`, `search_chunks` (full-text), `get_chunk`, `get_neighbors`, `verify_quote` |
| MCP server | `app/mcp_server/` | exposes those tools read-only over stdio; scoped to an investigation's documents |
| Investigator | `app/agent/` | LangGraph workflow, verification rules, persistence |
| Baseline | `app/baseline.py` | Approach A: one direct LLM pass over all passages |
| Eval | `eval/` | same cases + deterministic scoring for both approaches |
| UI (Streamlit) | `ui/app.py` | upload, job + investigation progress, evidence inspection, feedback, history |

## Ingest job (mirrors the ai-service `document_ingest` pattern)
`submit_job`: idempotency key (replay returns the same job; same key + different docs = 409) → documents exist (404) →
not already in a live job (409) → skip already-processed documents → queue. `run_job`: process-wide lock (analogue of the
per-deal lock), documents run concurrently and isolated (one failure never stops the others; the job fails if any
document failed), results stored per document with a `failure_code` (`EMPTY_TEXT`, `PARSE_ERROR`, ...). Only transient
`OSError` is retried. Jobs left `running` after a crash are marked failed on startup.
Deviation from ai-service: no Redis/arq (single process, DB-backed state), no Bedrock/Textract (scanned PDFs fail with `EMPTY_TEXT`
instead of OCR). Completion does not auto-start an investigation; the user starts it (requirement 2).

## Data model (Postgres via docker compose)
`documents` (incl. `summary`, `summary_status`), `chunks` (+ generated `tsv` tsvector with a GIN index for full-text search),
`ingest_jobs`, `investigations`, `claims`, `hypotheses`, `findings` (Pydantic JSON blob), `events` (append-only trace).
A chunk keeps `page`, `char_start`, `char_end` into that page's text, so a citation resolves back to an exact location
(tested invariant: `page_text[start:end] == chunk.text`). JSON-ish fields are `TEXT` holding Pydantic JSON; whether any should be
JSONB, or live in Mongo, is the architect's call - this keeps the schema minimal.

**Backends.** `app/db.py` picks the backend from the URL: `postgresql://` (real) or `sqlite:///` (FTS5; exists only so the default test
run needs no Docker). Application SQL is written once with `?` placeholders and translated; only chunk storage/search, DDL and
"insert returns id" are backend-specific, all inside `Database`. `DRI_DB_SCHEMA` isolates a Database in its own Postgres schema
(tests and the eval harness use throw-away schemas so they never touch real data). Postgres search is `to_tsquery('english', a | b)`
ranked by `ts_rank` (BM25 on SQLite) - ranking differs slightly between backends.
**Verification status:** the full suite (155 tests = 95 SQLite + the 64 backend-parametrized ones on Postgres) passes against a live
Postgres 17 container via `docker compose`, and a real API server + MCP subprocess ran ingest, tsvector search and `verify_quote`
on it (2026-11-11). Not yet exercised on Postgres: a live LLM-driven investigation (no API key in the build environment).

## Document summaries (before MCP)
After parse/chunk/index, each document is summarised (`DocSummary`: overview, topics, key figures; long documents are map-reduced).
Best-effort: no LLM / LLM failure leaves the document `processed` and searchable with `summary_status` `skipped` / `failed`.
The MCP server exposes them via `list_documents` and `get_document_summary`; the agent feeds the overview to hypothesis generation
and to every investigation step. **Summaries are orientation, never evidence**: citations are verified against source passages only,
so a quote copied from a summary fails verification (tested). Ingest-time LLM cost is recorded on the job and reported separately.

## Prompt caching (`app/llm.py`)
User turns are built as blocks ordered stable → volatile, with `cache_control` breakpoints at stability boundaries:
* investigation step: `[data-room overview ★] [hypothesis header] [evidence log ... ★]` + uncached `"Steps used: n/m"` suffix.
  The overview is identical for every step of every hypothesis; the evidence log is append-only so each step re-reads the previous
  step's prefix.
* baseline: the whole data room is the cached prefix (re-read by repeat runs within the 5-minute TTL).
* Not cached: extraction, hypothesis generation, support check, summaries (unique inputs, no reuse - a write premium with no reads).
Usage tracks cache write/read tokens separately; cost applies the write (1.25x) and read (0.1x) multipliers; `cache_hit_ratio` is shown in
the UI and eval report. The models only cache prefixes above a minimum length (512 tokens on the current models), so very small
prompts silently don't cache. **Not yet measured against the live API** - check `cache_read_tokens > 0` on a real run; expected savings
are modest for this workload (the large stable block is the overview + the growing evidence log).
Structured output uses `messages.parse` (`output_config.format`): forced `tool_choice` is rejected by the current models.

## LLM providers (`app/llm.py`)
Everything depends on the small `LLM` protocol (`structured(system, user, schema, purpose, prefix)` + `usage`); `create_llm(settings)`
picks `AnthropicLLM` or `GeminiLLM` from `DRI_LLM_PROVIDER`. `GeminiLLM` asks for JSON matching the Pydantic JSON schema and validates it
with the same retry path; transient 429/5xx get exponential backoff (free tiers rate-limit). Gemini caches repeated prefixes implicitly,
so `Block.cache` markers are only concatenated and cached tokens come from usage metadata (reported as cache reads, never writes).
Cost is `0` unless `DRI_PRICE_IN/OUT` are set (no stale price table). **Gemini adapter is tested with a faked client only** - not against
the live API: whether Gemini accepts every keyword in our schemas (nested models, enums, `Literal`, `ge/le`) is unverified.

## Agent workflow (LangGraph, `app/agent/runner.py`)
```
START ─route by saved phase─► extract_claims ─► hypothesize ─► pick_next ─► decide ⇄ run_tool
                                                                  ▲            │ conclude
                                                                  └─ finalize ◄┘            → complete → END
```
* **extract_claims** (LLM): per document, structured claims with subject/value/unit/**conditions**/verbatim quote.
* **hypothesize** (LLM + heuristic): which claims to compare; deterministic cross-document subject-overlap candidates are
  added so recall does not rest on one LLM decision.
* **decide / run_tool** (LLM chooses, code executes): search or get_neighbors through MCP, or conclude. Each step records the
  model's *belief*; changes are logged as `belief_changed` (hypothesis revision).
* **finalize**: verify citations → rule check → LLM support check → persist.

### What needs AI reasoning vs deterministic logic
| AI reasoning | Deterministic code |
|---|---|
| Extracting claims and their conditions | Phase routing, resume, step/hypothesis caps, duplicate-call guard |
| Choosing which claims to compare | Heuristic candidate generation, de-duplication, id validation |
| Choosing the next search / when evidence suffices | Tool execution, retrieval (BM25), quote existence check (`verify_quote`) |
| Classifying contradiction vs contextual vs unverifiable; revising belief | ≥2 distinct *verified* passages required, otherwise downgrade to `insufficient_evidence` |
| Judging whether quotes support the explanation | Confidence threshold → `established` vs `unresolved`; persistence, retries, failure states |

## Evidence and verification
Every claim and every citation is checked with `verify_quote` (via MCP): whitespace/case/quote-insensitive substring of the
cited passage; a quote found in a different passage of the same document is re-attributed, anything else is flagged
`verified=false` and never counts as support. A comparison finding needs ≥2 verified distinct passages and a passing support
check, else it is downgraded and listed as an unresolved question with the original model conclusion recorded in uncertainties.

## State, failure, intervention
* Statuses: investigation `queued|running|completed|partial|failed|interrupted`; hypothesis `pending|running|done|failed`.
* LLM output that fails validation is retried (`llm_retries`); a failed hypothesis becomes an unresolved finding and the run
  ends `partial`; failed extraction on every document ends `failed`; hypothesis generation failure degrades to heuristics.
* Resume: `POST /investigations/{id}/resume` continues from the saved phase; hypotheses left `running` are retried;
  failed ones are re-opened. Interrupted runs are detected on startup.
* User intervention: `POST /investigations/{id}/feedback` with a `finding_id` (+ clarification note) re-opens that hypothesis
  with the note in the agent's prompt, supersedes (does not delete) the old finding; or with a `question` adds a new hypothesis.

## Cross-repo / production notes
Standalone prototype: nothing here is imported by or changes vericap-ai-core / skills / ai-service contracts. New Python dependency
beyond the first cut: `psycopg[binary]` (Postgres driver). If this is
folded into ai-service, the job lifecycle maps to the existing arq/Redis flow and the chunk/claim schemas would live in ai-core
(Pydantic, versioned artifact) - that is a contract change needing a version bump for consumers.

## Known limitations
No OCR, no tables/XLSX/DOCX parsing, keyword (not semantic) retrieval, a claim and its counter-claim in the *same* passage cannot
count as two passages, single-process locking, `max_claims_in_prompt` truncation for very large rooms, scoring match rule is
keyword/document based (unmatched findings are listed for manual review rather than auto-judged).
