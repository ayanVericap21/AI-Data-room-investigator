# Data Room Investigator

Upload a data room → ingest it (parse, chunk, **summarise**) → an agent investigates discrepancies through MCP tools → evidence-backed findings you can challenge.

```
Upload (API) → data/uploads/ → ingest job (parse → chunk → index → summarise) → Postgres
                                                                                    │
                          findings ◄── LangGraph investigator ◄── MCP server (search / passages / summaries / verify_quote)
```

## Setup
Python 3.11+ (developed on 3.14 under WSL Ubuntu, because the Windows host's Application Control policy blocks `pydantic_core`).

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

### Configuration (environment variables - never put real values in a committed file)
| Variable | Purpose |
|---|---|
| `POSTGRES_PASSWORD` | **required** by `docker compose` (no default on purpose). Optional: `POSTGRES_USER` (default `dri`), `POSTGRES_DB` (default `dri`), `POSTGRES_PORT` (host port, default 5432 - use another if Windows reserves it). Compose reads them from your shell or a gitignored `.env`. Generate the password without a trailing newline: `python -c "import secrets,sys;sys.stdout.write(secrets.token_hex(16))"`. |
| `DRI_DATABASE_URL` | **required** to run the API/MCP server: `postgresql://<user>:<password>@localhost:5432/<db>` |
| `DRI_LLM_PROVIDER` | `anthropic` (default) or `gemini` |
| `ANTHROPIC_API_KEY` | needed for summaries, investigations and the eval with the Anthropic provider. Without a key, ingest still works and summaries are `skipped`. |
| `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) | same, for `DRI_LLM_PROVIDER=gemini`. A key (free tier available) comes from https://aistudio.google.com/apikey |
| `DRI_MODEL` | Anthropic default `claude-sonnet-5-5`. **Required for Gemini** (no default - model names change; see https://ai.google.dev/gemini-api/docs/models) |
| `DRI_PRICE_IN` / `DRI_PRICE_OUT` | USD per 1M tokens for the model. Anthropic defaults: $2 / $10 (`claude-sonnet-5-5` list price, 2026-10 - verify). **Gemini defaults to 0 = unpriced**: cost shows $0 until you set these. `DRI_CACHE_READ_MULT` (default 0.1) |
| `DRI_DATA_DIR`, `DRI_DB_SCHEMA`, `DRI_MAX_STEPS`, `DRI_MAX_HYPOTHESES` | optional |
| `DRI_TEST_DATABASE_URL` | optional: also run the test suite against Postgres |

## Run
```bash
docker compose up -d db                                    # Postgres 17 (waits for healthcheck)
uvicorn app.api.main:app_factory --factory --port 8000     # API (spawns the MCP server per investigation)
streamlit run ui/app.py                                    # UI (DRI_API_URL if the API is elsewhere)
python -m app.mcp_server                                   # MCP server standalone (stdio; DRI_DATABASE_URL, DRI_DOC_IDS)
```

## How it works in the UI (one page, four steps)
1. **Upload** - pick a folder (subfolders included) or individual files.
2. **Preprocessing** - starts automatically on upload, no button: parse → chunk → index → summarise. The table shows each file's
   status (queued / parsing & chunking / summarising / ready / failed). Anything left unprocessed is picked up again when the API starts.
3. **MCP server** - live status (a real handshake with the stdio server), the tools it exposes, and how many documents it serves.
4. **Investigate** - one button, enabled when preprocessing is finished; then progress, MCP tool calls, and findings with evidence.

## Uploading documents
In the UI (**1 · Upload**): **Upload a folder** picks a folder and uploads every PDF/TXT/MD/CSV inside it *and its subfolders*
(hidden folders such as `.git`, `node_modules`, and other file types are skipped and counted). Each document keeps its relative path
as its name (`dataroom/finance/2025/report.txt`), so same-named files in different folders stay distinct and citations show where
a passage came from. Individual files can still be uploaded with the second uploader. Identical content is stored once.
The folder picker runs in your browser and posts to the API directly, so the API must allow the UI's origin
(`DRI_CORS_ORIGINS`, default `http://localhost:8501`) and be reachable from the browser (`DRI_PUBLIC_API_URL`, default `DRI_API_URL`).
The API has no authentication: run it on `127.0.0.1` (`uvicorn ... --host 127.0.0.1`).
Files are stored in `data/uploads/`. Not supported: DOCX/XLSX, scanned PDFs without a text layer, zip archives.

## Test
```bash
node --test tests/js/folder_upload.test.js   # folder-upload JS logic (Node >= 20)
pytest -q                                   # SQLite backend: no Docker, no network, no API key
DRI_TEST_DATABASE_URL=postgresql://... pytest -q    # same suite ALSO on Postgres (each test gets its own schema)
```
The header line of the pytest output says which backends ran. The LLM is scripted in tests; the Anthropic SDK client is faked.

## Experiment (baseline vs agent)
```bash
ANTHROPIC_API_KEY=... python -m eval.run_eval --case eval/cases/sample_plant --runs 3 --out eval/results
# or with Gemini:
DRI_LLM_PROVIDER=gemini GEMINI_API_KEY=... DRI_MODEL=<gemini model> python -m eval.run_eval --case eval/cases/sample_plant --runs 3
```
Both approaches use the same provider and model, so the comparison is fair, but the numbers describe that model only.
Add your own case as `eval/cases/<name>/docs/*` + `ground_truth.json`. The sample is a development fixture - do not tune prompts to it.

See [ARCHITECTURE.md](ARCHITECTURE.md) for design decisions and the AI-vs-deterministic split.
