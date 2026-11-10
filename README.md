# Data Room Investigator

AI-assisted investigation of a predefined data room: extract claims, find cross-document
discrepancies, gather evidence, and produce traceable findings. PostgreSQL is the source of truth;
ChromaDB (Stage 2+) only holds vectors; LangGraph (Stage 3+) orchestrates the investigation.

> **Status: Stage 1 (foundation) complete.** Document scanning, extraction, schema/migrations and the
> data room dashboard work. Claims, retrieval, the LangGraph workflow, evaluation and the remaining
> pages are later stages. Full docs (`docs/`) are written in Stage 5.

## Repository layout

```
client/   React + TS + Vite + MUI + React Query
server/   FastAPI backend (app/), Alembic (alembic/), tests (tests/), data_room/ (input files)
docker-compose.yml   PostgreSQL + ChromaDB
```

## Prerequisites
Python 3.11+, Node 20+, Docker Desktop.

## Setup
```bash
cp .env.example server/.env          # edit values locally; never commit .env
docker compose up -d                 # postgres :5432, chroma :8001
docker compose logs -f postgres      # inspect logs
docker compose down -v               # reset dev volumes (deletes data)

cd server
python -m venv .venv && .venv/Scripts/activate   # Windows; use bin/activate elsewhere
pip install -r requirements.txt
alembic upgrade head                 # create tables
uvicorn app.main:app --reload --port 8000

cd ../client
npm install && npm run dev           # http://localhost:5173 (proxies /api to :8000)
```

Put `.pdf`, `.docx`, `.xlsx` files in `server/data_room/` (subfolders are scanned), then click
**Scan / refresh documents**. Document identity is the SHA-256 of the content; only paths relative to
the data room are exposed.

## Tests
```bash
cd server && python -m pytest
```
Unit tests run on SQLite (JSONB falls back to JSON); PostgreSQL integration tests arrive with Stage 2+.

## Extraction behaviour worth knowing
- **PDF:** page-wise paragraphs + best-effort tables, with page numbers.
- **DOCX:** body order preserved; headings tracked; paragraph/table indexes in the locator.
- **XLSX:** one unit per row with sheet + range locator; formulas kept in metadata. openpyxl does not
  calculate formulas, so a formula cell with no cached value is flagged `cached value not available`.
