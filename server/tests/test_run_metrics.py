"""Run metrics: the script builds them from a finished run, the API serves them by investigation."""
from __future__ import annotations

import importlib.util
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import app.services.run_metrics as store
from app.core.database import get_session
from app.llm.embeddings import MockEmbeddingProvider
from app.llm.metrics import CallMetric
from app.llm.mock import MockLLMClient
from app.main import create_app
from app.services.claim_service import ExtractionSummary
from app.services.pair_judge import JudgeStats
from tests.test_tools import room  # noqa: F401  (fixture)

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ledger_run.py"


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("ledger_run_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SCRIPT.parent))
    spec.loader.exec_module(module)
    return module


def finished_run(**overrides):
    llm = MockLLMClient()
    llm.call_metrics.extend([
        CallMetric(kind="llm", latency_ms=1000, prompt_tokens=100, completion_tokens=50, total_tokens=150, started_at=1.0),
        CallMetric(kind="llm", latency_ms=3000, prompt_tokens=300, completion_tokens=70, total_tokens=370, started_at=2.0),
        CallMetric(kind="llm", latency_ms=500, ok=False, error="429", started_at=3.0),
    ])
    llm.retry_count = 1
    embedder = MockEmbeddingProvider(dimensions=4)
    embedder.call_metrics.append(CallMetric(kind="embedding", latency_ms=20, items=5))
    judge = JudgeStats(pairs_exact=2, pairs_judged=3, findings_saved=1, findings_rejected=2)
    judge.verdict_counts.update({"contradiction": 1, "not_comparable": 2})
    run = SimpleNamespace(
        llm=llm, embedder=embedder, doc_stats={"a": dict(chunks=4, readable=4, ok=True), "b": dict(chunks=2, readable=2, ok=False)},
        extraction=ExtractionSummary(claims_created=7, claims_rejected=1), judge=judge, pairs=[object()] * 5, wall_s=12.34,
        started=datetime(2026, 11, 11, 10, 0, 0), finished=datetime(2026, 11, 11, 10, 0, 12), config="ui-run", error=None, stopped=False,
        result=SimpleNamespace(investigation_id="inv_1"), llm_bounds=[2, 2, 3], stage={"1. read": 8000, "3. judge": 4000},
        args=SimpleNamespace(extractor="llm", pairing="llm", vocabulary="off", coverage_check="off", batch_chunks=0, batch_chars=8000),
        settings=SimpleNamespace(llm_model="m", embedding_model="e"))
    for key, value in overrides.items():
        setattr(run, key, value)
    return run


def test_the_script_builds_the_same_numbers_the_results_file_shows(script):
    m = script.build_metrics(finished_run(), Path("/x/TEST-RUN-1.txt"))
    assert (m.investigation_id, m.state, m.wall_s) == ("inv_1", "completed", 12.3)
    assert (m.counts.documents, m.counts.documents_fully_read, m.counts.chunks, m.counts.claims_recorded, m.counts.claims_rejected) == (2, 1, 6, 7, 1)
    assert (m.counts.candidate_pairs, m.counts.pairs_exact, m.counts.pairs_judged, m.counts.findings_saved) == (5, 2, 3, 1)
    assert m.counts.verdicts == {"contradiction": 1, "not_comparable": 2} and m.counts.embedding_texts == 5
    assert (m.llm.attempts, m.llm.failed, m.llm.retries, m.llm.total_s) == (3, 1, 1, 4.5)
    assert (m.llm.min_context_tokens, m.llm.avg_context_tokens, m.llm.max_context_tokens) == (100, 200, 300)
    assert (m.llm.prompt_tokens, m.llm.completion_tokens) == (400, 120) or m.llm.total_tokens == 0  # tokens come from the client's own counter
    assert [(p.phase, p.attempts, p.failed) for p in m.phases] == [("extract claims", 2, 0), ("judge pairs", 1, 1)]
    assert [(s.stage, s.seconds, s.percent) for s in m.stages] == [("1. read", 8.0, 64.8), ("3. judge", 4.0, 32.4)]
    assert m.settings.pairing == "llm" and m.settings.batch_chunks is None


def test_a_stopped_or_failed_run_says_so(script):
    assert script.build_metrics(finished_run(stopped=True), Path("/x/a.txt")).state == "stopped"
    assert script.build_metrics(finished_run(error="boom"), Path("/x/a.txt")).state == "failed"


def test_metrics_are_written_next_to_the_results_file_and_found_by_investigation(script, tmp_path, monkeypatch):
    results = tmp_path / "TEST-RUN-1-ui-run.txt"
    results.write_text("report", encoding="utf-8")
    path = store.write_metrics(results, script.build_metrics(finished_run(), results))
    assert path.name == "TEST-RUN-1-ui-run.metrics.json"
    assert store.find_metrics("inv_1", tmp_path).config_name == "ui-run"
    assert store.find_metrics("inv_other", tmp_path) is None
    (tmp_path / "broken.metrics.json").write_text("{not json", encoding="utf-8")
    assert store.find_metrics("inv_1", tmp_path) is not None  # an unreadable file is skipped, not fatal


def test_the_endpoint_serves_metrics_and_404s_for_runs_without_them(script, session, room, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr(store, "RESULTS_DIR", tmp_path)
    app = create_app()
    app.dependency_overrides[get_session] = lambda: session
    client = TestClient(app)
    assert client.get("/api/investigations/inv_nope/metrics").status_code == 404  # unknown run
    assert client.get(f"/api/investigations/{room.investigation.id}/metrics").status_code == 404  # known run, no file
    results = tmp_path / "TEST-RUN-2-ui-run.txt"
    metrics = script.build_metrics(finished_run(result=SimpleNamespace(investigation_id=room.investigation.id)), results)
    store.write_metrics(results, metrics)
    body = client.get(f"/api/investigations/{room.investigation.id}/metrics").json()
    assert body["config_name"] == "ui-run" and body["llm"]["attempts"] == 3 and body["counts"]["claims_recorded"] == 7
