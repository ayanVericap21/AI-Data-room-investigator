import pytest

from app import jobs
from app.documents import DocumentNotFound, save_upload
from app.models import JobStatus
from app.tools import DataRoomTools, normalize, quote_in_text


def _doc(db, settings, name, text):
    d, _ = save_upload(db, settings, name, text.encode())
    return d


# ---------- jobs ----------
def test_job_happy_path(db, settings):
    a = _doc(db, settings, "a.txt", "Capacity is 30,000 tonnes per year.")
    b = _doc(db, settings, "b.txt", "Supplier guarantees 25,000 tonnes.")
    job, created = jobs.submit_job(db, [a.id, b.id], "k1")
    assert created and job.status == JobStatus.QUEUED
    done = jobs.run_job(db, settings, job.id)
    assert done.status == JobStatus.COMPLETED and len(done.results) == 2
    assert done.started_at and done.finished_at


def test_idempotent_replay_returns_same_job(db, settings):
    a = _doc(db, settings, "a.txt", "some text content here")
    j1, c1 = jobs.submit_job(db, [a.id], "k")
    j2, c2 = jobs.submit_job(db, [a.id], "k")
    assert c1 and not c2 and j1.id == j2.id


def test_same_key_different_docs_conflicts(db, settings):
    a = _doc(db, settings, "a.txt", "text a is here")
    b = _doc(db, settings, "b.txt", "text b is here")
    jobs.submit_job(db, [a.id], "k")
    with pytest.raises(jobs.JobConflict):
        jobs.submit_job(db, [b.id], "k")


def test_documents_in_live_job_conflict(db, settings):
    a = _doc(db, settings, "a.txt", "text a is here")
    jobs.submit_job(db, [a.id], "k1")
    with pytest.raises(jobs.JobConflict):
        jobs.submit_job(db, [a.id], "k2")


def test_unknown_document_and_empty_inputs(db):
    with pytest.raises(DocumentNotFound):
        jobs.submit_job(db, [42], "k")
    with pytest.raises(ValueError):
        jobs.submit_job(db, [], "k")
    with pytest.raises(ValueError):
        jobs.submit_job(db, [1], "  ")


def test_already_processed_documents_are_skipped(db, settings):
    a = _doc(db, settings, "a.txt", "text a is here")
    j, _ = jobs.submit_job(db, [a.id], "k1")
    jobs.run_job(db, settings, j.id)
    with pytest.raises(jobs.NothingToProcess):
        jobs.submit_job(db, [a.id], "k2")
    b = _doc(db, settings, "b.txt", "text b is here")
    j2, _ = jobs.submit_job(db, [a.id, b.id], "k3")
    assert j2.document_ids == [b.id]


def test_one_bad_document_fails_job_but_others_complete(db, settings):
    good = _doc(db, settings, "g.txt", "good document with content")
    bad, _ = save_upload(db, settings, "bad.pdf", b"garbage")
    j, _ = jobs.submit_job(db, [good.id, bad.id], "k")
    done = jobs.run_job(db, settings, j.id)
    assert done.status == JobStatus.FAILED and "1 of 2" in done.error
    statuses = {r.document_id: r.status.value for r in done.results}
    assert statuses[good.id] == "processed" and statuses[bad.id] == "failed"


def test_run_job_not_rerun_and_recovery(db, settings):
    a = _doc(db, settings, "a.txt", "text a is here")
    j, _ = jobs.submit_job(db, [a.id], "k")
    jobs.run_job(db, settings, j.id)
    assert jobs.run_job(db, settings, j.id).status == JobStatus.COMPLETED  # no-op
    db.execute("UPDATE ingest_jobs SET status='running' WHERE id=?", (j.id,))
    assert jobs.recover_stuck_jobs(db) == 1
    assert jobs.get_job(db, j.id).status == JobStatus.FAILED
    with pytest.raises(jobs.JobNotFound):
        jobs.get_job(db, 999)


# ---------- tools / evidence ----------
@pytest.fixture
def room(db, settings):
    a = _doc(db, settings, "supplier.txt", "The supplier guarantees only 25,000 tonnes per year.\n\nDelivery within 90 days.")
    b = _doc(db, settings, "model.txt", "Revenue assumes 30,000 tonnes of annual production.")
    j, _ = jobs.submit_job(db, [a.id, b.id], "k")
    jobs.run_job(db, settings, j.id)
    return DataRoomTools(db), a, b


def test_normalize_and_quote_matching():
    assert normalize("  Hello\n\tWORLD’s ") == "hello world's"
    assert quote_in_text("guarantees  ONLY 25,000\ntonnes", "The supplier guarantees only 25,000 tonnes per year")
    assert not quote_in_text("guarantees 30,000 tonnes", "The supplier guarantees only 25,000 tonnes")
    assert not quote_in_text("a", "a b c")  # too short to be meaningful


def test_search_and_scope(room, db):
    tools, a, b = room
    res = tools.search_chunks("tonnes")["results"]
    assert {r["filename"] for r in res} == {"supplier.txt", "model.txt"}
    only_b = DataRoomTools(db, [b.id]).search_chunks("tonnes")["results"]
    assert {r["filename"] for r in only_b} == {"model.txt"}
    assert tools.search_chunks("   ")["results"] == []
    assert tools.search_chunks('"; DROP TABLE chunks; --')["results"] == []  # FTS input is sanitised


def test_get_chunk_and_neighbors(room):
    tools, a, _ = room
    cid = tools.search_chunks("delivery", document_id=a.id)["results"][0]["chunk_id"]
    assert "chunk" in tools.get_chunk(cid)
    assert "error" in tools.get_chunk(9999)
    assert tools.get_neighbors(cid, 1)["chunks"]
    assert "error" in tools.get_neighbors(9999)


def test_verify_quote(room):
    tools, a, b = room
    cid = tools.search_chunks("guarantees", document_id=a.id)["results"][0]["chunk_id"]
    assert tools.verify_quote(cid, "guarantees only 25,000 tonnes")["found"]
    assert not tools.verify_quote(cid, "guarantees 40,000 tonnes per year")["found"]  # fabricated
    assert not tools.verify_quote(9999, "anything long enough")["found"]
    # citing a real quote under the wrong document's chunk must not verify
    other = tools.search_chunks("revenue", document_id=b.id)["results"][0]["chunk_id"]
    assert not tools.verify_quote(other, "guarantees only 25,000 tonnes")["found"]
