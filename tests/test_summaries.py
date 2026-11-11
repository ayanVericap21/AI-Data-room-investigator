import pytest

from app import jobs
from app.documents import save_upload
from app.ingest.summarize import summarize_document
from app.llm import LLMError
from app.models import DocSummary, JobStatus
from app.tools import DataRoomTools
from tests.test_agent import FakeLLM

SUMMARY = DocSummary(overview="Supplier quotation for a synthesis loop.", key_topics=["guarantee"], key_figures=["25,000 tonnes/yr guaranteed"])


def _doc(db, settings, name, text):
    return save_upload(db, settings, name, text.encode())[0]


def _ingested(db, settings, llm=None, text="The supplier guarantees only 25,000 tonnes per year."):
    d = _doc(db, settings, "s.txt", text)
    j, _ = jobs.submit_job(db, [d.id], "k")
    jobs.run_job(db, settings, j.id, (lambda: llm) if llm else None)
    return d.id


def test_summary_is_stored_before_documents_are_served(db, settings):
    llm = FakeLLM({"summarize": [SUMMARY]})
    did = _ingested(db, settings, llm)
    row = db.one("SELECT summary_status, summary FROM documents WHERE id=?", (did,))
    assert row["summary_status"] == "done" and DocSummary.model_validate_json(row["summary"]) == SUMMARY
    listed = DataRoomTools(db).list_documents()["documents"][0]          # what the MCP server returns
    assert listed["summary"]["overview"].startswith("Supplier quotation") and listed["summary_status"] == "done"
    one = DataRoomTools(db).get_document_summary(did)
    assert one["summary"]["key_figures"] == ["25,000 tonnes/yr guaranteed"] and "not evidence" in one["note"]
    assert "error" in DataRoomTools(db).get_document_summary(999)


def test_no_llm_means_skipped_not_failed(db, settings):
    did = _ingested(db, settings, llm=None)
    row = db.one("SELECT status, summary_status, summary_error FROM documents WHERE id=?", (did,))
    assert row["status"] == "processed" and row["summary_status"] == "skipped" and "no LLM" in row["summary_error"]
    assert DataRoomTools(db).list_documents()["documents"][0]["summary"] is None  # still searchable
    assert DataRoomTools(db).search_chunks("supplier")["results"]


def test_llm_factory_failure_skips_summaries_but_job_completes(db, settings):
    d = _doc(db, settings, "s.txt", "The supplier guarantees only 25,000 tonnes per year.")
    j, _ = jobs.submit_job(db, [d.id], "k")

    def broken():
        raise LLMError("ANTHROPIC_API_KEY is not set")

    done = jobs.run_job(db, settings, j.id, broken)
    assert done.status == JobStatus.COMPLETED and done.results[0].summary_status == "skipped"
    assert "ANTHROPIC_API_KEY" in db.one("SELECT summary_error FROM documents WHERE id=?", (d.id,))["summary_error"]


def test_summary_failure_is_isolated(db, settings):
    llm = FakeLLM({"summarize": [LLMError("model overloaded")]})
    did = _ingested(db, settings, llm)
    row = db.one("SELECT status, summary_status, summary_error FROM documents WHERE id=?", (did,))
    assert row["status"] == "processed" and row["summary_status"] == "failed" and "overloaded" in row["summary_error"]
    assert jobs.get_job(db, 1).status == JobStatus.COMPLETED


def test_unexpected_summary_crash_is_isolated(db, settings):
    class Boom(FakeLLM):
        def structured(self, **kw):
            raise RuntimeError("kaboom")

    did = _ingested(db, settings, Boom({}))
    assert db.one("SELECT summary_status FROM documents WHERE id=?", (did,))["summary_status"] == "failed"


def test_long_document_is_map_reduced(db, settings):
    settings.summary_batch_chars = 400
    text = "\n\n".join(f"Paragraph {i} describes capacity item number {i} in some detail here." * 2 for i in range(12))
    d = _doc(db, settings, "long.txt", text)
    jobs.submit_job(db, [d.id], "k")
    from app.ingest.pipeline import ingest_document
    ingest_document(db, settings, d.id)
    llm = FakeLLM({"summarize": [DocSummary(overview="part")], "summarize_merge": [SUMMARY]})
    assert summarize_document(db, settings, llm, d.id) == "done"
    purposes = [p for p, _ in llm.prompts]
    assert purposes.count("summarize") > 1 and purposes[-1] == "summarize_merge"
    assert "Part 1:" in llm.prompts[-1][1]


def test_reingest_resets_summary(db, settings):
    did = _ingested(db, settings, FakeLLM({"summarize": [SUMMARY]}))
    from app.ingest.pipeline import ingest_document
    ingest_document(db, settings, did)
    row = db.one("SELECT summary, summary_status FROM documents WHERE id=?", (did,))
    assert row["summary"] is None and row["summary_status"] == "pending"


def test_job_usage_is_recorded(db, settings):
    llm = FakeLLM({"summarize": [SUMMARY]})
    d = _doc(db, settings, "s.txt", "The supplier guarantees only 25,000 tonnes per year.")
    j, _ = jobs.submit_job(db, [d.id], "k")
    done = jobs.run_job(db, settings, j.id, lambda: llm)
    assert done.usage["calls"] == 1 and done.usage["cost_usd"] >= 0


def test_empty_document_summary_skipped(db, settings):
    d = _doc(db, settings, "e.txt", "placeholder text")
    db.execute("DELETE FROM chunks WHERE document_id=?", (d.id,))
    assert summarize_document(db, settings, FakeLLM({}), d.id) == "skipped"
