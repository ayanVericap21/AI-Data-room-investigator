"""Typed fact extraction: facts are tied to chunks by their quote; only observations and assertions become claims; zero-fact documents are asked again."""
import threading
import time
from pathlib import Path

from sqlalchemy import select

from app.investigator.facts import SCHEMA, SYSTEM_PROMPT, FactExtractor, locate
from app.llm.llama_extract import LlamaJob
from app.models import Claim, Document
from tests.inv_support import RoutedLLM, build_store, dive_room
from tests.test_llama_claims import FakeClient
from tests.test_tools import room  # noqa: F401  (fixture)


def fact(**kw):
    base = dict(type="observation", subject="lab", attribute="mean sample count per dive", operator="=", value="5.7", unit="samples",
                quote="Mean sample count per dive was 5.7.", value_type="quantity")
    base.update(kw)
    return base


def extractor(session, tmp_path, client, workers=1, name="t.sqlite"):
    store = build_store(session, tmp_path, name=name)
    ex = FactExtractor(session, client, Path("/data_room"), store, RoutedLLM({}), workers=workers)
    return ex, store


def docs(session):
    return {d.filename: d for d in session.scalars(select(Document))}


def test_schema_and_prompt_describe_all_seven_types():
    for t in ("observation", "constraint", "event", "assertion", "reference", "definition", "conclusion"):
        assert t in SYSTEM_PROMPT and t in SCHEMA["properties"]["facts"]["items"]["properties"]["type"]["enum"]
    assert len(SYSTEM_PROMPT) < 9000


def test_a_fact_is_tied_to_the_chunk_that_holds_its_quote(session, tmp_path):
    dive_room(session)
    ex, store = extractor(session, tmp_path, FakeClient({"facts": [fact()]}))
    d = docs(session)["lab.pdf"]
    stats = ex.run([d])
    assert stats.facts == 1 and ex.facts[0].chunk_id and ex.facts[0].quote == "Mean sample count per dive was 5.7."
    assert store.query("SELECT type, value FROM _facts").rows == [("observation", "5.7")]


def test_observations_and_assertions_become_claims_and_other_types_do_not(session, tmp_path):
    dive_room(session)
    rule = fact(type="constraint", subject="site", attribute="fragments collected", operator="<=", value="8", unit="fragments", modality="must",
                quote="No more than 8 fragments may be collected per site.")
    ex, _ = extractor(session, tmp_path, FakeClient({"facts": [rule]}))
    ex.run([docs(session)["permit.docx"]])
    assert session.scalars(select(Claim)).all() == [] and ex.facts[0].type == "constraint" and ex.facts[0].modality == "must"
    ex2, _ = extractor(session, tmp_path, FakeClient({"facts": [fact()]}), name="second.sqlite")
    stats = ex2.run([docs(session)["lab.pdf"]])
    claim = session.scalars(select(Claim)).one()
    assert stats.claims == 1 and claim.metric == "mean sample count per dive" and claim.claim_category == "quantity"


def test_a_fact_whose_quote_is_not_in_the_document_is_rejected_and_counted(session, tmp_path):
    dive_room(session)
    ex, _ = extractor(session, tmp_path, FakeClient({"facts": [fact(quote="This sentence is not in the document at all.", value="999")]}))
    stats = ex.run([docs(session)["lab.pdf"]])
    assert stats.facts == 0 and stats.rejected == 1 and "quote not found" in stats.rejections[0]


def test_a_fact_with_an_unknown_type_is_ignored(session, tmp_path):
    dive_room(session)
    ex, _ = extractor(session, tmp_path, FakeClient({"facts": [fact(type="opinion")]}))
    assert ex.run([docs(session)["lab.pdf"]]).facts == 0


def test_locate_falls_back_to_the_value_and_subject_when_the_quote_is_missing(session):
    permit, register, lab = dive_room(session)
    from app.models import DocumentChunk
    chunks = list(session.scalars(select(DocumentChunk).where(DocumentChunk.document_id == lab.id).order_by(DocumentChunk.chunk_index)))
    chunk, quote = locate("", "5.7", "lab", chunks)
    assert chunk.page_number == 2 and "5.7" in quote


class EmptyThenFull:
    def __init__(self):
        self.calls = 0

    def extract(self, path, schema, **kw):
        self.calls += 1
        result = {"facts": []} if self.calls == 1 else {"facts": [fact()]}
        return LlamaJob(job_id="x", status="COMPLETED", result=result, metadata={}, credits=2.0)


def test_a_document_with_numbers_and_no_tables_that_returns_no_facts_is_asked_once_more(session, tmp_path):
    permit, register, lab = dive_room(session)
    store = build_store(session, tmp_path)
    client = EmptyThenFull()
    ex = FactExtractor(session, client, Path("/data_room"), store, RoutedLLM({}))
    from app.investigator.tables import TableStore
    coverage = TableStore(tmp_path / "other.sqlite").build(session)
    stats = ex.run([lab], coverage)
    assert client.calls == 2 and stats.facts == 1 and stats.credits == 4.0 and coverage[lab.id].ok


class Slow:
    def __init__(self):
        self.live, self.peak, self.guard = 0, 0, threading.Lock()

    def extract(self, path, schema, **kw):
        with self.guard:
            self.live += 1
            self.peak = max(self.peak, self.live)
        time.sleep(0.05)
        with self.guard:
            self.live -= 1
        return LlamaJob(job_id="x", status="COMPLETED", result={"facts": []}, metadata={}, credits=1.0)


def test_documents_are_read_in_parallel_with_workers(session, tmp_path):
    dive_room(session)
    client = Slow()
    ex, _ = extractor(session, tmp_path, client, workers=3)
    ex.run(list(docs(session).values()))
    assert client.peak >= 2


def test_a_failed_extraction_is_reported_and_does_not_stop_the_others(session, tmp_path):
    dive_room(session)
    ex, _ = extractor(session, tmp_path, FakeClient({"facts": []}, status="FAILED"))
    stats = ex.run(list(docs(session).values()))
    assert len(stats.failed) == 3 and stats.documents == 0
