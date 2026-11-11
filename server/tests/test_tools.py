"""Tool layer: reading tools, calculate, submit_finding validation, registry behaviour."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from sqlalchemy import func, select

from app.core.exceptions import AppError
from app.models import Document, DocumentChunk, Evidence, Finding, Investigation, InvestigationEvent
from app.services.retrieval_service import RetrievalService, RetrievedChunk
from app.tools import ToolBudget, ToolContext, ToolRegistry, build_registry
from app.tools.base import ToolError, ToolSpec
from app.tools.schemas import SearchInput
from app.tools.submit import find_source_span


def add_doc(session, filename, file_type, chunks, status="processed"):
    doc = Document(filename=filename, file_hash=f"hash-{filename}", file_type=file_type, file_size=1,
                   processing_status=status, last_modified_at=__import__("datetime").datetime.now())
    session.add(doc)
    session.flush()
    rows = []
    for i, kw in enumerate(chunks):
        row = DocumentChunk(document_id=doc.id, chunk_index=i, content_type=kw.pop("content_type", "paragraph"), **kw)
        session.add(row)
        rows.append(row)
    session.flush()
    return doc, rows


@pytest.fixture
def room(session):
    deck, deck_chunks = add_doc(session, "deck.pdf", "pdf", [
        dict(content="Annual production capacity is 120,000 tonnes per year at the Pune plant.", page_number=1),
        dict(content="EBITDA margin for 2024 was 18% of revenue.", page_number=2),
    ])
    capex, capex_chunks = add_doc(session, "capex.xlsx", "xlsx", [
        dict(content="Capacity | 100,000 tonnes", sheet_name="Current", cell_reference="B3", content_type="table_row"),
        dict(content="Capacity after phase 2 | 150,000 tonnes (2027)", sheet_name="Expansion", cell_reference="B4",
             content_type="table_row"),
    ])
    lease, lease_chunks = add_doc(session, "lease.docx", "docx", [
        dict(content="Annual rent is INR 1.2 crore.\nPayable quarterly in advance.", heading="Rent"),
        dict(content="Rent escalates 5% each year.", heading="Rent"),
        dict(content="The lease term is 9 years.", heading="Term"),
    ])
    gone, _ = add_doc(session, "old.pdf", "pdf", [dict(content="stale", page_number=1)], status="missing")
    investigation = Investigation(name="test run")
    session.add(investigation)
    session.commit()
    return SimpleNamespace(
        deck=deck, capex=capex, lease=lease, gone=gone, investigation=investigation,
        deck_p1=deck_chunks[0].id, deck_p2=deck_chunks[1].id, capex_current=capex_chunks[0].id,
        capex_expansion=capex_chunks[1].id, rent_1=lease_chunks[0].id, rent_2=lease_chunks[1].id, term=lease_chunks[2].id,
    )


@pytest.fixture
def registry(session, room):
    return build_registry(session, retrieval=None, investigation_id=room.investigation.id)


def finding_args(room, **overrides):
    args = dict(
        title="Plant capacity differs", description="The deck says 120,000 t; the capex plan says 100,000 t current.",
        classification="contradiction", severity="high", confidence=0.8, evidence_status="adequate",
        evidence=[
            dict(chunk_id=room.deck_p1, quoted_text="capacity is 120,000 tonnes per year", role="contradicting",
                 relevance_reason="States current capacity", support_assessment="Says 120,000 t"),
            dict(chunk_id=room.capex_current, quoted_text="Capacity | 100,000 tonnes", role="contradicting",
                 relevance_reason="States current capacity", support_assessment="Says 100,000 t"),
        ],
    )
    args.update(overrides)
    return args


# ---- reading tools ------------------------------------------------------------
def test_list_documents_excludes_missing_and_counts_chunks(registry):
    result = registry.execute("list_documents")
    assert result.ok
    docs = {d.filename: d for d in result.output.documents}
    assert set(docs) == {"deck.pdf", "capex.xlsx", "lease.docx"}
    assert docs["lease.docx"].chunk_count == 3 and docs["deck.pdf"].status == "processed"


def test_outline_follows_each_format(registry, room):
    pdf = registry.execute("get_outline", {"document_id": room.deck.id}).output
    assert [s.title for s in pdf.sections] == ["Page 1", "Page 2"]
    xlsx = registry.execute("get_outline", {"document_id": room.capex.id}).output
    assert [s.title for s in xlsx.sections] == ["Sheet: Current", "Sheet: Expansion"]
    docx = registry.execute("get_outline", {"document_id": room.lease.id}).output
    assert [(s.title, s.chunk_count) for s in docx.sections] == [("Rent", 2), ("Term", 1)]


def test_outline_unknown_document_is_a_structured_error(registry):
    result = registry.execute("get_outline", {"document_id": "doc_nope"})
    assert not result.ok and result.error.code == "not_found"


def test_get_chunk_with_neighbors(registry, room):
    out = registry.execute("get_chunk", {"chunk_id": room.rent_2, "neighbors": 1}).output
    assert out.chunk.chunk_id == room.rent_2 and out.chunk.locator == "heading Rent"
    assert [c.chunk_id for c in out.before] == [room.rent_1] and [c.chunk_id for c in out.after] == [room.term]
    assert registry.execute("get_chunk", {"chunk_id": "chk_nope"}).error.code == "not_found"


def test_read_section_by_page_sheet_and_heading(registry, room):
    page = registry.execute("read_section", {"document_id": room.deck.id, "page_number": 2}).output
    assert [c.chunk_id for c in page.chunks] == [room.deck_p2]
    sheet = registry.execute("read_section", {"document_id": room.capex.id, "sheet_name": "Current"}).output
    assert sheet.chunks[0].locator == "sheet Current, cell B3"
    heading = registry.execute("read_section", {"document_id": room.lease.id, "heading": "rent"}).output  # case-insensitive
    assert len(heading.chunks) == 2 and not heading.truncated


def test_read_section_requires_exactly_one_selector_and_reports_misses(registry, room):
    both = registry.execute("read_section", {"document_id": room.deck.id, "page_number": 1, "heading": "x"})
    assert both.error.code == "invalid_arguments" and "exactly one" in both.error.message
    none = registry.execute("read_section", {"document_id": room.deck.id})
    assert none.error.code == "invalid_arguments"
    miss = registry.execute("read_section", {"document_id": room.deck.id, "page_number": 99})
    assert miss.error.code == "not_found" and "get_outline" in miss.error.message


def test_read_section_truncates_by_max_chunks(registry, room):
    out = registry.execute("read_section", {"document_id": room.lease.id, "heading": "Rent", "max_chunks": 1}).output
    assert out.truncated and out.total_chunks == 2 and len(out.chunks) == 1


# ---- calculate ----------------------------------------------------------------
@pytest.mark.parametrize("expression,expected", [
    ("120000 - 100000", "20000"),
    ("(120000 - 100000) / 100000 * 100", "20"),
    ("0.1 + 0.2", "0.3"),  # exact decimal arithmetic, not float
    ("2 ** 10", "1024"),
    ("-5 + 10 % 3", "-4"),
    ("1 / 3", "0.3333333333333333333333333333"),
])
def test_calculate_is_exact(registry, expression, expected):
    result = registry.execute("calculate", {"expression": expression})
    assert result.ok and result.output.result == expected


@pytest.mark.parametrize("expression", ["1 / 0", "__import__('os')", "abs(-1)", "x + 1", "2 ** 0.5", "2 ** 100", "1 +", "1e999999"])
def test_calculate_rejects_anything_but_arithmetic(registry, expression):
    result = registry.execute("calculate", {"expression": expression})
    assert not result.ok and result.error.code == "invalid_expression"


# ---- record_fact --------------------------------------------------------------
def fact_args(room, **overrides):
    fact = dict(label="Plant capacity", value="120,000", unit="tonnes per year", qualifier="Pune plant",
                chunk_id=room.deck_p1, quoted_text="capacity is 120,000 tonnes per year")
    fact.update(overrides)
    return {"facts": [fact]}


def test_record_fact_keeps_verified_facts_in_memory_with_their_source(registry, room):
    result = registry.execute("record_fact", {"facts": fact_args(room)["facts"] + fact_args(
        room, label="Current capacity", value="100,000", chunk_id=room.capex_current,
        quoted_text="Capacity | 100,000 tonnes")["facts"]})
    assert result.ok, result.error
    assert result.output.recorded == ["fact_1", "fact_2"] and result.output.total_facts == 2
    first, second = registry.context.facts
    assert (first.filename, first.locator, first.value, first.qualifier) == ("deck.pdf", "page 1", "120,000", "Pune plant")
    assert second.filename == "capex.xlsx" and second.quoted_text == "Capacity | 100,000 tonnes"
    registry.execute("record_fact", fact_args(room, chunk_id=room.deck_p2, quoted_text="EBITDA margin for 2024 was 18%"))
    assert registry.context.facts[-1].fact_id == "fact_3"


def test_record_fact_rejects_the_whole_batch_when_one_quote_is_wrong(registry, room):
    good = fact_args(room)["facts"][0]
    bad = fact_args(room, quoted_text="capacity is 130,000 tonnes")["facts"][0]
    result = registry.execute("record_fact", {"facts": [good, bad]})
    assert not result.ok and result.error.code == "validation_failed" and "fact 2" in result.error.message
    assert registry.context.facts == []  # nothing from the batch was kept


def test_record_fact_rejects_unknown_chunks_and_empty_batches(registry, room):
    result = registry.execute("record_fact", fact_args(room, chunk_id="chunk_nope"))
    assert not result.ok and "does not exist" in result.error.message
    assert registry.execute("record_fact", {"facts": []}).error.code == "invalid_arguments"


def test_record_fact_stores_the_exact_source_text(registry, room):
    registry.execute("record_fact", fact_args(room, chunk_id=room.rent_1,
                                              quoted_text="Annual rent is INR 1.2 crore. Payable quarterly"))
    assert registry.context.facts[0].quoted_text == "Annual rent is INR 1.2 crore.\nPayable quarterly"


# ---- submit_finding -----------------------------------------------------------
def test_submit_finding_saves_verified_finding_and_evidence(registry, room, session):
    result = registry.execute("submit_finding", finding_args(room))
    assert result.ok, result.error
    finding = session.get(Finding, result.output.finding_id)
    assert finding.verification_status == "verified" and finding.investigation_id == room.investigation.id
    quotes = {e.quoted_text for e in session.scalars(select(Evidence).where(Evidence.finding_id == finding.id))}
    assert quotes == {"capacity is 120,000 tonnes per year", "Capacity | 100,000 tonnes"}
    assert registry.context.finding_ids == [finding.id]


def test_whitespace_differences_store_the_exact_source_text(registry, room, session):
    args = finding_args(room, classification="missing_information", evidence_status="partial", evidence=[dict(
        chunk_id=room.rent_1, quoted_text="Annual rent is INR 1.2 crore. Payable quarterly in advance.",
        role="supporting", relevance_reason="Rent", support_assessment="States rent")])
    result = registry.execute("submit_finding", args)
    assert result.ok
    stored = session.scalar(select(Evidence.quoted_text).where(Evidence.finding_id == result.output.finding_id))
    assert stored == "Annual rent is INR 1.2 crore.\nPayable quarterly in advance."  # verbatim, newline kept
    assert any("whitespace" in n for n in result.output.notes)


def test_unverifiable_quote_rejects_the_whole_finding(registry, room, session):
    args = finding_args(room)
    args["evidence"][1]["quoted_text"] = "Capacity | 110,000 tonnes"  # one digit changed
    result = registry.execute("submit_finding", args)
    assert not result.ok and result.error.code == "validation_failed"
    assert "evidence 2" in result.error.message and "not in chunk" in result.error.message
    assert session.scalar(select(func.count()).select_from(Finding)) == 0
    assert session.scalar(select(func.count()).select_from(Evidence)) == 0
    assert registry.context.finding_ids == []


def test_unknown_chunk_is_rejected(registry, room, session):
    args = finding_args(room)
    args["evidence"][0]["chunk_id"] = "chk_invented"
    result = registry.execute("submit_finding", args)
    assert result.error.code == "validation_failed" and "does not exist" in result.error.message
    assert session.scalar(select(func.count()).select_from(Finding)) == 0


@pytest.mark.parametrize("classification", ["contradiction", "contextual_difference", "consistent"])
def test_comparing_classifications_need_two_chunks(registry, room, classification):
    args = finding_args(room, classification=classification)
    args["evidence"] = args["evidence"][:1]
    result = registry.execute("submit_finding", args)
    assert result.error.code == "validation_failed" and "two different chunks" in result.error.message


def test_missing_information_may_cite_a_single_chunk(registry, room):
    args = finding_args(room, classification="missing_information", evidence_status="partial",
                        unresolved_questions=["Is there a provision in the financials?"])
    args["evidence"] = args["evidence"][:1]
    assert registry.execute("submit_finding", args).ok


def test_invalid_enum_and_confidence_come_back_as_errors(registry, room):
    bad = registry.execute("submit_finding", finding_args(room, classification="maybe_wrong", confidence=1.5))
    assert bad.error.code == "invalid_arguments"
    assert "classification" in bad.error.message and "confidence" in bad.error.message


def test_short_quotes_are_refused(registry, room):
    args = finding_args(room)
    args["evidence"][0]["quoted_text"] = "120"
    assert registry.execute("submit_finding", args).error.code == "invalid_arguments"


def test_finding_budget_and_missing_investigation(session, room):
    registry = build_registry(session, None, room.investigation.id, ToolBudget(max_findings=1))
    assert registry.execute("submit_finding", finding_args(room)).ok
    second = registry.execute("submit_finding", finding_args(room, title="Again"))
    assert second.error.code == "finding_budget_exhausted"
    orphan = build_registry(session, None, None)
    assert orphan.execute("submit_finding", finding_args(room)).error.code == "no_investigation"


def test_saving_a_finding_writes_a_trace_event_with_the_searches_run(session, room):
    class Stub:
        def search(self, query, **kwargs):
            return []

    registry = build_registry(session, Stub(), room.investigation.id)
    registry.execute("search", {"query": "capex rent provision"})
    registry.execute("submit_finding", finding_args(room))
    event = session.scalars(select(InvestigationEvent).where(InvestigationEvent.event_type == "finding_saved")).one()
    assert event.details["searches_since_last_finding"] == ["capex rent provision"]
    assert registry.context.recent_queries == []


def test_find_source_span():
    assert find_source_span("a  b\nc", "a  b") == ("a  b", False)
    assert find_source_span("a  b\nc", "a b c") == ("a  b\nc", True)
    assert find_source_span("a b c", "a b d") is None
    assert find_source_span("a b c", "   ") is None


# ---- search -------------------------------------------------------------------
def test_search_returns_hits_with_locators_and_logs_the_query(session, room):
    class Stub:
        def search(self, query, top_k=5, **kwargs):
            return [RetrievedChunk(chunk_id=room.capex_current, document_id=room.capex.id, filename="capex.xlsx",
                                   score=0.83456, content="Capacity | 100,000 tonnes", content_type="table_row",
                                   sheet_name="Current", cell_reference="B3")]

    registry = build_registry(session, Stub(), room.investigation.id)
    out = registry.execute("search", {"query": "plant capacity"}).output
    assert out.hits[0].locator == "sheet Current, cell B3" and out.hits[0].score == 0.8346
    assert registry.context.recent_queries == ["plant capacity"]


def test_search_without_a_backend_or_on_sqlite_is_a_clean_error(session, room):
    assert build_registry(session, None, None).execute("search", {"query": "x"}).error.code == "search_unavailable"
    from app.llm.embeddings import MockEmbeddingProvider
    sqlite_backed = build_registry(session, RetrievalService(session, MockEmbeddingProvider()), None)
    result = sqlite_backed.execute("search", {"query": "x"})
    assert result.error.code == "search_unavailable" and "PostgreSQL" in result.error.message


def test_search_top_k_is_bounded(registry):
    assert registry.execute("search", {"query": "x", "top_k": 50}).error.code == "invalid_arguments"
    assert SearchInput(query="x").top_k == 5


# ---- registry -----------------------------------------------------------------
def test_unknown_tool_lists_the_available_ones(registry):
    result = registry.execute("delete_everything")
    assert result.error.code == "unknown_tool" and "list_documents" in result.error.message


def test_unexpected_arguments_are_rejected_not_ignored(registry):
    result = registry.execute("list_documents", {"limit": 3})
    assert result.error.code == "invalid_arguments"


def test_budget_counts_every_call_including_failures(session, room):
    registry = build_registry(session, None, None, ToolBudget(max_tool_calls=2))
    registry.execute("get_chunk", {"chunk_id": "chk_nope"})  # fails, still counts
    registry.execute("list_documents")
    third = registry.execute("list_documents")
    assert third.error.code == "budget_exhausted" and registry.context.tool_calls_used == 2


def test_call_log_and_latency_by_tool(registry, room):
    registry.execute("list_documents")
    registry.execute("calculate", {"expression": "1+1"})
    registry.execute("get_chunk", {"chunk_id": "chk_nope"})
    log = registry.context.calls
    assert [(c.index, c.tool, c.ok) for c in log] == [(1, "list_documents", True), (2, "calculate", True), (3, "get_chunk", False)]
    assert log[2].error_code == "not_found" and log[1].result_summary == "1+1 = 2"
    summary = registry.latency_summary()
    assert summary.calls == 3 and summary.failed_calls == 1
    assert set(registry.latency_by_tool()) == {"list_documents", "calculate", "get_chunk"}


def test_a_crashing_tool_returns_an_error_instead_of_raising(session):
    def boom(ctx, inp):
        raise RuntimeError("secret internal detail")

    from app.tools.schemas import ListDocumentsInput
    registry = ToolRegistry(ToolContext(session=session), [ToolSpec("boom", "x", ListDocumentsInput, boom, str)])
    result = registry.execute("boom")
    assert result.error.code == "internal_error" and "secret" not in result.error.message


def test_describe_lists_every_tool_with_its_input_schema(registry):
    described = {d.name: d for d in registry.describe()}
    assert set(described) == {"list_documents", "get_outline", "search", "get_chunk", "read_section", "calculate",
                              "record_fact", "submit_finding"}
    assert "document_id" in described["get_outline"].input_schema["properties"]
    schema = str(described["submit_finding"].input_schema)
    assert "contradiction" in schema and "missing_information" in schema


# ---- fix: sheet and heading names match ignoring case and surrounding spaces ---
def test_read_section_finds_a_sheet_whose_real_name_ends_with_a_space(session, room, registry):
    doc, _ = add_doc(session, "model.xlsm", "xlsx", [
        dict(content="[Biochar CAPEX OPEX !A1:C1] Total: 14475000", sheet_name="Biochar CAPEX OPEX ", cell_reference="A1:C1", content_type="table_row")])
    session.commit()
    for asked in ("Biochar CAPEX OPEX", "biochar capex opex", "  Biochar CAPEX OPEX  "):
        result = registry.execute("read_section", {"document_id": doc.id, "sheet_name": asked})
        assert result.ok and result.output.chunks[0].sheet_name == "Biochar CAPEX OPEX ", asked


def test_read_section_headings_ignore_surrounding_spaces_too(registry, room):
    assert registry.execute("read_section", {"document_id": room.lease.id, "heading": "  RENT "}).ok


def test_a_wrong_sheet_name_gets_close_suggestions(session, room, registry):
    doc, _ = add_doc(session, "model.xlsm", "xlsx", [
        dict(content="x", sheet_name="Biochar CAPEX OPEX ", cell_reference="A1:B1", content_type="table_row"),
        dict(content="y", sheet_name="Revenue", cell_reference="A1:B1", content_type="table_row")])
    session.commit()
    result = registry.execute("read_section", {"document_id": doc.id, "sheet_name": "Biochar CAPEX"})
    assert result.error.code == "not_found" and "Did you mean: 'Biochar CAPEX OPEX'" in result.error.message
    assert "get_outline" in result.error.message


# ---- fix: coverage tracking -----------------------------------------------------

def test_a_chunk_counts_as_read_once_its_text_was_shown(session, room, registry):
    class Stub:
        def search(self, query, **kwargs):
            return [RetrievedChunk(chunk_id=room.deck_p1, document_id=room.deck.id, filename="deck.pdf", score=0.9,
                                   content="x", content_type="paragraph", page_number=1)]

    registry.context.retrieval = Stub()
    state = lambda: {d.filename: (d.chunks_read, d.chunks, d.unread_sections) for d in registry.coverage().not_fully_read}
    assert state() == {"capex.xlsx": (0, 2, ["Sheet: Current", "Sheet: Expansion"]), "deck.pdf": (0, 2, ["Page 1", "Page 2"]),
                       "lease.docx": (0, 3, ["Rent", "Term"])}  # the missing document is not counted

    registry.execute("get_outline", {"document_id": room.deck.id})
    assert state()["deck.pdf"][0] == 0  # an outline shows no chunk text

    registry.execute("search", {"query": "capacity"})  # a search hit shows the whole chunk
    assert state()["deck.pdf"] == (1, 2, ["Page 2"])

    registry.execute("get_chunk", {"chunk_id": room.capex_current, "neighbors": 1})  # neighbours are shown too
    assert "capex.xlsx" not in state()

    registry.execute("read_section", {"document_id": room.lease.id, "heading": "Rent"})
    assert state()["lease.docx"] == (2, 3, ["Term"])

    registry.execute("read_section", {"document_id": room.lease.id, "heading": "Term"})
    registry.execute("read_section", {"document_id": room.deck.id, "page_number": 2})
    cov = registry.coverage()
    assert (cov.total, cov.fully_read, cov.not_fully_read) == (3, 3, [])



def test_list_documents_shows_how_many_chunks_were_read(registry, room):
    registry.execute("read_section", {"document_id": room.capex.id, "sheet_name": "Current"})
    docs = {d.filename: d.chunks_read for d in registry.execute("list_documents").output.documents}
    assert docs == {"capex.xlsx": 1, "deck.pdf": 0, "lease.docx": 0}


def test_a_document_without_chunks_is_not_expected_to_be_opened(session, room, registry):
    add_doc(session, "scan.pdf", "pdf", [], status="skipped")
    session.commit()
    assert registry.coverage().total == 3
