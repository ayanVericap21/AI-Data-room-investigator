"""The findings endpoints: findings grouped by category with their evidence and source location."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core.database import get_session
from app.main import create_app
from app.models import Investigation
from tests.test_tools import finding_args, registry, room  # noqa: F401  (fixtures/helpers)


@pytest.fixture
def client(session):
    app = create_app()
    app.dependency_overrides[get_session] = lambda: session
    return TestClient(app)


def save(registry, room, **overrides):  # noqa: F811
    result = registry.execute("submit_finding", finding_args(room, **overrides))
    assert result.ok, result.error
    return result.output.finding_id


def test_nothing_run_yet_gives_an_empty_answer(client):
    body = client.get("/api/findings").json()
    assert body == {"investigation": None, "counts": {}, "findings": []}
    assert client.get("/api/investigations").json() == []


def test_findings_are_grouped_contradictions_first_then_by_severity_with_their_sources(client, registry, room):  # noqa: F811
    context_id = save(registry, room, classification="contextual_difference", severity="low", title="Context")
    medium_id = save(registry, room, severity="medium", title="Medium contradiction")
    high_id = save(registry, room, severity="high", title="High contradiction")
    body = client.get("/api/findings").json()
    assert body["investigation"]["id"] == room.investigation.id and body["investigation"]["finding_count"] == 3
    assert body["counts"] == {"contradiction": 2, "contextual_difference": 1}
    assert [f["id"] for f in body["findings"]] == [high_id, medium_id, context_id]
    first = body["findings"][0]
    assert first["verification_status"] == "verified" and len(first["evidence"]) == 2
    sources = {e["source"]["filename"]: e for e in first["evidence"]}
    assert sources["deck.pdf"]["source"]["page_number"] == 1 and sources["deck.pdf"]["quoted_text"] == "capacity is 120,000 tonnes per year"
    assert sources["capex.xlsx"]["source"]["sheet_name"] == "Current" and sources["capex.xlsx"]["source"]["cell_reference"] == "B3"
    assert sources["deck.pdf"]["source"]["chunk_id"] == room.deck_p1


def test_a_run_can_be_chosen_by_id_and_unknown_ids_are_404(client, registry, room, session):  # noqa: F811
    save(registry, room)
    other = Investigation(name="second run", status="completed")
    session.add(other)
    session.commit()
    assert client.get(f"/api/findings?investigation_id={other.id}").json()["findings"] == []
    assert client.get(f"/api/findings?investigation_id={room.investigation.id}").json()["counts"] == {"contradiction": 1}
    assert client.get("/api/findings?investigation_id=inv_nope").status_code == 404
    runs = client.get("/api/investigations").json()
    assert {r["id"]: r["finding_count"] for r in runs} == {room.investigation.id: 1, other.id: 0}
