import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.core.database import get_session
from app.main import create_app
from tests.conftest import make_pdf, make_xlsx


@pytest.fixture
def client(session_factory, settings):
    app = create_app()

    def _session():
        with session_factory() as s:
            yield s

    app.dependency_overrides[get_session] = _session
    app.dependency_overrides[get_settings] = lambda: settings
    return TestClient(app)


def test_scan_then_list_and_get(client, data_room):
    make_pdf(data_room / "a.pdf", ["p1", "p2", "p3"])
    make_xlsx(data_room / "b.xlsx")

    assert client.get("/api/documents").json() == {"items": [], "total": 0}
    scan = client.post("/api/documents/scan").json()
    assert scan["added"] == 2

    listing = client.get("/api/documents").json()
    assert listing["total"] == 2
    by_name = {d["filename"]: d for d in listing["items"]}
    assert by_name["a.pdf"]["unit_count"] == 3
    assert by_name["b.xlsx"]["unit_count"] == 2
    assert by_name["a.pdf"]["processing_status"] == "pending"
    assert "path" not in by_name["a.pdf"]
    assert str(data_room) not in client.get("/api/documents").text

    one = client.get(f"/api/documents/{by_name['a.pdf']['id']}")
    assert one.status_code == 200 and one.json()["file_type"] == "pdf"


def test_unknown_document_gives_consistent_error(client):
    r = client.get("/api/documents/doc_nope")
    assert r.status_code == 404
    assert r.json()["error"] == "not_found"
