import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.documents import InvalidUpload, safe_filename, safe_relpath, save_upload


# ---------- path sanitising ----------
@pytest.mark.parametrize("raw,expected", [
    ("report.pdf", "report.pdf"),
    ("data-room/legal/lease.pdf", "data-room/legal/lease.pdf"),
    ("data room\\finance\\model v2.csv", "data room/finance/model v2.csv"),   # Windows separators
    ("../../etc/passwd.txt", "etc/passwd.txt"),                                # traversal dropped
    ("/abs/olute/./path/../a.md", "abs/olute/path/a.md"),
    ("a//b///c.txt", "a/b/c.txt"),
    ("weird<>name?.txt", "weird__name_.txt"),
])
def test_safe_relpath(raw, expected):
    assert safe_relpath(raw) == expected


def test_safe_relpath_limits_and_rejects_empty():
    deep = "/".join(f"d{i}" for i in range(20)) + "/f.txt"
    out = safe_relpath(deep)
    assert out.count("/") == 7 and out.endswith("d19/f.txt") and len(out) <= 200
    for bad in ["", "   ", "../..", "/", "\\"]:
        with pytest.raises(InvalidUpload):
            safe_relpath(bad)
    assert safe_filename("a/b/c.txt") == "c.txt"


# ---------- storage keeps folder context ----------
def test_relative_path_is_kept_as_document_name(db, settings):
    doc, _ = save_upload(db, settings, "data-room/legal/lease.txt", b"lease terms are described here")
    assert doc.filename == "data-room/legal/lease.txt"
    from pathlib import Path
    stored = Path(doc.stored_path)
    assert stored.parent == settings.uploads_dir and stored.name.endswith("_lease.txt") and "/" not in stored.name


def test_same_filename_in_different_folders_are_distinct_documents(db, settings):
    a, _ = save_upload(db, settings, "2024/report.txt", b"Annual report for 2024: capacity was 25,000 tonnes.")
    b, _ = save_upload(db, settings, "2025/report.txt", b"Annual report for 2025: capacity was 30,000 tonnes.")
    assert a.id != b.id and {a.filename, b.filename} == {"2024/report.txt", "2025/report.txt"}
    assert a.stored_path != b.stored_path


def test_extension_checked_on_the_file_not_the_folder(db, settings):
    with pytest.raises(InvalidUpload):
        save_upload(db, settings, "archive.pdf/notes.exe", b"data")
    doc, _ = save_upload(db, settings, "weird.exe/notes.txt", b"plain notes inside a folder named like an exe")
    assert doc.filename == "weird.exe/notes.txt"


# ---------- through the API (what the browser sends) ----------
@pytest.fixture
def client(settings):
    return TestClient(create_app(settings, run_inline=True))


def test_api_folder_upload_preserves_structure(client):
    files = [("files", ("proj/finance/model.csv", b"year,revenue\n2025,10\n", "text/csv")),
             ("files", ("proj/legal/sub/lease.txt", b"The lease runs for ten years.", "text/plain")),
             ("files", ("proj/legal/sub/photo.png", b"\x89PNG", "image/png"))]
    body = client.post("/api/v1/documents", files=files).json()
    assert [b["document"]["filename"] if b["document"] else None for b in body] == \
           ["proj/finance/model.csv", "proj/legal/sub/lease.txt", None]
    assert "unsupported" in body[2]["error"]          # one bad file does not reject the folder
    assert {d["filename"] for d in client.get("/api/v1/documents").json()} == {"proj/finance/model.csv", "proj/legal/sub/lease.txt"}


def test_cors_allows_only_configured_ui_origins(client):
    pre = client.options("/api/v1/documents", headers={"Origin": "http://localhost:8501",
                                                       "Access-Control-Request-Method": "POST"})
    assert pre.status_code == 200 and pre.headers["access-control-allow-origin"] == "http://localhost:8501"
    other = client.options("/api/v1/documents", headers={"Origin": "http://evil.example",
                                                         "Access-Control-Request-Method": "POST"})
    assert "access-control-allow-origin" not in other.headers
    ok = client.get("/api/v1/health", headers={"Origin": "http://127.0.0.1:8501"})
    assert ok.headers["access-control-allow-origin"] == "http://127.0.0.1:8501"


def test_cors_origins_from_env(monkeypatch):
    from app.config import get_settings
    monkeypatch.setenv("DRI_CORS_ORIGINS", "http://a.test, http://b.test")
    assert get_settings().cors_origins == ["http://a.test", "http://b.test"]
