import os
import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings  # noqa: E402
from app.db import Database  # noqa: E402

PG_URL = os.environ.get("DRI_TEST_DATABASE_URL")
# SQLite always runs (zero setup). The identical suite also runs on Postgres when DRI_TEST_DATABASE_URL is set
# (e.g. the docker compose `db` service); each test gets its own schema, dropped afterwards.
BACKENDS = ["sqlite"] + (["postgres"] if PG_URL else [])


def pytest_report_header(config):
    return ("database backends: sqlite" + (", postgres" if PG_URL else
            "  (postgres NOT exercised: set DRI_TEST_DATABASE_URL, e.g. after `docker compose up -d db`)"))


def make_pdf(pages: list[str]) -> bytes:
    """Hand-built minimal PDF with one text line per page (no extra deps)."""
    objs: list[bytes] = []
    n = len(pages)
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(n))
    objs.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objs.append(f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode())
    for i, text in enumerate(pages):
        content = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {4 + 2 * i} 0 R "
                    f"/Resources << /Font << /F1 {3 + 2 * n} 0 R >> >> >>".encode())
        objs.append(b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream")
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, o in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + o + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    return bytes(out)


@pytest.fixture(params=BACKENDS)
def settings(request, tmp_path):
    if request.param == "sqlite":
        url, schema = f"sqlite:///{tmp_path / 'test.sqlite3'}", None
    else:
        url, schema = PG_URL, f"t_{uuid.uuid4().hex[:16]}"
    s = Settings(data_dir=tmp_path / "data", chunk_max_chars=300, database_url=url, db_schema=schema)
    yield s
    if request.param == "postgres":
        Database(url, schema, init=False).drop_schema()


@pytest.fixture
def db(settings):
    return Database(settings.database_url, settings.db_schema)
