from __future__ import annotations

import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

SERVER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVER_DIR))

from app.core.config import Settings  # noqa: E402
from app.models import Base  # noqa: E402


@pytest.fixture
def data_room(tmp_path: Path) -> Path:
    root = tmp_path / "data_room"
    root.mkdir()
    return root


@pytest.fixture
def settings(data_room: Path) -> Settings:
    return Settings(data_room_path=data_room)


@pytest.fixture
def session_factory():
    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    yield sessionmaker(bind=engine, expire_on_commit=False)
    engine.dispose()


@pytest.fixture
def session(session_factory):
    with session_factory() as s:
        yield s


# ---- sample document builders -------------------------------------------------
def make_pdf(path: Path, pages: list[str]) -> Path:
    import pymupdf as fitz

    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 100), text)
    doc.save(path)
    doc.close()
    return path


def make_docx(path: Path) -> Path:
    import docx

    d = docx.Document()
    d.add_heading("Operations", level=1)
    d.add_paragraph("Annual production capacity is estimated at 30000 tonnes.")
    table = d.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Metric"
    table.cell(0, 1).text = "Value"
    table.cell(1, 0).text = "Capacity"
    table.cell(1, 1).text = "30000"
    d.add_heading("Finance", level=1)
    d.add_paragraph("Revenue projections assume full capacity.")
    d.save(path)
    return path


def make_xlsx(path: Path) -> Path:
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Revenue"
    ws.append(["Product", "Tonnes", "Price", "Revenue"])
    ws.append(["Steel", 100, 5, "=B2*C2"])
    wb.create_sheet("Notes").append(["Only a note"])
    wb.save(path)  # saved by openpyxl: formulas have NO cached values
    return path
