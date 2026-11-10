"""The Alembic migration must produce the same tables/columns as the ORM models."""
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect

from app.models import Base

SERVER = Path(__file__).resolve().parents[1]


def test_migration_matches_models(tmp_path):
    url = f"sqlite:///{tmp_path / 'm.db'}"
    cfg = Config(str(SERVER / "alembic.ini"))
    cfg.set_main_option("script_location", str(SERVER / "alembic"))
    cfg.set_main_option("sqlalchemy.url", url)
    command.upgrade(cfg, "head")

    insp = inspect(create_engine(url))
    for name, table in Base.metadata.tables.items():
        assert name in insp.get_table_names(), name
        migrated = {c["name"] for c in insp.get_columns(name)}
        assert migrated == {c.name for c in table.columns}, name

    command.downgrade(cfg, "base")
    assert set(inspect(create_engine(url)).get_table_names()) <= {"alembic_version"}
