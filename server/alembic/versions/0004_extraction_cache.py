"""extraction_cache: the raw LlamaExtract response per (file hash, schema hash, prompt hash, tier, pages)

Revision ID: 0004
Revises: 0003

A run on an unchanged file, schema and prompt reuses the stored response instead of paying for the job again.
See app/models/extraction_cache.py. Dropping the table only means the next run extracts again.
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None

JSON = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.create_table(
        "extraction_cache",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("file_sha256", sa.String(64), nullable=False),
        sa.Column("schema_hash", sa.String(64), nullable=False),
        sa.Column("prompt_hash", sa.String(64), nullable=False),
        sa.Column("tier", sa.String(32), nullable=False),
        sa.Column("pages", sa.String(32), nullable=False, server_default=""),
        sa.Column("result", JSON, nullable=False),
        sa.Column("extract_metadata", JSON, nullable=True),
        sa.Column("credits", sa.Float, nullable=False, server_default="0"),
        sa.Column("pages_billed", sa.Integer, nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("file_sha256", "schema_hash", "prompt_hash", "tier", "pages", name="uq_extraction_cache_key"),
    )


def downgrade() -> None:
    op.drop_table("extraction_cache")
