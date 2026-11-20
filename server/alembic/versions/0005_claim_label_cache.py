"""claim_label_cache: the labelling model's answer per claim (or per set of attribute names), by hash

Revision ID: 0005
Revises: 0004

Used by key grouping (--grouping key). See app/models/claim_label_cache.py. Dropping the table only means the next run asks the model again.
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None

JSON = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.create_table(
        "claim_label_cache",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column("key_hash", sa.String(64), nullable=False),
        sa.Column("prompt_hash", sa.String(64), nullable=False),
        sa.Column("payload", JSON, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("kind", "key_hash", "prompt_hash", name="uq_claim_label_cache_key"),
    )


def downgrade() -> None:
    op.drop_table("claim_label_cache")
