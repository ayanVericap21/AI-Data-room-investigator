"""add pgvector embedding column to document_chunks

Revision ID: 0002
Revises: 0001
Embeddings live in PostgreSQL (no separate vector database). The column width must match
app.models.base.EMBEDDING_DIM.
"""
import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None

DIM = 1536


def upgrade() -> None:
    if op.get_bind().dialect.name == "postgresql":
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")
        op.execute(f"ALTER TABLE document_chunks ADD COLUMN embedding vector({DIM})")
        op.execute(
            "CREATE INDEX ix_chunks_embedding_hnsw ON document_chunks "
            "USING hnsw (embedding vector_cosine_ops)"
        )
    else:  # SQLite (unit tests): plain JSON, no similarity index
        op.add_column("document_chunks", sa.Column("embedding", sa.JSON))


def downgrade() -> None:
    if op.get_bind().dialect.name == "postgresql":
        op.execute("DROP INDEX IF EXISTS ix_chunks_embedding_hnsw")
    with op.batch_alter_table("document_chunks") as batch:
        batch.drop_column("embedding")
