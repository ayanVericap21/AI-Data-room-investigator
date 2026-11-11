"""document_chunks.embedding: vector(1536) -> vector(384) for the local all-MiniLM-L6-v2 model

Revision ID: 0003
Revises: 0002

Vectors of one size cannot be converted to another, so existing embeddings are cleared and the
documents that had them are set back to ``pending``. Their chunk text is kept; the next
"Process documents" run re-embeds the chunks without parsing the files again.
SQLite (unit tests) stores embeddings as JSON, so nothing changes there.
"""
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None

NEW_DIM = 384
OLD_DIM = 1536


def _resize(dim: int) -> None:
    if op.get_bind().dialect.name != "postgresql":
        return
    op.execute("DROP INDEX IF EXISTS ix_chunks_embedding_hnsw")
    op.execute("UPDATE document_chunks SET embedding = NULL")
    op.execute("UPDATE documents SET processing_status = 'pending' WHERE processing_status = 'processed'")
    op.execute(f"ALTER TABLE document_chunks ALTER COLUMN embedding TYPE vector({dim})")
    op.execute("CREATE INDEX ix_chunks_embedding_hnsw ON document_chunks USING hnsw (embedding vector_cosine_ops)")


def upgrade() -> None:
    _resize(NEW_DIM)


def downgrade() -> None:
    _resize(OLD_DIM)
