"""initial schema

Revision ID: 0001
Revises:
See app/models/*.py for the documentation of every table and column.
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

JSON = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")
TS = sa.DateTime(timezone=True)


def upgrade() -> None:
    op.create_table(
        "documents",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("filename", sa.String(512), nullable=False),
        sa.Column("file_hash", sa.String(64), nullable=False),
        sa.Column("file_type", sa.String(8), nullable=False),
        sa.Column("file_size", sa.Integer, nullable=False),
        sa.Column("processing_status", sa.String(16), nullable=False),
        sa.Column("last_modified_at", TS, nullable=False),
        sa.Column("created_at", TS, nullable=False),
        sa.UniqueConstraint("file_hash", name="uq_documents_file_hash"),
    )
    op.create_index("ix_documents_filename", "documents", ["filename"])

    op.create_table(
        "document_chunks",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("document_id", sa.String(32), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("chunk_index", sa.Integer, nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("content_type", sa.String(32), nullable=False),
        sa.Column("page_number", sa.Integer),
        sa.Column("sheet_name", sa.String(255)),
        sa.Column("cell_reference", sa.String(64)),
        sa.Column("heading", sa.Text),
        sa.Column("metadata", JSON, nullable=False),
        sa.Column("created_at", TS, nullable=False),
        sa.UniqueConstraint("document_id", "chunk_index", name="uq_chunks_doc_index"),
    )
    op.create_index("ix_chunks_document", "document_chunks", ["document_id"])

    op.create_table(
        "claims",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("document_id", sa.String(32), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("chunk_id", sa.String(32), sa.ForeignKey("document_chunks.id", ondelete="CASCADE"), nullable=False),
        sa.Column("subject", sa.Text),
        sa.Column("metric", sa.Text),
        sa.Column("value", JSON),
        sa.Column("unit", sa.String(64)),
        sa.Column("period", sa.String(128)),
        sa.Column("conditions", sa.Text),
        sa.Column("assumptions", JSON),
        sa.Column("claim_text", sa.Text, nullable=False),
        sa.Column("claim_category", sa.String(64)),
        sa.Column("extraction_confidence", sa.Float),
        sa.Column("created_at", TS, nullable=False),
    )
    op.create_index("ix_claims_document", "claims", ["document_id"])
    op.create_index("ix_claims_metric", "claims", ["metric"])

    op.create_table(
        "investigations",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("status", sa.String(24), nullable=False),
        sa.Column("approach", sa.String(16), nullable=False),
        sa.Column("started_at", TS),
        sa.Column("completed_at", TS),
        sa.Column("stopped_at", TS),
        sa.Column("error_message", sa.Text),
        sa.Column("execution_config", JSON, nullable=False),
        sa.Column("total_tokens", sa.BigInteger, nullable=False),
        sa.Column("estimated_cost", sa.Numeric(12, 6), nullable=False),
        sa.Column("duration_ms", sa.BigInteger),
        sa.Column("created_at", TS, nullable=False),
    )
    op.create_index("ix_investigations_status", "investigations", ["status"])

    op.create_table(
        "investigation_events",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("investigation_id", sa.String(32), sa.ForeignKey("investigations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("event_type", sa.String(64), nullable=False),
        sa.Column("node_name", sa.String(64)),
        sa.Column("message", sa.Text, nullable=False),
        sa.Column("status", sa.String(24), nullable=False),
        sa.Column("details", JSON, nullable=False),
        sa.Column("created_at", TS, nullable=False),
    )
    op.create_index("ix_events_investigation_created", "investigation_events", ["investigation_id", "created_at"])

    op.create_table(
        "findings",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("investigation_id", sa.String(32), sa.ForeignKey("investigations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("title", sa.String(512), nullable=False),
        sa.Column("description", sa.Text, nullable=False),
        sa.Column("classification", sa.String(32), nullable=False),
        sa.Column("severity", sa.String(16), nullable=False),
        sa.Column("confidence", sa.Float, nullable=False),
        sa.Column("materiality_reasoning", sa.Text),
        sa.Column("business_impact", sa.Text),
        sa.Column("evidence_status", sa.String(16), nullable=False),
        sa.Column("verification_status", sa.String(16), nullable=False),
        sa.Column("unresolved_questions", JSON, nullable=False),
        sa.Column("created_at", TS, nullable=False),
        sa.Column("updated_at", TS, nullable=False),
    )
    op.create_index("ix_findings_investigation", "findings", ["investigation_id"])

    op.create_table(
        "finding_claims",
        sa.Column("finding_id", sa.String(32), sa.ForeignKey("findings.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("claim_id", sa.String(32), sa.ForeignKey("claims.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("relationship_type", sa.String(16), nullable=False),
    )

    op.create_table(
        "evidence",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("finding_id", sa.String(32), sa.ForeignKey("findings.id", ondelete="CASCADE"), nullable=False),
        sa.Column("chunk_id", sa.String(32), sa.ForeignKey("document_chunks.id"), nullable=False),
        sa.Column("claim_id", sa.String(32), sa.ForeignKey("claims.id")),
        sa.Column("evidence_role", sa.String(16), nullable=False),
        sa.Column("quoted_text", sa.Text, nullable=False),
        sa.Column("relevance_reason", sa.Text, nullable=False),
        sa.Column("support_assessment", sa.Text, nullable=False),
        sa.Column("created_at", TS, nullable=False),
    )
    op.create_index("ix_evidence_finding", "evidence", ["finding_id"])


def downgrade() -> None:
    for table in (
        "evidence", "finding_claims", "findings", "investigation_events",
        "investigations", "claims", "document_chunks", "documents",
    ):
        op.drop_table(table)
