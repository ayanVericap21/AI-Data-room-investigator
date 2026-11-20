"""claim_label_cache table."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, JsonType, new_id, tz_datetime, utcnow


class ClaimLabelCacheEntry(Base):
    """What the labelling model said about one claim (or about a set of attribute names), kept so a rerun does not ask again.

    Written by: the claim labeller (key grouping), after a labelling or vocabulary call. Used by: the same, before it calls the model.
    ``kind`` is "label" (one claim: ``key_hash`` is the hash of its text, subject, metric, value, unit and period) or "vocabulary" (one merge of
    attribute names: ``key_hash`` is the hash of the sorted names). ``prompt_hash`` is the hash of the instructions sent, so an edited prompt never
    reads an old answer. Claim ids are not part of the key: claims are rebuilt with new ids on every run.
    Not tied to ``claims``: clearing results does not touch it. Dropping the table only means the next run asks again.
    """

    __tablename__ = "claim_label_cache"
    __table_args__ = (UniqueConstraint("kind", "key_hash", "prompt_hash", name="uq_claim_label_cache_key"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: new_id("clb"))
    kind: Mapped[str] = mapped_column(String(16))
    key_hash: Mapped[str] = mapped_column(String(64))
    prompt_hash: Mapped[str] = mapped_column(String(64))
    payload: Mapped[object] = mapped_column(JsonType)
    created_at: Mapped[datetime] = mapped_column(tz_datetime(), default=utcnow)
