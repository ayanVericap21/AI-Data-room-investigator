"""Duplicates by claims and quotes, evidence keeps its role, agreements about different subjects are judged, missing items per sentence,
insufficient evidence absorbed by a consistent finding, consistent findings grouped by set of documents and attribute."""
from sqlalchemy import select

from app.llm.mock import MockLLMClient
from app.models import Claim, Evidence, Finding
from app.services.missing_items import DocReference, MissingItemChecker
from app.services.claim_group_judge import ClaimGroupJudge
from app.tools import build_registry
from tests.test_claim_group_judge import make_claims  # noqa: F401
from tests.test_tools import add_doc, room  # noqa: F401  (fixtures and helper)




# ---- 1: a start date and a volume stated by the same two passages are two findings ---------------


# ---- 2: merged evidence keeps its own role ---------------------------------------------------------


# ---- 3: equal values about different subjects are not an agreement until the judge says so --------------


# ---- 4: missing items named by one sentence are one finding ------------------------------------------------
def test_items_named_by_one_sentence_are_one_missing_information_finding(session, room):  # noqa: F811
    sentence = "Annual rent is INR 1.2 crore."
    refs = [DocReference(room.lease.id, room.rent_1, item, sentence) for item in ("MES exports", "uptime logs", "acceptance records")]
    stats = MissingItemChecker(session, build_registry(session, None, room.investigation.id)).run(refs)
    finding = session.scalars(select(Finding)).one()
    assert stats.saved == 1 and stats.missing_items == ["MES exports", "uptime logs", "acceptance records"]
    assert all(item in finding.title for item in ("MES exports", "uptime logs", "acceptance records"))
    assert len(finding.unresolved_questions) == 3


# ---- 5: a consistent finding absorbs an insufficient-evidence one about the same fact ---------------------------


# ---- 6: consistent findings are grouped by set of documents and attribute -------------------------------------
