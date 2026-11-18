"""One discrepancy, one finding; label-only claims skipped; declared absences asked for; no finding about unrelated claims; no repeated unit."""
from sqlalchemy import select

from app.llm.mock import MockLLMClient
from app.models import Claim, Evidence, Finding
from app.schemas.claim import Candidate
from app.services.llama_claims import SCHEMA, SYSTEM_PROMPT, LlamaClaim, _is_own_label
from app.services.claim_group_judge import ClaimGroupJudge, _shown
from app.tools import build_registry
from tests.test_claim_group_judge import comparison, make_claims
from tests.test_tools import add_doc, room  # noqa: F401  (fixtures and helper)


# ---- 5 and 2: extraction -------------------------------------------------------
def test_a_claim_whose_value_is_only_its_own_row_label_is_not_a_claim():
    assert _is_own_label(LlamaClaim(subject="MET-004", metric="Tool_ID", value="MET-004"))
    assert _is_own_label(LlamaClaim(subject=" met-004 ", metric="Tool_ID", value="MET-004"))
    assert not _is_own_label(LlamaClaim(subject="MET-004", metric="Installed_Cost", value="26"))
    assert not _is_own_label(LlamaClaim(subject=None, metric="x", value="MET-004"))




# ---- 4: unit text -------------------------------------------------------------------
def test_a_unit_already_in_the_value_is_not_repeated():
    assert _shown(Claim(value="12,000 WSPM", unit="WSPM")) == "12,000 WSPM"
    assert _shown(Claim(value="24000", unit="WSPM")) == "24000 WSPM"
    assert _shown(Claim(value="96.8%", unit="%")) == "96.8%"
    assert _shown(Claim(value="Karnataka", unit=None)) == "Karnataka"


# ---- 3: no finding about unrelated claims -------------------------------------------


# ---- 1: one discrepancy, one finding --------------------------------------------------


def test_a_claim_that_names_an_entity_with_a_field_label_is_kept_while_titles_and_page_furniture_are_not():
    name = "Riverside Heritage Development LLC"
    assert not _is_own_label(LlamaClaim(subject=name, metric="taxpayer", value=name, claim_text="Taxpayer: Riverside Heritage Development LLC"))
    assert not _is_own_label(LlamaClaim(subject="Ashby Mill", metric="property name", value="Ashby Mill", claim_text="Property name and address: Ashby Mill, 410 Bridge Street"))
    assert _is_own_label(LlamaClaim(subject="Cost Certification Draft", metric="document title", value="Cost Certification Draft", claim_text="Ashby Mill Lofts | Cost Certification Draft"))
    assert _is_own_label(LlamaClaim(subject="Ashby Mill Lofts", metric="name", value="Ashby Mill Lofts", claim_text="Ashby Mill Lofts | Cost Certification Draft | Page 5 of 8"))
    assert _is_own_label(LlamaClaim(subject="Article 5 - Allocations", metric="section", value="Article 5 - Allocations", claim_text="Article 5 - Allocations"))
