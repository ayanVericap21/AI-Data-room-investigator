

def test_a_matching_view_without_a_claim_category_is_read_by_its_value_type():
    """The checks inside a document pass ClaimView objects (value_type), the judge passes claim rows (claim_category): both can be rated."""
    from app.schemas.claim import ClaimView
    from app.services.severity import contradiction_severity

    a = ClaimView(id="a", document_id="d1", chunk_id="c1", subject="Managing Member", metric="role", value="Managing Member", claim_text="shall manage", value_type="name")
    b = ClaimView(id="b", document_id="d2", chunk_id="c2", subject="Managing Member", metric="role", value="Administrator", claim_text="administers", value_type="name")
    assert contradiction_severity([a, b]) == "critical"  # a name that cannot be both, in a prominent (shall) statement
