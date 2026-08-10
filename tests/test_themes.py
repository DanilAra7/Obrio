import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.themes import verify_citations


def test_verify_citations_keeps_real_citations():
    recs = [{"theme": "Billing", "recommendation": "Fix refunds.", "cited_review_ids": ["1", "2"]}]
    valid = {"Billing": {"1", "2", "3"}}
    out = verify_citations(recs, valid)
    assert out[0]["cited_review_ids"] == ["1", "2"]
    assert out[0]["citations_valid"] is True
    assert out[0]["hallucinated_citations_dropped"] == 0


def test_verify_citations_drops_hallucinated_ids():
    recs = [{"theme": "Billing", "recommendation": "Fix refunds.", "cited_review_ids": ["1", "999"]}]
    valid = {"Billing": {"1", "2", "3"}}
    out = verify_citations(recs, valid)
    assert out[0]["cited_review_ids"] == ["1"]
    assert out[0]["hallucinated_citations_dropped"] == 1
    assert out[0]["citations_valid"] is True  # still has one real citation


def test_verify_citations_flags_fully_hallucinated_recommendation():
    recs = [{"theme": "Billing", "recommendation": "Fix refunds.", "cited_review_ids": ["999", "888"]}]
    valid = {"Billing": {"1", "2", "3"}}
    out = verify_citations(recs, valid)
    assert out[0]["cited_review_ids"] == []
    assert out[0]["citations_valid"] is False
    assert out[0]["hallucinated_citations_dropped"] == 2


def test_verify_citations_unknown_theme_has_no_valid_ids():
    recs = [{"theme": "Nonexistent Theme", "recommendation": "...", "cited_review_ids": ["1"]}]
    out = verify_citations(recs, {"Billing": {"1"}})
    assert out[0]["citations_valid"] is False
