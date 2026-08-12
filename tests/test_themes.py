"""Tests for the pure (no-network) logic in app/themes.py: citation
verification, candidate-pair selection for cluster merging, and union-find
merging. The LLM-calling parts (embedding, merge judgment, naming) are
validated live against real data in eval/run_cluster_eval.py — mocking a
generative model's output convincingly is its own can of worms, so instead
of fake-mocking it, the pure logic around it is unit-tested and the LLM
calls themselves are smoke-tested live (same pattern as the rest of the
project's LLM integration)."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import themes
from app.themes import _candidate_pairs, _UnionFind, name_and_recommend, verify_citations


# --------------------------------------------------------------------------- #
# citation verification
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# union-find merging
# --------------------------------------------------------------------------- #
def test_union_find_merges_transitively():
    uf = _UnionFind(4)
    uf.union(0, 1)
    uf.union(1, 2)
    assert uf.find(0) == uf.find(2)
    assert uf.find(0) != uf.find(3)


def test_union_find_no_op_on_already_merged():
    uf = _UnionFind(3)
    uf.union(0, 1)
    root_before = uf.find(0)
    uf.union(0, 1)  # merging the same pair again should not break anything
    assert uf.find(0) == root_before


# --------------------------------------------------------------------------- #
# candidate-pair selection for the merge step
# --------------------------------------------------------------------------- #
def fake_review(rid):
    return {"id": rid, "title": "", "text": ""}


def test_candidate_pairs_only_returns_pairs_in_the_band():
    clusters = [[fake_review("a")], [fake_review("b")], [fake_review("c")]]
    ids = ["a", "b", "c"]
    matrix = [
        [1.0, 0.60, 0.30],   # a-b in band [0.55,0.68), a-c below band
        [0.60, 1.0, 0.90],   # b-c above band (would already have merged during clustering)
        [0.30, 0.90, 1.0],
    ]
    pairs = _candidate_pairs(clusters, matrix, ids, low=0.55, high=0.68)
    assert pairs == [(0, 1, 0.60)]


def test_candidate_pairs_sorted_most_plausible_first():
    clusters = [[fake_review("a")], [fake_review("b")], [fake_review("c")]]
    ids = ["a", "b", "c"]
    matrix = [
        [1.0, 0.56, 0.60],
        [0.56, 1.0, 0.30],
        [0.60, 0.30, 1.0],
    ]
    pairs = _candidate_pairs(clusters, matrix, ids, low=0.55, high=0.68)
    assert [p[:2] for p in pairs] == [(0, 2), (0, 1)]  # 0.60 before 0.56


def test_candidate_pairs_empty_when_nothing_in_band():
    clusters = [[fake_review("a")], [fake_review("b")]]
    ids = ["a", "b"]
    matrix = [[1.0, 0.2], [0.2, 1.0]]
    assert _candidate_pairs(clusters, matrix, ids, low=0.55, high=0.68) == []


# --------------------------------------------------------------------------- #
# name_and_recommend()'s output shape — the real function, network mocked
# --------------------------------------------------------------------------- #
def test_name_and_recommend_output_has_every_field_downstream_needs(monkeypatch):
    """Regression test: name_and_recommend() previously omitted
    share_of_negative, which both analysis.actions_and_summary() and
    report.py read directly (t["share_of_negative"]) — a KeyError that
    reached a live end-to-end run (main.py -> real Gemini call) undetected,
    because every other test mocked llm_theme_analysis()/llm.call() at a
    level that let a hand-written "expected" shape stand in for the real
    one. This test calls the actual name_and_recommend(), mocking only the
    network boundary (llm.call), so a future dropped field fails here
    instead of at request time."""
    clusters = [
        [{"id": "1", "title": "Scam", "text": "charged twice", "rating": 1},
         {"id": "2", "title": "Scam2", "text": "billed again", "rating": 2}],
    ]

    async def fake_call(system, payload, schema, **kwargs):
        return {"themes": [{"cluster_id": "0", "name": "Billing", "description": "...",
                            "recommendation": "Fix billing.", "cited_review_ids": ["1", "2"]}]}

    monkeypatch.setattr(themes.llm, "call", fake_call)
    result = asyncio.run(name_and_recommend(clusters))

    required = {"theme", "description", "negative_reviews", "share_of_negative", "avg_rating",
               "recommendation", "sample_quotes", "cited_review_ids", "citations_valid"}
    assert required.issubset(result[0].keys())
    assert result[0]["share_of_negative"] == 100.0
    assert result[0]["negative_reviews"] == 2
    assert result[0]["avg_rating"] == 1.5
