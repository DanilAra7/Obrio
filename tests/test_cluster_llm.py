"""Tests for the parts of eval/cluster_llm.py that don't need the network:
candidate-pair selection and union-find merging. The LLM-calling parts
(merge judgment, naming) are validated live against real data in
eval/run_cluster_eval.py, same pattern as the rest of the LLM integration."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.cluster_llm import _UnionFind, _candidate_pairs


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
