import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.cluster import cluster_pair_similarity, cluster_reviews, complete_linkage_cluster, similarity_matrix


def fake_cosine(a, b):
    """1D toy embeddings: distance is just |a[0]-b[0]|, similarity = 1-distance."""
    return 1.0 - abs(a[0] - b[0])


def test_two_tight_groups_stay_separate():
    # group A near 0.0, group B near 1.0 — clearly two different topics
    vectors = [[0.0], [0.02], [0.03], [1.0], [0.98], [0.97]]
    ids = ["a1", "a2", "a3", "b1", "b2", "b3"]
    clusters = cluster_reviews(ids, vectors, fake_cosine, threshold=0.9)
    clusters_as_sets = [set(c) for c in clusters]
    assert {"a1", "a2", "a3"} in clusters_as_sets
    assert {"b1", "b2", "b3"} in clusters_as_sets
    assert len(clusters) == 2


def test_high_threshold_over_segments_rather_than_mixing_themes():
    # one item (0.5) sits between two tight groups — with a strict threshold
    # it must NOT drag them together (that's the whole point of "strict").
    vectors = [[0.0], [0.05], [0.5], [1.0], [0.95]]
    ids = ["a1", "a2", "mid", "b1", "b2"]
    clusters = cluster_reviews(ids, vectors, fake_cosine, threshold=0.85)
    for cluster in clusters:
        if "a1" in cluster:
            assert "b1" not in cluster and "b2" not in cluster
        if "b1" in cluster:
            assert "a1" not in cluster and "a2" not in cluster


def test_no_close_neighbor_stays_its_own_singleton_cluster():
    vectors = [[0.0], [0.02], [5.0]]  # third item is far from everything
    ids = ["a1", "a2", "loner"]
    clusters = cluster_reviews(ids, vectors, fake_cosine, threshold=0.9)
    singleton = [c for c in clusters if c == ["loner"]]
    assert singleton


def test_every_review_appears_in_exactly_one_cluster():
    vectors = [[0.0], [0.1], [0.5], [0.9], [1.0]]
    ids = [f"r{i}" for i in range(5)]
    clusters = cluster_reviews(ids, vectors, fake_cosine, threshold=0.7)
    flattened = [rid for c in clusters for rid in c]
    assert sorted(flattened) == sorted(ids)
    assert len(flattened) == len(set(flattened))


def test_complete_linkage_uses_worst_case_not_average():
    # merging {0,1} then comparing to 2: sim(0,2)=0.9 but sim(1,2)=0.3 ->
    # complete linkage must report the min (0.3), not the average (0.6).
    matrix = [
        [1.0, 0.95, 0.9],
        [0.95, 1.0, 0.3],
        [0.9, 0.3, 1.0],
    ]
    clusters = complete_linkage_cluster(3, matrix, threshold=0.5)
    # {0,1} merge (0.95 >= 0.5); merged-vs-2 similarity is min(0.9,0.3)=0.3 < 0.5 -> stays separate
    sets = [set(c) for c in clusters]
    assert {0, 1} in sets
    assert {2} in sets


def test_cluster_pair_similarity_is_the_minimum_cross_pair():
    matrix = [
        [1.0, 0.9, 0.4, 0.2],
        [0.9, 1.0, 0.5, 0.1],
        [0.4, 0.5, 1.0, 0.8],
        [0.2, 0.1, 0.8, 1.0],
    ]
    assert cluster_pair_similarity([0, 1], [2, 3], matrix) == 0.1


def test_similarity_matrix_diagonal_and_symmetry():
    vectors = [[0.0], [1.0], [0.5]]
    m = similarity_matrix(vectors, fake_cosine)
    assert all(m[i][i] == 1.0 for i in range(3))
    assert m[0][1] == m[1][0]


def test_empty_input_returns_no_clusters():
    assert cluster_reviews([], [], fake_cosine) == []


def test_single_review_is_its_own_cluster():
    clusters = cluster_reviews(["only"], [[0.5]], fake_cosine, threshold=0.9)
    assert clusters == [["only"]]
