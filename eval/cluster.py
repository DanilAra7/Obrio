"""Strict agglomerative clustering on a precomputed cosine-similarity matrix
— no numpy, no network, pure algorithm (this is what makes it unit-testable
without mocking an embeddings API).

Complete-linkage, deliberately: a cluster's similarity to another is the
MINIMUM similarity across every pair of members — the worst case, not the
average or the best case. That is what "strict" means here: two items only
end up in the same cluster if EVERY member on both sides agrees they belong
together, so one stray review with mixed vocabulary can't drag two distinct
themes into a single blob (the failure mode single-linkage is prone to via
chaining). The trade-off, taken deliberately per the design brief: this
over-segments into more, smaller clusters rather than under-segmenting into
fewer, impure ones — over-segmentation is the recoverable failure (a later
merge step can fix it), under-segmentation usually isn't.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple


def similarity_matrix(vectors: Sequence[Sequence[float]], cosine_fn) -> List[List[float]]:
    n = len(vectors)
    matrix = [[1.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            s = cosine_fn(vectors[i], vectors[j])
            matrix[i][j] = matrix[j][i] = s
    return matrix


def complete_linkage_cluster(n: int, matrix: List[List[float]], threshold: float) -> List[List[int]]:
    """Merge the most-similar pair of clusters repeatedly, stopping as soon as
    the best remaining pair falls below `threshold`. Returns a list of
    clusters, each a list of original indices (0..n-1). Every index appears
    in exactly one cluster, including size-1 clusters for items that never
    matched anything closely enough — that's intentional: an item with no
    close neighbor is real signal ("nothing else looks like this"), not a
    bug to paper over by force-joining it to the nearest cluster anyway.
    """
    if n == 0:
        return []
    sim = [row[:] for row in matrix]  # local mutable copy — merges update it in place
    members: Dict[int, List[int]] = {i: [i] for i in range(n)}
    active = list(range(n))

    while len(active) > 1:
        best: Tuple[float, int, int] = (-2.0, -1, -1)
        for a_idx in range(len(active)):
            for b_idx in range(a_idx + 1, len(active)):
                a, b = active[a_idx], active[b_idx]
                if sim[a][b] > best[0]:
                    best = (sim[a][b], a, b)
        score, a, b = best
        if score < threshold:
            break

        members[a].extend(members[b])
        del members[b]
        active.remove(b)
        for k in active:
            if k != a:
                # complete linkage: new similarity is the WORSE of the two —
                # merging can only ever tighten the bar for further merges.
                sim[a][k] = sim[k][a] = min(sim[a][k], sim[b][k])

    return [members[i] for i in active]


def cluster_reviews(review_ids: Sequence[str], vectors: Sequence[Sequence[float]],
                    cosine_fn, threshold: float = 0.68) -> List[List[str]]:
    """Convenience wrapper: ids + embedding vectors in, list of id-clusters out."""
    matrix = similarity_matrix(vectors, cosine_fn)
    index_clusters = complete_linkage_cluster(len(review_ids), matrix, threshold)
    return [[review_ids[i] for i in cluster] for cluster in index_clusters]


def cluster_pair_similarity(cluster_a: List[int], cluster_b: List[int], matrix: List[List[float]]) -> float:
    """Complete-linkage similarity between two (already-final) clusters —
    used to find merge *candidates* for the LLM judgment step: pairs whose
    similarity fell just short of the strict clustering threshold, not pairs
    that are obviously unrelated (no point spending an LLM call on those)."""
    return min(matrix[i][j] for i in cluster_a for j in cluster_b)
