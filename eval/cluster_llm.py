"""Embeddings -> strict clustering -> LLM merge judgment -> naming +
cited recommendations. The alternative theme-discovery pipeline the user
asked for, instead of eval/themes.py's one-shot "read a sample, invent
5-9 themes" approach.

Why this exists alongside eval/themes.py's discovery step: that approach
only ever looks at a SAMPLE (<=60) of the complaint corpus and asks the LLM
to invent categories from it in one creative pass — on Nebula's 101-review
corpus that left 7 reviews matched to no theme at all, because whatever
generated the taxonomy never saw an example of their specific complaint.
This pipeline processes every review (via its embedding) before any LLM
judgment happens, so a rare-but-real complaint pattern gets its own tight
cluster instead of being sampled out. The LLM is only asked two much
narrower questions — "are these two groups the same complaint?" and "name
this group" — instead of "invent categories from nothing", which is both
easier to get right and easier to audit.

Pipeline:
  1. embed every review in the complaint corpus (eval/embeddings.py)
  2. cluster STRICTLY (eval/cluster.py, complete-linkage, high threshold) —
     deliberately over-segments; merging bad splits is step 3's job
  3. for cluster pairs whose similarity falls in the candidate band (close,
     but not close enough to have auto-merged), ask the LLM a narrow
     same-theme-or-different question and union-merge on "same"
  4. name + describe + recommend each surviving cluster, citing specific
     review ids — citations are verified against real cluster membership
     exactly like eval/themes.py's write_recommendations (verify_citations
     is imported from there rather than re-implemented)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import llm  # noqa: E402
from eval.cluster import cluster_pair_similarity, cluster_reviews, similarity_matrix  # noqa: E402
from eval.embeddings import cosine, embed_reviews  # noqa: E402
from eval.themes import verify_citations  # noqa: E402

CLUSTER_THRESHOLD = 0.68     # complete-linkage merge bar — calibrated live against Nebula's real
                             # similarity distribution (median pairwise sim was 0.67, so anything
                             # near that is "not obviously related"; see eval/run_cluster_eval.py)
CANDIDATE_LOW = 0.55         # below this, two clusters are almost certainly unrelated — not worth
                             # spending an LLM call to confirm the obvious
MERGE_BATCH_SIZE = 10        # candidate pairs per LLM call
SAMPLE_PER_GROUP_MERGE = 4   # reviews shown per side when judging a merge
SAMPLE_PER_GROUP_NAME = 10   # reviews shown per cluster when naming/recommending


# --------------------------------------------------------------------------- #
# stage 1-2: embed + strict cluster
# --------------------------------------------------------------------------- #
async def discover_clusters(reviews: List[Dict], threshold: float = CLUSTER_THRESHOLD
                            ) -> Tuple[List[List[Dict]], List[List[float]], List[str]]:
    """Returns (clusters as lists of review dicts, full similarity matrix, id order) —
    the matrix and id order are needed by the merge step to find candidate pairs
    without re-embedding anything."""
    by_id = {r["id"]: r for r in reviews}
    embeddings = await embed_reviews(reviews)
    ids = list(embeddings.keys())
    vectors = [embeddings[i] for i in ids]

    matrix = similarity_matrix(vectors, cosine)
    id_clusters = cluster_reviews(ids, vectors, cosine, threshold=threshold)
    review_clusters = [[by_id[rid] for rid in cluster] for cluster in id_clusters]
    return review_clusters, matrix, ids


# --------------------------------------------------------------------------- #
# stage 3: LLM merge judgment
# --------------------------------------------------------------------------- #
MERGE_SYSTEM = """You are given pairs of App Store review groups. Each group was formed by
semantic similarity, not by you. For each pair, decide whether the two groups represent the
SAME underlying complaint/theme (should be merged into one) or genuinely DIFFERENT complaints
(should stay separate). Judge by what the reviews are actually unhappy about, not surface
wording — e.g. two groups both about unexpected billing charges are the same theme even if one
says "scam" and the other says "charged twice"."""

MERGE_SCHEMA = {
    "type": "object",
    "properties": {
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "pair_id": {"type": "string"},
                    "same_theme": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["pair_id", "same_theme", "reason"],
            },
        }
    },
    "required": ["decisions"],
}


def _candidate_pairs(clusters: List[List[Dict]], matrix: List[List[float]], ids: List[str],
                     low: float = CANDIDATE_LOW, high: float = CLUSTER_THRESHOLD) -> List[Tuple[int, int, float]]:
    idx_of = {rid: i for i, rid in enumerate(ids)}
    cluster_idx = [[idx_of[r["id"]] for r in c] for c in clusters]
    pairs = []
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            sim = cluster_pair_similarity(cluster_idx[i], cluster_idx[j], matrix)
            if low <= sim < high:
                pairs.append((i, j, sim))
    pairs.sort(key=lambda p: -p[2])  # judge the most-plausible merges first
    return pairs


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


async def merge_clusters(clusters: List[List[Dict]], matrix: List[List[float]], ids: List[str]
                         ) -> List[List[Dict]]:
    """Judges every candidate pair, then union-merges the ones the LLM calls
    "same theme". If the LLM is unavailable, returns the clusters unmerged
    (over-segmented) rather than failing — an extra, slightly redundant
    theme in the output is a far cheaper failure than crashing the endpoint.
    """
    pairs = _candidate_pairs(clusters, matrix, ids)
    if not pairs:
        return clusters

    uf = _UnionFind(len(clusters))
    for batch_start in range(0, len(pairs), MERGE_BATCH_SIZE):
        batch = pairs[batch_start:batch_start + MERGE_BATCH_SIZE]
        payload = []
        for k, (i, j, sim) in enumerate(batch):
            payload.append({
                "pair_id": str(k),
                "group_a": [f"{r.get('title','')}: {r.get('text','')}"[:200] for r in clusters[i][:SAMPLE_PER_GROUP_MERGE]],
                "group_b": [f"{r.get('title','')}: {r.get('text','')}"[:200] for r in clusters[j][:SAMPLE_PER_GROUP_MERGE]],
            })
        try:
            response = await llm.call(MERGE_SYSTEM, payload, MERGE_SCHEMA)
        except llm.LLMError:
            continue  # leave this batch's pairs unmerged rather than fail the whole pipeline
        decisions = {d["pair_id"]: d["same_theme"] for d in response.get("decisions", [])}
        for k, (i, j, sim) in enumerate(batch):
            if decisions.get(str(k)):
                uf.union(i, j)

    groups: Dict[int, List[Dict]] = {}
    for i, cluster in enumerate(clusters):
        root = uf.find(i)
        groups.setdefault(root, []).extend(cluster)
    return list(groups.values())


# --------------------------------------------------------------------------- #
# stage 4: name + describe + recommend, cited and verified
# --------------------------------------------------------------------------- #
NAME_SYSTEM = """Each numbered group below is a cluster of App Store reviews sharing a complaint
theme (grouped by semantic similarity, not by you — your job is only to describe and act on
what's already grouped). For each group: give a short theme name (2-4 words), a one-sentence
description, ONE actionable product recommendation, and cite 2-4 specific review ids from that
exact group as evidence. Do not merge or split groups — describe each one as given."""

NAME_SCHEMA = {
    "type": "object",
    "properties": {
        "themes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "cluster_id": {"type": "string"},
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "recommendation": {"type": "string"},
                    "cited_review_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["cluster_id", "name", "description", "recommendation", "cited_review_ids"],
            },
        }
    },
    "required": ["themes"],
}


async def name_and_recommend(clusters: List[List[Dict]], min_size: int = 2) -> List[Dict]:
    """Singleton clusters (min_size not met) are dropped from the LLM call —
    "what should we fix" isn't a meaningful question for a single review —
    but are still reported by the caller as an unclustered long tail if
    desired; this function focuses on clusters worth naming."""
    named_targets = [(str(i), c) for i, c in enumerate(clusters) if len(c) >= min_size]
    if not named_targets:
        return []

    payload = {
        cid: [{"id": r["id"], "title": r.get("title", ""), "text": r.get("text", "")}
             for r in c[:SAMPLE_PER_GROUP_NAME]]
        for cid, c in named_targets
    }
    response = await llm.call(NAME_SYSTEM, payload, NAME_SCHEMA)

    valid_ids_by_cluster = {cid: {r["id"] for r in c} for cid, c in named_targets}
    verified = verify_citations(
        [{"theme": t["cluster_id"], "recommendation": t["recommendation"],
          "cited_review_ids": t["cited_review_ids"]} for t in response["themes"]],
        valid_ids_by_cluster,
    )
    verified_by_id = {v["theme"]: v for v in verified}
    by_id = {cid: {r["id"]: r for r in c} for cid, c in named_targets}
    sizes = {cid: len(c) for cid, c in named_targets}

    out = []
    for t in response["themes"]:
        cid = t["cluster_id"]
        v = verified_by_id.get(cid, {})
        cited = v.get("cited_review_ids", [])
        quotes = [by_id[cid][rid]["text"] or by_id[cid][rid]["title"] for rid in cited[:2] if rid in by_id.get(cid, {})]
        out.append({
            "theme": t["name"],
            "description": t["description"],
            "negative_reviews": sizes.get(cid, 0),
            "recommendation": t["recommendation"],
            "sample_quotes": [q[:280] for q in quotes],
            "cited_review_ids": cited,
            "citations_valid": v.get("citations_valid", False),
        })
    out.sort(key=lambda t: -t["negative_reviews"])
    return out


# --------------------------------------------------------------------------- #
# full pipeline
# --------------------------------------------------------------------------- #
async def run(reviews: List[Dict], threshold: float = CLUSTER_THRESHOLD) -> Dict:
    clusters, matrix, ids = await discover_clusters(reviews, threshold=threshold)
    merged = await merge_clusters(clusters, matrix, ids)
    themes = await name_and_recommend(merged)
    corpus_size = len(reviews)
    for t in themes:
        t["share_of_negative"] = round(100 * t["negative_reviews"] / corpus_size, 2) if corpus_size else 0.0
    unclustered = sum(1 for c in merged if len(c) < 2)
    return {"themes": themes, "raw_cluster_count": len(clusters), "merged_cluster_count": len(merged),
           "unnamed_singletons": unclustered, "corpus_size": corpus_size}
