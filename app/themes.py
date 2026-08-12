"""LLM theme discovery — the MISTRAL_API_KEY-enabled upgrade over
analysis.py's hardcoded 9-regex theme_analysis().

Pipeline: embed every review -> cluster STRICTLY -> LLM-judge whether close
clusters should merge -> name + describe + recommend each survivor, citing
specific review ids.

This replaced an earlier one-shot version ("read a sample of ≤60 reviews,
ask the LLM to invent 5-9 themes from it") after a head-to-head run on
Nebula's real 101-review complaint corpus: the one-shot version only ever
saw a sample, so 7 of 101 reviews matched no theme at all — nothing had
generated a category their specific complaint fit. This pipeline embeds
every review before any LLM judgment happens, so a rare-but-real pattern
still gets its own cluster instead of being sampled out. Same run: 99 of
101 reviews covered, 0 hallucinated citations across 9 merged themes (vs
94/101 and 8 themes for the one-shot version) — see eval/data/clusters_*.json
and eval/data/themes_*.json for the raw comparison data, and
eval/run_cluster_eval.py to reproduce it.

Stages:
  1. embed every review in the complaint corpus (app/embeddings.py) and
     cluster STRICTLY (app/cluster.py, complete-linkage, high threshold) —
     deliberately over-segments; merging bad splits is stage 2's job. The
     threshold is calibrated against a live similarity distribution, not
     guessed (see CLUSTER_THRESHOLD below).
  2. for cluster pairs whose similarity falls in the candidate band (close,
     but not close enough to have auto-merged), ask the LLM a narrow
     same-theme-or-different question and union-merge on "same" — a much
     easier question to get right than "invent categories from nothing"
  3. name + describe + recommend each surviving cluster, citing specific
     review ids. Citations are checked programmatically against the real
     cluster membership before being trusted — a hallucinated id is dropped
     and the theme is flagged `citations_valid: false` rather than shipping
     a fabricated citation unnoticed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import httpx

from . import llm
from .cluster import cluster_pair_similarity, cluster_reviews, similarity_matrix
from .embeddings import cosine, embed_reviews

CLUSTER_THRESHOLD = 0.81     # complete-linkage merge bar — calibrated live against Nebula's real
                             # mistral-embed similarity distribution (median pairwise sim was 0.81,
                             # so anything near that is "not obviously related"; see
                             # eval/run_cluster_eval.py). Re-calibrate if the embedding model changes —
                             # this number is specific to mistral-embed's similarity range, not portable.
CANDIDATE_LOW = 0.77         # below this, two clusters are almost certainly unrelated — not worth
                             # spending an LLM call to confirm the obvious. Narrower band than a
                             # first pass used (0.70): at 0.70 nearly 70% of all cluster pairs
                             # qualified as "candidates" on mistral-embed's similarity distribution
                             # (denser/less discriminative than gemini-embedding-001's was), and
                             # with that many pairs even a modest false-positive rate on individual
                             # merge judgments chained transitively through union-find into one
                             # 90-review mega-cluster on a live run. See MAX_THEME_SHARE below for
                             # the second, structural safeguard against the same failure mode.
MERGE_BATCH_SIZE = 10        # candidate pairs per LLM call
SAMPLE_PER_GROUP_MERGE = 4   # reviews shown per side when judging a merge
SAMPLE_PER_GROUP_NAME = 10   # reviews shown per cluster when naming/recommending
MIN_THEME_SIZE = 2           # clusters smaller than this aren't worth naming ("fix what?" for n=1)
MAX_THEME_SHARE = 0.35       # a merge that would push one theme past this share of the whole
                             # complaint corpus is refused outright — a math-level circuit breaker
                             # so an overly agreeable LLM merge judgment can't cascade into one
                             # blob theme no matter how many individual "same theme" calls it makes


# --------------------------------------------------------------------------- #
# shared: citation verification (also used by app/keywords.py)
# --------------------------------------------------------------------------- #
def verify_citations(recommendations: List[Dict], valid_ids_by_group: Dict[str, set]) -> List[Dict]:
    """Any cited id not actually in that group's review set is dropped;
    citations_valid=False means zero real citations survived — the caller
    treats that as "unverified", never as "trust it anyway"."""
    out = []
    for rec in recommendations:
        group = rec["theme"]
        valid_ids = valid_ids_by_group.get(group, set())
        cited = [rid for rid in rec.get("cited_review_ids", []) if rid in valid_ids]
        dropped = len(rec.get("cited_review_ids", [])) - len(cited)
        out.append({"theme": group, "recommendation": rec["recommendation"], "cited_review_ids": cited,
                    "citations_valid": len(cited) > 0, "hallucinated_citations_dropped": dropped})
    return out


# --------------------------------------------------------------------------- #
# stage 1: embed + strict cluster
# --------------------------------------------------------------------------- #
async def discover_clusters(reviews: List[Dict], threshold: float = CLUSTER_THRESHOLD
                            ) -> Tuple[List[List[Dict]], List[List[float]], List[str]]:
    """Returns (clusters as lists of review dicts, full similarity matrix, id order) —
    the matrix and id order let stage 2 find candidate pairs without re-embedding."""
    by_id = {r["id"]: r for r in reviews}
    embeddings = await embed_reviews(reviews)
    ids = list(embeddings.keys())
    vectors = [embeddings[i] for i in ids]

    matrix = similarity_matrix(vectors, cosine)
    id_clusters = cluster_reviews(ids, vectors, cosine, threshold=threshold)
    review_clusters = [[by_id[rid] for rid in cluster] for cluster in id_clusters]
    return review_clusters, matrix, ids


# --------------------------------------------------------------------------- #
# stage 2: LLM merge judgment
# --------------------------------------------------------------------------- #
MERGE_SYSTEM = """You are given pairs of App Store review groups. Each group was formed by
semantic similarity, not by you. For each pair, decide whether the two groups describe the SAME
SPECIFIC complaint mechanism (should be merged) or merely the same broad category (should stay
separate). Judge the mechanism, not the topic label: "charged again right after cancelling" and
"billed for a lifetime plan after a $1 trial" are both nominally "billing," but are DIFFERENT
mechanisms and must stay separate. Only merge when a reader would describe both groups with the
same one-sentence complaint, e.g. two groups both specifically about being charged again despite
already cancelling, regardless of tone (one panicked "SCAM!!!", one matter-of-fact). When in
doubt, keep them separate — false negatives here just mean two adjacent themes in the report
instead of one; false positives blur distinct problems together and lose the specific one an
engineer would need to act on."""

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
    "same theme" — except a union that would push one group past
    MAX_THEME_SHARE of the whole corpus is refused regardless of what the
    LLM said. This is a deliberate, math-level check on top of the model's
    judgment, not just a fallback for when the LLM is unavailable: a live
    run surfaced exactly the failure it guards against — a merge-happy LLM
    response chained transitively through union-find (A merges with B, B
    with C, so A and C end up together even though no one judged A and C
    directly) into a single cluster covering 90 of 101 reviews. If the LLM
    is unavailable outright, this function returns the clusters unmerged
    (over-segmented) rather than failing — an extra, slightly redundant
    theme in the output is a far cheaper failure than crashing the endpoint.
    """
    pairs = _candidate_pairs(clusters, matrix, ids)
    if not pairs:
        return clusters

    corpus_size = sum(len(c) for c in clusters)
    max_size = max(1, int(MAX_THEME_SHARE * corpus_size))
    uf = _UnionFind(len(clusters))
    sizes = [len(c) for c in clusters]  # sizes[root] is only meaningful once root == uf.find(root)

    async with httpx.AsyncClient() as client:
        for batch_start in range(0, len(pairs), MERGE_BATCH_SIZE):
            batch = pairs[batch_start:batch_start + MERGE_BATCH_SIZE]
            payload = []
            for k, (i, j, sim) in enumerate(batch):
                payload.append({
                    "pair_id": str(k),
                    "group_a": [f"{r.get('title','')}: {r.get('text','')}"[:200]
                               for r in clusters[i][:SAMPLE_PER_GROUP_MERGE]],
                    "group_b": [f"{r.get('title','')}: {r.get('text','')}"[:200]
                               for r in clusters[j][:SAMPLE_PER_GROUP_MERGE]],
                })
            try:
                response = await llm.call(MERGE_SYSTEM, payload, MERGE_SCHEMA, client=client)
            except llm.LLMError:
                continue  # leave this batch's pairs unmerged rather than fail the whole pipeline
            decisions = {d["pair_id"]: d["same_theme"] for d in response.get("decisions", [])}
            for k, (i, j, sim) in enumerate(batch):
                if not decisions.get(str(k)):
                    continue
                ra, rb = uf.find(i), uf.find(j)
                if ra == rb:
                    continue
                if sizes[ra] + sizes[rb] > max_size:
                    continue  # refuse: would blow past MAX_THEME_SHARE regardless of the LLM's call
                uf.union(ra, rb)
                sizes[uf.find(ra)] = sizes[ra] + sizes[rb]

    groups: Dict[int, List[Dict]] = {}
    for i, cluster in enumerate(clusters):
        root = uf.find(i)
        groups.setdefault(root, []).extend(cluster)
    return list(groups.values())


# --------------------------------------------------------------------------- #
# stage 3: name + describe + recommend, cited and verified
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


async def name_and_recommend(clusters: List[List[Dict]], min_size: int = MIN_THEME_SIZE) -> List[Dict]:
    """Singleton clusters (below min_size) are dropped — "what should we fix"
    isn't a meaningful question for a single review. Output shape matches the
    regex fallback's theme_analysis() (theme/negative_reviews/share_of_negative/
    avg_rating/recommendation/sample_quotes) so callers don't need two code
    paths, plus cited_review_ids/citations_valid for the anti-hallucination
    signal the regex path has no equivalent of."""
    named_targets = [(str(i), c) for i, c in enumerate(clusters) if len(c) >= min_size]
    if not named_targets:
        return []
    # Denominator for share_of_negative is the WHOLE complaint corpus (every
    # cluster, including singletons too small to name) — matching the regex
    # fallback's theme_analysis(), so the two are comparable percentages.
    corpus_size = sum(len(c) for c in clusters)

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
    reviews_by_cluster = {cid: c for cid, c in named_targets}
    by_id = {cid: {r["id"]: r for r in c} for cid, c in named_targets}

    out = []
    for t in response["themes"]:
        cid = t["cluster_id"]
        v = verified_by_id.get(cid, {})
        cited = v.get("cited_review_ids", [])
        revs = reviews_by_cluster.get(cid, [])
        quotes = [by_id[cid][rid]["text"] or by_id[cid][rid]["title"] for rid in cited[:2] if rid in by_id.get(cid, {})]
        out.append({
            "theme": t["name"],
            "description": t["description"],
            "negative_reviews": len(revs),
            "share_of_negative": round(100 * len(revs) / corpus_size, 2) if corpus_size else 0.0,
            "avg_rating": round(sum(r["rating"] for r in revs) / len(revs), 2) if revs else 0.0,
            "recommendation": t["recommendation"],
            "sample_quotes": [q[:280] for q in quotes],
            "cited_review_ids": cited,
            "citations_valid": v.get("citations_valid", False),
        })
    out.sort(key=lambda t: -t["negative_reviews"])
    return out


# --------------------------------------------------------------------------- #
# entry point used by main.py
# --------------------------------------------------------------------------- #
async def llm_theme_analysis(reviews: List[Dict], app_id: Any, max_quotes: int = 2) -> List[Dict]:
    """reviews: the full collected batch (negative ∪ has_complaint filtering
    happens here, same corpus definition as the regex fallback). Raises
    llm.LLMError on failure — callers must catch it and fall back to
    analysis.theme_analysis(), never let a broken LLM/embedding call take
    down the endpoint. `app_id` is accepted for interface parity with the
    regex fallback's call site in main.py; clustering re-derives themes from
    the full corpus every call rather than pinning a taxonomy to disk, so it
    isn't otherwise used — unlike the old one-shot version, there's no
    separately-invented category list that needs to stay stable across runs.
    """
    corpus = [r for r in reviews if r.get("sentiment") == "negative" or r.get("has_complaint")]
    if not corpus:
        return []

    clusters, matrix, ids = await discover_clusters(corpus)
    merged = await merge_clusters(clusters, matrix, ids)
    return await name_and_recommend(merged)
