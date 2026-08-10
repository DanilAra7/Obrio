"""Compare the old ad-hoc keyword score against the new log-odds method on
the real labeled Nebula corpus, using the gold has_complaint split.

    python -m eval.run_keywords_eval
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.analysis import negative_keywords  # noqa: E402
from eval.keywords import extract_negative_keywords  # noqa: E402
from eval.metrics import DATA_DIR, to_label  # noqa: E402


def bootstrap_stability(reviews: list, top_n: int = 10, n_resamples: int = 200, seed: int = 7) -> None:
    """How often does each of today's top-N terms survive if we resample the
    same population with replacement? Low survival = the ranking is mostly
    sample noise, not a stable signal — the intuitive companion to the
    z-score significance check above."""
    rng = random.Random(seed)
    today = [k["term"] for k in extract_negative_keywords(reviews, top_n=top_n, min_count=3)]
    survival = Counter()
    for _ in range(n_resamples):
        sample = [reviews[rng.randrange(len(reviews))] for _ in range(len(reviews))]
        resampled_terms = {k["term"] for k in extract_negative_keywords(sample, top_n=top_n, min_count=3)}
        for term in today:
            if term in resampled_terms:
                survival[term] += 1

    print(f"\n=== Bootstrap stability of today's top-{top_n} ({n_resamples} resamples) ===")
    for term in today:
        pct = 100 * survival[term] / n_resamples
        print(f"  {term:16s} stays in top-{top_n} in {pct:5.1f}% of resamples")


def main() -> None:
    labels = json.loads((DATA_DIR / "labels.json").read_text(encoding="utf-8"))
    pool = json.loads((DATA_DIR / "pool.json").read_text(encoding="utf-8"))

    reviews = []
    for r in pool["items"]:
        label = labels.get(r["id"])
        if label is None:
            continue
        reviews.append({
            "title": r["title"], "text": r["text"],
            "sentiment": to_label(label["score"]),
            "has_complaint": label.get("has_complaint", False),
        })

    corpus_size = sum(1 for r in reviews if r["sentiment"] == "negative" or r["has_complaint"])
    print(f"Corpus: {len(reviews)} reviews, {corpus_size} in the negative-analysis corpus "
          f"(negative ∪ mixed)\n")

    print("=== OLD: count * log(1+lift), corpus = sentiment=='negative' only ===")
    old_reviews = [{**r, "sentiment": "negative" if (r["sentiment"] == "negative") else "positive"}
                   for r in reviews]  # old code has no mixed concept
    for k in negative_keywords(old_reviews, top_n=15, min_count=2):
        print(f"  {k['term']:20s} count={k['count']:2d}  share={k['share_of_negative']:5.1f}%  "
              f"lift=x{k['lift_vs_rest']}")

    print("\n=== NEW: log-odds z-score + Dirichlet prior, corpus = negative ∪ mixed ===")
    new_results = extract_negative_keywords(reviews, top_n=15, min_count=3)
    n_sig = sum(1 for k in new_results if k["significant"])
    for k in new_results:
        flag = "***" if k["significant"] else ("*  " if abs(k["z_score"]) >= 1.5 else "   ")
        print(f"  {flag}{k['term']:20s} count={k['count']:2d}  share={k['share_of_corpus']:5.1f}%  "
              f"z={k['z_score']}")
    print(f"\n{n_sig} of {len(new_results)} terms clear |z|>=1.96 (conventional p<0.05).")

    bootstrap_stability(reviews)


if __name__ == "__main__":
    main()
