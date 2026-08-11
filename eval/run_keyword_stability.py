"""Bootstrap stability check for the statistical keyword method
(app/analysis.py's negative_keywords): resample the labeled corpus with
replacement N times and see how often each of today's top-N terms survives
in the top-N of the resampled result.

This is the practical companion to the z-score significance test — on
Nebula's real corpus, z_score alone says "zero terms individually clear
conventional significance" (see README), which is mathematically correct
but not the whole practical picture: bootstrap resampling shows the top 3
terms are still robust (>99% survival) even without individually clearing
that bar, while the tail (5th-15th place) is genuinely unstable (15-50%
survival) — a concrete signal for which results to trust versus treat as
"worth a human look", that a single z-score threshold can't distinguish.

    python -m eval.run_keyword_stability
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.analysis import negative_keywords  # noqa: E402
from eval.metrics import DATA_DIR, to_label  # noqa: E402


def bootstrap_stability(reviews: list, top_n: int = 10, n_resamples: int = 200, seed: int = 7) -> None:
    rng = random.Random(seed)
    today = [k["term"] for k in negative_keywords(reviews, top_n=top_n, min_count=3)]
    survival: Counter = Counter()
    for _ in range(n_resamples):
        sample = [reviews[rng.randrange(len(reviews))] for _ in range(len(reviews))]
        resampled_terms = {k["term"] for k in negative_keywords(sample, top_n=top_n, min_count=3)}
        for term in today:
            if term in resampled_terms:
                survival[term] += 1

    print(f"=== Bootstrap stability of today's top-{top_n} ({n_resamples} resamples) ===")
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
        reviews.append({"id": r["id"], "title": r["title"], "text": r["text"],
                        "sentiment": to_label(label["score"]),
                        "has_complaint": label.get("has_complaint", False)})

    bootstrap_stability(reviews)


if __name__ == "__main__":
    main()
