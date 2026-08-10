"""Compare the LLM phrase-extraction against the log-odds statistical method
on the real labeled Nebula corpus.

    python -m eval.run_llm_keywords_eval
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import llm  # noqa: E402
from app.analysis import negative_keywords  # noqa: E402
from eval.llm_keywords import extract_keywords_llm  # noqa: E402
from eval.metrics import DATA_DIR, to_label  # noqa: E402


def main() -> None:
    labels = json.loads((DATA_DIR / "labels.json").read_text(encoding="utf-8"))
    pool = json.loads((DATA_DIR / "pool.json").read_text(encoding="utf-8"))

    # negative_keywords() needs the FULL review set (complaints + clean
    # reviews) to contrast against — passing only the pre-filtered complaint
    # corpus leaves it with an empty "rest" group and every z-score
    # collapses to exactly 0.0 (mathematically: with no contrast group,
    # target counts == pooled counts, so log_odds_target == log_odds_rest
    # for every term). The LLM extractor below has no such requirement — it
    # doesn't compute a statistical contrast, so it gets the complaint-only
    # corpus, matching what it's actually meant to read.
    full_reviews = []
    corpus = []
    for r in pool["items"]:
        label = labels.get(r["id"])
        if label is None:
            continue
        sentiment = to_label(label["score"])
        has_complaint = label.get("has_complaint", False)
        review = {"id": r["id"], "title": r["title"], "text": r["text"],
                 "sentiment": sentiment, "has_complaint": has_complaint}
        full_reviews.append(review)
        if sentiment == "negative" or has_complaint:
            corpus.append(review)

    print(f"Full set: {len(full_reviews)} reviews. Complaint corpus: {len(corpus)} reviews.\n")

    print("=== Statistical: log-odds z-score (app/analysis.py) ===")
    for k in negative_keywords(full_reviews, top_n=15, min_count=3):
        flag = "***" if k["significant"] else "   "
        print(f"  {flag}{k['term']:20s} count={k['count']:2d}  z={k['z_score']}")

    print("\n=== LLM phrase extraction (eval/llm_keywords.py) ===")
    try:
        phrases = asyncio.run(extract_keywords_llm(corpus, top_n=15))
    except llm.LLMError as exc:
        print(f"LLM call failed (quota/network): {exc}")
        return

    for p in phrases:
        flag = f"⚠ {p['hallucinated_citations_dropped']} dropped" if p["hallucinated_citations_dropped"] else ""
        print(f"  \"{p['phrase']}\" — {p['mention_count']} reviews {flag}")
        print(f"      e.g. \"{p['example_quote'][:100]}\"")

    out_path = DATA_DIR / f"llm_keywords_{pool['app_id']}.json"
    out_path.write_text(json.dumps(phrases, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
