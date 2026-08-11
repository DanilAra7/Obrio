"""Compare the production LLM phrase-extraction (app/keywords.py) against the
log-odds statistical method (app/analysis.py) on the real labeled Nebula
corpus — both are exactly what /apps/{id}/insights runs.

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
from app.keywords import llm_keywords  # noqa: E402
from eval.metrics import DATA_DIR, to_label  # noqa: E402


def main() -> None:
    labels = json.loads((DATA_DIR / "labels.json").read_text(encoding="utf-8"))
    pool = json.loads((DATA_DIR / "pool.json").read_text(encoding="utf-8"))

    # Both functions expect the FULL batch (complaints + clean reviews) and
    # filter internally: negative_keywords() needs the "rest" group to
    # contrast against (an empty rest collapses every z-score to exactly
    # 0.0), and llm_keywords() filters to the same negative∪has_complaint
    # corpus so the two methods are compared on identical input.
    reviews = []
    complaint_count = 0
    for r in pool["items"]:
        label = labels.get(r["id"])
        if label is None:
            continue
        sentiment = to_label(label["score"])
        has_complaint = label.get("has_complaint", False)
        reviews.append({"id": r["id"], "title": r["title"], "text": r["text"],
                        "sentiment": sentiment, "has_complaint": has_complaint})
        if sentiment == "negative" or has_complaint:
            complaint_count += 1

    print(f"Full set: {len(reviews)} reviews. Complaint corpus: {complaint_count} reviews.\n")

    print("=== Statistical: log-odds z-score (app/analysis.py) ===")
    for k in negative_keywords(reviews, top_n=15, min_count=3):
        flag = "***" if k["significant"] else "   "
        print(f"  {flag}{k['term']:20s} count={k['count']:2d}  z={k['z_score']}")

    print("\n=== LLM phrase extraction (app/keywords.py) ===")
    try:
        phrases = asyncio.run(llm_keywords(reviews, top_n=15))
    except llm.LLMError as exc:
        print(f"LLM call failed (quota/network): {exc}")
        return

    for p in phrases:
        flag = f"⚠ {p['hallucinated_citations_dropped']} dropped" if p["hallucinated_citations_dropped"] else ""
        print(f"  \"{p['term']}\" — {p['count']} reviews {flag}")
        print(f"      e.g. \"{p['example_quote'][:100]}\"")

    out_path = DATA_DIR / f"llm_keywords_{pool['app_id']}.json"
    out_path.write_text(json.dumps(phrases, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
