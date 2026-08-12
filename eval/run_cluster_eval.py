"""Live smoke-test of the production theme pipeline (app/themes.py — embed,
cluster, LLM-merge, name+recommend) against the real labeled Nebula corpus.

This is what /apps/{id}/insights actually runs when MISTRAL_API_KEY is set;
running it here separately makes it easy to inspect the raw output and
re-verify the numbers quoted in the README (99/101 coverage, 0 hallucinated
citations) without spinning up the API.

    python -m eval.run_cluster_eval
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import llm  # noqa: E402
from app.themes import discover_clusters, llm_theme_analysis  # noqa: E402
from eval.metrics import DATA_DIR, to_label  # noqa: E402


def main() -> None:
    labels = json.loads((DATA_DIR / "labels.json").read_text(encoding="utf-8"))
    pool = json.loads((DATA_DIR / "pool.json").read_text(encoding="utf-8"))

    reviews = []
    corpus_size = 0
    for r in pool["items"]:
        label = labels.get(r["id"])
        if label is None:
            continue
        review = {"id": r["id"], "title": r["title"], "text": r["text"],
                  "rating": r["rating"], "sentiment": to_label(label["score"]),
                  "has_complaint": label.get("has_complaint", False)}
        reviews.append(review)
        if review["sentiment"] == "negative" or review["has_complaint"]:
            corpus_size += 1

    print(f"Running the production theme pipeline (app.themes.llm_theme_analysis) "
         f"on {corpus_size} complaint reviews out of {len(reviews)}...")
    try:
        themes_out = asyncio.run(llm_theme_analysis(reviews, app_id=pool["app_id"]))
    except llm.LLMError as exc:
        print(f"\nLLM step failed (quota/network): {exc}")
        print("Falling back to embeddings+clustering only (no merge, no naming) so the "
             "clustering quality itself can still be inspected:")
        corpus = [r for r in reviews if r["sentiment"] == "negative" or r["has_complaint"]]
        clusters, _matrix, _ids = asyncio.run(discover_clusters(corpus))
        clusters.sort(key=len, reverse=True)
        for c in clusters[:8]:
            print(f"\n--- raw cluster, size={len(c)} ---")
            for r in c[:4]:
                print(f"  [{r['title'][:35]:35s}] {(r['text'] or '')[:70]}")
        return

    covered = sum(t["negative_reviews"] for t in themes_out)
    print(f"\n{len(themes_out)} themes, {covered}/{corpus_size} complaint reviews covered, "
         f"{sum(1 for t in themes_out if not t['citations_valid'])} with unverified citations")

    for t in themes_out:
        flag = "✓" if t["citations_valid"] else "⚠ NO VALID CITATIONS"
        print(f"\n{t['theme']} — {t['negative_reviews']} reviews ({t['share_of_negative']}%)  {flag}")
        print(f"  {t['description']}")
        print(f"  → {t['recommendation']}")
        print(f"  cited: {t['cited_review_ids']}")

    out_path = DATA_DIR / f"clusters_{pool['app_id']}.json"
    out_path.write_text(json.dumps(themes_out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
