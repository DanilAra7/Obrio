"""Run the embeddings -> strict-cluster -> LLM-merge -> name pipeline on the
real labeled Nebula corpus and compare against the one-shot discovery result
already saved in eval/data/themes_1459969523.json.

    python -m eval.run_cluster_eval
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import llm  # noqa: E402
from eval.cluster_llm import discover_clusters, run  # noqa: E402
from eval.metrics import DATA_DIR, to_label  # noqa: E402


def main() -> None:
    labels = json.loads((DATA_DIR / "labels.json").read_text(encoding="utf-8"))
    pool = json.loads((DATA_DIR / "pool.json").read_text(encoding="utf-8"))

    corpus = []
    for r in pool["items"]:
        label = labels.get(r["id"])
        if label is None:
            continue
        sentiment = to_label(label["score"])
        if sentiment == "negative" or label.get("has_complaint", False):
            corpus.append({"id": r["id"], "title": r["title"], "text": r["text"]})

    print(f"Running cluster pipeline on {len(corpus)} reviews...")
    try:
        result = asyncio.run(run(corpus))
    except llm.LLMError as exc:
        print(f"\nLLM step failed (quota/network): {exc}")
        print("Falling back to embeddings+clustering only (no merge, no naming) so the "
             "clustering quality itself can still be inspected:")
        clusters, _matrix, _ids = asyncio.run(discover_clusters(corpus))
        clusters.sort(key=len, reverse=True)
        for c in clusters[:8]:
            print(f"\n--- raw cluster, size={len(c)} ---")
            for r in c[:4]:
                print(f"  [{r['title'][:35]:35s}] {(r['text'] or '')[:70]}")
        return

    print(f"\nRaw clusters (strict, before merge): {result['raw_cluster_count']}")
    print(f"After LLM merge: {result['merged_cluster_count']}")
    print(f"Named themes (size>=2): {len(result['themes'])}")
    print(f"Unnamed singletons: {result['unnamed_singletons']}")

    covered = sum(t["negative_reviews"] for t in result["themes"])
    print(f"\nCoverage: {covered}/{result['corpus_size']} reviews in a named theme "
         f"({result['unnamed_singletons']} singletons + "
         f"{result['corpus_size'] - covered - result['unnamed_singletons']} unaccounted)")

    print("\n=== Themes ===")
    for t in result["themes"]:
        flag = "✓" if t["citations_valid"] else "⚠ NO VALID CITATIONS"
        print(f"\n{t['theme']} — {t['negative_reviews']} reviews ({t['share_of_negative']}%)  {flag}")
        print(f"  {t['description']}")
        print(f"  → {t['recommendation']}")
        print(f"  cited: {t['cited_review_ids']}")

    out_path = DATA_DIR / f"clusters_{pool['app_id']}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")

    # side-by-side with the one-shot discovery pipeline, if available
    oneshot_path = DATA_DIR / f"themes_{pool['app_id']}.json"
    if oneshot_path.exists():
        oneshot = json.loads(oneshot_path.read_text(encoding="utf-8"))
        print(f"\n=== For comparison: one-shot discovery (eval/themes.py) found "
             f"{len(oneshot['themes'])} themes ===")
        for t in oneshot["themes"]:
            print(f"  {t['theme']} — {t['review_count']} reviews")
        oneshot_unassigned = sum(1 for ids in oneshot["assignment"].values() if not ids)
        print(f"  ({oneshot_unassigned} of {len(oneshot['assignment'])} reviews matched no theme)")


if __name__ == "__main__":
    main()
