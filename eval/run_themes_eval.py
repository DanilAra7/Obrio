"""Run the full Trek C pipeline (discover -> assign -> recommend) on the real
labeled Nebula corpus and print a readable report.

    python -m eval.run_themes_eval
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.metrics import DATA_DIR, to_label  # noqa: E402
from eval.themes import run_pipeline  # noqa: E402


def main() -> None:
    labels = json.loads((DATA_DIR / "labels.json").read_text(encoding="utf-8"))
    pool = json.loads((DATA_DIR / "pool.json").read_text(encoding="utf-8"))

    corpus = []
    for r in pool["items"]:
        label = labels.get(r["id"])
        if label is None:
            continue
        sentiment = to_label(label["score"])
        has_complaint = label.get("has_complaint", False)
        if sentiment == "negative" or has_complaint:
            corpus.append({"id": r["id"], "title": r["title"], "text": r["text"]})

    print(f"Running theme discovery + assignment on {len(corpus)} reviews (negative ∪ mixed)...")
    result = asyncio.run(run_pipeline(pool["app_id"], corpus))

    print(f"\n=== Discovered taxonomy ({len(result['taxonomy'])} themes) ===")
    for t in result["taxonomy"]:
        print(f"  {t['name']}: {t['description']}")

    print(f"\n=== Themes ranked by coverage ===")
    for t in result["themes"]:
        flag = "✓" if t["citations_valid"] else "⚠ NO VALID CITATIONS"
        print(f"\n{t['theme']}  —  {t['review_count']} reviews ({t['share_of_corpus']}% of corpus)  {flag}")
        print(f"  {t['recommendation']}")
        print(f"  cited: {t['cited_review_ids']}")

    unassigned = sum(1 for ids in result["assignment"].values() if not ids)
    print(f"\n{unassigned} of {len(corpus)} reviews matched no theme.")

    out_path = DATA_DIR / f"themes_{pool['app_id']}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote full result -> {out_path}")


if __name__ == "__main__":
    main()
