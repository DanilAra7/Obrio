"""Score the labeled set with Mistral and compare against baselines + human labels.

    python -m eval.run_llm_eval
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.llm_sentiment import score_items  # noqa: E402
from eval.metrics import DATA_DIR, complaint_recall, evaluate, load_labeled_items, print_report  # noqa: E402
from eval.run_baseline import shipped_predict, text_only_predict  # noqa: E402


def main() -> None:
    labels_path = DATA_DIR / "labels.json"
    if not labels_path.exists():
        print(f"No labels yet at {labels_path} — label some reviews first (eval/label_app.py).")
        return

    items = load_labeled_items()
    if not items:
        print("labels.json exists but is empty.")
        return

    print(f"Scoring {len(items)} labeled reviews with Mistral (cached results are reused)...")
    llm_results = asyncio.run(score_items(items))

    errors = {k: v for k, v in llm_results.items() if "error" in v}
    if errors:
        print(f"{len(errors)}/{len(items)} reviews failed after retries — excluded from the LLM row.")

    scored_items = [it for it in items if "error" not in llm_results.get(it["id"], {"error": 1})]

    def llm_predict(it: dict) -> float:
        return llm_results[it["id"]]["score"]

    print_report(evaluate(items, shipped_predict, name="shipped (VADER+rating blend)"))
    print_report(evaluate(items, text_only_predict, name="text_only (VADER)"))
    if scored_items:
        print_report(evaluate(scored_items, llm_predict, name=f"mistral (n={len(scored_items)})"))

    # has_complaint: the shipped pipeline has no such signal, so the only proxy
    # available today is "did we classify it net-negative?" — this is exactly
    # the gap the user's requirement exposed (mixed reviews score net-positive
    # and would be silently dropped from the negative-analysis corpus).
    print("\n=== has_complaint recall (\"did we catch every review with dissatisfaction?\") ===")
    proxy = complaint_recall(items, lambda it: shipped_predict(it) <= -0.15)
    print(f"Proxy: shipped score <= -0.15   precision={proxy['precision']}  recall={proxy['recall']}  "
          f"f1={proxy['f1']}  (missed {proxy['fn']} of {proxy['support_true']} true complaints)")

    if scored_items:
        gem = complaint_recall(scored_items, lambda it: llm_results[it["id"]].get("has_complaint", False))
        print(f"Mistral native flag             precision={gem['precision']}  recall={gem['recall']}  "
              f"f1={gem['f1']}  (missed {gem['fn']} of {gem['support_true']} true complaints)")


if __name__ == "__main__":
    main()
