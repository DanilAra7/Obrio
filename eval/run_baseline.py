"""Score the labeled set with the pipeline(s) currently in app/analysis.py and
report how they compare to the human -1..1 labels.

Two variants are measured, because they answer different questions:
  * "shipped"   — classify_sentiment(text, rating): what the API returns today.
  * "text_only" — VADER on the text alone, no rating blended in.
Since the human labels are blind to the star rating (see eval/label_app.py),
"text_only" is the fair apples-to-apples comparison; "shipped" tells us how
much blending in the rating pulls predictions away from a blind text read.

    python -m eval.run_baseline
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.analysis import _ANALYZER, classify_sentiment  # noqa: E402
from eval.metrics import evaluate, print_report  # noqa: E402
from eval.metrics import DATA_DIR, load_labeled_items  # noqa: E402


def shipped_predict(item: dict) -> float:
    text = f"{item.get('title', '')}. {item.get('text', '')}".strip(". ")
    return classify_sentiment(text, item["rating"])[1]


def text_only_predict(item: dict) -> float:
    text = f"{item.get('title', '')}. {item.get('text', '')}".strip(". ")
    if len(text) < 3:
        return 0.0
    return _ANALYZER.polarity_scores(text)["compound"]


def main() -> None:
    labels_path = DATA_DIR / "labels.json"
    if not labels_path.exists():
        print(f"No labels yet at {labels_path} — label some reviews first (eval/label_app.py).")
        return

    items = load_labeled_items()
    if not items:
        print("labels.json exists but is empty.")
        return

    for name, fn in [("shipped (VADER+rating blend)", shipped_predict), ("text_only (VADER)", text_only_predict)]:
        print_report(evaluate(items, fn, name=name))


if __name__ == "__main__":
    main()
