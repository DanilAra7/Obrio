"""Grid-search the three constants that were hand-picked in app/analysis.py:
the text/rating blend weight and the two class thresholds.

This is the entirety of the "tuning" in this project — no gradients, no
training. Just: try every combination, keep the one that scores best on the
labeled data, and report how much it beat the hand-picked values by.

To avoid reporting a number that only holds on the data it was tuned on, the
sweep runs under k-fold cross-validation: weights are chosen on k-1 folds and
scored on the held-out fold, so the reported figure is out-of-sample.

    python -m eval.calibrate
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.metrics import DATA_DIR, load_labeled_items, macro_f1, mae, spearman, to_label  # noqa: E402
from eval.run_baseline import text_only_predict  # noqa: E402

WEIGHTS = [round(w * 0.05, 2) for w in range(21)]          # 0.00 .. 1.00 text weight
THRESHOLDS = [round(-1.0 + i * 0.05, 2) for i in range(41)]  # -1.00 .. 1.00


def blended(vader: float, rating: int, w_text: float) -> float:
    rating_score = (rating - 3) / 2
    return w_text * vader + (1 - w_text) * rating_score


def score_config(items: List[Dict], vaders: List[float], w_text: float,
                 neg_t: float, pos_t: float) -> float:
    truth = [to_label(it["human_score"]) for it in items]
    pred = [to_label(blended(v, it["rating"], w_text), pos_t, neg_t)
            for it, v in zip(items, vaders)]
    return macro_f1(truth, pred)[0]


def best_config(items: List[Dict], vaders: List[float]) -> Tuple[float, float, float, float]:
    best = (-1.0, 0.6, -0.15, 0.15)
    for w in WEIGHTS:
        for neg_t in THRESHOLDS:
            for pos_t in THRESHOLDS:
                if pos_t <= neg_t:
                    continue
                f1 = score_config(items, vaders, w, neg_t, pos_t)
                if f1 > best[0]:
                    best = (f1, w, neg_t, pos_t)
    return best


def cross_validated(items: List[Dict], vaders: List[float], k: int = 5) -> float:
    """Out-of-sample macro-F1: pick the config on k-1 folds, score on the rest."""
    folds = [list(range(i, len(items), k)) for i in range(k)]
    scores = []
    for held_out in folds:
        train_idx = [i for i in range(len(items)) if i not in set(held_out)]
        train = [items[i] for i in train_idx]
        train_v = [vaders[i] for i in train_idx]
        _, w, neg_t, pos_t = best_config(train, train_v)

        test = [items[i] for i in held_out]
        test_v = [vaders[i] for i in held_out]
        scores.append(score_config(test, test_v, w, neg_t, pos_t))
    return sum(scores) / len(scores)


def main() -> None:
    if not (DATA_DIR / "labels.json").exists():
        print("No labels yet — nothing to calibrate against.")
        return

    items = load_labeled_items()
    vaders = [text_only_predict(it) for it in items]
    truth = [to_label(it["human_score"]) for it in items]

    shipped_f1 = score_config(items, vaders, 0.6, -0.15, 0.15)
    print(f"Hand-picked config   w_text=0.60  thresholds=(-0.15, +0.15)  macro-F1 = {shipped_f1:.3f}")

    f1, w, neg_t, pos_t = best_config(items, vaders)
    print(f"Best in-sample       w_text={w:.2f}  thresholds=({neg_t:+.2f}, {pos_t:+.2f})  macro-F1 = {f1:.3f}")

    cv_f1 = cross_validated(items, vaders)
    print(f"Cross-validated (5-fold, out-of-sample)                     macro-F1 = {cv_f1:.3f}")
    print(f"\nGain over hand-picked constants: {cv_f1 - shipped_f1:+.3f} macro-F1 (out-of-sample)")

    pred_scores = [blended(v, it["rating"], w) for it, v in zip(items, vaders)]
    pred_labels = [to_label(s, pos_t, neg_t) for s in pred_scores]
    _, per_class = macro_f1(truth, pred_labels)
    print("\nPer class with the tuned config:")
    for c, v in per_class.items():
        print(f"  {c:9s} precision {v['precision']:.3f}  recall {v['recall']:.3f}  f1 {v['f1']:.3f}  (n={v['support']})")

    human = [it["human_score"] for it in items]
    print(f"\nMAE {mae(human, pred_scores):.4f}   Spearman {spearman(human, pred_scores):.4f}")
    print("\nNote: Spearman barely moves when tuning thresholds — thresholds change *where the "
          "cut points are*, not the ranking. A high Spearman with a low F1 is the signature of "
          "a scorer that orders reviews well but is mapped to labels badly.")


if __name__ == "__main__":
    main()
