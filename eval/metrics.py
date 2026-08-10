"""Eval harness: compare a sentiment scorer's predictions against hand labels.

No scipy dependency (project stays lightweight) — Spearman is Pearson on ranks,
which is exactly what scipy.stats.spearmanr computes under the hood.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

DATA_DIR = Path(__file__).parent / "data"


# --------------------------------------------------------------------------- #
# data loading
# --------------------------------------------------------------------------- #
def load_labeled_items(pool_path: Path = DATA_DIR / "pool.json",
                       labels_path: Path = DATA_DIR / "labels.json") -> List[Dict]:
    """Join pool items with their human labels. Unlabeled items are skipped."""
    pool = json.loads(pool_path.read_text(encoding="utf-8"))
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    items = []
    for r in pool["items"]:
        label = labels.get(r["id"])
        if label is None:
            continue
        items.append({**r, "human_score": label["score"],
                      "has_complaint": label.get("has_complaint", False), "note": label.get("note", "")})
    return items


# --------------------------------------------------------------------------- #
# stats (no scipy)
# --------------------------------------------------------------------------- #
def _ranks(values: Sequence[float]) -> List[float]:
    """Average ranks, ties get the mean of the ranks they span (standard method)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2 + 1  # 1-indexed
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def pearson(a: Sequence[float], b: Sequence[float]) -> float:
    n = len(a)
    if n < 2:
        return 0.0
    mean_a, mean_b = sum(a) / n, sum(b) / n
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    var_a = sum((x - mean_a) ** 2 for x in a)
    var_b = sum((y - mean_b) ** 2 for y in b)
    denom = (var_a * var_b) ** 0.5
    return cov / denom if denom else 0.0


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    return pearson(_ranks(a), _ranks(b))


def mae(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a) if a else 0.0


def rmse(a: Sequence[float], b: Sequence[float]) -> float:
    return (sum((x - y) ** 2 for x, y in zip(a, b)) / len(a)) ** 0.5 if a else 0.0


# --------------------------------------------------------------------------- #
# 3-class projection + macro F1
# --------------------------------------------------------------------------- #
def to_label(score: float, pos_thresh: float = 0.15, neg_thresh: float = -0.15) -> str:
    if score >= pos_thresh:
        return "positive"
    if score <= neg_thresh:
        return "negative"
    return "neutral"


def complaint_recall(items: List[Dict], predict_flag: Callable[[Dict], bool]) -> Dict:
    """Precision/recall for the has_complaint flag. Recall is what matters most
    here — the whole point of the flag is to not miss dissatisfaction, so a
    false positive (flagging a clean review) is far cheaper than a false
    negative (a real complaint never reaching the keyword/theme pipeline)."""
    truth = [bool(it.get("has_complaint", False)) for it in items]
    pred = [bool(predict_flag(it)) for it in items]
    tp = sum(1 for t, p in zip(truth, pred) if t and p)
    fp = sum(1 for t, p in zip(truth, pred) if not t and p)
    fn = sum(1 for t, p in zip(truth, pred) if t and not p)
    tn = sum(1 for t, p in zip(truth, pred) if not t and not p)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn, "support_true": tp + fn}


def macro_f1(true_labels: Sequence[str], pred_labels: Sequence[str],
            classes: Sequence[str] = ("positive", "neutral", "negative")) -> Tuple[float, Dict[str, Dict]]:
    per_class = {}
    for c in classes:
        tp = sum(1 for t, p in zip(true_labels, pred_labels) if t == c and p == c)
        fp = sum(1 for t, p in zip(true_labels, pred_labels) if t != c and p == c)
        fn = sum(1 for t, p in zip(true_labels, pred_labels) if t == c and p != c)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[c] = {"precision": round(precision, 3), "recall": round(recall, 3),
                        "f1": round(f1, 3), "support": tp + fn}
    return round(sum(v["f1"] for v in per_class.values()) / len(classes), 3), per_class


# --------------------------------------------------------------------------- #
# main entry point
# --------------------------------------------------------------------------- #
def evaluate(items: List[Dict], predict: Callable[[Dict], float], name: str = "model") -> Dict:
    """`predict` takes one labeled item (id/title/text/rating/human_score) and
    returns a float in -1..1. Returns overall metrics plus breakdowns."""
    human = [it["human_score"] for it in items]
    pred = [max(-1.0, min(1.0, predict(it))) for it in items]

    true_labels = [to_label(s) for s in human]
    pred_labels = [to_label(s) for s in pred]
    f1, per_class = macro_f1(true_labels, pred_labels)

    result = {
        "name": name,
        "n": len(items),
        "mae": round(mae(human, pred), 4),
        "rmse": round(rmse(human, pred), 4),
        "spearman": round(spearman(human, pred), 4),
        "macro_f1": f1,
        "per_class": per_class,
        "by_rating": _breakdown(items, human, pred, key=lambda it: it["rating"]),
        "by_length": _breakdown(items, human, pred, key=lambda it: _length_bucket(it["text"])),
    }
    return result


def _length_bucket(text: str) -> str:
    n = len(text)
    if n < 40:
        return "short (<40 chars)"
    if n < 150:
        return "medium (40-150)"
    return "long (150+)"


def _breakdown(items: List[Dict], human: List[float], pred: List[float], key) -> Dict:
    groups: Dict = {}
    for it, h, p in zip(items, human, pred):
        groups.setdefault(key(it), {"human": [], "pred": []})
        groups[key(it)]["human"].append(h)
        groups[key(it)]["pred"].append(p)
    return {
        str(k): {"n": len(v["human"]), "mae": round(mae(v["human"], v["pred"]), 4),
                 "spearman": round(spearman(v["human"], v["pred"]), 4) if len(v["human"]) > 1 else None}
        for k, v in sorted(groups.items(), key=lambda kv: str(kv[0]))
    }


def print_report(result: Dict) -> None:
    print(f"\n=== {result['name']} (n={result['n']}) ===")
    print(f"MAE {result['mae']}  RMSE {result['rmse']}  Spearman {result['spearman']}  macro-F1 {result['macro_f1']}")
    print("Per class:")
    for c, v in result["per_class"].items():
        print(f"  {c:9s} precision {v['precision']:.3f}  recall {v['recall']:.3f}  f1 {v['f1']:.3f}  (n={v['support']})")
    print("By star rating:")
    for k, v in result["by_rating"].items():
        print(f"  {k}★  n={v['n']:3d}  MAE={v['mae']}  Spearman={v['spearman']}")
    print("By review length:")
    for k, v in result["by_length"].items():
        print(f"  {k:16s} n={v['n']:3d}  MAE={v['mae']}  Spearman={v['spearman']}")
