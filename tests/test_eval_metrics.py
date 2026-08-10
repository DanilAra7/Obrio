"""Sanity checks for eval/metrics.py against known values (no scipy on hand to
cross-check against, so these are hand-computed)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

from eval.metrics import complaint_recall, evaluate, load_labeled_items, macro_f1, mae, spearman, to_label


def test_mae_and_spearman_perfect_agreement():
    a = [-1.0, -0.5, 0.0, 0.5, 1.0]
    assert mae(a, a) == 0.0
    assert spearman(a, a) == 1.0


def test_spearman_perfect_disagreement():
    a = [-1.0, -0.5, 0.0, 0.5, 1.0]
    b = [1.0, 0.5, 0.0, -0.5, -1.0]
    assert spearman(a, b) == -1.0


def test_spearman_handles_ties():
    # constant predictions -> zero variance -> defined as 0, not a crash
    assert spearman([0.1, 0.2, 0.3], [0.5, 0.5, 0.5]) == 0.0


def test_to_label_thresholds():
    assert to_label(0.2) == "positive"
    assert to_label(-0.2) == "negative"
    assert to_label(0.0) == "neutral"
    assert to_label(0.15) == "positive"   # boundary is inclusive
    assert to_label(-0.15) == "negative"


def test_macro_f1_perfect_predictions():
    labels = ["positive", "negative", "neutral", "positive"]
    f1, per_class = macro_f1(labels, labels)
    assert f1 == 1.0
    assert all(v["f1"] == 1.0 for v in per_class.values() if v["support"] > 0)


def test_macro_f1_all_wrong():
    true = ["positive", "positive"]
    pred = ["negative", "negative"]
    f1, _ = macro_f1(true, pred)
    assert f1 < 0.5


def test_complaint_recall_penalizes_missed_complaints_not_false_positives():
    # 3 true complaints; predictor catches 2, plus 1 false alarm on a clean review.
    items = [
        {"has_complaint": True}, {"has_complaint": True}, {"has_complaint": True},
        {"has_complaint": False}, {"has_complaint": False},
    ]
    caught = iter([True, True, False, True, False])  # 2 hits, 1 miss, 1 false positive
    result = complaint_recall(items, lambda it: next(caught))
    assert result["tp"] == 2 and result["fn"] == 1 and result["fp"] == 1
    assert result["recall"] == round(2 / 3, 3)


def test_complaint_recall_perfect():
    items = [{"has_complaint": True}, {"has_complaint": False}]
    result = complaint_recall(items, lambda it: it["has_complaint"])
    assert result["precision"] == 1.0 and result["recall"] == 1.0


def test_load_labeled_items_propagates_has_complaint(tmp_path):
    # Regression test: load_labeled_items used to drop has_complaint from the
    # merged item, silently zeroing out complaint_recall (support_true=0).
    pool = {"items": [{"id": "1", "title": "t", "text": "x", "rating": 3}]}
    labels = {"1": {"score": 0.6, "has_complaint": True, "note": ""}}
    pool_path = tmp_path / "pool.json"
    labels_path = tmp_path / "labels.json"
    pool_path.write_text(json.dumps(pool))
    labels_path.write_text(json.dumps(labels))

    items = load_labeled_items(pool_path, labels_path)
    assert items[0]["has_complaint"] is True
    assert items[0]["human_score"] == 0.6


def test_evaluate_end_to_end_with_perfect_predictor():
    items = [
        {"id": "1", "rating": 5, "text": "great", "human_score": 1.0},
        {"id": "2", "rating": 1, "text": "bad", "human_score": -1.0},
        {"id": "3", "rating": 3, "text": "ok", "human_score": 0.0},
    ]
    result = evaluate(items, predict=lambda it: it["human_score"], name="oracle")
    assert result["mae"] == 0.0
    assert result["macro_f1"] == 1.0
    assert result["by_rating"]["1"]["n"] == 1
