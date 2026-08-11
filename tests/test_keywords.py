"""Deep-dive tests for the statistical keyword algorithm's internals (stemmer
correctness, log-odds math on edge cases) — complements the integration-level
checks in test_pipeline.py (does negative_keywords() respect has_complaint,
does it show up correctly in build_insights())."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.analysis import negative_keywords, stem


def test_stem_merges_inflections_of_the_same_word():
    groups = [
        ["charge", "charged", "charging", "charges"],
        ["cancel", "cancelled", "cancelling", "cancels"],
        ["scam", "scams", "scammed", "scamming"],
        ["refund", "refunded", "refunds"],
        ["crash", "crashes", "crashed", "crashing"],
    ]
    for words in groups:
        stems = {stem(w) for w in words}
        assert len(stems) == 1, f"{words} produced different stems: {stems}"


def test_stem_does_not_mangle_short_or_double_s_words():
    for word in ["less", "was", "this", "happy", "buggy", "process", "address"]:
        assert stem(word) == word


def _review(text, complaint, rid=None):
    return {"id": rid or text[:8], "title": "", "text": text,
           "sentiment": "negative" if complaint else "positive", "has_complaint": complaint}


def test_log_odds_ranks_distinctive_terms_above_shared_vocabulary():
    target = [_review(t, True) for t in [
        "the app charged me twice after I cancelled my subscription",
        "they charged my card again, refund never came",
        "cancelled the subscription but still got charged",
    ]]
    rest = [_review(t, False) for t in [
        "the app is great, I use it every day",
        "nice app, the horoscope is fun to read every day",
        "great app for daily horoscope reading",
    ]]
    scores = negative_keywords(target + rest, min_count=2)
    terms = [r["term"] for r in scores]
    assert "charg" in terms[:3]
    assert "app" not in terms  # shared/common word should not rank as distinctive


def test_log_odds_variance_penalizes_low_count_terms():
    # "rare" appears 2/2 in target (small corpus) vs a common word appearing
    # in most of a much larger rest corpus — the rare term's z-score should
    # still be a finite number (not inf/NaN from a division-by-zero edge
    # case) despite the extreme count imbalance.
    target = [_review(t, True) for t in ["a rare weird glitch happened here", "another rare weird glitch"]]
    rest = [_review("fine app good app nice app", False, rid=f"r{i}") for i in range(20)]
    scores = negative_keywords(target + rest, min_count=2)
    assert all(_is_finite(r["z_score"]) for r in scores)


def _is_finite(x: float) -> bool:
    return x == x and abs(x) != float("inf")


def test_negative_keywords_ignores_has_complaint_when_key_is_entirely_absent():
    # a caller that never ran classify_sentiment/prepare (no has_complaint key
    # at all, not even False) should still get sensible output via the
    # sentiment=='negative' fallback in _is_complaint(), not crash on a
    # missing dict key.
    reviews = [
        {"title": "Love it", "text": "great app works well", "sentiment": "positive"},
        {"title": "Bad", "text": "crashes constantly, unusable crashes crashes", "sentiment": "negative"},
    ]
    result = negative_keywords(reviews, min_count=1)
    assert any(k["term"] == "crash" for k in result)


def test_negative_keywords_empty_when_no_complaints():
    reviews = [{"title": "Great", "text": "love it", "sentiment": "positive", "has_complaint": False}]
    assert negative_keywords(reviews) == []
