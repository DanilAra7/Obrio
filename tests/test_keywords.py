import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.keywords import extract_negative_keywords, log_odds_scores, stem


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


def test_log_odds_ranks_distinctive_terms_above_shared_vocabulary():
    target = [
        "the app charged me twice after I cancelled my subscription",
        "they charged my card again, refund never came",
        "cancelled the subscription but still got charged",
    ]
    rest = [
        "the app is great, I use it every day",
        "nice app, the horoscope is fun to read every day",
        "great app for daily horoscope reading",
    ]
    scores = log_odds_scores(target, rest, min_count=2)
    terms = [r["term"] for r in scores]
    assert terms[0] in {"charg", "cancel", "subscript"} or "charg" in terms[:3]
    assert "app" not in terms  # shared/common word should not rank as distinctive


def test_log_odds_variance_penalizes_low_count_terms():
    # "rare" appears 2/2 in target (small corpus) vs a common word appearing
    # in most of a much larger rest corpus — the rare word's z-score should
    # still reflect real uncertainty (finite, not runaway) despite 100% share.
    target = ["a rare weird glitch happened here", "another rare weird glitch"]
    rest = ["fine app good app nice app"] * 20
    scores = log_odds_scores(target, rest, min_count=2)
    assert all(math_is_finite(r["z_score"]) for r in scores)


def math_is_finite(x: float) -> bool:
    return x == x and abs(x) != float("inf")


def test_extract_negative_keywords_uses_has_complaint_when_present():
    reviews = [
        {"title": "Love it", "text": "great app, works well", "sentiment": "positive", "has_complaint": False},
        {"title": "Mostly good", "text": "great app but the billing charged me twice",
         "sentiment": "positive", "has_complaint": True},  # net-positive, still a complaint
        {"title": "Bad", "text": "billing charged me for nothing, terrible", "sentiment": "negative",
         "has_complaint": True},
    ]
    keywords = extract_negative_keywords(reviews, min_count=2)
    terms = [k["term"] for k in keywords]
    assert any("bill" in t or "charg" in t for t in terms), terms


def test_extract_negative_keywords_falls_back_to_sentiment_without_has_complaint():
    reviews = [
        {"title": "Love it", "text": "great app works well", "sentiment": "positive"},
        {"title": "Bad", "text": "crashes constantly, unusable crashes crashes", "sentiment": "negative"},
    ]
    keywords = extract_negative_keywords(reviews, min_count=1)
    assert any(k["term"] == "crash" for k in keywords)


def test_extract_negative_keywords_empty_when_no_complaints():
    reviews = [{"title": "Great", "text": "love it", "sentiment": "positive", "has_complaint": False}]
    assert extract_negative_keywords(reviews) == []
