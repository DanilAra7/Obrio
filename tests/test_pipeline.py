"""Unit tests for parsing + analysis (no network access needed)."""

import asyncio

import pytest
from fastapi.testclient import TestClient

from app import itunes, store
from app.analysis import (build_insights, calculate_metrics, classify_sentiment,
                          clean_text, negative_keywords, prepare, tokenize)
from app.main import app as fastapi_app


def entry(rating="5", title="Great", content="I love it", rid="1"):
    return {
        "im:rating": {"label": rating},
        "title": {"label": title},
        "content": {"label": content, "attributes": {"type": "text"}},
        "id": {"label": rid},
        "author": {"name": {"label": "tester"}},
        "im:version": {"label": "1.0.0"},
        "updated": {"label": "2026-08-08T17:48:01-07:00"},
        "im:voteSum": {"label": "3"},
    }


# --------------------------- parsing --------------------------------------- #
def test_parse_entry_extracts_key_fields():
    review = itunes.parse_entry(entry(rating="4", title="Nice", content="Works well"))
    assert review["rating"] == 4
    assert review["title"] == "Nice"
    assert review["text"] == "Works well"
    assert review["votes"] == 3


@pytest.mark.parametrize("bad", [
    {},                                            # the app-description entry has no rating
    {"im:rating": {"label": "not-a-number"}},      # malformed rating
    {"im:rating": {"label": "9"}},                 # out of range
    "garbage",                                     # wrong type entirely
])
def test_parse_entry_rejects_non_reviews(bad):
    assert itunes.parse_entry(bad) is None


def test_parse_feed_handles_missing_and_single_entry():
    assert itunes.parse_feed({}) == []
    assert itunes.parse_feed({"feed": {}}) == []
    assert len(itunes.parse_feed({"feed": {"entry": entry()}})) == 1
    assert len(itunes.parse_feed({"feed": {"entry": [entry(), {}, entry(rid="2")]}})) == 2


@pytest.mark.parametrize("limit", [0, 501])
def test_fetch_reviews_validates_limit(limit):
    with pytest.raises(ValueError):
        asyncio.run(itunes.fetch_reviews(1, limit=limit))


# --------------------------- preprocessing --------------------------------- #
def test_clean_text_strips_markup_urls_and_whitespace():
    assert clean_text("Great   &amp; <b>fast</b>\n\napp http://x.io/y") == "Great & fast app"
    assert clean_text("") == ""


def test_tokenize_drops_stopwords_and_short_tokens():
    assert tokenize("The app is really SLOW and it crashes") == ["slow", "crashes"]


# --------------------------- sentiment ------------------------------------- #
@pytest.mark.parametrize("text,rating,expected", [
    ("Absolutely love this, it is amazing", 5, "positive"),
    ("Crashes constantly and stole my money, terrible", 1, "negative"),
    ("", 1, "negative"),          # empty text falls back to the rating
    ("", 5, "positive"),
    ("It is okay", 3, "neutral"),
])
def test_classify_sentiment(text, rating, expected):
    assert classify_sentiment(text, rating)[0] == expected


def test_rating_pulls_lukewarm_praise_down():
    # VADER alone would call this positive; the 2-star rating must temper it.
    assert classify_sentiment("Nice idea", 2)[0] != "positive"


# --------------------------- metrics & insights ---------------------------- #
def sample():
    return prepare([
        {"id": "1", "rating": 5, "title": "Love it", "text": "Amazing app, accurate readings"},
        {"id": "2", "rating": 5, "title": "Great", "text": "Very helpful and fun"},
        {"id": "3", "rating": 1, "title": "Scam", "text": "It charged me twice, the subscription is a rip-off"},
        {"id": "4", "rating": 1, "title": "Awful", "text": "Charged me after I cancelled the subscription, no refund"},
    ])


def test_calculate_metrics():
    m = calculate_metrics(sample())
    assert m["total_reviews"] == 4
    assert m["average_rating"] == 3.0
    assert m["rating_distribution"]["5_star"] == {"count": 2, "percentage": 50.0}
    assert m["rating_distribution"]["3_star"]["count"] == 0
    assert m["positive_share_4_5"] == 50.0


def test_calculate_metrics_on_empty_input_does_not_crash():
    assert calculate_metrics([])["average_rating"] == 0.0


def test_negative_keywords_prefer_terms_distinctive_of_negatives():
    terms = [k["term"] for k in negative_keywords(sample(), min_count=2)]
    assert "subscription" in terms
    assert "charg" in terms  # stemmed form of charge/charged/charging/charges


def test_negative_keywords_corpus_includes_mixed_not_just_pure_negative():
    # review 2 is net-positive (sentiment="positive") but has_complaint=True —
    # negative_keywords must still pull it into the target corpus rather than
    # treating "not sentiment=='negative'" as "nothing to see here". This is
    # the has_complaint field set directly (bypassing classify_sentiment's own
    # heuristic, which is a separate, already-documented approximation) so the
    # test isolates negative_keywords()'s corpus-selection logic specifically.
    reviews = [
        {"id": "1", "title": "Great", "text": "amazing app love it best purchase ever",
         "sentiment": "positive", "has_complaint": False},
        {"id": "2", "title": "Great but", "text": "love the app but they charged my card twice billing support",
         "sentiment": "positive", "has_complaint": True},
    ]
    terms = [k["term"] for k in negative_keywords(reviews, min_count=1)]
    assert "charg" in terms


def test_build_insights_surfaces_billing_theme_and_actions():
    insights = build_insights(sample())
    assert insights["sentiment"]["counts"]["negative"] == 2
    assert insights["themes"][0]["theme"] == "Billing & subscriptions"
    assert insights["actionable_insights"]


def test_insights_on_all_positive_sample_have_no_themes():
    insights = build_insights(prepare([{"id": "1", "rating": 5, "title": "Love", "text": "Perfect app"}]))
    assert insights["themes"] == []
    assert insights["negative_keywords"] == []


# --------------------------- API ------------------------------------------- #
def test_api_endpoints_use_the_cached_batch(monkeypatch):
    store.clear()
    store.save(123, "us", {"app_id": 123, "name": "Demo", "developer": "Dev",
                           "store_rating": 4.5, "store_rating_count": 10}, sample())
    client = TestClient(fastapi_app)

    metrics = client.get("/apps/123/metrics").json()
    assert metrics["metrics"]["average_rating"] == 3.0

    insights = client.get("/apps/123/insights").json()
    assert insights["sentiment"]["counts"]["negative"] == 2

    csv_response = client.get("/apps/123/reviews?format=csv")
    assert csv_response.status_code == 200
    assert "attachment" in csv_response.headers["content-disposition"]
    assert csv_response.text.splitlines()[0].startswith("id,rating,title,text")

    assert "Rating distribution" in client.get("/apps/123/report").text
    store.clear()


def test_collect_rejects_request_without_app_identifier():
    client = TestClient(fastapi_app)
    assert client.post("/reviews/collect", json={"country": "us"}).status_code == 422


def test_collect_maps_upstream_failure_to_404(monkeypatch):
    async def boom(*args, **kwargs):
        raise itunes.AppNotFoundError("No app with id 1")

    monkeypatch.setattr(itunes, "lookup_app", boom)
    client = TestClient(fastapi_app, raise_server_exceptions=False)
    response = client.post("/reviews/collect", json={"app_id": 1, "country": "us"})
    assert response.status_code == 404
