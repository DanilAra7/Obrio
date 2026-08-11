"""Tests for the LLM-upgrade wiring in main.py — mocked, no network. Verifies
the graceful-degradation contract: an LLM outage or missing key must never
break an endpoint, only fall back to the deterministic regex/statistical/
VADER path — and that the keywords and themes upgrades are independent of
each other (one can succeed while the other fails).

Every test that sets a (fake) API key mocks BOTH keywords.llm_keywords and
themes.llm_theme_analysis, even when only one is under test: _apply_llm_upgrades
calls both once a key is present, so leaving either one unmocked means it
actually hits the network with a bogus key — slow, flaky, and exactly what
conftest.py's fast/free/deterministic contract exists to prevent. (An earlier
version of this file had that gap and a run silently ballooned from 0.1s to
20s from three real retried HTTP calls before failing.)

Uses asyncio.run() directly rather than pytest-asyncio, to avoid adding a
dependency for a handful of tests.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import keywords, llm, themes
from app.analysis import prepare
from app.main import _apply_llm_upgrades


def sample_reviews():
    return prepare([
        {"id": "1", "rating": 1, "title": "Scam", "text": "It charged me twice, the subscription is a rip-off"},
        {"id": "2", "rating": 1, "title": "Awful", "text": "Charged me after I cancelled the subscription"},
        {"id": "3", "rating": 5, "title": "Love it", "text": "Amazing app, accurate readings"},
    ])


async def _no_op_keywords(reviews):
    return []


async def _no_op_themes(reviews, app_id):
    return []


def _mock_key_and_defaults(monkeypatch):
    """Sets a fake key and stubs both upgrade paths to a harmless no-op, so a
    test can then override just the one it's exercising."""
    monkeypatch.setattr(llm, "api_key", lambda: "fake-key-for-test")
    monkeypatch.setattr(keywords, "llm_keywords", _no_op_keywords)
    monkeypatch.setattr(themes, "llm_theme_analysis", _no_op_themes)


# --------------------------------------------------------------------------- #
# no key at all -> both upgrades no-op
# --------------------------------------------------------------------------- #
def test_no_api_key_returns_fully_deterministic_output_unchanged(monkeypatch):
    monkeypatch.setattr(llm, "api_key", lambda: None)
    insights = asyncio.run(_apply_llm_upgrades(sample_reviews(), app_id=1))
    assert insights["themes_source"] == "regex"
    assert insights["keywords_source"] == "statistical"
    assert insights["themes"]  # regex path still found the billing theme


# --------------------------------------------------------------------------- #
# themes upgrade
# --------------------------------------------------------------------------- #
def test_theme_llm_failure_falls_back_to_regex_themes(monkeypatch):
    _mock_key_and_defaults(monkeypatch)

    async def boom(reviews, app_id):
        raise llm.LLMError("simulated outage")

    monkeypatch.setattr(themes, "llm_theme_analysis", boom)
    insights = asyncio.run(_apply_llm_upgrades(sample_reviews(), app_id=1))
    assert insights["themes_source"] == "regex"
    assert insights["themes"]


def test_theme_llm_success_swaps_in_llm_themes_and_regenerates_text(monkeypatch):
    _mock_key_and_defaults(monkeypatch)

    fake_themes = [{
        "theme": "Billing & Refunds", "description": "...", "negative_reviews": 2,
        "share_of_negative": 100.0, "avg_rating": 1.0, "recommendation": "Fix the refund flow.",
        "sample_quotes": ["It charged me twice..."], "cited_review_ids": ["1", "2"], "citations_valid": True,
    }]

    async def fake_pipeline(reviews, app_id):
        return fake_themes

    monkeypatch.setattr(themes, "llm_theme_analysis", fake_pipeline)
    insights = asyncio.run(_apply_llm_upgrades(sample_reviews(), app_id=1))

    assert insights["themes_source"] == "llm"
    assert insights["themes"] == fake_themes
    assert "Billing & Refunds" in insights["actionable_insights"][0]
    assert "Fix the refund flow." in insights["actionable_insights"][0]
    assert "Billing & Refunds" in insights["summary"]


def test_theme_llm_returning_no_themes_keeps_regex_fallback(monkeypatch):
    _mock_key_and_defaults(monkeypatch)
    insights = asyncio.run(_apply_llm_upgrades(sample_reviews(), app_id=1))  # both stubs already return []
    assert insights["themes_source"] == "regex"


# --------------------------------------------------------------------------- #
# keywords upgrade
# --------------------------------------------------------------------------- #
def test_keyword_llm_failure_falls_back_to_statistical(monkeypatch):
    _mock_key_and_defaults(monkeypatch)

    async def boom(reviews):
        raise llm.LLMError("simulated outage")

    monkeypatch.setattr(keywords, "llm_keywords", boom)
    insights = asyncio.run(_apply_llm_upgrades(sample_reviews(), app_id=1))
    assert insights["keywords_source"] == "statistical"


def test_keyword_llm_success_swaps_in_llm_phrases(monkeypatch):
    _mock_key_and_defaults(monkeypatch)

    fake_phrases = [{"term": "charged twice after cancelling", "count": 2, "share_of_corpus": 100.0,
                     "example_review_ids": ["1", "2"], "example_quote": "It charged me twice",
                     "hallucinated_citations_dropped": 0}]

    async def fake_extract(reviews):
        return fake_phrases

    monkeypatch.setattr(keywords, "llm_keywords", fake_extract)
    insights = asyncio.run(_apply_llm_upgrades(sample_reviews(), app_id=1))
    assert insights["keywords_source"] == "llm"
    assert insights["negative_keywords"] == fake_phrases


def test_keyword_llm_returning_empty_keeps_statistical_fallback(monkeypatch):
    _mock_key_and_defaults(monkeypatch)
    insights = asyncio.run(_apply_llm_upgrades(sample_reviews(), app_id=1))  # both stubs already return []
    assert insights["keywords_source"] == "statistical"


# --------------------------------------------------------------------------- #
# the two upgrades are independent
# --------------------------------------------------------------------------- #
def test_keyword_success_and_theme_failure_are_independent(monkeypatch):
    _mock_key_and_defaults(monkeypatch)

    async def fake_extract(reviews):
        return [{"term": "x", "count": 1, "share_of_corpus": 50.0, "example_review_ids": ["1"],
                 "example_quote": "x", "hallucinated_citations_dropped": 0}]

    async def theme_boom(reviews, app_id):
        raise llm.LLMError("themes down")

    monkeypatch.setattr(keywords, "llm_keywords", fake_extract)
    monkeypatch.setattr(themes, "llm_theme_analysis", theme_boom)
    insights = asyncio.run(_apply_llm_upgrades(sample_reviews(), app_id=1))

    assert insights["keywords_source"] == "llm"
    assert insights["themes_source"] == "regex"
