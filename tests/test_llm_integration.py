"""Tests for the LLM-upgrade wiring in main.py — mocked, no network. Verifies
the graceful-degradation contract: an LLM outage or missing key must never
break an endpoint, only fall back to the deterministic regex/VADER path.

Uses asyncio.run() directly rather than pytest-asyncio, to avoid adding a
dependency for four tests.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import llm, themes
from app.analysis import prepare
from app.main import _insights_with_llm_themes


def sample_reviews():
    return prepare([
        {"id": "1", "rating": 1, "title": "Scam", "text": "It charged me twice, the subscription is a rip-off"},
        {"id": "2", "rating": 1, "title": "Awful", "text": "Charged me after I cancelled the subscription"},
        {"id": "3", "rating": 5, "title": "Love it", "text": "Amazing app, accurate readings"},
    ])


def test_no_api_key_returns_regex_fallback_unchanged(monkeypatch):
    monkeypatch.setattr(llm, "api_key", lambda: None)
    insights = asyncio.run(_insights_with_llm_themes(sample_reviews(), app_id=1))
    assert insights["themes_source"] == "regex"
    assert insights["themes"]  # regex path still found the billing theme


def test_llm_failure_falls_back_to_regex_themes(monkeypatch):
    monkeypatch.setattr(llm, "api_key", lambda: "fake-key-for-test")

    async def boom(reviews, app_id):
        raise llm.LLMError("simulated outage")

    monkeypatch.setattr(themes, "llm_theme_analysis", boom)
    insights = asyncio.run(_insights_with_llm_themes(sample_reviews(), app_id=1))
    assert insights["themes_source"] == "regex"
    assert insights["themes"]


def test_llm_success_swaps_in_llm_themes_and_regenerates_text(monkeypatch):
    monkeypatch.setattr(llm, "api_key", lambda: "fake-key-for-test")

    fake_themes = [{
        "theme": "Billing & Refunds", "description": "...", "negative_reviews": 2,
        "share_of_negative": 100.0, "avg_rating": 1.0, "recommendation": "Fix the refund flow.",
        "sample_quotes": ["It charged me twice..."], "cited_review_ids": ["1", "2"], "citations_valid": True,
    }]

    async def fake_pipeline(reviews, app_id):
        return fake_themes

    monkeypatch.setattr(themes, "llm_theme_analysis", fake_pipeline)
    insights = asyncio.run(_insights_with_llm_themes(sample_reviews(), app_id=1))

    assert insights["themes_source"] == "llm"
    assert insights["themes"] == fake_themes
    assert "Billing & Refunds" in insights["actionable_insights"][0]
    assert "Fix the refund flow." in insights["actionable_insights"][0]
    assert "Billing & Refunds" in insights["summary"]


def test_llm_returning_no_themes_keeps_regex_fallback(monkeypatch):
    monkeypatch.setattr(llm, "api_key", lambda: "fake-key-for-test")

    async def empty_pipeline(reviews, app_id):
        return []

    monkeypatch.setattr(themes, "llm_theme_analysis", empty_pipeline)
    insights = asyncio.run(_insights_with_llm_themes(sample_reviews(), app_id=1))
    assert insights["themes_source"] == "regex"
