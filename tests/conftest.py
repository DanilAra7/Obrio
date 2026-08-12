"""Unit tests must stay fast, free and deterministic — they exercise the
offline VADER/regex fallback path, not the live Mistral integration (that's
what eval/run_llm_eval.py, eval/run_cluster_eval.py and
eval/run_llm_keywords_eval.py are for, run deliberately against real data).
Without this fixture, a developer's local .env with a real MISTRAL_API_KEY
would make every test that touches build_insights()/insights endpoints
silently start making network calls — slow, flaky, and burning quota on
every `pytest` run.
"""

import pytest

from app import llm


@pytest.fixture(autouse=True)
def no_llm_api_key(monkeypatch):
    monkeypatch.setattr(llm, "api_key", lambda: None)
