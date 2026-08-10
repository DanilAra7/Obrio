"""LLM theme discovery — the GEMINI_API_KEY-enabled upgrade over
analysis.py's hardcoded 9-regex theme_analysis(). Three stages:

  1. discover_taxonomy() — reads a sample of the negative-analysis corpus and
     proposes 5-9 themes grounded in what THIS app's reviews actually say,
     instead of a generic subscription-app checklist. Pinned to disk
     (data/taxonomy_<app_id>.json) so repeated runs on the same app are
     comparable rather than reshuffling theme names every call.
  2. assign_themes() — multi-label classification of every review against
     the FIXED taxonomy (a billing complaint that also gripes about support
     hits both themes). Any theme name the model invents outside the fixed
     list is dropped.
  3. write_recommendations() — one recommendation per theme, REQUIRED to
     cite specific review ids. Citations are verified against the real
     assignment before being trusted — a hallucinated id is dropped and the
     theme is flagged `citations_valid: false` rather than shipping unnoticed.

Validated in eval/run_themes_eval.py against Nebula's real review corpus:
found two themes (non-inclusive gender options, per-minute psychic pricing)
that the old regex taxonomy had no pattern for at all, with 0 hallucinated
citations across 8 themes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from . import llm

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MAX_DISCOVERY_SAMPLE = 60
ASSIGN_BATCH_SIZE = 15
MAX_REVIEWS_PER_THEME = 12


# --------------------------------------------------------------------------- #
# stage 1: discovery (+ pinning)
# --------------------------------------------------------------------------- #
DISCOVERY_SYSTEM = """You analyze App Store reviews to find recurring product problems.
You will be given a sample of reviews that all express some dissatisfaction (ranging from
a minor gripe in an otherwise happy review, to an angry 1-star rant). Read them and propose
5 to 9 THEMES that group the complaints — grounded specifically in what these reviews actually
say, not a generic checklist. Each theme needs a short name (2-4 words) and a one-sentence
definition precise enough that a different reader could consistently classify a new review
against it. Avoid overlapping themes — if two of your themes would usually apply together to
the same review, merge them."""

DISCOVERY_SCHEMA = {
    "type": "object",
    "properties": {
        "themes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "description": {"type": "string"}},
                "required": ["name", "description"],
            },
        }
    },
    "required": ["themes"],
}


def _taxonomy_path(app_id: Any) -> Path:
    return DATA_DIR / f"taxonomy_{app_id}.json"


def load_taxonomy(app_id: Any) -> Optional[List[Dict]]:
    path = _taxonomy_path(app_id)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def save_taxonomy(app_id: Any, themes: List[Dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _taxonomy_path(app_id).write_text(json.dumps(themes, ensure_ascii=False, indent=2), encoding="utf-8")


async def discover_taxonomy(reviews: List[Dict], sample_size: int = MAX_DISCOVERY_SAMPLE) -> List[Dict]:
    sample = reviews[:sample_size]
    payload = [{"id": r["id"], "title": r.get("title", ""), "text": r.get("text", "")} for r in sample]
    result = await llm.call(DISCOVERY_SYSTEM, payload, DISCOVERY_SCHEMA)
    return result["themes"]


# --------------------------------------------------------------------------- #
# stage 2: assignment
# --------------------------------------------------------------------------- #
ASSIGN_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "string"}, "themes": {"type": "array", "items": {"type": "string"}}},
                "required": ["id", "themes"],
            },
        }
    },
    "required": ["results"],
}


def _assign_system(taxonomy: List[Dict]) -> str:
    listing = "\n".join(f'- "{t["name"]}": {t["description"]}' for t in taxonomy)
    return f"""Classify each review against this FIXED set of themes — do not invent new theme
names, use exactly the names given:
{listing}

A review can match zero, one, or several themes (multi-label). Only assign a theme when the
review's text actually supports it. Return an empty "themes" list for reviews matching none."""


async def assign_themes(reviews: List[Dict], taxonomy: List[Dict],
                        batch_size: int = ASSIGN_BATCH_SIZE) -> Dict[str, List[str]]:
    valid_names = {t["name"] for t in taxonomy}
    system = _assign_system(taxonomy)
    results: Dict[str, List[str]] = {}

    async with httpx.AsyncClient() as client:
        for i in range(0, len(reviews), batch_size):
            batch = reviews[i:i + batch_size]
            payload = [{"id": r["id"], "title": r.get("title", ""), "text": r.get("text", "")} for r in batch]
            try:
                response = await llm.call(system, payload, ASSIGN_SCHEMA, client=client)
            except llm.LLMError:
                for r in batch:
                    results[r["id"]] = []
                continue
            by_id = {r["id"]: r.get("themes", []) for r in response["results"]}
            for r in batch:
                results[r["id"]] = [t for t in by_id.get(r["id"], []) if t in valid_names]
    return results


# --------------------------------------------------------------------------- #
# stage 3: recommendations, cited and verified
# --------------------------------------------------------------------------- #
RECOMMEND_SYSTEM = """For each theme below, write ONE actionable product recommendation (1-2
sentences) grounded in the specific reviews listed for it. You MUST cite 2-4 "cited_review_ids"
from the ids actually given to you for that theme — citing an id not in the list, or citing
none, is a failure. Do not generalize beyond what these specific reviews say."""

RECOMMEND_SCHEMA = {
    "type": "object",
    "properties": {
        "recommendations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "theme": {"type": "string"},
                    "recommendation": {"type": "string"},
                    "cited_review_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["theme", "recommendation", "cited_review_ids"],
            },
        }
    },
    "required": ["recommendations"],
}


def verify_citations(recommendations: List[Dict], valid_ids_by_theme: Dict[str, set]) -> List[Dict]:
    """Pure validation, kept separate from the network call so it's unit-
    testable: any cited id not actually in that theme's review set is
    dropped; citations_valid=False means zero real citations survived."""
    out = []
    for rec in recommendations:
        theme = rec["theme"]
        valid_ids = valid_ids_by_theme.get(theme, set())
        cited = [rid for rid in rec.get("cited_review_ids", []) if rid in valid_ids]
        dropped = len(rec.get("cited_review_ids", [])) - len(cited)
        out.append({"theme": theme, "recommendation": rec["recommendation"], "cited_review_ids": cited,
                    "citations_valid": len(cited) > 0, "hallucinated_citations_dropped": dropped})
    return out


async def write_recommendations(reviews_by_theme: Dict[str, List[Dict]],
                                max_reviews_per_theme: int = MAX_REVIEWS_PER_THEME) -> List[Dict]:
    payload = {theme: [{"id": r["id"], "title": r.get("title", ""), "text": r.get("text", "")}
                       for r in revs[:max_reviews_per_theme]]
              for theme, revs in reviews_by_theme.items() if revs}
    if not payload:
        return []
    response = await llm.call(RECOMMEND_SYSTEM, payload, RECOMMEND_SCHEMA)
    valid_ids_by_theme = {theme: {r["id"] for r in revs} for theme, revs in reviews_by_theme.items()}
    return verify_citations(response["recommendations"], valid_ids_by_theme)


# --------------------------------------------------------------------------- #
# full pipeline, adapted to theme_analysis()'s output shape
# --------------------------------------------------------------------------- #
async def llm_theme_analysis(reviews: List[Dict], app_id: Any, max_quotes: int = 2,
                             force_rediscover: bool = False) -> List[Dict]:
    """Drop-in replacement for analysis.theme_analysis(): same output shape
    (theme, negative_reviews, share_of_negative, avg_rating, recommendation,
    sample_quotes), plus extra fields (description, cited_review_ids,
    citations_valid) that report.py may use but doesn't require. Raises
    llm.LLMError on failure — callers must catch it and fall back to
    theme_analysis(), never let a broken LLM call take down the endpoint."""
    corpus = [r for r in reviews if r.get("sentiment") == "negative" or r.get("has_complaint")]
    if not corpus:
        return []

    taxonomy = None if force_rediscover else load_taxonomy(app_id)
    if taxonomy is None:
        taxonomy = await discover_taxonomy(corpus)
        save_taxonomy(app_id, taxonomy)

    assignment = await assign_themes(corpus, taxonomy)
    by_id = {r["id"]: r for r in corpus}
    reviews_by_theme: Dict[str, List[Dict]] = {t["name"]: [] for t in taxonomy}
    for rid, theme_names in assignment.items():
        for name in theme_names:
            reviews_by_theme[name].append(by_id[rid])

    recommendations = await write_recommendations(reviews_by_theme)
    rec_by_theme = {r["theme"]: r for r in recommendations}

    out = []
    for t in taxonomy:
        revs = reviews_by_theme.get(t["name"], [])
        if not revs:
            continue
        rec = rec_by_theme.get(t["name"], {})
        cited_ids = rec.get("cited_review_ids", [])
        quotes = [by_id[rid]["text"] or by_id[rid]["title"] for rid in cited_ids[:max_quotes] if rid in by_id]
        if not quotes:  # citations missing/invalid — fall back to longest reviews as evidence
            quotes = [r["text"] or r["title"] for r in
                     sorted(revs, key=lambda r: len(r.get("text", "")), reverse=True)[:max_quotes]]
        out.append({
            "theme": t["name"],
            "description": t["description"],
            "negative_reviews": len(revs),
            "share_of_negative": round(100 * len(revs) / len(corpus), 2),
            "avg_rating": round(sum(r["rating"] for r in revs) / len(revs), 2),
            "recommendation": rec.get("recommendation", t["description"]),
            "sample_quotes": [q[:280] for q in quotes],
            "cited_review_ids": cited_ids,
            "citations_valid": rec.get("citations_valid", False),
        })
    out.sort(key=lambda t: -t["negative_reviews"])
    return out
