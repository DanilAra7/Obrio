"""Trek C: LLM-driven theme discovery + multi-label assignment + cited
recommendations, replacing app/analysis.py's 9 hardcoded regex themes.

Three stages, each a separate LLM call (or batch of calls):

  1. discover_taxonomy() — reads a sample of the negative∪mixed corpus and
     proposes 5-9 themes with names + definitions, grounded in what's
     actually in THIS app's reviews rather than a generic subscription-app
     checklist. Pinned to disk (eval/data/taxonomy_<app_id>.json) so re-runs
     of the same app are comparable — see the pinning rationale below.

  2. assign_themes() — labels every review against the FIXED taxonomy,
     multi-label (a billing complaint that also gripes about support hits
     both themes). Any theme name the model invents that isn't in the fixed
     taxonomy is dropped — the assignment step must not silently redefine
     the taxonomy it was handed.

  3. write_recommendations() — one recommendation per theme, grounded in
     that theme's actual reviews. The prompt REQUIRES citing specific
     review ids, and the citations are checked programmatically against the
     real assignment before being trusted: any cited id that doesn't
     actually belong to that theme is dropped and the theme is flagged, so
     a hallucinated citation surfaces as "needs review" instead of shipping
     silently as if it were verified.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.env import load_env  # noqa: E402
from eval.llm_sentiment import LLMScorerError, MODEL, API_URL, MAX_RETRIES  # noqa: E402

load_env()

DATA_DIR = Path(__file__).parent / "data"
ASSIGN_BATCH_SIZE = 15
MAX_DISCOVERY_SAMPLE = 60  # enough diversity without spending the whole context window


async def _call(client: httpx.AsyncClient, api_key: str, system: str, user_payload, schema: dict) -> dict:
    payload = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": json.dumps(user_payload, ensure_ascii=False)}]}],
        "generationConfig": {"temperature": 0, "responseMimeType": "application/json", "responseSchema": schema},
    }
    last_error: Optional[Exception] = None
    for attempt in range(MAX_RETRIES):
        try:
            response = await client.post(API_URL, params={"key": api_key}, json=payload, timeout=90.0)
            if response.status_code == 429 or response.status_code >= 500:
                raise LLMScorerError(f"HTTP {response.status_code}: {response.text[:200]}")
            response.raise_for_status()
            text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
            return json.loads(text)
        except (httpx.HTTPError, LLMScorerError, KeyError, IndexError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(2 ** attempt)
    raise LLMScorerError(f"Gemini call failed after {MAX_RETRIES} attempts: {last_error}")


# --------------------------------------------------------------------------- #
# stage 1: discovery
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
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["name", "description"],
            },
        }
    },
    "required": ["themes"],
}


async def discover_taxonomy(reviews: List[Dict], api_key: str, sample_size: int = MAX_DISCOVERY_SAMPLE) -> List[Dict]:
    sample = reviews[:sample_size]
    payload = [{"id": r["id"], "title": r.get("title", ""), "text": r.get("text", "")} for r in sample]
    async with httpx.AsyncClient() as client:
        result = await _call(client, api_key, DISCOVERY_SYSTEM, payload, DISCOVERY_SCHEMA)
    return result["themes"]


def taxonomy_path(app_id) -> Path:
    return DATA_DIR / f"taxonomy_{app_id}.json"


def load_taxonomy(app_id) -> Optional[List[Dict]]:
    path = taxonomy_path(app_id)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def save_taxonomy(app_id, themes: List[Dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    taxonomy_path(app_id).write_text(json.dumps(themes, ensure_ascii=False, indent=2), encoding="utf-8")


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
                "properties": {
                    "id": {"type": "string"},
                    "themes": {"type": "array", "items": {"type": "string"}},
                },
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
review's text actually supports it — do not assign a theme just because the review is
negative in general. Return an empty "themes" list for reviews matching none of the above."""


async def assign_themes(reviews: List[Dict], taxonomy: List[Dict], api_key: str,
                        batch_size: int = ASSIGN_BATCH_SIZE) -> Dict[str, List[str]]:
    valid_names = {t["name"] for t in taxonomy}
    system = _assign_system(taxonomy)
    results: Dict[str, List[str]] = {}

    async with httpx.AsyncClient() as client:
        for i in range(0, len(reviews), batch_size):
            batch = reviews[i:i + batch_size]
            payload = [{"id": r["id"], "title": r.get("title", ""), "text": r.get("text", "")} for r in batch]
            try:
                response = await _call(client, api_key, system, payload, ASSIGN_SCHEMA)
            except LLMScorerError:
                for r in batch:
                    results[r["id"]] = []
                continue
            by_id = {r["id"]: r.get("themes", []) for r in response["results"]}
            for r in batch:
                # drop any hallucinated theme name not in the fixed taxonomy
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


async def write_recommendations(reviews_by_theme: Dict[str, List[Dict]], api_key: str,
                                max_reviews_per_theme: int = 12) -> List[Dict]:
    """reviews_by_theme: {theme_name: [review dicts with id/title/text]}.

    Returns one entry per theme with a verified `cited_review_ids` (hallucinated
    ids removed) and a `citations_valid` flag — False means the model cited
    zero real ids and the recommendation should be treated as unverified,
    not silently trusted."""
    payload = {
        theme: [{"id": r["id"], "title": r.get("title", ""), "text": r.get("text", "")}
                for r in revs[:max_reviews_per_theme]]
        for theme, revs in reviews_by_theme.items() if revs
    }
    if not payload:
        return []

    async with httpx.AsyncClient() as client:
        response = await _call(client, api_key, RECOMMEND_SYSTEM, payload, RECOMMEND_SCHEMA)

    valid_ids_by_theme = {theme: {r["id"] for r in revs} for theme, revs in reviews_by_theme.items()}
    return verify_citations(response["recommendations"], valid_ids_by_theme)


def verify_citations(recommendations: List[Dict], valid_ids_by_theme: Dict[str, set]) -> List[Dict]:
    """Pure validation step, split out from write_recommendations so it's
    testable without a network call: any cited id not actually in that
    theme's review set is dropped, and citations_valid=False signals the
    caller that this recommendation has no verified grounding at all."""
    out = []
    for rec in recommendations:
        theme = rec["theme"]
        valid_ids = valid_ids_by_theme.get(theme, set())
        cited = [rid for rid in rec.get("cited_review_ids", []) if rid in valid_ids]
        dropped = len(rec.get("cited_review_ids", [])) - len(cited)
        out.append({
            "theme": theme,
            "recommendation": rec["recommendation"],
            "cited_review_ids": cited,
            "citations_valid": len(cited) > 0,
            "hallucinated_citations_dropped": dropped,
        })
    return out


# --------------------------------------------------------------------------- #
# full pipeline
# --------------------------------------------------------------------------- #
async def run_pipeline(app_id, reviews: List[Dict], api_key: Optional[str] = None,
                       force_rediscover: bool = False) -> Dict:
    """reviews: the negative∪mixed corpus (already filtered by the caller)."""
    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise LLMScorerError("GEMINI_API_KEY not set (env var or .env file)")

    taxonomy = None if force_rediscover else load_taxonomy(app_id)
    if taxonomy is None:
        taxonomy = await discover_taxonomy(reviews, api_key)
        save_taxonomy(app_id, taxonomy)

    assignment = await assign_themes(reviews, taxonomy, api_key)

    by_id = {r["id"]: r for r in reviews}
    reviews_by_theme: Dict[str, List[Dict]] = {t["name"]: [] for t in taxonomy}
    for rid, theme_names in assignment.items():
        for name in theme_names:
            reviews_by_theme[name].append(by_id[rid])

    recommendations = await write_recommendations(reviews_by_theme, api_key)
    rec_by_theme = {r["theme"]: r for r in recommendations}

    themes_out = []
    for t in taxonomy:
        revs = reviews_by_theme.get(t["name"], [])
        if not revs:
            continue
        rec = rec_by_theme.get(t["name"], {})
        themes_out.append({
            "theme": t["name"],
            "description": t["description"],
            "review_count": len(revs),
            "share_of_corpus": round(100 * len(revs) / len(reviews), 1) if reviews else 0.0,
            "recommendation": rec.get("recommendation", ""),
            "cited_review_ids": rec.get("cited_review_ids", []),
            "citations_valid": rec.get("citations_valid", False),
        })
    themes_out.sort(key=lambda t: -t["review_count"])
    return {"taxonomy": taxonomy, "assignment": assignment, "themes": themes_out}
