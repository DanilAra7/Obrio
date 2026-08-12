"""LLM-based sentiment scorer (Mistral) — Trek A candidate to replace/augment
the VADER+rating blend in app/analysis.py.

Design choices (see session discussion):
  * Blind to the star rating — the prompt only ever sees title+text, mirroring
    exactly what the human labels in eval/label_app.py were produced from.
    Mixing in the rating belongs to a separate fusion step, not to this scorer.
  * Same -1..1 anchored scale as the labeling UI, so eval numbers are directly
    comparable between "what a human said" and "what the model said".
  * Batched requests (default 15 reviews/call) — free-tier quotas are rate-
    limited per request, not per token, so batching is the main lever.
  * Disk cache keyed by (review id, model, prompt version) — re-running eval
    after a code change should not re-spend quota on unchanged reviews.
  * On failure after retries, the caller gets an explicit `error` per item
    instead of a silently wrong 0.0 — callers decide the fallback policy
    (see app-side integration, which falls back to the VADER blend).

Reuses app.llm.call() for the actual network call (retries, auth, schema
enforcement) rather than maintaining a second HTTP client — this module adds
only what's specific to evaluation: confidence/reason fields (useful for
eyeballing model reasoning, not needed in production), a disk cache, and
per-batch error tolerance instead of raising.

    python -m eval.llm_sentiment --pool eval/data/pool.json --out eval/data/llm_scores.json
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import llm  # noqa: E402

PROMPT_VERSION = "v2"  # v2 adds has_complaint — bumped so v1 cache entries aren't reused
BATCH_SIZE = 15
CACHE_PATH = Path(__file__).parent / "data" / "llm_cache.json"

SYSTEM_PROMPT = """You are scoring App Store review sentiment for a product-analytics pipeline
whose job is to catch every piece of user dissatisfaction, even when it's buried inside an
otherwise positive review. For EACH review, read only its title and text (you are NOT given its
star rating — that is intentional, judge the words alone) and output TWO independent signals:

1. "score": the review's overall/net tone as a float from -1.0 to 1.0:
     -1.0  Very negative — angry, feels scammed, demands a refund
     -0.5  Negative — a real complaint, frustrated
      0.0  Neutral — factual, no strong emotion either way
     +0.5  Positive — satisfied, mild praise
     +1.0  Very positive — enthusiastic, glowing
   Not restricted to these five anchors — use any value in between (e.g. -0.3, 0.7).

2. "has_complaint": true if the review expresses ANY dissatisfaction, however minor, partial,
   or ultimately resolved — even inside a review whose net tone is clearly positive. Examples
   that must be true: praise for the app with one gripe about ads; a billing problem that
   support later fixed to the user's satisfaction; a feature request framed politely. Only
   false when there is truly nothing the user is unhappy about. This flag is what feeds the
   "what should we fix" pipeline downstream, so it should err toward true on any real signal
   of a problem, not just when the review reads negative overall.

Judge sarcasm and irony by intent, not surface words: "great, another crash" is negative
despite containing "great". A short review with too little signal to judge should get a score
near 0.0, has_complaint false, and low confidence rather than a guess.

Return one result per input review, in the same order, referencing its "id" exactly as given."""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "score": {"type": "number"},
                    "has_complaint": {"type": "boolean"},
                    "confidence": {"type": "number", "description": "0..1, how sure you are"},
                    "reason": {"type": "string", "description": "5-12 words justifying the score"},
                },
                "required": ["id", "score", "has_complaint", "confidence", "reason"],
            },
        }
    },
    "required": ["results"],
}


class LLMScorerError(Exception):
    pass


def _cache_key(review_id: str) -> str:
    raw = f"{PROMPT_VERSION}:{llm.MODEL}:{review_id}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def _load_cache() -> Dict[str, dict]:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def _save_cache(cache: Dict[str, dict]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(CACHE_PATH)


def _batches(items: List[dict], size: int) -> List[List[dict]]:
    return [items[i:i + size] for i in range(0, len(items), size)]


async def _score_batch(batch: List[dict]) -> Dict[str, dict]:
    payload = [{"id": it["id"], "title": it.get("title", ""), "text": it.get("text", "")} for it in batch]
    parsed = await llm.call(SYSTEM_PROMPT, payload, RESPONSE_SCHEMA)
    return {r["id"]: r for r in parsed["results"]}


async def score_items(items: List[dict], batch_size: int = BATCH_SIZE, use_cache: bool = True) -> Dict[str, dict]:
    """Score a list of {id, title, text} dicts. Returns {id: {score, confidence, reason, source}}.

    Cached results are reused; only uncached ids hit the API. Items whose batch
    fails after retries get {"error": "..."} instead of a fabricated score —
    the caller decides whether to fall back (see app-side wiring)."""
    if not llm.api_key():
        raise LLMScorerError("MISTRAL_API_KEY not set (env var or .env file)")

    cache = _load_cache() if use_cache else {}
    results: Dict[str, dict] = {}
    to_fetch = []
    for it in items:
        key = _cache_key(it["id"])
        if use_cache and key in cache:
            results[it["id"]] = cache[key]
        else:
            to_fetch.append(it)

    for batch in _batches(to_fetch, batch_size):
        try:
            batch_result = await _score_batch(batch)
        except llm.LLMError as exc:
            for it in batch:
                results[it["id"]] = {"error": str(exc)}
            continue
        for it in batch:
            r = batch_result.get(it["id"])
            if r is None:
                results[it["id"]] = {"error": "missing from LLM response"}
                continue
            entry = {"score": float(r["score"]), "has_complaint": bool(r.get("has_complaint", False)),
                     "confidence": float(r.get("confidence", 0.5)),
                     "reason": r.get("reason", ""), "source": "mistral"}
            results[it["id"]] = entry
            if use_cache:
                cache[_cache_key(it["id"])] = entry
    if to_fetch and use_cache:
        _save_cache(cache)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", default="eval/data/pool.json")
    parser.add_argument("--out", default="eval/data/llm_scores.json")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    pool = json.loads(Path(args.pool).read_text(encoding="utf-8"))
    results = asyncio.run(score_items(pool["items"], use_cache=not args.no_cache))

    errors = {k: v for k, v in results.items() if "error" in v}
    print(f"Scored {len(results) - len(errors)}/{len(results)} reviews. {len(errors)} errors.")
    if errors:
        print("First error:", next(iter(errors.values())))

    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
