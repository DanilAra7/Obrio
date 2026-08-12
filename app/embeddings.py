"""Mistral embeddings client — no numpy dependency, pure Python cosine
similarity (the project stays lightweight; at ~100 reviews this is
milliseconds either way).

Sanity-checked live before building anything on top of it: same-topic review
pairs scored clearly higher cosine similarity than cross-topic pairs — a
clean gap a threshold can sit in (see cluster.py). The clustering threshold
in cluster.py/themes.py is calibrated against a live similarity distribution
on the real complaint corpus, not a fixed guess.
"""

from __future__ import annotations

import asyncio
from typing import Dict, List, Optional

import httpx

from . import llm

EMBED_MODEL = "mistral-embed"
EMBED_URL = "https://api.mistral.ai/v1/embeddings"
BATCH_SIZE = 100  # safety cap per request; chunked regardless of any server-side limit
MAX_RETRIES = 3


def cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


async def embed_texts(texts: List[str], client: Optional[httpx.AsyncClient] = None) -> List[List[float]]:
    """Returns one embedding vector per input text, same order. Raises
    llm.LLMError on failure (caller falls back to the non-embedding path,
    same contract as every other LLM call in this project). Retries on
    429/5xx with backoff, matching app/llm.py's call()."""
    key = llm.api_key()
    if not key:
        raise llm.LLMError("MISTRAL_API_KEY not set")
    headers = {"Authorization": f"Bearer {key}"}

    owns_client = client is None
    client = client or httpx.AsyncClient()
    try:
        vectors: List[List[float]] = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            payload = {"model": EMBED_MODEL, "input": batch}
            last_error: Optional[Exception] = None
            for attempt in range(MAX_RETRIES):
                try:
                    response = await client.post(EMBED_URL, headers=headers, json=payload, timeout=60.0)
                    if response.status_code == 429 or response.status_code >= 500:
                        raise llm.LLMError(f"HTTP {response.status_code}: {response.text[:200]}")
                    response.raise_for_status()
                    body = response.json()
                    vectors.extend(e["embedding"] for e in body["data"])
                    break
                except (httpx.HTTPError, llm.LLMError, KeyError) as exc:
                    last_error = exc
                    if attempt < MAX_RETRIES - 1:
                        await asyncio.sleep(2 ** attempt)
            else:
                raise llm.LLMError(f"Embedding call failed after {MAX_RETRIES} attempts: {last_error}")
        return vectors
    finally:
        if owns_client:
            await client.aclose()


async def embed_reviews(reviews: List[Dict], client: Optional[httpx.AsyncClient] = None) -> Dict[str, List[float]]:
    """reviews: dicts with 'id'/'title'/'text'. Returns {id: embedding}."""
    texts = [f"{r.get('title', '')}. {r.get('text', '')}".strip(". ") for r in reviews]
    vectors = await embed_texts(texts, client=client)
    return {r["id"]: v for r, v in zip(reviews, vectors)}
