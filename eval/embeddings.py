"""Gemini embeddings client — no numpy dependency, pure Python cosine
similarity (the project stays lightweight; at ~100 reviews x 3072 dims this
is milliseconds either way).

Sanity-checked live before building anything on top of it: same-topic review
pairs scored 0.76-0.81 cosine similarity, cross-topic pairs 0.50-0.54 — a
clean gap a threshold can sit in (see cluster.py).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import llm  # noqa: E402 — reuses api_key()/LLMError, not the generateContent caller

EMBED_MODEL = "gemini-embedding-001"
EMBED_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{EMBED_MODEL}:batchEmbedContents"
BATCH_SIZE = 100  # Gemini's batchEmbedContents cap per request


def cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


async def embed_texts(texts: List[str], client: httpx.AsyncClient = None) -> List[List[float]]:
    """Returns one embedding vector per input text, same order. Raises
    llm.LLMError on failure (caller falls back to the non-embedding pipeline,
    same contract as every other LLM call in this project)."""
    key = llm.api_key()
    if not key:
        raise llm.LLMError("GEMINI_API_KEY not set")

    owns_client = client is None
    client = client or httpx.AsyncClient()
    try:
        vectors: List[List[float]] = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            payload = {"requests": [
                {"model": f"models/{EMBED_MODEL}", "content": {"parts": [{"text": t}]}} for t in batch
            ]}
            try:
                response = await client.post(EMBED_URL, params={"key": key}, json=payload, timeout=60.0)
                response.raise_for_status()
                body = response.json()
                vectors.extend(e["values"] for e in body["embeddings"])
            except (httpx.HTTPError, KeyError) as exc:
                raise llm.LLMError(f"Embedding call failed: {exc}") from exc
        return vectors
    finally:
        if owns_client:
            await client.aclose()


async def embed_reviews(reviews: List[Dict], client: httpx.AsyncClient = None) -> Dict[str, List[float]]:
    """reviews: dicts with 'id'/'title'/'text'. Returns {id: embedding}."""
    texts = [f"{r.get('title', '')}. {r.get('text', '')}".strip(". ") for r in reviews]
    vectors = await embed_texts(texts, client=client)
    return {r["id"]: v for r, v in zip(reviews, vectors)}
