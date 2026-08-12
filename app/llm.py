"""Shared Mistral client — one retrying, structured-output call used by
sentiment enrichment (analysis.py), theme discovery (themes.py) and keyword
extraction (keywords.py).

Everything in this module is optional: if MISTRAL_API_KEY is not set, callers
are expected to catch LLMError and fall back to the deterministic VADER/regex
path (see analysis.py's module docstring). No code path in the API requires
this module to succeed.

Chosen over Gemini for its free tier: 1 request/second, 500K tokens/minute,
1B tokens/month — comfortably enough headroom to run a full 100-review batch
(sentiment + themes + keywords, ~15-25 calls total) without hitting a rate
limit mid-run, which Gemini's free tier repeatedly did during development
(see git history / README for the numbers that motivated the switch).
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Optional

import httpx

from .env import load_env

load_env()

MODEL = os.environ.get("MISTRAL_MODEL", "mistral-small-latest")
API_URL = "https://api.mistral.ai/v1/chat/completions"
MAX_RETRIES = 3
TIMEOUT = 60.0


class LLMError(Exception):
    """Any failure calling the LLM — network, quota, malformed response.
    Callers are expected to catch this and fall back, not propagate it."""


def api_key() -> Optional[str]:
    return os.environ.get("MISTRAL_API_KEY")


async def call(system: str, user_payload: Any, schema: dict, *,
               client: Optional[httpx.AsyncClient] = None, key: Optional[str] = None) -> dict:
    """One structured-output call: system prompt + JSON payload + response
    schema in, parsed JSON dict out. Retries on 429/5xx with backoff."""
    key = key or api_key()
    if not key:
        raise LLMError("MISTRAL_API_KEY not set")

    payload = {
        "model": MODEL,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "response", "schema": schema, "strict": True},
        },
    }
    headers = {"Authorization": f"Bearer {key}"}

    owns_client = client is None
    client = client or httpx.AsyncClient()
    try:
        last_error: Optional[Exception] = None
        for attempt in range(MAX_RETRIES):
            try:
                response = await client.post(API_URL, headers=headers, json=payload, timeout=TIMEOUT)
                if response.status_code == 429 or response.status_code >= 500:
                    raise LLMError(f"HTTP {response.status_code}: {response.text[:200]}")
                response.raise_for_status()
                text = response.json()["choices"][0]["message"]["content"]
                return json.loads(text)
            except (httpx.HTTPError, LLMError, KeyError, IndexError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(2 ** attempt)
        raise LLMError(f"Mistral call failed after {MAX_RETRIES} attempts: {last_error}")
    finally:
        if owns_client:
            await client.aclose()
