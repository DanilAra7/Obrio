"""Shared Gemini client — one retrying, structured-output call used by both
sentiment enrichment (analysis.py) and theme discovery (themes.py).

Everything in this module is optional: if GEMINI_API_KEY is not set, callers
are expected to catch LLMError and fall back to the deterministic VADER/regex
path (see analysis.py's module docstring). No code path in the API requires
this module to succeed.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Optional

import httpx

from .env import load_env

load_env()

MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
MAX_RETRIES = 3
TIMEOUT = 60.0


class LLMError(Exception):
    """Any failure calling the LLM — network, quota, malformed response.
    Callers are expected to catch this and fall back, not propagate it."""


def api_key() -> Optional[str]:
    return os.environ.get("GEMINI_API_KEY")


async def call(system: str, user_payload: Any, schema: dict, *,
               client: Optional[httpx.AsyncClient] = None, key: Optional[str] = None) -> dict:
    """One structured-output call: system prompt + JSON payload + response
    schema in, parsed JSON dict out. Retries on 429/5xx with backoff."""
    key = key or api_key()
    if not key:
        raise LLMError("GEMINI_API_KEY not set")

    payload = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": json.dumps(user_payload, ensure_ascii=False)}]}],
        "generationConfig": {"temperature": 0, "responseMimeType": "application/json", "responseSchema": schema},
    }

    owns_client = client is None
    client = client or httpx.AsyncClient()
    try:
        last_error: Optional[Exception] = None
        for attempt in range(MAX_RETRIES):
            try:
                response = await client.post(API_URL, params={"key": key}, json=payload, timeout=TIMEOUT)
                if response.status_code == 429 or response.status_code >= 500:
                    raise LLMError(f"HTTP {response.status_code}: {response.text[:200]}")
                response.raise_for_status()
                text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
                return json.loads(text)
            except (httpx.HTTPError, LLMError, KeyError, IndexError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(2 ** attempt)
        raise LLMError(f"Gemini call failed after {MAX_RETRIES} attempts: {last_error}")
    finally:
        if owns_client:
            await client.aclose()
