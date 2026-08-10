"""Client for the public Apple/iTunes endpoints (search + customer reviews RSS)."""

from __future__ import annotations

import asyncio
import math
import random
from typing import Any, Dict, List, Optional, Sequence

import httpx

SEARCH_URL = "https://itunes.apple.com/search"
LOOKUP_URL = "https://itunes.apple.com/lookup"
REVIEWS_URL = "https://itunes.apple.com/{country}/rss/customerreviews/page={page}/id={app_id}/sortby={sort}/json"

PAGE_SIZE = 50          # Apple returns at most 50 reviews per RSS page
MAX_PAGES = 10          # ...and refuses to paginate past page 10 (500 reviews max)
TIMEOUT = httpx.Timeout(15.0)


class ITunesError(Exception):
    """Upstream failure (network, timeout, malformed payload)."""


class AppNotFoundError(ITunesError):
    """The app id / search term does not resolve to an App Store app."""


class NoReviewsError(ITunesError):
    """The app exists but has no reviews in the requested storefront."""


# --------------------------------------------------------------------------- #
# parsing (pure, unit-testable)
# --------------------------------------------------------------------------- #
def parse_entry(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert one RSS entry into a review dict, or None if it is not a review.

    The first entry of the feed describes the app itself and carries no rating,
    so it is skipped here rather than special-cased by index.
    """
    if not isinstance(entry, dict) or "im:rating" not in entry:
        return None
    try:
        rating = int(entry["im:rating"]["label"])
    except (KeyError, TypeError, ValueError):
        return None
    if not 1 <= rating <= 5:
        return None

    def label(key: str, default: str = "") -> str:
        node = entry.get(key)
        if isinstance(node, dict):
            value = node.get("label")
            if isinstance(value, str):
                return value.strip()
        return default

    return {
        "id": label("id"),
        "title": label("title"),
        "text": label("content"),
        "rating": rating,
        "author": (entry.get("author") or {}).get("name", {}).get("label", ""),
        "version": label("im:version"),
        "updated": label("updated"),
        "votes": int(label("im:voteSum", "0") or 0),
    }


def parse_feed(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract reviews from one RSS page payload (tolerant of Apple's quirks)."""
    entries = ((payload or {}).get("feed") or {}).get("entry") or []
    if isinstance(entries, dict):  # Apple collapses a single-item list into a dict
        entries = [entries]
    reviews = (parse_entry(e) for e in entries)
    return [r for r in reviews if r is not None]


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
async def _get_json(client: httpx.AsyncClient, url: str, **kwargs: Any) -> Dict[str, Any]:
    try:
        response = await client.get(url, **kwargs)
        response.raise_for_status()
        # The RSS endpoint answers with text/javascript, so parse explicitly.
        return response.json()
    except httpx.HTTPStatusError as exc:
        raise ITunesError(f"App Store returned HTTP {exc.response.status_code} for {url}") from exc
    except httpx.HTTPError as exc:
        raise ITunesError(f"Could not reach the App Store: {exc}") from exc
    except ValueError as exc:  # not JSON -> usually an empty/404 RSS page
        raise ITunesError(f"App Store returned a malformed response for {url}") from exc


async def search_apps(term: str, country: str = "us", limit: int = 5) -> List[Dict[str, Any]]:
    """Find apps by name so the caller does not have to know the numeric id."""
    if not term.strip():
        raise AppNotFoundError("Search term must not be empty")
    params = {"term": term, "country": country, "entity": "software", "limit": limit}
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        payload = await _get_json(client, SEARCH_URL, params=params)
    return [_app_info(r) for r in payload.get("results", [])]


async def lookup_app(app_id: int, country: str = "us") -> Dict[str, Any]:
    params = {"id": app_id, "country": country, "entity": "software"}
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        payload = await _get_json(client, LOOKUP_URL, params=params)
    results = payload.get("results") or []
    if not results:
        raise AppNotFoundError(f"No app with id {app_id} in storefront '{country}'")
    return _app_info(results[0])


def _app_info(raw: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "app_id": raw.get("trackId"),
        "name": raw.get("trackName"),
        "developer": raw.get("sellerName"),
        "store_rating": raw.get("averageUserRating"),
        "store_rating_count": raw.get("userRatingCount"),
        "url": raw.get("trackViewUrl"),
    }


async def fetch_reviews(
    app_id: int,
    country: str = "us",
    limit: int = 100,
    sort: str = "mostrecent",
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Collect `limit` random reviews for `app_id`.

    Apple exposes at most 500 reviews (10 pages x 50). We download the pages
    needed to build a pool that is comfortably larger than `limit` and then take
    a uniform random sample of it, so the result is not just "the newest N".
    """
    if limit < 1:
        raise ValueError("limit must be >= 1")
    if limit > PAGE_SIZE * MAX_PAGES:
        raise ValueError(f"limit must be <= {PAGE_SIZE * MAX_PAGES} (App Store hard limit)")

    # Over-fetch (x2, capped at the 10 available pages) to make sampling meaningful.
    pages = min(MAX_PAGES, max(1, math.ceil(limit * 2 / PAGE_SIZE)))
    async with httpx.AsyncClient(timeout=TIMEOUT, follow_redirects=True) as client:
        tasks = [
            _get_json(client, REVIEWS_URL.format(country=country, page=p, app_id=app_id, sort=sort))
            for p in range(1, pages + 1)
        ]
        payloads = await asyncio.gather(*tasks, return_exceptions=True)

    pool: List[Dict[str, Any]] = []
    errors = [p for p in payloads if isinstance(p, Exception)]
    for payload in payloads:
        if not isinstance(payload, Exception):
            pool.extend(parse_feed(payload))

    if not pool:
        if errors:
            raise ITunesError(str(errors[0]))
        raise NoReviewsError(f"App {app_id} has no reviews in storefront '{country}'")

    # Apple's pages overlap occasionally - de-duplicate by review id.
    unique = list({r["id"]: r for r in pool}.values())
    if len(unique) <= limit:
        return unique
    return random.Random(seed).sample(unique, limit)


async def fetch_pool(
    app_id: int,
    country: str = "us",
    sorts: Sequence[str] = ("mostrecent", "mosthelpful"),
) -> List[Dict[str, Any]]:
    """Fetch and de-duplicate every review Apple exposes for `app_id`.

    Not used by the API (which samples `limit` reviews) — this is for building
    a large, diverse pool to draw an eval/labeling set from. Querying multiple
    sort orders surfaces different reviews (Apple ranks each list differently),
    which pushes the pool above what a single sort order (max 500) returns.
    """
    async with httpx.AsyncClient(timeout=TIMEOUT, follow_redirects=True) as client:
        tasks = [
            _get_json(client, REVIEWS_URL.format(country=country, page=p, app_id=app_id, sort=s))
            for s in sorts
            for p in range(1, MAX_PAGES + 1)
        ]
        payloads = await asyncio.gather(*tasks, return_exceptions=True)

    pool: List[Dict[str, Any]] = []
    errors = [p for p in payloads if isinstance(p, Exception)]
    for payload in payloads:
        if not isinstance(payload, Exception):
            pool.extend(parse_feed(payload))

    if not pool:
        if errors:
            raise ITunesError(str(errors[0]))
        raise NoReviewsError(f"App {app_id} has no reviews in storefront '{country}'")
    return list({r["id"]: r for r in pool}.values())
