"""REST API for App Store review collection, metrics and insights."""

from __future__ import annotations

import csv
import io
import json
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Path, Query
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from pydantic import BaseModel, Field

from . import itunes, report, store
from .analysis import apply_llm_upgrades, calculate_metrics, enrich_sentiment_llm, prepare, sentiment_breakdown

app = FastAPI(
    title="Apple Store Review Analysis API",
    version="1.0.0",
    description=(
        "Collects reviews from the public App Store RSS feed, computes rating metrics, "
        "runs sentiment analysis and turns negative feedback into actionable insights."
    ),
)

MAX_LIMIT = itunes.PAGE_SIZE * itunes.MAX_PAGES


class CollectRequest(BaseModel):
    app_id: Optional[int] = Field(None, description="Numeric App Store id, e.g. 1459969523", examples=[1459969523])
    app_name: Optional[str] = Field(None, description="Used to resolve the id when app_id is not given")
    country: str = Field("us", min_length=2, max_length=2, description="Storefront code, e.g. us / gb / ua")
    limit: int = Field(100, ge=1, le=MAX_LIMIT, description=f"Number of random reviews (max {MAX_LIMIT})")
    sort: str = Field("mostrecent", pattern="^(mostrecent|mosthelpful)$")
    seed: Optional[int] = Field(None, description="Seed for reproducible sampling")


@app.exception_handler(itunes.AppNotFoundError)
@app.exception_handler(itunes.NoReviewsError)
async def _not_found_handler(_, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(itunes.ITunesError)
async def _upstream_handler(_, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=502, content={"detail": str(exc)})


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
async def _collect(app_id: Optional[int], app_name: Optional[str], country: str,
                   limit: int, sort: str = "mostrecent", seed: Optional[int] = None) -> Dict[str, Any]:
    if app_id is None:
        if not app_name:
            raise HTTPException(status_code=422, detail="Provide either app_id or app_name")
        matches = await itunes.search_apps(app_name, country=country, limit=1)
        if not matches:
            raise itunes.AppNotFoundError(f"No app found for '{app_name}' in storefront '{country}'")
        app_id = matches[0]["app_id"]

    info = await itunes.lookup_app(app_id, country=country)
    raw = await itunes.fetch_reviews(app_id, country=country, limit=limit, sort=sort, seed=seed)
    reviews = prepare(raw)
    # No-op if MISTRAL_API_KEY isn't set (returns the VADER-scored reviews
    # unchanged) — see enrich_sentiment_llm's docstring for why this upgrade
    # is worth it when a key is available: has_complaint recall 1.00 vs 0.37.
    reviews = await enrich_sentiment_llm(reviews)
    return store.save(app_id, country, info, reviews)


async def _batch(app_id: int, country: str, limit: int = 100, refresh: bool = False) -> Dict[str, Any]:
    """Return a cached batch, collecting it on demand so every endpoint works standalone."""
    cached = store.get(app_id, country)
    if cached and not refresh:
        return cached
    return await _collect(app_id, None, country, limit)


# --------------------------------------------------------------------------- #
# endpoints
# --------------------------------------------------------------------------- #
@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse("/docs")


@app.get("/health", tags=["meta"])
async def health() -> Dict[str, Any]:
    return {"status": "ok", "collections": store.collections()}


@app.get("/apps/search", tags=["collection"], summary="Find an app id by name")
async def search(
    q: str = Query(..., min_length=1, description="App name, e.g. 'nebula horoscope'"),
    country: str = Query("us", min_length=2, max_length=2),
    limit: int = Query(5, ge=1, le=25),
) -> Dict[str, Any]:
    return {"results": await itunes.search_apps(q, country=country, limit=limit)}


@app.post("/reviews/collect", tags=["collection"], summary="Collect N random reviews for an app")
async def collect(payload: CollectRequest) -> Dict[str, Any]:
    batch = await _collect(payload.app_id, payload.app_name, payload.country,
                           payload.limit, payload.sort, payload.seed)
    return {
        "app": batch["app"],
        "country": batch["country"],
        "collected_at": batch["collected_at"],
        "collected_reviews": len(batch["reviews"]),
        "metrics": calculate_metrics(batch["reviews"]),
        "sentiment": sentiment_breakdown(batch["reviews"]),
        "links": {
            "metrics": f"/apps/{batch['app']['app_id']}/metrics?country={batch['country']}",
            "insights": f"/apps/{batch['app']['app_id']}/insights?country={batch['country']}",
            "raw_json": f"/apps/{batch['app']['app_id']}/reviews?country={batch['country']}&format=json",
            "raw_csv": f"/apps/{batch['app']['app_id']}/reviews?country={batch['country']}&format=csv",
            "report": f"/apps/{batch['app']['app_id']}/report?country={batch['country']}",
        },
    }


@app.get("/apps/{app_id}/metrics", tags=["analysis"], summary="Rating metrics for a collected app")
async def metrics(
    app_id: int = Path(..., gt=0),
    country: str = Query("us", min_length=2, max_length=2),
    limit: int = Query(100, ge=1, le=MAX_LIMIT, description="Used only if the app has not been collected yet"),
    refresh: bool = Query(False, description="Re-collect before computing"),
) -> Dict[str, Any]:
    batch = await _batch(app_id, country, limit, refresh)
    return {"app": batch["app"], "collected_at": batch["collected_at"],
            "metrics": calculate_metrics(batch["reviews"]),
            "sentiment": sentiment_breakdown(batch["reviews"])}


@app.get("/apps/{app_id}/insights", tags=["analysis"], summary="Sentiment, negative keywords and recommendations")
async def insights(
    app_id: int = Path(..., gt=0),
    country: str = Query("us", min_length=2, max_length=2),
    limit: int = Query(100, ge=1, le=MAX_LIMIT),
    refresh: bool = Query(False),
) -> Dict[str, Any]:
    batch = await _batch(app_id, country, limit, refresh)
    return {"app": batch["app"], "collected_at": batch["collected_at"],
            **await apply_llm_upgrades(batch["reviews"], app_id)}


@app.get("/apps/{app_id}/reviews", tags=["analysis"], summary="Download the raw collected reviews (JSON or CSV)")
async def raw_reviews(
    app_id: int = Path(..., gt=0),
    country: str = Query("us", min_length=2, max_length=2),
    limit: int = Query(100, ge=1, le=MAX_LIMIT),
    fmt: str = Query("json", alias="format", pattern="^(json|csv)$"),
    refresh: bool = Query(False),
):
    batch = await _batch(app_id, country, limit, refresh)
    reviews: List[Dict[str, Any]] = batch["reviews"]
    filename = f"reviews_{app_id}_{country}"
    if fmt == "json":
        body = json.dumps({"app": batch["app"], "collected_at": batch["collected_at"], "reviews": reviews},
                          ensure_ascii=False, indent=2)
        return Response(
            content=body,
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{filename}.json"'},
        )

    buffer = io.StringIO()
    columns = ["id", "rating", "title", "text", "author", "version", "updated", "votes",
               "sentiment", "sentiment_score", "has_complaint", "sentiment_source"]
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(reviews)
    return Response(
        content=buffer.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}.csv"'},
    )


@app.get("/apps/{app_id}/report", response_class=HTMLResponse, tags=["analysis"],
         summary="Visual HTML report (rating + sentiment distribution, themes)")
async def html_report(
    app_id: int = Path(..., gt=0),
    country: str = Query("us", min_length=2, max_length=2),
    limit: int = Query(100, ge=1, le=MAX_LIMIT),
    refresh: bool = Query(False),
) -> HTMLResponse:
    batch = await _batch(app_id, country, limit, refresh)
    insights = await apply_llm_upgrades(batch["reviews"], app_id)
    return HTMLResponse(report.render_html(batch, insights))
