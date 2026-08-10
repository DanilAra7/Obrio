"""Standalone collection script — same pipeline as the API, no server required.

    python -m app.cli 1459969523 --limit 100 --out sample_report
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from . import itunes, llm, report, store, themes
from .analysis import actions_and_summary, build_insights, complaint_corpus_size, enrich_sentiment_llm, prepare


async def _run(args: argparse.Namespace) -> int:
    try:
        app_id = args.app
        if not str(app_id).isdigit():
            matches = await itunes.search_apps(str(app_id), country=args.country, limit=1)
            if not matches:
                raise itunes.AppNotFoundError(f"No app found for '{app_id}'")
            app_id = matches[0]["app_id"]
            print(f"Resolved '{args.app}' -> {matches[0]['name']} ({app_id})", file=sys.stderr)
        app_id = int(app_id)

        info = await itunes.lookup_app(app_id, country=args.country)
        reviews = prepare(await itunes.fetch_reviews(app_id, country=args.country,
                                                     limit=args.limit, seed=args.seed))
    except itunes.ITunesError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    # No-op without GEMINI_API_KEY (or on any failure) — same graceful
    # fallback as the API, see analysis.enrich_sentiment_llm's docstring.
    reviews = await enrich_sentiment_llm(reviews)
    batch = store.save(app_id, args.country, info, reviews)
    insights = build_insights(reviews)

    if llm.api_key():
        try:
            llm_themes = await themes.llm_theme_analysis(reviews, app_id)
            if llm_themes:
                corpus_size = complaint_corpus_size(reviews)
                actions, summary = actions_and_summary(insights["metrics"], insights["sentiment"],
                                                        llm_themes, corpus_size)
                insights.update(themes=llm_themes, actionable_insights=actions, summary=summary,
                                themes_source="llm")
        except llm.LLMError as exc:
            print(f"note: LLM theme discovery unavailable ({exc}), using regex fallback", file=sys.stderr)

    if args.out:
        with open(f"{args.out}.md", "w", encoding="utf-8") as fh:
            fh.write(report.render_markdown(batch, insights))
        with open(f"{args.out}.html", "w", encoding="utf-8") as fh:
            fh.write(report.render_html(batch, insights))
        with open(f"{args.out}.json", "w", encoding="utf-8") as fh:
            json.dump({"app": info, "insights": insights, "reviews": reviews}, fh,
                      ensure_ascii=False, indent=2)
        print(f"Wrote {args.out}.md / .html / .json ({len(reviews)} reviews)", file=sys.stderr)
    else:
        print(json.dumps(insights, ensure_ascii=False, indent=2))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect App Store reviews and print/write an analysis report")
    parser.add_argument("app", help="numeric App Store id or app name")
    parser.add_argument("--country", default="us")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", help="basename for .md/.html/.json report files")
    sys.exit(asyncio.run(_run(parser.parse_args())))


if __name__ == "__main__":
    main()
