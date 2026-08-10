"""Build a stratified labeling pool: fetch every review Apple exposes for an
app, group by star rating, and sample evenly across ratings so the 1★/2★
bucket (usually a small minority) isn't drowned out.

    python -m eval.sample 1459969523 --per-rating 50 --out eval/data/pool.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import itunes  # noqa: E402
from app.analysis import clean_text  # noqa: E402


async def build_pool(app_id: int, country: str, per_rating: int, seed: int) -> dict:
    raw = await itunes.fetch_pool(app_id, country=country)
    by_rating: dict[int, list] = defaultdict(list)
    for r in raw:
        by_rating[r["rating"]].append(r)

    rng = random.Random(seed)
    items = []
    for rating in range(1, 6):
        bucket = by_rating.get(rating, [])
        rng.shuffle(bucket)
        items.extend(bucket[:per_rating])
    rng.shuffle(items)  # interleave ratings so labeling order carries no pattern

    labeling_items = [
        {
            "id": r["id"],
            "title": clean_text(r["title"]),
            "text": clean_text(r["text"]),
            "rating": r["rating"],  # kept for later analysis, hidden in the labeling UI
        }
        for r in items
    ]
    return {
        "app_id": app_id,
        "country": country,
        "pool_size": len(raw),
        "bucket_sizes": {str(k): len(v) for k, v in sorted(by_rating.items())},
        "items": labeling_items,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("app_id", type=int)
    parser.add_argument("--country", default="us")
    parser.add_argument("--per-rating", type=int, default=50,
                        help="Max reviews to keep per star rating (1-5)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="eval/data/pool.json")
    args = parser.parse_args()

    result = asyncio.run(build_pool(args.app_id, args.country, args.per_rating, args.seed))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Pool: {result['pool_size']} unique reviews. Per-rating available: {result['bucket_sizes']}")
    print(f"Sampled {len(result['items'])} reviews -> {args.out}")


if __name__ == "__main__":
    main()
