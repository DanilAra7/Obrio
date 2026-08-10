"""In-memory store for collected review batches.

Deliberately trivial: the API is stateless apart from this cache, so swapping it
for Redis/Postgres later only means reimplementing `save`/`get`.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

_LOCK = threading.Lock()
_DATA: Dict[tuple, Dict[str, Any]] = {}


def save(app_id: int, country: str, app: Dict[str, Any], reviews: List[Dict[str, Any]]) -> Dict[str, Any]:
    batch = {
        "app": app,
        "country": country,
        "collected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reviews": reviews,
    }
    with _LOCK:
        _DATA[(app_id, country)] = batch
    return batch


def get(app_id: int, country: str) -> Optional[Dict[str, Any]]:
    with _LOCK:
        return _DATA.get((app_id, country))


def collections() -> List[Dict[str, Any]]:
    with _LOCK:
        return [
            {"app_id": key[0], "country": key[1], "name": batch["app"].get("name"),
             "reviews": len(batch["reviews"]), "collected_at": batch["collected_at"]}
            for key, batch in _DATA.items()
        ]


def clear() -> None:
    with _LOCK:
        _DATA.clear()
