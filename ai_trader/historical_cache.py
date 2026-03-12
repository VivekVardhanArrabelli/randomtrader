"""Persistent cache for historical Polygon responses used by backtests."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_HISTORICAL_CACHE_DB_PATH = Path(__file__).parent / "logs" / "historical_data.db"


def _normalize_params(params: dict[str, Any] | None) -> dict[str, Any]:
    if not params:
        return {}
    return {
        str(key): params[key]
        for key in sorted(params)
    }


class PolygonResponseStore:
    """SQLite-backed raw response store keyed by request path and params."""

    def __init__(self, db_path: Path = DEFAULT_HISTORICAL_CACHE_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS polygon_http_cache (
                    cache_key TEXT PRIMARY KEY,
                    path TEXT NOT NULL,
                    params_json TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    fetched_at TEXT NOT NULL
                )
                """
            )

    def _cache_key(self, path: str, params: dict[str, Any] | None) -> str:
        normalized = _normalize_params(params)
        payload = json.dumps(
            {"path": path, "params": normalized},
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
        cache_key = self._cache_key(path, params)
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT response_json
                FROM polygon_http_cache
                WHERE cache_key = ?
                """,
                (cache_key,),
            ).fetchone()
        if row is None:
            return None
        return json.loads(str(row[0]))

    def put(
        self,
        path: str,
        params: dict[str, Any] | None,
        response: dict[str, Any],
    ) -> None:
        cache_key = self._cache_key(path, params)
        normalized = _normalize_params(params)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO polygon_http_cache (
                    cache_key,
                    path,
                    params_json,
                    response_json,
                    fetched_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    cache_key,
                    path,
                    json.dumps(normalized, sort_keys=True, default=str),
                    json.dumps(response, sort_keys=True, default=str),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def entry_count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM polygon_http_cache").fetchone()
        return int(row[0] if row else 0)
