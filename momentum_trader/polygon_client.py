"""Polygon REST client helpers."""

from __future__ import annotations

from typing import Iterable
import os
from datetime import datetime, timedelta, timezone

import requests

from . import config
from .bars import Bar
from .utils import parse_timestamp


class PolygonClient:
    def __init__(self, api_key: str, base_url: str) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    @classmethod
    def from_env(cls) -> "PolygonClient":
        api_key = os.environ.get("POLYGON_API_KEY")
        if not api_key:
            raise ValueError("Missing POLYGON_API_KEY")
        base_url = os.environ.get("POLYGON_BASE_URL", config.POLYGON_BASE_URL)
        return cls(api_key=api_key, base_url=base_url)

    def _request(self, path: str, params: dict | None = None) -> dict:
        params = params or {}
        params["apiKey"] = self.api_key
        url = f"{self.base_url}{path}"
        response = self.session.get(url, params=params, timeout=config.HTTP_TIMEOUT_SECONDS)
        if response.status_code >= 400:
            raise RuntimeError(f"Polygon API error {response.status_code}: {response.text}")
        return response.json()

    def get_gainers_snapshot(self, limit: int) -> list[dict]:
        data = self._request(
            "/v2/snapshot/locale/us/markets/stocks/gainers",
            params={"limit": limit},
        )
        return data.get("tickers", [])

    def get_gainers(self, limit: int) -> list[str]:
        tickers = self.get_gainers_snapshot(limit)
        return [item["ticker"] for item in tickers if "ticker" in item]

    def get_ticker_snapshot(self, symbol: str) -> dict:
        data = self._request(f"/v2/snapshot/locale/us/markets/stocks/tickers/{symbol}")
        return data.get("ticker", data)

    def get_fundamentals(self, symbol: str) -> dict:
        try:
            data = self._request(f"/v3/reference/tickers/{symbol}")
        except RuntimeError as exc:
            # Polygon's snapshot endpoints can occasionally include symbols that don't resolve
            # through the reference endpoint. Treat those as "no fundamentals" instead of
            # crashing the scan.
            if "Polygon API error 404" in str(exc):
                return {}
            raise
        return data.get("results", {}) or {}

    def get_aggregate_bars(
        self,
        symbol: str,
        multiplier: int,
        timespan: str,
        start_ms: int,
        end_ms: int,
        limit: int | None = None,
        sort: str = "asc",
        adjusted: bool = True,
    ) -> list[dict]:
        params: dict[str, str] = {
            "adjusted": "true" if adjusted else "false",
            "sort": sort,
        }
        if limit is not None:
            params["limit"] = str(limit)
        data = self._request(
            f"/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start_ms}/{end_ms}",
            params=params,
        )
        return data.get("results", []) or []

    def get_minute_bars(
        self,
        symbol: str,
        minutes: int,
        end: datetime | None = None,
    ) -> list[Bar]:
        if minutes <= 0:
            return []
        end_dt = end or datetime.now(tz=timezone.utc)
        start_dt = end_dt - timedelta(minutes=minutes)
        results = self.get_aggregate_bars(
            symbol=symbol,
            multiplier=1,
            timespan="minute",
            start_ms=int(start_dt.timestamp() * 1000),
            end_ms=int(end_dt.timestamp() * 1000),
            limit=minutes,
            sort="asc",
        )
        bars: list[Bar] = []
        for item in results:
            timestamp = parse_timestamp(item.get("t"))
            if timestamp is None:
                continue
            bars.append(
                Bar(
                    timestamp=timestamp,
                    open=float(item.get("o") or 0.0),
                    high=float(item.get("h") or 0.0),
                    low=float(item.get("l") or 0.0),
                    close=float(item.get("c") or 0.0),
                    volume=float(item.get("v") or 0.0),
                )
            )
        return bars

    def get_minute_bars_between(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        limit: int | None = None,
    ) -> list[Bar]:
        """Fetch 1-minute bars between start and end datetimes (inclusive)."""
        if end <= start:
            return []
        start_dt = start.astimezone(timezone.utc)
        end_dt = end.astimezone(timezone.utc)
        expected = int((end_dt - start_dt).total_seconds() / 60) + 1
        if expected <= 0:
            return []
        if limit is None:
            limit = expected
        results = self.get_aggregate_bars(
            symbol=symbol,
            multiplier=1,
            timespan="minute",
            start_ms=int(start_dt.timestamp() * 1000),
            end_ms=int(end_dt.timestamp() * 1000),
            limit=limit,
            sort="asc",
        )
        bars: list[Bar] = []
        for item in results:
            timestamp = parse_timestamp(item.get("t"))
            if timestamp is None:
                continue
            bars.append(
                Bar(
                    timestamp=timestamp,
                    open=float(item.get("o") or 0.0),
                    high=float(item.get("h") or 0.0),
                    low=float(item.get("l") or 0.0),
                    close=float(item.get("c") or 0.0),
                    volume=float(item.get("v") or 0.0),
                )
            )
        return bars

    @staticmethod
    def extract_market_cap(results: dict) -> float | None:
        value = results.get("market_cap")
        if value is None:
            return None
        return float(value)

    @staticmethod
    def extract_shares_outstanding(results: dict) -> float | None:
        for key in (
            "share_class_shares_outstanding",
            "weighted_shares_outstanding",
            "shares_outstanding",
        ):
            value = results.get(key)
            if value is not None:
                return float(value)
        return None
