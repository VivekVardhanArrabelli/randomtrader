"""Polygon data access for gainers and quotes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import requests

from . import config
from .entry import QuoteSnapshot
from .scanner import MarketSnapshot


@dataclass(frozen=True)
class PolygonSnapshot:
    market: MarketSnapshot
    quote: QuoteSnapshot


class PolygonClient:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def fetch_gainers(self) -> list[PolygonSnapshot]:
        url = "https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/gainers"
        response = requests.get(url, params={"apiKey": self.api_key, "limit": config.POLYGON_GAINERS_LIMIT})
        response.raise_for_status()
        payload = response.json()
        tickers = payload.get("tickers", [])
        snapshots: list[PolygonSnapshot] = []
        for item in tickers:
            snapshot = self._parse_snapshot(item)
            if snapshot is not None:
                snapshots.append(snapshot)
        return snapshots

    def _parse_snapshot(self, item: dict[str, Any]) -> PolygonSnapshot | None:
        symbol = item.get("ticker")
        day = item.get("day", {})
        last_quote = item.get("lastQuote", {})
        last_trade = item.get("lastTrade", {})
        prev_day = item.get("prevDay", {})

        open_price = day.get("o")
        current_price = last_trade.get("p") or day.get("c")
        day_high = day.get("h")
        bid = last_quote.get("p") or 0.0
        ask = last_quote.get("P") or 0.0

        if symbol is None or open_price in (None, 0) or current_price is None:
            return None

        day_volume = day.get("v") or 0.0
        prev_volume = prev_day.get("v") or 0.0
        relative_volume = day_volume / prev_volume if prev_volume else 0.0

        market_snapshot = MarketSnapshot(
            symbol=symbol,
            open_price=open_price,
            current_price=current_price,
            relative_volume=relative_volume,
            market_cap=None,
            shares_outstanding=None,
            is_tradeable=True,
        )
        quote_snapshot = QuoteSnapshot(
            symbol=symbol,
            last_price=current_price,
            day_high=day_high or current_price,
            bid=bid,
            ask=ask if ask else current_price,
            halted=False,
        )
        return PolygonSnapshot(market=market_snapshot, quote=quote_snapshot)

    def fetch_share_float(self, symbol: str, as_of: date | None = None) -> tuple[float | None, float | None]:
        if as_of is None:
            as_of = date.today()
        url = f"https://api.polygon.io/v3/reference/tickers/{symbol}"
        response = requests.get(url, params={"apiKey": self.api_key, "date": as_of.isoformat()})
        response.raise_for_status()
        data = response.json().get("results", {})
        market_cap = data.get("market_cap")
        shares_outstanding = data.get("share_class_shares_outstanding")
        return market_cap, shares_outstanding
