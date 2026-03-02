"""Extended Alpaca client with options support."""

from __future__ import annotations

import os
import time as time_module
from typing import Iterator

import requests

from . import config


def _chunked(items: list[str], size: int) -> Iterator[list[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


class AlpacaClient:
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        base_url: str,
        data_url: str,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.data_url = data_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update(
            {
                "APCA-API-KEY-ID": api_key,
                "APCA-API-SECRET-KEY": api_secret,
            }
        )

    @classmethod
    def from_env(cls) -> "AlpacaClient":
        api_key = os.environ.get("ALPACA_API_KEY")
        api_secret = os.environ.get("ALPACA_API_SECRET")
        if not api_key or not api_secret:
            raise ValueError("Missing ALPACA_API_KEY or ALPACA_API_SECRET")
        base_url = os.environ.get(
            "ALPACA_BASE_URL", config.ALPACA_BASE_URL
        ).rstrip("/")
        if base_url.endswith("/v2"):
            base_url = base_url[:-3]
        data_url = os.environ.get(
            "ALPACA_DATA_URL", config.ALPACA_DATA_URL
        ).rstrip("/")
        if data_url.endswith("/v2"):
            data_url = data_url[:-3]
        return cls(
            api_key=api_key,
            api_secret=api_secret,
            base_url=base_url,
            data_url=data_url,
        )

    def _request(
        self,
        base_url: str,
        method: str,
        path: str,
        params: dict | None = None,
        payload: dict | None = None,
    ) -> dict | list:
        url = f"{base_url}{path}"
        backoff = max(1, config.ALPACA_RATE_LIMIT_BACKOFF_INITIAL_SECONDS)
        max_backoff = max(backoff, config.ALPACA_RATE_LIMIT_BACKOFF_MAX_SECONDS)
        max_retries = config.ALPACA_RATE_LIMIT_MAX_RETRIES
        response = None
        for attempt in range(max_retries + 1):
            response = self.session.request(
                method, url, params=params, json=payload,
                timeout=config.HTTP_TIMEOUT_SECONDS,
            )
            if response.status_code == 429 and attempt < max_retries:
                retry_after = response.headers.get("Retry-After")
                sleep_s = backoff
                if retry_after:
                    try:
                        sleep_s = max(sleep_s, int(float(retry_after)))
                    except ValueError:
                        pass
                sleep_s = min(sleep_s, max_backoff)
                time_module.sleep(sleep_s)
                backoff = min(max_backoff, backoff * 2)
                continue
            if response.status_code >= 400:
                raise RuntimeError(
                    f"Alpaca API error {response.status_code}: {response.text}"
                )
            break
        if response is None:
            raise RuntimeError("No response from Alpaca API")
        if response.status_code == 204 or not response.content:
            return {}
        try:
            return response.json()
        except ValueError:
            return {}

    # ------------------------------------------------------------------ account
    def get_account(self) -> dict:
        return self._request(self.base_url, "GET", "/v2/account")

    # ----------------------------------------------------------------- positions
    def get_positions(self) -> list[dict]:
        result = self._request(self.base_url, "GET", "/v2/positions")
        return result if isinstance(result, list) else []

    def close_position(self, symbol_or_id: str) -> dict:
        return self._request(
            self.base_url, "DELETE", f"/v2/positions/{symbol_or_id}"
        )

    # ------------------------------------------------------------------- orders
    def get_orders(
        self,
        status: str = "open",
        limit: int | None = None,
    ) -> list[dict]:
        params: dict[str, str] = {"status": status}
        if limit is not None:
            params["limit"] = str(limit)
        result = self._request(self.base_url, "GET", "/v2/orders", params=params)
        return result if isinstance(result, list) else []

    def submit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str,
        time_in_force: str,
        limit_price: float | None = None,
    ) -> dict:
        payload: dict = {
            "symbol": symbol,
            "qty": str(int(qty)),
            "side": side,
            "type": order_type,
            "time_in_force": time_in_force,
        }
        if limit_price is not None:
            payload["limit_price"] = str(round(limit_price, 2))
        return self._request(self.base_url, "POST", "/v2/orders", payload=payload)

    def get_order(self, order_id: str) -> dict:
        return self._request(self.base_url, "GET", f"/v2/orders/{order_id}")

    def cancel_order(self, order_id: str) -> None:
        self._request(self.base_url, "DELETE", f"/v2/orders/{order_id}")

    # ----------------------------------------------------------------- options
    def get_option_contracts(
        self,
        underlying: str,
        option_type: str | None = None,
        expiration_date_gte: str | None = None,
        expiration_date_lte: str | None = None,
        strike_price_gte: float | None = None,
        strike_price_lte: float | None = None,
        status: str = "active",
        limit: int = 100,
    ) -> list[dict]:
        params: dict[str, str] = {
            "underlying_symbols": underlying,
            "status": status,
            "limit": str(limit),
        }
        if option_type:
            params["type"] = option_type
        if expiration_date_gte:
            params["expiration_date_gte"] = expiration_date_gte
        if expiration_date_lte:
            params["expiration_date_lte"] = expiration_date_lte
        if strike_price_gte is not None:
            params["strike_price_gte"] = str(strike_price_gte)
        if strike_price_lte is not None:
            params["strike_price_lte"] = str(strike_price_lte)
        data = self._request(
            self.base_url, "GET", "/v2/options/contracts", params=params
        )
        if isinstance(data, list):
            return data
        return data.get("option_contracts", []) if isinstance(data, dict) else []

    def get_option_snapshots(self, underlying: str) -> dict:
        """Fetch options snapshots for an underlying symbol."""
        return self._request(
            self.data_url,
            "GET",
            f"/v1beta1/options/snapshots/{underlying}",
            params={"limit": "100"},
        )

    def get_option_latest_quotes(self, symbols: list[str]) -> dict:
        if not symbols:
            return {}
        return self._request(
            self.data_url,
            "GET",
            "/v1beta1/options/quotes/latest",
            params={"symbols": ",".join(symbols)},
        )

    # ----------------------------------------------------------------- market data
    def get_snapshots(self, symbols: list[str]) -> dict[str, dict]:
        snapshots: dict[str, dict] = {}
        for chunk in _chunked(symbols, 50):
            if not chunk:
                continue
            data = self._request(
                self.data_url, "GET", "/v2/stocks/snapshots",
                params={"symbols": ",".join(chunk)},
            )
            if isinstance(data, dict):
                if "snapshots" in data and isinstance(data["snapshots"], dict):
                    snapshots.update(data["snapshots"])
                else:
                    for sym, snap in data.items():
                        if isinstance(snap, dict):
                            snapshots[sym] = snap
        return snapshots

    def get_movers(self, top: int = 20) -> list[str]:
        data = self._request(
            self.data_url, "GET", "/v1beta1/screener/stocks/movers",
            params={"top": str(top)},
        )
        if not isinstance(data, dict):
            return []
        gainers = data.get("gainers") or []
        losers = data.get("losers") or []
        symbols = []
        for item in gainers + losers:
            if isinstance(item, dict) and item.get("symbol"):
                symbols.append(item["symbol"])
        return symbols

    # ----------------------------------------------------------------- bars
    def get_bars(
        self,
        symbol: str,
        timeframe: str = "1Day",
        start: str | None = None,
        end: str | None = None,
        limit: int = 30,
    ) -> list[dict]:
        """Fetch historical bars from Alpaca data API.

        Args:
            symbol: Ticker symbol.
            timeframe: Bar timeframe (e.g. "1Day", "1Hour").
            start: RFC-3339 or YYYY-MM-DD start date.
            end: RFC-3339 or YYYY-MM-DD end date.
            limit: Max bars to return.

        Returns:
            List of bar dicts sorted ascending by time.
        """
        params: dict[str, str] = {
            "timeframe": timeframe,
            "limit": str(limit),
            "sort": "asc",
        }
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        data = self._request(
            self.data_url, "GET", f"/v2/stocks/{symbol}/bars", params=params
        )
        if isinstance(data, dict):
            return data.get("bars", [])
        return data if isinstance(data, list) else []

    # ----------------------------------------------------------------- news
    def get_news(
        self,
        symbols: list[str] | None = None,
        limit: int = 50,
        start: str | None = None,
        end: str | None = None,
    ) -> list[dict]:
        params: dict[str, str] = {
            "limit": str(limit),
            "sort": "desc",
        }
        if symbols:
            params["symbols"] = ",".join(symbols)
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        data = self._request(
            self.data_url, "GET", "/v1beta1/news", params=params
        )
        if isinstance(data, dict):
            return data.get("news", [])
        return data if isinstance(data, list) else []
