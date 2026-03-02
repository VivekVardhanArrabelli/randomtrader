"""Alpaca REST client helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator
import os
import time as time_module

import requests

from . import config


@dataclass(frozen=True)
class AlpacaCredentials:
    api_key: str
    api_secret: str
    base_url: str
    data_url: str


def _chunked(items: list[str], size: int) -> Iterator[list[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


class AlpacaClient:
    def __init__(self, credentials: AlpacaCredentials) -> None:
        self.base_url = credentials.base_url.rstrip("/")
        self.data_url = credentials.data_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update(
            {
                "APCA-API-KEY-ID": credentials.api_key,
                "APCA-API-SECRET-KEY": credentials.api_secret,
            }
        )

    @classmethod
    def from_env(cls) -> "AlpacaClient":
        api_key = os.environ.get("ALPACA_API_KEY")
        api_secret = os.environ.get("ALPACA_API_SECRET")
        if not api_key or not api_secret:
            raise ValueError("Missing ALPACA_API_KEY or ALPACA_API_SECRET")
        # Alpaca docs often show the trading endpoint including "/v2". Our client
        # includes "/v2" in the request paths, so accept either format here.
        base_url = os.environ.get("ALPACA_BASE_URL", config.ALPACA_BASE_URL).rstrip("/")
        if base_url.endswith("/v2"):
            base_url = base_url[:-3]
        data_url = os.environ.get("ALPACA_DATA_URL", config.ALPACA_DATA_URL).rstrip("/")
        if data_url.endswith("/v2"):
            data_url = data_url[:-3]
        return cls(
            AlpacaCredentials(
                api_key=api_key,
                api_secret=api_secret,
                base_url=base_url,
                data_url=data_url,
            )
        )

    def _request(
        self,
        base_url: str,
        method: str,
        path: str,
        params: dict | None = None,
        payload: dict | None = None,
    ) -> dict:
        url = f"{base_url}{path}"
        backoff_seconds = max(1, int(config.ALPACA_RATE_LIMIT_BACKOFF_INITIAL_SECONDS))
        max_backoff_seconds = max(backoff_seconds, int(config.ALPACA_RATE_LIMIT_BACKOFF_MAX_SECONDS))
        max_retries = max(0, int(getattr(config, "ALPACA_RATE_LIMIT_MAX_RETRIES", 0)))
        for attempt in range(max_retries + 1):
            response = self.session.request(
                method,
                url,
                params=params,
                json=payload,
                timeout=config.HTTP_TIMEOUT_SECONDS,
            )
            if response.status_code == 429 and attempt < max_retries:
                retry_after = response.headers.get("Retry-After")
                retry_after_seconds = None
                if retry_after:
                    try:
                        retry_after_seconds = int(float(retry_after))
                    except ValueError:
                        retry_after_seconds = None
                sleep_seconds = min(backoff_seconds, max_backoff_seconds)
                if retry_after_seconds is not None:
                    sleep_seconds = max(sleep_seconds, retry_after_seconds)
                time_module.sleep(sleep_seconds)
                backoff_seconds = min(max_backoff_seconds, backoff_seconds * 2)
                continue
            if response.status_code >= 400:
                raise RuntimeError(f"Alpaca API error {response.status_code}: {response.text}")
        # Some Alpaca endpoints (notably DELETE /v2/orders/{id}) return 204 No Content.
        if response.status_code == 204 or not response.content:
            return {}
        try:
            return response.json()
        except ValueError:
            # Be defensive: treat unexpected non-JSON success responses as empty payload.
            return {}

    def get_account(self) -> dict:
        return self._request(self.base_url, "GET", "/v2/account")

    def get_assets(self) -> list[dict]:
        return self._request(
            self.base_url,
            "GET",
            "/v2/assets",
            params={"status": "active", "asset_class": "us_equity"},
        )

    def get_positions(self) -> list[dict]:
        return self._request(self.base_url, "GET", "/v2/positions")

    def get_orders(
        self,
        status: str = "open",
        limit: int | None = None,
        after: str | None = None,
    ) -> list[dict]:
        params: dict[str, str] = {"status": status}
        if limit is not None:
            params["limit"] = str(limit)
        if after is not None:
            params["after"] = after
        return self._request(self.base_url, "GET", "/v2/orders", params=params)

    def get_snapshots(self, symbols: Iterable[str]) -> dict[str, dict]:
        snapshots: dict[str, dict] = {}
        symbol_list = list(symbols)
        for chunk in _chunked(symbol_list, config.SNAPSHOT_SYMBOL_CHUNK):
            if not chunk:
                continue
            data = self._request(
                self.data_url,
                "GET",
                "/v2/stocks/snapshots",
                params={"symbols": ",".join(chunk)},
            )
            # Alpaca's response shape can be either:
            # - {"snapshots": {"AAPL": {...}, ...}}
            # - {"AAPL": {...}, "TSLA": {...}, ...}
            # Accept both.
            if "snapshots" in data and isinstance(data.get("snapshots"), dict):
                snapshots.update(data.get("snapshots", {}))
            else:
                for symbol, snapshot in data.items():
                    if isinstance(snapshot, dict):
                        snapshots[symbol] = snapshot
        return snapshots

    def get_movers(self, top: int | None = None) -> dict:
        params: dict[str, str] = {}
        if top is not None:
            params["top"] = str(top)
        return self._request(
            self.data_url,
            "GET",
            "/v1beta1/screener/stocks/movers",
            params=params,
        )

    def get_mover_gainer_symbols(self, top: int) -> list[str]:
        data = self.get_movers(top=top)
        gainers = data.get("gainers") or []
        symbols: list[str] = []
        for item in gainers:
            if not isinstance(item, dict):
                continue
            symbol = item.get("symbol")
            if symbol:
                symbols.append(symbol)
        return symbols

    def get_most_actives(self, top: int | None = None) -> dict:
        params: dict[str, str] = {}
        if top is not None:
            params["top"] = str(top)
        return self._request(
            self.data_url,
            "GET",
            "/v1beta1/screener/stocks/most-actives",
            params=params,
        )

    def get_most_active_symbols(self, top: int) -> list[str]:
        data = self.get_most_actives(top=top)
        actives = data.get("most_actives") or []
        symbols: list[str] = []
        for item in actives:
            if not isinstance(item, dict):
                continue
            symbol = item.get("symbol")
            if symbol:
                symbols.append(symbol)
        return symbols

    def submit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str,
        time_in_force: str,
        limit_price: float | None = None,
        order_class: str | None = None,
        take_profit: dict | None = None,
        stop_loss: dict | None = None,
    ) -> dict:
        payload = {
            "symbol": symbol,
            "qty": str(qty),
            "side": side,
            "type": order_type,
            "time_in_force": time_in_force,
        }
        if limit_price is not None:
            payload["limit_price"] = str(limit_price)
        if order_class is not None:
            payload["order_class"] = order_class
        if take_profit is not None:
            payload["take_profit"] = take_profit
        if stop_loss is not None:
            payload["stop_loss"] = stop_loss
        return self._request(self.base_url, "POST", "/v2/orders", payload=payload)

    def get_order(self, order_id: str) -> dict:
        return self._request(self.base_url, "GET", f"/v2/orders/{order_id}")

    def cancel_order(self, order_id: str) -> None:
        self._request(self.base_url, "DELETE", f"/v2/orders/{order_id}")
