"""Alpaca paper trading client wrapper."""

from __future__ import annotations

from dataclasses import dataclass

import alpaca_trade_api as tradeapi


@dataclass(frozen=True)
class AlpacaOrder:
    symbol: str
    qty: float
    side: str
    order_type: str
    limit_price: float | None


class AlpacaClient:
    def __init__(self, api_key: str, api_secret: str, base_url: str) -> None:
        self.api = tradeapi.REST(api_key, api_secret, base_url, api_version="v2")

    def get_account(self) -> tradeapi.entity.Account:
        return self.api.get_account()

    def submit_limit_buy(self, symbol: str, qty: float, limit_price: float) -> None:
        self.api.submit_order(
            symbol=symbol,
            qty=qty,
            side="buy",
            type="limit",
            time_in_force="day",
            limit_price=limit_price,
        )
