"""Risk management utilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time

from . import config
from .utils import EASTERN_TZ


@dataclass(frozen=True)
class RiskState:
    daily_loss_limit_hit: bool
    reason: str


def evaluate_daily_risk(account_equity: float, day_pl: float) -> RiskState:
    loss_limit = account_equity * config.DAILY_LOSS_LIMIT
    if day_pl <= -loss_limit:
        return RiskState(True, "daily loss limit reached")
    return RiskState(False, "ok")


def within_trading_window(current: datetime) -> bool:
    market_open = time(9, 30, tzinfo=EASTERN_TZ)
    market_close = time(16, 0, tzinfo=EASTERN_TZ)

    open_minutes = 9 * 60 + 30 + config.NO_TRADE_MINUTES_AFTER_OPEN
    close_minutes = 16 * 60 - config.NO_TRADE_MINUTES_BEFORE_CLOSE

    no_trade_start = time(open_minutes // 60, open_minutes % 60, tzinfo=EASTERN_TZ)
    no_trade_end = time(close_minutes // 60, close_minutes % 60, tzinfo=EASTERN_TZ)

    return no_trade_start <= current.timetz() <= no_trade_end and market_open <= current.timetz() <= market_close
