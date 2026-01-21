"""Position monitoring and exit logic."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time

from . import config
from .utils import EASTERN_TZ


@dataclass(frozen=True)
class OpenPosition:
    symbol: str
    entry_price: float
    qty: float
    entry_time: datetime


@dataclass(frozen=True)
class ExitDecision:
    symbol: str
    should_exit: bool
    reason: str
    exit_price: float


class PositionManager:
    def __init__(self) -> None:
        self.positions: dict[str, OpenPosition] = {}

    def add_position(self, position: OpenPosition) -> None:
        self.positions[position.symbol] = position

    def remove_position(self, symbol: str) -> None:
        self.positions.pop(symbol, None)

    def evaluate_exit(self, symbol: str, last_price: float, now: datetime) -> ExitDecision:
        position = self.positions[symbol]
        profit_target = position.entry_price * (1 + config.PROFIT_TARGET_PCT)
        stop_loss = position.entry_price * (1 - config.STOP_LOSS_PCT)
        time_stop = time(config.TIME_STOP_HOUR, config.TIME_STOP_MINUTE, tzinfo=EASTERN_TZ)

        if last_price >= profit_target:
            return ExitDecision(symbol, True, "profit target", last_price)
        if last_price <= stop_loss:
            return ExitDecision(symbol, True, "stop loss", last_price)
        if now.timetz() >= time_stop:
            return ExitDecision(symbol, True, "time stop", last_price)

        return ExitDecision(symbol, False, "hold", last_price)
