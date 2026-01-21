"""Entry logic for momentum trade candidates."""

from __future__ import annotations

from dataclasses import dataclass

from . import config
from .scanner import ScanCandidate


@dataclass(frozen=True)
class QuoteSnapshot:
    symbol: str
    last_price: float
    day_high: float
    bid: float
    ask: float
    halted: bool

    @property
    def bid_ask_spread_pct(self) -> float:
        if self.ask == 0:
            return 1.0
        return (self.ask - self.bid) / self.ask

    @property
    def within_day_high_band(self) -> bool:
        if self.day_high == 0:
            return False
        return self.last_price >= self.day_high * 0.90


@dataclass(frozen=True)
class EntryDecision:
    symbol: str
    should_enter: bool
    reason: str
    position_size: float


def evaluate_entry(
    candidate: ScanCandidate,
    quote: QuoteSnapshot,
    account_equity: float,
    open_positions: int,
) -> EntryDecision:
    if open_positions >= config.MAX_POSITIONS:
        return EntryDecision(candidate.symbol, False, "max positions reached", 0.0)
    if quote.halted:
        return EntryDecision(candidate.symbol, False, "symbol halted", 0.0)
    if not quote.within_day_high_band:
        return EntryDecision(candidate.symbol, False, "price below day-high band", 0.0)
    if quote.bid_ask_spread_pct >= 0.01:
        return EntryDecision(candidate.symbol, False, "bid-ask spread too wide", 0.0)

    risk_budget = account_equity * config.MAX_RISK_PER_TRADE
    stop_distance = quote.last_price * config.STOP_LOSS_PCT
    if stop_distance == 0:
        return EntryDecision(candidate.symbol, False, "invalid price", 0.0)

    position_size = risk_budget / stop_distance
    return EntryDecision(candidate.symbol, True, "entry conditions met", position_size)
