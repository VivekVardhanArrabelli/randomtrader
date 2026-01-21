"""Main orchestrator for the momentum trader workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from . import config
from .db import ScanRecord, TradeLogger
from .entry import EntryDecision, QuoteSnapshot, evaluate_entry
from .positions import OpenPosition, PositionManager
from .risk import evaluate_daily_risk, within_trading_window
from .scanner import MarketScanner, MarketSnapshot, ScanCandidate
from .utils import AccountSnapshot, now_eastern


@dataclass(frozen=True)
class MockMarketData:
    snapshots: list[MarketSnapshot]
    quotes: dict[str, QuoteSnapshot]


class TraderEngine:
    def __init__(self, logger: TradeLogger) -> None:
        self.logger = logger
        self.positions = PositionManager()

    def run_scan(self, snapshots: Iterable[MarketSnapshot]) -> list[ScanCandidate]:
        scanner = MarketScanner(snapshots)
        candidates = scanner.scan()
        symbols = [candidate.symbol for candidate in candidates]
        self.logger.log_scan(
            ScanRecord(
                timestamp=now_eastern(),
                symbols=",".join(symbols),
                criteria="momentum scan",
            )
        )
        return candidates

    def evaluate_entries(
        self,
        candidates: Iterable[ScanCandidate],
        quotes: dict[str, QuoteSnapshot],
        account: AccountSnapshot,
    ) -> list[EntryDecision]:
        decisions: list[EntryDecision] = []
        for candidate in candidates:
            quote = quotes[candidate.symbol]
            decision = evaluate_entry(
                candidate=candidate,
                quote=quote,
                account_equity=account.equity,
                open_positions=len(self.positions.positions),
            )
            decisions.append(decision)
        return decisions

    def open_position(self, symbol: str, price: float, qty: float) -> None:
        self.positions.add_position(
            OpenPosition(symbol=symbol, entry_price=price, qty=qty, entry_time=now_eastern())
        )


def run_once(market_data: MockMarketData, account: AccountSnapshot) -> list[EntryDecision]:
    if not within_trading_window(now_eastern()):
        return []

    risk_state = evaluate_daily_risk(account.equity, account.day_pl)
    if risk_state.daily_loss_limit_hit:
        return []

    logger = TradeLogger()
    engine = TraderEngine(logger)
    candidates = engine.run_scan(market_data.snapshots)
    decisions = engine.evaluate_entries(candidates, market_data.quotes, account)

    for decision in decisions:
        if decision.should_enter:
            quote = market_data.quotes[decision.symbol]
            engine.open_position(decision.symbol, quote.ask, decision.position_size)

    return decisions
