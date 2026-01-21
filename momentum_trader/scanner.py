"""Market scanner logic for momentum candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from . import config


@dataclass(frozen=True)
class MarketSnapshot:
    symbol: str
    open_price: float
    current_price: float
    relative_volume: float
    market_cap: float | None
    shares_outstanding: float | None
    is_tradeable: bool

    @property
    def gain_pct(self) -> float:
        return (self.current_price - self.open_price) / self.open_price


@dataclass(frozen=True)
class ScanCandidate:
    symbol: str
    gain_pct: float
    relative_volume: float
    score: float


class MarketScanner:
    def __init__(self, snapshots: Iterable[MarketSnapshot]) -> None:
        self.snapshots = list(snapshots)

    def scan(self) -> list[ScanCandidate]:
        candidates: list[ScanCandidate] = []
        for snapshot in self.snapshots:
            if not snapshot.is_tradeable:
                continue
            if not (config.MIN_PRICE <= snapshot.current_price <= config.MAX_PRICE):
                continue
            if not (config.MIN_GAIN_PCT <= snapshot.gain_pct <= config.MAX_GAIN_PCT):
                continue
            if snapshot.relative_volume < config.MIN_REL_VOLUME:
                continue
            if snapshot.market_cap is not None:
                if not (config.MIN_MARKET_CAP <= snapshot.market_cap <= config.MAX_MARKET_CAP):
                    continue
            if snapshot.shares_outstanding is not None:
                if snapshot.shares_outstanding > config.MAX_SHARES_OUTSTANDING:
                    continue
            score = snapshot.gain_pct * snapshot.relative_volume
            candidates.append(
                ScanCandidate(
                    symbol=snapshot.symbol,
                    gain_pct=snapshot.gain_pct,
                    relative_volume=snapshot.relative_volume,
                    score=score,
                )
            )
        return sorted(candidates, key=lambda item: item.score, reverse=True)
