"""Small helpers for 1-minute bar analysis (ATR/VWAP + simple breakout checks)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable


@dataclass(frozen=True)
class Bar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


def _true_range(prev_close: float, high: float, low: float) -> float:
    return max(high - low, abs(high - prev_close), abs(low - prev_close))


def atr(bars: list[Bar], period: int = 14) -> float | None:
    """Simple ATR as the average true range over the last `period` bars."""
    if period <= 0:
        return None
    if len(bars) < period + 1:
        return None
    trs: list[float] = []
    for idx in range(-period, 0):
        prev = bars[idx - 1].close
        current = bars[idx]
        trs.append(_true_range(prev, current.high, current.low))
    return sum(trs) / len(trs) if trs else None


def vwap(bars: Iterable[Bar]) -> float | None:
    total_pv = 0.0
    total_volume = 0.0
    for bar in bars:
        typical = (bar.high + bar.low + bar.close) / 3.0
        total_pv += typical * bar.volume
        total_volume += bar.volume
    if total_volume <= 0:
        return None
    return total_pv / total_volume


def consolidation_breakout(
    bars: list[Bar],
    consolidation_bars: int = 5,
    max_range_pct: float = 0.03,
    breakout_buffer_pct: float = 0.001,
    volume_multiplier: float = 1.5,
) -> tuple[bool, str]:
    """Detect a simple tight-range consolidation followed by an upside breakout bar."""
    if consolidation_bars <= 0:
        return False, "invalid consolidation window"
    if len(bars) < consolidation_bars + 1:
        return False, "insufficient bars"

    consolidation = bars[-(consolidation_bars + 1) : -1]
    breakout = bars[-1]
    high = max(bar.high for bar in consolidation)
    low = min(bar.low for bar in consolidation)
    mid = (high + low) / 2.0
    if mid <= 0:
        return False, "invalid consolidation prices"
    range_pct = (high - low) / mid
    if range_pct > max_range_pct:
        return False, "no consolidation"

    if breakout.close <= high * (1 + breakout_buffer_pct):
        return False, "no breakout"

    avg_volume = sum(bar.volume for bar in consolidation) / len(consolidation)
    if avg_volume > 0 and breakout.volume < avg_volume * volume_multiplier:
        return False, "no volume expansion"

    return True, "breakout confirmed"
