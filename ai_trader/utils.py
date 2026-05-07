"""Shared utilities for the AI trader."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, time
from zoneinfo import ZoneInfo

EASTERN_TZ = ZoneInfo("America/New_York")
_EQUITY_CANDIDATE_RE = re.compile(r"^[A-Z][A-Z0-9.-]{0,5}$")
_NON_COMMON_STOCK_SUFFIXES = ("W", "WS", "WT", "U", "R")


def now_eastern() -> datetime:
    return datetime.now(tz=EASTERN_TZ)


def is_market_open(current: datetime | None = None) -> bool:
    current = current or now_eastern()
    market_open = time(9, 30, tzinfo=EASTERN_TZ)
    market_close = time(16, 0, tzinfo=EASTERN_TZ)
    return market_open <= current.timetz() <= market_close


def parse_timestamp(value) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        from datetime import timezone

        if value > 1e12:
            value = value / 1000
        return datetime.fromtimestamp(value, tz=timezone.utc)
    if isinstance(value, str):
        for fmt in (
            "%Y-%m-%dT%H:%M:%S.%f%z",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
        ):
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
    return None


def is_equity_candidate_symbol(symbol: str) -> bool:
    """Return True for symbols suitable for stock/options candidate research."""
    normalized = str(symbol or "").upper().strip()
    if not normalized or not _EQUITY_CANDIDATE_RE.match(normalized):
        return False
    return not (
        len(normalized) >= 5
        and any(normalized.endswith(suffix) for suffix in _NON_COMMON_STOCK_SUFFIXES)
    )


def prioritized_symbol_watchlist(*symbol_groups: list[str], limit: int) -> list[str]:
    """Build a de-duplicated symbol list while preserving group priority."""
    ordered: list[str] = []
    seen: set[str] = set()
    for group in symbol_groups:
        for symbol in group:
            normalized = str(symbol or "").upper().strip()
            if not normalized or normalized in seen:
                continue
            ordered.append(normalized)
            seen.add(normalized)
            if len(ordered) >= limit:
                return ordered
    return ordered


@dataclass(frozen=True)
class AccountSnapshot:
    equity: float
    cash: float
    buying_power: float
    day_pl: float
    positions_value: float


def log(message: str) -> None:
    timestamp = now_eastern().strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"[{timestamp}] {message}", flush=True)
