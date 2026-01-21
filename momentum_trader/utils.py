"""Shared utilities for scheduling and time handling."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from zoneinfo import ZoneInfo

EASTERN_TZ = ZoneInfo("America/New_York")


def now_eastern() -> datetime:
    return datetime.now(tz=EASTERN_TZ)


def time_in_eastern(hour: int, minute: int) -> datetime:
    current = now_eastern()
    return current.replace(hour=hour, minute=minute, second=0, microsecond=0)


def is_within_market_window(current: datetime, start: time, end: time) -> bool:
    return start <= current.time() <= end


@dataclass(frozen=True)
class AccountSnapshot:
    equity: float
    cash: float
    buying_power: float
    day_pl: float
