"""SQLite logging for scans, trades, and daily summaries."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

DEFAULT_DB_PATH = Path(__file__).parent / "logs" / "trades.db"


@dataclass(frozen=True)
class ScanRecord:
    timestamp: datetime
    symbols: str
    criteria: str


@dataclass(frozen=True)
class TradeRecord:
    entry_time: datetime
    symbol: str
    entry_price: float
    exit_price: float | None
    exit_time: datetime | None
    pnl: float | None
    exit_reason: str | None


@dataclass(frozen=True)
class DailySummary:
    date: str
    trades: int
    wins: int
    losses: int
    net_pnl: float


class TradeLogger:
    def __init__(self, db_path: Path = DEFAULT_DB_PATH) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS scans (
                    timestamp TEXT,
                    symbols TEXT,
                    criteria TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trades (
                    entry_time TEXT,
                    symbol TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    exit_time TEXT,
                    pnl REAL,
                    exit_reason TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS daily_summary (
                    date TEXT,
                    trades INTEGER,
                    wins INTEGER,
                    losses INTEGER,
                    net_pnl REAL
                )
                """
            )

    def log_scan(self, record: ScanRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO scans VALUES (?, ?, ?)",
                (record.timestamp.isoformat(), record.symbols, record.criteria),
            )

    def log_trades(self, records: Iterable[TradeRecord]) -> None:
        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO trades VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        record.entry_time.isoformat(),
                        record.symbol,
                        record.entry_price,
                        record.exit_price,
                        record.exit_time.isoformat() if record.exit_time else None,
                        record.pnl,
                        record.exit_reason,
                    )
                    for record in records
                ],
            )

    def log_daily_summary(self, summary: DailySummary) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO daily_summary VALUES (?, ?, ?, ?, ?)",
                (
                    summary.date,
                    summary.trades,
                    summary.wins,
                    summary.losses,
                    summary.net_pnl,
                ),
            )
