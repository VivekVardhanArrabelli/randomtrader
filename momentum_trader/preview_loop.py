"""Run the Polygon-only preview periodically until a cutoff time.

This is meant for paper-testing without Alpaca keys: it repeatedly runs the
preview scanner and, if a trade is taken, prints the realized PnL once the
simulation exits.
"""

from __future__ import annotations

import argparse
import time as time_module
from datetime import datetime, time, timedelta
from pathlib import Path

from dotenv import load_dotenv

from .db import PREVIEW_DB_PATH, TradeLogger
from .preview import run as preview_run
from .utils import EASTERN_TZ, now_eastern


def _log(message: str) -> None:
    timestamp = now_eastern().strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"[{timestamp}] {message}")


def _parse_hhmm(value: str) -> time:
    hour_str, minute_str = value.split(":")
    return time(int(hour_str), int(minute_str), tzinfo=EASTERN_TZ)


def _cutoff_datetime(cutoff: time) -> datetime:
    now = now_eastern()
    return now.replace(hour=cutoff.hour, minute=cutoff.minute, second=0, microsecond=0)


def run() -> None:
    parser = argparse.ArgumentParser(description="Run preview every N minutes until a cutoff time (ET).")
    parser.add_argument("--until", default="12:00", help="Cutoff time in ET (HH:MM). Default: 12:00")
    parser.add_argument("--every-minutes", type=int, default=30, help="Interval minutes between attempts. Default: 30")
    args = parser.parse_args()

    env_path = Path(__file__).with_name(".env")
    load_dotenv(env_path, override=True)

    interval_minutes = max(1, args.every_minutes)
    cutoff_time = _parse_hhmm(args.until)
    cutoff_dt = _cutoff_datetime(cutoff_time)
    if now_eastern() >= cutoff_dt:
        _log(f"cutoff {args.until} ET already passed; nothing to do")
        return

    logger = TradeLogger(PREVIEW_DB_PATH)
    baseline_count = len(logger.fetch_trades())
    attempt = 0

    while True:
        now = now_eastern()
        if now >= cutoff_dt:
            break

        attempt += 1
        _log(f"attempt {attempt}: running preview")
        preview_run()

        trades = logger.fetch_trades()
        if len(trades) > baseline_count:
            trade = trades[-1]
            if trade.exit_time is not None and trade.pnl is not None:
                _log(
                    "trade completed "
                    f"symbol={trade.symbol} entry={trade.entry_price:.2f} "
                    f"exit={trade.exit_price:.2f} pnl={trade.pnl:.2f} reason={trade.exit_reason}"
                )
            else:
                _log(
                    "trade opened "
                    f"symbol={trade.symbol} entry={trade.entry_price:.2f} qty={trade.qty} "
                    "exit not recorded yet"
                )
            return

        sleep_seconds = interval_minutes * 60
        remaining = (cutoff_dt - now_eastern()).total_seconds()
        if remaining <= 0:
            break
        sleep_seconds = min(sleep_seconds, int(remaining))
        _log(f"no trades; sleeping {sleep_seconds}s")
        time_module.sleep(sleep_seconds)

    _log("cutoff reached; no trades taken")


if __name__ == "__main__":
    run()
