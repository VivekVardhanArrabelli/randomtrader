"""Performance report for logged trades."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from . import config
from .db import DEFAULT_DB_PATH, EntryAttemptRecord, ScanAuditRecord, TradeLogger, TradeRecord


@dataclass(frozen=True)
class TradeMetrics:
    total: int
    wins: int
    losses: int
    breakeven: int
    win_rate: float | None
    avg_win: float
    avg_loss: float
    profit_factor: float | None
    expectancy: float
    gross_profit: float
    gross_loss: float
    max_drawdown: float
    largest_win: float
    largest_loss: float
    avg_hold_minutes: float | None
    median_hold_minutes: float | None
    best_day: tuple[str, float] | None
    worst_day: tuple[str, float] | None


@dataclass(frozen=True)
class ExecutionMetrics:
    attempts: int
    filled: int
    rejected: int
    reject_rate: float | None
    avg_slippage_pct: float | None
    median_slippage_pct: float | None
    avg_fill_seconds: float | None
    median_fill_seconds: float | None


def _safe_div(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2 == 1:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2


def _format_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def _format_percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.2f}%"


def _compute_metrics(trades: list[TradeRecord]) -> TradeMetrics:
    completed = [trade for trade in trades if trade.exit_time and trade.pnl is not None]
    total = len(completed)
    if total == 0:
        return TradeMetrics(
            total=0,
            wins=0,
            losses=0,
            breakeven=0,
            win_rate=None,
            avg_win=0.0,
            avg_loss=0.0,
            profit_factor=None,
            expectancy=0.0,
            gross_profit=0.0,
            gross_loss=0.0,
            max_drawdown=0.0,
            largest_win=0.0,
            largest_loss=0.0,
            avg_hold_minutes=None,
            median_hold_minutes=None,
            best_day=None,
            worst_day=None,
        )

    wins = [trade for trade in completed if trade.pnl and trade.pnl > 0]
    losses = [trade for trade in completed if trade.pnl and trade.pnl < 0]
    breakeven = [trade for trade in completed if trade.pnl == 0]
    gross_profit = sum(trade.pnl for trade in wins if trade.pnl)
    gross_loss = sum(trade.pnl for trade in losses if trade.pnl)
    avg_win = gross_profit / len(wins) if wins else 0.0
    avg_loss = abs(gross_loss / len(losses)) if losses else 0.0
    profit_factor = _safe_div(gross_profit, abs(gross_loss)) if gross_loss else None
    win_rate = _safe_div(len(wins), len(wins) + len(losses))
    expectancy = sum(trade.pnl for trade in completed if trade.pnl) / total
    largest_win = max((trade.pnl for trade in completed if trade.pnl), default=0.0)
    largest_loss = min((trade.pnl for trade in completed if trade.pnl), default=0.0)

    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for trade in sorted(completed, key=lambda t: t.exit_time or t.entry_time):
        equity += trade.pnl or 0.0
        if equity > peak:
            peak = equity
        drawdown = peak - equity
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    hold_minutes: list[float] = []
    for trade in completed:
        if trade.exit_time and trade.entry_time:
            delta = trade.exit_time - trade.entry_time
            hold_minutes.append(delta.total_seconds() / 60)
    avg_hold_minutes = sum(hold_minutes) / len(hold_minutes) if hold_minutes else None
    median_hold_minutes = _median(hold_minutes)

    daily_pnl: dict[str, float] = defaultdict(float)
    for trade in completed:
        date_key = trade.entry_time.date().isoformat()
        daily_pnl[date_key] += trade.pnl or 0.0
    best_day = max(daily_pnl.items(), key=lambda item: item[1]) if daily_pnl else None
    worst_day = min(daily_pnl.items(), key=lambda item: item[1]) if daily_pnl else None

    return TradeMetrics(
        total=total,
        wins=len(wins),
        losses=len(losses),
        breakeven=len(breakeven),
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        profit_factor=profit_factor,
        expectancy=expectancy,
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        max_drawdown=max_drawdown,
        largest_win=largest_win,
        largest_loss=largest_loss,
        avg_hold_minutes=avg_hold_minutes,
        median_hold_minutes=median_hold_minutes,
        best_day=best_day,
        worst_day=worst_day,
    )


def _compute_execution_metrics(attempts: list[EntryAttemptRecord]) -> ExecutionMetrics:
    if not attempts:
        return ExecutionMetrics(
            attempts=0,
            filled=0,
            rejected=0,
            reject_rate=None,
            avg_slippage_pct=None,
            median_slippage_pct=None,
            avg_fill_seconds=None,
            median_fill_seconds=None,
        )
    filled_attempts = [attempt for attempt in attempts if attempt.filled_qty > 0]
    rejected_attempts = [
        attempt
        for attempt in attempts
        if attempt.status in {"rejected", "canceled", "expired"}
        and attempt.filled_qty == 0
    ]
    slippages = [
        attempt.slippage_pct
        for attempt in filled_attempts
        if attempt.slippage_pct is not None
    ]
    fill_seconds = [
        (attempt.fill_time - attempt.signal_time).total_seconds()
        for attempt in filled_attempts
        if attempt.fill_time is not None
    ]
    return ExecutionMetrics(
        attempts=len(attempts),
        filled=len(filled_attempts),
        rejected=len(rejected_attempts),
        reject_rate=_safe_div(len(rejected_attempts), len(attempts)),
        avg_slippage_pct=sum(slippages) / len(slippages) if slippages else None,
        median_slippage_pct=_median(slippages),
        avg_fill_seconds=sum(fill_seconds) / len(fill_seconds) if fill_seconds else None,
        median_fill_seconds=_median(fill_seconds),
    )


def _print_report(metrics: TradeMetrics, execution: ExecutionMetrics) -> None:
    if metrics.total == 0:
        print("No completed trades found.")
    else:
        print("Trade Summary")
        print(f"Total trades: {metrics.total}")
        print(
            f"Wins: {metrics.wins}  Losses: {metrics.losses}  Breakeven: {metrics.breakeven}"
        )
        print(f"Win rate: {_format_percent(metrics.win_rate)}")
        print(f"Gross profit: {_format_float(metrics.gross_profit)}")
        print(f"Gross loss: {_format_float(metrics.gross_loss)}")
        print(f"Avg win: {_format_float(metrics.avg_win)}")
        print(f"Avg loss: {_format_float(metrics.avg_loss)}")
        print(f"Profit factor: {_format_float(metrics.profit_factor)}")
        print(f"Expectancy (avg pnl/trade): {_format_float(metrics.expectancy)}")
        print(f"Largest win: {_format_float(metrics.largest_win)}")
        print(f"Largest loss: {_format_float(metrics.largest_loss)}")
        print(f"Max drawdown: {_format_float(metrics.max_drawdown)}")
        print(f"Avg hold (min): {_format_float(metrics.avg_hold_minutes)}")
        print(f"Median hold (min): {_format_float(metrics.median_hold_minutes)}")
        if metrics.best_day:
            print(
                f"Best day: {metrics.best_day[0]} pnl={_format_float(metrics.best_day[1])}"
            )
        if metrics.worst_day:
            print(
                f"Worst day: {metrics.worst_day[0]} pnl={_format_float(metrics.worst_day[1])}"
            )
    if execution.attempts == 0:
        print("No entry attempts found.")
        return
    print("Execution Summary")
    print(
        f"Attempts: {execution.attempts}  Filled: {execution.filled}  Rejected: {execution.rejected}"
    )
    print(f"Reject rate: {_format_percent(execution.reject_rate)}")
    print(f"Avg slippage: {_format_percent(execution.avg_slippage_pct)}")
    print(f"Median slippage: {_format_percent(execution.median_slippage_pct)}")
    print(f"Avg fill time (s): {_format_float(execution.avg_fill_seconds)}")
    print(f"Median fill time (s): {_format_float(execution.median_fill_seconds)}")


def _bucket_label(minutes_since_midnight: int) -> str:
    hour = minutes_since_midnight // 60
    minute = minutes_since_midnight % 60
    return f"{hour:02d}:{minute:02d}"


def _print_time_buckets(trades: list[TradeRecord], bucket_minutes: int) -> None:
    completed = [trade for trade in trades if trade.exit_time and trade.pnl is not None]
    if not completed:
        return
    bucket_minutes = max(1, int(bucket_minutes))
    buckets: dict[str, list[TradeRecord]] = defaultdict(list)
    for trade in completed:
        entry = trade.entry_time
        minutes = entry.hour * 60 + entry.minute
        start = (minutes // bucket_minutes) * bucket_minutes
        buckets[_bucket_label(start)].append(trade)

    print(f"By Entry Time ({bucket_minutes}m buckets)")
    for bucket in sorted(buckets.keys()):
        items = buckets[bucket]
        wins = sum(1 for t in items if (t.pnl or 0.0) > 0)
        losses = sum(1 for t in items if (t.pnl or 0.0) < 0)
        total = len(items)
        net = sum(t.pnl or 0.0 for t in items)
        win_rate = _safe_div(wins, wins + losses)
        avg = net / total if total else 0.0
        print(
            f"{bucket}  trades={total} wins={wins} losses={losses} "
            f"win_rate={_format_percent(win_rate)} net_pnl={_format_float(net)} avg_pnl={_format_float(avg)}"
        )


def _print_scan_audit_summary(logger: TradeLogger, bucket_minutes: int) -> None:
    records = logger.fetch_scan_audit()
    if not records:
        return

    total = len(records)
    tradeable_total = sum(1 for record in records if record.is_tradeable)
    passed_total = sum(1 for record in records if record.passed_scan)
    passed_tradeable = sum(
        1 for record in records if record.is_tradeable and record.passed_scan
    )
    not_tradeable = total - tradeable_total
    pass_rate_tradeable = _safe_div(passed_tradeable, tradeable_total)

    print("Scan Audit (Top Gainers)")
    print(
        f"Rows: {total} tradeable={tradeable_total} not_tradeable={not_tradeable} "
        f"passed_scan={passed_total} passed_tradeable={passed_tradeable} "
        f"pass_rate_tradeable={_format_percent(pass_rate_tradeable)}"
    )

    failure_reasons = Counter(
        (record.reject_reason or "unknown")
        for record in records
        if record.is_tradeable and not record.passed_scan
    )
    if failure_reasons:
        print("Top Scan Reject Reasons (tradeable only)")
        for reason, count in failure_reasons.most_common(10):
            pct = count / tradeable_total if tradeable_total else 0.0
            print(f"{reason}: {count} ({pct * 100:.2f}%)")

    bucket_minutes = max(1, int(bucket_minutes))
    buckets: dict[str, list[ScanAuditRecord]] = defaultdict(list)
    for record in records:
        minutes = record.scan_time.hour * 60 + record.scan_time.minute
        start = (minutes // bucket_minutes) * bucket_minutes
        buckets[_bucket_label(start)].append(record)

    print(f"By Scan Time ({bucket_minutes}m buckets)")
    for bucket in sorted(buckets.keys()):
        items = buckets[bucket]
        bucket_tradeable = sum(1 for record in items if record.is_tradeable)
        bucket_passed = sum(1 for record in items if record.is_tradeable and record.passed_scan)
        bucket_pass_rate = _safe_div(bucket_passed, bucket_tradeable)
        bucket_reasons = Counter(
            (record.reject_reason or "unknown")
            for record in items
            if record.is_tradeable and not record.passed_scan
        )
        top_reason = bucket_reasons.most_common(1)[0][0] if bucket_reasons else "none"
        print(
            f"{bucket}  rows={len(items)} tradeable={bucket_tradeable} "
            f"passed_tradeable={bucket_passed} pass_rate_tradeable={_format_percent(bucket_pass_rate)} "
            f"top_reject={top_reason}"
        )


def _print_path_summary(logger: TradeLogger, trades: list[TradeRecord]) -> None:
    paths = logger.fetch_trade_paths()
    if not paths:
        return
    completed = {(t.symbol, t.entry_time.isoformat()): t for t in trades if t.exit_time and t.pnl is not None}
    mfe_values: list[float] = []
    mae_values: list[float] = []
    givebacks: list[float] = []
    hit_target = 0
    for path in paths:
        key = (path.symbol, path.entry_time.isoformat())
        trade = completed.get(key)
        if path.mfe_pct is not None:
            mfe_values.append(path.mfe_pct)
            if path.mfe_pct >= config.PROFIT_TARGET_PCT:
                hit_target += 1
        if path.mae_pct is not None:
            mae_values.append(path.mae_pct)
        if trade and trade.entry_price > 0 and trade.qty > 0 and trade.pnl is not None and path.mfe_pct is not None:
            realized_pct = trade.pnl / (trade.entry_price * trade.qty)
            givebacks.append(path.mfe_pct - realized_pct)

    print("Trade Path (MFE/MAE)")
    print(f"Samples: {len(paths)}")
    print(f"Avg MFE: {_format_percent(sum(mfe_values) / len(mfe_values) if mfe_values else None)}")
    print(f"Median MFE: {_format_percent(_median(mfe_values))}")
    print(f"Avg MAE: {_format_percent(sum(mae_values) / len(mae_values) if mae_values else None)}")
    print(f"Median MAE: {_format_percent(_median(mae_values))}")
    if givebacks:
        print(f"Avg giveback (MFE - realized): {_format_percent(sum(givebacks) / len(givebacks))}")
        print(f"Median giveback: {_format_percent(_median(givebacks))}")
    print(f"MFE >= target ({config.PROFIT_TARGET_PCT:.2%}): {hit_target}")


def _print_exit_updates(logger: TradeLogger, limit: int = 10) -> None:
    updates = logger.fetch_exit_updates(limit=limit)
    if not updates:
        return
    print(f"Recent Stop Updates (last {len(updates)})")
    for update in updates:
        old_stop = _format_float(update.old_stop_price) if update.old_stop_price is not None else "n/a"
        print(
            f"{update.timestamp.isoformat()} {update.symbol} "
            f"reason={update.reason} old_stop={old_stop} new_stop={update.new_stop_price:.2f} "
            f"peak={update.peak_price:.2f} current={update.current_price:.2f}"
        )


def run() -> None:
    parser = argparse.ArgumentParser(description="Generate a performance report from the trade log DB.")
    parser.add_argument(
        "--db",
        type=str,
        default=str(DEFAULT_DB_PATH),
        help=f"Path to SQLite DB (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--bucket-minutes",
        type=int,
        default=30,
        help="Entry-time bucket size in minutes for the time-of-day breakdown.",
    )
    args = parser.parse_args()

    logger = TradeLogger(Path(args.db))
    trades = logger.fetch_trades()
    attempts = logger.fetch_entry_attempts()
    metrics = _compute_metrics(trades)
    execution = _compute_execution_metrics(attempts)
    _print_report(metrics, execution)
    _print_scan_audit_summary(logger, args.bucket_minutes)
    _print_time_buckets(trades, args.bucket_minutes)
    _print_path_summary(logger, trades)
    _print_exit_updates(logger)


if __name__ == "__main__":
    run()
