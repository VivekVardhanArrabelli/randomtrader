"""Polygon-only preview runner."""

from __future__ import annotations

import math
import os
import time as time_module
from collections import Counter
from datetime import timedelta, time

from dotenv import load_dotenv
from pathlib import Path

from . import config
from .bars import atr as bars_atr, consolidation_breakout, vwap as bars_vwap
from .db import PREVIEW_DB_PATH, TradeLogger, TradeRecord
from .entry import EntryDecision, QuoteSnapshot, evaluate_entry
from .polygon_client import PolygonClient
from .positions import OpenPosition, PositionManager
from .scanner import MarketScanner, MarketSnapshot, scan_reject_reason
from .utils import EASTERN_TZ, is_within_market_window, now_eastern, parse_timestamp


def _log(message: str) -> None:
    timestamp = now_eastern().strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"[{timestamp}] {message}")


def _preview_equity() -> float:
    value = os.environ.get("PREVIEW_ACCOUNT_EQUITY")
    if value:
        try:
            return float(value)
        except ValueError:
            pass
    return config.PREVIEW_ACCOUNT_EQUITY


def _apply_preview_overrides() -> None:
    if not config.PREVIEW_RELAX_FILTERS:
        return
    config.MIN_GAIN_PCT = config.PREVIEW_MIN_GAIN_PCT
    config.MIN_REL_VOLUME = config.PREVIEW_MIN_REL_VOLUME
    config.MIN_DOLLAR_VOLUME = config.PREVIEW_MIN_DOLLAR_VOLUME
    config.MIN_QUOTE_SIZE = config.PREVIEW_MIN_QUOTE_SIZE
    config.MAX_BID_ASK_SPREAD_PCT = config.PREVIEW_MAX_BID_ASK_SPREAD_PCT
    config.ENFORCE_DAY_HIGH_BAND = config.PREVIEW_ENFORCE_DAY_HIGH_BAND
    config.MAX_PRICE = config.PREVIEW_MAX_PRICE
    config.MAX_LAST_TRADE_AGE_SECONDS = config.PREVIEW_MAX_LAST_TRADE_AGE_SECONDS
    config.ENFORCE_FUNDAMENTALS = config.PREVIEW_ENFORCE_FUNDAMENTALS


def _format_counter(counter: Counter) -> str:
    if not counter:
        return "none"
    return ", ".join(f"{key}={value}" for key, value in sorted(counter.items()))


def _debug_rejections(
    snapshots: list[MarketSnapshot], decisions: list[EntryDecision]
) -> None:
    if not config.PREVIEW_DEBUG_REJECTIONS:
        return
    scan_reasons: Counter[str] = Counter()
    scan_passed = 0
    now = now_eastern()
    market_open = time(9, 30, tzinfo=EASTERN_TZ)
    market_close = time(16, 0, tzinfo=EASTERN_TZ)
    market_open_now = is_within_market_window(now, market_open, market_close)
    for snapshot in snapshots:
        reject_reason, _ = scan_reject_reason(snapshot, now=now, market_open_now=market_open_now)
        if reject_reason is None:
            scan_passed += 1
        else:
            scan_reasons[reject_reason] += 1

    entry_reasons: Counter[str] = Counter()
    entry_passed = 0
    for decision in decisions:
        if decision.should_enter:
            entry_passed += 1
        else:
            entry_reasons[decision.reason] += 1

    _log(f"debug scan: passed={scan_passed} rejects={_format_counter(scan_reasons)}")
    _log(
        "debug entry: candidates="
        f"{len(decisions)} passed={entry_passed} rejects={_format_counter(entry_reasons)}"
    )


def _build_market_data(
    polygon: PolygonClient, tickers: list[dict]
) -> tuple[list[MarketSnapshot], dict[str, QuoteSnapshot]]:
    snapshots: list[MarketSnapshot] = []
    quotes: dict[str, QuoteSnapshot] = {}
    now = now_eastern()
    market_open = time(9, 30, tzinfo=EASTERN_TZ)
    market_close = time(16, 0, tzinfo=EASTERN_TZ)
    market_open_now = is_within_market_window(now, market_open, market_close)
    total_minutes = 6.5 * 60  # Regular session length.
    minutes_since_open = (now.hour * 60 + now.minute) - (9 * 60 + 30)
    if minutes_since_open < 1:
        minutes_since_open = 1
    if minutes_since_open > total_minutes:
        minutes_since_open = total_minutes
    time_fraction = (minutes_since_open / total_minutes) if market_open_now else 1.0

    for item in tickers:
        symbol = item.get("ticker")
        if not symbol:
            continue
        day = item.get("day") or {}
        minute = item.get("min") or {}
        prev_day = item.get("prevDay") or {}
        last_trade = item.get("lastTrade") or {}
        last_quote = item.get("lastQuote") or {}

        open_price = float(day.get("o") or 0.0)
        current_price = float(last_trade.get("p") or day.get("c") or minute.get("c") or 0.0)
        day_high = float(day.get("h") or 0.0)
        bid = float(last_quote.get("bp") or 0.0)
        ask = float(last_quote.get("ap") or 0.0)
        bid_size = float(last_quote.get("bs") or 0.0)
        ask_size = float(last_quote.get("as") or 0.0)
        last_trade_time = parse_timestamp(last_trade.get("t") or minute.get("t"))
        if config.PREVIEW_USE_SYNTHETIC_QUOTES and bid == 0.0 and ask == 0.0:
            bid = current_price
            ask = current_price

        day_volume = day.get("v")
        prev_volume = prev_day.get("v")
        day_volume_value = float(day_volume) if day_volume else 0.0
        relative_volume = 0.0
        if day_volume and prev_volume:
            prev_value = float(prev_volume)
            if prev_value > 0:
                relative_volume = float(day_volume) / prev_value
                if market_open_now and time_fraction > 0:
                    relative_volume = relative_volume / time_fraction
        dollar_volume = day_volume_value * current_price
        if market_open_now and time_fraction > 0:
            # Time-normalize dollar volume (pace) so early-session previews behave
            # similarly to the live scan.
            dollar_volume = dollar_volume / time_fraction

        market_cap = None
        shares_outstanding = None
        security_type = None
        if config.ENFORCE_FUNDAMENTALS or config.ENFORCE_SECURITY_TYPE_FILTER:
            fundamentals = polygon.get_fundamentals(symbol)
            security_type = fundamentals.get("type")
            if config.ENFORCE_FUNDAMENTALS:
                market_cap = polygon.extract_market_cap(fundamentals)
                shares_outstanding = polygon.extract_shares_outstanding(fundamentals)

        snapshots.append(
            MarketSnapshot(
                symbol=symbol,
                open_price=open_price,
                current_price=current_price,
                relative_volume=relative_volume,
                dollar_volume=dollar_volume,
                market_cap=market_cap,
                shares_outstanding=shares_outstanding,
                security_type=security_type,
                is_tradeable=True,
                last_trade_time=last_trade_time,
            )
        )
        quotes[symbol] = QuoteSnapshot(
            symbol=symbol,
            last_price=current_price,
            day_high=day_high,
            bid=bid,
            ask=ask,
            bid_size=bid_size,
            ask_size=ask_size,
            halted=False,
        )

    return snapshots, quotes


def _snapshot_last_price(snapshot: dict) -> float:
    day = snapshot.get("day") or {}
    last_trade = snapshot.get("lastTrade") or {}
    minute = snapshot.get("min") or {}
    last_price = float(last_trade.get("p") or day.get("c") or minute.get("c") or 0.0)
    return last_price


def _evaluate_entries(
    polygon: PolygonClient,
    snapshots: list[MarketSnapshot],
    quotes: dict[str, QuoteSnapshot],
    account_equity: float,
) -> list[EntryDecision]:
    scanner = MarketScanner(snapshots)
    scan_candidates = scanner.scan()
    decisions: list[EntryDecision] = []
    open_positions = 0
    for candidate in scan_candidates:
        quote = quotes.get(candidate.symbol)
        if quote is None:
            decisions.append(EntryDecision(candidate.symbol, False, "missing quote", 0.0))
            continue
        decision = evaluate_entry(
            candidate=candidate,
            quote=quote,
            account_equity=account_equity,
            open_positions=open_positions,
        )
        if decision.should_enter and (
            config.BAR_CONFIRMATION_ENABLED or config.STOP_DISTANCE_MODE == "atr"
        ):
            bars = None
            try:
                bars = polygon.get_minute_bars(candidate.symbol, config.BAR_LOOKBACK_MINUTES)
            except Exception as exc:
                bars = None
                if config.BAR_CONFIRMATION_ENABLED:
                    decision = EntryDecision(
                        candidate.symbol,
                        False,
                        f"bar confirmation failed: bar data error: {exc}",
                        0.0,
                    )
            if decision.should_enter and config.BAR_CONFIRMATION_ENABLED:
                ok, reason = _bar_confirms_entry(
                    polygon,
                    candidate.symbol,
                    quote,
                    bars=bars,
                )
                if not ok:
                    decision = EntryDecision(
                        candidate.symbol,
                        False,
                        f"bar confirmation failed: {reason}",
                        0.0,
                    )
            if decision.should_enter and config.STOP_DISTANCE_MODE in {"atr", "structure"} and bars:
                stop_distance = None
                if config.STOP_DISTANCE_MODE == "atr":
                    atr_value = bars_atr(bars, period=config.ATR_PERIOD)
                    if atr_value is not None and quote.ask > 0:
                        raw_pct = (atr_value * config.ATR_STOP_MULTIPLIER) / quote.ask
                        pct = min(
                            max(raw_pct, config.DYNAMIC_STOP_MIN_PCT),
                            config.DYNAMIC_STOP_MAX_PCT,
                        )
                        stop_distance = quote.ask * pct
                elif config.STOP_DISTANCE_MODE == "structure":
                    lookback = max(1, config.STRUCTURE_STOP_LOOKBACK_BARS)
                    window = bars[-(lookback + 1) : -1] if len(bars) > 1 else []
                    if window and quote.ask > 0:
                        stop_ref = min(bar.low for bar in window)
                        stop_ref *= 1 - config.STRUCTURE_STOP_BUFFER_PCT
                        raw_distance = quote.ask - stop_ref
                        if raw_distance > 0:
                            raw_pct = raw_distance / quote.ask
                            pct = min(
                                max(raw_pct, config.DYNAMIC_STOP_MIN_PCT),
                                config.DYNAMIC_STOP_MAX_PCT,
                            )
                            stop_distance = quote.ask * pct
                if stop_distance:
                    decision = evaluate_entry(
                        candidate=candidate,
                        quote=quote,
                        account_equity=account_equity,
                        open_positions=open_positions,
                        stop_distance=stop_distance,
                    )
        decisions.append(decision)
        if decision.should_enter:
            open_positions += 1
    return decisions


def _bar_confirms_entry(
    polygon: PolygonClient,
    symbol: str,
    quote: QuoteSnapshot,
    bars: list | None = None,
) -> tuple[bool, str]:
    if not config.BAR_CONFIRMATION_ENABLED:
        return True, "disabled"
    try:
        bars = bars or polygon.get_minute_bars(symbol, config.BAR_LOOKBACK_MINUTES)
    except Exception as exc:
        return False, f"bar data error: {exc}"
    if config.BAR_CONFIRMATION_METHOD == "vwap":
        value = bars_vwap(bars)
        if value is None:
            return False, "missing vwap"
        threshold = value * (1 + config.BAR_VWAP_MIN_PREMIUM_PCT)
        if quote.last_price < threshold:
            return False, "below vwap"
        return True, "vwap confirmed"
    if config.BAR_CONFIRMATION_METHOD == "consolidation_breakout":
        return consolidation_breakout(
            bars,
            consolidation_bars=config.BAR_CONSOLIDATION_BARS,
            max_range_pct=config.BAR_CONSOLIDATION_MAX_RANGE_PCT,
            breakout_buffer_pct=config.BAR_BREAKOUT_BUFFER_PCT,
            volume_multiplier=config.BAR_VOLUME_MULTIPLIER,
        )
    return False, "unknown bar confirmation method"


def _confirm_entry(
    polygon: PolygonClient, account_equity: float
) -> tuple[EntryDecision | None, QuoteSnapshot | None, list[MarketSnapshot], list[EntryDecision]]:
    required = max(1, config.CONFIRMATION_SNAPSHOTS)
    tickers = polygon.get_gainers_snapshot(config.GAINERS_LIMIT)
    if not tickers:
        _log("no gainers returned")
        return None, None, [], []
    initial_symbols = {
        item["ticker"] for item in tickers if item.get("ticker")
    }
    remaining_symbols: set[str] | None = None
    last_snapshots: list[MarketSnapshot] = []
    last_quotes: dict[str, QuoteSnapshot] = {}
    last_decisions: list[EntryDecision] = []

    for index in range(required):
        if index > 0:
            tickers = polygon.get_gainers_snapshot(config.GAINERS_LIMIT)
        if remaining_symbols is None:
            filtered = [item for item in tickers if item.get("ticker") in initial_symbols]
        else:
            filtered = [item for item in tickers if item.get("ticker") in remaining_symbols]
        snapshots, quotes = _build_market_data(polygon, filtered)
        decisions = _evaluate_entries(polygon, snapshots, quotes, account_equity)
        entry_symbols = {decision.symbol for decision in decisions if decision.should_enter}
        remaining_symbols = entry_symbols if remaining_symbols is None else remaining_symbols & entry_symbols
        _debug_rejections(snapshots, decisions)
        _log(
            f"confirm {index + 1}/{required}: snapshots={len(snapshots)} "
            f"entries={len(entry_symbols)}"
        )
        last_snapshots = snapshots
        last_quotes = quotes
        last_decisions = decisions
        if not remaining_symbols:
            return None, None, last_snapshots, last_decisions
        if index < required - 1 and config.CONFIRMATION_INTERVAL_SECONDS > 0:
            time_module.sleep(config.CONFIRMATION_INTERVAL_SECONDS)

    scanner = MarketScanner(last_snapshots)
    ranked = scanner.scan()
    for candidate in ranked:
        if candidate.symbol not in remaining_symbols:
            continue
        for decision in last_decisions:
            if decision.symbol == candidate.symbol and decision.should_enter:
                return decision, last_quotes.get(candidate.symbol), last_snapshots, last_decisions
    return None, None, last_snapshots, last_decisions


def run() -> None:
    env_path = Path(__file__).with_name(".env")
    load_dotenv(env_path, override=True)
    _apply_preview_overrides()
    polygon = PolygonClient.from_env()
    equity = _preview_equity()
    entry, quote, snapshots, decisions = _confirm_entry(polygon, equity)
    candidates = [decision for decision in decisions if decision.should_enter]

    if not snapshots:
        return
    _log(f"preview equity={equity:.2f}")
    _log(f"decisions={len(decisions)} entries={len(candidates)}")

    if config.PREVIEW_SHOW_TOP > 0:
        scanner = MarketScanner(snapshots)
        ranked = scanner.scan()[: config.PREVIEW_SHOW_TOP]
        for candidate in ranked:
            _log(
                f"candidate {candidate.symbol} gain={candidate.gain_pct:.2%} "
                f"relvol={candidate.relative_volume:.2f} score={candidate.score:.2f}"
            )

    if entry is None:
        _log("no confirmed entries")
        return
    if quote is None or quote.ask <= 0:
        _log(f"would enter {entry.symbol} qty=0 reason=missing ask")
        return
    risk_qty = math.floor(entry.position_size)
    max_qty = math.floor(equity / quote.ask) if quote.ask else 0
    qty = min(risk_qty, max_qty)
    if qty <= 0:
        _log(f"would enter {entry.symbol} qty=0 reason=position size too small")
        return
    _log(f"would enter {entry.symbol} qty={qty} ask={quote.ask:.2f} reason={entry.reason}")
    if not config.PREVIEW_SIMULATE_EXIT:
        return
    market_open = time(9, 30, tzinfo=EASTERN_TZ)
    market_close = time(16, 0, tzinfo=EASTERN_TZ)
    if not config.PREVIEW_IGNORE_MARKET_HOURS:
        if not (market_open <= now_eastern().timetz() <= market_close):
            _log("market closed; skipping simulated exit")
            return
    logger = TradeLogger(PREVIEW_DB_PATH) if config.PREVIEW_LOG_TRADES else None
    entry_time = now_eastern()
    if logger:
        logger.log_trades(
            [
                TradeRecord(
                    entry_time=entry_time,
                    symbol=entry.symbol,
                    entry_price=quote.ask,
                    qty=qty,
                    exit_price=None,
                    exit_time=None,
                    pnl=None,
                    exit_reason=None,
                )
            ]
        )
    _simulate_exit(polygon, entry.symbol, quote.ask, qty, entry_time, logger, entry.stop_distance)


def _simulate_exit(
    polygon: PolygonClient,
    symbol: str,
    entry_price: float,
    qty: int,
    entry_time: datetime,
    logger: TradeLogger | None,
    stop_distance: float,
) -> None:
    position_manager = PositionManager()
    profit_target = entry_price * (1 + config.PROFIT_TARGET_PCT)
    stop_loss = entry_price * (1 - config.STOP_LOSS_PCT)
    if stop_distance > 0:
        stop_loss = entry_price - stop_distance
    position_manager.add_position(
        OpenPosition(
            symbol=symbol,
            entry_price=entry_price,
            qty=qty,
            entry_time=entry_time,
            profit_target=profit_target,
            stop_loss=stop_loss,
        )
    )
    start_time = now_eastern()
    end_time = None
    if config.PREVIEW_MAX_MONITOR_SECONDS > 0:
        end_time = start_time + timedelta(seconds=config.PREVIEW_MAX_MONITOR_SECONDS)
    _log(f"monitoring {symbol} for exit signals")
    last_price = 0.0
    while True:
        if end_time and now_eastern() >= end_time:
            _log("preview monitor timeout")
            if logger and last_price > 0:
                pnl = (last_price - entry_price) * qty
                logger.log_trade_exit(
                    symbol=symbol,
                    exit_price=last_price,
                    exit_time=now_eastern(),
                    pnl=pnl,
                    exit_reason="timeout",
                )
                _log(
                    f"exit {symbol} reason=timeout entry={entry_price:.2f} "
                    f"exit={last_price:.2f} pnl={pnl:.2f}"
                )
            return
        snapshot = polygon.get_ticker_snapshot(symbol)
        last_price = _snapshot_last_price(snapshot)
        if last_price <= 0:
            time_module.sleep(config.PREVIEW_POLL_INTERVAL_SECONDS)
            continue
        decision = position_manager.evaluate_exit(symbol, last_price, now_eastern())
        if decision.should_exit:
            pnl = (decision.exit_price - entry_price) * qty
            if logger:
                logger.log_trade_exit(
                    symbol=symbol,
                    exit_price=decision.exit_price,
                    exit_time=now_eastern(),
                    pnl=pnl,
                    exit_reason=decision.reason,
                )
            _log(
                f"exit {symbol} reason={decision.reason} entry={entry_price:.2f} "
                f"exit={decision.exit_price:.2f} pnl={pnl:.2f}"
            )
            position_manager.remove_position(symbol)
            return
        time_module.sleep(config.PREVIEW_POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    run()
