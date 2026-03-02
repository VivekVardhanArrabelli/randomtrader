"""Live paper trading runner."""

from __future__ import annotations

import math
import os
import time as time_module
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, time, timedelta

from dotenv import load_dotenv
from pathlib import Path

from . import config
from .alpaca_client import AlpacaClient
from .bars import atr as bars_atr, consolidation_breakout, vwap as bars_vwap
from .db import (
    DecisionFeatureRecord,
    EntryAttemptRecord,
    ExitUpdateRecord,
    OrderEventRecord,
    ScanAuditRecord,
    ScanRecord,
    TradeLogger,
    TradePathRecord,
)
from .entry import EntryDecision, QuoteSnapshot, evaluate_entry
from .main import TraderEngine
from .polygon_client import PolygonClient
from .risk import (
    count_consecutive_losses,
    evaluate_daily_risk,
    last_loss_time,
    within_trading_window,
)
from .scanner import MarketSnapshot, scan_reject_reason
from .utils import AccountSnapshot, EASTERN_TZ, is_within_market_window, now_eastern, parse_timestamp


def _log(message: str) -> None:
    timestamp = now_eastern().strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"[{timestamp}] {message}")


def _acquire_single_instance_lock() -> object | None:
    """Best-effort single instance lock to prevent duplicate live traders.

    Duplicate runners can hammer Alpaca and/or double-trade. We use an OS-level
    advisory lock so it releases automatically if the process dies.
    """

    try:
        import fcntl  # type: ignore
    except Exception:
        return None

    lock_path = Path(__file__).with_name("logs") / "live.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(lock_path, "w", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None
    try:
        handle.write(str(os.getpid()))
        handle.flush()
    except Exception:
        # Lock is still held even if we fail to write the pid.
        pass
    return handle


def _parse_scan_time(value: str) -> time:
    hour_str, minute_str = value.split(":")
    return time(int(hour_str), int(minute_str), tzinfo=EASTERN_TZ)


def _account_snapshot(alpaca: AlpacaClient) -> AccountSnapshot:
    account = alpaca.get_account()
    equity = float(account["equity"])
    last_equity = float(account.get("last_equity", equity))
    return AccountSnapshot(
        equity=equity,
        cash=float(account["cash"]),
        buying_power=float(account["buying_power"]),
        day_pl=equity - last_equity,
    )


@dataclass
class WatchlistItem:
    symbol: str
    added_time: datetime
    last_seen_time: datetime
    score: float
    gainer_rank: int | None = None
    scan_rank: int | None = None
    peak_price: float = 0.0
    last_price: float = 0.0
    dip_seen: bool = False
    below_vwap_seen: bool = False
    last_vwap: float | None = None
    last_atr: float | None = None
    last_bars_at: datetime | None = None


class LiveTrader:
    def __init__(self) -> None:
        env_path = Path(__file__).with_name(".env")
        load_dotenv(env_path, override=True)

        self._lock_handle = _acquire_single_instance_lock()
        if self._lock_handle is None:
            _log("another momentum_trader.live instance is already running; exiting")
            raise SystemExit(0)

        self.alpaca = AlpacaClient.from_env()
        self.polygon = PolygonClient.from_env()
        self.logger = TradeLogger()
        self.engine = TraderEngine(self.logger)
        self.fundamentals_cache: dict[str, dict] = {}
        self.tradable_symbols = self._load_tradable_symbols()
        self.scan_time = _parse_scan_time(config.SCAN_TIME)
        self.scan_start_time = _parse_scan_time(config.SCAN_START_TIME)
        self.scan_end_time = _parse_scan_time(config.SCAN_END_TIME)
        self.exit_time = _parse_scan_time(config.LIVE_EXIT_TIME) if getattr(config, "LIVE_EXIT_TIME", "") else None
        self.last_scan_date: str | None = None
        self.last_scan_at: datetime | None = None
        self.position_peaks: dict[str, float] = {}
        self.position_stops: dict[str, float] = {}
        self.rate_limit_backoff_seconds: int = 0
        self.watchlist: dict[str, WatchlistItem] = {}
        self.last_watchlist_monitor_at: datetime | None = None

        _log(
            "config "
            f"scan_mode={config.SCAN_MODE} start={config.SCAN_START_TIME} end={config.SCAN_END_TIME} "
            f"interval_s={config.SCAN_INTERVAL_SECONDS} "
            f"profit_target={config.PROFIT_TARGET_PCT:.2%} stop_loss={config.STOP_LOSS_PCT:.2%} "
            f"stop_mode={config.STOP_DISTANCE_MODE} bar_confirm={config.BAR_CONFIRMATION_ENABLED}:{config.BAR_CONFIRMATION_METHOD} "
            f"entry_style={config.ENTRY_STYLE}"
        )

    def _load_tradable_symbols(self) -> set[str]:
        # This is a "nice to have" pre-filter, but the live runner must be able
        # to start even if Alpaca is temporarily rate limiting us (429). Without
        # this retry, launchd can get stuck in a crash/restart loop at open.
        backoff_seconds = max(1, int(config.ALPACA_RATE_LIMIT_BACKOFF_INITIAL_SECONDS))
        max_backoff_seconds = max(backoff_seconds, int(config.ALPACA_RATE_LIMIT_BACKOFF_MAX_SECONDS))
        for attempt in range(1, 11):
            try:
                assets = self.alpaca.get_assets()
                return {
                    asset["symbol"]
                    for asset in assets
                    if asset.get("tradable") and asset.get("status") == "active"
                }
            except RuntimeError as exc:
                message = str(exc)
                if "Alpaca API error 429" not in message and "rate limit" not in message.lower():
                    raise
                sleep_seconds = min(backoff_seconds, max_backoff_seconds)
                _log(f"alpaca rate limit while loading assets; retry {attempt}/10 in {sleep_seconds}s")
                time_module.sleep(sleep_seconds)
                backoff_seconds = min(max_backoff_seconds, backoff_seconds * 2)
        _log("alpaca assets still rate-limited; continuing without tradable-symbol cache")
        return set()

    def _get_fundamentals(self, symbol: str) -> dict:
        cached = self.fundamentals_cache.get(symbol)
        if cached is not None:
            return cached
        fundamentals = self.polygon.get_fundamentals(symbol)
        self.fundamentals_cache[symbol] = fundamentals
        return fundamentals

    def _build_market_data(
        self, symbols: list[str]
    ) -> tuple[list[MarketSnapshot], dict[str, QuoteSnapshot]]:
        snapshots = self.alpaca.get_snapshots(symbols)
        market_snapshots: list[MarketSnapshot] = []
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

        for symbol in symbols:
            snapshot = snapshots.get(symbol)
            if not snapshot:
                continue
            daily_bar = snapshot.get("dailyBar") or {}
            prev_daily_bar = snapshot.get("prevDailyBar") or {}
            latest_trade = snapshot.get("latestTrade") or {}
            latest_quote = snapshot.get("latestQuote") or {}

            open_price = float(daily_bar.get("o") or 0.0)
            current_price = float(latest_trade.get("p") or daily_bar.get("c") or 0.0)
            day_high = float(daily_bar.get("h") or 0.0)
            bid = float(latest_quote.get("bp") or 0.0)
            ask = float(latest_quote.get("ap") or 0.0)
            bid_size = float(latest_quote.get("bs") or 0.0)
            ask_size = float(latest_quote.get("as") or 0.0)
            last_trade_time = parse_timestamp(latest_trade.get("t"))

            day_volume = daily_bar.get("v")
            prev_volume = prev_daily_bar.get("v")
            day_volume_value = float(day_volume) if day_volume else 0.0
            relative_volume = 0.0
            if day_volume and prev_volume:
                prev_value = float(prev_volume)
                if prev_value > 0:
                    # Polygon/Alpaca daily bars represent today's volume so far. Normalize by
                    # elapsed session time so we can qualify early movers (e.g., 09:45 ET).
                    relative_volume = float(day_volume) / prev_value
                    if market_open_now and time_fraction > 0:
                        relative_volume = relative_volume / time_fraction
            dollar_volume = day_volume_value * current_price
            if market_open_now and time_fraction > 0:
                # Use time-normalized dollar volume (pace) so early-session scans
                # don't discard legitimate movers purely because the day just started.
                dollar_volume = dollar_volume / time_fraction

            fundamentals = self._get_fundamentals(symbol)
            market_cap = self.polygon.extract_market_cap(fundamentals)
            shares_outstanding = self.polygon.extract_shares_outstanding(fundamentals)
            security_type = fundamentals.get("type")
            is_tradeable = (symbol in self.tradable_symbols) if self.tradable_symbols else True

            market_snapshots.append(
                MarketSnapshot(
                    symbol=symbol,
                    open_price=open_price,
                    current_price=current_price,
                    relative_volume=relative_volume,
                    dollar_volume=dollar_volume,
                    market_cap=market_cap,
                    shares_outstanding=shares_outstanding,
                    security_type=security_type,
                    is_tradeable=is_tradeable,
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
        return market_snapshots, quotes

    def _quote_allows_entry(self, quote: QuoteSnapshot, enforce_day_high_band: bool | None = None) -> bool:
        if enforce_day_high_band is None:
            enforce_day_high_band = config.ENFORCE_DAY_HIGH_BAND
        if quote.bid <= 0 or quote.ask <= 0:
            return False
        if enforce_day_high_band and not quote.within_day_high_band:
            return False
        if min(quote.bid_size, quote.ask_size) < config.MIN_QUOTE_SIZE:
            return False
        if quote.bid_ask_spread_pct >= config.MAX_BID_ASK_SPREAD_PCT:
            return False
        return True

    def _bar_confirms_entry(
        self,
        symbol: str,
        quote: QuoteSnapshot,
        bars: list | None = None,
    ) -> tuple[bool, str, dict]:
        if not config.BAR_CONFIRMATION_ENABLED:
            return True, "disabled", {}
        try:
            bars = bars or self.polygon.get_minute_bars(symbol, config.BAR_LOOKBACK_MINUTES)
        except Exception as exc:
            return False, f"bar data error: {exc}", {}
        if config.BAR_CONFIRMATION_METHOD == "vwap":
            value = bars_vwap(bars)
            if value is None:
                return False, "missing vwap", {}
            threshold = value * (1 + config.BAR_VWAP_MIN_PREMIUM_PCT)
            premium_pct = None
            if value > 0:
                premium_pct = (quote.last_price - value) / value
            if quote.last_price < threshold:
                return (
                    False,
                    "below vwap",
                    {"bar_vwap": value, "bar_vwap_premium_pct": premium_pct, "bar_threshold": threshold},
                )
            return (
                True,
                "vwap confirmed",
                {"bar_vwap": value, "bar_vwap_premium_pct": premium_pct, "bar_threshold": threshold},
            )
        if config.BAR_CONFIRMATION_METHOD == "consolidation_breakout":
            ok, reason = consolidation_breakout(
                bars,
                consolidation_bars=config.BAR_CONSOLIDATION_BARS,
                max_range_pct=config.BAR_CONSOLIDATION_MAX_RANGE_PCT,
                breakout_buffer_pct=config.BAR_BREAKOUT_BUFFER_PCT,
                volume_multiplier=config.BAR_VOLUME_MULTIPLIER,
            )
            return ok, reason, {}
        return False, "unknown bar confirmation method", {}

    def _format_counter(self, counter: Counter[str]) -> str:
        if not counter:
            return "none"
        return ", ".join(f"{key}={value}" for key, value in sorted(counter.items()))

    def _debug_rejections(
        self, snapshots: list[MarketSnapshot], decisions: list[EntryDecision]
    ) -> None:
        if not config.LIVE_DEBUG:
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

        _log(f"debug scan: passed={scan_passed} rejects={self._format_counter(scan_reasons)}")
        _log(
            "debug entry: candidates="
            f"{len(decisions)} passed={entry_passed} rejects={self._format_counter(entry_reasons)}"
        )

    def _log_top_decisions(
        self,
        candidates: list,
        decisions: list[EntryDecision],
        quotes: dict[str, QuoteSnapshot],
    ) -> None:
        if not config.LIVE_DEBUG or config.LIVE_LOG_TOP_DECISIONS <= 0:
            return
        decision_by_symbol = {decision.symbol: decision for decision in decisions}
        top = candidates[: config.LIVE_LOG_TOP_DECISIONS]
        for candidate in top:
            decision = decision_by_symbol.get(candidate.symbol)
            quote = quotes.get(candidate.symbol)
            ask = quote.ask if quote else 0.0
            spread_pct = quote.bid_ask_spread_pct if quote else 0.0
            min_qsz = min(quote.bid_size, quote.ask_size) if quote else 0.0
            stop_pct = None
            if decision and ask > 0 and decision.stop_distance > 0:
                stop_pct = decision.stop_distance / ask
            stop_str = f"{stop_pct:.2%}" if stop_pct is not None else "n/a"
            reason = decision.reason if decision else "missing decision"
            enter = decision.should_enter if decision else False
            _log(
                f"decision {candidate.symbol} enter={enter} reason={reason} "
                f"gain={candidate.gain_pct:.2%} relvol={candidate.relative_volume:.2f} score={candidate.score:.2f} "
                f"ask={ask:.2f} spread={spread_pct:.2%} qsz={min_qsz:.0f} stop={stop_str}"
            )

    def _persist_decision_features(
        self,
        scan_time: datetime,
        confirm_step: int,
        gainers: list[str],
        market_snapshots: list[MarketSnapshot],
        candidates: list,
        decisions: list[EntryDecision],
        quotes: dict[str, QuoteSnapshot],
        extras: dict[str, dict],
    ) -> None:
        if not config.LIVE_LOG_SCANS:
            return
        gainer_ranks = {symbol: index + 1 for index, symbol in enumerate(gainers)}
        snapshot_by_symbol = {snapshot.symbol: snapshot for snapshot in market_snapshots}
        decision_by_symbol = {decision.symbol: decision for decision in decisions}
        records: list[DecisionFeatureRecord] = []

        for scan_rank, candidate in enumerate(candidates, start=1):
            symbol = candidate.symbol
            decision = decision_by_symbol.get(symbol) or EntryDecision(symbol, False, "missing decision", 0.0)
            quote = quotes.get(symbol)
            snapshot = snapshot_by_symbol.get(symbol)
            extra = extras.get(symbol, {})

            ask = quote.ask if quote else None
            bid = quote.bid if quote else None
            spread_pct = quote.bid_ask_spread_pct if quote else None
            within_day_high_band = quote.within_day_high_band if quote else None
            bid_size = quote.bid_size if quote else None
            ask_size = quote.ask_size if quote else None
            day_high = quote.day_high if quote else None

            last_trade_age_s = None
            if snapshot and snapshot.last_trade_time:
                last_trade_age_s = (scan_time - snapshot.last_trade_time).total_seconds()
                if last_trade_age_s < 0:
                    last_trade_age_s = 0.0

            stop_distance = decision.stop_distance if decision.stop_distance > 0 else None
            stop_pct = None
            if stop_distance is not None and ask and ask > 0:
                stop_pct = stop_distance / ask

            bar_confirm_ok = None
            if "bar_confirm_ok" in extra:
                bar_confirm_ok = bool(extra.get("bar_confirm_ok"))
            bar_confirm_reason = extra.get("bar_confirm_reason") or extra.get("bar_error")

            records.append(
                DecisionFeatureRecord(
                    scan_time=scan_time,
                    confirm_step=confirm_step,
                    symbol=symbol,
                    gainer_rank=gainer_ranks.get(symbol),
                    scan_rank=scan_rank,
                    should_enter=decision.should_enter,
                    reason=decision.reason,
                    score=getattr(candidate, "score", None),
                    gain_pct=getattr(candidate, "gain_pct", None),
                    relative_volume=getattr(candidate, "relative_volume", None),
                    dollar_volume=snapshot.dollar_volume if snapshot else None,
                    current_price=snapshot.current_price if snapshot else None,
                    bid=bid,
                    ask=ask,
                    spread_pct=spread_pct,
                    bid_size=bid_size,
                    ask_size=ask_size,
                    day_high=day_high,
                    within_day_high_band=within_day_high_band,
                    last_trade_age_s=last_trade_age_s,
                    market_cap=snapshot.market_cap if snapshot else None,
                    shares_outstanding=snapshot.shares_outstanding if snapshot else None,
                    security_type=snapshot.security_type if snapshot else None,
                    bar_method=config.BAR_CONFIRMATION_METHOD if config.BAR_CONFIRMATION_ENABLED else None,
                    bar_confirm_ok=bar_confirm_ok,
                    bar_confirm_reason=bar_confirm_reason,
                    bar_vwap=extra.get("bar_vwap"),
                    bar_vwap_premium_pct=extra.get("bar_vwap_premium_pct"),
                    atr=extra.get("atr"),
                    atr_pct=extra.get("atr_pct"),
                    stop_distance=stop_distance,
                    stop_pct=stop_pct,
                    position_size=decision.position_size if decision.position_size > 0 else None,
                )
            )

        if not records:
            return
        try:
            self.logger.log_decision_features(records)
        except Exception as exc:
            if config.LIVE_DEBUG:
                _log(f"decision feature log failed: {exc}")

    def _persist_scan_audit(
        self,
        *,
        scan_time: datetime,
        gainers: list[str],
        market_snapshots: list[MarketSnapshot],
    ) -> None:
        if not config.LIVE_LOG_SCANS:
            return

        snapshot_by_symbol = {snapshot.symbol: snapshot for snapshot in market_snapshots}
        market_open = time(9, 30, tzinfo=EASTERN_TZ)
        market_close = time(16, 0, tzinfo=EASTERN_TZ)
        market_open_now = is_within_market_window(scan_time, market_open, market_close)

        records: list[ScanAuditRecord] = []
        for rank, symbol in enumerate(gainers, start=1):
            is_tradeable = (symbol in self.tradable_symbols) if self.tradable_symbols else True
            snapshot = snapshot_by_symbol.get(symbol)

            if not is_tradeable:
                records.append(
                    ScanAuditRecord(
                        scan_time=scan_time,
                        symbol=symbol,
                        gainer_rank=rank,
                        is_tradeable=False,
                        security_type=None,
                        open_price=None,
                        current_price=None,
                        gain_pct=None,
                        relative_volume=None,
                        dollar_volume=None,
                        market_cap=None,
                        shares_outstanding=None,
                        last_trade_time=None,
                        last_trade_age_s=None,
                        passed_scan=False,
                        reject_reason="not_tradeable",
                    )
                )
                continue

            if snapshot is None:
                records.append(
                    ScanAuditRecord(
                        scan_time=scan_time,
                        symbol=symbol,
                        gainer_rank=rank,
                        is_tradeable=True,
                        security_type=None,
                        open_price=None,
                        current_price=None,
                        gain_pct=None,
                        relative_volume=None,
                        dollar_volume=None,
                        market_cap=None,
                        shares_outstanding=None,
                        last_trade_time=None,
                        last_trade_age_s=None,
                        passed_scan=False,
                        reject_reason="missing_snapshot",
                    )
                )
                continue

            reject_reason, last_trade_age_s = scan_reject_reason(
                snapshot,
                now=scan_time,
                market_open_now=market_open_now,
            )
            passed_scan = reject_reason is None
            records.append(
                ScanAuditRecord(
                    scan_time=scan_time,
                    symbol=symbol,
                    gainer_rank=rank,
                    is_tradeable=snapshot.is_tradeable,
                    security_type=snapshot.security_type,
                    open_price=snapshot.open_price,
                    current_price=snapshot.current_price,
                    gain_pct=snapshot.gain_pct,
                    relative_volume=snapshot.relative_volume,
                    dollar_volume=snapshot.dollar_volume,
                    market_cap=snapshot.market_cap,
                    shares_outstanding=snapshot.shares_outstanding,
                    last_trade_time=snapshot.last_trade_time,
                    last_trade_age_s=last_trade_age_s,
                    passed_scan=passed_scan,
                    reject_reason=reject_reason,
                )
            )

        try:
            self.logger.log_scan_audit(records)
        except Exception as exc:
            if config.LIVE_DEBUG:
                _log(f"scan audit log failed: {exc}")

    def _safe_log_order_event(self, record: OrderEventRecord) -> None:
        if not config.LIVE_LOG_SCANS:
            return
        try:
            self.logger.log_order_event(record)
        except Exception as exc:
            if config.LIVE_DEBUG:
                _log(f"order event log failed: {exc}")

    def _safe_log_exit_update(self, record: ExitUpdateRecord) -> None:
        if not config.LIVE_LOG_SCANS:
            return
        try:
            self.logger.log_exit_update(record)
        except Exception as exc:
            if config.LIVE_DEBUG:
                _log(f"exit update log failed: {exc}")

    def _safe_log_trade_path(self, record: TradePathRecord) -> None:
        if not config.LIVE_LOG_SCANS:
            return
        try:
            self.logger.log_trade_path(record)
        except Exception as exc:
            if config.LIVE_DEBUG:
                _log(f"trade path log failed: {exc}")

    def _build_quotes(self, symbols: list[str]) -> dict[str, QuoteSnapshot]:
        if not symbols:
            return {}
        snapshots = self.alpaca.get_snapshots(symbols)
        quotes: dict[str, QuoteSnapshot] = {}
        for symbol in symbols:
            snapshot = snapshots.get(symbol)
            if not snapshot:
                continue
            latest_quote = snapshot.get("latestQuote") or {}
            latest_trade = snapshot.get("latestTrade") or {}
            daily_bar = snapshot.get("dailyBar") or {}
            bid = float(latest_quote.get("bp") or 0.0)
            ask = float(latest_quote.get("ap") or 0.0)
            bid_size = float(latest_quote.get("bs") or 0.0)
            ask_size = float(latest_quote.get("as") or 0.0)
            last_price = float(latest_trade.get("p") or daily_bar.get("c") or 0.0)
            day_high = float(daily_bar.get("h") or 0.0)
            quotes[symbol] = QuoteSnapshot(
                symbol=symbol,
                last_price=last_price,
                day_high=day_high,
                bid=bid,
                ask=ask,
                bid_size=bid_size,
                ask_size=ask_size,
                halted=False,
            )
        return quotes

    def _update_watchlist(self, scan_time: datetime, gainers: list[str], candidates: list, quotes: dict[str, QuoteSnapshot]) -> None:
        if config.ENTRY_STYLE != "dip_watchlist":
            return
        if config.WATCHLIST_MAX_SYMBOLS <= 0:
            self.watchlist.clear()
            return
        keep = candidates[: config.WATCHLIST_MAX_SYMBOLS]
        keep_symbols = {candidate.symbol for candidate in keep}
        gainer_ranks = {symbol: index + 1 for index, symbol in enumerate(gainers)}

        for scan_rank, candidate in enumerate(keep, start=1):
            symbol = candidate.symbol
            quote = quotes.get(symbol)
            price = 0.0
            if quote is not None:
                price = quote.last_price if quote.last_price > 0 else quote.ask
            item = self.watchlist.get(symbol)
            if item is None:
                item = WatchlistItem(
                    symbol=symbol,
                    added_time=scan_time,
                    last_seen_time=scan_time,
                    score=float(getattr(candidate, "score", 0.0) or 0.0),
                    gainer_rank=gainer_ranks.get(symbol),
                    scan_rank=scan_rank,
                    peak_price=price,
                    last_price=price,
                )
                self.watchlist[symbol] = item
                if config.LIVE_DEBUG:
                    _log(f"watchlist add {symbol} rank={scan_rank} score={item.score:.2f}")
            else:
                item.last_seen_time = scan_time
                item.score = float(getattr(candidate, "score", item.score) or item.score)
                item.gainer_rank = gainer_ranks.get(symbol)
                item.scan_rank = scan_rank
                if price > 0:
                    item.last_price = price
                    if price > item.peak_price:
                        item.peak_price = price
                        item.dip_seen = False
                        item.below_vwap_seen = False

        ttl = timedelta(minutes=max(1, int(config.WATCHLIST_TTL_MINUTES)))
        removed: list[str] = []
        for symbol, item in list(self.watchlist.items()):
            if symbol in keep_symbols:
                continue
            if scan_time - item.last_seen_time > ttl:
                removed.append(symbol)
                del self.watchlist[symbol]
        if removed and config.LIVE_DEBUG:
            _log(f"watchlist removed stale: {','.join(sorted(removed))}")

    def _get_cached_bars_metrics(self, item: WatchlistItem, now: datetime) -> tuple[float | None, float | None]:
        if item.last_bars_at is not None:
            age = (now - item.last_bars_at).total_seconds()
            if age >= 0 and age < max(1, int(config.DIP_VWAP_CACHE_SECONDS)):
                return item.last_vwap, item.last_atr
        bars = self.polygon.get_minute_bars(item.symbol, config.DIP_VWAP_LOOKBACK_MINUTES)
        item.last_bars_at = now
        item.last_vwap = bars_vwap(bars)
        item.last_atr = bars_atr(bars, period=config.ATR_PERIOD)
        return item.last_vwap, item.last_atr

    def _dip_reclaim_trigger(self, item: WatchlistItem, quote: QuoteSnapshot) -> tuple[bool, dict]:
        now = now_eastern()
        current_price = quote.last_price if quote.last_price > 0 else quote.ask
        if current_price <= 0:
            return False, {}
        item.last_price = current_price

        # Reset dip state on new highs.
        if current_price > item.peak_price:
            item.peak_price = current_price
            item.dip_seen = False
            item.below_vwap_seen = False

        pullback_pct = 0.0
        if item.peak_price > 0:
            pullback_pct = (item.peak_price - current_price) / item.peak_price
        if pullback_pct >= config.DIP_MIN_PULLBACK_PCT and pullback_pct <= config.DIP_MAX_PULLBACK_PCT:
            item.dip_seen = True

        vwap, atr_value = self._get_cached_bars_metrics(item, now)
        if vwap is None or vwap <= 0:
            return False, {"pullback_pct": pullback_pct}

        if config.DIP_REQUIRE_BELOW_VWAP and current_price <= vwap:
            item.below_vwap_seen = True

        reclaim_threshold = vwap * (1 + config.DIP_RECLAIM_VWAP_BUFFER_PCT)
        should_trigger = (
            item.dip_seen
            and (not config.DIP_REQUIRE_BELOW_VWAP or item.below_vwap_seen)
            and current_price >= reclaim_threshold
        )
        return should_trigger, {
            "pullback_pct": pullback_pct,
            "dip_seen": item.dip_seen,
            "below_vwap_seen": item.below_vwap_seen,
            "peak_price": item.peak_price,
            "current_price": current_price,
            "vwap": vwap,
            "reclaim_threshold": reclaim_threshold,
            "atr": atr_value,
        }

    def _maybe_trade_watchlist(self) -> None:
        if config.ENTRY_STYLE != "dip_watchlist":
            return
        now = now_eastern()
        if now.timetz() < self.scan_start_time or now.timetz() > self.scan_end_time:
            return
        if not within_trading_window(now):
            return
        if self.logger.has_trade_for_date(now.date().isoformat()):
            return
        if self.alpaca.get_positions():
            return
        if self._cooldown_active():
            return
        if not self.watchlist:
            return
        if self.last_watchlist_monitor_at is not None:
            interval = timedelta(seconds=max(1, int(config.WATCHLIST_MONITOR_INTERVAL_SECONDS)))
            if now - self.last_watchlist_monitor_at < interval:
                return
        self.last_watchlist_monitor_at = now

        items = sorted(self.watchlist.values(), key=lambda item: item.score, reverse=True)
        symbols = [item.symbol for item in items[: config.WATCHLIST_MAX_SYMBOLS]]
        quotes = self._build_quotes(symbols)

        for symbol in symbols:
            item = self.watchlist.get(symbol)
            quote = quotes.get(symbol)
            if item is None or quote is None:
                continue
            triggered, ctx = self._dip_reclaim_trigger(item, quote)
            if not triggered:
                continue
            _log(
                f"dip trigger {symbol} pullback={ctx.get('pullback_pct', 0.0):.2%} "
                f"vwap={ctx.get('vwap', 0.0):.2f} px={ctx.get('current_price', 0.0):.2f} "
                f"peak={ctx.get('peak_price', 0.0):.2f}"
            )
            self._safe_log_order_event(
                OrderEventRecord(
                    timestamp=now_eastern(),
                    symbol=symbol,
                    order_id=None,
                    event="dip_trigger",
                    side=None,
                    order_type=None,
                    order_class=None,
                    qty=None,
                    limit_price=None,
                    stop_price=None,
                    take_profit_price=None,
                    filled_qty=None,
                    filled_avg_price=None,
                    status=None,
                    signal_price=quote.ask if quote.ask > 0 else quote.last_price,
                    slippage_pct=None,
                    fill_seconds=None,
                    reject_reason=None,
                    notes=(
                        f"pullback_pct={ctx.get('pullback_pct', 0.0):.4f} "
                        f"vwap={ctx.get('vwap', 0.0):.4f} "
                        f"reclaim={ctx.get('reclaim_threshold', 0.0):.4f}"
                    ),
                )
            )

            account = _account_snapshot(self.alpaca)
            risk_state = evaluate_daily_risk(account.equity, account.day_pl)
            if risk_state.daily_loss_limit_hit:
                _log("daily loss limit hit; skipping entry")
                return

            market_snapshots, md_quotes = self._build_market_data([symbol])
            if not market_snapshots:
                _log(f"dip trigger ignored for {symbol}: missing market snapshot")
                return
            candidates = self.engine.run_scan(market_snapshots)
            candidate = next((c for c in candidates if c.symbol == symbol), None)
            if candidate is None:
                _log(f"dip trigger ignored for {symbol}: no longer passes scan")
                return
            md_quote = md_quotes.get(symbol) or quote
            stop_distance = md_quote.ask * config.STOP_LOSS_PCT if md_quote.ask > 0 else 0.0
            if config.STOP_DISTANCE_MODE == "atr":
                atr_value = ctx.get("atr")
                if atr_value is None:
                    _, atr_value = self._get_cached_bars_metrics(item, now_eastern())
                if atr_value is not None and md_quote.ask > 0:
                    raw_pct = (atr_value * config.ATR_STOP_MULTIPLIER) / md_quote.ask
                    pct = min(
                        max(raw_pct, config.DYNAMIC_STOP_MIN_PCT),
                        config.DYNAMIC_STOP_MAX_PCT,
                    )
                    stop_distance = md_quote.ask * pct

            decision = evaluate_entry(
                candidate=candidate,
                quote=md_quote,
                account_equity=account.equity,
                open_positions=0,
                stop_distance=stop_distance,
                enforce_day_high_band=False,
            )
            if not decision.should_enter:
                _log(f"dip trigger blocked for {symbol}: {decision.reason}")
                self._safe_log_order_event(
                    OrderEventRecord(
                        timestamp=now_eastern(),
                        symbol=symbol,
                        order_id=None,
                        event="dip_trigger_blocked",
                        side=None,
                        order_type=None,
                        order_class=None,
                        qty=None,
                        limit_price=None,
                        stop_price=None,
                        take_profit_price=None,
                        filled_qty=None,
                        filled_avg_price=None,
                        status=None,
                        signal_price=md_quote.ask if md_quote.ask > 0 else md_quote.last_price,
                        slippage_pct=None,
                        fill_seconds=None,
                        reject_reason=decision.reason,
                        notes=None,
                    )
                )
                return
            self._execute_entry(decision, md_quote, account)
            return

    def _evaluate_entries(
        self,
        candidates: list,
        quotes: dict[str, QuoteSnapshot],
        account: AccountSnapshot,
        enforce_day_high_band: bool | None = None,
    ) -> tuple[list[EntryDecision], dict[str, dict]]:
        decisions: list[EntryDecision] = []
        extras: dict[str, dict] = {}
        open_positions = len(self.engine.positions.positions)
        for candidate in candidates:
            quote = quotes.get(candidate.symbol)
            extras[candidate.symbol] = {}
            if quote is None:
                decisions.append(EntryDecision(candidate.symbol, False, "missing quote", 0.0))
                continue
            decision = evaluate_entry(
                candidate=candidate,
                quote=quote,
                account_equity=account.equity,
                open_positions=open_positions,
                enforce_day_high_band=enforce_day_high_band,
            )
            extra = extras[candidate.symbol]

            bars = None
            needs_bars = config.BAR_CONFIRMATION_ENABLED or config.STOP_DISTANCE_MODE in {"atr", "structure"}
            if needs_bars:
                try:
                    bars = self.polygon.get_minute_bars(candidate.symbol, config.BAR_LOOKBACK_MINUTES)
                except Exception as exc:
                    bars = None
                    extra["bar_error"] = str(exc)
                    if config.BAR_CONFIRMATION_ENABLED:
                        decision = EntryDecision(
                            candidate.symbol,
                            False,
                            f"bar confirmation failed: bar data error: {exc}",
                            0.0,
                        )

            if bars:
                extra["bars_count"] = len(bars)
                atr_value = bars_atr(bars, period=config.ATR_PERIOD)
                if atr_value is not None:
                    extra["atr"] = atr_value
                    if quote.ask > 0:
                        extra["atr_pct"] = atr_value / quote.ask

            if decision.should_enter and config.BAR_CONFIRMATION_ENABLED and bars:
                ok, reason, ctx = self._bar_confirms_entry(candidate.symbol, quote, bars=bars)
                extra["bar_confirm_ok"] = ok
                extra["bar_confirm_reason"] = reason
                extra.update(ctx)
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
                    atr_value = extra.get("atr")
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
                    extra["stop_distance_calc"] = stop_distance
                    decision = evaluate_entry(
                        candidate=candidate,
                        quote=quote,
                        account_equity=account.equity,
                        open_positions=open_positions,
                        stop_distance=stop_distance,
                        enforce_day_high_band=enforce_day_high_band,
                    )
            decisions.append(decision)
            if decision.should_enter:
                open_positions += 1
        return decisions, extras

    def _confirm_entry(
        self, symbols: list[str], account: AccountSnapshot
    ) -> tuple[EntryDecision | None, QuoteSnapshot | None]:
        required = max(1, config.CONFIRMATION_SNAPSHOTS)
        remaining_symbols: set[str] | None = None
        last_candidates = []
        last_decisions: list[EntryDecision] = []
        last_quotes: dict[str, QuoteSnapshot] = {}

        for index in range(required):
            scan_time = now_eastern()
            symbols_to_check = symbols
            if remaining_symbols is not None:
                symbols_to_check = list(remaining_symbols)
            market_snapshots, quotes = self._build_market_data(symbols_to_check)
            candidates = self.engine.run_scan(market_snapshots)
            decisions, extras = self._evaluate_entries(candidates, quotes, account)
            self._debug_rejections(market_snapshots, decisions)
            self._log_top_decisions(candidates, decisions, quotes)
            self._persist_decision_features(
                scan_time=scan_time,
                confirm_step=index + 1,
                gainers=symbols,
                market_snapshots=market_snapshots,
                candidates=candidates,
                decisions=decisions,
                quotes=quotes,
                extras=extras,
            )
            entry_symbols = {decision.symbol for decision in decisions if decision.should_enter}
            remaining_symbols = entry_symbols if remaining_symbols is None else remaining_symbols & entry_symbols
            _log(
                f"confirm {index + 1}/{required}: checked={len(symbols_to_check)} "
                f"candidates={len(candidates)} entries={len(entry_symbols)} remaining={len(remaining_symbols or set())}"
            )
            last_candidates = candidates
            last_decisions = decisions
            last_quotes = quotes
            if not remaining_symbols:
                return None, None
            if index < required - 1 and config.CONFIRMATION_INTERVAL_SECONDS > 0:
                time_module.sleep(config.CONFIRMATION_INTERVAL_SECONDS)

        for candidate in last_candidates:
            if candidate.symbol not in remaining_symbols:
                continue
            for decision in last_decisions:
                if decision.symbol == candidate.symbol and decision.should_enter:
                    return decision, last_quotes.get(candidate.symbol)
        return None, None

    def _await_order_fill(self, order_id: str) -> dict:
        deadline = now_eastern() + timedelta(seconds=config.ENTRY_FILL_TIMEOUT_SECONDS)
        last_order = {}
        while now_eastern() < deadline:
            order = self.alpaca.get_order(order_id)
            last_order = order
            status = order.get("status")
            if status in {"filled", "canceled", "expired", "rejected"}:
                return order
            time_module.sleep(config.ENTRY_POLL_INTERVAL_SECONDS)
        try:
            self.alpaca.cancel_order(order_id)
        except RuntimeError:
            pass
        time_module.sleep(config.ENTRY_POLL_INTERVAL_SECONDS)
        order = self.alpaca.get_order(order_id)
        order["_cancel_reason"] = "fill_timeout"
        return order

    def _log_entry_attempt(
        self,
        signal_time: datetime,
        signal_price: float,
        order: dict,
        rejection_reason: str | None = None,
    ) -> None:
        filled_qty = float(order.get("filled_qty") or 0.0)
        filled_avg_price = order.get("filled_avg_price")
        avg_price = float(filled_avg_price) if filled_avg_price else None
        filled_at = parse_timestamp(order.get("filled_at"))
        status = order.get("status", "unknown")
        if filled_qty > 0 and status != "filled":
            status = "partial"
        slippage_pct = None
        if avg_price is not None and signal_price > 0:
            slippage_pct = (avg_price - signal_price) / signal_price
        self.logger.log_entry_attempt(
            EntryAttemptRecord(
                signal_time=signal_time,
                symbol=order.get("symbol", ""),
                signal_price=signal_price,
                order_id=order.get("id"),
                status=status,
                filled_qty=filled_qty,
                filled_avg_price=avg_price,
                fill_time=filled_at,
                slippage_pct=slippage_pct,
                rejection_reason=rejection_reason,
            )
        )

    def _submit_exit_oco(
        self,
        symbol: str,
        qty: float,
        entry_price: float,
        stop_price: float | None = None,
    ) -> tuple[dict, float, float]:
        take_profit_price = entry_price * (1 + config.PROFIT_TARGET_PCT)
        stop_price_value = stop_price if stop_price is not None else entry_price * (1 - config.STOP_LOSS_PCT)
        order = self.alpaca.submit_order(
            symbol=symbol,
            qty=qty,
            side="sell",
            order_type="limit",
            time_in_force=config.ORDER_TIME_IN_FORCE,
            order_class="oco",
            take_profit={"limit_price": str(round(take_profit_price, 2))},
            stop_loss={"stop_price": str(round(stop_price_value, 2))},
        )
        return order, take_profit_price, stop_price_value

    def _last_logged_stop_price(self, symbol: str) -> float | None:
        """Best-effort stop recovery if the process restarted or OCO submission failed."""

        try:
            events = self.logger.fetch_order_events(limit=200)
        except Exception:
            return None
        for event in events:
            if event.symbol != symbol:
                continue
            if event.event not in {"exit_oco_submit", "exit_oco_update"}:
                continue
            if event.stop_price is None or event.stop_price <= 0:
                continue
            return float(event.stop_price)
        return None

    def _compute_initial_stop_price(self, symbol: str, entry_price: float) -> float:
        """Compute the configured initial stop for an existing position (fallback path)."""

        stop_price = entry_price * (1 - config.STOP_LOSS_PCT)
        if entry_price <= 0:
            return stop_price
        if config.STOP_DISTANCE_MODE not in {"atr", "structure"}:
            return stop_price
        try:
            bars = self.polygon.get_minute_bars(symbol, config.BAR_LOOKBACK_MINUTES)
        except Exception:
            return stop_price
        if not bars:
            return stop_price

        stop_distance = 0.0
        if config.STOP_DISTANCE_MODE == "atr":
            atr_value = bars_atr(bars, period=config.ATR_PERIOD)
            if atr_value is not None and atr_value > 0:
                raw_pct = (atr_value * config.ATR_STOP_MULTIPLIER) / entry_price
                pct = min(
                    max(raw_pct, config.DYNAMIC_STOP_MIN_PCT),
                    config.DYNAMIC_STOP_MAX_PCT,
                )
                stop_distance = entry_price * pct
        elif config.STOP_DISTANCE_MODE == "structure":
            lookback = max(1, config.STRUCTURE_STOP_LOOKBACK_BARS)
            window = bars[-(lookback + 1) : -1] if len(bars) > 1 else []
            if window:
                stop_ref = min(bar.low for bar in window)
                stop_ref *= 1 - config.STRUCTURE_STOP_BUFFER_PCT
                raw_distance = entry_price - stop_ref
                if raw_distance > 0:
                    raw_pct = raw_distance / entry_price
                    pct = min(
                        max(raw_pct, config.DYNAMIC_STOP_MIN_PCT),
                        config.DYNAMIC_STOP_MAX_PCT,
                    )
                    stop_distance = entry_price * pct

        if stop_distance > 0:
            stop_price = entry_price - stop_distance
        return stop_price

    def _refresh_quote(self, symbol: str) -> QuoteSnapshot | None:
        snapshots = self.alpaca.get_snapshots([symbol])
        snapshot = snapshots.get(symbol)
        if not snapshot:
            return None
        latest_quote = snapshot.get("latestQuote") or {}
        latest_trade = snapshot.get("latestTrade") or {}
        daily_bar = snapshot.get("dailyBar") or {}
        bid = float(latest_quote.get("bp") or 0.0)
        ask = float(latest_quote.get("ap") or 0.0)
        bid_size = float(latest_quote.get("bs") or 0.0)
        ask_size = float(latest_quote.get("as") or 0.0)
        last_price = float(latest_trade.get("p") or daily_bar.get("c") or 0.0)
        day_high = float(daily_bar.get("h") or 0.0)
        return QuoteSnapshot(
            symbol=symbol,
            last_price=last_price,
            day_high=day_high,
            bid=bid,
            ask=ask,
            bid_size=bid_size,
            ask_size=ask_size,
            halted=False,
        )

    def _execute_entry(self, decision: EntryDecision, quote: QuoteSnapshot, account: AccountSnapshot) -> None:
        enforce_day_high_band = config.ENFORCE_DAY_HIGH_BAND
        if config.ENTRY_STYLE == "dip_watchlist":
            enforce_day_high_band = False
        if not self._quote_allows_entry(quote, enforce_day_high_band=enforce_day_high_band):
            _log(f"skip entry for {decision.symbol}: quote invalid")
            return
        signal_time = now_eastern()
        base_signal_price = quote.ask
        risk_qty = math.floor(decision.position_size)
        max_qty = math.floor(account.buying_power / quote.ask) if quote.ask else 0
        qty = min(risk_qty, max_qty)
        if qty <= 0:
            _log(f"skip entry for {decision.symbol}: position size too small")
            return
        attempt = 0
        limit_price = base_signal_price
        while attempt <= config.ENTRY_RETRY_COUNT:
            submit_time = now_eastern()
            order = self.alpaca.submit_order(
                symbol=decision.symbol,
                qty=qty,
                side="buy",
                order_type="limit",
                time_in_force=config.ORDER_TIME_IN_FORCE,
                limit_price=limit_price,
            )
            rejection_reason = order.get("reject_reason") or order.get("rejected_reason")
            self._safe_log_order_event(
                OrderEventRecord(
                    timestamp=submit_time,
                    symbol=decision.symbol,
                    order_id=order.get("id"),
                    event="entry_submit",
                    side="buy",
                    order_type="limit",
                    order_class=order.get("order_class"),
                    qty=qty,
                    limit_price=limit_price,
                    stop_price=None,
                    take_profit_price=None,
                    filled_qty=None,
                    filled_avg_price=None,
                    status=order.get("status"),
                    signal_price=base_signal_price,
                    slippage_pct=None,
                    fill_seconds=None,
                    reject_reason=rejection_reason,
                    notes=f"attempt={attempt}",
                )
            )
            if order.get("status") in {"rejected", "canceled"}:
                self._log_entry_attempt(signal_time, base_signal_price, order, rejection_reason)
                self._safe_log_order_event(
                    OrderEventRecord(
                        timestamp=now_eastern(),
                        symbol=decision.symbol,
                        order_id=order.get("id"),
                        event="entry_rejected",
                        side="buy",
                        order_type=order.get("type") or "limit",
                        order_class=order.get("order_class"),
                        qty=qty,
                        limit_price=limit_price,
                        stop_price=None,
                        take_profit_price=None,
                        filled_qty=float(order.get("filled_qty") or 0.0),
                        filled_avg_price=None,
                        status=order.get("status"),
                        signal_price=base_signal_price,
                        slippage_pct=None,
                        fill_seconds=None,
                        reject_reason=rejection_reason,
                        notes=f"attempt={attempt}",
                    )
                )
                _log(f"entry rejected for {decision.symbol}: {order.get('status')}")
                return

            order = self._await_order_fill(order["id"])
            self._log_entry_attempt(signal_time, base_signal_price, order, rejection_reason)
            filled_qty = float(order.get("filled_qty") or 0.0)
            avg_price = order.get("filled_avg_price")
            filled_avg_price = float(avg_price) if avg_price else None
            fill_time = parse_timestamp(order.get("filled_at")) or now_eastern()
            fill_seconds = (fill_time - signal_time).total_seconds()
            slippage_pct = None
            if filled_avg_price is not None and base_signal_price > 0:
                slippage_pct = (filled_avg_price - base_signal_price) / base_signal_price
            cancel_reason = order.get("_cancel_reason")

            if filled_qty > 0 and filled_avg_price:
                self.engine.open_position(decision.symbol, filled_avg_price, filled_qty, fill_time)
                take_profit_price = filled_avg_price * (1 + config.PROFIT_TARGET_PCT)
                stop_price = filled_avg_price * (1 - config.STOP_LOSS_PCT)
                if decision.stop_distance > 0:
                    stop_price = filled_avg_price - decision.stop_distance
                if stop_price > 0:
                    # If OCO submission fails, monitor_positions will use this as a fallback stop.
                    self.position_stops[decision.symbol] = stop_price
                _log(
                    f"entry filled for {decision.symbol} qty={filled_qty} price={filled_avg_price:.2f} "
                    f"signal={base_signal_price:.2f} slip={(slippage_pct or 0.0):.2%} fill_s={fill_seconds:.1f} "
                    f"tp={take_profit_price:.2f} stop={stop_price:.2f} order_id={order.get('id')}"
                )
                self._safe_log_order_event(
                    OrderEventRecord(
                        timestamp=fill_time,
                        symbol=decision.symbol,
                        order_id=order.get("id"),
                        event="entry_filled",
                        side="buy",
                        order_type=order.get("type") or "limit",
                        order_class=order.get("order_class"),
                        qty=qty,
                        limit_price=limit_price,
                        stop_price=None,
                        take_profit_price=None,
                        filled_qty=filled_qty,
                        filled_avg_price=filled_avg_price,
                        status=order.get("status"),
                        signal_price=base_signal_price,
                        slippage_pct=slippage_pct,
                        fill_seconds=fill_seconds,
                        reject_reason=rejection_reason,
                        notes=f"attempt={attempt}",
                    )
                )
                try:
                    exit_order, tp_price, stop_price_value = self._submit_exit_oco(
                        decision.symbol,
                        filled_qty,
                        filled_avg_price,
                        stop_price=stop_price,
                    )
                    self.position_stops[decision.symbol] = stop_price_value
                    self._safe_log_order_event(
                        OrderEventRecord(
                            timestamp=now_eastern(),
                            symbol=decision.symbol,
                            order_id=exit_order.get("id"),
                            event="exit_oco_submit",
                            side="sell",
                            order_type=exit_order.get("type") or "limit",
                            order_class=exit_order.get("order_class") or "oco",
                            qty=filled_qty,
                            limit_price=None,
                            stop_price=stop_price_value,
                            take_profit_price=tp_price,
                            filled_qty=float(exit_order.get("filled_qty") or 0.0),
                            filled_avg_price=None,
                            status=exit_order.get("status"),
                            signal_price=filled_avg_price,
                            slippage_pct=None,
                            fill_seconds=None,
                            reject_reason=None,
                            notes=None,
                        )
                    )
                except RuntimeError as exc:
                    _log(f"exit OCO submit failed for {decision.symbol}: {exc}")
                return

            if attempt >= config.ENTRY_RETRY_COUNT:
                event = "entry_not_filled"
                notes = f"attempt={attempt}"
                if cancel_reason:
                    event = "entry_canceled"
                    notes = f"attempt={attempt} cancel_reason={cancel_reason}"
                self._safe_log_order_event(
                    OrderEventRecord(
                        timestamp=fill_time,
                        symbol=decision.symbol,
                        order_id=order.get("id"),
                        event=event,
                        side="buy",
                        order_type=order.get("type") or "limit",
                        order_class=order.get("order_class"),
                        qty=qty,
                        limit_price=limit_price,
                        stop_price=None,
                        take_profit_price=None,
                        filled_qty=filled_qty,
                        filled_avg_price=filled_avg_price,
                        status=order.get("status"),
                        signal_price=base_signal_price,
                        slippage_pct=slippage_pct,
                        fill_seconds=fill_seconds,
                        reject_reason=rejection_reason,
                        notes=notes,
                    )
                )
                _log(
                    f"entry not filled for {decision.symbol}; giving up status={order.get('status')} "
                    f"filled_qty={order.get('filled_qty')} order_id={order.get('id')}"
                )
                return
            refreshed = self._refresh_quote(decision.symbol)
            if refreshed is None or not self._quote_allows_entry(refreshed, enforce_day_high_band=enforce_day_high_band):
                _log(f"entry retry failed for {decision.symbol}: quote invalid")
                return
            max_price = base_signal_price * (1 + config.ENTRY_MAX_SLIPPAGE_PCT)
            if refreshed.ask > max_price:
                _log(f"entry retry skipped for {decision.symbol}: slippage too high")
                return
            limit_price = refreshed.ask
            attempt += 1

    def _should_scan_now(self) -> bool:
        now = now_eastern()
        if config.SCAN_MODE == "once":
            today = now.date().isoformat()
            if self.last_scan_date == today:
                return False
            if now.timetz() < self.scan_time:
                return False
            return True

        # Default: interval scanning between start/end times.
        if now.timetz() < self.scan_start_time or now.timetz() > self.scan_end_time:
            return False
        if self.last_scan_at is not None and config.SCAN_INTERVAL_SECONDS > 0:
            if now - self.last_scan_at < timedelta(seconds=config.SCAN_INTERVAL_SECONDS):
                return False
        return True

    def _cooldown_active(self) -> bool:
        now = now_eastern()
        trades = self.logger.fetch_trades()
        if config.MAX_CONSECUTIVE_LOSSES > 0:
            losses = count_consecutive_losses(trades)
            if losses >= config.MAX_CONSECUTIVE_LOSSES:
                _log(f"max consecutive losses reached ({losses}); skipping entries")
                return True
        if config.COOLDOWN_MINUTES_AFTER_LOSS > 0:
            loss_time = last_loss_time(trades)
            if loss_time:
                cooldown = timedelta(minutes=config.COOLDOWN_MINUTES_AFTER_LOSS)
                if now - loss_time < cooldown:
                    _log("cooldown after loss active; skipping entries")
                    return True
        if config.COOLDOWN_MINUTES_AFTER_ENTRY_FAILURE > 0:
            attempts = self.logger.fetch_entry_attempts(limit=1)
            if attempts:
                attempt = attempts[0]
                if attempt.status in {"rejected", "canceled", "expired"} and attempt.filled_qty == 0:
                    cooldown = timedelta(minutes=config.COOLDOWN_MINUTES_AFTER_ENTRY_FAILURE)
                    if now - attempt.signal_time < cooldown:
                        _log("cooldown after failed entry active; skipping entries")
                        return True
        return False

    def scan_and_trade(self) -> None:
        now = now_eastern()
        if not within_trading_window(now):
            if config.LIVE_DEBUG:
                _log("scan skipped: outside trading window")
            return

        if self.logger.has_trade_for_date(now.date().isoformat()):
            if config.LIVE_DEBUG:
                _log("scan skipped: already traded today")
            return

        if self.alpaca.get_positions():
            if config.LIVE_DEBUG:
                _log("scan skipped: existing position open")
            return

        account = _account_snapshot(self.alpaca)
        risk_state = evaluate_daily_risk(account.equity, account.day_pl)
        if risk_state.daily_loss_limit_hit:
            _log("daily loss limit hit; skipping entries")
            return
        if self._cooldown_active():
            return

        source = getattr(config, "LIVE_SYMBOL_SOURCE", "polygon_gainers")
        if source == "alpaca_movers":
            top = getattr(config, "ALPACA_MOVERS_TOP", config.GAINERS_LIMIT)
            gainers = self.alpaca.get_mover_gainer_symbols(int(top))
        else:
            gainers = self.polygon.get_gainers(config.GAINERS_LIMIT)
        if config.LIVE_DEBUG:
            _log(f"scan source={source} symbols={len(gainers)} head={','.join(gainers[:10])}")
        symbols = gainers
        if self.tradable_symbols:
            symbols = [symbol for symbol in gainers if symbol in self.tradable_symbols]
            if config.LIVE_DEBUG:
                _log(f"tradable gainers: {len(symbols)}")
        elif config.LIVE_DEBUG:
            _log("tradable-symbol cache unavailable; skipping tradable filter")
        if not symbols:
            _log("no candidates from gainer list")
            self._persist_scan_audit(scan_time=now, gainers=gainers, market_snapshots=[])
            return

        if config.LIVE_LOG_SCANS:
            criteria = (
                f"source={source} "
                f"min_gain={config.MIN_GAIN_PCT} max_gain={config.MAX_GAIN_PCT} "
                f"min_relvol={config.MIN_REL_VOLUME} min_dvol={config.MIN_DOLLAR_VOLUME} "
                f"price=[{config.MIN_PRICE},{config.MAX_PRICE}] spread<={config.MAX_BID_ASK_SPREAD_PCT} "
                f"min_qsz={config.MIN_QUOTE_SIZE} bar={config.BAR_CONFIRMATION_ENABLED}:{config.BAR_CONFIRMATION_METHOD} "
                f"stop_mode={config.STOP_DISTANCE_MODE} pt={config.PROFIT_TARGET_PCT} sl={config.STOP_LOSS_PCT} "
                f"entry_style={config.ENTRY_STYLE}"
            )
            try:
                self.logger.log_scan(
                    ScanRecord(
                        timestamp=now,
                        symbols=",".join(symbols),
                        criteria=criteria,
                    )
                )
            except Exception as exc:
                if config.LIVE_DEBUG:
                    _log(f"scan log failed: {exc}")

        if config.ENTRY_STYLE == "breakout":
            entry_decision, quote = self._confirm_entry(symbols, account)
            if entry_decision is None:
                _log("no confirmed entry signals")
                return
            if quote is None:
                _log(f"missing quote for {entry_decision.symbol}")
                return
            self._execute_entry(entry_decision, quote, account)
            return

        # Watchlist scan (dip entry mode): refresh watchlist and then let the dip trigger place the trade.
        scan_time = now_eastern()
        market_snapshots, quotes = self._build_market_data(symbols)
        self._persist_scan_audit(scan_time=scan_time, gainers=gainers, market_snapshots=market_snapshots)
        candidates = self.engine.run_scan(market_snapshots)
        decisions, extras = self._evaluate_entries(
            candidates,
            quotes,
            account,
            enforce_day_high_band=False,
        )
        self._debug_rejections(market_snapshots, decisions)
        self._log_top_decisions(candidates, decisions, quotes)
        self._persist_decision_features(
            scan_time=scan_time,
            confirm_step=1,
            gainers=gainers,
            market_snapshots=market_snapshots,
            candidates=candidates,
            decisions=decisions,
            quotes=quotes,
            extras=extras,
        )
        self._update_watchlist(scan_time, gainers, candidates, quotes)
        if config.LIVE_DEBUG:
            _log(f"watchlist size={len(self.watchlist)}")
        # Watchlist monitoring happens in the main loop; avoid an extra Alpaca poll here.

    def _time_stop_reached(self) -> bool:
        time_stop = time(config.TIME_STOP_HOUR, config.TIME_STOP_MINUTE, tzinfo=EASTERN_TZ)
        return now_eastern().timetz() >= time_stop

    def _latest_filled_sell_order(self, orders: list[dict], symbol: str) -> dict | None:
        latest_order = None
        latest_time = None
        for order in orders:
            if order.get("symbol") != symbol or order.get("side") != "sell":
                continue
            filled_at = parse_timestamp(order.get("filled_at"))
            if filled_at is None:
                continue
            if latest_time is None or filled_at > latest_time:
                latest_time = filled_at
                latest_order = order
        return latest_order

    def _compute_trade_path_metrics(
        self,
        symbol: str,
        entry_time: datetime,
        exit_time: datetime,
        entry_price: float,
    ) -> TradePathRecord | None:
        if exit_time <= entry_time or entry_price <= 0:
            return None
        try:
            bars = self.polygon.get_minute_bars_between(symbol, entry_time, exit_time)
        except Exception as exc:
            if config.LIVE_DEBUG:
                _log(f"trade path fetch failed for {symbol}: {exc}")
            return None
        bars = [bar for bar in bars if bar.high > 0 and bar.low > 0]
        if not bars:
            return None
        max_bar = max(bars, key=lambda bar: bar.high)
        min_bar = min(bars, key=lambda bar: bar.low)
        max_high = max_bar.high
        min_low = min_bar.low
        mfe_pct = (max_high - entry_price) / entry_price if max_high > 0 else None
        mae_pct = (min_low - entry_price) / entry_price if min_low > 0 else None
        time_to_mfe_s = (max_bar.timestamp - entry_time).total_seconds()
        time_to_mae_s = (min_bar.timestamp - entry_time).total_seconds()
        if time_to_mfe_s < 0:
            time_to_mfe_s = 0.0
        if time_to_mae_s < 0:
            time_to_mae_s = 0.0
        return TradePathRecord(
            symbol=symbol,
            entry_time=entry_time,
            exit_time=exit_time,
            entry_price=entry_price,
            bars_count=len(bars),
            max_high=max_high,
            min_low=min_low,
            mfe_pct=mfe_pct,
            mae_pct=mae_pct,
            time_to_mfe_s=time_to_mfe_s,
            time_to_mae_s=time_to_mae_s,
        )

    def _sync_closed_positions(self, positions: list[dict]) -> None:
        open_trades = self.logger.fetch_open_trades()
        if not open_trades:
            return
        open_symbols = {position["symbol"] for position in positions}
        closed_trades = [trade for trade in open_trades if trade.symbol not in open_symbols]
        if not closed_trades:
            return
        closed_orders = self.alpaca.get_orders(status="closed", limit=100)
        for trade in closed_trades:
            order = self._latest_filled_sell_order(closed_orders, trade.symbol)
            if not order:
                continue
            exit_price = float(order.get("filled_avg_price") or 0.0)
            if exit_price <= 0:
                continue
            exit_time = parse_timestamp(order.get("filled_at")) or now_eastern()
            pnl = (exit_price - trade.entry_price) * trade.qty
            self.logger.log_trade_exit(
                symbol=trade.symbol,
                exit_price=exit_price,
                exit_time=exit_time,
                pnl=pnl,
                exit_reason="broker_exit",
            )
            self.position_stops.pop(trade.symbol, None)
            self.position_peaks.pop(trade.symbol, None)
            self._safe_log_order_event(
                OrderEventRecord(
                    timestamp=exit_time,
                    symbol=trade.symbol,
                    order_id=order.get("id"),
                    event="exit_filled",
                    side="sell",
                    order_type=order.get("type") or order.get("order_type"),
                    order_class=order.get("order_class"),
                    qty=float(order.get("filled_qty") or 0.0),
                    limit_price=float(order.get("limit_price") or 0.0) or None,
                    stop_price=float(order.get("stop_price") or 0.0) or None,
                    take_profit_price=None,
                    filled_qty=float(order.get("filled_qty") or 0.0),
                    filled_avg_price=exit_price,
                    status=order.get("status"),
                    signal_price=trade.entry_price,
                    slippage_pct=None,
                    fill_seconds=None,
                    reject_reason=None,
                    notes=order.get("type") or order.get("order_type"),
                )
            )
            path = self._compute_trade_path_metrics(
                symbol=trade.symbol,
                entry_time=trade.entry_time,
                exit_time=exit_time,
                entry_price=trade.entry_price,
            )
            if path:
                self._safe_log_trade_path(path)
            _log(
                f"exit logged for {trade.symbol} price={exit_price:.2f} pnl={pnl:.2f}"
            )

    def _ensure_exit_orders(self, symbol: str, qty: float, entry_price: float, open_sell_orders: list[dict]) -> None:
        if open_sell_orders:
            return
        stop_price = self.position_stops.get(symbol)
        if stop_price is None:
            stop_price = self._last_logged_stop_price(symbol)
        if stop_price is None or stop_price <= 0:
            stop_price = self._compute_initial_stop_price(symbol, entry_price)
        try:
            exit_order, tp_price, stop_price_value = self._submit_exit_oco(
                symbol, qty, entry_price, stop_price=stop_price
            )
            self.position_stops[symbol] = stop_price_value
            _log(f"exit OCO submitted for {symbol} qty={qty}")
            self._safe_log_order_event(
                OrderEventRecord(
                    timestamp=now_eastern(),
                    symbol=symbol,
                    order_id=exit_order.get("id"),
                    event="exit_oco_submit",
                    side="sell",
                    order_type=exit_order.get("type") or "limit",
                    order_class=exit_order.get("order_class") or "oco",
                    qty=qty,
                    limit_price=None,
                    stop_price=stop_price_value,
                    take_profit_price=tp_price,
                    filled_qty=float(exit_order.get("filled_qty") or 0.0),
                    filled_avg_price=None,
                    status=exit_order.get("status"),
                    signal_price=entry_price,
                    slippage_pct=None,
                    fill_seconds=None,
                    reject_reason=None,
                    notes="ensure_exit_orders",
                )
            )
        except RuntimeError as exc:
            _log(f"exit OCO submit failed for {symbol}: {exc}")

    def _stop_price_from_orders(self, orders: list[dict]) -> float | None:
        for order in orders:
            stop_price = order.get("stop_price")
            if stop_price is None:
                continue
            try:
                value = float(stop_price)
            except (TypeError, ValueError):
                continue
            if value > 0:
                return value
        return None

    def _maybe_adjust_exit_orders(
        self,
        symbol: str,
        qty: float,
        entry_price: float,
        current_price: float,
        open_sell_orders: list[dict],
    ) -> None:
        if not open_sell_orders:
            return
        if current_price <= 0 or entry_price <= 0:
            return

        peak = self.position_peaks.get(symbol, entry_price)
        if current_price > peak:
            peak = current_price
        self.position_peaks[symbol] = peak

        take_profit_price = entry_price * (1 + config.PROFIT_TARGET_PCT)
        current_stop = self.position_stops.get(symbol)
        if current_stop is None:
            current_stop = self._stop_price_from_orders(open_sell_orders)
        desired_stop = current_stop if current_stop is not None else entry_price * (1 - config.STOP_LOSS_PCT)

        profit_lock_active = (
            config.PROFIT_LOCK_ENABLED and peak >= entry_price * (1 + config.PROFIT_LOCK_TRIGGER_PCT)
        )
        trail_active = (
            config.TRAIL_STOP_ENABLED and peak >= entry_price * (1 + config.TRAIL_STOP_TRIGGER_PCT)
        )
        if config.PROFIT_LOCK_ENABLED and peak >= entry_price * (1 + config.PROFIT_LOCK_TRIGGER_PCT):
            desired_stop = max(desired_stop, entry_price * (1 + config.PROFIT_LOCK_STOP_PCT))
        if config.TRAIL_STOP_ENABLED and peak >= entry_price * (1 + config.TRAIL_STOP_TRIGGER_PCT):
            desired_stop = max(desired_stop, peak * (1 - config.TRAIL_STOP_DISTANCE_PCT))

        desired_stop = round(desired_stop, 2)
        if desired_stop <= 0:
            return
        if desired_stop >= take_profit_price:
            return
        if desired_stop >= current_price:
            return
        if current_stop is not None and desired_stop <= round(current_stop, 2):
            return

        for order in open_sell_orders:
            try:
                self.alpaca.cancel_order(order["id"])
            except RuntimeError:
                continue
        try:
            exit_order, tp_price, stop_price_value = self._submit_exit_oco(
                symbol, qty, entry_price, stop_price=desired_stop
            )
            self.position_stops[symbol] = stop_price_value
            _log(f"updated stop for {symbol} stop={desired_stop:.2f} peak={peak:.2f}")
            reason_parts: list[str] = []
            if profit_lock_active:
                reason_parts.append("profit_lock")
            if trail_active:
                reason_parts.append("trail")
            reason = "+".join(reason_parts) if reason_parts else "update"
            self._safe_log_exit_update(
                ExitUpdateRecord(
                    timestamp=now_eastern(),
                    symbol=symbol,
                    entry_price=entry_price,
                    qty=qty,
                    current_price=current_price,
                    peak_price=peak,
                    old_stop_price=current_stop,
                    new_stop_price=stop_price_value,
                    reason=reason,
                    notes=None,
                )
            )
            self._safe_log_order_event(
                OrderEventRecord(
                    timestamp=now_eastern(),
                    symbol=symbol,
                    order_id=exit_order.get("id"),
                    event="exit_oco_update",
                    side="sell",
                    order_type=exit_order.get("type") or "limit",
                    order_class=exit_order.get("order_class") or "oco",
                    qty=qty,
                    limit_price=None,
                    stop_price=stop_price_value,
                    take_profit_price=tp_price,
                    filled_qty=float(exit_order.get("filled_qty") or 0.0),
                    filled_avg_price=None,
                    status=exit_order.get("status"),
                    signal_price=entry_price,
                    slippage_pct=None,
                    fill_seconds=None,
                    reject_reason=None,
                    notes=reason,
                )
            )
        except RuntimeError as exc:
            _log(f"exit OCO update failed for {symbol}: {exc}")

    def _force_time_exit(self, symbol: str, qty: float, entry_price: float, open_sell_orders: list[dict]) -> None:
        trade_entry_time = None
        for trade in self.logger.fetch_open_trades():
            if trade.symbol == symbol:
                trade_entry_time = trade.entry_time
                break
        if trade_entry_time is None:
            trade_entry_time = now_eastern()

        for order in open_sell_orders:
            try:
                self.alpaca.cancel_order(order["id"])
            except RuntimeError:
                continue
        self.position_stops.pop(symbol, None)
        self._safe_log_order_event(
            OrderEventRecord(
                timestamp=now_eastern(),
                symbol=symbol,
                order_id=None,
                event="time_stop_exit_submit",
                side="sell",
                order_type="market",
                order_class=None,
                qty=qty,
                limit_price=None,
                stop_price=None,
                take_profit_price=None,
                filled_qty=None,
                filled_avg_price=None,
                status=None,
                signal_price=entry_price,
                slippage_pct=None,
                fill_seconds=None,
                reject_reason=None,
                notes=None,
            )
        )
        order = self.alpaca.submit_order(
            symbol=symbol,
            qty=qty,
            side="sell",
            order_type="market",
            time_in_force=config.ORDER_TIME_IN_FORCE,
        )
        order = self._await_order_fill(order["id"])
        exit_price = float(order.get("filled_avg_price") or 0.0)
        if exit_price <= 0:
            _log(f"time stop exit failed for {symbol}")
            return
        exit_time = parse_timestamp(order.get("filled_at")) or now_eastern()
        pnl = (exit_price - entry_price) * qty
        self.logger.log_trade_exit(
            symbol=symbol,
            exit_price=exit_price,
            exit_time=exit_time,
            pnl=pnl,
            exit_reason="time_stop",
        )
        path = self._compute_trade_path_metrics(
            symbol=symbol,
            entry_time=trade_entry_time,
            exit_time=exit_time,
            entry_price=entry_price,
        )
        if path:
            self._safe_log_trade_path(path)
        self._safe_log_order_event(
            OrderEventRecord(
                timestamp=exit_time,
                symbol=symbol,
                order_id=order.get("id"),
                event="time_stop_exit_filled",
                side="sell",
                order_type=order.get("type") or "market",
                order_class=order.get("order_class"),
                qty=qty,
                limit_price=None,
                stop_price=None,
                take_profit_price=None,
                filled_qty=float(order.get("filled_qty") or 0.0),
                filled_avg_price=exit_price,
                status=order.get("status"),
                signal_price=entry_price,
                slippage_pct=None,
                fill_seconds=None,
                reject_reason=None,
                notes=None,
            )
        )
        _log(
            f"time stop exit {symbol} entry={entry_price:.2f} exit={exit_price:.2f} pnl={pnl:.2f}"
        )

    def monitor_positions(self) -> None:
        positions = self.alpaca.get_positions()
        self._sync_closed_positions(positions)
        if not positions:
            self.position_peaks.clear()
            self.position_stops.clear()
            return
        open_orders = self.alpaca.get_orders(status="open")
        open_sell_orders: dict[str, list[dict]] = {}
        for order in open_orders:
            if order.get("side") != "sell":
                continue
            open_sell_orders.setdefault(order["symbol"], []).append(order)

        for position in positions:
            symbol = position["symbol"]
            entry_price = float(position["avg_entry_price"])
            qty = abs(float(position["qty"]))
            current_price = float(position.get("current_price") or 0.0)
            if self._time_stop_reached():
                self._force_time_exit(
                    symbol,
                    qty,
                    entry_price,
                    open_sell_orders.get(symbol, []),
                )
                continue
            self._ensure_exit_orders(
                symbol,
                qty,
                entry_price,
                open_sell_orders.get(symbol, []),
            )
            self._maybe_adjust_exit_orders(
                symbol,
                qty,
                entry_price,
                current_price,
                open_sell_orders.get(symbol, []),
            )

    def run(self) -> None:
        _log("live trader started")
        market_open = time(9, 30, tzinfo=EASTERN_TZ)
        market_close = time(16, 0, tzinfo=EASTERN_TZ)
        while True:
            now = now_eastern()
            in_market_hours = is_within_market_window(now, market_open, market_close)
            if self.exit_time is not None and now.timetz() >= self.exit_time:
                try:
                    if not self.alpaca.get_positions():
                        _log("exit time reached; shutting down live trader")
                        return
                except Exception as exc:
                    _log(f"exit check error: {exc}")
            base_sleep = config.MARKET_POLL_INTERVAL_SECONDS if in_market_hours else config.OFF_HOURS_POLL_INTERVAL_SECONDS
            try:
                base_sleep_seconds = max(1, int(base_sleep))
            except (TypeError, ValueError):
                base_sleep_seconds = max(1, int(config.POLL_INTERVAL_SECONDS))
            sleep_seconds = max(base_sleep_seconds, int(self.rate_limit_backoff_seconds or 0))
            try:
                if self._should_scan_now():
                    self.scan_and_trade()
                    if config.SCAN_MODE == "once":
                        self.last_scan_date = now.date().isoformat()
                    self.last_scan_at = now
                if config.ENTRY_STYLE == "dip_watchlist":
                    self._maybe_trade_watchlist()
                self.monitor_positions()
                self.rate_limit_backoff_seconds = 0
            except RuntimeError as exc:
                message = str(exc)
                if "Alpaca API error 429" in message or "rate limit" in message.lower():
                    if self.rate_limit_backoff_seconds <= 0:
                        self.rate_limit_backoff_seconds = config.ALPACA_RATE_LIMIT_BACKOFF_INITIAL_SECONDS
                    else:
                        self.rate_limit_backoff_seconds = min(
                            max(
                                self.rate_limit_backoff_seconds * 2,
                                config.ALPACA_RATE_LIMIT_BACKOFF_INITIAL_SECONDS,
                            ),
                            config.ALPACA_RATE_LIMIT_BACKOFF_MAX_SECONDS,
                        )
                    sleep_seconds = max(sleep_seconds, self.rate_limit_backoff_seconds)
                else:
                    self.rate_limit_backoff_seconds = 0
                _log(f"error: {exc}")
            except Exception as exc:
                self.rate_limit_backoff_seconds = 0
                _log(f"error: {exc}")
            time_module.sleep(sleep_seconds)


def run() -> None:
    LiveTrader().run()


if __name__ == "__main__":
    run()
