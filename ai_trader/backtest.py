"""Historical backtester for the AI options trader.

Replays historical news and market data through the LLM brain,
uses real Polygon options market data for pricing, and tracks P&L.

Run:  python -m ai_trader.backtest --start 2025-01-02 --end 2025-01-31

Requires: ANTHROPIC_API_KEY, POLYGON_API_KEY (paid tier for options data)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time as time_module
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

from . import config
from .brain import MarketAnalysis, TradingBrain, TradeDecision
from .db import format_trade_history
from .journal import ThesisJournal
from .news import NewsItem, build_news_events, classify_catalyst_reaction, format_news_for_llm
from .options import approx_delta, target_dte_for_expiry_preference, target_strike_for_preference
from .risk import size_for_risk_budget
from .utils import EASTERN_TZ, log, now_eastern


# ---------------------------------------------------------------------------
# Historical data fetching (Polygon — equities)
# ---------------------------------------------------------------------------

def _polygon_request(api_key: str, path: str, params: dict | None = None) -> dict:
    import requests

    params = params or {}
    params["apiKey"] = api_key
    url = f"https://api.polygon.io{path}"
    response = requests.get(url, params=params, timeout=30)
    for attempt in range(3):
        if response.status_code != 429:
            break
        time_module.sleep(12 + attempt * 5)
        response = requests.get(url, params=params, timeout=30)
    if response.status_code >= 400:
        raise RuntimeError(f"Polygon {response.status_code}: {response.text[:200]}")
    return response.json()


def fetch_historical_news_window(
    api_key: str,
    start_time: datetime,
    end_time: datetime,
    limit: int = 50,
) -> list[dict]:
    """Fetch Polygon news published within a precise timestamp window."""
    if end_time <= start_time:
        return []

    start = start_time.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    end = end_time.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        data = _polygon_request(
            api_key,
            "/v2/reference/news",
            params={
                "published_utc.gte": start,
                "published_utc.lte": end,
                "limit": str(limit),
                "sort": "published_utc",
                "order": "desc",
            },
        )
    except Exception as exc:
        log(f"news fetch error for {start_time} -> {end_time}: {exc}")
        return []
    return data.get("results", [])


def fetch_historical_news(
    api_key: str, trade_date: date, limit: int = 50,
) -> list[dict]:
    """Fetch all news for a given date from Polygon."""
    start = datetime.combine(trade_date, time(0, 0, tzinfo=EASTERN_TZ))
    end = datetime.combine(trade_date, time(23, 59, 59, tzinfo=EASTERN_TZ))
    return fetch_historical_news_window(api_key, start, end, limit=limit)


def fetch_historical_daily_bar(
    api_key: str, symbol: str, trade_date: date,
) -> dict | None:
    """Fetch OHLCV for an equity symbol on a given date."""
    try:
        data = _polygon_request(
            api_key,
            f"/v1/open-close/{symbol}/{trade_date}",
        )
        if data.get("status") == "OK":
            return data
    except Exception:
        pass
    return None


def fetch_historical_daily_bars_range(
    api_key: str, symbol: str, start: date, end: date,
) -> list[dict]:
    """Fetch daily bars for an equity over a date range."""
    try:
        data = _polygon_request(
            api_key,
            f"/v2/aggs/ticker/{symbol}/range/1/day"
            f"/{start.isoformat()}/{end.isoformat()}",
            params={"adjusted": "true", "sort": "asc", "limit": "250"},
        )
    except Exception as exc:
        log(f"bars fetch error for {symbol}: {exc}")
        return []
    return data.get("results", [])


def fetch_historical_intraday_bars(
    api_key: str,
    ticker: str,
    trading_day: date,
    multiplier: int = 5,
    cache: PolygonCache | None = None,
) -> list[dict]:
    """Fetch regular-session intraday aggregate bars for one ticker and day."""
    cache_key = (ticker, trading_day.isoformat(), multiplier)
    if cache and cache_key in cache.intraday_bars:
        return cache.intraday_bars[cache_key]

    start_ms = _to_epoch_ms(_market_open_dt(trading_day))
    end_ms = _to_epoch_ms(_market_close_dt(trading_day))
    try:
        data = _polygon_request(
            api_key,
            f"/v2/aggs/ticker/{ticker}/range/{multiplier}/minute/{start_ms}/{end_ms}",
            params={"adjusted": "true", "sort": "asc", "limit": "5000"},
        )
    except Exception as exc:
        log(f"intraday bars fetch error for {ticker} on {trading_day}: {exc}")
        return []

    results = data.get("results", [])
    if cache is not None:
        cache.intraday_bars[cache_key] = results
    return results


# ---------------------------------------------------------------------------
# Polygon options data
# ---------------------------------------------------------------------------

@dataclass
class PolygonCache:
    """In-memory cache for Polygon lookups and aggregate bars."""
    # (underlying, type, expiry_gte, expiry_lte, strike_gte, strike_lte) -> list[contract_dict]
    contracts: dict[tuple, list[dict]] = field(default_factory=dict)
    # option_ticker -> {date_iso -> bar_dict}
    option_bars: dict[str, dict[str, dict]] = field(default_factory=dict)
    # (ticker, date_iso, multiplier) -> list[agg_bar_dict]
    intraday_bars: dict[tuple[str, str, int], list[dict]] = field(default_factory=dict)


def _coerce_eastern_datetime(
    value: date | datetime,
    default_time: time = time(16, 0),
) -> datetime:
    """Normalize a date/datetime to an aware Eastern datetime."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=EASTERN_TZ)
        return value.astimezone(EASTERN_TZ)

    if default_time.tzinfo is None:
        default_time = time(
            default_time.hour,
            default_time.minute,
            default_time.second,
            default_time.microsecond,
            tzinfo=EASTERN_TZ,
        )
    return datetime.combine(value, default_time)


def _market_open_dt(trading_day: date) -> datetime:
    return datetime.combine(trading_day, time(9, 30, tzinfo=EASTERN_TZ))


def _market_close_dt(trading_day: date) -> datetime:
    return datetime.combine(trading_day, time(16, 0, tzinfo=EASTERN_TZ))


def _to_epoch_ms(value: datetime) -> int:
    return int(value.astimezone(timezone.utc).timestamp() * 1000)


def _bar_timestamp_eastern(bar: dict) -> datetime | None:
    ts_ms = bar.get("t")
    if not ts_ms:
        return None
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).astimezone(EASTERN_TZ)


def _latest_bar_before(bars: list[dict], cutoff: datetime) -> dict | None:
    """Return the most recent aggregate bar that started before cutoff."""
    for bar in reversed(bars):
        ts = _bar_timestamp_eastern(bar)
        if ts and ts < cutoff:
            return bar
    return None


def _first_bar_at_or_after(bars: list[dict], cutoff: datetime) -> dict | None:
    """Return the first aggregate bar that starts at or after cutoff."""
    for bar in bars:
        ts = _bar_timestamp_eastern(bar)
        if ts and ts >= cutoff:
            return bar
    return None


def _decision_timestamps_for_day(
    trading_day: date,
    interval_minutes: int,
    start_delay_minutes: int,
    end_buffer_minutes: int,
) -> list[datetime]:
    """Generate intraday decision timestamps aligned to the live scan cadence."""
    start = _market_open_dt(trading_day) + timedelta(minutes=start_delay_minutes)
    end = _market_close_dt(trading_day) - timedelta(minutes=end_buffer_minutes)
    timestamps: list[datetime] = []
    current = start
    while current <= end:
        timestamps.append(current)
        current += timedelta(minutes=interval_minutes)
    return timestamps


def fetch_polygon_option_contracts(
    api_key: str,
    underlying: str,
    contract_type: str,
    expiry_gte: date,
    expiry_lte: date,
    strike_gte: float,
    strike_lte: float,
    as_of: date | None = None,
    cache: PolygonCache | None = None,
) -> list[dict]:
    """Query Polygon for available option contracts matching criteria.

    For backtesting past dates, pass as_of=trade_date so Polygon returns
    contracts that existed at that time (even if they've since expired).
    """
    cache_key = (underlying, contract_type, expiry_gte.isoformat(),
                 expiry_lte.isoformat(), strike_gte, strike_lte,
                 as_of.isoformat() if as_of else "")
    if cache and cache_key in cache.contracts:
        return cache.contracts[cache_key]

    params = {
        "underlying_ticker": underlying,
        "contract_type": contract_type,
        "expiration_date.gte": expiry_gte.isoformat(),
        "expiration_date.lte": expiry_lte.isoformat(),
        "strike_price.gte": str(strike_gte),
        "strike_price.lte": str(strike_lte),
        "limit": "100",
        "sort": "strike_price",
        "order": "asc",
    }
    if as_of:
        params["as_of"] = as_of.isoformat()

    try:
        data = _polygon_request(
            api_key,
            "/v3/reference/options/contracts",
            params=params,
        )
    except Exception as exc:
        log(f"option contracts fetch error for {underlying}: {exc}")
        return []

    results = data.get("results", [])
    if cache:
        cache.contracts[cache_key] = results
    return results


def fetch_option_daily_bar(
    api_key: str,
    option_ticker: str,
    trade_date: date,
    cache: PolygonCache | None = None,
) -> dict | None:
    """Get a single-day OHLCV bar for an option contract. Checks cache first."""
    date_key = trade_date.isoformat()
    if cache and option_ticker in cache.option_bars:
        bar = cache.option_bars[option_ticker].get(date_key)
        if bar is not None:
            return bar

    # Fetch just this one day
    try:
        data = _polygon_request(
            api_key,
            f"/v2/aggs/ticker/{option_ticker}/range/1/day"
            f"/{date_key}/{date_key}",
            params={"adjusted": "true", "sort": "asc", "limit": "1"},
        )
    except Exception as exc:
        log(f"option bar fetch error for {option_ticker} on {trade_date}: {exc}")
        return None

    results = data.get("results", [])
    if not results:
        return None

    bar = results[0]
    if cache:
        cache.option_bars.setdefault(option_ticker, {})[date_key] = bar
    return bar


def fetch_option_daily_bars_range(
    api_key: str,
    option_ticker: str,
    start: date,
    end: date,
    cache: PolygonCache | None = None,
) -> list[dict]:
    """Bulk fetch daily bars for an option contract and populate cache."""
    try:
        data = _polygon_request(
            api_key,
            f"/v2/aggs/ticker/{option_ticker}/range/1/day"
            f"/{start.isoformat()}/{end.isoformat()}",
            params={"adjusted": "true", "sort": "asc", "limit": "250"},
        )
    except Exception as exc:
        log(f"option bars range fetch error for {option_ticker}: {exc}")
        return []

    results = data.get("results", [])
    if cache and results:
        bars_by_date = cache.option_bars.setdefault(option_ticker, {})
        for bar in results:
            # Polygon agg timestamps are ms since epoch — convert to date ISO
            ts_ms = bar.get("t", 0)
            if ts_ms:
                bar_date = date.fromtimestamp(ts_ms / 1000)
                bars_by_date[bar_date.isoformat()] = bar
    return results


def _option_bar_price(bar: dict) -> float:
    """Extract price from an option bar: prefer VWAP, then close."""
    vwap = bar.get("vw", 0)
    if vwap and vwap > 0:
        return float(vwap)
    close = bar.get("c", 0)
    return float(close) if close else 0.0


def _equity_bar_price(bar: dict) -> float:
    """Extract price from an equity bar."""
    close = bar.get("c", 0)
    return float(close) if close else 0.0


def _latest_intraday_bar_before(
    api_key: str,
    ticker: str,
    as_of: date | datetime,
    cache: PolygonCache | None = None,
    multiplier: int = 5,
) -> dict | None:
    as_of_dt = _coerce_eastern_datetime(as_of)
    bars = fetch_historical_intraday_bars(
        api_key, ticker, as_of_dt.date(), multiplier=multiplier, cache=cache,
    )
    return _latest_bar_before(bars, as_of_dt)


def _first_intraday_bar_at_or_after(
    api_key: str,
    ticker: str,
    as_of: date | datetime,
    cache: PolygonCache | None = None,
    multiplier: int = 5,
) -> dict | None:
    as_of_dt = _coerce_eastern_datetime(as_of)
    bars = fetch_historical_intraday_bars(
        api_key, ticker, as_of_dt.date(), multiplier=multiplier, cache=cache,
    )
    return _first_bar_at_or_after(bars, as_of_dt)


def _current_option_bar(
    api_key: str,
    option_ticker: str,
    as_of: date | datetime,
    cache: PolygonCache | None = None,
    bar_minutes: int = 5,
) -> dict | None:
    if isinstance(as_of, datetime):
        return _latest_intraday_bar_before(
            api_key, option_ticker, as_of, cache=cache, multiplier=bar_minutes,
        )
    return fetch_option_daily_bar(api_key, option_ticker, as_of, cache=cache)


def _current_equity_bar(
    api_key: str,
    symbol: str,
    as_of: date | datetime,
    cache: PolygonCache | None = None,
    bar_minutes: int = 5,
) -> dict | None:
    if isinstance(as_of, datetime):
        return _latest_intraday_bar_before(
            api_key, symbol, as_of, cache=cache, multiplier=bar_minutes,
        )
    return fetch_historical_daily_bar(api_key, symbol, as_of)


def _session_intraday_bars_before(
    api_key: str,
    ticker: str,
    as_of: date | datetime,
    cache: PolygonCache | None = None,
    multiplier: int = 5,
) -> list[dict]:
    as_of_dt = _coerce_eastern_datetime(as_of)
    bars = fetch_historical_intraday_bars(
        api_key, ticker, as_of_dt.date(), multiplier=multiplier, cache=cache,
    )
    eligible: list[dict] = []
    for bar in bars:
        ts = _bar_timestamp_eastern(bar)
        if ts and ts < as_of_dt:
            eligible.append(bar)
    return eligible


def _mark_to_market_equity(
    realized_equity: float,
    positions: list["SimPosition"],
    as_of: date | datetime,
    api_key: str,
    cache: PolygonCache,
    bar_minutes: int = 5,
) -> tuple[float, float]:
    """Return (account_equity, total_unrealized_pnl) at a timestamp."""
    total_unrealized = 0.0
    for pos in positions:
        if not pos.polygon_ticker:
            continue
        bar = _current_option_bar(
            api_key, pos.polygon_ticker, as_of, cache=cache, bar_minutes=bar_minutes,
        )
        premium = _option_bar_price(bar) if bar else 0.0
        if premium <= 0 or pos.entry_premium <= 0:
            continue
        total_unrealized += (premium - pos.entry_premium) * pos.qty * 100
    return realized_equity + total_unrealized, total_unrealized


def _current_underlying_price(
    api_key: str,
    symbol: str,
    as_of: date | datetime,
    cache: PolygonCache,
    bar_minutes: int = 5,
) -> float:
    bar = _current_equity_bar(api_key, symbol, as_of, cache=cache, bar_minutes=bar_minutes)
    if bar is None:
        return 0.0
    return _equity_bar_price(bar)


def _next_fill_option_bar(
    api_key: str,
    option_ticker: str,
    decision_time: datetime,
    cache: PolygonCache,
    bar_minutes: int = 5,
) -> dict | None:
    return _first_intraday_bar_at_or_after(
        api_key, option_ticker, decision_time, cache=cache, multiplier=bar_minutes,
    )


def _select_real_contract(
    api_key: str,
    underlying: str,
    option_type: str,
    spot: float,
    trade_date: date,
    strike_preference: str,
    expiry_preference: str,
    default_dte: int,
    contract_symbol: str | None = None,
    target_delta: float | None = None,
    min_dte: int | None = None,
    max_dte: int | None = None,
    max_spread_pct: float | None = None,
    cache: PolygonCache | None = None,
) -> dict | None:
    """Select a real Polygon option contract based on LLM preferences.

    Returns the contract dict from Polygon or None if nothing suitable found.
    """
    # Determine expiry range
    if min_dte is not None or max_dte is not None:
        resolved_min_dte = max(min_dte if min_dte is not None else config.PREFERRED_DTE_MIN, 1)
        resolved_max_dte = max_dte if max_dte is not None else config.PREFERRED_DTE_MAX
        resolved_max_dte = max(resolved_max_dte, resolved_min_dte)
        expiry_gte = trade_date + timedelta(days=resolved_min_dte)
        expiry_lte = trade_date + timedelta(days=resolved_max_dte)
    elif expiry_preference == "this_week":
        days_until_friday = (4 - trade_date.weekday()) % 7
        if days_until_friday < 2:
            days_until_friday += 7
        expiry_gte = trade_date + timedelta(days=2)
        expiry_lte = trade_date + timedelta(days=days_until_friday)
    elif expiry_preference == "next_week":
        days_until_friday = (4 - trade_date.weekday()) % 7 + 7
        expiry_gte = trade_date + timedelta(days=5)
        expiry_lte = trade_date + timedelta(days=days_until_friday + 2)
    else:  # monthly
        expiry_gte = trade_date + timedelta(days=max(7, default_dte - 7))
        expiry_lte = trade_date + timedelta(days=default_dte + 7)

    strike_band_pct = 0.35 if (contract_symbol or target_delta is not None) else 0.15
    strike_gte = round(spot * max(0.05, 1.0 - strike_band_pct), 2)
    strike_lte = round(spot * (1.0 + strike_band_pct), 2)

    contracts = fetch_polygon_option_contracts(
        api_key, underlying, option_type,
        expiry_gte, expiry_lte, strike_gte, strike_lte,
        as_of=trade_date, cache=cache,
    )
    if not contracts:
        return None

    if contract_symbol:
        exact = [c for c in contracts if str(c.get("ticker", "")).upper() == contract_symbol.upper()]
        return exact[0] if exact else None

    if max_spread_pct is not None:
        spread_filtered = []
        for contract in contracts:
            bid = float(contract.get("bid") or 0.0)
            ask = float(contract.get("ask") or 0.0)
            if ask <= 0 or bid < 0:
                spread_filtered.append(contract)
                continue
            spread_pct = (ask - bid) / ask
            if spread_pct <= max_spread_pct:
                spread_filtered.append(contract)
        if not spread_filtered:
            return None
        contracts = spread_filtered

    target_strike = target_strike_for_preference(spot, option_type, strike_preference)
    if min_dte is not None or max_dte is not None:
        lower = min_dte if min_dte is not None else max(1, (max_dte or config.PREFERRED_DTE_MAX) - 7)
        upper = max_dte if max_dte is not None else lower
        target_dte = int((lower + upper) / 2)
    else:
        target_dte = target_dte_for_expiry_preference(expiry_preference)

    def _contract_score(contract: dict) -> tuple[float, float, int, float]:
        strike = float(contract.get("strike_price", 0) or 0)
        expiry_raw = str(contract.get("expiration_date") or "")
        try:
            expiry = date.fromisoformat(expiry_raw)
            dte = max((expiry - trade_date).days, 0)
        except ValueError:
            dte = target_dte
        strike_penalty = abs(strike - target_strike) / spot * 100 if spot > 0 else 0.0
        dte_penalty = abs(dte - target_dte)
        delta_penalty = 0.0
        if target_delta is not None and strike > 0 and spot > 0:
            delta_penalty = abs(abs(approx_delta(strike, spot, max(dte, 1), option_type)) - target_delta)
        return (delta_penalty, strike_penalty, dte_penalty, dte)

    best = min(contracts, key=_contract_score)
    return best


# ---------------------------------------------------------------------------
# Simulated position
# ---------------------------------------------------------------------------

@dataclass
class SimPosition:
    underlying: str
    option_type: str
    strike: float
    entry_date: date
    expiry_date: date
    entry_premium: float
    qty: int
    conviction: float
    reasoning: str
    polygon_ticker: str = ""

    @property
    def dte_from(self) -> int:
        return (self.expiry_date - self.entry_date).days


@dataclass
class SimTrade:
    entry_date: date
    exit_date: date
    underlying: str
    option_type: str
    strike: float
    entry_premium: float
    exit_premium: float
    qty: int
    pnl: float
    exit_reason: str
    conviction: float
    polygon_ticker: str = ""
    reasoning: str = ""


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

@dataclass
class BacktestConfig:
    start_date: date
    end_date: date
    initial_equity: float = 100_000.0
    max_risk_per_trade: float = 0.40
    profit_target_pct: float = 0.50
    stop_loss_pct: float = 0.40
    time_stop_dte: int = 2
    default_dte: int = 14
    max_positions: int = 5
    llm_delay_seconds: float = 1.0
    decision_interval_minutes: int = config.SCAN_INTERVAL_MINUTES
    signal_bar_minutes: int = 5
    news_lookback_hours: int = config.NEWS_LOOKBACK_HOURS
    no_trade_minutes_after_open: int = config.NO_TRADE_MINUTES_AFTER_OPEN
    no_trade_minutes_before_close: int = config.NO_TRADE_MINUTES_BEFORE_CLOSE
    use_journal: bool = True
    journal_max_active: int = 8
    journal_max_full_display: int = 5
    journal_stale_cycles: int = 8
    journal_stale_conviction: float = 0.4


@dataclass
class BacktestResult:
    trades: list[SimTrade] = field(default_factory=list)
    equity_curve: list[tuple[str, float]] = field(default_factory=list)
    initial_equity: float = 100_000.0
    final_equity: float = 100_000.0
    total_return_pct: float = 0.0
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float | None = None
    net_pnl: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float | None = None
    profit_factor: float | None = None
    avg_conviction: float = 0.0
    days_tested: int = 0
    decision_log: list[dict] = field(default_factory=list)


def _trading_days(start: date, end: date) -> list[date]:
    """Generate weekdays between start and end (inclusive)."""
    days = []
    current = start
    while current <= end:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return days


def _previous_trading_day(trade_date: date) -> date:
    """Return the prior weekday for daily backtest signal generation."""
    previous = trade_date - timedelta(days=1)
    while previous.weekday() >= 5:
        previous -= timedelta(days=1)
    return previous


def _last_completed_trading_day(as_of: date | datetime) -> date:
    """Return the most recent fully completed session available at as_of."""
    as_of_dt = _coerce_eastern_datetime(as_of)
    session_day = as_of_dt.date()
    while session_day.weekday() >= 5:
        session_day = _previous_trading_day(session_day)
    if as_of_dt < _market_close_dt(session_day):
        return _previous_trading_day(session_day)
    return session_day


# ---------------------------------------------------------------------------
# News quality filter — remove lawsuit spam & corporate housekeeping
# ---------------------------------------------------------------------------

_JUNK_TITLE_PATTERNS = [
    # Lawsuit/legal spam
    "securities fraud",
    "class action",
    "loss recovery",
    "reminds investors",
    "encourages.*investors",
    "investor rights",
    "deadline reminder",
    "deadline tuesday",
    "deadline alert",
    "recover your losses",
    "leading law firm",
    "have suffered losses",
    "securities class action",
    # Corporate housekeeping
    "inducement grant",
    "listing rule 5635",
    "repurchase of own shares",
    "omien osakkeiden",            # Nokia Finnish filings
]

_JUNK_RE = re.compile("|".join(_JUNK_TITLE_PATTERNS), re.IGNORECASE)


def _filter_news_quality(articles: list[dict]) -> list[dict]:
    """Remove junk articles (lawsuit spam, corporate housekeeping) by title."""
    return [a for a in articles if not _JUNK_RE.search(a.get("title", ""))]


def _format_news_for_backtest(
    articles: list[dict],
    focus_symbols: list[str] | None = None,
    reference_time: datetime | None = None,
) -> str:
    items = _news_items_from_backtest_articles(articles, reference_time=reference_time)
    return format_news_for_llm(
        items,
        focus_symbols=focus_symbols,
        reference_time=reference_time,
    )


def _news_items_from_backtest_articles(
    articles: list[dict],
    reference_time: datetime | None = None,
) -> list[NewsItem]:
    items: list[NewsItem] = []
    for article in articles:
        published_raw = article.get("published_utc", "")
        try:
            published_at = datetime.fromisoformat(str(published_raw).replace("Z", "+00:00")).astimezone(EASTERN_TZ)
        except ValueError:
            published_at = reference_time or now_eastern()
        items.append(
            NewsItem(
                headline=article.get("title", ""),
                summary=(article.get("description") or "")[:1000],
                source=article.get("publisher", {}).get("name", "") or "unknown",
                symbols=[str(t).upper() for t in article.get("tickers", []) if t],
                published_at=published_at,
                url=article.get("article_url") or "",
            )
        )
    return items


def _build_performance_summary(
    closed_trades: list[dict], equity: float, initial_equity: float,
) -> str:
    """Build a running performance summary from closed trades for LLM context."""
    if not closed_trades:
        return ""

    wins = [t for t in closed_trades if (t.get("pnl") or 0) > 0]
    losses = [t for t in closed_trades if (t.get("pnl") or 0) < 0]
    total = len(closed_trades)
    n_wins = len(wins)
    n_losses = len(losses)
    win_rate = n_wins / total * 100 if total > 0 else 0
    net_pnl = sum(t.get("pnl", 0) for t in closed_trades)
    ret_pct = (equity - initial_equity) / initial_equity * 100 if initial_equity > 0 else 0

    avg_win = sum(t["pnl"] for t in wins) / n_wins if n_wins else 0
    avg_loss = sum(t["pnl"] for t in losses) / n_losses if n_losses else 0
    wl_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf") if avg_win > 0 else 0

    # Recent streak (last 5)
    recent = closed_trades[-5:]
    streak = ", ".join("W" if (t.get("pnl") or 0) > 0 else "L" for t in recent)

    # Per-ticker stats (top 5 most traded)
    ticker_stats: dict[str, dict] = {}
    for t in closed_trades:
        sym = t.get("underlying", "???")
        s = ticker_stats.setdefault(sym, {"wins": 0, "losses": 0, "pnl": 0.0, "trades": []})
        pnl = t.get("pnl", 0)
        if pnl > 0:
            s["wins"] += 1
        elif pnl < 0:
            s["losses"] += 1
        s["pnl"] += pnl
        s["trades"].append(t)
    top_tickers = sorted(ticker_stats.items(), key=lambda x: x[1]["wins"] + x[1]["losses"], reverse=True)[:5]

    lines = ["\n--- Running Performance ---"]

    # Repeat losers FIRST — most actionable signal, must not get buried
    repeat_losers = [
        (sym, s) for sym, s in ticker_stats.items()
        if s["losses"] >= 2 and s["wins"] == 0
    ]
    repeat_losers.sort(key=lambda x: x[1]["losses"], reverse=True)
    if repeat_losers:
        lines.append("WARNING — REPEAT LOSERS (you keep losing on these tickers):")
        for sym, s in repeat_losers:
            n_trades = s["losses"]
            total_lost = s["pnl"]
            avg_loss_val = total_lost / n_trades if n_trades else 0
            hold_days = []
            for t in s["trades"]:
                entry_str = t.get("entry_date", "")
                exit_str = t.get("timestamp", "")
                if entry_str and exit_str:
                    try:
                        ed = date.fromisoformat(str(entry_str)[:10])
                        xd = date.fromisoformat(str(exit_str)[:10])
                        hold_days.append((xd - ed).days)
                    except (ValueError, TypeError):
                        pass
            avg_hold = sum(hold_days) / len(hold_days) if hold_days else 0
            lines.append(
                f"  {sym}: {n_trades} trades, 0 wins, ${total_lost:+,.0f} total lost"
                f" — avg hold: {avg_hold:.1f} days, avg loss: ${avg_loss_val:+,.0f}"
            )

    # Equity momentum (last 5 trades)
    if recent:
        momentum = sum(t.get("pnl", 0) for t in recent)
        m_wins = sum(1 for t in recent if (t.get("pnl") or 0) > 0)
        m_losses = sum(1 for t in recent if (t.get("pnl") or 0) < 0)
        lines.append(
            f"Equity momentum (last {len(recent)} trades): ${momentum:+,.0f} "
            f"({m_wins}W/{m_losses}L)"
        )

    lines.extend([
        f"Total trades: {total} | Wins: {n_wins} | Losses: {n_losses} | Win rate: {win_rate:.1f}%",
        f"Net P&L: ${net_pnl:+,.2f} | Equity: ${equity:,.2f} ({ret_pct:+.1f}% vs start)",
        f"Avg winner: ${avg_win:+,.2f} | Avg loser: ${avg_loss:+,.2f} | W/L ratio: {wl_ratio:.2f}",
        f"Recent streak (last {len(recent)}): {streak}",
    ])
    if top_tickers:
        lines.append("Top tickers:")
        for sym, s in top_tickers:
            lines.append(f"  {sym}: {s['wins']}W/{s['losses']}L net=${s['pnl']:+,.2f}")

    return "\n".join(lines)


def _build_market_trend_context(
    api_key: str,
    as_of: date | datetime,
    lookback_days: int = 10,
    cache: PolygonCache | None = None,
    bar_minutes: int = 5,
) -> str:
    """Build market trend context using only information available at as_of."""
    is_intraday = isinstance(as_of, datetime)
    as_of_dt = _coerce_eastern_datetime(as_of)
    last_completed_day = _last_completed_trading_day(as_of_dt)
    start = last_completed_day - timedelta(days=int(lookback_days * 1.8))
    if is_intraday:
        lines = [
            f"Major Indices ({lookback_days}-day view) as of "
            f"{as_of_dt.strftime('%Y-%m-%d %H:%M %Z')}:"
        ]
        change_label = "now"
    else:
        lines = [f"Major Indices ({lookback_days}-day view):"]
        change_label = "today"

    for idx_sym in ["SPY", "QQQ", "IWM"]:
        bars = fetch_historical_daily_bars_range(api_key, idx_sym, start, last_completed_day)
        time_module.sleep(0.25)
        if not bars:
            continue

        current_price = float(bars[-1].get("c", 0))
        intraday_chg = 0.0

        if isinstance(as_of, datetime):
            intraday_bars = _session_intraday_bars_before(
                api_key, idx_sym, as_of_dt, cache=cache, multiplier=bar_minutes,
            )
            if intraday_bars:
                first_bar = intraday_bars[0]
                latest_bar = intraday_bars[-1]
                current_price = _equity_bar_price(latest_bar)
                session_open = float(first_bar.get("o", 0))
                intraday_chg = (
                    (current_price - session_open) / session_open * 100
                    if session_open > 0 and current_price > 0 else 0.0
                )
        else:
            today_bar = bars[-1]
            today_open = float(today_bar.get("o", 0))
            intraday_chg = (
                (current_price - today_open) / today_open * 100
                if today_open > 0 and current_price > 0 else 0.0
            )

        if current_price <= 0:
            continue

        five_d_chg = 0.0
        if len(bars) >= 6:
            ref = float(bars[-6].get("c", 0))
            if ref > 0:
                five_d_chg = (current_price - ref) / ref * 100

        ten_d_chg = 0.0
        if len(bars) >= 11:
            ref = float(bars[-11].get("c", 0))
            if ref > 0:
                ten_d_chg = (current_price - ref) / ref * 100

        recent_bars = bars[-lookback_days:] if len(bars) >= lookback_days else bars
        avg_close = sum(float(b.get("c", 0)) for b in recent_bars) / len(recent_bars)
        trend = "up" if current_price >= avg_close else "down"

        lines.append(
            f"  {idx_sym}: ${current_price:.2f}"
            f" {change_label}({intraday_chg:+.2f}%)"
            f" 5d({five_d_chg:+.1f}%)"
            f" 10d({ten_d_chg:+.1f}%)"
            f" trend={trend}"
        )

    return "\n".join(lines)


def _build_ticker_price_context(
    api_key: str,
    ticker: str,
    as_of: date | datetime,
    lookback_days: int = 10,
    cache: PolygonCache | None = None,
    bar_minutes: int = 5,
) -> tuple[str | None, float]:
    """Build price trend context for a single ticker as of a timestamp."""
    metrics = _ticker_price_metrics_as_of(
        api_key,
        ticker,
        as_of,
        lookback_days=lookback_days,
        cache=cache,
        bar_minutes=bar_minutes,
    )
    if not metrics:
        return None, 0.0

    ctx = (
        f"spot=${metrics['price']:.2f}"
        f" {'now' if metrics['is_intraday'] else 'today'}({metrics['intraday_chg']:+.1f}%)"
        f" 5d({metrics['five_d_chg']:+.1f}%)"
        f" 10d({metrics['ten_d_chg']:+.1f}%)"
        f" {'session' if metrics['is_intraday'] else 'hi/lo'}=${metrics['session_low']:.0f}/${metrics['session_high']:.0f}"
    )
    if metrics["is_intraday"]:
        ctx += (
            f" 10d_hi/lo=${metrics['recent_high']:.0f}/{metrics['recent_low']:.0f}"
        )
    return ctx, metrics["price"]


def _ticker_price_metrics_as_of(
    api_key: str,
    ticker: str,
    as_of: date | datetime,
    lookback_days: int = 10,
    cache: PolygonCache | None = None,
    bar_minutes: int = 5,
) -> dict | None:
    """Return price/trend metrics for a ticker using only info available at as_of."""
    is_intraday = isinstance(as_of, datetime)
    as_of_dt = _coerce_eastern_datetime(as_of)
    last_completed_day = _last_completed_trading_day(as_of_dt)
    start = last_completed_day - timedelta(days=int(lookback_days * 1.8))
    bars = fetch_historical_daily_bars_range(api_key, ticker, start, last_completed_day)
    if not bars:
        return None

    current_price = float(bars[-1].get("c", 0))
    intraday_chg = 0.0
    session_high = current_price
    session_low = current_price

    if isinstance(as_of, datetime):
        intraday_bars = _session_intraday_bars_before(
            api_key, ticker, as_of_dt, cache=cache, multiplier=bar_minutes,
        )
        if intraday_bars:
            latest_bar = intraday_bars[-1]
            current_price = _equity_bar_price(latest_bar)
            session_open = float(intraday_bars[0].get("o", 0))
            intraday_chg = (
                (current_price - session_open) / session_open * 100
                if session_open > 0 and current_price > 0 else 0.0
            )
            session_high = max(float(b.get("h", 0)) for b in intraday_bars)
            lows = [float(b.get("l", 0)) for b in intraday_bars if float(b.get("l", 0)) > 0]
            session_low = min(lows) if lows else current_price
    else:
        today_bar = bars[-1]
        today_open = float(today_bar.get("o", 0))
        intraday_chg = (
            (current_price - today_open) / today_open * 100
            if today_open > 0 and current_price > 0 else 0.0
        )
        session_high = float(today_bar.get("h", current_price) or current_price)
        session_low = float(today_bar.get("l", current_price) or current_price)

    if current_price <= 0:
        return None

    five_d_chg = 0.0
    if len(bars) >= 6:
        ref = float(bars[-6].get("c", 0))
        if ref > 0:
            five_d_chg = (current_price - ref) / ref * 100

    ten_d_chg = 0.0
    if len(bars) >= 11:
        ref = float(bars[-11].get("c", 0))
        if ref > 0:
            ten_d_chg = (current_price - ref) / ref * 100

    recent_bars = bars[-lookback_days:] if len(bars) >= lookback_days else bars
    recent_high = max(float(b.get("h", 0)) for b in recent_bars)
    recent_low = min(float(b.get("l", float("inf"))) for b in recent_bars)
    if not is_intraday:
        session_high = recent_high
        session_low = recent_low
    trend = "up" if current_price >= (
        sum(float(b.get("c", 0)) for b in recent_bars) / len(recent_bars)
    ) else "down"
    return {
        "price": current_price,
        "intraday_chg": intraday_chg,
        "five_d_chg": five_d_chg,
        "ten_d_chg": ten_d_chg,
        "session_high": session_high,
        "session_low": session_low,
        "recent_high": recent_high,
        "recent_low": recent_low,
        "trend": trend,
        "is_intraday": is_intraday,
    }


def _build_catalyst_reaction_context(
    api_key: str,
    news_events: list,
    as_of: date | datetime,
    cache: PolygonCache | None = None,
    bar_minutes: int = 5,
    max_events: int = 5,
) -> str:
    lines = ["Catalyst Reaction Snapshot:"]
    seen: set[str] = set()
    for event in news_events:
        symbol = next((sym.upper() for sym in event.symbols if sym), "")
        if not symbol or symbol in seen:
            continue
        metrics = _ticker_price_metrics_as_of(
            api_key,
            symbol,
            as_of,
            cache=cache,
            bar_minutes=bar_minutes,
        )
        if not metrics:
            continue
        reaction = classify_catalyst_reaction(
            event.age_minutes,
            metrics["intraday_chg"],
            metrics["five_d_chg"],
        )
        lines.append(
            f"  {symbol}: event={event.event_type}/{event.freshness}"
            f" age={event.age_minutes}m"
            f" now({metrics['intraday_chg']:+.1f}%)"
            f" 5d({metrics['five_d_chg']:+.1f}%)"
            f" trend={metrics['trend']}"
            f" reaction={reaction}"
        )
        seen.add(symbol)
        if len(seen) >= max_events:
            break
    return "\n".join(lines) if len(lines) > 1 else ""


def _annotate_closed_trades(
    closed_trades: list[dict],
    api_key: str,
    cache: PolygonCache,
) -> list[dict]:
    """Annotate closed trades with underlying price movement over holding period.

    Adds a 'context' field to all closed trades with valid entry/exit dates,
    showing what the underlying stock did while the position was held.
    Uses cached bars when available to minimize API calls.
    """
    annotated = []
    # Cache underlying bars per ticker to avoid redundant fetches
    underlying_bars_cache: dict[str, list[dict]] = {}

    for trade in closed_trades:
        trade = dict(trade)  # shallow copy to avoid mutating original
        underlying = trade.get("underlying", "")
        entry_date_str = trade.get("entry_date", "")
        exit_date_str = trade.get("timestamp", "")  # timestamp is exit date

        if underlying and entry_date_str and exit_date_str:
            try:
                entry_dt = (
                    date.fromisoformat(str(entry_date_str)[:10])
                    if isinstance(entry_date_str, str) else entry_date_str
                )
                exit_dt = (
                    date.fromisoformat(str(exit_date_str)[:10])
                    if isinstance(exit_date_str, str) else exit_date_str
                )
            except (ValueError, TypeError):
                annotated.append(trade)
                continue

            cache_key = underlying
            if cache_key not in underlying_bars_cache:
                # Fetch full range — usually already available from other queries
                bars = fetch_historical_daily_bars_range(
                    api_key, underlying, entry_dt, exit_dt,
                )
                underlying_bars_cache[cache_key] = bars

            bars = underlying_bars_cache[cache_key]
            if len(bars) >= 2:
                entry_price = float(bars[0].get("c", 0))
                exit_price = float(bars[-1].get("c", 0))
                if entry_price > 0 and exit_price > 0:
                    move_pct = (exit_price - entry_price) / entry_price * 100
                    option_type = trade.get("option_type", "")
                    direction = "bearish" if option_type == "put" else "bullish"
                    trade["context"] = (
                        f"Underlying moved: {underlying}"
                        f" ${entry_price:.0f}→${exit_price:.0f}"
                        f" ({move_pct:+.1f}%)"
                        f" while holding {direction} position"
                    )

        annotated.append(trade)

    return annotated


def _extract_top_news_tickers(
    news: list[dict],
    exclude: set[str] | None = None,
    max_tickers: int = 5,
) -> list[str]:
    """Count ticker mentions across news articles, return top N.

    Excludes broad index ETFs by default so the LLM focuses on
    individual stocks with actual options chains.
    """
    if exclude is None:
        exclude = {"SPY", "QQQ", "IWM", "DIA"}

    counts: Counter[str] = Counter()
    for article in news:
        for ticker in article.get("tickers", []):
            t = ticker.strip().upper()
            if t and t not in exclude:
                counts[t] += 1

    return [t for t, _ in counts.most_common(max_tickers)]


def _build_focus_tickers(
    news: list[dict],
    positions: list[SimPosition],
    journal: ThesisJournal | None,
    max_tickers: int = config.WATCHLIST_SIZE,
) -> list[str]:
    """Build a compact focus list from live theses, open risk, and fresh news."""
    tickers: list[str] = []
    for pos in positions:
        if pos.underlying:
            tickers.append(pos.underlying.upper())
    if journal:
        for entry in journal.active_entries():
            if entry.underlying:
                tickers.append(entry.underlying.upper())
    tickers.extend(_extract_top_news_tickers(news, max_tickers=max_tickers))
    return list(dict.fromkeys(tickers))[:max_tickers]


def _build_options_context(
    api_key: str,
    tickers: list[str],
    as_of: date | datetime,
    cache: PolygonCache,
    default_dte: int = 14,
    bar_minutes: int = 5,
) -> str:
    """Build a real options context string for the most-discussed tickers.

    For each ticker: fetch spot price, find ATM call+put contracts,
    and show the latest premium available before as_of.
    """
    if not tickers:
        return "(Backtest mode: real Polygon options data used for pricing)"

    as_of_dt = _coerce_eastern_datetime(as_of)
    trade_date = as_of_dt.date()
    lines: list[str] = [
        "Available options for the current focus tickers "
        f"(as of {as_of_dt.strftime('%H:%M %Z')}):"
    ]
    found_any = False

    expiry_gte = trade_date + timedelta(days=max(7, default_dte - 7))
    expiry_lte = trade_date + timedelta(days=default_dte + 7)

    for ticker in tickers:
        # Fetch spot price + trend context (single range query, no extra call)
        trend_ctx, spot = _build_ticker_price_context(
            api_key, ticker, as_of_dt, cache=cache, bar_minutes=bar_minutes,
        )
        if spot <= 0:
            bar = _current_equity_bar(
                api_key, ticker, as_of_dt, cache=cache, bar_minutes=bar_minutes,
            )
            if bar is None:
                continue
            spot = _equity_bar_price(bar)
            if spot <= 0:
                continue

        if trend_ctx:
            lines.append(f"  {ticker} ({trend_ctx}):")
        else:
            lines.append(f"  {ticker} (spot=${spot:.2f}):")
        strike_gte = round(spot * 0.97, 2)
        strike_lte = round(spot * 1.03, 2)

        for opt_type in ("call", "put"):
            contracts = fetch_polygon_option_contracts(
                api_key, ticker, opt_type,
                expiry_gte, expiry_lte, strike_gte, strike_lte,
                as_of=trade_date, cache=cache,
            )
            if not contracts:
                continue
            # Pick ATM contract
            best = min(contracts, key=lambda c: abs(c.get("strike_price", 0) - spot))
            opt_ticker = best.get("ticker", "")
            strike = best.get("strike_price", 0)
            expiry = best.get("expiration_date", "")
            if not opt_ticker:
                continue

            session_bars = _session_intraday_bars_before(
                api_key, opt_ticker, as_of_dt, cache=cache, multiplier=bar_minutes,
            )
            if session_bars:
                opt_bar = session_bars[-1]
                premium = _option_bar_price(opt_bar)
                vol = sum(int(b.get("v") or 0) for b in session_bars)
                lows = [float(b.get("l", 0)) for b in session_bars if float(b.get("l", 0)) > 0]
                highs = [float(b.get("h", 0)) for b in session_bars if float(b.get("h", 0)) > 0]
                session_range = (
                    f" session_range=${min(lows):.2f}-${max(highs):.2f}"
                    if lows and highs else ""
                )
            else:
                opt_bar = fetch_option_daily_bar(api_key, opt_ticker, trade_date, cache=cache)
                premium = _option_bar_price(opt_bar) if opt_bar else 0
                vol = int(opt_bar.get("v", 0)) if opt_bar else 0
                session_range = ""

            if premium > 0:
                bar_time = _bar_timestamp_eastern(opt_bar) if opt_bar else None
                bar_label = f" asof={bar_time.strftime('%H:%M')}" if bar_time else ""
                lines.append(
                    f"    {opt_ticker} {opt_type.upper()} ${strike:.2f} exp={expiry}"
                    f" premium=${premium:.2f} vol={vol}{session_range}{bar_label}"
                )
                found_any = True

        # Rate limit between tickers (range bars + contract lookups + option bars)
        time_module.sleep(1.0)

    if not found_any:
        return "(Backtest mode: real Polygon options data used for pricing)"

    return "\n".join(lines)


def _build_enriched_portfolio_context(
    equity: float,
    initial_equity: float,
    positions: list[SimPosition],
    trade_date: date | datetime,
    api_key: str,
    cache: PolygonCache,
    profit_target_pct: float,
    stop_loss_pct: float,
    time_stop_dte: int,
    bar_minutes: int = 5,
) -> str:
    """Build mark-to-market portfolio context with unrealized P&L.

    Uses the latest option bar available before as_of.
    """
    as_of_dt = _coerce_eastern_datetime(trade_date)
    trade_day = as_of_dt.date()
    ret_pct = (equity - initial_equity) / initial_equity * 100 if initial_equity > 0 else 0
    lines = [
        f"Account Equity: ${equity:,.2f} (starting: ${initial_equity:,.2f}, return: {ret_pct:+.1f}%)",
        f"As of: {as_of_dt.strftime('%Y-%m-%d %H:%M %Z')}",
        f"Open Positions: {len(positions)}",
    ]

    total_unrealized = 0.0
    for pos in positions:
        dte = (pos.expiry_date - trade_day).days
        line = (
            f"  {pos.polygon_ticker or pos.underlying} {pos.option_type} "
            f"${pos.strike:.2f} exp={pos.expiry_date} DTE={dte}"
        )
        lines.append(line)

        # Try to get current price (usually cached from exit check)
        current_premium = 0.0
        if pos.polygon_ticker:
            opt_bar = _current_option_bar(
                api_key, pos.polygon_ticker, trade_date, cache=cache, bar_minutes=bar_minutes,
            )
            if opt_bar:
                current_premium = _option_bar_price(opt_bar)

        if current_premium > 0 and pos.entry_premium > 0:
            pnl_pct = (current_premium - pos.entry_premium) / pos.entry_premium * 100
            pnl_dollar = (current_premium - pos.entry_premium) * pos.qty * 100
            total_unrealized += pnl_dollar
            lines.append(
                f"    entry=${pos.entry_premium:.2f} current=${current_premium:.2f} "
                f"unrealized={pnl_pct:+.1f}% (${pnl_dollar:+,.2f}) qty={pos.qty}"
            )
            if dte > 0:
                daily_decay = current_premium / dte
                total_daily = daily_decay * pos.qty * 100
                lines.append(f"    time decay ≈${total_daily:.0f}/day")
            # Flag proximity to exit triggers
            flags = []
            if pnl_pct >= profit_target_pct * 100 * 0.8:
                flags.append(f"approaching profit target of {profit_target_pct:.0%}")
            if pnl_pct <= -stop_loss_pct * 100 * 0.8:
                flags.append(f"approaching stop loss of {stop_loss_pct:.0%}")
            if dte <= time_stop_dte + 1:
                flags.append(f"approaching time stop ({time_stop_dte} DTE)")
            if flags:
                lines.append(f"    [{'; '.join(flags)}]")
        else:
            lines.append(
                f"    entry=${pos.entry_premium:.2f} qty={pos.qty}"
            )

    if positions:
        lines.append(f"Total unrealized P&L: ${total_unrealized:+,.2f}")

    return "\n".join(lines)


def run_backtest(bt_config: BacktestConfig) -> BacktestResult:
    """Run the full backtest using real Polygon options data."""

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    polygon_key = os.environ.get("POLYGON_API_KEY")
    if not anthropic_key:
        raise ValueError("ANTHROPIC_API_KEY required for backtesting")
    if not polygon_key:
        raise ValueError("POLYGON_API_KEY required for backtesting")

    brain = TradingBrain(api_key=anthropic_key)
    journal = ThesisJournal(
        max_active=bt_config.journal_max_active,
        max_full_display=bt_config.journal_max_full_display,
        stale_cycles=bt_config.journal_stale_cycles,
        stale_conviction=bt_config.journal_stale_conviction,
    ) if bt_config.use_journal else None
    trading_days = _trading_days(bt_config.start_date, bt_config.end_date)

    result = BacktestResult(initial_equity=bt_config.initial_equity)
    equity = bt_config.initial_equity
    positions: list[SimPosition] = []
    closed_trades: list[dict] = []
    peak_equity = equity
    max_dd = 0.0
    daily_returns: list[float] = []

    # Polygon cache — avoids redundant API calls for option bars
    cache = PolygonCache()

    log(f"backtest: {bt_config.start_date} to {bt_config.end_date} ({len(trading_days)} days)")
    log(f"initial equity: ${equity:,.2f}")
    prior_day_equity = bt_config.initial_equity
    for day_idx, trade_date in enumerate(trading_days):
        log(
            f"\n=== {trade_date} (day {day_idx + 1}/{len(trading_days)}) "
            f"realized_equity=${equity:,.2f} ==="
        )
        decision_times = _decision_timestamps_for_day(
            trade_date,
            interval_minutes=bt_config.decision_interval_minutes,
            start_delay_minutes=bt_config.no_trade_minutes_after_open,
            end_buffer_minutes=bt_config.no_trade_minutes_before_close,
        )

        for decision_time in decision_times:
            remaining: list[SimPosition] = []
            for pos in positions:
                dte = (pos.expiry_date - decision_time.date()).days
                if not pos.polygon_ticker:
                    remaining.append(pos)
                    continue

                option_bar = _current_option_bar(
                    polygon_key,
                    pos.polygon_ticker,
                    decision_time,
                    cache=cache,
                    bar_minutes=bt_config.signal_bar_minutes,
                )
                if option_bar is None:
                    remaining.append(pos)
                    continue

                current_premium = _option_bar_price(option_bar)
                if current_premium <= 0:
                    remaining.append(pos)
                    continue

                pnl_pct = (
                    (current_premium - pos.entry_premium) / pos.entry_premium
                    if pos.entry_premium > 0 else 0
                )
                exit_reason = None
                if pnl_pct >= bt_config.profit_target_pct:
                    exit_reason = "profit_target"
                elif pnl_pct <= -bt_config.stop_loss_pct:
                    exit_reason = "stop_loss"
                elif dte <= bt_config.time_stop_dte:
                    exit_reason = "time_stop"
                elif decision_time.date() >= pos.expiry_date:
                    exit_reason = "expiry"

                if exit_reason:
                    trade_pnl = (current_premium - pos.entry_premium) * pos.qty * 100
                    equity += trade_pnl
                    result.trades.append(
                        SimTrade(
                            entry_date=pos.entry_date,
                            exit_date=decision_time.date(),
                            underlying=pos.underlying,
                            option_type=pos.option_type,
                            strike=pos.strike,
                            entry_premium=pos.entry_premium,
                            exit_premium=current_premium,
                            qty=pos.qty,
                            pnl=trade_pnl,
                            exit_reason=exit_reason,
                            conviction=pos.conviction,
                            polygon_ticker=pos.polygon_ticker,
                            reasoning=pos.reasoning,
                        )
                    )
                    closed_trades.append({
                        "timestamp": decision_time.isoformat(),
                        "entry_date": pos.entry_date.isoformat(),
                        "underlying": pos.underlying,
                        "option_type": pos.option_type,
                        "entry_premium": pos.entry_premium,
                        "exit_premium": current_premium,
                        "pnl": trade_pnl,
                        "reason": exit_reason,
                        "polygon_ticker": pos.polygon_ticker,
                    })
                    log(
                        f"  {decision_time.strftime('%H:%M')} AUTO EXIT {pos.polygon_ticker} "
                        f"reason={exit_reason} pnl=${trade_pnl:+,.2f}"
                    )
                else:
                    remaining.append(pos)
            positions = remaining

            account_equity, _ = _mark_to_market_equity(
                equity,
                positions,
                decision_time,
                polygon_key,
                cache,
                bar_minutes=bt_config.signal_bar_minutes,
            )

            news_window_start = decision_time - timedelta(hours=bt_config.news_lookback_hours)
            news_raw = fetch_historical_news_window(
                polygon_key,
                news_window_start,
                decision_time,
                limit=50,
            )
            news = _filter_news_quality(news_raw)
            log(
                f"  {decision_time.strftime('%H:%M')}: "
                f"news {len(news_raw)} raw → {len(news)}"
            )
            context_focus_symbols = [p.underlying for p in positions]
            if journal:
                context_focus_symbols.extend(entry.underlying for entry in journal.active_entries())
            news_items = _news_items_from_backtest_articles(
                news,
                reference_time=decision_time,
            )
            news_events = build_news_events(news_items, reference_time=decision_time)
            news_context = _format_news_for_backtest(
                news,
                focus_symbols=list(dict.fromkeys(context_focus_symbols)),
                reference_time=decision_time,
            )
            catalyst_reaction_context = _build_catalyst_reaction_context(
                polygon_key,
                news_events,
                decision_time,
                cache=cache,
                bar_minutes=bt_config.signal_bar_minutes,
            )
            if catalyst_reaction_context:
                news_context = f"{catalyst_reaction_context}\n\n{news_context}"

            market_context = _build_market_trend_context(
                polygon_key,
                decision_time,
                cache=cache,
                bar_minutes=bt_config.signal_bar_minutes,
            )
            portfolio_context = _build_enriched_portfolio_context(
                account_equity,
                bt_config.initial_equity,
                positions,
                decision_time,
                polygon_key,
                cache,
                bt_config.profit_target_pct,
                bt_config.stop_loss_pct,
                bt_config.time_stop_dte,
                bar_minutes=bt_config.signal_bar_minutes,
            )

            journal_context = journal.to_context_str() if journal else ""
            annotated_trades = _annotate_closed_trades(closed_trades, polygon_key, cache)
            trade_history_context = format_trade_history([], annotated_trades[-10:]) if journal else ""
            perf_summary = _build_performance_summary(
                closed_trades,
                account_equity,
                bt_config.initial_equity,
            )
            if perf_summary:
                trade_history_context += perf_summary

            focus_tickers = _build_focus_tickers(news, positions, journal)
            options_context = _build_options_context(
                polygon_key,
                focus_tickers,
                decision_time,
                cache,
                bt_config.default_dte,
                bar_minutes=bt_config.signal_bar_minutes,
            )

            trade_results: list[dict] = []
            if bt_config.llm_delay_seconds > 0:
                time_module.sleep(bt_config.llm_delay_seconds)

            analysis = brain.analyze(
                portfolio_context=portfolio_context,
                news_context=news_context,
                market_context=market_context,
                options_context=options_context,
                journal_context=journal_context,
                trade_history_context=trade_history_context,
            )
            log(f"  {decision_time.strftime('%H:%M')} LLM: {analysis.analysis}")

            if journal and analysis.thesis_updates:
                journal.apply_updates(analysis.thesis_updates)
                log(
                    f"  Journal: {len(analysis.thesis_updates)} updates, "
                    f"{len(journal.active_entries())} active"
                )

            for decision in analysis.trades:
                trade_record = {
                    "underlying": decision.underlying,
                    "action": decision.action,
                    "conviction": decision.conviction,
                    "risk_pct": decision.risk_pct,
                    "reasoning": decision.reasoning,
                    "status": "skipped",
                    "skip_reason": "",
                    "contract": "",
                    "qty": 0,
                    "premium": 0.0,
                }

                if decision.action == "close_position":
                    target = decision.target_symbol
                    matched_pos = None
                    if target:
                        matched_pos = next((p for p in positions if p.polygon_ticker == target), None)
                    if matched_pos is None:
                        matched_pos = next(
                            (p for p in positions if p.underlying.upper() == decision.underlying.upper()),
                            None,
                        )
                    if matched_pos is None:
                        trade_record["skip_reason"] = "no matching position"
                        trade_results.append(trade_record)
                        continue

                    exit_bar = _next_fill_option_bar(
                        polygon_key,
                        matched_pos.polygon_ticker,
                        decision_time,
                        cache,
                        bar_minutes=bt_config.signal_bar_minutes,
                    )
                    if exit_bar is None:
                        trade_record["skip_reason"] = "no fill bar after decision time"
                        trade_results.append(trade_record)
                        continue

                    exit_premium = _option_bar_price(exit_bar)
                    if exit_premium <= 0:
                        trade_record["skip_reason"] = "invalid exit price"
                        trade_results.append(trade_record)
                        continue

                    trade_pnl = (exit_premium - matched_pos.entry_premium) * matched_pos.qty * 100
                    equity += trade_pnl
                    result.trades.append(
                        SimTrade(
                            entry_date=matched_pos.entry_date,
                            exit_date=decision_time.date(),
                            underlying=matched_pos.underlying,
                            option_type=matched_pos.option_type,
                            strike=matched_pos.strike,
                            entry_premium=matched_pos.entry_premium,
                            exit_premium=exit_premium,
                            qty=matched_pos.qty,
                            pnl=trade_pnl,
                            exit_reason="manual_close",
                            conviction=matched_pos.conviction,
                            polygon_ticker=matched_pos.polygon_ticker,
                            reasoning=matched_pos.reasoning,
                        )
                    )
                    closed_trades.append({
                        "timestamp": decision_time.isoformat(),
                        "entry_date": matched_pos.entry_date.isoformat(),
                        "underlying": matched_pos.underlying,
                        "option_type": matched_pos.option_type,
                        "entry_premium": matched_pos.entry_premium,
                        "exit_premium": exit_premium,
                        "pnl": trade_pnl,
                        "reason": "manual_close",
                        "polygon_ticker": matched_pos.polygon_ticker,
                    })
                    positions.remove(matched_pos)
                    trade_record["status"] = "executed"
                    trade_record["contract"] = matched_pos.polygon_ticker
                    trade_record["qty"] = matched_pos.qty
                    trade_record["premium"] = exit_premium
                    trade_record["pnl"] = trade_pnl
                    trade_results.append(trade_record)
                    account_equity, _ = _mark_to_market_equity(
                        equity,
                        positions,
                        decision_time,
                        polygon_key,
                        cache,
                        bar_minutes=bt_config.signal_bar_minutes,
                    )
                    continue

                if decision.action not in ("buy_call", "buy_put"):
                    trade_record["skip_reason"] = f"unsupported action in backtest: {decision.action}"
                    trade_results.append(trade_record)
                    continue

                if len(positions) >= bt_config.max_positions:
                    trade_record["skip_reason"] = "max positions reached"
                    trade_results.append(trade_record)
                    continue

                option_type = "call" if decision.action == "buy_call" else "put"
                underlying = decision.underlying
                spot = _current_underlying_price(
                    polygon_key,
                    underlying,
                    decision_time,
                    cache,
                    bar_minutes=bt_config.signal_bar_minutes,
                )
                if spot <= 0:
                    trade_record["skip_reason"] = "no price data before decision time"
                    trade_results.append(trade_record)
                    continue

                contract = _select_real_contract(
                    polygon_key,
                    underlying,
                    option_type,
                    spot,
                    trade_date,
                    decision.strike_preference or "atm",
                    decision.expiry_preference or "next_week",
                    bt_config.default_dte,
                    contract_symbol=decision.contract_symbol,
                    target_delta=decision.target_delta,
                    min_dte=decision.min_dte,
                    max_dte=decision.max_dte,
                    max_spread_pct=decision.max_spread_pct,
                    cache=cache,
                )
                if contract is None:
                    trade_record["skip_reason"] = "no contract found"
                    trade_results.append(trade_record)
                    continue

                polygon_ticker = contract.get("ticker", "")
                strike = float(contract.get("strike_price", 0))
                expiry_str = contract.get("expiration_date", "")
                if not polygon_ticker or strike <= 0 or not expiry_str:
                    trade_record["skip_reason"] = "invalid contract data"
                    trade_results.append(trade_record)
                    continue
                expiry = date.fromisoformat(expiry_str)

                entry_bar = _next_fill_option_bar(
                    polygon_key,
                    polygon_ticker,
                    decision_time,
                    cache,
                    bar_minutes=bt_config.signal_bar_minutes,
                )
                if entry_bar is None:
                    trade_record["skip_reason"] = "no fill bar after decision time"
                    trade_results.append(trade_record)
                    continue

                volume = int(entry_bar.get("v", 0) or 0)
                if volume <= 0:
                    trade_record["skip_reason"] = "zero volume"
                    trade_results.append(trade_record)
                    continue

                premium = _option_bar_price(entry_bar)
                if premium < 0.01:
                    trade_record["skip_reason"] = "premium too low"
                    trade_results.append(trade_record)
                    continue

                requested_risk_pct = min(max(decision.risk_pct, 0.0), bt_config.max_risk_per_trade)
                max_cost = account_equity * requested_risk_pct
                cost_per_contract = premium * 100
                qty = size_for_risk_budget(max_cost, cost_per_contract)
                if qty <= 0:
                    trade_record["skip_reason"] = "risk budget too small for 1 contract"
                    trade_results.append(trade_record)
                    continue

                positions.append(
                    SimPosition(
                        underlying=underlying,
                        option_type=option_type,
                        strike=strike,
                        entry_date=trade_date,
                        expiry_date=expiry,
                        entry_premium=premium,
                        qty=qty,
                        conviction=decision.conviction,
                        reasoning=decision.reasoning,
                        polygon_ticker=polygon_ticker,
                    )
                )
                trade_record["status"] = "executed"
                trade_record["contract"] = polygon_ticker
                trade_record["qty"] = qty
                trade_record["premium"] = premium
                trade_results.append(trade_record)
                account_equity, _ = _mark_to_market_equity(
                    equity,
                    positions,
                    decision_time,
                    polygon_key,
                    cache,
                    bar_minutes=bt_config.signal_bar_minutes,
                )

            thesis_updates_serialized = [
                {
                    "id": tu.id,
                    "underlying": tu.underlying,
                    "direction": tu.direction,
                    "thesis": tu.thesis,
                    "conviction": tu.conviction,
                    "status": tu.status,
                    "new_observation": tu.new_observation,
                }
                for tu in analysis.thesis_updates
            ]
            marked_equity, _ = _mark_to_market_equity(
                equity,
                positions,
                decision_time,
                polygon_key,
                cache,
                bar_minutes=bt_config.signal_bar_minutes,
            )
            result.decision_log.append({
                "date": trade_date.isoformat(),
                "decision_time": decision_time.isoformat(),
                "news_window_start": news_window_start.isoformat(),
                "market_analysis": analysis.analysis,
                "thesis_updates": thesis_updates_serialized,
                "trades_proposed": trade_results,
                "trades_executed": sum(1 for t in trade_results if t["status"] == "executed"),
                "trades_skipped": sum(1 for t in trade_results if t["status"] == "skipped"),
                "equity": marked_equity,
                "open_positions": len(positions),
            })

        day_close_equity, _ = _mark_to_market_equity(
            equity,
            positions,
            _market_close_dt(trade_date),
            polygon_key,
            cache,
            bar_minutes=bt_config.signal_bar_minutes,
        )
        day_return = day_close_equity - prior_day_equity
        prior_day_equity = day_close_equity
        daily_returns.append(day_return)
        result.equity_curve.append((trade_date.isoformat(), day_close_equity))

        if day_close_equity > peak_equity:
            peak_equity = day_close_equity
        dd = peak_equity - day_close_equity
        if dd > max_dd:
            max_dd = dd

        time_module.sleep(0.25)

    # Close any remaining positions at last day
    last_date = trading_days[-1] if trading_days else bt_config.end_date
    for pos in positions:
        exit_premium = 0.0
        if pos.polygon_ticker:
            close_time = _market_close_dt(last_date)
            option_bar = _current_option_bar(
                polygon_key,
                pos.polygon_ticker,
                close_time,
                cache=cache,
                bar_minutes=bt_config.signal_bar_minutes,
            )
            if option_bar is not None:
                exit_premium = _option_bar_price(option_bar)
            if exit_premium <= 0:
                for lookback in range(1, 4):
                    check_date = last_date - timedelta(days=lookback)
                    option_bar = fetch_option_daily_bar(
                        polygon_key, pos.polygon_ticker, check_date, cache=cache,
                    )
                    if option_bar is not None:
                        exit_premium = _option_bar_price(option_bar)
                        if exit_premium > 0:
                            break

        trade_pnl = (exit_premium - pos.entry_premium) * pos.qty * 100
        equity += trade_pnl
        result.trades.append(
            SimTrade(
                entry_date=pos.entry_date,
                exit_date=last_date,
                underlying=pos.underlying,
                option_type=pos.option_type,
                strike=pos.strike,
                entry_premium=pos.entry_premium,
                exit_premium=exit_premium,
                qty=pos.qty,
                pnl=trade_pnl,
                exit_reason="backtest_end",
                conviction=pos.conviction,
                polygon_ticker=pos.polygon_ticker,
                reasoning=pos.reasoning,
            )
        )

    # Compute results
    result.final_equity = equity
    result.total_return_pct = (equity - bt_config.initial_equity) / bt_config.initial_equity
    result.total_trades = len(result.trades)
    result.wins = sum(1 for t in result.trades if t.pnl > 0)
    result.losses = sum(1 for t in result.trades if t.pnl < 0)
    result.win_rate = (
        result.wins / (result.wins + result.losses)
        if (result.wins + result.losses) > 0 else None
    )
    result.net_pnl = sum(t.pnl for t in result.trades)
    result.max_drawdown = max_dd
    result.days_tested = len(trading_days)

    gross_profit = sum(t.pnl for t in result.trades if t.pnl > 0)
    gross_loss = abs(sum(t.pnl for t in result.trades if t.pnl < 0))
    result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else None

    convictions = [t.conviction for t in result.trades]
    result.avg_conviction = sum(convictions) / len(convictions) if convictions else 0

    # Sharpe
    if len(daily_returns) >= 2:
        mean_r = sum(daily_returns) / len(daily_returns)
        var_r = sum((r - mean_r) ** 2 for r in daily_returns) / (len(daily_returns) - 1)
        std_r = math.sqrt(var_r) if var_r > 0 else 0
        if std_r > 0:
            result.sharpe_ratio = round((mean_r / std_r) * math.sqrt(252), 2)

    return result


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_backtest_result(r: BacktestResult) -> None:
    print("=" * 60)
    print("  AI TRADER BACKTEST RESULTS")
    print("=" * 60)

    print(f"\n  Period:           {r.days_tested} trading days")
    print(f"  Initial equity:   ${r.initial_equity:,.2f}")
    print(f"  Final equity:     ${r.final_equity:,.2f}")
    print(f"  Total return:     {r.total_return_pct:+.2%}")
    print(f"  Net P&L:          ${r.net_pnl:+,.2f}")

    print(f"\n  Total trades:     {r.total_trades}")
    print(f"  Wins:             {r.wins}")
    print(f"  Losses:           {r.losses}")
    if r.win_rate is not None:
        print(f"  Win rate:         {r.win_rate:.1%}")

    print(f"\n  Max drawdown:     ${r.max_drawdown:,.2f}")
    if r.sharpe_ratio is not None:
        print(f"  Sharpe ratio:     {r.sharpe_ratio:.2f}")
    if r.profit_factor is not None:
        print(f"  Profit factor:    {r.profit_factor:.2f}")
    print(f"  Avg conviction:   {r.avg_conviction:.2f}")

    if r.trades:
        print(f"\n--- Trade Log ---")
        for t in r.trades:
            ticker_str = f" [{t.polygon_ticker}]" if t.polygon_ticker else ""
            print(
                f"  {t.entry_date} -> {t.exit_date} | "
                f"{t.underlying} {t.option_type} ${t.strike:.0f}{ticker_str} | "
                f"entry=${t.entry_premium:.2f} exit=${t.exit_premium:.2f} | "
                f"qty={t.qty} pnl=${t.pnl:+,.2f} | "
                f"{t.exit_reason} (conv={t.conviction:.2f})"
                + (f" | {t.reasoning[:80]}" if t.reasoning else "")
            )

    print("=" * 60)


def save_backtest_result(r: BacktestResult, path: Path) -> None:
    """Save backtest results to a JSON file."""
    out = {
        "initial_equity": r.initial_equity,
        "final_equity": r.final_equity,
        "total_return_pct": r.total_return_pct,
        "net_pnl": r.net_pnl,
        "total_trades": r.total_trades,
        "wins": r.wins,
        "losses": r.losses,
        "win_rate": r.win_rate,
        "max_drawdown": r.max_drawdown,
        "sharpe_ratio": r.sharpe_ratio,
        "profit_factor": r.profit_factor,
        "avg_conviction": r.avg_conviction,
        "days_tested": r.days_tested,
        "equity_curve": [{"date": d, "equity": e} for d, e in r.equity_curve],
        "trades": [
            {
                "entry_date": t.entry_date.isoformat(),
                "exit_date": t.exit_date.isoformat(),
                "underlying": t.underlying,
                "option_type": t.option_type,
                "strike": t.strike,
                "entry_premium": t.entry_premium,
                "exit_premium": t.exit_premium,
                "qty": t.qty,
                "pnl": t.pnl,
                "exit_reason": t.exit_reason,
                "conviction": t.conviction,
                "polygon_ticker": t.polygon_ticker,
                "reasoning": t.reasoning,
            }
            for t in r.trades
        ],
        "decision_log": r.decision_log,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    log(f"results saved to {path}")


def save_debug_log(r: BacktestResult, path: Path) -> None:
    """Write a human-readable markdown debug log of the backtest."""
    lines: list[str] = []

    # Header
    win_loss = f"{r.wins}W/{r.losses}L" if r.total_trades else "0 trades"
    lines.append("# Backtest Debug Log")
    lines.append(
        f"Period: {r.days_tested} trading days | "
        f"Final: ${r.final_equity:,.0f} ({r.total_return_pct:+.1%}) | {win_loss}"
    )
    lines.append("")

    for entry in r.decision_log:
        lines.append("---")
        lines.append("")
        heading = entry.get("decision_time") or entry["date"]
        lines.append(f"## {heading}")
        lines.append("")

        # Market analysis
        analysis_text = entry.get("market_analysis", "")
        if analysis_text:
            lines.append(f"**Market Analysis:** {analysis_text}")
            lines.append("")

        # Thesis updates
        thesis_updates = entry.get("thesis_updates", [])
        if thesis_updates:
            lines.append("**Thesis Updates:**")
            for tu in thesis_updates:
                tid = tu.get("id") or "new"
                status = tu.get("status", "")
                tag = "NEW" if tid == "new" or not tu.get("id") else "UPDATE"
                lines.append(
                    f"- [{tag}] {tid}: {tu.get('underlying', '?')} "
                    f"{tu.get('direction', '?')} — {tu.get('thesis', '')} "
                    f"(conviction: {tu.get('conviction', 0)}, {status})"
                )
            lines.append("")

        # Trades
        trades_proposed = entry.get("trades_proposed", [])
        if trades_proposed:
            lines.append("**Trades:**")
            for tr in trades_proposed:
                action = tr.get("action", "?").upper().replace("_", " ")
                underlying = tr.get("underlying", "?")
                conv = tr.get("conviction", 0)
                risk = tr.get("risk_pct", 0)
                reasoning = tr.get("reasoning", "")
                reasoning_short = reasoning[:100] if reasoning else ""
                lines.append(
                    f"- {action} {underlying} | conviction={conv:.2f} "
                    f"risk={risk:.0%} | \"{reasoning_short}\""
                )
                status = tr.get("status", "skipped")
                if status == "executed":
                    contract = tr.get("contract", "")
                    qty = tr.get("qty", 0)
                    premium = tr.get("premium", 0)
                    if tr.get("action") == "close_position":
                        pnl = tr.get("pnl", 0)
                        lines.append(
                            f"  → Closed: {contract} qty={qty} "
                            f"exit=${premium:.2f} pnl=${pnl:+,.2f}"
                        )
                    else:
                        lines.append(
                            f"  → Executed: {contract} qty={qty} premium=${premium:.2f}"
                        )
                else:
                    skip_reason = tr.get("skip_reason", "unknown")
                    lines.append(f"  → SKIPPED: {skip_reason}")
        else:
            lines.append("**Trades:** None")
        lines.append("")

        # Equity / positions
        eq = entry.get("equity", 0)
        pos_count = entry.get("open_positions", 0)
        lines.append(f"**Equity:** ${eq:,.0f} | Positions: {pos_count}")
        lines.append("")

    lines.append("---")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    log(f"debug log saved to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest the AI options trader on historical data."
    )
    parser.add_argument(
        "--start", required=True,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end", required=True,
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--equity", type=float, default=100_000,
        help="Starting equity (default: 100000)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Save results to JSON file",
    )
    parser.add_argument(
        "--delay", type=float, default=1.0,
        help="Seconds between LLM calls (cost control). Default: 1.0",
    )
    parser.add_argument(
        "--decision-interval", type=int, default=config.SCAN_INTERVAL_MINUTES,
        help="Minutes between simulated decision points (default: live scan interval)",
    )
    parser.add_argument(
        "--bar-minutes", type=int, default=5,
        help="Intraday bar size for state/fills (default: 5)",
    )
    parser.add_argument(
        "--news-lookback-hours", type=int, default=config.NEWS_LOOKBACK_HOURS,
        help="Hours of news to include before each simulated decision time",
    )
    parser.add_argument(
        "--no-journal", action="store_true",
        help="Disable thesis journal and trade history (stateless mode)",
    )
    parser.add_argument(
        "--journal-max-active", type=int, default=8,
        help="Max active theses in journal (0=unlimited, default=8)",
    )
    parser.add_argument(
        "--journal-max-display", type=int, default=5,
        help="Max theses shown in full detail (0=all, default=5)",
    )
    args = parser.parse_args()

    env_path = Path(__file__).with_name(".env")
    if not env_path.exists():
        env_path = Path(__file__).parent.parent / "momentum_trader" / ".env"
    load_dotenv(env_path, override=True)

    bt_config = BacktestConfig(
        start_date=date.fromisoformat(args.start),
        end_date=date.fromisoformat(args.end),
        initial_equity=args.equity,
        llm_delay_seconds=args.delay,
        decision_interval_minutes=args.decision_interval,
        signal_bar_minutes=args.bar_minutes,
        news_lookback_hours=args.news_lookback_hours,
        use_journal=not args.no_journal,
        journal_max_active=args.journal_max_active,
        journal_max_full_display=args.journal_max_display,
    )

    result = run_backtest(bt_config)
    print_backtest_result(result)

    if args.output:
        output_path = Path(args.output)
        save_backtest_result(result, output_path)
        save_debug_log(result, Path(f"{output_path}.debug.md"))


if __name__ == "__main__":
    run()
