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
import time as time_module
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv

from . import config
from .brain import TradingBrain, TradeDecision
from .db import format_trade_history
from .journal import ThesisJournal
from .utils import EASTERN_TZ, log


# ---------------------------------------------------------------------------
# Historical data fetching (Polygon — equities)
# ---------------------------------------------------------------------------

def _polygon_request(api_key: str, path: str, params: dict | None = None) -> dict:
    import requests

    params = params or {}
    params["apiKey"] = api_key
    url = f"https://api.polygon.io{path}"
    response = requests.get(url, params=params, timeout=30)
    if response.status_code == 429:
        time_module.sleep(12)
        response = requests.get(url, params=params, timeout=30)
    if response.status_code >= 400:
        raise RuntimeError(f"Polygon {response.status_code}: {response.text[:200]}")
    return response.json()


def fetch_historical_news(
    api_key: str, trade_date: date, limit: int = 50,
) -> list[dict]:
    """Fetch news for a given date from Polygon."""
    start = f"{trade_date}T00:00:00Z"
    end = f"{trade_date}T23:59:59Z"
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
        log(f"news fetch error for {trade_date}: {exc}")
        return []
    return data.get("results", [])


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


# ---------------------------------------------------------------------------
# Polygon options data
# ---------------------------------------------------------------------------

@dataclass
class PolygonCache:
    """In-memory cache for Polygon option contract lookups and daily bars."""
    # (underlying, type, expiry_gte, expiry_lte, strike_gte, strike_lte) -> list[contract_dict]
    contracts: dict[tuple, list[dict]] = field(default_factory=dict)
    # option_ticker -> {date_iso -> bar_dict}
    option_bars: dict[str, dict[str, dict]] = field(default_factory=dict)


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


def _select_real_contract(
    api_key: str,
    underlying: str,
    option_type: str,
    spot: float,
    trade_date: date,
    strike_preference: str,
    expiry_preference: str,
    default_dte: int,
    cache: PolygonCache | None = None,
) -> dict | None:
    """Select a real Polygon option contract based on LLM preferences.

    Returns the contract dict from Polygon or None if nothing suitable found.
    """
    # Determine expiry range
    if expiry_preference == "this_week":
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

    # Strike range: ±15% of spot
    strike_gte = round(spot * 0.85, 2)
    strike_lte = round(spot * 1.15, 2)

    contracts = fetch_polygon_option_contracts(
        api_key, underlying, option_type,
        expiry_gte, expiry_lte, strike_gte, strike_lte,
        as_of=trade_date, cache=cache,
    )
    if not contracts:
        return None

    # Determine target strike
    if strike_preference == "atm":
        target_strike = spot
    elif strike_preference == "otm":
        if option_type == "call":
            target_strike = spot * 1.03
        else:
            target_strike = spot * 0.97
    else:  # itm
        if option_type == "call":
            target_strike = spot * 0.97
        else:
            target_strike = spot * 1.03

    # Pick closest contract to target strike
    best = min(
        contracts,
        key=lambda c: abs(c.get("strike_price", 0) - target_strike),
    )
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
    use_journal: bool = True


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


def _trading_days(start: date, end: date) -> list[date]:
    """Generate weekdays between start and end (inclusive)."""
    days = []
    current = start
    while current <= end:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return days


def _format_news_for_backtest(articles: list[dict]) -> str:
    lines = []
    for a in articles[:30]:
        title = a.get("title", "")
        desc = a.get("description", "")[:300]
        tickers = ", ".join(a.get("tickers", []))
        source = a.get("publisher", {}).get("name", "")
        ts = a.get("published_utc", "")[:16]
        lines.append(f"[{ts}] ({source}) {title}")
        if tickers:
            lines.append(f"  Tickers: {tickers}")
        if desc:
            lines.append(f"  {desc}")
        lines.append("")
    return "\n".join(lines) if lines else "No news available."


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
        s = ticker_stats.setdefault(sym, {"wins": 0, "losses": 0, "pnl": 0.0})
        pnl = t.get("pnl", 0)
        if pnl > 0:
            s["wins"] += 1
        elif pnl < 0:
            s["losses"] += 1
        s["pnl"] += pnl
    top_tickers = sorted(ticker_stats.items(), key=lambda x: x[1]["wins"] + x[1]["losses"], reverse=True)[:5]

    lines = [
        "\n--- Running Performance ---",
        f"Total trades: {total} | Wins: {n_wins} | Losses: {n_losses} | Win rate: {win_rate:.1f}%",
        f"Net P&L: ${net_pnl:+,.2f} | Equity: ${equity:,.2f} ({ret_pct:+.1f}% vs start)",
        f"Avg winner: ${avg_win:+,.2f} | Avg loser: ${avg_loss:+,.2f} | W/L ratio: {wl_ratio:.2f}",
        f"Recent streak (last {len(recent)}): {streak}",
    ]
    if top_tickers:
        lines.append("Top tickers:")
        for sym, s in top_tickers:
            lines.append(f"  {sym}: {s['wins']}W/{s['losses']}L net=${s['pnl']:+,.2f}")

    return "\n".join(lines)


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


def _build_options_context(
    api_key: str,
    tickers: list[str],
    trade_date: date,
    cache: PolygonCache,
    default_dte: int = 14,
) -> str:
    """Build a real options context string for the most-discussed tickers.

    For each ticker: fetch spot price, find ATM call+put contracts,
    get today's bar for premium & volume.
    """
    if not tickers:
        return "(Backtest mode: real Polygon options data used for pricing)"

    lines: list[str] = ["Available options for today's most-discussed tickers:"]
    found_any = False

    expiry_gte = trade_date + timedelta(days=max(7, default_dte - 7))
    expiry_lte = trade_date + timedelta(days=default_dte + 7)

    for ticker in tickers:
        # Fetch spot price
        bar = fetch_historical_daily_bar(api_key, ticker, trade_date)
        if bar is None:
            continue
        spot = float(bar.get("close") or bar.get("c") or 0)
        if spot <= 0:
            continue

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

            opt_bar = fetch_option_daily_bar(api_key, opt_ticker, trade_date, cache=cache)
            premium = _option_bar_price(opt_bar) if opt_bar else 0
            vol = opt_bar.get("v", 0) if opt_bar else 0

            if premium > 0:
                lines.append(
                    f"    {opt_ticker} {opt_type.upper()} ${strike:.2f} exp={expiry}"
                    f" premium=${premium:.2f} vol={vol}"
                )
                found_any = True

        # Rate limit between tickers
        time_module.sleep(0.3)

    if not found_any:
        return "(Backtest mode: real Polygon options data used for pricing)"

    return "\n".join(lines)


def _build_enriched_portfolio_context(
    equity: float,
    initial_equity: float,
    positions: list[SimPosition],
    trade_date: date,
    api_key: str,
    cache: PolygonCache,
    profit_target_pct: float,
    stop_loss_pct: float,
    time_stop_dte: int,
) -> str:
    """Build mark-to-market portfolio context with unrealized P&L.

    Uses option bars that are almost always already cached from
    the exit-check loop earlier in the same day.
    """
    ret_pct = (equity - initial_equity) / initial_equity * 100 if initial_equity > 0 else 0
    lines = [
        f"Account Equity: ${equity:,.2f} (starting: ${initial_equity:,.2f}, return: {ret_pct:+.1f}%)",
        f"Open Positions: {len(positions)}",
    ]

    total_unrealized = 0.0
    for pos in positions:
        dte = (pos.expiry_date - trade_date).days
        line = (
            f"  {pos.polygon_ticker or pos.underlying} {pos.option_type} "
            f"${pos.strike:.2f} exp={pos.expiry_date} DTE={dte}"
        )
        lines.append(line)

        # Try to get current price (usually cached from exit check)
        current_premium = 0.0
        if pos.polygon_ticker:
            opt_bar = fetch_option_daily_bar(api_key, pos.polygon_ticker, trade_date, cache=cache)
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
    journal = ThesisJournal() if bt_config.use_journal else None
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

    for day_idx, trade_date in enumerate(trading_days):
        day_start_equity = equity
        log(f"\n=== {trade_date} (day {day_idx + 1}/{len(trading_days)}) equity=${equity:,.2f} ===")

        # 1. Check existing positions for exits
        remaining: list[SimPosition] = []
        for pos in positions:
            dte = (pos.expiry_date - trade_date).days

            # Get current option price from cache/Polygon
            if not pos.polygon_ticker:
                remaining.append(pos)
                continue

            option_bar = fetch_option_daily_bar(
                polygon_key, pos.polygon_ticker, trade_date, cache=cache,
            )
            if option_bar is None:
                # No bar for this day (holiday, illiquid) — keep position
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
            elif trade_date >= pos.expiry_date:
                exit_reason = "expiry"

            if exit_reason:
                trade_pnl = (current_premium - pos.entry_premium) * pos.qty * 100
                equity += trade_pnl
                sim_trade = SimTrade(
                    entry_date=pos.entry_date,
                    exit_date=trade_date,
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
                )
                result.trades.append(sim_trade)
                closed_trades.append({
                    "timestamp": trade_date.isoformat(),
                    "underlying": pos.underlying,
                    "entry_premium": pos.entry_premium,
                    "exit_premium": current_premium,
                    "pnl": trade_pnl,
                    "reason": exit_reason,
                    "polygon_ticker": pos.polygon_ticker,
                })
                log(
                    f"  EXIT {pos.polygon_ticker} "
                    f"reason={exit_reason} pnl=${trade_pnl:+,.2f}"
                )
            else:
                remaining.append(pos)

        positions = remaining

        # 2. Fetch news for this day
        news = fetch_historical_news(polygon_key, trade_date)
        news_context = _format_news_for_backtest(news)

        # 3. Build market context from index prices
        market_lines = ["Major Indices:"]
        for idx_sym in ["SPY", "QQQ", "IWM"]:
            bar = fetch_historical_daily_bar(polygon_key, idx_sym, trade_date)
            if bar:
                o = float(bar.get("open") or bar.get("o") or 0)
                c = float(bar.get("close") or bar.get("c") or 0)
                chg = ((c - o) / o * 100) if o > 0 else 0
                market_lines.append(f"  {idx_sym}: ${c:.2f} ({chg:+.2f}%)")
        market_context = "\n".join(market_lines)

        # 4. Portfolio context (enriched with mark-to-market)
        portfolio_context = _build_enriched_portfolio_context(
            equity, bt_config.initial_equity, positions, trade_date,
            polygon_key, cache,
            bt_config.profit_target_pct, bt_config.stop_loss_pct,
            bt_config.time_stop_dte,
        )

        # 5. Build journal and trade history context
        journal_context = journal.to_context_str() if journal else ""
        trade_history_context = format_trade_history([], closed_trades[-10:]) if journal else ""

        # 5b. Append running performance summary
        perf_summary = _build_performance_summary(closed_trades, equity, bt_config.initial_equity)
        if perf_summary:
            trade_history_context += perf_summary

        # 5c. Build real options context from top news tickers
        top_tickers = _extract_top_news_tickers(news)
        options_context = _build_options_context(
            polygon_key, top_tickers, trade_date, cache, bt_config.default_dte,
        )

        # 6. Ask LLM (only if we have room for more positions)
        if len(positions) < bt_config.max_positions:
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
            log(f"  LLM: {analysis.analysis[:120]}")

            # Apply thesis journal updates
            if journal and analysis.thesis_updates:
                journal.apply_updates(analysis.thesis_updates)
                log(f"  Journal: {len(analysis.thesis_updates)} updates, {len(journal.active_entries())} active")

            # 7. Execute new trades
            for decision in analysis.trades:
                if decision.action == "close_position":
                    continue
                if len(positions) >= bt_config.max_positions:
                    break

                option_type = "call" if decision.action == "buy_call" else "put"
                underlying = decision.underlying

                # Get underlying price
                bar = fetch_historical_daily_bar(polygon_key, underlying, trade_date)
                if bar is None:
                    log(f"  SKIP {underlying}: no price data")
                    continue
                spot = float(bar.get("close") or bar.get("c") or 0)
                if spot <= 0:
                    continue

                # Find a real option contract via Polygon
                contract = _select_real_contract(
                    polygon_key, underlying, option_type, spot, trade_date,
                    decision.strike_preference or "atm",
                    decision.expiry_preference or "next_week",
                    bt_config.default_dte,
                    cache=cache,
                )
                if contract is None:
                    log(f"  SKIP {underlying}: no matching option contracts found")
                    continue

                polygon_ticker = contract.get("ticker", "")
                strike = float(contract.get("strike_price", 0))
                expiry_str = contract.get("expiration_date", "")
                if not polygon_ticker or strike <= 0 or not expiry_str:
                    log(f"  SKIP {underlying}: invalid contract data")
                    continue
                expiry = date.fromisoformat(expiry_str)

                # Get entry-day price from Polygon
                entry_bar = fetch_option_daily_bar(
                    polygon_key, polygon_ticker, trade_date, cache=cache,
                )
                if entry_bar is None:
                    log(f"  SKIP {polygon_ticker}: no option price data on {trade_date}")
                    continue

                # Skip zero-volume contracts
                volume = entry_bar.get("v", 0)
                if volume <= 0:
                    log(f"  SKIP {polygon_ticker}: zero volume")
                    continue

                premium = _option_bar_price(entry_bar)
                if premium < 0.01:
                    log(f"  SKIP {polygon_ticker}: premium too low (${premium:.4f})")
                    continue

                # Pre-fetch all future bars for this contract through expiry
                fetch_option_daily_bars_range(
                    polygon_key, polygon_ticker,
                    trade_date + timedelta(days=1), expiry,
                    cache=cache,
                )

                # Position sizing (40% max rule)
                max_cost = equity * min(decision.risk_pct, bt_config.max_risk_per_trade)
                cost_per_contract = premium * 100
                qty = max(1, int(max_cost / cost_per_contract))
                total_cost = qty * cost_per_contract

                if total_cost > equity * bt_config.max_risk_per_trade:
                    qty = max(1, int((equity * bt_config.max_risk_per_trade) / cost_per_contract))
                    total_cost = qty * cost_per_contract

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
                log(
                    f"  OPEN {polygon_ticker} "
                    f"premium=${premium:.2f} qty={qty} cost=${total_cost:,.2f} "
                    f"vol={volume} conviction={decision.conviction:.2f}"
                )

        # Track daily equity
        day_return = equity - day_start_equity
        daily_returns.append(day_return)
        result.equity_curve.append((trade_date.isoformat(), equity))

        if equity > peak_equity:
            peak_equity = equity
        dd = peak_equity - equity
        if dd > max_dd:
            max_dd = dd

        # Rate limit for Polygon
        time_module.sleep(0.5)

    # Close any remaining positions at last day
    last_date = trading_days[-1] if trading_days else bt_config.end_date
    for pos in positions:
        exit_premium = 0.0
        if pos.polygon_ticker:
            # Try last_date, then look back up to 3 days
            for lookback in range(4):
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
            }
            for t in r.trades
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    log(f"results saved to {path}")


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
        "--no-journal", action="store_true",
        help="Disable thesis journal and trade history (stateless mode)",
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
        use_journal=not args.no_journal,
    )

    result = run_backtest(bt_config)
    print_backtest_result(result)

    if args.output:
        save_backtest_result(result, Path(args.output))


if __name__ == "__main__":
    run()
