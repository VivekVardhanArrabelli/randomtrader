"""Main autonomous trading loop.

Runs continuously during market hours:
1. Fetches portfolio state
2. Checks existing positions for risk exits
3. Loads thesis journal + trade history
4. Fetches news and market data
5. Sends everything to the LLM brain
6. LLM updates journal + makes trade decisions
7. Executes trade decisions
8. Sleeps until next cycle
"""

from __future__ import annotations

import json
import os
import time as time_module
from datetime import datetime, time, timedelta, date
from pathlib import Path

from dotenv import load_dotenv

from . import config
from .alpaca_client import AlpacaClient
from .brain import TradingBrain
from .candidates import (
    build_candidate_ideas,
    format_candidate_table,
    select_candidate_finalists,
)
from .db import (
    AIDecisionRecord,
    AITradeLogger,
    PortfolioSnapshotRecord,
    expression_guidance_lines,
    format_trade_history,
)
from .executor import _execute_close, execute_trade, reconcile_pending_orders
from .journal import ThesisJournal
from .llm import api_key_env_name, infer_provider, resolve_api_key
from .news import (
    build_news_events,
    classify_catalyst_reaction,
    expand_symbols_with_relationships,
    fetch_news,
    fetch_targeted_news,
    format_symbol_setup_context,
    format_news_for_llm,
    map_best_events_by_symbol,
    merge_news_items,
    rank_symbols_from_events,
    score_news_event,
)
from .options import fetch_option_chain, format_chain_for_llm
from .portfolio import get_portfolio_state
from .risk import PositionRiskAlert, PositionRiskState, assess_position_risk, evaluate_trade_risk
from .utils import (
    EASTERN_TZ,
    is_market_open,
    log,
    now_eastern,
    prioritized_symbol_watchlist,
)


def _within_trading_window(current: datetime) -> bool:
    """Check if we're within the active trading window."""
    market_open = time(9, 30, tzinfo=EASTERN_TZ)
    market_close = time(16, 0, tzinfo=EASTERN_TZ)
    ct = current.timetz()
    if not (market_open <= ct <= market_close):
        return False

    open_minutes = 9 * 60 + 30 + config.NO_TRADE_MINUTES_AFTER_OPEN
    close_minutes = 16 * 60 - config.NO_TRADE_MINUTES_BEFORE_CLOSE
    no_trade_start = time(open_minutes // 60, open_minutes % 60, tzinfo=EASTERN_TZ)
    no_trade_end = time(close_minutes // 60, close_minutes % 60, tzinfo=EASTERN_TZ)
    return no_trade_start <= ct <= no_trade_end


def _portfolio_snapshot_record(portfolio) -> PortfolioSnapshotRecord:
    """Build a normalized end-of-cycle portfolio snapshot for reporting."""
    positions: list[dict] = []
    for pos in portfolio.option_positions:
        positions.append(
            {
                "asset_type": "option",
                "symbol": pos.symbol,
                "underlying": pos.underlying,
                "option_type": pos.option_type,
                "strike": pos.strike,
                "expiration": pos.expiration,
                "qty": pos.qty,
                "avg_entry_price": pos.avg_entry_price,
                "current_price": pos.current_price,
                "market_value": pos.market_value,
                "unrealized_pl": pos.unrealized_pl,
                "cost_basis": pos.cost_basis,
                "pnl_pct": pos.pnl_pct,
                "dte": pos.dte,
                "risk_alert": pos.risk_alert,
            }
        )
    for pos in portfolio.equity_positions:
        positions.append(
            {
                "asset_type": "stock",
                "symbol": pos.symbol,
                "underlying": pos.symbol,
                "qty": pos.qty,
                "avg_entry_price": pos.avg_entry_price,
                "current_price": pos.current_price,
                "market_value": pos.market_value,
                "unrealized_pl": pos.unrealized_pl,
                "cost_basis": pos.cost_basis,
                "pnl_pct": pos.pnl_pct,
            }
        )

    return PortfolioSnapshotRecord(
        timestamp=now_eastern(),
        equity=portfolio.account.equity,
        cash=portfolio.account.cash,
        buying_power=portfolio.account.buying_power,
        day_pl=portfolio.account.day_pl,
        total_options_exposure=portfolio.total_options_exposure,
        total_equity_exposure=portfolio.total_equity_exposure,
        total_exposure=portfolio.total_exposure,
        open_option_count=portfolio.open_option_count,
        open_equity_count=portfolio.open_equity_count,
        positions_json=json.dumps(positions),
    )


def _compute_bar_trends(bars: list[dict]) -> dict:
    """Compute trend metrics from a list of daily bars (ascending by date).

    Returns dict with keys: price, intraday_chg, five_d_chg, ten_d_chg, trend.
    """
    if not bars:
        return {}
    today_bar = bars[-1]
    today_close = float(today_bar.get("c") or today_bar.get("close") or 0)
    today_open = float(today_bar.get("o") or today_bar.get("open") or 0)
    if today_close <= 0:
        return {}

    intraday_chg = ((today_close - today_open) / today_open * 100) if today_open > 0 else 0

    five_d_chg = 0.0
    if len(bars) >= 6:
        ref = float(bars[-6].get("c") or bars[-6].get("close") or 0)
        if ref > 0:
            five_d_chg = (today_close - ref) / ref * 100

    ten_d_chg = 0.0
    if len(bars) >= 11:
        ref = float(bars[-11].get("c") or bars[-11].get("close") or 0)
        if ref > 0:
            ten_d_chg = (today_close - ref) / ref * 100

    lookback = min(10, len(bars))
    recent_bars = bars[-lookback:]
    avg_close = sum(float(b.get("c") or b.get("close") or 0) for b in recent_bars) / len(recent_bars)
    trend = "up" if today_close >= avg_close else "down"

    recent_high = max(float(b.get("h") or b.get("high") or 0) for b in recent_bars)
    recent_low = min(float(b.get("l") or b.get("low") or float("inf")) for b in recent_bars)
    if recent_high > recent_low:
        range_pos_pct = max(0.0, min(100.0, (today_close - recent_low) / (recent_high - recent_low) * 100))
    else:
        range_pos_pct = 50.0
    if range_pos_pct >= 85:
        range_label = "near_10d_high"
    elif range_pos_pct <= 15:
        range_label = "near_10d_low"
    else:
        range_label = "mid_range"

    return {
        "price": today_close,
        "intraday_chg": intraday_chg,
        "five_d_chg": five_d_chg,
        "ten_d_chg": ten_d_chg,
        "trend": trend,
        "high": recent_high,
        "low": recent_low,
        "range_pos_pct": range_pos_pct,
        "range_label": range_label,
    }


def _get_market_context(alpaca: AlpacaClient) -> str:
    """Build market overview context with multi-day trends for the LLM."""
    index_symbols = ["SPY", "QQQ", "IWM"]
    today = now_eastern().date()
    start = (today - timedelta(days=18)).isoformat()  # buffer for weekends/holidays

    lines = ["Major Indices (10-day view):"]
    for sym in index_symbols:
        try:
            bars = alpaca.get_bars(sym, timeframe="1Day", start=start, limit=15)
        except Exception:
            bars = []

        t = _compute_bar_trends(bars)
        if not t:
            # Fallback to snapshot
            try:
                snaps = alpaca.get_snapshots([sym])
                snap = snaps.get(sym, {})
                trade = snap.get("latestTrade") or snap.get("latest_trade") or {}
                price = float(trade.get("p") or trade.get("price") or 0)
                prev = float((snap.get("prevDailyBar") or snap.get("prev_daily_bar") or {}).get("c") or 0)
                chg = ((price - prev) / prev * 100) if prev > 0 else 0
                lines.append(f"  {sym}: ${price:.2f} ({chg:+.2f}%)")
            except Exception:
                pass
            continue

        lines.append(
            f"  {sym}: ${t['price']:.2f}"
            f" today({t['intraday_chg']:+.2f}%)"
            f" 5d({t['five_d_chg']:+.1f}%)"
            f" 10d({t['ten_d_chg']:+.1f}%)"
            f" trend={t['trend']}"
        )

    # Sector ETFs (5-day view)
    sector_etfs = ["XLK", "XLE", "XLF", "XLV", "XLI"]
    sector_lines = []
    for sym in sector_etfs:
        try:
            bars = alpaca.get_bars(sym, timeframe="1Day", start=start, limit=15)
        except Exception:
            bars = []
        t = _compute_bar_trends(bars)
        if t:
            sector_lines.append(
                f"  {sym}: ${t['price']:.2f}"
                f" today({t['intraday_chg']:+.1f}%)"
                f" 5d({t['five_d_chg']:+.1f}%)"
                f" trend={t['trend']}"
            )
    if sector_lines:
        lines.append("\nSector ETFs (5-day view):")
        lines.extend(sector_lines)

    # Volatility (VIXY as VIX proxy)
    try:
        vixy_bars = alpaca.get_bars("VIXY", timeframe="1Day", start=start, limit=15)
        vt = _compute_bar_trends(vixy_bars)
        if vt:
            level = "elevated" if vt["price"] > 30 else "high" if vt["price"] > 25 else "moderate" if vt["price"] > 20 else "low"
            lines.append(
                f"\nVolatility (VIXY): ${vt['price']:.2f}"
                f" today({vt['intraday_chg']:+.1f}%)"
                f" 5d({vt['five_d_chg']:+.1f}%)"
                f" 10d({vt['ten_d_chg']:+.1f}%)"
                f" level={level}"
                f" — options premiums are {'expensive' if level in ('elevated', 'high') else 'normal'}"
            )
    except Exception:
        pass

    # Top movers
    try:
        movers = alpaca.get_movers(top=10)
        if movers:
            lines.append(f"\nTop Movers: {', '.join(movers[:10])}")
    except Exception:
        pass

    return "\n".join(lines)


def _get_ticker_trend(
    alpaca: AlpacaClient,
    symbol: str,
    metrics_cache: dict[str, dict | None] | None = None,
) -> str | None:
    metrics = _get_ticker_trend_metrics(alpaca, symbol, metrics_cache=metrics_cache)
    if not metrics:
        return None

    return (
        f"spot=${metrics['price']:.2f}"
        f" today({metrics['intraday_chg']:+.1f}%)"
        f" 5d({metrics['five_d_chg']:+.1f}%)"
        f" 10d({metrics['ten_d_chg']:+.1f}%)"
        f" hi/lo=${metrics['high']:.0f}/${metrics['low']:.0f}"
        f" range={metrics['range_pos_pct']:.0f}% {metrics['range_label']}"
    )


def _get_ticker_trend_metrics(
    alpaca: AlpacaClient,
    symbol: str,
    metrics_cache: dict[str, dict | None] | None = None,
) -> dict | None:
    """Return daily trend metrics for a single ticker."""
    normalized = symbol.upper()
    if metrics_cache is not None and normalized in metrics_cache:
        return metrics_cache[normalized]
    today = now_eastern().date()
    start = (today - timedelta(days=18)).isoformat()
    try:
        bars = alpaca.get_bars(normalized, timeframe="1Day", start=start, limit=15)
    except Exception:
        if metrics_cache is not None:
            metrics_cache[normalized] = None
        return None

    t = _compute_bar_trends(bars)
    if not t:
        if metrics_cache is not None:
            metrics_cache[normalized] = None
        return None
    if metrics_cache is not None:
        metrics_cache[normalized] = t
    return t


def _build_catalyst_reaction_context(
    alpaca: AlpacaClient,
    news_events: list,
    focus_symbols: list[str] | None = None,
    metrics_cache: dict[str, dict | None] | None = None,
    max_events: int = 5,
) -> str:
    lines = ["Catalyst Reaction Snapshot:"]
    seen: set[str] = set()
    focus_set = {symbol.upper() for symbol in (focus_symbols or [])}
    for event in news_events:
        symbol = next((sym.upper() for sym in event.symbols if sym), "")
        if not symbol or symbol in seen:
            continue
        if focus_set and symbol not in focus_set:
            continue
        metrics = _get_ticker_trend_metrics(
            alpaca,
            symbol,
            metrics_cache=metrics_cache,
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
            f" today({metrics['intraday_chg']:+.1f}%)"
            f" 5d({metrics['five_d_chg']:+.1f}%)"
            f" trend={metrics['trend']}"
            f" reaction={reaction}"
        )
        seen.add(symbol)
        if len(seen) >= max_events:
            break
    return "\n".join(lines) if len(lines) > 1 else ""


def _get_options_context(
    alpaca: AlpacaClient,
    watchlist: list[str],
    best_events: dict[str, object] | None = None,
    recent_trades: list[dict] | None = None,
    recent_closes: list[dict] | None = None,
    metrics_cache: dict[str, dict | None] | None = None,
) -> str:
    """Fetch options chains for watchlist symbols to give LLM context."""
    if not watchlist:
        return ""

    guidance = expression_guidance_lines(recent_closes or [], recent_trades or [])
    all_lines = []
    show_guidance = True
    for sym in watchlist:
        price = _get_price(alpaca, sym)
        if price <= 0:
            continue
        metrics = _get_ticker_trend_metrics(alpaca, sym, metrics_cache=metrics_cache)
        # Add per-ticker trend context
        trend = _get_ticker_trend(alpaca, sym, metrics_cache=metrics_cache)
        if trend:
            all_lines.append(f"\n{sym} ({trend}):")
        else:
            all_lines.append(f"\n{sym} (${price:.2f}):")
        all_lines.append(
            format_symbol_setup_context(
                sym,
                metrics,
                (best_events or {}).get(sym.upper()),
            )
        )
        chain = fetch_option_chain(alpaca, sym, price)
        if chain:
            all_lines.append(
                format_chain_for_llm(
                    chain,
                    max_contracts=20,
                    underlying_price=price,
                    expression_guidance=guidance if show_guidance else None,
                )
            )
            show_guidance = False

    return "\n".join(all_lines) if all_lines else ""


def _build_candidate_context(
    alpaca: AlpacaClient,
    news_events: list,
    position_symbols: list[str],
    thesis_symbols: list[str],
    direct_symbols: list[str],
    spillover_symbols: list[str],
    metrics_cache: dict[str, dict | None] | None = None,
) -> tuple[str, list[str]]:
    source_tags_by_symbol: dict[str, set[str]] = {}

    def _tag_symbols(symbols: list[str], tag: str) -> None:
        for symbol in symbols:
            normalized = symbol.upper()
            if not normalized:
                continue
            source_tags_by_symbol.setdefault(normalized, set()).add(tag)

    _tag_symbols(position_symbols, "position")
    _tag_symbols(thesis_symbols, "thesis")
    _tag_symbols(direct_symbols, "direct")
    spillover_only = [
        symbol for symbol in spillover_symbols
        if symbol.upper() not in {s.upper() for s in direct_symbols}
    ]
    _tag_symbols(spillover_only, "spillover")
    try:
        movers = alpaca.get_movers(top=10)
    except Exception:
        movers = []
    _tag_symbols([str(symbol).upper() for symbol in movers], "mover")
    _tag_symbols(["SPY", "QQQ", "IWM", "XLK", "XLF", "XLE", "XLV"], "macro")

    best_events = map_best_events_by_symbol(news_events)
    event_scores = {
        symbol: score_news_event(event)
        for symbol, event in best_events.items()
    }
    metrics_by_symbol: dict[str, dict] = {}
    for symbol in list(source_tags_by_symbol):
        metrics = _get_ticker_trend_metrics(
            alpaca,
            symbol,
            metrics_cache=metrics_cache,
        )
        if not metrics:
            continue
        event = best_events.get(symbol)
        reaction = ""
        if event is not None:
            reaction = classify_catalyst_reaction(
                event.age_minutes,
                metrics["intraday_chg"],
                metrics["five_d_chg"],
            )
        metrics_by_symbol[symbol] = {
            **metrics,
            "reaction": reaction,
        }

    candidates = build_candidate_ideas(
        source_tags_by_symbol,
        metrics_by_symbol,
        best_events,
        event_scores,
    )
    candidate_context = format_candidate_table(
        candidates,
        max_rows=config.CANDIDATE_TABLE_SIZE,
    )
    finalists = select_candidate_finalists(
        candidates,
        max_symbols=config.CANDIDATE_FINALISTS,
    )
    return candidate_context, finalists


def _get_price(alpaca: AlpacaClient, symbol: str) -> float:
    try:
        data = alpaca.get_snapshots([symbol])
        snap = data.get(symbol, {})
        trade = snap.get("latestTrade") or snap.get("latest_trade") or {}
        price = float(trade.get("p") or trade.get("price") or 0.0)
        if price > 0:
            return price
        bar = snap.get("dailyBar") or snap.get("daily_bar") or {}
        return float(bar.get("c") or bar.get("close") or 0.0)
    except Exception:
        return 0.0


def _build_watchlist(alpaca: AlpacaClient, news_symbols: list[str]) -> list[str]:
    """Build a watchlist from news mentions and market movers."""
    symbols: list[str] = []
    seen: set[str] = set()

    # Add symbols mentioned in news
    for sym in news_symbols:
        if sym not in seen and sym not in ("SPY", "QQQ", "IWM", "DIA"):
            symbols.append(sym)
            seen.add(sym)

    # Add top movers
    try:
        movers = alpaca.get_movers(top=config.WATCHLIST_SIZE)
        for sym in movers:
            if sym not in seen:
                symbols.append(sym)
                seen.add(sym)
    except Exception:
        pass

    return symbols[: config.WATCHLIST_SIZE]


def run_cycle(
    alpaca: AlpacaClient,
    brain: TradingBrain,
    logger: AITradeLogger,
    journal: ThesisJournal,
) -> int:
    """Run one complete trading cycle. Returns number of trades executed."""

    # 1. Get portfolio state
    portfolio = get_portfolio_state(alpaca)

    # 1b. Reconcile pending orders that may have filled asynchronously.
    status_updates, closes_backfilled = reconcile_pending_orders(alpaca, logger)
    if status_updates or closes_backfilled:
        log(
            f"reconciled pending orders: status_updates={status_updates} "
            f"close_backfills={closes_backfilled}"
        )
        if closes_backfilled > 0:
            portfolio = get_portfolio_state(alpaca)

    # 2. Enrich positions with underlying spot prices and run risk assessment
    if portfolio.option_positions:
        # Batch-fetch underlying spot prices
        underlyings = list({p.underlying for p in portfolio.option_positions})
        try:
            snap_data = alpaca.get_snapshots(underlyings)
        except Exception:
            snap_data = {}
        for pos in portfolio.option_positions:
            snap = snap_data.get(pos.underlying, {})
            trade = snap.get("latestTrade") or snap.get("latest_trade") or {}
            spot = float(trade.get("p") or trade.get("price") or 0.0)
            if spot <= 0:
                bar = snap.get("dailyBar") or snap.get("daily_bar") or {}
                spot = float(bar.get("c") or bar.get("close") or 0.0)
            pos.underlying_spot = spot

        # Run risk assessment on each position
        catastrophic_exits = False
        for pos in list(portfolio.option_positions):
            result = assess_position_risk(
                entry_premium=pos.avg_entry_price,
                current_premium=pos.current_price,
                dte=pos.dte,
            )
            if isinstance(result, PositionRiskState) and result.should_close:
                # Catastrophic stop — auto-close immediately
                log(f"catastrophic stop for {pos.symbol}: {result.reason}")
                from .brain import TradeDecision
                decision = TradeDecision(
                    action="close_position",
                    underlying=pos.underlying,
                    strike_preference="",
                    expiry_preference="",
                    conviction=1.0,
                    risk_pct=0.0,
                    reasoning=f"auto-exit: {result.reason}",
                    target_symbol=pos.symbol,
                )
                close_result = _execute_close(alpaca, decision, portfolio, logger, "catastrophic risk exit")
                if close_result.filled:
                    log(f"catastrophic exit executed: {pos.symbol}")
                    catastrophic_exits = True
                elif close_result.success:
                    log(f"catastrophic exit submitted (pending): {pos.symbol}")
            elif isinstance(result, PositionRiskAlert):
                # Soft alert — attach to position for LLM to see
                pos.risk_alert = result.message
                log(f"risk alert for {pos.symbol}: {result.message}")

        # Refresh portfolio after any catastrophic exits
        if catastrophic_exits:
            portfolio = get_portfolio_state(alpaca)
            # Re-enrich the refreshed positions
            try:
                snap_data = alpaca.get_snapshots(
                    list({p.underlying for p in portfolio.option_positions})
                )
            except Exception:
                snap_data = {}
            for pos in portfolio.option_positions:
                snap = snap_data.get(pos.underlying, {})
                trade = snap.get("latestTrade") or snap.get("latest_trade") or {}
                spot = float(trade.get("p") or trade.get("price") or 0.0)
                if spot <= 0:
                    bar = snap.get("dailyBar") or snap.get("daily_bar") or {}
                    spot = float(bar.get("c") or bar.get("close") or 0.0)
                pos.underlying_spot = spot

    # 3. Load trade history for context
    recent_trades = logger.get_recent_trades(limit=15)
    recent_closes = logger.get_recent_closes(limit=10)
    trade_history_context = format_trade_history(recent_trades, recent_closes)

    # 4. Get thesis journal context
    journal_context = journal.to_context_str()

    # 5. Fetch news — targeted for tickers the model cares about
    # Focus symbols = open positions + active theses
    position_symbols: list[str] = [p.underlying for p in portfolio.option_positions]
    thesis_symbols = [e.underlying for e in journal.active_entries()]
    focus_symbols = list(dict.fromkeys(position_symbols + thesis_symbols))

    cycle_time = now_eastern()
    metrics_cache: dict[str, dict | None] = {}
    news_items = fetch_targeted_news(
        alpaca, focus_symbols, lookback_hours=config.NEWS_LOOKBACK_HOURS,
    )
    news_events = build_news_events(news_items, reference_time=cycle_time)
    news_symbols = rank_symbols_from_events(
        news_events, focus_symbols=focus_symbols, max_symbols=config.WATCHLIST_SIZE,
    )
    expanded_news_symbols = expand_symbols_with_relationships(
        news_symbols,
        events=news_events,
        max_symbols=config.WATCHLIST_SIZE,
    )
    extra_news_symbols = [
        symbol
        for symbol in expanded_news_symbols
        if symbol not in set(news_symbols) and symbol not in set(focus_symbols)
    ][:6]
    if extra_news_symbols:
        related_news_items = fetch_news(
            alpaca,
            symbols=extra_news_symbols,
            lookback_hours=config.NEWS_LOOKBACK_HOURS,
        )
        news_items = merge_news_items(news_items, related_news_items, max_items=70)
        news_events = build_news_events(news_items, reference_time=cycle_time)
        news_symbols = rank_symbols_from_events(
            news_events,
            focus_symbols=focus_symbols,
            max_symbols=config.WATCHLIST_SIZE,
        )
        expanded_news_symbols = expand_symbols_with_relationships(
            news_symbols,
            events=news_events,
            max_symbols=config.WATCHLIST_SIZE,
        )
    candidate_context, finalists = _build_candidate_context(
        alpaca,
        news_events,
        position_symbols=position_symbols,
        thesis_symbols=thesis_symbols,
        direct_symbols=news_symbols,
        spillover_symbols=expanded_news_symbols,
        metrics_cache=metrics_cache,
    )
    deep_focus_symbols = list(dict.fromkeys(focus_symbols + finalists))
    news_context = format_news_for_llm(
        news_items,
        focus_symbols=deep_focus_symbols,
        reference_time=cycle_time,
    )
    catalyst_reaction_context = _build_catalyst_reaction_context(
        alpaca,
        news_events,
        focus_symbols=deep_focus_symbols,
        metrics_cache=metrics_cache,
    )
    if catalyst_reaction_context:
        news_context = f"{catalyst_reaction_context}\n\n{news_context}"

    # 6. Get market context
    market_context = _get_market_context(alpaca)

    # 7. Get deep-dive options context only for finalists
    finalist_watchlist = prioritized_symbol_watchlist(
        position_symbols,
        thesis_symbols,
        finalists,
        limit=min(
            config.WATCHLIST_SIZE,
            len(position_symbols) + len(thesis_symbols) + config.CANDIDATE_FINALISTS,
        ),
    )
    options_context = _get_options_context(
        alpaca,
        finalist_watchlist,
        best_events=map_best_events_by_symbol(news_events),
        recent_trades=recent_trades,
        recent_closes=recent_closes,
        metrics_cache=metrics_cache,
    )

    # 8. Send to LLM brain
    portfolio_context = portfolio.to_context_str()
    run_result = brain.run(
        portfolio_context=portfolio_context,
        candidate_context=candidate_context,
        news_context=news_context,
        market_context=market_context,
        options_context=options_context,
        journal_context=journal_context,
        trade_history_context=trade_history_context,
    )
    analysis = run_result.analysis

    log(f"LLM analysis: {analysis.analysis[:200]}")

    decisions_json = json.dumps(
        {
            "market_analysis": analysis.analysis,
            "thesis_updates": [
                {
                    "id": update.id,
                    "underlying": update.underlying,
                    "direction": update.direction,
                    "thesis": update.thesis,
                    "conviction": update.conviction,
                    "status": update.status,
                    "new_observation": update.new_observation,
                }
                for update in analysis.thesis_updates
            ],
            "trades": [
                {
                    "action": d.action,
                    "underlying": d.underlying,
                    "strike_preference": d.strike_preference,
                    "expiry_preference": d.expiry_preference,
                    "conviction": d.conviction,
                    "risk_pct": d.risk_pct,
                    "reasoning": d.reasoning,
                    "target_symbol": d.target_symbol,
                    "expression_profile": d.expression_profile,
                    "contract_symbol": d.contract_symbol,
                    "target_delta_range": d.target_delta_range,
                    "target_dte_range": d.target_dte_range,
                    "max_spread_pct": d.max_spread_pct,
                }
                for d in analysis.trades
            ],
        }
    )
    decision_id = logger.log_decision(
        AIDecisionRecord(
            timestamp=now_eastern(),
            market_analysis=analysis.analysis,
            news_summary=news_context[:1000],
            portfolio_state=portfolio_context[:1000],
            decisions_json=decisions_json,
            trades_executed=0,
            llm_provider=run_result.packet.provider,
            llm_model=run_result.packet.model,
            packet_json=json.dumps(run_result.packet.to_payload()),
            response_json=json.dumps(run_result.completion.to_payload()),
        )
    )

    # 9. Apply thesis journal updates
    if analysis.thesis_updates:
        journal.apply_updates(analysis.thesis_updates)

    # 10. Execute trades
    trades_executed = 0
    for decision in analysis.trades:
        log(
            f"trade decision: {decision.action} {decision.underlying} "
            f"conviction={decision.conviction:.2f} risk={decision.risk_pct:.1%} "
            f"reason: {decision.reasoning}"
        )
        result = execute_trade(
            alpaca, decision, portfolio, logger, analysis.analysis
        )
        if result.success:
            if result.filled:
                trades_executed += 1
                log(f"trade filled: {result.symbol} qty={result.qty} premium=${result.premium:.2f}")
                # Refresh portfolio for next trade in this cycle
                portfolio = get_portfolio_state(alpaca)
            else:
                log(f"trade accepted (pending fill): {result.symbol} - {result.message}")
        else:
            log(f"trade failed: {result.symbol} - {result.message}")

    # 11. Update the logged cycle with realized execution count.
    logger.update_decision_trade_count(decision_id, trades_executed)
    logger.log_portfolio_snapshot(_portfolio_snapshot_record(portfolio))

    return trades_executed


def run() -> None:
    """Main entry point - runs the autonomous trading loop."""

    env_path = Path(__file__).with_name(".env")
    if not env_path.exists():
        # Fall back to momentum_trader .env
        env_path = Path(__file__).parent.parent / "momentum_trader" / ".env"
    load_dotenv(env_path, override=True)

    # Validate required env vars
    llm_model = config.resolved_llm_model()
    llm_provider = infer_provider(
        model=llm_model,
        provider=os.environ.get("LLM_PROVIDER") or config.LLM_PROVIDER,
    )
    llm_api_key = resolve_api_key(llm_provider)
    if not llm_api_key:
        log(f"ERROR: {api_key_env_name(llm_provider)} not set. Add it to ai_trader/.env")
        return

    alpaca_key = os.environ.get("ALPACA_API_KEY")
    alpaca_secret = os.environ.get("ALPACA_API_SECRET")
    if not alpaca_key or not alpaca_secret:
        log("ERROR: ALPACA_API_KEY / ALPACA_API_SECRET not set")
        return

    alpaca = AlpacaClient.from_env()
    brain = TradingBrain(
        api_key=llm_api_key,
        provider=llm_provider,
        model=llm_model,
    )
    logger = AITradeLogger()

    # Initialize thesis journal (persisted to SQLite)
    journal_db = Path(__file__).parent / "logs" / "ai_trades.db"
    journal = ThesisJournal(
        db_path=journal_db,
        max_active=config.JOURNAL_MAX_ACTIVE,
        max_full_display=config.JOURNAL_MAX_FULL_DISPLAY,
        stale_cycles=config.JOURNAL_STALE_CYCLES,
        stale_conviction=config.JOURNAL_STALE_CONVICTION,
    )

    # Verify account access
    try:
        account = alpaca.get_account()
        equity = float(account.get("equity") or 0)
        log(f"connected to Alpaca | equity=${equity:,.2f}")
        if account.get("account_blocked"):
            log("ERROR: account is blocked")
            return
    except Exception as exc:
        log(f"ERROR: cannot connect to Alpaca: {exc}")
        return

    active_theses = len(journal.active_entries())
    log("=" * 60)
    log("AI AUTONOMOUS OPTIONS TRADER - STARTING")
    log(f"  Paper trading: {config.PAPER_TRADING}")
    log(f"  Max risk per trade: {config.MAX_RISK_PER_TRADE:.0%}")
    log(f"  Scan interval: {config.SCAN_INTERVAL_MINUTES} min")
    log(f"  LLM provider/model: {llm_provider}/{llm_model}")
    log(f"  Active theses: {active_theses}")
    log("=" * 60)

    cycle_count = 0
    total_trades = 0

    while True:
        now = now_eastern()

        if not is_market_open(now):
            # Wait for market to open
            market_open_today = now.replace(
                hour=9, minute=30, second=0, microsecond=0
            )
            if now < market_open_today:
                wait = (market_open_today - now).total_seconds()
                if wait > 0:
                    log(f"market closed. waiting {wait / 60:.0f} min until open")
                    time_module.sleep(min(wait, 300))  # Check every 5 min
                    continue
            # After 4 PM - done for the day
            log("market closed for today. shutting down.")
            break

        if not _within_trading_window(now):
            log("outside trading window, waiting...")
            time_module.sleep(60)
            continue

        cycle_count += 1
        log(f"--- cycle {cycle_count} ---")

        try:
            trades = run_cycle(alpaca, brain, logger, journal)
            total_trades += trades
            active = len(journal.active_entries())
            log(
                f"cycle {cycle_count} complete: {trades} trades | "
                f"total today: {total_trades} | active theses: {active}"
            )
        except Exception as exc:
            log(f"cycle error: {exc}")
            import traceback
            traceback.print_exc()

        # Sleep until next cycle
        sleep_seconds = config.SCAN_INTERVAL_MINUTES * 60
        log(f"sleeping {config.SCAN_INTERVAL_MINUTES} min until next cycle")
        time_module.sleep(sleep_seconds)


if __name__ == "__main__":
    run()
