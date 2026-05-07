"""Performance report for the AI trader.

Run:  python -m ai_trader.report [--db path] [--json]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from . import config
from .db import DEFAULT_DB_PATH


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class TradeMetrics:
    total_trades: int = 0
    filled: int = 0
    rejected: int = 0
    wins: int = 0
    losses: int = 0
    breakeven: int = 0
    win_rate: float | None = None
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    net_pnl: float = 0.0
    profit_factor: float | None = None
    expectancy: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    max_drawdown: float = 0.0
    avg_conviction: float = 0.0
    avg_hold_hours: float | None = None
    sharpe_ratio: float | None = None
    calmar_ratio: float | None = None
    best_day: tuple[str, float] | None = None
    worst_day: tuple[str, float] | None = None
    calls_traded: int = 0
    puts_traded: int = 0
    call_win_rate: float | None = None
    put_win_rate: float | None = None
    top_underlying: list[tuple[str, float]] | None = None
    avg_risk_pct: float = 0.0
    decision_cycles: int = 0
    cycles_with_trades: int = 0
    conviction_vs_outcome: list[tuple[float, float]] | None = None
    latest_portfolio_snapshot: dict | None = None
    option_loss_streak: int = 0
    option_guard_active: bool = False
    option_guard_min_conviction: float = 0.0
    llm_failure_cycles: int = 0
    llm_parse_failures: int = 0
    llm_api_failures: int = 0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_trades(db_path: Path) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM ai_trades ORDER BY timestamp ASC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _load_closes(db_path: Path) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM position_closes ORDER BY timestamp ASC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _load_decisions(db_path: Path) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM ai_decisions ORDER BY timestamp ASC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _load_latest_portfolio_snapshot(db_path: Path) -> dict | None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'portfolio_snapshots'"
    ).fetchone()
    if exists is None:
        conn.close()
        return None

    row = conn.execute(
        "SELECT * FROM portfolio_snapshots ORDER BY timestamp DESC LIMIT 1"
    ).fetchone()
    conn.close()
    if row is None:
        return None

    snapshot = dict(row)
    raw_positions = snapshot.pop("positions_json", "") or "[]"
    try:
        positions = json.loads(raw_positions)
    except (TypeError, ValueError):
        positions = []
    snapshot["positions"] = positions if isinstance(positions, list) else []
    return snapshot


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------

def _safe_div(a: float, b: float) -> float | None:
    return a / b if b != 0 else None


def _decision_rows_from_json(raw: object) -> list[dict]:
    if not raw:
        return []
    try:
        parsed = json.loads(str(raw))
    except (TypeError, ValueError):
        return []
    if isinstance(parsed, list):
        return [row for row in parsed if isinstance(row, dict)]
    if isinstance(parsed, dict):
        trades = parsed.get("trades")
        if isinstance(trades, list):
            return [row for row in trades if isinstance(row, dict)]
    return []


def _decision_has_trade_activity(decision: dict) -> bool:
    """Return True when a decision cycle produced at least one trade decision."""
    if int(decision.get("trades_executed") or 0) > 0:
        return True
    return len(_decision_rows_from_json(decision.get("decisions_json"))) > 0


def _is_entry_action(action: object) -> bool:
    return str(action or "") in {"buy_call", "buy_put", "buy_stock"}


def _entry_risk_pcts_from_decisions(decisions: list[dict]) -> list[float]:
    """Return requested risk_pct values for opening trade decisions."""
    risk_pcts: list[float] = []
    for decision in decisions:
        for row in _decision_rows_from_json(decision.get("decisions_json")):
            if not _is_entry_action(row.get("action")):
                continue
            try:
                risk_pct = float(row.get("risk_pct"))
            except (TypeError, ValueError):
                continue
            if risk_pct >= 0:
                risk_pcts.append(risk_pct)
    return risk_pcts


def _is_option_symbol(symbol: object) -> bool:
    return re.match(
        r"^(?:O:)?[A-Z]+[0-9]{6}[CP][0-9]{8}$",
        str(symbol or "").upper(),
    ) is not None


def _current_option_loss_streak(closes: list[dict]) -> int:
    streak = 0
    for close in reversed(closes):
        option_type = str(close.get("option_type") or "").strip().lower()
        if option_type not in {"call", "put"} and not _is_option_symbol(close.get("symbol")):
            continue
        if float(close.get("pnl") or 0.0) < 0:
            streak += 1
            continue
        break
    return streak


def _llm_failure_counts(decisions: list[dict]) -> tuple[int, int, int]:
    failures = parse_failures = api_failures = 0
    for decision in decisions:
        analysis = str(decision.get("market_analysis") or "").lower()
        is_parse = (
            "parse error" in analysis
            or "not valid json" in analysis
            or "malformed llm" in analysis
            or "did not include content or tool calls" in analysis
        )
        is_api = (
            "deepseek error" in analysis
            or "llm error" in analysis
            or "unsupported llm provider" in analysis
        )
        if is_parse:
            parse_failures += 1
        if is_api:
            api_failures += 1
        if is_parse or is_api:
            failures += 1
    return failures, parse_failures, api_failures


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    s = sorted(values)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2


def compute_metrics(db_path: Path) -> TradeMetrics:
    trades = _load_trades(db_path)
    closes = _load_closes(db_path)
    decisions = _load_decisions(db_path)

    m = TradeMetrics()
    m.latest_portfolio_snapshot = _load_latest_portfolio_snapshot(db_path)
    m.total_trades = len(trades)
    m.filled = sum(1 for t in trades if t["status"] == "filled")
    m.rejected = sum(1 for t in trades if t["status"] in ("rejected", "risk_rejected", "error"))
    m.decision_cycles = len(decisions)
    m.cycles_with_trades = sum(1 for d in decisions if _decision_has_trade_activity(d))
    entry_risks = _entry_risk_pcts_from_decisions(decisions)
    m.avg_risk_pct = sum(entry_risks) / len(entry_risks) if entry_risks else 0.0
    m.option_loss_streak = _current_option_loss_streak(closes)
    m.option_guard_min_conviction = config.OPTION_LOSS_STREAK_GUARD_MIN_CONVICTION
    m.option_guard_active = (
        m.option_loss_streak >= config.OPTION_LOSS_STREAK_GUARD_LOOKBACK
        and config.OPTION_LOSS_STREAK_GUARD_LOOKBACK > 0
    )
    m.llm_failure_cycles, m.llm_parse_failures, m.llm_api_failures = _llm_failure_counts(decisions)

    if not closes:
        return m

    # Win / loss / breakeven
    pnls: list[float] = []
    daily_pnl: dict[str, float] = defaultdict(float)
    convictions: list[float] = []
    conviction_outcomes: list[tuple[float, float]] = []
    call_wins = call_total = put_wins = put_total = 0
    underlying_pnl: dict[str, float] = defaultdict(float)

    for c in closes:
        pnl = c["pnl"]
        pnls.append(pnl)
        day = c["timestamp"][:10]
        daily_pnl[day] += pnl
        underlying_pnl[c["underlying"]] += pnl

        if pnl > 0:
            m.wins += 1
        elif pnl < 0:
            m.losses += 1
        else:
            m.breakeven += 1

    # Match closes back to trade convictions
    trade_by_symbol: dict[str, dict] = {}
    for t in trades:
        if t["status"] == "filled" and _is_entry_action(t.get("action")):
            trade_by_symbol[t["symbol"]] = t

    for c in closes:
        trade = trade_by_symbol.get(c["symbol"])
        if trade:
            conv = trade.get("conviction", 0)
            convictions.append(conv)
            conviction_outcomes.append((conv, c["pnl"]))
            opt_type = trade.get("option_type", "")
            if "call" in opt_type:
                call_total += 1
                if c["pnl"] > 0:
                    call_wins += 1
            elif "put" in opt_type:
                put_total += 1
                if c["pnl"] > 0:
                    put_wins += 1

    total_closed = m.wins + m.losses + m.breakeven
    m.win_rate = _safe_div(m.wins, m.wins + m.losses)
    m.gross_profit = sum(p for p in pnls if p > 0)
    m.gross_loss = sum(p for p in pnls if p < 0)
    m.net_pnl = sum(pnls)
    m.profit_factor = _safe_div(m.gross_profit, abs(m.gross_loss))
    m.expectancy = m.net_pnl / total_closed if total_closed else 0.0
    m.avg_win = m.gross_profit / m.wins if m.wins else 0.0
    m.avg_loss = abs(m.gross_loss / m.losses) if m.losses else 0.0
    m.largest_win = max(pnls) if pnls else 0.0
    m.largest_loss = min(pnls) if pnls else 0.0
    m.avg_conviction = sum(convictions) / len(convictions) if convictions else 0.0

    # Max drawdown
    equity_curve = 0.0
    peak = 0.0
    max_dd = 0.0
    for pnl in pnls:
        equity_curve += pnl
        if equity_curve > peak:
            peak = equity_curve
        dd = peak - equity_curve
        if dd > max_dd:
            max_dd = dd
    m.max_drawdown = max_dd

    # Sharpe ratio (daily)
    if len(daily_pnl) >= 2:
        daily_returns = list(daily_pnl.values())
        mean_r = sum(daily_returns) / len(daily_returns)
        var_r = sum((r - mean_r) ** 2 for r in daily_returns) / (len(daily_returns) - 1)
        std_r = math.sqrt(var_r) if var_r > 0 else 0
        if std_r > 0:
            m.sharpe_ratio = round((mean_r / std_r) * math.sqrt(252), 2)

    # Calmar ratio
    if max_dd > 0 and daily_pnl:
        total_days = len(daily_pnl)
        annualized_return = (m.net_pnl / total_days) * 252 if total_days else 0
        m.calmar_ratio = round(annualized_return / max_dd, 2) if max_dd > 0 else None

    # Best / worst day
    if daily_pnl:
        m.best_day = max(daily_pnl.items(), key=lambda x: x[1])
        m.worst_day = min(daily_pnl.items(), key=lambda x: x[1])

    # Call vs put
    m.calls_traded = call_total
    m.puts_traded = put_total
    m.call_win_rate = _safe_div(call_wins, call_total)
    m.put_win_rate = _safe_div(put_wins, put_total)

    # Top underlyings by P&L
    sorted_ul = sorted(underlying_pnl.items(), key=lambda x: x[1], reverse=True)
    m.top_underlying = sorted_ul[:10]

    # Conviction vs outcome
    m.conviction_vs_outcome = conviction_outcomes

    return m


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def _fmt(val: float | None, prefix: str = "$", decimals: int = 2) -> str:
    if val is None:
        return "n/a"
    if prefix == "%":
        return f"{val * 100:.{decimals}f}%"
    return f"{prefix}{val:,.{decimals}f}"


def _print_open_exposure(snapshot: dict | None) -> None:
    print("\n--- Current Open Exposure ---")
    if not snapshot:
        print("  No portfolio snapshot logged yet.")
        return

    equity = float(snapshot.get("equity") or 0.0)
    total_exposure = float(snapshot.get("total_exposure") or 0.0)
    exposure_pct = _safe_div(total_exposure, equity)
    print(f"  Snapshot:             {snapshot.get('timestamp', 'unknown')}")
    print(f"  Equity:               {_fmt(equity)}")
    print(f"  Cash:                 {_fmt(float(snapshot.get('cash') or 0.0))}")
    print(
        f"  Total exposure:       {_fmt(total_exposure)} "
        f"({_fmt(exposure_pct, '%')})"
    )
    print(f"  Options exposure:     {_fmt(float(snapshot.get('total_options_exposure') or 0.0))}")
    print(f"  Equity exposure:      {_fmt(float(snapshot.get('total_equity_exposure') or 0.0))}")

    positions = snapshot.get("positions") or []
    if not positions:
        print("  Open positions:       none")
        return

    print("  Open positions:")
    ranked = sorted(
        [p for p in positions if isinstance(p, dict)],
        key=lambda p: abs(float(p.get("market_value") or 0.0)),
        reverse=True,
    )
    for pos in ranked[:8]:
        symbol = str(pos.get("symbol") or "?")
        asset_type = str(pos.get("asset_type") or "?")
        qty = int(float(pos.get("qty") or 0))
        value = float(pos.get("market_value") or 0.0)
        upl = float(pos.get("unrealized_pl") or 0.0)
        pnl_pct = float(pos.get("pnl_pct") or 0.0)
        detail = ""
        if asset_type == "option":
            detail = (
                f" {str(pos.get('option_type') or '').upper()} "
                f"${float(pos.get('strike') or 0.0):.2f} "
                f"exp={pos.get('expiration', '')}"
            )
        print(
            f"    {symbol:18s} {asset_type:6s} qty={qty:<5d}"
            f" value={_fmt(value):>12s} UPL={_fmt(upl):>12s} "
            f"({pnl_pct:.1%}){detail}"
        )


def print_report(m: TradeMetrics) -> None:
    print("=" * 60)
    print("  AI TRADER PERFORMANCE REPORT")
    print("=" * 60)

    print("\n--- Activity ---")
    print(f"  Decision cycles:       {m.decision_cycles}")
    print(f"  Cycles with orders:    {m.cycles_with_trades}")
    print(f"  Total orders:          {m.total_trades}")
    print(f"  Filled:                {m.filled}")
    print(f"  Rejected / errors:     {m.rejected}")

    _print_open_exposure(m.latest_portfolio_snapshot)

    if m.wins + m.losses + m.breakeven == 0:
        print("\n  No closed positions yet.")
        return

    print("\n--- P&L ---")
    print(f"  Net P&L:               {_fmt(m.net_pnl)}")
    print(f"  Gross profit:          {_fmt(m.gross_profit)}")
    print(f"  Gross loss:            {_fmt(m.gross_loss)}")
    print(f"  Largest win:           {_fmt(m.largest_win)}")
    print(f"  Largest loss:          {_fmt(m.largest_loss)}")

    print("\n--- Win / Loss ---")
    print(f"  Wins:                  {m.wins}")
    print(f"  Losses:                {m.losses}")
    print(f"  Breakeven:             {m.breakeven}")
    print(f"  Win rate:              {_fmt(m.win_rate, '%')}")
    print(f"  Avg win:               {_fmt(m.avg_win)}")
    print(f"  Avg loss:              {_fmt(m.avg_loss)}")
    print(f"  Expectancy:            {_fmt(m.expectancy)}")
    print(f"  Profit factor:         {_fmt(m.profit_factor, '', 2)}")

    print("\n--- Risk ---")
    print(f"  Max drawdown:          {_fmt(m.max_drawdown)}")
    print(f"  Sharpe ratio:          {_fmt(m.sharpe_ratio, '', 2)}")
    print(f"  Calmar ratio:          {_fmt(m.calmar_ratio, '', 2)}")

    print("\n--- Strategy Guardrails ---")
    print(f"  Option loss streak:    {m.option_loss_streak}")
    if m.option_guard_active:
        print(
            "  Marginal options:      blocked below "
            f"{m.option_guard_min_conviction:.2f} conviction"
        )
    else:
        print("  Marginal options:      normal conviction threshold")

    print("\n--- Direction ---")
    print(f"  Calls traded:          {m.calls_traded}  (win rate: {_fmt(m.call_win_rate, '%')})")
    print(f"  Puts traded:           {m.puts_traded}  (win rate: {_fmt(m.put_win_rate, '%')})")

    print("\n--- LLM Quality ---")
    print(f"  Avg conviction:        {_fmt(m.avg_conviction, '', 2)}")
    print(f"  Avg requested risk:    {_fmt(m.avg_risk_pct, '%')}")
    print(f"  Failure cycles:        {m.llm_failure_cycles}")
    print(f"  Parse failures:        {m.llm_parse_failures}")
    print(f"  API/provider failures: {m.llm_api_failures}")
    if m.conviction_vs_outcome:
        high_conv = [p for c, p in m.conviction_vs_outcome if c >= 0.80]
        low_conv = [p for c, p in m.conviction_vs_outcome if c < 0.80]
        high_wr = _safe_div(sum(1 for p in high_conv if p > 0), len(high_conv))
        low_wr = _safe_div(sum(1 for p in low_conv if p > 0), len(low_conv))
        print(f"  High conviction (>=80%) win rate:  {_fmt(high_wr, '%')}  (n={len(high_conv)})")
        print(f"  Low conviction (<80%) win rate:     {_fmt(low_wr, '%')}  (n={len(low_conv)})")

    if m.best_day:
        print(f"\n  Best day:  {m.best_day[0]}  {_fmt(m.best_day[1])}")
    if m.worst_day:
        print(f"  Worst day: {m.worst_day[0]}  {_fmt(m.worst_day[1])}")

    if m.top_underlying:
        print("\n--- Top Underlyings by P&L ---")
        for sym, pnl in m.top_underlying[:5]:
            print(f"  {sym:8s} {_fmt(pnl)}")

    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run() -> None:
    parser = argparse.ArgumentParser(description="AI trader performance report.")
    parser.add_argument(
        "--db", type=str, default=str(DEFAULT_DB_PATH),
        help=f"Path to SQLite DB (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON.")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"No database found at {db_path}")
        print("Run the AI trader first to generate trade data.")
        return

    metrics = compute_metrics(db_path)
    if args.json:
        out = {}
        for k, v in vars(metrics).items():
            if isinstance(v, list) and v and isinstance(v[0], tuple):
                out[k] = [{"key": a, "value": b} for a, b in v]
            elif isinstance(v, tuple):
                out[k] = {"key": v[0], "value": v[1]}
            else:
                out[k] = v
        print(json.dumps(out, indent=2))
    else:
        print_report(metrics)


if __name__ == "__main__":
    run()
