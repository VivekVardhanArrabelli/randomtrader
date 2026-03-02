"""Performance report for the AI trader.

Run:  python -m ai_trader.report [--db path] [--json]
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

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


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------

def _safe_div(a: float, b: float) -> float | None:
    return a / b if b != 0 else None


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
    m.total_trades = len(trades)
    m.filled = sum(1 for t in trades if t["status"] == "filled")
    m.rejected = sum(1 for t in trades if t["status"] in ("rejected", "risk_rejected", "error"))
    m.decision_cycles = len(decisions)
    m.cycles_with_trades = sum(1 for d in decisions if d["trades_executed"] > 0)

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
        if t["status"] == "filled":
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


def print_report(m: TradeMetrics) -> None:
    print("=" * 60)
    print("  AI TRADER PERFORMANCE REPORT")
    print("=" * 60)

    print("\n--- Activity ---")
    print(f"  Decision cycles:       {m.decision_cycles}")
    print(f"  Cycles with trades:    {m.cycles_with_trades}")
    print(f"  Total orders:          {m.total_trades}")
    print(f"  Filled:                {m.filled}")
    print(f"  Rejected / errors:     {m.rejected}")

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

    print("\n--- Direction ---")
    print(f"  Calls traded:          {m.calls_traded}  (win rate: {_fmt(m.call_win_rate, '%')})")
    print(f"  Puts traded:           {m.puts_traded}  (win rate: {_fmt(m.put_win_rate, '%')})")

    print("\n--- LLM Quality ---")
    print(f"  Avg conviction:        {_fmt(m.avg_conviction, '', 2)}")
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
