"""SQLite logging for AI trader decisions and trades."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

DEFAULT_DB_PATH = Path(__file__).parent / "logs" / "ai_trades.db"
_PENDING_ORDER_STATUSES = (
    "new",
    "accepted",
    "pending_new",
    "accepted_for_bidding",
    "partially_filled",
    "pending_replace",
    "replaced",
    "calculated",
    "held",
)


@dataclass(frozen=True)
class AITradeRecord:
    timestamp: datetime
    symbol: str              # OCC option symbol
    underlying: str
    option_type: str
    strike: float
    expiration: str
    action: str              # buy / sell / close
    qty: int
    premium: float           # per-share premium
    total_cost: float
    conviction: float
    reasoning: str
    market_analysis: str
    order_id: str | None
    status: str              # submitted / filled / rejected / error


@dataclass(frozen=True)
class AIDecisionRecord:
    timestamp: datetime
    market_analysis: str
    news_summary: str
    portfolio_state: str
    decisions_json: str      # full JSON of LLM decisions
    trades_executed: int


@dataclass(frozen=True)
class PositionCloseRecord:
    timestamp: datetime
    symbol: str
    underlying: str
    qty: int
    entry_premium: float
    exit_premium: float
    pnl: float
    reason: str
    order_id: str | None = None


class AITradeLogger:
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
                CREATE TABLE IF NOT EXISTS ai_trades (
                    timestamp TEXT,
                    symbol TEXT,
                    underlying TEXT,
                    option_type TEXT,
                    strike REAL,
                    expiration TEXT,
                    action TEXT,
                    qty INTEGER,
                    premium REAL,
                    total_cost REAL,
                    conviction REAL,
                    reasoning TEXT,
                    market_analysis TEXT,
                    order_id TEXT,
                    status TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ai_decisions (
                    timestamp TEXT,
                    market_analysis TEXT,
                    news_summary TEXT,
                    portfolio_state TEXT,
                    decisions_json TEXT,
                    trades_executed INTEGER
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS position_closes (
                    timestamp TEXT,
                    symbol TEXT,
                    underlying TEXT,
                    qty INTEGER,
                    entry_premium REAL,
                    exit_premium REAL,
                    pnl REAL,
                    reason TEXT,
                    order_id TEXT
                )
                """
            )
            close_cols = {
                str(row[1])
                for row in conn.execute("PRAGMA table_info(position_closes)")
            }
            if "order_id" not in close_cols:
                conn.execute("ALTER TABLE position_closes ADD COLUMN order_id TEXT")

    def log_trade(self, record: AITradeRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO ai_trades VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record.timestamp.isoformat(),
                    record.symbol,
                    record.underlying,
                    record.option_type,
                    record.strike,
                    record.expiration,
                    record.action,
                    record.qty,
                    record.premium,
                    record.total_cost,
                    record.conviction,
                    record.reasoning,
                    record.market_analysis,
                    record.order_id,
                    record.status,
                ),
            )

    def log_decision(self, record: AIDecisionRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO ai_decisions VALUES (?, ?, ?, ?, ?, ?)",
                (
                    record.timestamp.isoformat(),
                    record.market_analysis,
                    record.news_summary,
                    record.portfolio_state,
                    record.decisions_json,
                    record.trades_executed,
                ),
            )

    def log_position_close(self, record: PositionCloseRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO position_closes (
                    timestamp, symbol, underlying, qty, entry_premium,
                    exit_premium, pnl, reason, order_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.timestamp.isoformat(),
                    record.symbol,
                    record.underlying,
                    record.qty,
                    record.entry_premium,
                    record.exit_premium,
                    record.pnl,
                    record.reason,
                    record.order_id,
                ),
            )

    def get_pending_trades(self, limit: int = 100) -> list[dict]:
        placeholders = ",".join("?" for _ in _PENDING_ORDER_STATUSES)
        query = (
            "SELECT * FROM ai_trades "
            "WHERE order_id IS NOT NULL AND order_id != '' "
            f"AND lower(status) IN ({placeholders}) "
            "ORDER BY timestamp DESC LIMIT ?"
        )
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                query,
                (*_PENDING_ORDER_STATUSES, limit),
            ).fetchall()
            return [dict(row) for row in rows]

    def update_trade_status(self, order_id: str, status: str) -> int:
        if not order_id:
            return 0
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE ai_trades SET status = ? WHERE order_id = ?",
                (status, order_id),
            )
            return int(cur.rowcount or 0)

    def has_position_close_for_order(self, order_id: str) -> bool:
        if not order_id:
            return False
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM position_closes WHERE order_id = ? LIMIT 1",
                (order_id,),
            ).fetchone()
            return row is not None

    def get_trade_count_today(self) -> int:
        today_str = datetime.now().strftime("%Y-%m-%d")
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM ai_trades WHERE timestamp LIKE ? AND status = 'filled'",
                (f"{today_str}%",),
            ).fetchone()
            return row[0] if row else 0

    def get_recent_trades(self, limit: int = 20) -> list[dict]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM ai_trades ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_todays_pnl(self) -> float:
        today_str = datetime.now().strftime("%Y-%m-%d")
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(pnl), 0) FROM position_closes WHERE timestamp LIKE ?",
                (f"{today_str}%",),
            ).fetchone()
            return float(row[0]) if row else 0.0

    def get_recent_closes(self, limit: int = 10) -> list[dict]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM position_closes ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(row) for row in rows]


def _conviction_calibration(trades: list[dict], closes: list[dict]) -> list[str]:
    """Group closes by conviction bucket and show win rate per bucket."""
    # Build a map from underlying -> conviction from the trades table
    # (trades table has conviction; closes table does not)
    underlying_conviction: dict[str, float] = {}
    for t in trades:
        action = str(t.get("action") or "")
        if t.get("status") == "filled" and t.get("conviction") and action.startswith("buy_"):
            sym = t.get("underlying", "")
            underlying_conviction[sym] = float(t["conviction"])

    # Bucket: conviction -> list of pnl
    buckets: dict[str, list[float]] = {}
    for c in closes:
        sym = c.get("underlying", "")
        conv = underlying_conviction.get(sym)
        if conv is None:
            continue
        if conv < 0.7:
            bucket = "0.60-0.69"
        elif conv < 0.8:
            bucket = "0.70-0.79"
        elif conv < 0.9:
            bucket = "0.80-0.89"
        else:
            bucket = "0.90-1.00"
        buckets.setdefault(bucket, []).append(c.get("pnl", 0))

    if not buckets:
        return []

    lines = ["  Conviction calibration:"]
    for bucket in sorted(buckets.keys()):
        pnls = buckets[bucket]
        w = sum(1 for p in pnls if p > 0)
        total = len(pnls)
        wr = w / total * 100 if total > 0 else 0
        lines.append(f"    {bucket}: {w}/{total} wins ({wr:.0f}%)")
    return lines


def format_trade_history(trades: list[dict], closes: list[dict]) -> str:
    """Format recent trades + closes as context for the LLM."""
    if not trades and not closes:
        return "No prior trades yet. This is your first session."

    lines = []

    if closes:
        lines.append("Recent closed positions (newest first):")
        for c in closes[:10]:
            pnl = c.get("pnl", 0)
            result = "WIN" if pnl > 0 else "LOSS" if pnl < 0 else "FLAT"
            lines.append(
                f"  {c.get('timestamp', '')[:16]} {c.get('underlying', '')} "
                f"| entry=${c.get('entry_premium', 0):.2f} exit=${c.get('exit_premium', 0):.2f} "
                f"| pnl=${pnl:+,.2f} ({result}) | reason: {c.get('reason', '')}"
            )
            if c.get("context"):
                lines.append(f"    {c['context']}")

        # Summary stats
        total = len(closes)
        wins = sum(1 for c in closes if (c.get("pnl") or 0) > 0)
        losses = sum(1 for c in closes if (c.get("pnl") or 0) < 0)
        net = sum(c.get("pnl", 0) for c in closes)
        lines.append(f"\n  Recent stats: {wins}W/{losses}L net=${net:+,.2f}")

        # Performance by type (calls vs puts)
        call_w, call_l, call_net = 0, 0, 0.0
        put_w, put_l, put_net = 0, 0, 0.0
        for c in closes:
            sym = c.get("symbol", "")
            pnl = c.get("pnl") or 0
            is_call = "C" in sym[6:] if len(sym) > 6 else False
            if is_call:
                call_net += pnl
                if pnl > 0:
                    call_w += 1
                elif pnl < 0:
                    call_l += 1
            else:
                put_net += pnl
                if pnl > 0:
                    put_w += 1
                elif pnl < 0:
                    put_l += 1
        lines.append(
            f"  By type: Calls {call_w}W/{call_l}L net=${call_net:+,.0f}"
            f" | Puts {put_w}W/{put_l}L net=${put_net:+,.0f}"
        )

        # Average winner vs loser
        win_pnls = [c.get("pnl", 0) for c in closes if (c.get("pnl") or 0) > 0]
        loss_pnls = [c.get("pnl", 0) for c in closes if (c.get("pnl") or 0) < 0]
        avg_win = sum(win_pnls) / len(win_pnls) if win_pnls else 0
        avg_loss = sum(loss_pnls) / len(loss_pnls) if loss_pnls else 0
        ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        lines.append(
            f"  Avg winner: ${avg_win:+,.0f} | Avg loser: ${avg_loss:+,.0f}"
            f" | Win/loss ratio: {ratio:.2f}"
        )

        # Best and worst trade
        if closes:
            best = max(closes, key=lambda c: c.get("pnl") or 0)
            worst = min(closes, key=lambda c: c.get("pnl") or 0)
            lines.append(
                f"  Best trade: {best.get('underlying', '?')} ${best.get('pnl', 0):+,.0f}"
                f" | Worst trade: {worst.get('underlying', '?')} ${worst.get('pnl', 0):+,.0f}"
            )

        # Conviction calibration
        cal_lines = _conviction_calibration(trades, closes)
        if cal_lines:
            lines.extend(cal_lines)

        # Repeat losers: tickers with 2+ losses and 0 wins in recent closes
        ticker_stats: dict[str, dict] = {}
        for c in closes:
            sym = c.get("underlying", "???")
            s = ticker_stats.setdefault(sym, {"wins": 0, "losses": 0, "pnl": 0.0})
            pnl = c.get("pnl") or 0
            if pnl > 0:
                s["wins"] += 1
            elif pnl < 0:
                s["losses"] += 1
            s["pnl"] += pnl
        repeat_losers = [
            (sym, s) for sym, s in ticker_stats.items()
            if s["losses"] >= 2 and s["wins"] == 0
        ]
        if repeat_losers:
            repeat_losers.sort(key=lambda x: x[1]["losses"], reverse=True)
            lines.append("WARNING — REPEAT LOSERS (you keep losing on these tickers):")
            for sym, s in repeat_losers:
                n = s["losses"]
                lines.append(
                    f"  {sym}: {n} trades, 0 wins, ${s['pnl']:+,.0f} total lost"
                )

        # Equity momentum (last 5 closed trades — newest first)
        recent_5 = closes[:5]
        if recent_5:
            momentum = sum(c.get("pnl", 0) for c in recent_5)
            m_wins = sum(1 for c in recent_5 if (c.get("pnl") or 0) > 0)
            m_losses = sum(1 for c in recent_5 if (c.get("pnl") or 0) < 0)
            lines.append(
                f"Equity momentum (last {len(recent_5)} trades): ${momentum:+,.0f} "
                f"({m_wins}W/{m_losses}L)"
            )

    if trades:
        # Show recent unfilled/rejected for learning
        rejected = [t for t in trades if t.get("status") in ("rejected", "risk_rejected", "error")]
        if rejected:
            lines.append(f"\nRecent rejected orders ({len(rejected)}):")
            for t in rejected[:5]:
                lines.append(
                    f"  {t.get('underlying', '')} {t.get('action', '')} "
                    f"| status={t.get('status', '')} | {t.get('reasoning', '')[:80]}"
                )

    return "\n".join(lines) if lines else "No trade history available."
