"""Tests for the AI trader report module."""

import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path

from ai_trader.db import AITradeLogger, AITradeRecord, PositionCloseRecord, AIDecisionRecord
from ai_trader.report import compute_metrics


def _seed_db(db_path: Path) -> None:
    logger = AITradeLogger(db_path)
    now = datetime(2025, 3, 1, 10, 0, 0)

    # Log some trades
    for i, (sym, pnl_val) in enumerate([
        ("AAPL", 500.0), ("TSLA", -200.0), ("MSFT", 300.0),
        ("GOOG", -100.0), ("NVDA", 800.0),
    ]):
        logger.log_trade(AITradeRecord(
            timestamp=now,
            symbol=f"{sym}250321C00150000",
            underlying=sym,
            option_type="call" if i % 2 == 0 else "put",
            strike=150.0,
            expiration="2025-03-21",
            action="buy_call" if i % 2 == 0 else "buy_put",
            qty=2,
            premium=5.0,
            total_cost=1000.0,
            conviction=0.7 + i * 0.05,
            reasoning=f"test trade {i}",
            market_analysis="bullish",
            order_id=f"order-{i}",
            status="filled",
        ))
        logger.log_position_close(PositionCloseRecord(
            timestamp=now,
            symbol=f"{sym}250321C00150000",
            underlying=sym,
            qty=2,
            entry_premium=5.0,
            exit_premium=5.0 + pnl_val / 200,
            pnl=pnl_val,
            reason="profit_target" if pnl_val > 0 else "stop_loss",
        ))

    logger.log_decision(AIDecisionRecord(
        timestamp=now,
        market_analysis="test",
        news_summary="test",
        portfolio_state="test",
        decisions_json="[]",
        trades_executed=3,
    ))


def test_compute_metrics_basic():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        _seed_db(db_path)
        m = compute_metrics(db_path)

        assert m.total_trades == 5
        assert m.filled == 5
        assert m.wins == 3
        assert m.losses == 2
        assert m.net_pnl == 1300.0  # 500-200+300-100+800
        assert m.win_rate is not None
        assert abs(m.win_rate - 0.60) < 0.01
        assert m.largest_win == 800.0
        assert m.largest_loss == -200.0
        assert m.decision_cycles == 1
        assert m.max_drawdown >= 0


def test_compute_metrics_empty_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "empty.db"
        AITradeLogger(db_path)  # just create schema
        m = compute_metrics(db_path)
        assert m.total_trades == 0
        assert m.net_pnl == 0.0
        assert m.win_rate is None


def test_profit_factor():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "pf.db"
        _seed_db(db_path)
        m = compute_metrics(db_path)
        # gross_profit = 1600, gross_loss = 300
        assert m.profit_factor is not None
        assert m.profit_factor > 1.0
