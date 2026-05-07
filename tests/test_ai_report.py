"""Tests for the AI trader report module."""

import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path

from ai_trader.db import (
    AIDecisionRecord,
    AITradeLogger,
    AITradeRecord,
    PortfolioSnapshotRecord,
    PositionCloseRecord,
)
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


def test_compute_metrics_counts_legacy_decision_json_activity():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "legacy.db"
        logger = AITradeLogger(db_path)
        logger.log_decision(AIDecisionRecord(
            timestamp=datetime(2026, 5, 1, 14, 24, 0),
            market_analysis="legacy pending order",
            news_summary="",
            portfolio_state="",
            decisions_json='[{"action":"buy_call","underlying":"AAPL"}]',
            trades_executed=0,
        ))

        m = compute_metrics(db_path)

        assert m.decision_cycles == 1
        assert m.cycles_with_trades == 1


def test_compute_metrics_counts_nested_trade_activity():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "nested.db"
        logger = AITradeLogger(db_path)
        logger.log_decision(AIDecisionRecord(
            timestamp=datetime(2026, 5, 1, 14, 24, 0),
            market_analysis="newer decision format",
            news_summary="",
            portfolio_state="",
            decisions_json='{"trades":[{"action":"buy_put","underlying":"SPY","risk_pct":0.06}]}',
            trades_executed=0,
        ))

        m = compute_metrics(db_path)

        assert m.cycles_with_trades == 1
        assert m.avg_risk_pct == 0.06


def test_compute_metrics_uses_entry_conviction_for_closed_pnl():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "entry_conviction.db"
        logger = AITradeLogger(db_path)
        now = datetime(2026, 5, 1, 10, 0, 0)
        symbol = "AAPL260515C00150000"

        logger.log_trade(AITradeRecord(
            timestamp=now,
            symbol=symbol,
            underlying="AAPL",
            option_type="call",
            strike=150.0,
            expiration="2026-05-15",
            action="buy_call",
            qty=1,
            premium=5.0,
            total_cost=500.0,
            conviction=0.65,
            reasoning="entry decision",
            market_analysis="test",
            order_id="entry-order",
            status="filled",
        ))
        logger.log_trade(AITradeRecord(
            timestamp=now,
            symbol=symbol,
            underlying="AAPL",
            option_type="call",
            strike=150.0,
            expiration="2026-05-15",
            action="close_position",
            qty=1,
            premium=5.0,
            total_cost=0.0,
            conviction=0.95,
            reasoning="exit confidence should not calibrate entry quality",
            market_analysis="test",
            order_id="close-order",
            status="filled",
        ))
        logger.log_position_close(PositionCloseRecord(
            timestamp=now,
            symbol=symbol,
            underlying="AAPL",
            qty=1,
            entry_premium=5.0,
            exit_premium=3.0,
            pnl=-200.0,
            reason="stop_loss",
        ))

        m = compute_metrics(db_path)

        assert m.avg_conviction == 0.65
        assert m.conviction_vs_outcome == [(0.65, -200.0)]
        assert m.calls_traded == 1


def test_compute_metrics_loads_latest_portfolio_snapshot():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "snapshot.db"
        logger = AITradeLogger(db_path)
        logger.log_portfolio_snapshot(PortfolioSnapshotRecord(
            timestamp=datetime(2026, 5, 6, 15, 30, 0),
            equity=95_000.0,
            cash=88_000.0,
            buying_power=176_000.0,
            day_pl=125.0,
            total_options_exposure=0.0,
            total_equity_exposure=4_500.0,
            total_exposure=4_500.0,
            open_option_count=0,
            open_equity_count=1,
            positions_json=(
                '[{"asset_type":"stock","symbol":"NVDA","qty":22,'
                '"market_value":4500.0,"unrealized_pl":125.0,"pnl_pct":0.028}]'
            ),
        ))

        m = compute_metrics(db_path)

        snapshot = m.latest_portfolio_snapshot
        assert snapshot is not None
        assert snapshot["equity"] == 95_000.0
        assert snapshot["total_exposure"] == 4_500.0
        assert snapshot["open_equity_count"] == 1
        assert snapshot["positions"][0]["symbol"] == "NVDA"


def test_compute_metrics_reports_active_option_loss_guardrail():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "option_guard.db"
        logger = AITradeLogger(db_path)
        now = datetime(2026, 5, 6, 10, 0, 0)
        for symbol in (
            "LLY260508C00950000",
            "TT260515C00490000",
            "W260515C00065000",
        ):
            logger.log_position_close(PositionCloseRecord(
                timestamp=now,
                symbol=symbol,
                underlying=symbol[:3].rstrip("0123456789"),
                qty=1,
                entry_premium=5.0,
                exit_premium=3.0,
                pnl=-200.0,
                reason="option loss",
            ))

        m = compute_metrics(db_path)

        assert m.option_loss_streak == 3
        assert m.option_guard_active
        assert m.option_guard_min_conviction == 0.80


def test_compute_metrics_counts_llm_failure_cycles():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "llm_failures.db"
        logger = AITradeLogger(db_path)
        now = datetime(2026, 5, 1, 10, 0, 0)
        for analysis in (
            "DeepSeek response parse error: Unterminated string",
            "DeepSeek error: read timed out",
            "Broad market constructive; no trades.",
        ):
            logger.log_decision(AIDecisionRecord(
                timestamp=now,
                market_analysis=analysis,
                news_summary="",
                portfolio_state="",
                decisions_json="[]",
                trades_executed=0,
            ))

        m = compute_metrics(db_path)

        assert m.llm_failure_cycles == 2
        assert m.llm_parse_failures == 1
        assert m.llm_api_failures == 1
