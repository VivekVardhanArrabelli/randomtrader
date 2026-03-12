"""Tests for AI trader decision/trade history formatting."""

from ai_trader.db import format_trade_history


def test_format_trade_history_includes_profile_calibration_lines():
    trades = [
        {
            "symbol": "O:VST260123C00170000",
            "underlying": "VST",
            "action": "buy_call",
            "status": "filled",
            "expression_profile": "convex",
        },
        {
            "symbol": "O:NOC260116C00620000",
            "underlying": "NOC",
            "action": "buy_call",
            "status": "filled",
            "expression_profile": "convex",
        },
        {
            "symbol": "O:XLP260227C00086000",
            "underlying": "XLP",
            "action": "buy_call",
            "status": "filled",
            "expression_profile": "time_cushion",
        },
    ]
    closes = [
        {
            "timestamp": "2026-02-11T09:35:00-05:00",
            "underlying": "VST",
            "symbol": "O:VST260123C00170000",
            "entry_date": "2026-01-12",
            "entry_premium": 7.67,
            "exit_premium": 4.00,
            "pnl": -3675.0,
            "reason": "stop_loss",
        },
        {
            "timestamp": "2026-02-12T09:35:00-05:00",
            "underlying": "NOC",
            "symbol": "O:NOC260116C00620000",
            "entry_date": "2026-01-08",
            "entry_premium": 18.39,
            "exit_premium": 3.40,
            "pnl": -4498.5,
            "reason": "stop_loss",
        },
        {
            "timestamp": "2026-02-13T09:35:00-05:00",
            "underlying": "XLP",
            "symbol": "O:XLP260227C00086000",
            "entry_date": "2026-02-11",
            "entry_premium": 2.09,
            "exit_premium": 3.90,
            "pnl": 3982.0,
            "reason": "backtest_end",
        },
    ]

    result = format_trade_history(trades, closes)

    assert "Expression profiles: convex 0/2 wins net=$-8,174 | time_cushion 1/1 wins net=$+3,982" in result
    assert "Expression review: Calls 1/3 wins net=$-4,192" in result
    assert "Short-dated calls (<=14 DTE): 0/2 wins net=$-8,174" in result
    assert "Stop-loss cluster: 2 trades net=$-8,174 | calls=2 puts=0" in result
