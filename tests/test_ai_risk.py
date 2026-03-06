"""Tests for the AI trader risk management module."""

from ai_trader.risk import (
    evaluate_position_risk,
    evaluate_stock_trade_risk,
    evaluate_trade_risk,
    size_for_risk_budget,
)


def test_trade_risk_approved():
    result = evaluate_trade_risk(
        equity=100_000,
        cash=50_000,
        current_exposure=10_000,
        open_positions=1,
        option_ask=5.00,
        day_pl=0.0,
    )
    assert result.approved
    assert result.max_contracts > 0
    # Max premium should not exceed 40% of equity
    assert result.max_premium <= 100_000 * 0.40 + 0.01


def test_trade_risk_daily_loss_limit():
    result = evaluate_trade_risk(
        equity=100_000,
        cash=50_000,
        current_exposure=0,
        open_positions=0,
        option_ask=5.00,
        day_pl=-15_000,  # -15% loss
    )
    assert not result.approved
    assert "daily loss" in result.reason


def test_trade_risk_max_positions():
    result = evaluate_trade_risk(
        equity=100_000,
        cash=50_000,
        current_exposure=0,
        open_positions=5,  # at max
        option_ask=5.00,
        day_pl=0.0,
    )
    assert not result.approved
    assert "max open positions" in result.reason


def test_trade_risk_max_exposure():
    result = evaluate_trade_risk(
        equity=100_000,
        cash=50_000,
        current_exposure=85_000,  # over 80% exposure
        open_positions=1,
        option_ask=5.00,
        day_pl=0.0,
    )
    assert not result.approved
    assert "exposure" in result.reason


def test_trade_risk_expensive_option():
    result = evaluate_trade_risk(
        equity=10_000,
        cash=5_000,
        current_exposure=0,
        open_positions=0,
        option_ask=50.00,  # $5000/contract, over 40% of $10k
        day_pl=0.0,
    )
    assert not result.approved
    assert "expensive" in result.reason


def test_position_risk_profit_target():
    result = evaluate_position_risk(
        entry_premium=5.00,
        current_premium=8.00,  # +60%, above 50% target
        dte=30,
    )
    assert result.should_close
    assert "profit" in result.reason


def test_position_risk_stop_loss():
    result = evaluate_position_risk(
        entry_premium=5.00,
        current_premium=2.50,  # -50%, below 40% stop
        dte=5,
    )
    assert result.should_close
    assert "stop" in result.reason


def test_position_risk_time_stop():
    result = evaluate_position_risk(
        entry_premium=5.00,
        current_premium=5.50,
        dte=1,  # 1 DTE, below 2 DTE time stop
    )
    assert result.should_close
    assert "time" in result.reason


def test_position_risk_hold():
    result = evaluate_position_risk(
        entry_premium=5.00,
        current_premium=5.50,  # +10%, within normal range
        dte=20,
    )
    assert not result.should_close
    assert result.reason == "hold"


def test_stock_trade_risk_approved():
    result = evaluate_stock_trade_risk(
        equity=100_000,
        cash=50_000,
        current_exposure=10_000,
        open_positions=1,
        share_price=200.0,
        day_pl=0.0,
    )
    assert result.approved
    assert result.max_shares > 0
    assert result.max_notional <= 100_000 * 0.40 + 0.01


def test_stock_trade_risk_expensive_share():
    result = evaluate_stock_trade_risk(
        equity=10_000,
        cash=1_000,
        current_exposure=0,
        open_positions=0,
        share_price=1_500.0,
        day_pl=0.0,
    )
    assert not result.approved
    assert "expensive" in result.reason


def test_size_for_risk_budget_returns_zero_when_unit_is_too_expensive():
    assert size_for_risk_budget(400.0, 500.0) == 0


def test_size_for_risk_budget_floors_to_whole_units():
    assert size_for_risk_budget(1_250.0, 500.0) == 2
