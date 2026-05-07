from datetime import datetime

from ai_trader.brain import TradeDecision
from ai_trader.db import AITradeLogger, PositionCloseRecord
from ai_trader.executor import (
    _execute_close_option,
    _execute_open_option,
    _execute_open_stock,
    _option_loss_streak_guard_reason,
)
from ai_trader.portfolio import EquityPosition, OptionPosition, PortfolioState
from ai_trader.utils import AccountSnapshot


class _FakeAlpaca:
    def __init__(self) -> None:
        self.submitted_orders = []

    def get_orders(self, status="open", limit=200):
        return []

    def get_option_latest_quotes(self, symbols):
        return {
            "quotes": {
                symbols[0]: {
                    "bp": 24.9,
                    "ap": 25.1,
                }
            }
        }

    def get_snapshots(self, symbols):
        return {
            symbols[0]: {
                "latestQuote": {
                    "bp": 199.0,
                    "ap": 201.0,
                },
                "latestTrade": {
                    "p": 200.0,
                },
            }
        }

    def submit_order(self, **kwargs):
        self.submitted_orders.append(kwargs)
        return {"id": "order-1", "status": "filled"}

    def get_order(self, order_id):
        return {
            "id": order_id,
            "status": "filled",
            "filled_avg_price": "25.00",
            "filled_qty": "3",
        }


def _portfolio_with_lly_call() -> PortfolioState:
    return PortfolioState(
        account=AccountSnapshot(
            equity=100_000.0,
            cash=75_000.0,
            buying_power=150_000.0,
            day_pl=0.0,
            positions_value=7_500.0,
        ),
        option_positions=[
            OptionPosition(
                symbol="LLY260508C00950000",
                underlying="LLY",
                option_type="call",
                strike=950.0,
                expiration="2026-05-08",
                qty=3,
                avg_entry_price=30.0,
                current_price=25.0,
                market_value=7_500.0,
                unrealized_pl=-1_500.0,
                cost_basis=9_000.0,
            )
        ],
        equity_positions=[],
    )


def _portfolio_with_nvda_stock() -> PortfolioState:
    return PortfolioState(
        account=AccountSnapshot(
            equity=100_000.0,
            cash=75_000.0,
            buying_power=150_000.0,
            day_pl=0.0,
            positions_value=2_400.0,
        ),
        option_positions=[],
        equity_positions=[
            EquityPosition(
                symbol="NVDA",
                qty=12,
                avg_entry_price=200.0,
                current_price=200.0,
                market_value=2_400.0,
                unrealized_pl=0.0,
                cost_basis=2_400.0,
            )
        ],
    )


def _log_recent_option_losses(logger: AITradeLogger) -> None:
    for symbol, pnl in (
        ("LLY260508C00950000", -1500.0),
        ("TT260515C00490000", -2590.0),
        ("W260515C00065000", -2109.0),
    ):
        logger.log_position_close(PositionCloseRecord(
            timestamp=datetime(2026, 5, 6, 10, 0, 0),
            symbol=symbol,
            underlying=symbol[:3].rstrip("0123456789"),
            qty=1,
            entry_premium=5.0,
            exit_premium=3.0,
            pnl=pnl,
            reason="closed option loss",
        ))


def test_close_option_accepts_underlying_in_target_symbol(tmp_path) -> None:
    alpaca = _FakeAlpaca()
    logger = AITradeLogger(tmp_path / "trades.db")
    decision = TradeDecision(
        action="close_position",
        underlying="LLY",
        strike_preference="",
        expiry_preference="",
        conviction=0.9,
        risk_pct=0.0,
        reasoning="Close LLY call before theta decay accelerates.",
        target_symbol="LLY",
    )

    result = _execute_close_option(
        alpaca=alpaca,
        decision=decision,
        portfolio=_portfolio_with_lly_call(),
        logger=logger,
        market_analysis="test close",
    )

    assert result.success
    assert result.filled
    assert result.symbol == "LLY260508C00950000"
    assert alpaca.submitted_orders[0]["symbol"] == "LLY260508C00950000"


def test_close_option_accepts_polygon_prefixed_target_symbol(tmp_path) -> None:
    alpaca = _FakeAlpaca()
    logger = AITradeLogger(tmp_path / "trades.db")
    decision = TradeDecision(
        action="close_position",
        underlying="LLY",
        strike_preference="",
        expiry_preference="",
        conviction=0.9,
        risk_pct=0.0,
        reasoning="Close exact contract.",
        target_symbol="O:LLY260508C00950000",
    )

    result = _execute_close_option(
        alpaca=alpaca,
        decision=decision,
        portfolio=_portfolio_with_lly_call(),
        logger=logger,
        market_analysis="test close",
    )

    assert result.success
    assert result.symbol == "LLY260508C00950000"


def test_open_stock_add_respects_existing_symbol_risk_budget(tmp_path) -> None:
    alpaca = _FakeAlpaca()
    logger = AITradeLogger(tmp_path / "trades.db")
    decision = TradeDecision(
        action="buy_stock",
        underlying="NVDA",
        strike_preference="",
        expiry_preference="",
        conviction=0.65,
        risk_pct=0.05,
        reasoning="Add to existing NVDA thesis after another cycle.",
        target_symbol=None,
    )

    result = _execute_open_stock(
        alpaca=alpaca,
        decision=decision,
        portfolio=_portfolio_with_nvda_stock(),
        logger=logger,
        market_analysis="test add-on sizing",
    )

    assert not result.success
    assert "existing NVDA exposure" in result.message
    assert alpaca.submitted_orders == []
    rows = logger.get_recent_trades(limit=1)
    assert rows[0]["status"] == "risk_rejected"
    assert rows[0]["qty"] == 0


def test_open_option_rejects_marginal_trade_after_option_loss_streak(tmp_path) -> None:
    alpaca = _FakeAlpaca()
    logger = AITradeLogger(tmp_path / "trades.db")
    _log_recent_option_losses(logger)
    decision = TradeDecision(
        action="buy_call",
        underlying="AAPL",
        strike_preference="atm",
        expiry_preference="next_week",
        conviction=0.75,
        risk_pct=0.10,
        reasoning="Marginal option entry after three option losses.",
        target_symbol=None,
    )

    result = _execute_open_option(
        alpaca=alpaca,
        decision=decision,
        portfolio=_portfolio_with_lly_call(),
        logger=logger,
        market_analysis="test option loss guard",
    )

    assert not result.success
    assert "closed option trades were losses" in result.message
    assert alpaca.submitted_orders == []
    row = logger.get_recent_trades(limit=1)[0]
    assert row["status"] == "risk_rejected"
    assert row["action"] == "buy_call"
    assert row["symbol"] == "AAPL"


def test_option_loss_streak_guard_allows_high_conviction_option(tmp_path) -> None:
    logger = AITradeLogger(tmp_path / "trades.db")
    _log_recent_option_losses(logger)
    decision = TradeDecision(
        action="buy_call",
        underlying="AAPL",
        strike_preference="atm",
        expiry_preference="next_week",
        conviction=0.85,
        risk_pct=0.10,
        reasoning="High-conviction setup should still be allowed.",
        target_symbol=None,
    )

    assert _option_loss_streak_guard_reason(logger, decision) == ""
