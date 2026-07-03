"""Tests for portfolio state parsing and the executor-facing risk contract.

The executor's risk rails read `total_exposure` and `open_position_count`
from PortfolioState and close stock positions via `EquityPosition`. These
tests pin that contract so a refactor of portfolio.py cannot silently break
the live loop again (see the EquityPosition ImportError regression).
"""

import importlib

from ai_trader.portfolio import (
    EquityPosition,
    OptionPosition,
    PortfolioState,
    get_portfolio_state,
)
from ai_trader.utils import AccountSnapshot


class FakeAlpaca:
    def __init__(self, account: dict, positions: list[dict]):
        self._account = account
        self._positions = positions

    def get_account(self) -> dict:
        return self._account

    def get_positions(self) -> list[dict]:
        return self._positions


def _account_snapshot(equity: float = 100_000.0) -> AccountSnapshot:
    return AccountSnapshot(
        equity=equity,
        cash=equity / 2,
        buying_power=equity,
        day_pl=0.0,
        positions_value=0.0,
    )


def _option_position(market_value: float = 1_000.0) -> OptionPosition:
    return OptionPosition(
        symbol="AAPL260117C00200000",
        underlying="AAPL",
        option_type="call",
        strike=200.0,
        expiration="2026-01-17",
        qty=2,
        avg_entry_price=5.0,
        current_price=5.0,
        market_value=market_value,
        unrealized_pl=0.0,
        cost_basis=1_000.0,
    )


def _equity_position(market_value: float = 2_000.0, qty: int = 10) -> EquityPosition:
    return EquityPosition(
        symbol="MSFT",
        qty=qty,
        avg_entry_price=200.0,
        current_price=market_value / qty if qty else 0.0,
        market_value=market_value,
        unrealized_pl=0.0,
        cost_basis=2_000.0,
    )


def test_live_trading_modules_import():
    # The paper trader once shipped broken because nothing imported these
    # modules in CI. Keep them importable.
    for module in (
        "ai_trader.portfolio",
        "ai_trader.executor",
        "ai_trader.loop",
        "ai_trader.paper_runner",
    ):
        importlib.import_module(module)


def test_total_exposure_counts_options_and_stock():
    state = PortfolioState(
        account=_account_snapshot(),
        option_positions=[_option_position(market_value=1_500.0)],
        equity_positions=[_equity_position(market_value=2_500.0)],
    )
    assert state.total_options_exposure == 1_500.0
    assert state.total_equity_exposure == 2_500.0
    assert state.total_exposure == 4_000.0
    assert state.open_option_count == 1
    assert state.open_position_count == 2


def test_short_equity_counts_as_positive_exposure():
    state = PortfolioState(
        account=_account_snapshot(),
        option_positions=[],
        equity_positions=[_equity_position(market_value=-3_000.0, qty=-10)],
    )
    assert state.total_exposure == 3_000.0
    assert state.open_position_count == 1


def test_context_includes_stock_positions():
    state = PortfolioState(
        account=_account_snapshot(),
        option_positions=[_option_position()],
        equity_positions=[_equity_position()],
    )
    context = state.to_context_str()
    assert "MSFT STOCK LONG" in context
    assert "1 options, 1 stock" in context


def test_get_portfolio_state_parses_mixed_positions():
    alpaca = FakeAlpaca(
        account={
            "equity": "100000",
            "cash": "60000",
            "buying_power": "120000",
            "last_equity": "99000",
            "long_market_value": "40000",
        },
        positions=[
            {
                "symbol": "AAPL260117C00200000",
                "asset_class": "options",
                "underlying_symbol": "AAPL",
                "strike_price": "200",
                "expiration_date": "2026-01-17",
                "qty": "2",
                "avg_entry_price": "5.10",
                "current_price": "5.50",
                "market_value": "1100",
                "unrealized_pl": "80",
                "cost_basis": "1020",
            },
            {
                "symbol": "MSFT",
                "asset_class": "us_equity",
                "qty": "10",
                "avg_entry_price": "400.00",
                "current_price": "410.00",
                "market_value": "4100",
                "unrealized_pl": "100",
                "cost_basis": "4000",
            },
        ],
    )

    state = get_portfolio_state(alpaca)

    assert state.account.day_pl == 1_000.0
    assert len(state.option_positions) == 1
    assert len(state.equity_positions) == 1

    equity = state.equity_positions[0]
    assert isinstance(equity, EquityPosition)
    assert equity.symbol == "MSFT"
    assert equity.qty == 10
    assert equity.market_value == 4_100.0

    assert state.total_exposure == 1_100.0 + 4_100.0
    assert state.open_position_count == 2


def test_get_portfolio_state_handles_missing_account_fields():
    alpaca = FakeAlpaca(
        account={"equity": "50000", "last_equity": None},
        positions=[],
    )
    state = get_portfolio_state(alpaca)
    assert state.account.day_pl == 50_000.0
    assert state.total_exposure == 0.0
