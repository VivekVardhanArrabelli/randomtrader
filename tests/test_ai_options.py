"""Tests for the AI trader options module."""

from datetime import date

from ai_trader.options import OptionContract, select_contract


def _make_contract(
    symbol: str = "AAPL250321C00150000",
    underlying: str = "AAPL",
    option_type: str = "call",
    strike: float = 150.0,
    expiration: date = date(2025, 3, 21),
    bid: float = 4.50,
    ask: float = 5.00,
    volume: int = 100,
    open_interest: int = 500,
    dte: int = 20,
) -> OptionContract:
    return OptionContract(
        symbol=symbol,
        underlying=underlying,
        option_type=option_type,
        strike=strike,
        expiration=expiration,
        bid=bid,
        ask=ask,
        mid=(bid + ask) / 2,
        volume=volume,
        open_interest=open_interest,
        dte=dte,
    )


def test_select_contract_atm():
    contracts = [
        _make_contract(strike=145.0, symbol="C145"),
        _make_contract(strike=150.0, symbol="C150"),
        _make_contract(strike=155.0, symbol="C155"),
    ]
    result = select_contract(contracts, underlying_price=150.0, strike_preference="atm")
    assert result is not None
    assert result.strike == 150.0


def test_select_contract_otm_call():
    contracts = [
        _make_contract(strike=145.0, symbol="C145"),
        _make_contract(strike=150.0, symbol="C150"),
        _make_contract(strike=155.0, symbol="C155"),
    ]
    result = select_contract(contracts, underlying_price=150.0, strike_preference="otm")
    assert result is not None
    assert result.strike >= 150.0


def test_select_contract_empty():
    result = select_contract([], underlying_price=150.0)
    assert result is None


def test_spread_pct():
    c = _make_contract(bid=4.50, ask=5.00)
    assert abs(c.spread_pct - 0.10) < 0.01  # 10% spread


def test_context_str():
    c = _make_contract()
    s = c.to_context_str()
    assert "AAPL" in s
    assert "CALL" in s
    assert "150.00" in s
