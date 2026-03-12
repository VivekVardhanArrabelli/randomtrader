"""Tests for the AI trader options module."""

from datetime import date

from ai_trader.options import OptionContract, format_chain_for_llm, select_contract


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
    delta: float | None = None,
    implied_volatility: float | None = None,
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
        delta=delta,
        implied_volatility=implied_volatility,
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


def test_select_contract_prefers_quality_when_strikes_are_close():
    contracts = [
        _make_contract(
            strike=150.0,
            symbol="C150_BAD",
            bid=4.30,
            ask=5.00,
            volume=10,
            open_interest=50,
            dte=9,
        ),
        _make_contract(
            strike=151.0,
            symbol="C151_GOOD",
            bid=4.90,
            ask=5.00,
            volume=400,
            open_interest=2000,
            dte=9,
        ),
    ]
    result = select_contract(
        contracts,
        underlying_price=150.0,
        strike_preference="atm",
        expiry_preference="next_week",
    )
    assert result is not None
    assert result.symbol == "C151_GOOD"


def test_select_contract_monthly_prefers_more_appropriate_dte():
    contracts = [
        _make_contract(symbol="C150_14D", strike=150.0, dte=14),
        _make_contract(symbol="C150_30D", strike=150.0, dte=30),
    ]
    result = select_contract(
        contracts,
        underlying_price=150.0,
        strike_preference="atm",
        expiry_preference="monthly",
    )
    assert result is not None
    assert result.symbol == "C150_30D"


def test_select_contract_exact_symbol_overrides_bucket_heuristics():
    contracts = [
        _make_contract(symbol="C150", strike=150.0, dte=9),
        _make_contract(symbol="C155", strike=155.0, dte=18),
    ]
    result = select_contract(
        contracts,
        underlying_price=150.0,
        strike_preference="atm",
        expiry_preference="next_week",
        contract_symbol="C155",
    )
    assert result is not None
    assert result.symbol == "C155"


def test_select_contract_respects_delta_and_dte_ranges():
    contracts = [
        _make_contract(symbol="C150_FAST", strike=150.0, dte=5),
        _make_contract(symbol="C150_TARGET", strike=150.0, dte=14),
        _make_contract(symbol="C145_DEEP", strike=145.0, dte=20),
    ]
    result = select_contract(
        contracts,
        underlying_price=150.0,
        strike_preference="atm",
        expiry_preference="next_week",
        target_delta_range=(0.45, 0.60),
        target_dte_range=(10, 18),
    )
    assert result is not None
    assert result.symbol == "C150_TARGET"


def test_select_contract_respects_expression_profile_stock_proxy():
    contracts = [
        _make_contract(symbol="C150_BAL", strike=150.0, dte=10, delta=0.52),
        _make_contract(symbol="C145_STOCK", strike=145.0, dte=28, delta=0.76),
        _make_contract(symbol="C155_CONVEX", strike=155.0, dte=7, delta=0.28),
    ]
    result = select_contract(
        contracts,
        underlying_price=150.0,
        expression_profile="stock_proxy",
    )
    assert result is not None
    assert result.symbol == "C145_STOCK"


def test_select_contract_empty():
    result = select_contract([], underlying_price=150.0)
    assert result is None


def test_spread_pct():
    c = _make_contract(bid=4.50, ask=5.00)
    assert abs(c.spread_pct - 0.10) < 0.01  # 10% spread


def test_context_str():
    c = _make_contract(delta=0.56, implied_volatility=0.32)
    s = c.to_context_str(underlying_price=150.0)
    assert "AAPL" in s
    assert "CALL" in s
    assert "150.00" in s
    assert "delta=0.56" in s
    assert "iv=32.0%" in s
    assert "premium=" in s
    assert "be_move=" in s
    assert "spread=" in s


def test_format_chain_for_llm_includes_shortlist():
    contracts = [
        _make_contract(symbol="CALL0", option_type="call", strike=147.0, dte=28, delta=0.76),
        _make_contract(symbol="CALL1", option_type="call", strike=150.0),
        _make_contract(symbol="CALL2", option_type="call", strike=151.0, dte=12),
        _make_contract(symbol="CALL3", option_type="call", strike=150.0, dte=25),
        _make_contract(symbol="CALL4", option_type="call", strike=154.0, dte=7),
        _make_contract(symbol="PUT0", option_type="put", strike=153.0, dte=28, delta=-0.74),
        _make_contract(symbol="PUT1", option_type="put", strike=150.0),
        _make_contract(symbol="PUT2", option_type="put", strike=149.0, dte=12),
        _make_contract(symbol="PUT3", option_type="put", strike=150.0, dte=25),
        _make_contract(symbol="PUT4", option_type="put", strike=146.0, dte=7),
    ]
    text = format_chain_for_llm(
        contracts,
        max_contracts=8,
        underlying_price=150.0,
        expression_guidance=["Recent expression outcomes: time_cushion 2/3 wins net=$+4,500"],
    )
    assert "Recent expression outcomes: time_cushion 2/3 wins net=$+4,500" in text
    assert "Suggested contract shortlist" in text
    assert "More time = slower decay / thesis room" in text
    assert "Calls:" in text
    assert "Puts:" in text
    assert "Primary:" in text
    assert "More time:" in text
    assert "Stock proxy:" in text
    assert "Convex upside:" in text
    assert "CALL1" in text
    assert "CALL3" in text
    assert "PUT1" in text
    assert "PUT3" in text
