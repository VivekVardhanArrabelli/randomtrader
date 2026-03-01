"""Tests for the AI trader backtest module."""

from datetime import date, timedelta

from ai_trader.backtest import (
    BacktestConfig,
    BacktestResult,
    PolygonCache,
    SimPosition,
    SimTrade,
    _build_enriched_portfolio_context,
    _build_performance_summary,
    _extract_top_news_tickers,
    _option_bar_price,
    _select_real_contract,
    _trading_days,
    print_backtest_result,
)


# ---------------------------------------------------------------------------
# _trading_days
# ---------------------------------------------------------------------------

def test_trading_days():
    # Mon Jan 6 to Fri Jan 10 2025 = 5 weekdays
    days = _trading_days(date(2025, 1, 6), date(2025, 1, 10))
    assert len(days) == 5
    # Sat-Sun excluded
    days2 = _trading_days(date(2025, 1, 4), date(2025, 1, 5))
    assert len(days2) == 0


def test_trading_days_single():
    days = _trading_days(date(2025, 1, 6), date(2025, 1, 6))
    assert len(days) == 1


# ---------------------------------------------------------------------------
# PolygonCache
# ---------------------------------------------------------------------------

def test_polygon_cache_init():
    cache = PolygonCache()
    assert cache.contracts == {}
    assert cache.option_bars == {}


def test_polygon_cache_contracts():
    cache = PolygonCache()
    key = ("AAPL", "call", "2025-01-10", "2025-01-17", 140.0, 160.0)
    contracts = [{"ticker": "O:AAPL250117C00150000", "strike_price": 150}]
    cache.contracts[key] = contracts
    assert cache.contracts[key] == contracts


def test_polygon_cache_option_bars():
    cache = PolygonCache()
    ticker = "O:AAPL250117C00150000"
    bar = {"o": 5.0, "h": 5.5, "l": 4.8, "c": 5.2, "v": 100, "vw": 5.1}
    cache.option_bars[ticker] = {"2025-01-10": bar}
    assert cache.option_bars[ticker]["2025-01-10"] == bar


# ---------------------------------------------------------------------------
# _option_bar_price
# ---------------------------------------------------------------------------

def test_option_bar_price_vwap():
    bar = {"o": 5.0, "h": 5.5, "l": 4.8, "c": 5.2, "v": 100, "vw": 5.1}
    assert _option_bar_price(bar) == 5.1  # prefers VWAP


def test_option_bar_price_close_fallback():
    bar = {"o": 5.0, "h": 5.5, "l": 4.8, "c": 5.2, "v": 100}
    assert _option_bar_price(bar) == 5.2  # no VWAP, uses close


def test_option_bar_price_zero_vwap():
    bar = {"o": 5.0, "c": 5.2, "vw": 0}
    assert _option_bar_price(bar) == 5.2  # zero VWAP, falls back to close


def test_option_bar_price_empty():
    bar = {}
    assert _option_bar_price(bar) == 0.0


# ---------------------------------------------------------------------------
# _select_real_contract (with mocked cache)
# ---------------------------------------------------------------------------

def test_select_real_contract_atm(monkeypatch):
    """ATM strike preference should pick the contract closest to spot."""
    contracts = [
        {"ticker": "O:AAPL250117C00145000", "strike_price": 145, "expiration_date": "2025-01-17"},
        {"ticker": "O:AAPL250117C00150000", "strike_price": 150, "expiration_date": "2025-01-17"},
        {"ticker": "O:AAPL250117C00155000", "strike_price": 155, "expiration_date": "2025-01-17"},
    ]

    # Mock fetch_polygon_option_contracts to return our contracts
    import ai_trader.backtest as bt_mod
    monkeypatch.setattr(
        bt_mod, "fetch_polygon_option_contracts",
        lambda *args, **kwargs: contracts,
    )

    result = _select_real_contract(
        api_key="fake",
        underlying="AAPL",
        option_type="call",
        spot=150.0,
        trade_date=date(2025, 1, 10),
        strike_preference="atm",
        expiry_preference="next_week",
        default_dte=14,
    )
    assert result is not None
    assert result["strike_price"] == 150


def test_select_real_contract_otm_call(monkeypatch):
    """OTM call should pick strike above spot (3% OTM)."""
    contracts = [
        {"ticker": "O:AAPL250117C00150000", "strike_price": 150, "expiration_date": "2025-01-17"},
        {"ticker": "O:AAPL250117C00155000", "strike_price": 155, "expiration_date": "2025-01-17"},
        {"ticker": "O:AAPL250117C00160000", "strike_price": 160, "expiration_date": "2025-01-17"},
    ]
    import ai_trader.backtest as bt_mod
    monkeypatch.setattr(
        bt_mod, "fetch_polygon_option_contracts",
        lambda *args, **kwargs: contracts,
    )

    result = _select_real_contract(
        api_key="fake",
        underlying="AAPL",
        option_type="call",
        spot=150.0,
        trade_date=date(2025, 1, 10),
        strike_preference="otm",
        expiry_preference="next_week",
        default_dte=14,
    )
    assert result is not None
    # 150 * 1.03 = 154.5, closest is 155
    assert result["strike_price"] == 155


def test_select_real_contract_otm_put(monkeypatch):
    """OTM put should pick strike below spot (3% OTM)."""
    contracts = [
        {"ticker": "O:AAPL250117P00140000", "strike_price": 140, "expiration_date": "2025-01-17"},
        {"ticker": "O:AAPL250117P00145000", "strike_price": 145, "expiration_date": "2025-01-17"},
        {"ticker": "O:AAPL250117P00150000", "strike_price": 150, "expiration_date": "2025-01-17"},
    ]
    import ai_trader.backtest as bt_mod
    monkeypatch.setattr(
        bt_mod, "fetch_polygon_option_contracts",
        lambda *args, **kwargs: contracts,
    )

    result = _select_real_contract(
        api_key="fake",
        underlying="AAPL",
        option_type="put",
        spot=150.0,
        trade_date=date(2025, 1, 10),
        strike_preference="otm",
        expiry_preference="next_week",
        default_dte=14,
    )
    assert result is not None
    # 150 * 0.97 = 145.5, closest is 145
    assert result["strike_price"] == 145


def test_select_real_contract_itm_call(monkeypatch):
    """ITM call should pick strike below spot."""
    contracts = [
        {"ticker": "O:AAPL250117C00140000", "strike_price": 140, "expiration_date": "2025-01-17"},
        {"ticker": "O:AAPL250117C00145000", "strike_price": 145, "expiration_date": "2025-01-17"},
        {"ticker": "O:AAPL250117C00150000", "strike_price": 150, "expiration_date": "2025-01-17"},
    ]
    import ai_trader.backtest as bt_mod
    monkeypatch.setattr(
        bt_mod, "fetch_polygon_option_contracts",
        lambda *args, **kwargs: contracts,
    )

    result = _select_real_contract(
        api_key="fake",
        underlying="AAPL",
        option_type="call",
        spot=150.0,
        trade_date=date(2025, 1, 10),
        strike_preference="itm",
        expiry_preference="next_week",
        default_dte=14,
    )
    assert result is not None
    # 150 * 0.97 = 145.5, closest is 145
    assert result["strike_price"] == 145


def test_select_real_contract_no_contracts(monkeypatch):
    """Returns None when no contracts found."""
    import ai_trader.backtest as bt_mod
    monkeypatch.setattr(
        bt_mod, "fetch_polygon_option_contracts",
        lambda *args, **kwargs: [],
    )

    result = _select_real_contract(
        api_key="fake",
        underlying="AAPL",
        option_type="call",
        spot=150.0,
        trade_date=date(2025, 1, 10),
        strike_preference="atm",
        expiry_preference="next_week",
        default_dte=14,
    )
    assert result is None


# ---------------------------------------------------------------------------
# SimPosition / SimTrade
# ---------------------------------------------------------------------------

def test_sim_position_polygon_ticker():
    pos = SimPosition(
        underlying="AAPL",
        option_type="call",
        strike=150.0,
        entry_date=date(2025, 1, 10),
        expiry_date=date(2025, 1, 17),
        entry_premium=5.0,
        qty=2,
        conviction=0.85,
        reasoning="test",
        polygon_ticker="O:AAPL250117C00150000",
    )
    assert pos.polygon_ticker == "O:AAPL250117C00150000"
    assert pos.dte_from == 7


def test_sim_position_default_polygon_ticker():
    pos = SimPosition(
        underlying="AAPL",
        option_type="call",
        strike=150.0,
        entry_date=date(2025, 1, 10),
        expiry_date=date(2025, 1, 17),
        entry_premium=5.0,
        qty=2,
        conviction=0.85,
        reasoning="test",
    )
    assert pos.polygon_ticker == ""


def test_sim_trade_polygon_ticker():
    trade = SimTrade(
        entry_date=date(2025, 1, 2),
        exit_date=date(2025, 1, 10),
        underlying="AAPL",
        option_type="call",
        strike=150.0,
        entry_premium=5.0,
        exit_premium=7.5,
        qty=2,
        pnl=500.0,
        exit_reason="profit_target",
        conviction=0.85,
        polygon_ticker="O:AAPL250117C00150000",
    )
    assert trade.polygon_ticker == "O:AAPL250117C00150000"


# ---------------------------------------------------------------------------
# BacktestResult / print
# ---------------------------------------------------------------------------

def test_backtest_result_defaults():
    r = BacktestResult()
    assert r.initial_equity == 100_000
    assert r.final_equity == 100_000
    assert r.total_trades == 0


def test_print_backtest_result_no_crash():
    r = BacktestResult(
        trades=[
            SimTrade(
                entry_date=date(2025, 1, 2),
                exit_date=date(2025, 1, 10),
                underlying="AAPL",
                option_type="call",
                strike=150.0,
                entry_premium=5.0,
                exit_premium=7.5,
                qty=2,
                pnl=500.0,
                exit_reason="profit_target",
                conviction=0.85,
                polygon_ticker="O:AAPL250117C00150000",
            )
        ],
        initial_equity=100_000,
        final_equity=100_500,
        total_return_pct=0.005,
        total_trades=1,
        wins=1,
        losses=0,
        win_rate=1.0,
        net_pnl=500.0,
        max_drawdown=0.0,
        days_tested=7,
    )
    # Should not raise
    print_backtest_result(r)


def test_print_backtest_result_no_ticker():
    """Print works when polygon_ticker is empty (backwards compat)."""
    r = BacktestResult(
        trades=[
            SimTrade(
                entry_date=date(2025, 1, 2),
                exit_date=date(2025, 1, 10),
                underlying="AAPL",
                option_type="call",
                strike=150.0,
                entry_premium=5.0,
                exit_premium=7.5,
                qty=2,
                pnl=500.0,
                exit_reason="profit_target",
                conviction=0.85,
            )
        ],
        initial_equity=100_000,
        final_equity=100_500,
        total_return_pct=0.005,
        total_trades=1,
        wins=1,
        losses=0,
        win_rate=1.0,
        net_pnl=500.0,
        max_drawdown=0.0,
        days_tested=7,
    )
    print_backtest_result(r)


# ---------------------------------------------------------------------------
# _build_performance_summary
# ---------------------------------------------------------------------------

def test_performance_summary_empty():
    assert _build_performance_summary([], 100_000, 100_000) == ""


def test_performance_summary_single_win():
    trades = [{"underlying": "AAPL", "pnl": 500, "timestamp": "2025-01-10"}]
    result = _build_performance_summary(trades, 100_500, 100_000)
    assert "Win rate: 100.0%" in result
    assert "Wins: 1" in result
    assert "Losses: 0" in result
    assert "$+500" in result
    assert "W" in result  # streak


def test_performance_summary_mixed():
    trades = [
        {"underlying": "AAPL", "pnl": 500, "timestamp": "2025-01-10"},
        {"underlying": "TSLA", "pnl": -300, "timestamp": "2025-01-11"},
        {"underlying": "AAPL", "pnl": 200, "timestamp": "2025-01-12"},
    ]
    result = _build_performance_summary(trades, 100_400, 100_000)
    assert "Total trades: 3" in result
    assert "Wins: 2" in result
    assert "Losses: 1" in result
    assert "Win rate: 66.7%" in result
    assert "W, L, W" in result  # streak
    assert "AAPL" in result
    assert "TSLA" in result


def test_performance_summary_per_ticker_stats():
    trades = [
        {"underlying": "AAPL", "pnl": 500},
        {"underlying": "AAPL", "pnl": -100},
        {"underlying": "TSLA", "pnl": -200},
    ]
    result = _build_performance_summary(trades, 100_200, 100_000)
    assert "AAPL: 1W/1L" in result
    assert "TSLA: 0W/1L" in result


def test_performance_summary_all_losses():
    trades = [
        {"underlying": "AAPL", "pnl": -500},
        {"underlying": "TSLA", "pnl": -300},
    ]
    result = _build_performance_summary(trades, 99_200, 100_000)
    assert "Win rate: 0.0%" in result
    assert "L, L" in result
    assert "-0.8%" in result  # return


# ---------------------------------------------------------------------------
# _extract_top_news_tickers
# ---------------------------------------------------------------------------

def test_extract_tickers_basic():
    news = [
        {"tickers": ["AAPL", "TSLA"]},
        {"tickers": ["AAPL", "MSFT"]},
        {"tickers": ["AAPL"]},
    ]
    result = _extract_top_news_tickers(news)
    assert result[0] == "AAPL"  # most mentioned
    assert "TSLA" in result
    assert "MSFT" in result


def test_extract_tickers_excludes_indices():
    news = [
        {"tickers": ["SPY", "QQQ", "AAPL"]},
        {"tickers": ["SPY", "IWM", "DIA"]},
    ]
    result = _extract_top_news_tickers(news)
    assert "SPY" not in result
    assert "QQQ" not in result
    assert "IWM" not in result
    assert "DIA" not in result
    assert result == ["AAPL"]


def test_extract_tickers_empty_news():
    assert _extract_top_news_tickers([]) == []


def test_extract_tickers_no_tickers_key():
    news = [{"title": "Some headline"}]
    assert _extract_top_news_tickers(news) == []


def test_extract_tickers_max_limit():
    news = [{"tickers": [f"T{i}"]} for i in range(20)]
    result = _extract_top_news_tickers(news, max_tickers=3)
    assert len(result) == 3


def test_extract_tickers_custom_exclude():
    news = [{"tickers": ["AAPL", "TSLA", "GOOG"]}]
    result = _extract_top_news_tickers(news, exclude={"AAPL"})
    assert "AAPL" not in result
    assert "TSLA" in result
    assert "GOOG" in result


# ---------------------------------------------------------------------------
# _build_enriched_portfolio_context
# ---------------------------------------------------------------------------

def test_enriched_portfolio_no_positions():
    result = _build_enriched_portfolio_context(
        equity=100_000, initial_equity=100_000,
        positions=[], trade_date=date(2025, 1, 10),
        api_key="fake", cache=PolygonCache(),
        profit_target_pct=0.50, stop_loss_pct=0.40, time_stop_dte=2,
    )
    assert "Account Equity: $100,000" in result
    assert "Open Positions: 0" in result
    assert "unrealized" not in result.lower()


def test_enriched_portfolio_with_cached_position():
    cache = PolygonCache()
    ticker = "O:AAPL250117C00150000"
    # Simulate a cached bar (as would exist from exit check)
    cache.option_bars[ticker] = {
        "2025-01-10": {"c": 7.5, "vw": 7.5, "v": 100},
    }
    pos = SimPosition(
        underlying="AAPL", option_type="call", strike=150.0,
        entry_date=date(2025, 1, 6), expiry_date=date(2025, 1, 17),
        entry_premium=5.0, qty=2, conviction=0.85, reasoning="test",
        polygon_ticker=ticker,
    )
    result = _build_enriched_portfolio_context(
        equity=100_000, initial_equity=100_000,
        positions=[pos], trade_date=date(2025, 1, 10),
        api_key="fake", cache=cache,
        profit_target_pct=0.50, stop_loss_pct=0.40, time_stop_dte=2,
    )
    assert "current=$7.50" in result
    assert "+50.0%" in result
    assert "$+500" in result
    assert "approaching profit target" in result
    assert "Total unrealized P&L" in result


def test_enriched_portfolio_losing_position():
    cache = PolygonCache()
    ticker = "O:TSLA250117P00200000"
    cache.option_bars[ticker] = {
        "2025-01-10": {"c": 3.0, "vw": 3.0, "v": 50},
    }
    pos = SimPosition(
        underlying="TSLA", option_type="put", strike=200.0,
        entry_date=date(2025, 1, 6), expiry_date=date(2025, 1, 17),
        entry_premium=5.0, qty=1, conviction=0.7, reasoning="test",
        polygon_ticker=ticker,
    )
    result = _build_enriched_portfolio_context(
        equity=95_000, initial_equity=100_000,
        positions=[pos], trade_date=date(2025, 1, 10),
        api_key="fake", cache=cache,
        profit_target_pct=0.50, stop_loss_pct=0.40, time_stop_dte=2,
    )
    assert "-5.0%" in result  # return vs start
    assert "-40.0%" in result  # unrealized
    assert "approaching stop loss" in result


def test_enriched_portfolio_no_bar_in_cache():
    """When no bar is cached, still shows basic position info."""
    cache = PolygonCache()
    pos = SimPosition(
        underlying="AAPL", option_type="call", strike=150.0,
        entry_date=date(2025, 1, 6), expiry_date=date(2025, 1, 17),
        entry_premium=5.0, qty=2, conviction=0.85, reasoning="test",
        polygon_ticker="O:AAPL250117C00150000",
    )
    result = _build_enriched_portfolio_context(
        equity=100_000, initial_equity=100_000,
        positions=[pos], trade_date=date(2025, 1, 10),
        api_key="fake", cache=cache,
        profit_target_pct=0.50, stop_loss_pct=0.40, time_stop_dte=2,
    )
    assert "entry=$5.00 qty=2" in result
    # No per-position unrealized line (no current price available)
    assert "unrealized=" not in result


def test_enriched_portfolio_time_stop_flag():
    cache = PolygonCache()
    ticker = "O:AAPL250113C00150000"
    cache.option_bars[ticker] = {
        "2025-01-10": {"c": 5.5, "vw": 5.5, "v": 100},
    }
    pos = SimPosition(
        underlying="AAPL", option_type="call", strike=150.0,
        entry_date=date(2025, 1, 6), expiry_date=date(2025, 1, 13),
        entry_premium=5.0, qty=1, conviction=0.8, reasoning="test",
        polygon_ticker=ticker,
    )
    result = _build_enriched_portfolio_context(
        equity=100_000, initial_equity=100_000,
        positions=[pos], trade_date=date(2025, 1, 10),
        api_key="fake", cache=cache,
        profit_target_pct=0.50, stop_loss_pct=0.40, time_stop_dte=2,
    )
    assert "approaching time stop" in result
