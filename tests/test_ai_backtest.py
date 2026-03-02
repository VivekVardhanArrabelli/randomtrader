"""Tests for the AI trader backtest module."""

from datetime import date, timedelta

from ai_trader.backtest import (
    BacktestConfig,
    BacktestResult,
    PolygonCache,
    SimPosition,
    SimTrade,
    _annotate_closed_trades,
    _build_enriched_portfolio_context,
    _build_market_trend_context,
    _build_performance_summary,
    _build_ticker_price_context,
    _extract_top_news_tickers,
    _filter_news_quality,
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


# ---------------------------------------------------------------------------
# _build_market_trend_context
# ---------------------------------------------------------------------------

def _make_bars(closes: list[float], start_date: date = date(2024, 12, 16)) -> list[dict]:
    """Helper: build mock daily bar list from close prices."""
    bars = []
    d = start_date
    for c in closes:
        bars.append({"o": c * 0.99, "h": c * 1.01, "l": c * 0.98, "c": c, "v": 1_000_000})
        d += timedelta(days=1)
    return bars


def test_build_market_trend_context_basic(monkeypatch):
    """Should show today's price, 5d/10d change, and trend direction."""
    import ai_trader.backtest as bt_mod

    # 12 bars: indices 0-11, so bars[-1] = today, bars[-6] = 5d ago, bars[-11] = 10d ago
    spy_closes = [570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 580, 582]
    qqq_closes = [490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 498]
    iwm_closes = [230, 229, 228, 227, 226, 225, 224, 223, 222, 221, 220, 218]

    def mock_bars_range(api_key, symbol, start, end):
        if symbol == "SPY":
            return _make_bars(spy_closes)
        elif symbol == "QQQ":
            return _make_bars(qqq_closes)
        elif symbol == "IWM":
            return _make_bars(iwm_closes)
        return []

    monkeypatch.setattr(bt_mod, "fetch_historical_daily_bars_range", mock_bars_range)

    result = _build_market_trend_context("fake", date(2025, 1, 10))
    assert "Major Indices (10-day view):" in result
    assert "SPY:" in result
    assert "QQQ:" in result
    assert "IWM:" in result
    assert "today(" in result
    assert "5d(" in result
    assert "10d(" in result
    assert "trend=" in result


def test_build_market_trend_context_trend_direction(monkeypatch):
    """Trend should be 'up' when price is above 10-day avg, 'down' when below."""
    import ai_trader.backtest as bt_mod

    # Rising: today=582, avg of last 10 ~ 575 → up
    spy_closes = [570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 580, 582]
    # Falling: today=218, avg of last 10 ~ 224 → down
    iwm_closes = [230, 229, 228, 227, 226, 225, 224, 223, 222, 221, 220, 218]

    def mock_bars(api_key, symbol, start, end):
        if symbol == "SPY":
            return _make_bars(spy_closes)
        elif symbol == "IWM":
            return _make_bars(iwm_closes)
        return []

    monkeypatch.setattr(bt_mod, "fetch_historical_daily_bars_range", mock_bars)

    result = _build_market_trend_context("fake", date(2025, 1, 10))
    # SPY rising → trend=up
    spy_line = [l for l in result.split("\n") if "SPY:" in l]
    assert spy_line and "trend=up" in spy_line[0]
    # IWM falling → trend=down
    iwm_line = [l for l in result.split("\n") if "IWM:" in l]
    assert iwm_line and "trend=down" in iwm_line[0]


def test_build_market_trend_context_no_data(monkeypatch):
    """Should handle empty bar data gracefully."""
    import ai_trader.backtest as bt_mod
    monkeypatch.setattr(bt_mod, "fetch_historical_daily_bars_range", lambda *a, **k: [])

    result = _build_market_trend_context("fake", date(2025, 1, 10))
    assert "Major Indices (10-day view):" in result
    # No index lines since no data
    lines = [l for l in result.split("\n") if l.strip().startswith("SPY:")]
    assert len(lines) == 0


# ---------------------------------------------------------------------------
# _build_ticker_price_context
# ---------------------------------------------------------------------------

def test_build_ticker_price_context_basic(monkeypatch):
    """Should show spot, today change, 5d/10d change, hi/lo and return spot price."""
    import ai_trader.backtest as bt_mod

    closes = [130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 142]
    monkeypatch.setattr(
        bt_mod, "fetch_historical_daily_bars_range",
        lambda *a, **k: _make_bars(closes),
    )

    ctx, spot = _build_ticker_price_context("fake", "NVDA", date(2025, 1, 10))
    assert ctx is not None
    assert spot == 142.0
    assert "spot=$142.00" in ctx
    assert "today(" in ctx
    assert "5d(" in ctx
    assert "10d(" in ctx
    assert "hi/lo=" in ctx


def test_build_ticker_price_context_no_data(monkeypatch):
    """Should return (None, 0.0) when no bars available."""
    import ai_trader.backtest as bt_mod
    monkeypatch.setattr(bt_mod, "fetch_historical_daily_bars_range", lambda *a, **k: [])

    ctx, spot = _build_ticker_price_context("fake", "NVDA", date(2025, 1, 10))
    assert ctx is None
    assert spot == 0.0


def test_build_ticker_price_context_short_history(monkeypatch):
    """Should handle fewer than 10 bars (no 10d change, limited hi/lo)."""
    import ai_trader.backtest as bt_mod

    # Only 3 bars — not enough for 5d or 10d change
    closes = [140, 141, 142]
    monkeypatch.setattr(
        bt_mod, "fetch_historical_daily_bars_range",
        lambda *a, **k: _make_bars(closes),
    )

    ctx, spot = _build_ticker_price_context("fake", "NVDA", date(2025, 1, 10))
    assert ctx is not None
    assert spot == 142.0
    assert "spot=$142.00" in ctx
    # 5d and 10d changes should be 0 (not enough data)
    assert "5d(+0.0%)" in ctx
    assert "10d(+0.0%)" in ctx


# ---------------------------------------------------------------------------
# _annotate_closed_trades
# ---------------------------------------------------------------------------

def test_annotate_closed_trades_stop_loss(monkeypatch):
    """Stop-loss trade should get underlying movement annotation."""
    import ai_trader.backtest as bt_mod

    bars = [
        {"o": 140, "h": 142, "l": 139, "c": 140, "v": 100},  # entry day
        {"o": 141, "h": 145, "l": 140, "c": 144, "v": 100},
        {"o": 144, "h": 149, "l": 143, "c": 148, "v": 100},  # exit day
    ]
    monkeypatch.setattr(bt_mod, "fetch_historical_daily_bars_range", lambda *a, **k: bars)

    trades = [{
        "timestamp": "2025-01-12",
        "entry_date": "2025-01-10",
        "underlying": "NVDA",
        "option_type": "put",
        "entry_premium": 5.0,
        "exit_premium": 3.0,
        "pnl": -200,
        "reason": "stop_loss",
        "polygon_ticker": "O:NVDA250117P00140000",
    }]
    result = _annotate_closed_trades(trades, "fake", PolygonCache())
    assert len(result) == 1
    assert "context" in result[0]
    assert "NVDA" in result[0]["context"]
    assert "$140→$148" in result[0]["context"]
    assert "bearish" in result[0]["context"]


def test_annotate_closed_trades_profit_target(monkeypatch):
    """Profit target trades should also get annotations."""
    import ai_trader.backtest as bt_mod

    bars = [
        {"o": 150, "h": 151, "l": 149, "c": 150, "v": 100},  # entry day
        {"o": 151, "h": 160, "l": 150, "c": 158, "v": 100},  # exit day
    ]
    monkeypatch.setattr(bt_mod, "fetch_historical_daily_bars_range", lambda *a, **k: bars)

    trades = [{
        "timestamp": "2025-01-12",
        "entry_date": "2025-01-10",
        "underlying": "AAPL",
        "option_type": "call",
        "entry_premium": 5.0,
        "exit_premium": 8.0,
        "pnl": 300,
        "reason": "profit_target",
        "polygon_ticker": "O:AAPL250117C00150000",
    }]
    result = _annotate_closed_trades(trades, "fake", PolygonCache())
    assert len(result) == 1
    assert "context" in result[0]
    assert "AAPL" in result[0]["context"]
    assert "bullish" in result[0]["context"]


def test_annotate_closed_trades_bullish_direction(monkeypatch):
    """Call option stop-loss should say 'bullish' direction."""
    import ai_trader.backtest as bt_mod

    bars = [
        {"o": 150, "h": 151, "l": 149, "c": 150, "v": 100},
        {"o": 149, "h": 150, "l": 145, "c": 145, "v": 100},
    ]
    monkeypatch.setattr(bt_mod, "fetch_historical_daily_bars_range", lambda *a, **k: bars)

    trades = [{
        "timestamp": "2025-01-11",
        "entry_date": "2025-01-10",
        "underlying": "AAPL",
        "option_type": "call",
        "entry_premium": 5.0,
        "exit_premium": 2.5,
        "pnl": -250,
        "reason": "stop_loss",
        "polygon_ticker": "O:AAPL250117C00150000",
    }]
    result = _annotate_closed_trades(trades, "fake", PolygonCache())
    assert "bullish" in result[0]["context"]


def test_annotate_closed_trades_empty():
    """Empty trades list should return empty list."""
    result = _annotate_closed_trades([], "fake", PolygonCache())
    assert result == []


def test_annotate_closed_trades_no_mutation(monkeypatch):
    """Original trade dicts should not be mutated."""
    import ai_trader.backtest as bt_mod

    bars = [
        {"o": 140, "h": 142, "l": 139, "c": 140, "v": 100},
        {"o": 138, "h": 140, "l": 135, "c": 136, "v": 100},
    ]
    monkeypatch.setattr(bt_mod, "fetch_historical_daily_bars_range", lambda *a, **k: bars)

    original = {
        "timestamp": "2025-01-12",
        "entry_date": "2025-01-10",
        "underlying": "NVDA",
        "option_type": "put",
        "pnl": -200,
        "reason": "time_stop",
        "polygon_ticker": "O:NVDA250117P00140000",
    }
    trades = [original]
    result = _annotate_closed_trades(trades, "fake", PolygonCache())
    assert "context" not in original  # original not modified
    assert "context" in result[0]  # but annotated copy has it


# ---------------------------------------------------------------------------
# _filter_news_quality
# ---------------------------------------------------------------------------

def test_filter_news_quality_removes_lawsuit_spam():
    articles = [
        {"title": "ROSEN, A LEADING LAW FIRM, Encourages Investors to Secure Counsel"},
        {"title": "Securities Fraud Class Action Filed Against XYZ Corp"},
        {"title": "DEADLINE REMINDER: Recover Your Losses in ABC Lawsuit"},
        {"title": "Investor Rights Law Firm Reminds Investors of Deadline"},
        {"title": "Have Suffered Losses? Contact Our Securities Class Action Team"},
        {"title": "Apple Reports Record Q4 Earnings"},  # real news — keep
    ]
    result = _filter_news_quality(articles)
    assert len(result) == 1
    assert result[0]["title"] == "Apple Reports Record Q4 Earnings"


def test_filter_news_quality_removes_corporate_housekeeping():
    articles = [
        {"title": "Company X Announces Inducement Grant Under Nasdaq Listing Rule 5635"},
        {"title": "Nokia Oyj: Repurchase of Own Shares on 15.1.2025"},
        {"title": "Nokia Oyj: Omien osakkeiden hankkiminen 15.1.2025"},
        {"title": "FDA Approves Breakthrough Cancer Drug"},  # real news — keep
    ]
    result = _filter_news_quality(articles)
    assert len(result) == 1
    assert result[0]["title"] == "FDA Approves Breakthrough Cancer Drug"


def test_filter_news_quality_keeps_real_news():
    articles = [
        {"title": "Hindenburg Research Releases Short Report on Carvana"},
        {"title": "Apple Reports Record Q4 Earnings, Stock Surges 5%"},
        {"title": "FDA Approves Pfizer's New COVID Booster for Fall 2025"},
        {"title": "Tesla Deliveries Miss Expectations, Shares Drop 8%"},
        {"title": "Fed Holds Rates Steady, Signals Cuts Coming in March"},
    ]
    result = _filter_news_quality(articles)
    assert len(result) == 5


def test_filter_news_quality_case_insensitive():
    articles = [
        {"title": "SECURITIES FRAUD class action FILED"},
        {"title": "deadline ALERT for loss RECOVERY"},
        {"title": "Leading Law Firm Encourages Acme Investors"},
    ]
    result = _filter_news_quality(articles)
    assert len(result) == 0


def test_filter_news_quality_empty():
    assert _filter_news_quality([]) == []


# ---------------------------------------------------------------------------
# _build_performance_summary — repeat losers
# ---------------------------------------------------------------------------

def test_performance_summary_repeat_losers():
    """Tickers with 2+ losses and 0 wins should appear in Repeat losers."""
    trades = [
        {"underlying": "NVDA", "pnl": -5000, "entry_date": "2025-01-02", "timestamp": "2025-01-04"},
        {"underlying": "NVDA", "pnl": -6000, "entry_date": "2025-01-06", "timestamp": "2025-01-08"},
        {"underlying": "NVDA", "pnl": -4000, "entry_date": "2025-01-10", "timestamp": "2025-01-13"},
        {"underlying": "AAPL", "pnl": 2000, "entry_date": "2025-01-03", "timestamp": "2025-01-05"},
    ]
    result = _build_performance_summary(trades, 87_000, 100_000)
    assert "Repeat losers:" in result
    assert "NVDA: 3 trades, 0 wins" in result
    assert "avg hold:" in result
    assert "avg loss:" in result
    # AAPL should NOT appear in repeat losers (it has a win)
    assert "AAPL" not in result.split("Repeat losers:")[1]


def test_performance_summary_no_repeat_losers():
    """No Repeat losers section when all tickers have at least 1 win."""
    trades = [
        {"underlying": "NVDA", "pnl": 500, "timestamp": "2025-01-05"},
        {"underlying": "NVDA", "pnl": -300, "timestamp": "2025-01-08"},
        {"underlying": "AAPL", "pnl": 200, "timestamp": "2025-01-06"},
        {"underlying": "AAPL", "pnl": -100, "timestamp": "2025-01-09"},
    ]
    result = _build_performance_summary(trades, 100_300, 100_000)
    assert "Repeat losers:" not in result


def test_performance_summary_repeat_losers_hold_days():
    """Avg hold days should be computed from entry_date and timestamp."""
    trades = [
        {"underlying": "META", "pnl": -1000, "entry_date": "2025-01-02", "timestamp": "2025-01-04"},  # 2 days
        {"underlying": "META", "pnl": -2000, "entry_date": "2025-01-06", "timestamp": "2025-01-10"},  # 4 days
    ]
    result = _build_performance_summary(trades, 97_000, 100_000)
    assert "Repeat losers:" in result
    assert "META: 2 trades, 0 wins" in result
    # avg hold = (2+4)/2 = 3.0 days
    assert "avg hold: 3.0 days" in result


# ---------------------------------------------------------------------------
# _annotate_closed_trades — all reasons
# ---------------------------------------------------------------------------

def test_annotate_closed_trades_all_reasons(monkeypatch):
    """Profit target and time_stop trades should also get annotations."""
    import ai_trader.backtest as bt_mod

    bars = [
        {"o": 100, "h": 102, "l": 99, "c": 100, "v": 100},
        {"o": 101, "h": 110, "l": 100, "c": 108, "v": 100},
    ]
    monkeypatch.setattr(bt_mod, "fetch_historical_daily_bars_range", lambda *a, **k: bars)

    trades = [
        {
            "timestamp": "2025-01-11",
            "entry_date": "2025-01-10",
            "underlying": "AAPL",
            "option_type": "call",
            "pnl": 500,
            "reason": "profit_target",
        },
        {
            "timestamp": "2025-01-11",
            "entry_date": "2025-01-10",
            "underlying": "TSLA",
            "option_type": "put",
            "pnl": -100,
            "reason": "time_stop",
        },
    ]
    result = _annotate_closed_trades(trades, "fake", PolygonCache())
    assert len(result) == 2
    assert "context" in result[0]
    assert "bullish" in result[0]["context"]
    assert "context" in result[1]
    assert "bearish" in result[1]["context"]
