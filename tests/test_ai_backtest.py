"""Tests for the AI trader backtest module."""

import sqlite3
from datetime import date, datetime, timedelta

import pytest

from ai_trader.brain import AnalysisRun, MarketAnalysis, TradeDecision
from ai_trader.backtest import (
    BacktestConfig,
    BacktestResult,
    PolygonCache,
    SimPosition,
    SimTrade,
    _decision_timestamps_for_day,
    _first_bar_at_or_after,
    _is_trading_day,
    _last_completed_trading_day,
    _latest_bar_before,
    _annotate_closed_trades,
    _build_enriched_portfolio_context,
    _build_market_trend_context,
    _build_options_context,
    _build_performance_summary,
    _build_ticker_price_context,
    _extract_top_news_tickers,
    _filter_news_quality,
    _option_bar_price,
    _exit_premium_for_position,
    _prefetch_prepare_option_data,
    _rank_prefetch_contracts,
    _previous_trading_day,
    _session_intraday_bars_before,
    _ticker_price_metrics_as_of,
    _select_real_contract,
    _trading_days,
    fetch_historical_daily_bars_range,
    print_backtest_result,
    save_debug_log,
)
from ai_trader.historical_cache import PolygonResponseStore
from ai_trader.llm import LLMCompletion, LLMDecisionPacket, ToolCall
from ai_trader.news import NewsEvent
from ai_trader.utils import EASTERN_TZ


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


def test_previous_trading_day_skips_weekend():
    assert _previous_trading_day(date(2025, 1, 6)) == date(2025, 1, 3)  # Monday -> Friday
    assert _previous_trading_day(date(2025, 1, 7)) == date(2025, 1, 6)  # Tuesday -> Monday


def test_trading_days_skip_nyse_holidays():
    days = _trading_days(date(2025, 12, 24), date(2025, 12, 26))

    assert days == [date(2025, 12, 24), date(2025, 12, 26)]
    assert not _is_trading_day(date(2025, 12, 25))


def test_previous_trading_day_skips_thanksgiving():
    assert _previous_trading_day(date(2025, 11, 28)) == date(2025, 11, 26)


def test_last_completed_trading_day_intraday_uses_prior_session():
    as_of = datetime(2025, 1, 6, 9, 35, tzinfo=EASTERN_TZ)
    assert _last_completed_trading_day(as_of) == date(2025, 1, 3)


def test_last_completed_trading_day_skips_holiday_session():
    as_of = datetime(2025, 12, 25, 12, 0, tzinfo=EASTERN_TZ)
    assert _last_completed_trading_day(as_of) == date(2025, 12, 24)


def test_decision_timestamps_for_day_match_scan_cadence():
    times = _decision_timestamps_for_day(
        date(2025, 1, 6),
        interval_minutes=15,
        start_delay_minutes=5,
        end_buffer_minutes=15,
    )
    assert times[0].strftime("%H:%M") == "09:35"
    assert times[1].strftime("%H:%M") == "09:50"
    assert times[-1].strftime("%H:%M") == "15:35"


def test_latest_bar_before_and_first_bar_after_cutoff():
    bars = [
        {"t": int(datetime(2025, 1, 6, 14, 30, tzinfo=EASTERN_TZ).timestamp() * 1000), "c": 100.0},
        {"t": int(datetime(2025, 1, 6, 14, 35, tzinfo=EASTERN_TZ).timestamp() * 1000), "c": 101.0},
        {"t": int(datetime(2025, 1, 6, 14, 40, tzinfo=EASTERN_TZ).timestamp() * 1000), "c": 102.0},
    ]
    cutoff = datetime(2025, 1, 6, 14, 35, tzinfo=EASTERN_TZ)
    assert _latest_bar_before(bars, cutoff)["c"] == 100.0
    assert _first_bar_at_or_after(bars, cutoff)["c"] == 101.0


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


def test_fetch_historical_daily_bars_range_uses_cache(monkeypatch):
    import ai_trader.backtest as bt_mod

    calls = {"count": 0}
    bars = _make_bars([100, 101, 102])

    def mock_polygon_request(api_key, path, params=None, store=None, offline=False):
        calls["count"] += 1
        return {"results": bars}

    monkeypatch.setattr(bt_mod, "_polygon_request", mock_polygon_request)

    cache = PolygonCache()
    first = fetch_historical_daily_bars_range(
        "fake", "AAPL", date(2025, 1, 1), date(2025, 1, 3), cache=cache,
    )
    second = fetch_historical_daily_bars_range(
        "fake", "AAPL", date(2025, 1, 1), date(2025, 1, 3), cache=cache,
    )

    assert first == second == bars
    assert calls["count"] == 1


def test_polygon_request_uses_persistent_store(tmp_path, monkeypatch):
    import ai_trader.backtest as bt_mod

    calls = {"count": 0}

    class DummyResponse:
        status_code = 200
        text = ""

        def json(self):
            return {"results": [{"ticker": "AAPL"}]}

    def mock_get(url, params=None, timeout=30):
        calls["count"] += 1
        return DummyResponse()

    monkeypatch.setattr("requests.get", mock_get)

    store = PolygonResponseStore(tmp_path / "historical.db")
    first = bt_mod._polygon_request("live-key", "/v2/test", {"ticker": "AAPL"}, store=store)
    second = bt_mod._polygon_request("", "/v2/test", {"ticker": "AAPL"}, store=store, offline=True)

    assert first == second == {"results": [{"ticker": "AAPL"}]}
    assert calls["count"] == 1


def test_polygon_request_offline_cache_miss_raises(tmp_path):
    import ai_trader.backtest as bt_mod

    store = PolygonResponseStore(tmp_path / "historical.db")
    with pytest.raises(RuntimeError, match="offline Polygon cache miss"):
        bt_mod._polygon_request("", "/v2/missing", {"ticker": "AAPL"}, store=store, offline=True)


def test_polygon_request_adapts_interval_after_429(monkeypatch):
    import ai_trader.backtest as bt_mod

    sleeps: list[float] = []
    calls = {"count": 0}
    base_interval = bt_mod.config.POLYGON_MIN_REQUEST_INTERVAL_SECONDS

    class DummyResponse:
        def __init__(self, status_code: int, payload: dict, *, headers: dict | None = None, text: str = ""):
            self.status_code = status_code
            self._payload = payload
            self.headers = headers or {}
            self.text = text

        def json(self):
            return self._payload

    responses = [
        DummyResponse(
            429,
            {"status": "ERROR"},
            headers={"Retry-After": "4"},
            text="rate limited",
        ),
        DummyResponse(200, {"results": [{"ticker": "AAPL"}]}),
    ]

    def mock_get(url, params=None, timeout=30):
        calls["count"] += 1
        return responses.pop(0)

    monkeypatch.setattr("requests.get", mock_get)
    monkeypatch.setattr(bt_mod.time_module, "sleep", lambda seconds: sleeps.append(seconds))
    monkeypatch.setattr(bt_mod, "_LAST_POLYGON_REQUEST_AT", 0.0)
    monkeypatch.setattr(bt_mod, "_POLYGON_REQUEST_INTERVAL_SECONDS", base_interval)

    result = bt_mod._polygon_request("live-key", "/v2/test", {"ticker": "AAPL"})

    assert result == {"results": [{"ticker": "AAPL"}]}
    assert calls["count"] == 2
    assert 4.0 in sleeps
    assert bt_mod._POLYGON_REQUEST_INTERVAL_SECONDS > base_interval


def test_fetch_historical_daily_bars_range_offline_reraises(monkeypatch):
    import ai_trader.backtest as bt_mod

    def mock_polygon_request(api_key, path, params=None, store=None, offline=False):
        raise RuntimeError("offline Polygon cache miss for /v2/aggs/ticker/AAPL")

    monkeypatch.setattr(bt_mod, "_polygon_request", mock_polygon_request)

    cache = PolygonCache(offline=True)
    with pytest.raises(RuntimeError, match="offline Polygon cache miss"):
        fetch_historical_daily_bars_range(
            "fake", "AAPL", date(2025, 1, 1), date(2025, 1, 3), cache=cache,
        )


def test_session_intraday_bars_before_uses_filtered_cache(monkeypatch):
    import ai_trader.backtest as bt_mod

    calls = {"count": 0}
    bars = [
        {"t": int(datetime(2025, 1, 6, 9, 30, tzinfo=EASTERN_TZ).timestamp() * 1000), "c": 100.0},
        {"t": int(datetime(2025, 1, 6, 9, 35, tzinfo=EASTERN_TZ).timestamp() * 1000), "c": 101.0},
    ]

    def mock_intraday(api_key, ticker, trading_day, multiplier=5, cache=None):
        calls["count"] += 1
        return bars

    monkeypatch.setattr(bt_mod, "fetch_historical_intraday_bars", mock_intraday)

    cache = PolygonCache()
    as_of = datetime(2025, 1, 6, 9, 36, tzinfo=EASTERN_TZ)
    first = _session_intraday_bars_before("fake", "AAPL", as_of, cache=cache)
    second = _session_intraday_bars_before("fake", "AAPL", as_of, cache=cache)

    assert first == second == bars
    assert calls["count"] == 1


def test_ticker_price_metrics_as_of_uses_metrics_cache(monkeypatch):
    import ai_trader.backtest as bt_mod

    calls = {"count": 0}
    bars = _make_bars([130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 142])

    def mock_bars_range(api_key, symbol, start, end, cache=None):
        calls["count"] += 1
        return bars

    monkeypatch.setattr(bt_mod, "fetch_historical_daily_bars_range", mock_bars_range)
    monkeypatch.setattr(bt_mod, "_session_intraday_bars_before", lambda *a, **k: [])

    cache = PolygonCache()
    metrics_cache: dict[str, dict | None] = {}
    first = _ticker_price_metrics_as_of(
        "fake",
        "NVDA",
        date(2025, 1, 10),
        cache=cache,
        metrics_cache=metrics_cache,
    )
    second = _ticker_price_metrics_as_of(
        "fake",
        "NVDA",
        date(2025, 1, 10),
        cache=cache,
        metrics_cache=metrics_cache,
    )

    assert first == second
    assert first is not None
    assert calls["count"] == 1


def test_rank_prefetch_contracts_balances_moneyness_buckets():
    contracts = [
        {"ticker": "ATM1", "strike_price": 100, "expiration_date": "2025-01-24"},
        {"ticker": "ITM1", "strike_price": 95, "expiration_date": "2025-01-24"},
        {"ticker": "OTM1", "strike_price": 105, "expiration_date": "2025-01-24"},
        {"ticker": "ATM2", "strike_price": 101, "expiration_date": "2025-01-31"},
        {"ticker": "ITM2", "strike_price": 94, "expiration_date": "2025-01-31"},
        {"ticker": "OTM2", "strike_price": 106, "expiration_date": "2025-01-31"},
    ]

    ranked = _rank_prefetch_contracts(
        contracts,
        spot=100.0,
        trade_date=date(2025, 1, 10),
        default_dte=14,
        option_type="call",
        limit=4,
    )

    assert [contract["ticker"] for contract in ranked[:3]] == ["ATM1", "ITM1", "OTM1"]
    assert len(ranked) == 4


def test_prefetch_prepare_option_data_fetches_broader_contract_bars(monkeypatch):
    import ai_trader.backtest as bt_mod

    call_contracts = [
        {"ticker": "CALLATM", "strike_price": 100, "expiration_date": "2025-01-24"},
        {"ticker": "CALLITM", "strike_price": 95, "expiration_date": "2025-01-24"},
        {"ticker": "CALLOTM", "strike_price": 105, "expiration_date": "2025-01-24"},
    ]
    put_contracts = [
        {"ticker": "PUTATM", "strike_price": 100, "expiration_date": "2025-01-24"},
        {"ticker": "PUTITM", "strike_price": 105, "expiration_date": "2025-01-24"},
        {"ticker": "PUTOTM", "strike_price": 95, "expiration_date": "2025-01-24"},
    ]
    intraday_calls: list[str] = []
    daily_calls: list[str] = []

    monkeypatch.setattr(
        bt_mod,
        "_build_ticker_price_context",
        lambda *args, **kwargs: ("spot=$100.00", 100.0),
    )

    def mock_fetch_contracts(api_key, ticker, option_type, *args, **kwargs):
        return call_contracts if option_type == "call" else put_contracts

    monkeypatch.setattr(bt_mod, "fetch_polygon_option_contracts", mock_fetch_contracts)
    monkeypatch.setattr(
        bt_mod,
        "fetch_historical_intraday_bars",
        lambda api_key, ticker, trading_day, multiplier=5, cache=None: intraday_calls.append(ticker) or [],
    )
    monkeypatch.setattr(
        bt_mod,
        "fetch_option_daily_bar",
        lambda api_key, ticker, trade_date, cache=None: daily_calls.append(ticker) or None,
    )

    count = _prefetch_prepare_option_data(
        "fake",
        ["NVDA"],
        datetime(2025, 1, 10, 9, 35, tzinfo=EASTERN_TZ),
        PolygonCache(),
        default_dte=14,
        contracts_per_side=2,
        max_symbols=1,
    )

    assert count == 4
    assert intraday_calls == ["CALLATM", "CALLITM", "PUTATM", "PUTITM"]
    assert daily_calls == intraday_calls


def test_build_options_context_fetches_intraday_only_for_primary_contracts(monkeypatch):
    import ai_trader.backtest as bt_mod

    call_contracts = [
        {"ticker": "CALLATM", "strike_price": 100, "expiration_date": "2025-01-24"},
        {"ticker": "CALLITM", "strike_price": 95, "expiration_date": "2025-01-24"},
        {"ticker": "CALLOTM", "strike_price": 105, "expiration_date": "2025-01-24"},
    ]
    put_contracts = [
        {"ticker": "PUTATM", "strike_price": 100, "expiration_date": "2025-01-24"},
        {"ticker": "PUTITM", "strike_price": 105, "expiration_date": "2025-01-24"},
        {"ticker": "PUTOTM", "strike_price": 95, "expiration_date": "2025-01-24"},
    ]
    intraday_calls: list[str] = []

    monkeypatch.setattr(
        bt_mod,
        "_build_ticker_price_context",
        lambda *args, **kwargs: ("spot=$100.00", 100.0),
    )
    monkeypatch.setattr(
        bt_mod,
        "fetch_polygon_option_contracts",
        lambda api_key, ticker, option_type, *args, **kwargs: call_contracts if option_type == "call" else put_contracts,
    )
    monkeypatch.setattr(
        bt_mod,
        "_session_intraday_bars_before",
        lambda api_key, ticker, as_of, cache=None, multiplier=5: intraday_calls.append(ticker) or [
            {
                "t": int(datetime(2025, 1, 10, 9, 35, tzinfo=EASTERN_TZ).timestamp() * 1000),
                "c": 1.5,
                "h": 1.6,
                "l": 1.4,
                "v": 10,
            }
        ],
    )
    monkeypatch.setattr(bt_mod.time_module, "sleep", lambda seconds: (_ for _ in ()).throw(AssertionError("unexpected sleep")))

    context = _build_options_context(
        "fake",
        ["NVDA"],
        datetime(2025, 1, 10, 9, 35, tzinfo=EASTERN_TZ),
        PolygonCache(),
        best_events={
            "NVDA": NewsEvent(
                headline="NVDA raises AI guidance",
                summary="Fresh hard catalyst",
                source_count=2,
                article_count=2,
                symbols=["NVDA"],
                event_type="guidance",
                freshness="fresh",
                first_seen=datetime(2025, 1, 10, 8, 45, tzinfo=EASTERN_TZ),
                last_seen=datetime(2025, 1, 10, 9, 10, tzinfo=EASTERN_TZ),
                supporting_sources=["Reuters", "Bloomberg"],
                supporting_headlines=["NVDA raises AI guidance"],
                age_minutes=25,
                catalyst_quality="hard_catalyst",
            )
        },
        closed_trades=[
            {
                "symbol": "CALLATM",
                "underlying": "NVDA",
                "pnl": 1250.0,
                "expression_profile": "time_cushion",
            }
        ],
    )

    assert intraday_calls == ["CALLATM", "PUTATM"]
    assert "Recent expression outcomes: time_cushion 1/1 wins net=$+1,250" in context
    assert "Setup (NVDA): catalyst=hard_catalyst/guidance/fresh/2src" in context
    assert "CALLITM CALL $95.00" in context
    assert "PUTITM PUT $105.00" in context
    assert "premium=$1.50" in context


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


def test_exit_premium_for_position_keeps_live_position_without_bar():
    pos = SimPosition(
        underlying="AAPL",
        option_type="call",
        strike=150.0,
        entry_date=date(2025, 1, 6),
        expiry_date=date(2025, 1, 17),
        entry_premium=5.0,
        qty=1,
        conviction=0.7,
        reasoning="test",
        polygon_ticker="O:AAPL250117C00150000",
    )
    decision_time = datetime(2025, 1, 10, 9, 35, tzinfo=EASTERN_TZ)
    assert _exit_premium_for_position(pos, decision_time, None) is None


def test_exit_premium_for_position_forces_expired_position_to_zero_without_bar():
    pos = SimPosition(
        underlying="RCL",
        option_type="call",
        strike=292.5,
        entry_date=date(2025, 12, 19),
        expiry_date=date(2025, 12, 26),
        entry_premium=7.0,
        qty=12,
        conviction=0.62,
        reasoning="test",
        polygon_ticker="O:RCL251226C00292500",
    )
    decision_time = datetime(2025, 12, 29, 9, 35, tzinfo=EASTERN_TZ)
    assert _exit_premium_for_position(pos, decision_time, None) == 0.0


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


def test_select_real_contract_monthly_prefers_target_dte(monkeypatch):
    contracts = [
        {"ticker": "O:AAPL250124C00150000", "strike_price": 150, "expiration_date": "2025-01-24"},
        {"ticker": "O:AAPL250207C00150000", "strike_price": 150, "expiration_date": "2025-02-07"},
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
        strike_preference="atm",
        expiry_preference="monthly",
        default_dte=30,
    )
    assert result is not None
    assert result["ticker"] == "O:AAPL250207C00150000"


def test_select_real_contract_exact_symbol(monkeypatch):
    contracts = [
        {"ticker": "O:AAPL250124C00150000", "strike_price": 150, "expiration_date": "2025-01-24"},
        {"ticker": "O:AAPL250207C00155000", "strike_price": 155, "expiration_date": "2025-02-07"},
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
        strike_preference="atm",
        expiry_preference="next_week",
        default_dte=14,
        contract_symbol="O:AAPL250207C00155000",
    )
    assert result is not None
    assert result["ticker"] == "O:AAPL250207C00155000"


def test_select_real_contract_respects_target_dte_range(monkeypatch):
    contracts = [
        {"ticker": "O:AAPL250117C00150000", "strike_price": 150, "expiration_date": "2025-01-17"},
        {"ticker": "O:AAPL250131C00150000", "strike_price": 150, "expiration_date": "2025-01-31"},
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
        strike_preference="atm",
        expiry_preference="next_week",
        default_dte=14,
        target_dte_range=(18, 25),
    )
    assert result is not None
    assert result["ticker"] == "O:AAPL250131C00150000"


def test_select_real_contract_respects_expression_profile(monkeypatch):
    contracts = [
        {"ticker": "O:AAPL250117C00150000", "strike_price": 150, "expiration_date": "2025-01-17"},
        {"ticker": "O:AAPL250131C00145000", "strike_price": 145, "expiration_date": "2025-01-31"},
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
        strike_preference="atm",
        expiry_preference="next_week",
        default_dte=14,
        expression_profile="stock_proxy",
    )
    assert result is not None
    assert result["ticker"] == "O:AAPL250131C00145000"


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


def test_performance_summary_includes_conviction_and_dte_reviews():
    trades = [
        {
            "underlying": "AAPL",
            "symbol": "AAPL250116C00150000",
            "entry_date": "2025-01-06",
            "conviction": 0.70,
            "pnl": -500,
        },
        {
            "underlying": "MSFT",
            "symbol": "MSFT250221C00420000",
            "entry_date": "2025-01-06",
            "conviction": 0.60,
            "pnl": 400,
        },
    ]
    result = _build_performance_summary(trades, 99_900, 100_000)
    assert "High-conviction review (>=0.65): 0/1 wins net=$-500" in result
    assert "Fast-decay review (<=10 DTE at entry): 0/1 wins net=$-500" in result
    assert "More-time review (>10 DTE at entry): 1/1 wins net=$+400" in result
    assert "Expression review: Calls 1/2 wins net=$-100" in result
    assert "Short-dated calls (<=14 DTE): 0/1 wins net=$-500" in result


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

    def mock_bars_range(api_key, symbol, start, end, cache=None):
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

    def mock_bars(api_key, symbol, start, end, cache=None):
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
    assert "range=" in ctx
    assert "near_10d_high" in ctx


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
    assert "range=" in ctx


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
    """Tickers with 2+ losses and 0 wins should appear in WARNING section."""
    trades = [
        {"underlying": "NVDA", "pnl": -5000, "entry_date": "2025-01-02", "timestamp": "2025-01-04"},
        {"underlying": "NVDA", "pnl": -6000, "entry_date": "2025-01-06", "timestamp": "2025-01-08"},
        {"underlying": "NVDA", "pnl": -4000, "entry_date": "2025-01-10", "timestamp": "2025-01-13"},
        {"underlying": "AAPL", "pnl": 2000, "entry_date": "2025-01-03", "timestamp": "2025-01-05"},
    ]
    result = _build_performance_summary(trades, 87_000, 100_000)
    assert "WARNING" in result
    assert "REPEAT LOSERS" in result
    assert "NVDA: 3 trades, 0 wins" in result
    assert "$-15,000 total lost" in result
    assert "avg hold:" in result
    assert "avg loss:" in result
    # AAPL should NOT appear in repeat losers (it has a win)
    assert "AAPL" not in result.split("REPEAT LOSERS")[1].split("Total trades")[0]


def test_performance_summary_repeat_losers_before_stats():
    """Repeat losers section should appear before the overall stats."""
    trades = [
        {"underlying": "NVDA", "pnl": -5000, "entry_date": "2025-01-02", "timestamp": "2025-01-04"},
        {"underlying": "NVDA", "pnl": -6000, "entry_date": "2025-01-06", "timestamp": "2025-01-08"},
    ]
    result = _build_performance_summary(trades, 89_000, 100_000)
    warning_pos = result.index("WARNING")
    stats_pos = result.index("Total trades")
    assert warning_pos < stats_pos


def test_performance_summary_no_repeat_losers():
    """No WARNING section when all tickers have at least 1 win."""
    trades = [
        {"underlying": "NVDA", "pnl": 500, "timestamp": "2025-01-05"},
        {"underlying": "NVDA", "pnl": -300, "timestamp": "2025-01-08"},
        {"underlying": "AAPL", "pnl": 200, "timestamp": "2025-01-06"},
        {"underlying": "AAPL", "pnl": -100, "timestamp": "2025-01-09"},
    ]
    result = _build_performance_summary(trades, 100_300, 100_000)
    assert "REPEAT LOSERS" not in result


def test_performance_summary_repeat_losers_hold_days():
    """Avg hold days should be computed from entry_date and timestamp."""
    trades = [
        {"underlying": "META", "pnl": -1000, "entry_date": "2025-01-02", "timestamp": "2025-01-04"},  # 2 days
        {"underlying": "META", "pnl": -2000, "entry_date": "2025-01-06", "timestamp": "2025-01-10"},  # 4 days
    ]
    result = _build_performance_summary(trades, 97_000, 100_000)
    assert "REPEAT LOSERS" in result
    assert "META: 2 trades, 0 wins" in result
    assert "$-3,000 total lost" in result
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


# ---------------------------------------------------------------------------
# SimTrade reasoning
# ---------------------------------------------------------------------------

def test_sim_trade_reasoning():
    """reasoning field is preserved on SimTrade."""
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
        reasoning="Strong earnings beat expected",
    )
    assert trade.reasoning == "Strong earnings beat expected"


def test_sim_trade_reasoning_default():
    """reasoning defaults to empty string for backwards compat."""
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
    )
    assert trade.reasoning == ""


# ---------------------------------------------------------------------------
# BacktestResult decision_log
# ---------------------------------------------------------------------------

def test_backtest_result_decision_log():
    """decision_log field exists and serializes."""
    r = BacktestResult()
    assert r.decision_log == []

    entry = {
        "date": "2025-01-10",
        "market_analysis": "Broad selloff continues",
        "thesis_updates": [
            {"id": "thesis-1", "underlying": "NVDA", "direction": "bullish",
             "status": "developing", "thesis": "AI capex", "conviction": 0.6,
             "new_observation": "earnings beat"}
        ],
        "trades_proposed": [
            {"underlying": "AAPL", "action": "buy_call", "conviction": 0.75,
             "risk_pct": 0.10, "reasoning": "test", "status": "executed",
             "skip_reason": "", "contract": "O:AAPL250117C00150000",
             "qty": 2, "premium": 5.0}
        ],
        "trades_executed": 1,
        "trades_skipped": 0,
        "equity": 95000.0,
        "open_positions": 2,
    }
    r.decision_log.append(entry)
    assert len(r.decision_log) == 1
    assert r.decision_log[0]["date"] == "2025-01-10"
    assert r.decision_log[0]["trades_executed"] == 1

    # Verify it's JSON-serializable
    import json
    serialized = json.dumps(r.decision_log)
    assert "Broad selloff continues" in serialized


# ---------------------------------------------------------------------------
# save_debug_log
# ---------------------------------------------------------------------------

def test_save_debug_log(tmp_path):
    """save_debug_log writes a readable markdown file."""
    r = BacktestResult(
        initial_equity=100_000,
        final_equity=95_000,
        total_return_pct=-0.05,
        total_trades=1,
        wins=0,
        losses=1,
        days_tested=5,
        decision_log=[
            {
                "date": "2025-01-10",
                "market_analysis": "Market is flat.",
                "thesis_updates": [
                    {"id": None, "underlying": "NVDA", "direction": "bullish",
                     "thesis": "AI hype", "conviction": 0.6, "status": "developing",
                     "new_observation": "new chip launch"}
                ],
                "trades_proposed": [
                    {"underlying": "NVDA", "action": "buy_call", "conviction": 0.8,
                     "risk_pct": 0.15, "reasoning": "AI narrative strong",
                     "status": "executed", "skip_reason": "",
                     "contract": "O:NVDA250117C00144000", "qty": 10, "premium": 3.49},
                    {"underlying": "AAPL", "action": "buy_put", "conviction": 0.5,
                     "risk_pct": 0.05, "reasoning": "weak guidance",
                     "status": "skipped", "skip_reason": "no contract found",
                     "contract": "", "qty": 0, "premium": 0.0},
                ],
                "trades_executed": 1,
                "trades_skipped": 1,
                "equity": 95000.0,
                "open_positions": 1,
            }
        ],
    )
    out_path = tmp_path / "test.debug.md"
    save_debug_log(r, out_path)
    assert out_path.exists()
    content = out_path.read_text()
    assert "# Backtest Debug Log" in content
    assert "2025-01-10" in content
    assert "Market is flat." in content
    assert "NVDA" in content
    assert "Executed" in content
    assert "SKIPPED" in content
    assert "no contract found" in content
    assert "$95,000" in content


# ---------------------------------------------------------------------------
# close_position support
# ---------------------------------------------------------------------------

def _make_position(underlying="AAPL", polygon_ticker="O:AAPL250117C00150000",
                   option_type="call", strike=150.0, entry_premium=5.0,
                   qty=2, conviction=0.85):
    """Helper: build a SimPosition for close_position tests."""
    return SimPosition(
        underlying=underlying,
        option_type=option_type,
        strike=strike,
        entry_date=date(2025, 1, 6),
        expiry_date=date(2025, 1, 17),
        entry_premium=entry_premium,
        qty=qty,
        conviction=conviction,
        reasoning="test position",
        polygon_ticker=polygon_ticker,
    )


def test_close_position_by_target_symbol(monkeypatch):
    """close_position with target_symbol matches by exact polygon_ticker."""
    import ai_trader.backtest as bt_mod

    pos = _make_position(polygon_ticker="O:AAPL250117C00150000")
    positions = [pos]

    # Simulate the matching logic from close_position block
    target = "O:AAPL250117C00150000"
    matched_pos = next(
        (p for p in positions if p.polygon_ticker == target), None
    )
    assert matched_pos is not None
    assert matched_pos.polygon_ticker == "O:AAPL250117C00150000"
    assert matched_pos.underlying == "AAPL"


def test_close_position_by_underlying():
    """close_position without target_symbol falls back to underlying match."""
    pos_aapl = _make_position(underlying="AAPL", polygon_ticker="O:AAPL250117C00150000")
    pos_tsla = _make_position(underlying="TSLA", polygon_ticker="O:TSLA250117P00200000",
                              option_type="put", strike=200.0)
    positions = [pos_aapl, pos_tsla]

    # No target_symbol, match by underlying
    target = None
    matched_pos = None
    if target:
        matched_pos = next(
            (p for p in positions if p.polygon_ticker == target), None
        )
    if matched_pos is None:
        matched_pos = next(
            (p for p in positions
             if p.underlying.upper() == "TSLA"),
            None,
        )
    assert matched_pos is not None
    assert matched_pos.underlying == "TSLA"
    assert matched_pos.polygon_ticker == "O:TSLA250117P00200000"


def test_close_position_no_match():
    """close_position with no matching position skips with reason."""
    positions = [_make_position(underlying="AAPL")]

    # Try to close NVDA — no match
    target = None
    matched_pos = None
    if target:
        matched_pos = next(
            (p for p in positions if p.polygon_ticker == target), None
        )
    if matched_pos is None:
        matched_pos = next(
            (p for p in positions
             if p.underlying.upper() == "NVDA"),
            None,
        )
    assert matched_pos is None  # no match found


def test_close_position_pnl():
    """P&L calculation for close_position: (exit - entry) * qty * 100."""
    pos = _make_position(entry_premium=5.0, qty=10)
    exit_premium = 7.5

    trade_pnl = (exit_premium - pos.entry_premium) * pos.qty * 100
    assert trade_pnl == 2500.0  # (7.5 - 5.0) * 10 * 100

    # Negative P&L case
    exit_premium_loss = 3.0
    trade_pnl_loss = (exit_premium_loss - pos.entry_premium) * pos.qty * 100
    assert trade_pnl_loss == -2000.0  # (3.0 - 5.0) * 10 * 100


def test_close_position_removes_from_positions():
    """After close_position, the position is removed from the list."""
    pos_aapl = _make_position(underlying="AAPL", polygon_ticker="O:AAPL250117C00150000")
    pos_tsla = _make_position(underlying="TSLA", polygon_ticker="O:TSLA250117P00200000",
                              option_type="put", strike=200.0)
    positions = [pos_aapl, pos_tsla]
    assert len(positions) == 2

    # Close AAPL
    matched_pos = pos_aapl
    positions.remove(matched_pos)
    assert len(positions) == 1
    assert positions[0].underlying == "TSLA"


def test_close_position_sim_trade_exit_reason():
    """SimTrade created for close_position has exit_reason='manual_close'."""
    pos = _make_position()
    exit_premium = 7.5
    trade_pnl = (exit_premium - pos.entry_premium) * pos.qty * 100
    trade_date = date(2025, 1, 10)

    sim_trade = SimTrade(
        entry_date=pos.entry_date,
        exit_date=trade_date,
        underlying=pos.underlying,
        option_type=pos.option_type,
        strike=pos.strike,
        entry_premium=pos.entry_premium,
        exit_premium=exit_premium,
        qty=pos.qty,
        pnl=trade_pnl,
        exit_reason="manual_close",
        conviction=pos.conviction,
        polygon_ticker=pos.polygon_ticker,
        reasoning=pos.reasoning,
    )
    assert sim_trade.exit_reason == "manual_close"
    assert sim_trade.exit_premium == 7.5
    assert sim_trade.pnl == 500.0
    assert sim_trade.polygon_ticker == "O:AAPL250117C00150000"


# ---------------------------------------------------------------------------
# _build_performance_summary — equity momentum
# ---------------------------------------------------------------------------

def test_performance_summary_equity_momentum_positive():
    """Winning streak shows correct P&L sum and W/L count."""
    trades = [
        {"underlying": "AAPL", "pnl": 1000, "timestamp": "2025-01-02"},
        {"underlying": "MSFT", "pnl": -500, "timestamp": "2025-01-03"},
        {"underlying": "NVDA", "pnl": 2000, "timestamp": "2025-01-04"},
        {"underlying": "TSLA", "pnl": 1500, "timestamp": "2025-01-05"},
        {"underlying": "AMZN", "pnl": -800, "timestamp": "2025-01-06"},
    ]
    result = _build_performance_summary(trades, 103_200, 100_000)
    assert "Equity momentum" in result
    assert "$+3,200" in result
    assert "3W/2L" in result


def test_performance_summary_equity_momentum_negative():
    """Losing streak shows correct negative P&L sum and W/L count."""
    trades = [
        {"underlying": "AAPL", "pnl": 500, "timestamp": "2025-01-02"},
        {"underlying": "MSFT", "pnl": -3000, "timestamp": "2025-01-03"},
        {"underlying": "NVDA", "pnl": -2000, "timestamp": "2025-01-04"},
        {"underlying": "TSLA", "pnl": -1500, "timestamp": "2025-01-05"},
        {"underlying": "AMZN", "pnl": -2400, "timestamp": "2025-01-06"},
    ]
    result = _build_performance_summary(trades, 91_600, 100_000)
    assert "Equity momentum" in result
    assert "$-8,400" in result
    assert "1W/4L" in result


def test_performance_summary_equity_momentum_few_trades():
    """Works correctly with fewer than 5 trades."""
    trades = [
        {"underlying": "AAPL", "pnl": 1000, "timestamp": "2025-01-02"},
        {"underlying": "MSFT", "pnl": 500, "timestamp": "2025-01-03"},
    ]
    result = _build_performance_summary(trades, 101_500, 100_000)
    assert "Equity momentum (last 2 trades)" in result
    assert "$+1,500" in result
    assert "2W/0L" in result


def test_performance_summary_equity_momentum_position():
    """Equity momentum should appear after repeat losers and before total trades."""
    trades = [
        {"underlying": "NVDA", "pnl": -5000, "entry_date": "2025-01-02", "timestamp": "2025-01-04"},
        {"underlying": "NVDA", "pnl": -6000, "entry_date": "2025-01-06", "timestamp": "2025-01-08"},
    ]
    result = _build_performance_summary(trades, 89_000, 100_000)
    warning_pos = result.index("WARNING")
    momentum_pos = result.index("Equity momentum")
    stats_pos = result.index("Total trades")
    assert warning_pos < momentum_pos < stats_pos


# ---------------------------------------------------------------------------
# _build_enriched_portfolio_context — time decay
# ---------------------------------------------------------------------------

def test_enriched_portfolio_time_decay():
    """Position with known current premium and DTE>0 should show time decay."""
    cache = PolygonCache()
    ticker = "O:AAPL250117C00150000"
    cache.option_bars[ticker] = {
        "2025-01-10": {"c": 3.0, "vw": 3.0, "v": 100},
    }
    pos = SimPosition(
        underlying="AAPL", option_type="call", strike=150.0,
        entry_date=date(2025, 1, 6), expiry_date=date(2025, 1, 15),
        entry_premium=5.0, qty=2, conviction=0.85, reasoning="test",
        polygon_ticker=ticker,
    )
    # trade_date=2025-01-10, expiry=2025-01-15 → DTE=5
    # daily_decay = 3.0/5 = 0.6, total = 0.6 * 2 * 100 = $120/day
    result = _build_enriched_portfolio_context(
        equity=100_000, initial_equity=100_000,
        positions=[pos], trade_date=date(2025, 1, 10),
        api_key="fake", cache=cache,
        profit_target_pct=0.50, stop_loss_pct=0.40, time_stop_dte=2,
    )
    assert "time decay" in result
    assert "$120/day" in result


def test_enriched_portfolio_time_decay_dte_zero():
    """No time decay line when DTE=0 (expiring today)."""
    cache = PolygonCache()
    ticker = "O:AAPL250110C00150000"
    cache.option_bars[ticker] = {
        "2025-01-10": {"c": 2.0, "vw": 2.0, "v": 100},
    }
    pos = SimPosition(
        underlying="AAPL", option_type="call", strike=150.0,
        entry_date=date(2025, 1, 6), expiry_date=date(2025, 1, 10),
        entry_premium=5.0, qty=2, conviction=0.85, reasoning="test",
        polygon_ticker=ticker,
    )
    result = _build_enriched_portfolio_context(
        equity=100_000, initial_equity=100_000,
        positions=[pos], trade_date=date(2025, 1, 10),
        api_key="fake", cache=cache,
        profit_target_pct=0.50, stop_loss_pct=0.40, time_stop_dte=2,
    )
    assert "time decay" not in result


def test_save_debug_log_close_position(tmp_path):
    """save_debug_log renders close_position trades with 'Closed:' line."""
    r = BacktestResult(
        initial_equity=100_000,
        final_equity=100_500,
        total_return_pct=0.005,
        total_trades=1,
        wins=1,
        losses=0,
        days_tested=3,
        decision_log=[
            {
                "date": "2025-01-10",
                "market_analysis": "Market steady.",
                "thesis_updates": [],
                "trades_proposed": [
                    {
                        "underlying": "AAPL",
                        "action": "close_position",
                        "conviction": 0.80,
                        "risk_pct": 0.0,
                        "reasoning": "Locking in profits",
                        "status": "executed",
                        "skip_reason": "",
                        "contract": "O:AAPL250117C00150000",
                        "qty": 2,
                        "premium": 7.5,
                        "pnl": 500.0,
                    },
                ],
                "trades_executed": 1,
                "trades_skipped": 0,
                "equity": 100500.0,
                "open_positions": 0,
            }
        ],
    )
    out_path = tmp_path / "test_close.debug.md"
    save_debug_log(r, out_path)
    content = out_path.read_text()
    assert "CLOSE POSITION" in content
    assert "Closed:" in content
    assert "exit=$7.50" in content
    assert "pnl=$+500" in content
    # Should NOT say "Executed:" for close trades
    assert "Executed:" not in content


def test_run_backtest_streams_trades_and_decisions_to_log_db(tmp_path, monkeypatch):
    import ai_trader.backtest as bt_mod

    trade_day = date(2025, 1, 6)
    decision_time = datetime(2025, 1, 6, 9, 35, tzinfo=EASTERN_TZ)
    log_db = tmp_path / "window.db"

    class FakeBrain:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, **kwargs):
            packet = LLMDecisionPacket(
                provider="openai",
                model="gpt-5.4",
                system_prompt="system",
                user_message="prompt",
                tool={"name": "submit_trade_decisions", "input_schema": {"type": "object"}},
                max_tokens=256,
                temperature=0.1,
                contexts=kwargs,
            )
            completion = LLMCompletion(
                provider="openai",
                model="gpt-5.4",
                tool_calls=[
                    ToolCall(
                        name="submit_trade_decisions",
                        input={
                            "market_analysis": "Constructive tape",
                            "thesis_updates": [],
                            "trades": [{"action": "buy_call", "underlying": "AAPL"}],
                        },
                    )
                ],
                raw_response={"id": "resp_test"},
            )
            analysis = MarketAnalysis(
                analysis="Constructive tape",
                thesis_updates=[],
                trades=[
                    TradeDecision(
                        action="buy_call",
                        underlying="AAPL",
                        strike_preference="atm",
                        expiry_preference="next_week",
                        conviction=0.7,
                        risk_pct=0.001,
                        reasoning="Fresh catalyst",
                        target_symbol=None,
                        expression_profile="balanced",
                    )
                ],
            )
            return AnalysisRun(packet=packet, completion=completion, analysis=analysis)

    monkeypatch.setenv("POLYGON_API_KEY", "test-polygon")
    monkeypatch.setattr(bt_mod, "TradingBrain", FakeBrain)
    monkeypatch.setattr(bt_mod, "infer_provider", lambda model=None, provider=None: "openai")
    monkeypatch.setattr(bt_mod, "resolve_api_key", lambda provider, api_key=None: "test-openai")
    monkeypatch.setattr(bt_mod, "_trading_days", lambda start, end: [trade_day])
    monkeypatch.setattr(bt_mod, "_decision_timestamps_for_day", lambda *args, **kwargs: [decision_time])
    monkeypatch.setattr(bt_mod, "_mark_to_market_equity", lambda equity, *args, **kwargs: (equity, 0.0))
    monkeypatch.setattr(bt_mod, "fetch_historical_news_window", lambda *args, **kwargs: [])
    monkeypatch.setattr(bt_mod, "_filter_news_quality", lambda news: news)
    monkeypatch.setattr(bt_mod, "_news_items_from_backtest_articles", lambda *args, **kwargs: [])
    monkeypatch.setattr(bt_mod, "build_news_events", lambda *args, **kwargs: [])
    monkeypatch.setattr(bt_mod, "_build_focus_tickers", lambda *args, **kwargs: ("", ["AAPL"]))
    monkeypatch.setattr(bt_mod, "_format_news_for_backtest", lambda *args, **kwargs: "")
    monkeypatch.setattr(bt_mod, "_build_catalyst_reaction_context", lambda *args, **kwargs: "")
    monkeypatch.setattr(bt_mod, "_build_market_trend_context", lambda *args, **kwargs: "market")
    monkeypatch.setattr(bt_mod, "_build_enriched_portfolio_context", lambda *args, **kwargs: "portfolio")
    monkeypatch.setattr(bt_mod, "_build_performance_summary", lambda *args, **kwargs: "")
    monkeypatch.setattr(bt_mod, "_build_options_context", lambda *args, **kwargs: "options")
    monkeypatch.setattr(bt_mod, "_current_underlying_price", lambda *args, **kwargs: 100.0)
    monkeypatch.setattr(
        bt_mod,
        "_select_real_contract",
        lambda *args, **kwargs: {
            "ticker": "O:AAPL250117C00100000",
            "strike_price": 100.0,
            "expiration_date": "2025-01-17",
        },
    )
    monkeypatch.setattr(bt_mod, "_next_fill_option_bar", lambda *args, **kwargs: {"c": 1.0, "v": 100})
    monkeypatch.setattr(bt_mod, "_current_option_bar", lambda *args, **kwargs: {"c": 1.5, "v": 100})

    result = bt_mod.run_backtest(
        BacktestConfig(
            start_date=trade_day,
            end_date=trade_day,
            llm_delay_seconds=0.0,
            cache_db_path=tmp_path / "cache.db",
            log_db_path=log_db,
        )
    )

    assert result.log_db_path == str(log_db)

    with sqlite3.connect(log_db) as conn:
        decision = conn.execute(
            "SELECT llm_provider, llm_model, trades_executed FROM ai_decisions"
        ).fetchone()
        trade = conn.execute(
            "SELECT symbol, underlying, action, status, expression_profile FROM ai_trades"
        ).fetchone()
        close = conn.execute(
            "SELECT symbol, underlying, pnl, reason FROM position_closes"
        ).fetchone()

    assert decision == ("openai", "gpt-5.4", 1)
    assert trade == ("O:AAPL250117C00100000", "AAPL", "buy_call", "filled", "balanced")
    assert close == ("O:AAPL250117C00100000", "AAPL", 50.0, "backtest_end")
