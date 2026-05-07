"""Historical backtester for the AI options trader.

Replays historical news and market data through the LLM brain,
uses real historical options market data for pricing, and tracks P&L.

Run:  python -m ai_trader.backtest --start 2025-01-02 --end 2025-01-31

Requires: provider API key (for the configured LLM) and POLYGON_API_KEY
(paid tier for options data)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time as time_module
from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import date, datetime, time, timedelta, timezone
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

from . import config
from .brain import MarketAnalysis, TradingBrain, TradeDecision
from .candidates import (
    build_candidate_ideas,
    format_candidate_table,
    select_candidate_finalists,
)
from .db import (
    AIDecisionRecord,
    AITradeLogger,
    AITradeRecord,
    PositionCloseRecord,
    expression_guidance_lines,
    format_trade_history,
    profile_calibration_lines,
)
from .historical_cache import (
    DEFAULT_HISTORICAL_CACHE_DB_PATH,
    PolygonResponseStore,
)
from .journal import ThesisJournal
from .llm import api_key_env_name, infer_provider, resolve_api_key
from .news import (
    NewsItem,
    build_news_events,
    classify_catalyst_reaction,
    expand_symbols_with_relationships,
    format_symbol_setup_context,
    format_news_for_llm,
    map_best_events_by_symbol,
    rank_symbols_from_events,
    score_news_event,
)
from .options import (
    OptionContract,
    approx_delta,
    filter_contracts_by_expiry_preference,
    resolve_expression_profile,
    select_contract,
)
from .risk import (
    evaluate_trade_risk,
    evaluate_stock_trade_risk,
    evaluate_position_risk,
    scale_risk_pct_for_conviction,
    size_for_risk_budget,
    stop_loss_for_dte,
)
from .utils import (
    EASTERN_TZ,
    is_equity_candidate_symbol,
    log,
    now_eastern,
    prioritized_symbol_watchlist,
)


# ---------------------------------------------------------------------------
# Historical data fetching (Polygon — equities)
# ---------------------------------------------------------------------------

_LAST_POLYGON_REQUEST_AT = 0.0
_POLYGON_REQUEST_INTERVAL_SECONDS = max(
    float(config.POLYGON_MIN_REQUEST_INTERVAL_SECONDS),
    0.0,
)
_THETA_TICKER_PREFIX = "THETA:"


def _base_polygon_request_interval() -> float:
    return max(float(config.POLYGON_MIN_REQUEST_INTERVAL_SECONDS), 0.0)


def _max_polygon_request_interval() -> float:
    return max(
        float(getattr(config, "POLYGON_MAX_REQUEST_INTERVAL_SECONDS", _base_polygon_request_interval())),
        _base_polygon_request_interval(),
    )


def _retry_after_seconds(headers: object) -> float | None:
    if not isinstance(headers, dict):
        return None
    raw_value = headers.get("Retry-After")
    if raw_value is None:
        return None
    try:
        retry_after = float(str(raw_value).strip())
    except ValueError:
        return None
    return retry_after if retry_after > 0 else None


def _increase_polygon_request_interval(*, retry_after: float | None = None) -> float:
    global _POLYGON_REQUEST_INTERVAL_SECONDS
    base_interval = _base_polygon_request_interval()
    current_interval = max(_POLYGON_REQUEST_INTERVAL_SECONDS, base_interval)
    max_interval = _max_polygon_request_interval()
    if retry_after is not None:
        next_interval = max(current_interval, min(retry_after, max_interval))
    else:
        next_interval = min(max_interval, max(current_interval * 1.5, base_interval + 1.0))
    _POLYGON_REQUEST_INTERVAL_SECONDS = next_interval
    return next_interval


def _decay_polygon_request_interval() -> None:
    global _POLYGON_REQUEST_INTERVAL_SECONDS
    base_interval = _base_polygon_request_interval()
    current_interval = max(_POLYGON_REQUEST_INTERVAL_SECONDS, base_interval)
    if current_interval <= base_interval:
        _POLYGON_REQUEST_INTERVAL_SECONDS = base_interval
        return
    _POLYGON_REQUEST_INTERVAL_SECONDS = max(base_interval, current_interval * 0.95)


def _respect_polygon_rate_limit() -> None:
    global _LAST_POLYGON_REQUEST_AT
    min_interval = max(_POLYGON_REQUEST_INTERVAL_SECONDS, _base_polygon_request_interval())
    if min_interval <= 0:
        return
    now = time_module.monotonic()
    elapsed = now - _LAST_POLYGON_REQUEST_AT
    if _LAST_POLYGON_REQUEST_AT > 0 and elapsed < min_interval:
        time_module.sleep(min_interval - elapsed)
    _LAST_POLYGON_REQUEST_AT = time_module.monotonic()

def _polygon_request(
    api_key: str,
    path: str,
    params: dict | None = None,
    *,
    store: PolygonResponseStore | None = None,
    offline: bool = False,
) -> dict:
    import requests

    params = params or {}
    cached = store.get(path, params) if store is not None else None
    if cached is not None:
        return cached
    if offline:
        raise RuntimeError(f"offline Polygon cache miss for {path}")

    request_params = dict(params)
    request_params["apiKey"] = api_key
    url = f"https://api.polygon.io{path}"
    max_attempts = max(int(getattr(config, "POLYGON_429_RETRY_ATTEMPTS", 5)), 1)

    response = None
    for attempt in range(max_attempts):
        _respect_polygon_rate_limit()
        response = requests.get(url, params=request_params, timeout=30)
        if response.status_code != 429:
            if response.status_code < 400:
                _decay_polygon_request_interval()
            break
        retry_after = _retry_after_seconds(getattr(response, "headers", None))
        backoff_interval = _increase_polygon_request_interval(retry_after=retry_after)
        sleep_seconds = retry_after if retry_after is not None else backoff_interval
        time_module.sleep(sleep_seconds)
    if response is None:
        raise RuntimeError(f"Polygon request failed without a response for {path}")
    if response.status_code >= 400:
        raise RuntimeError(f"Polygon {response.status_code}: {response.text[:200]}")
    data = response.json()
    if store is not None:
        store.put(path, params, data)
    return data


def _theta_request(
    path: str,
    params: dict | None = None,
    *,
    store: PolygonResponseStore | None = None,
    offline: bool = False,
) -> dict:
    import requests

    request_params = dict(params or {})
    if path.startswith("/v3/"):
        request_params.setdefault("format", "json")
    cache_path = f"theta:{path}"
    cached = store.get(cache_path, request_params) if store is not None else None
    if cached is not None:
        return cached
    if offline:
        raise RuntimeError(f"offline Theta cache miss for {path}")

    base_url = config.resolved_theta_base_url()
    if not base_url:
        raise RuntimeError("THETA_BASE_URL is not configured")
    url = f"{base_url}{path}"
    response = requests.get(url, params=request_params, timeout=30)
    if response.status_code >= 400:
        message = f"Theta {response.status_code}: {response.text[:200]}"
        if _theta_is_expected_empty_message(message):
            data = {
                "header": {"format": []},
                "response": [],
                "theta_empty": True,
                "theta_status": response.status_code,
                "theta_message": response.text[:200],
            }
            if store is not None:
                store.put(cache_path, request_params, data)
            return data
        raise RuntimeError(message)
    data = response.json()
    if store is not None:
        store.put(cache_path, request_params, data)
    return data


def _historical_options_provider() -> str:
    return config.resolved_historical_options_provider()


def _is_theta_option_ticker(ticker: str) -> bool:
    return str(ticker).startswith(_THETA_TICKER_PREFIX)


def _theta_contract_ticker(
    underlying: str,
    expiration: date,
    option_type: str,
    strike: float,
) -> str:
    return (
        f"{_THETA_TICKER_PREFIX}{underlying.upper()}:{expiration.isoformat()}:"
        f"{option_type.lower()}:{strike:.3f}"
    )


def _parse_theta_contract_ticker(option_ticker: str) -> tuple[str, date, str, float]:
    if not _is_theta_option_ticker(option_ticker):
        raise ValueError(f"not a theta option ticker: {option_ticker}")
    parts = option_ticker.split(":")
    if len(parts) != 5:
        raise ValueError(f"invalid theta option ticker: {option_ticker}")
    _, underlying, expiration_raw, option_type, strike_raw = parts
    return (
        underlying,
        date.fromisoformat(expiration_raw),
        option_type.lower(),
        float(strike_raw),
    )


def _theta_date(value: date) -> str:
    return value.strftime("%Y%m%d")


def _theta_strike_param(value: float) -> str:
    return str(int(round(value * 1000)))


def _theta_right_param(option_type: str) -> str:
    normalized = str(option_type).strip().lower()
    if normalized in {"c", "call"}:
        return "C"
    if normalized in {"p", "put"}:
        return "P"
    return normalized.upper()


def _theta_ohlc_interval(multiplier: int) -> str:
    return str(max(int(multiplier), 1) * 60_000)


def _theta_table_rows(data: dict) -> list[dict]:
    response = data.get("response", data.get("results", []))
    if not isinstance(response, list) or not response:
        return []
    if isinstance(response[0], dict):
        return [dict(row) for row in response]
    header = data.get("header", {})
    columns = header.get("format", [])
    if not isinstance(columns, list) or not columns:
        return []
    rows: list[dict] = []
    for raw_row in response:
        if not isinstance(raw_row, list):
            continue
        rows.append(
            {
                str(column): raw_row[index] if index < len(raw_row) else None
                for index, column in enumerate(columns)
            }
        )
    return rows


def _theta_parse_expiration(value: object) -> date | None:
    if isinstance(value, date):
        return value
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        if len(raw) == 8 and raw.isdigit():
            return datetime.strptime(raw, "%Y%m%d").date()
        return date.fromisoformat(raw)
    except ValueError:
        return None


def _theta_parse_strike(value: object) -> float:
    strike = _float_or_zero(value)
    if strike <= 0:
        return 0.0
    return strike / 1000.0 if strike >= 1000 else strike


def _theta_row_timestamp(value: object, *, fallback_date: date) -> int | None:
    if isinstance(value, (int, float)):
        numeric = int(float(value))
        if 0 <= numeric <= 86_400_000:
            return _to_epoch_ms(
                datetime.combine(fallback_date, time(0, 0), tzinfo=EASTERN_TZ)
                + timedelta(milliseconds=numeric)
            )
    if isinstance(value, str) and value.strip():
        normalized = value.strip().replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            parsed = None
        if parsed is not None:
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=EASTERN_TZ)
            return _to_epoch_ms(parsed.astimezone(EASTERN_TZ))
    return _to_epoch_ms(_market_close_dt(fallback_date))


def _theta_bar_to_polygon_bar(row: dict, *, fallback_date: date) -> dict | None:
    trade_date = _theta_parse_expiration(row.get("date")) or fallback_date
    timestamp_ms = _theta_row_timestamp(
        row.get("timestamp")
        or row.get("last_trade")
        or row.get("created")
        or row.get("ms_of_day2")
        or row.get("ms_of_day"),
        fallback_date=trade_date,
    )
    if timestamp_ms is None:
        return None
    bid = float(row.get("bid") or 0.0)
    ask = float(row.get("ask") or 0.0)
    mid_quote = (bid + ask) / 2 if bid > 0 and ask > 0 else max(bid, ask, 0.0)
    close_price = float(row.get("close") or 0.0)
    if close_price <= 0 and mid_quote > 0:
        close_price = mid_quote
    if close_price <= 0:
        return None
    open_price = float(row.get("open") or 0.0)
    high_price = float(row.get("high") or 0.0)
    low_price = float(row.get("low") or 0.0)
    if open_price <= 0:
        open_price = close_price
    if high_price <= 0:
        high_price = max(open_price, close_price)
    if low_price <= 0:
        low_price = min(open_price, close_price)
    bar = {
        "t": timestamp_ms,
        "o": open_price,
        "h": high_price,
        "l": low_price,
        "c": close_price,
        "v": int(float(row.get("volume") or 0)),
        "vw": float(row.get("vwap") or mid_quote or close_price),
    }
    if bid > 0:
        bar["bid"] = bid
    if ask > 0:
        bar["ask"] = ask
    return bar


def _theta_is_expected_empty_error(exc: Exception) -> bool:
    return _theta_is_expected_empty_message(str(exc))


def _theta_is_expected_empty_message(message: str) -> bool:
    if not message.startswith("Theta 472:"):
        return False
    return any(
        phrase in message
        for phrase in (
            "No listed contracts for the date",
            "No data for the specified timeframe & contract",
            "No data for contract",
        )
    )


def _theta_quote_rows_to_bars(
    data: dict,
    *,
    trading_day: date,
    multiplier: int,
) -> list[dict]:
    interval_ms = max(int(multiplier), 1) * 60_000
    session_start_ms = (9 * 60 + 30) * 60_000
    session_end_ms = 16 * 60 * 60_000
    buckets: dict[int, dict[str, float | int]] = {}

    for row in _theta_table_rows(data):
        ms_of_day = _int_or_zero(row.get("ms_of_day"))
        if ms_of_day < session_start_ms or ms_of_day > session_end_ms:
            continue
        bid = _float_or_zero(row.get("bid"))
        ask = _float_or_zero(row.get("ask"))
        price = (bid + ask) / 2 if bid > 0 and ask > 0 else max(bid, ask, 0.0)
        if price <= 0:
            continue
        bucket_ms = session_start_ms + ((ms_of_day - session_start_ms) // interval_ms) * interval_ms
        bucket = buckets.setdefault(
            bucket_ms,
            {
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "bid": bid,
                "ask": ask,
                "mid_total": 0.0,
                "count": 0,
            },
        )
        bucket["high"] = max(float(bucket["high"]), price)
        bucket["low"] = min(float(bucket["low"]), price)
        bucket["close"] = price
        bucket["mid_total"] = float(bucket["mid_total"]) + price
        bucket["count"] = int(bucket["count"]) + 1
        if bid > 0:
            bucket["bid"] = bid
        if ask > 0:
            bucket["ask"] = ask

    bars: list[dict] = []
    for bucket_ms in sorted(buckets):
        bucket = buckets[bucket_ms]
        count = int(bucket["count"])
        if count <= 0:
            continue
        timestamp_ms = _to_epoch_ms(
            datetime.combine(trading_day, time(0, 0), tzinfo=EASTERN_TZ)
            + timedelta(milliseconds=bucket_ms)
        )
        bar = {
            "t": timestamp_ms,
            "o": float(bucket["open"]),
            "h": float(bucket["high"]),
            "l": float(bucket["low"]),
            "c": float(bucket["close"]),
            "v": 0,
            "vw": float(bucket["mid_total"]) / count,
            "quote_count": count,
            "quote_only": True,
        }
        bid = float(bucket["bid"])
        ask = float(bucket["ask"])
        if bid > 0:
            bar["bid"] = bid
        if ask > 0:
            bar["ask"] = ask
        bars.append(bar)
    return bars


def _theta_contracts_from_response(
    underlying: str,
    contract_type: str,
    *,
    expiry_gte: date,
    expiry_lte: date,
    strike_gte: float,
    strike_lte: float,
    data: dict,
) -> list[dict]:
    results: list[dict] = []
    for row in _theta_table_rows(data):
        expiration = _theta_parse_expiration(row.get("expiration"))
        if expiration is None:
            continue
        strike = _theta_parse_strike(row.get("strike"))
        option_type = str(row.get("right") or "").strip().lower()
        if option_type == "c":
            option_type = "call"
        elif option_type == "p":
            option_type = "put"
        if option_type != contract_type:
            continue
        if not (expiry_gte <= expiration <= expiry_lte):
            continue
        if not (strike_gte <= strike <= strike_lte):
            continue
        results.append(
            {
                "ticker": _theta_contract_ticker(underlying, expiration, option_type, strike),
                "strike_price": strike,
                "expiration_date": expiration.isoformat(),
                "contract_type": option_type,
                "bid": _float_or_zero(row.get("bid")),
                "ask": _float_or_zero(row.get("ask")),
                "mid": _float_or_zero(row.get("mid")),
                "open_interest": _int_or_zero(row.get("open_interest")),
                "volume": _int_or_zero(row.get("volume")),
            }
        )
    return results


def _raise_if_offline(cache: PolygonCache | None, exc: Exception) -> None:
    if cache and cache.offline:
        raise RuntimeError(str(exc)) from exc


def _is_offline_cache_miss(exc: Exception) -> bool:
    message = str(exc)
    return "offline " in message and "cache miss" in message


def fetch_historical_news_window(
    api_key: str,
    start_time: datetime,
    end_time: datetime,
    limit: int = 50,
    cache: PolygonCache | None = None,
) -> list[dict]:
    """Fetch Polygon news published within a precise timestamp window."""
    if end_time <= start_time:
        return []

    start = start_time.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    end = end_time.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        data = _polygon_request(
            api_key,
            "/v2/reference/news",
            params={
                "published_utc.gte": start,
                "published_utc.lte": end,
                "limit": str(limit),
                "sort": "published_utc",
                "order": "desc",
            },
            store=cache.store if cache else None,
            offline=cache.offline if cache else False,
        )
    except Exception as exc:
        log(f"news fetch error for {start_time} -> {end_time}: {exc}")
        _raise_if_offline(cache, exc)
        return []
    return data.get("results", [])


def fetch_historical_news(
    api_key: str,
    trade_date: date,
    limit: int = 50,
    cache: PolygonCache | None = None,
) -> list[dict]:
    """Fetch all news for a given date from Polygon."""
    start = datetime.combine(trade_date, time(0, 0, tzinfo=EASTERN_TZ))
    end = datetime.combine(trade_date, time(23, 59, 59, tzinfo=EASTERN_TZ))
    return fetch_historical_news_window(api_key, start, end, limit=limit, cache=cache)


def fetch_historical_daily_bar(
    api_key: str,
    symbol: str,
    trade_date: date,
    cache: PolygonCache | None = None,
) -> dict | None:
    """Fetch OHLCV for an equity symbol on a given date."""
    try:
        data = _polygon_request(
            api_key,
            f"/v1/open-close/{symbol}/{trade_date}",
            store=cache.store if cache else None,
            offline=cache.offline if cache else False,
        )
        if data.get("status") == "OK":
            return data
    except Exception as exc:
        _raise_if_offline(cache, exc)
    return None


def fetch_historical_daily_bars_range(
    api_key: str,
    symbol: str,
    start: date,
    end: date,
    cache: PolygonCache | None = None,
) -> list[dict]:
    """Fetch daily bars for an equity over a date range."""
    cache_key = (symbol.upper(), start.isoformat(), end.isoformat())
    if cache and cache_key in cache.daily_bars:
        return cache.daily_bars[cache_key]
    try:
        data = _polygon_request(
            api_key,
            f"/v2/aggs/ticker/{symbol}/range/1/day"
            f"/{start.isoformat()}/{end.isoformat()}",
            params={"adjusted": "true", "sort": "asc", "limit": "250"},
            store=cache.store if cache else None,
            offline=cache.offline if cache else False,
        )
    except Exception as exc:
        log(f"bars fetch error for {symbol}: {exc}")
        _raise_if_offline(cache, exc)
        return []
    results = data.get("results", [])
    if cache is not None:
        cache.daily_bars[cache_key] = results
    return results


def fetch_historical_intraday_bars(
    api_key: str,
    ticker: str,
    trading_day: date,
    multiplier: int = 5,
    cache: PolygonCache | None = None,
) -> list[dict]:
    """Fetch regular-session intraday aggregate bars for one ticker and day."""
    cache_key = (ticker, trading_day.isoformat(), multiplier)
    if cache and cache_key in cache.intraday_bars:
        return cache.intraday_bars[cache_key]

    if _is_theta_option_ticker(ticker):
        data: dict | None = None
        try:
            underlying, expiration, option_type, strike = _parse_theta_contract_ticker(ticker)
            data = _theta_request(
                "/v2/hist/option/ohlc",
                params={
                    "root": underlying,
                    "exp": _theta_date(expiration),
                    "right": _theta_right_param(option_type),
                    "strike": _theta_strike_param(strike),
                    "start_date": _theta_date(trading_day),
                    "end_date": _theta_date(trading_day),
                    "ivl": _theta_ohlc_interval(multiplier),
                },
                store=cache.store if cache else None,
                offline=cache.offline if cache else False,
            )
        except Exception as exc:
            if not _theta_is_expected_empty_error(exc):
                log(f"theta intraday bars fetch error for {ticker} on {trading_day}: {exc}")
                _raise_if_offline(cache, exc)
                return []

        results = [
            bar
            for row in _theta_table_rows(data or {})
            if (bar := _theta_bar_to_polygon_bar(row, fallback_date=trading_day)) is not None
        ]
        if not results:
            try:
                quote_data = _theta_request(
                    "/v2/hist/option/quote",
                    params={
                        "root": underlying,
                        "exp": _theta_date(expiration),
                        "right": _theta_right_param(option_type),
                        "strike": _theta_strike_param(strike),
                        "start_date": _theta_date(trading_day),
                        "end_date": _theta_date(trading_day),
                        "ivl": _theta_ohlc_interval(multiplier),
                    },
                    store=cache.store if cache else None,
                    offline=cache.offline if cache else False,
                )
            except Exception as exc:
                if not _theta_is_expected_empty_error(exc):
                    log(f"theta intraday quote fallback error for {ticker} on {trading_day}: {exc}")
                    _raise_if_offline(cache, exc)
                quote_data = {}
            results = _theta_quote_rows_to_bars(
                quote_data,
                trading_day=trading_day,
                multiplier=multiplier,
            )
        if cache is not None:
            cache.intraday_bars[cache_key] = results
        return results

    start_ms = _to_epoch_ms(_market_open_dt(trading_day))
    end_ms = _to_epoch_ms(_market_close_dt(trading_day))
    try:
        data = _polygon_request(
            api_key,
            f"/v2/aggs/ticker/{ticker}/range/{multiplier}/minute/{start_ms}/{end_ms}",
            params={"adjusted": "true", "sort": "asc", "limit": "5000"},
            store=cache.store if cache else None,
            offline=cache.offline if cache else False,
        )
    except Exception as exc:
        log(f"intraday bars fetch error for {ticker} on {trading_day}: {exc}")
        _raise_if_offline(cache, exc)
        return []

    results = data.get("results", [])
    if cache is not None:
        cache.intraday_bars[cache_key] = results
    return results


# ---------------------------------------------------------------------------
# Polygon options data
# ---------------------------------------------------------------------------

@dataclass
class PolygonCache:
    """In-memory cache for Polygon lookups and aggregate bars."""
    # (underlying, type, expiry_gte, expiry_lte, strike_gte, strike_lte) -> list[contract_dict]
    contracts: dict[tuple, list[dict]] = field(default_factory=dict)
    # option_ticker -> {date_iso -> bar_dict}
    option_bars: dict[str, dict[str, dict]] = field(default_factory=dict)
    # (ticker, date_iso, multiplier) -> list[agg_bar_dict]
    intraday_bars: dict[tuple[str, str, int], list[dict]] = field(default_factory=dict)
    # (ticker, start_iso, end_iso) -> list[daily_bar_dict]
    daily_bars: dict[tuple[str, str, str], list[dict]] = field(default_factory=dict)
    # (ticker, as_of_iso, multiplier) -> list[agg_bar_dict] filtered to < as_of
    session_bars_before: dict[tuple[str, str, int], list[dict]] = field(default_factory=dict)
    # (ticker, as_of_iso, lookback_days, bar_minutes) -> metrics dict
    ticker_metrics: dict[tuple[str, str, int, int], dict | None] = field(default_factory=dict)
    store: PolygonResponseStore | None = None
    offline: bool = False


def _coerce_eastern_datetime(
    value: date | datetime,
    default_time: time = time(16, 0),
) -> datetime:
    """Normalize a date/datetime to an aware Eastern datetime."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=EASTERN_TZ)
        return value.astimezone(EASTERN_TZ)

    if default_time.tzinfo is None:
        default_time = time(
            default_time.hour,
            default_time.minute,
            default_time.second,
            default_time.microsecond,
            tzinfo=EASTERN_TZ,
        )
    return datetime.combine(value, default_time)


def _market_open_dt(trading_day: date) -> datetime:
    return datetime.combine(trading_day, time(9, 30, tzinfo=EASTERN_TZ))


def _market_close_dt(trading_day: date) -> datetime:
    return datetime.combine(trading_day, time(16, 0, tzinfo=EASTERN_TZ))


def _to_epoch_ms(value: datetime) -> int:
    return int(value.astimezone(timezone.utc).timestamp() * 1000)


def _bar_timestamp_eastern(bar: dict) -> datetime | None:
    ts_ms = bar.get("t")
    if not ts_ms:
        return None
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).astimezone(EASTERN_TZ)


def _latest_bar_before(bars: list[dict], cutoff: datetime) -> dict | None:
    """Return the most recent aggregate bar that started before cutoff."""
    for bar in reversed(bars):
        ts = _bar_timestamp_eastern(bar)
        if ts and ts < cutoff:
            return bar
    return None


def _first_bar_at_or_after(bars: list[dict], cutoff: datetime) -> dict | None:
    """Return the first aggregate bar that starts at or after cutoff."""
    for bar in bars:
        ts = _bar_timestamp_eastern(bar)
        if ts and ts >= cutoff:
            return bar
    return None


def _nth_weekday_of_month(year: int, month: int, weekday: int, occurrence: int) -> date:
    current = date(year, month, 1)
    while current.weekday() != weekday:
        current += timedelta(days=1)
    return current + timedelta(days=7 * (occurrence - 1))


def _last_weekday_of_month(year: int, month: int, weekday: int) -> date:
    if month == 12:
        current = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        current = date(year, month + 1, 1) - timedelta(days=1)
    while current.weekday() != weekday:
        current -= timedelta(days=1)
    return current


def _observed_weekday_holiday(holiday: date) -> date:
    if holiday.weekday() == 5:
        return holiday - timedelta(days=1)
    if holiday.weekday() == 6:
        return holiday + timedelta(days=1)
    return holiday


def _easter_sunday(year: int) -> date:
    """Anonymous Gregorian algorithm."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


@lru_cache(maxsize=None)
def _nyse_holidays(year: int) -> frozenset[date]:
    holidays = {
        _observed_weekday_holiday(date(year, 1, 1)),      # New Year's Day
        _nth_weekday_of_month(year, 1, 0, 3),             # Martin Luther King Jr. Day
        _nth_weekday_of_month(year, 2, 0, 3),             # Presidents Day
        _easter_sunday(year) - timedelta(days=2),         # Good Friday
        _last_weekday_of_month(year, 5, 0),               # Memorial Day
        _observed_weekday_holiday(date(year, 6, 19)),     # Juneteenth
        _observed_weekday_holiday(date(year, 7, 4)),      # Independence Day
        _nth_weekday_of_month(year, 9, 0, 1),             # Labor Day
        _nth_weekday_of_month(year, 11, 3, 4),            # Thanksgiving
        _observed_weekday_holiday(date(year, 12, 25)),    # Christmas
    }
    return frozenset(holidays)


def _is_trading_day(value: date) -> bool:
    return value.weekday() < 5 and value not in _nyse_holidays(value.year)


def _decision_timestamps_for_day(
    trading_day: date,
    interval_minutes: int,
    start_delay_minutes: int,
    end_buffer_minutes: int,
) -> list[datetime]:
    """Generate intraday decision timestamps aligned to the live scan cadence."""
    start = _market_open_dt(trading_day) + timedelta(minutes=start_delay_minutes)
    end = _market_close_dt(trading_day) - timedelta(minutes=end_buffer_minutes)
    timestamps: list[datetime] = []
    current = start
    while current <= end:
        timestamps.append(current)
        current += timedelta(minutes=interval_minutes)
    return timestamps


def fetch_polygon_option_contracts(
    api_key: str,
    underlying: str,
    contract_type: str,
    expiry_gte: date,
    expiry_lte: date,
    strike_gte: float,
    strike_lte: float,
    as_of: date | None = None,
    cache: PolygonCache | None = None,
) -> list[dict]:
    """Query the configured historical provider for available option contracts.

    For backtesting past dates, pass as_of=trade_date so the provider returns
    contracts that existed at that time (even if they've since expired).
    """
    cache_key = (underlying, contract_type, expiry_gte.isoformat(),
                 expiry_lte.isoformat(), strike_gte, strike_lte,
                 as_of.isoformat() if as_of else "")
    if cache and cache_key in cache.contracts:
        return cache.contracts[cache_key]

    try:
        provider = _historical_options_provider()
        if provider == "theta":
            data = _theta_request(
                "/v2/list/contracts/option/quote",
                params={
                    "root": underlying,
                    "start_date": _theta_date(as_of or expiry_gte),
                },
                store=cache.store if cache else None,
                offline=cache.offline if cache else False,
            )
            results = _theta_contracts_from_response(
                underlying,
                contract_type,
                expiry_gte=expiry_gte,
                expiry_lte=expiry_lte,
                strike_gte=strike_gte,
                strike_lte=strike_lte,
                data=data,
            )
        else:
            params = {
                "underlying_ticker": underlying,
                "contract_type": contract_type,
                "expiration_date.gte": expiry_gte.isoformat(),
                "expiration_date.lte": expiry_lte.isoformat(),
                "strike_price.gte": str(strike_gte),
                "strike_price.lte": str(strike_lte),
                "limit": "100",
                "sort": "strike_price",
                "order": "asc",
            }
            if as_of:
                params["as_of"] = as_of.isoformat()
            data = _polygon_request(
                api_key,
                "/v3/reference/options/contracts",
                params=params,
                store=cache.store if cache else None,
                offline=cache.offline if cache else False,
            )
            results = data.get("results", [])
    except Exception as exc:
        if not (provider == "theta" and _theta_is_expected_empty_error(exc)):
            log(f"option contracts fetch error for {underlying}: {exc}")
        _raise_if_offline(cache, exc)
        return []

    if cache:
        cache.contracts[cache_key] = results
    return results


def fetch_option_daily_bar(
    api_key: str,
    option_ticker: str,
    trade_date: date,
    cache: PolygonCache | None = None,
) -> dict | None:
    """Get a single-day OHLCV bar for an option contract. Checks cache first."""
    date_key = trade_date.isoformat()
    if cache and option_ticker in cache.option_bars:
        bar = cache.option_bars[option_ticker].get(date_key)
        if bar is not None:
            return bar

    if _is_theta_option_ticker(option_ticker):
        try:
            underlying, expiration, option_type, strike = _parse_theta_contract_ticker(option_ticker)
            data = _theta_request(
                "/v2/hist/option/eod",
                params={
                    "root": underlying,
                    "exp": _theta_date(expiration),
                    "right": _theta_right_param(option_type),
                    "strike": _theta_strike_param(strike),
                    "start_date": _theta_date(trade_date),
                    "end_date": _theta_date(trade_date),
                },
                store=cache.store if cache else None,
                offline=cache.offline if cache else False,
            )
        except Exception as exc:
            log(f"theta option bar fetch error for {option_ticker} on {trade_date}: {exc}")
            _raise_if_offline(cache, exc)
            return None

        results = _theta_table_rows(data)
        if not results:
            return None
        bar = _theta_bar_to_polygon_bar(results[0], fallback_date=trade_date)
        if bar is None:
            return None
        if cache:
            cache.option_bars.setdefault(option_ticker, {})[date_key] = bar
        return bar

    # Fetch just this one day
    try:
        data = _polygon_request(
            api_key,
            f"/v2/aggs/ticker/{option_ticker}/range/1/day"
            f"/{date_key}/{date_key}",
            params={"adjusted": "true", "sort": "asc", "limit": "1"},
            store=cache.store if cache else None,
            offline=cache.offline if cache else False,
        )
    except Exception as exc:
        log(f"option bar fetch error for {option_ticker} on {trade_date}: {exc}")
        _raise_if_offline(cache, exc)
        return None

    results = data.get("results", [])
    if not results:
        return None

    bar = results[0]
    if cache:
        cache.option_bars.setdefault(option_ticker, {})[date_key] = bar
    return bar


def fetch_option_daily_bars_range(
    api_key: str,
    option_ticker: str,
    start: date,
    end: date,
    cache: PolygonCache | None = None,
) -> list[dict]:
    """Bulk fetch daily bars for an option contract and populate cache."""
    if _is_theta_option_ticker(option_ticker):
        try:
            underlying, expiration, option_type, strike = _parse_theta_contract_ticker(option_ticker)
            data = _theta_request(
                "/v2/hist/option/eod",
                params={
                    "root": underlying,
                    "exp": _theta_date(expiration),
                    "right": _theta_right_param(option_type),
                    "strike": _theta_strike_param(strike),
                    "start_date": _theta_date(start),
                    "end_date": _theta_date(end),
                },
                store=cache.store if cache else None,
                offline=cache.offline if cache else False,
            )
        except Exception as exc:
            log(f"theta option bars range fetch error for {option_ticker}: {exc}")
            _raise_if_offline(cache, exc)
            return []

        results = [
            bar
            for row in _theta_table_rows(data)
            if (bar := _theta_bar_to_polygon_bar(row, fallback_date=start)) is not None
        ]
        if cache and results:
            bars_by_date = cache.option_bars.setdefault(option_ticker, {})
            for bar in results:
                ts_ms = bar.get("t", 0)
                if ts_ms:
                    bar_date = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).date()
                    bars_by_date[bar_date.isoformat()] = bar
        return results

    try:
        data = _polygon_request(
            api_key,
            f"/v2/aggs/ticker/{option_ticker}/range/1/day"
            f"/{start.isoformat()}/{end.isoformat()}",
            params={"adjusted": "true", "sort": "asc", "limit": "250"},
            store=cache.store if cache else None,
            offline=cache.offline if cache else False,
        )
    except Exception as exc:
        log(f"option bars range fetch error for {option_ticker}: {exc}")
        _raise_if_offline(cache, exc)
        return []

    results = data.get("results", [])
    if cache and results:
        bars_by_date = cache.option_bars.setdefault(option_ticker, {})
        for bar in results:
            # Polygon agg timestamps are ms since epoch — convert to date ISO
            ts_ms = bar.get("t", 0)
            if ts_ms:
                bar_date = date.fromtimestamp(ts_ms / 1000)
                bars_by_date[bar_date.isoformat()] = bar
    return results


def _option_bar_price(bar: dict) -> float:
    """Extract price from an option bar: prefer VWAP, then close."""
    vwap = bar.get("vw", 0)
    if vwap and vwap > 0:
        return float(vwap)
    close = bar.get("c", 0)
    return float(close) if close else 0.0


def _option_bar_bid(bar: dict | None) -> float:
    if not bar:
        return 0.0
    bid = _float_or_zero(bar.get("bid"))
    return bid if bid > 0 else 0.0


def _option_bar_ask(bar: dict | None) -> float:
    if not bar:
        return 0.0
    ask = _float_or_zero(bar.get("ask"))
    return ask if ask > 0 else 0.0


def _option_bar_quote_only(bar: dict | None) -> bool:
    return bool(bar and bar.get("quote_only"))


def _option_bar_entry_price(bar: dict | None) -> float:
    ask = _option_bar_ask(bar)
    if ask > 0:
        return ask
    return _option_bar_price(bar or {})


def _option_bar_exit_price(bar: dict | None) -> float:
    bid = _option_bar_bid(bar)
    if bid > 0:
        return bid
    return _option_bar_price(bar or {})


def _equity_bar_price(bar: dict) -> float:
    """Extract price from an equity bar."""
    close = bar.get("c", 0)
    return float(close) if close else 0.0


def _latest_intraday_bar_before(
    api_key: str,
    ticker: str,
    as_of: date | datetime,
    cache: PolygonCache | None = None,
    multiplier: int = 5,
) -> dict | None:
    as_of_dt = _coerce_eastern_datetime(as_of)
    bars = fetch_historical_intraday_bars(
        api_key, ticker, as_of_dt.date(), multiplier=multiplier, cache=cache,
    )
    return _latest_bar_before(bars, as_of_dt)


def _first_intraday_bar_at_or_after(
    api_key: str,
    ticker: str,
    as_of: date | datetime,
    cache: PolygonCache | None = None,
    multiplier: int = 5,
) -> dict | None:
    as_of_dt = _coerce_eastern_datetime(as_of)
    bars = fetch_historical_intraday_bars(
        api_key, ticker, as_of_dt.date(), multiplier=multiplier, cache=cache,
    )
    return _first_bar_at_or_after(bars, as_of_dt)


def _current_option_bar(
    api_key: str,
    option_ticker: str,
    as_of: date | datetime,
    cache: PolygonCache | None = None,
    bar_minutes: int = 5,
) -> dict | None:
    if isinstance(as_of, datetime):
        return _latest_intraday_bar_before(
            api_key, option_ticker, as_of, cache=cache, multiplier=bar_minutes,
        )
    return fetch_option_daily_bar(api_key, option_ticker, as_of, cache=cache)


def _current_equity_bar(
    api_key: str,
    symbol: str,
    as_of: date | datetime,
    cache: PolygonCache | None = None,
    bar_minutes: int = 5,
) -> dict | None:
    if isinstance(as_of, datetime):
        return _latest_intraday_bar_before(
            api_key, symbol, as_of, cache=cache, multiplier=bar_minutes,
        )
    return fetch_historical_daily_bar(api_key, symbol, as_of, cache=cache)


def _session_intraday_bars_before(
    api_key: str,
    ticker: str,
    as_of: date | datetime,
    cache: PolygonCache | None = None,
    multiplier: int = 5,
) -> list[dict]:
    as_of_dt = _coerce_eastern_datetime(as_of)
    cache_key = (ticker, as_of_dt.isoformat(), multiplier)
    if cache and cache_key in cache.session_bars_before:
        return cache.session_bars_before[cache_key]
    bars = fetch_historical_intraday_bars(
        api_key, ticker, as_of_dt.date(), multiplier=multiplier, cache=cache,
    )
    eligible: list[dict] = []
    for bar in bars:
        ts = _bar_timestamp_eastern(bar)
        if ts and ts < as_of_dt:
            eligible.append(bar)
    if cache is not None:
        cache.session_bars_before[cache_key] = eligible
    return eligible


def _mark_to_market_equity(
    realized_equity: float,
    positions: list["SimPosition"],
    as_of: date | datetime,
    api_key: str,
    cache: PolygonCache,
    bar_minutes: int = 5,
) -> tuple[float, float]:
    """Return (account_equity, total_unrealized_pnl) at a timestamp."""
    total_unrealized = 0.0
    for pos in positions:
        if not pos.polygon_ticker:
            continue
        if pos.option_type == "stock":
            bar = _current_equity_bar(
                api_key,
                pos.underlying,
                as_of,
                cache=cache,
                bar_minutes=bar_minutes,
            )
            price = _equity_bar_price(bar) if bar else 0.0
            if price <= 0 or pos.entry_premium <= 0:
                continue
            total_unrealized += (price - pos.entry_premium) * pos.qty
            continue
        bar = _current_option_bar(
            api_key, pos.polygon_ticker, as_of, cache=cache, bar_minutes=bar_minutes,
        )
        premium = _option_bar_price(bar) if bar else 0.0
        if premium <= 0 or pos.entry_premium <= 0:
            continue
        total_unrealized += (premium - pos.entry_premium) * pos.qty * 100
    return realized_equity + total_unrealized, total_unrealized


def _current_underlying_price(
    api_key: str,
    symbol: str,
    as_of: date | datetime,
    cache: PolygonCache,
    bar_minutes: int = 5,
) -> float:
    bar = _current_equity_bar(api_key, symbol, as_of, cache=cache, bar_minutes=bar_minutes)
    if bar is None:
        return 0.0
    return _equity_bar_price(bar)


def _float_or_zero(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _int_or_zero(value: object) -> int:
    try:
        return int(float(value or 0))
    except (TypeError, ValueError):
        return 0


def _option_bar_open_price(bar: dict | None) -> float:
    if not bar:
        return 0.0
    open_price = _float_or_zero(bar.get("o"))
    if open_price > 0:
        return open_price
    return _option_bar_price(bar)


def _option_bar_low_price(bar: dict | None) -> float:
    if not bar:
        return 0.0
    low_price = _float_or_zero(bar.get("l"))
    if low_price > 0:
        return low_price
    return _option_bar_price(bar)


def _option_bar_high_price(bar: dict | None) -> float:
    if not bar:
        return 0.0
    high_price = _float_or_zero(bar.get("h"))
    if high_price > 0:
        return high_price
    return _option_bar_price(bar)


def _historical_option_session_volume(
    api_key: str,
    option_ticker: str,
    as_of: date | datetime,
    cache: PolygonCache | None,
    *,
    bar_minutes: int = 5,
) -> int:
    if cache is None:
        return 0
    if isinstance(as_of, datetime):
        return sum(
            _int_or_zero(bar.get("v"))
            for bar in _session_intraday_bars_before(
                api_key,
                option_ticker,
                as_of,
                cache=cache,
                multiplier=bar_minutes,
            )
        )
    day_bar = fetch_option_daily_bar(api_key, option_ticker, as_of, cache=cache)
    return _int_or_zero(day_bar.get("v")) if day_bar else 0


def _historical_option_quote(
    contract: dict,
    observed_price: float,
    *,
    observed_bid: float = 0.0,
    observed_ask: float = 0.0,
) -> tuple[float, float, float]:
    bid = _float_or_zero(contract.get("bid")) or max(observed_bid, 0.0)
    ask = _float_or_zero(contract.get("ask")) or max(observed_ask, 0.0)
    mid = _float_or_zero(contract.get("mid"))
    if mid <= 0 and bid > 0 and ask > 0:
        mid = (bid + ask) / 2
    if mid <= 0 and observed_price > 0:
        mid = observed_price
    if ask <= 0 and mid > 0:
        ask = mid
    if bid <= 0 and mid > 0:
        bid = mid
    if mid <= 0 and bid > 0 and ask > 0:
        mid = (bid + ask) / 2
    return bid, ask, mid


def _contract_dte(contract: dict, trade_date: date) -> int:
    try:
        expiration = date.fromisoformat(str(contract.get("expiration_date") or ""))
    except ValueError:
        return 9999
    return max((expiration - trade_date).days, 0)


def _target_strike_for_preference(
    spot: float,
    strike_preference: str,
    option_type: str,
) -> float:
    preference = (strike_preference or "atm").lower()
    option_side = (option_type or "").lower()
    if preference == "otm":
        return spot * (1.03 if option_side == "call" else 0.97)
    if preference == "itm":
        return spot * (0.97 if option_side == "call" else 1.03)
    return spot


def _target_dte_for_preference(
    trade_date: date,
    expiry_preference: str,
    default_dte: int,
    target_dte_range: tuple[int, int] | None,
) -> float:
    if target_dte_range is not None:
        return (min(target_dte_range) + max(target_dte_range)) / 2
    if expiry_preference == "this_week":
        days_until_friday = (4 - trade_date.weekday()) % 7
        return max(days_until_friday, 2)
    if expiry_preference == "next_week":
        return 7
    return float(default_dte)


def _rank_contracts_for_hydration(
    contracts: list[dict],
    *,
    spot: float,
    trade_date: date,
    strike_preference: str,
    expiry_preference: str,
    option_type: str,
    default_dte: int,
    contract_symbol: str | None = None,
    target_dte_range: tuple[int, int] | None = None,
    limit: int | None = None,
) -> list[dict]:
    """Bound expensive option-bar hydration to the most plausible contracts."""
    if not contracts:
        return []
    limit = max(int(limit or len(contracts)), 1)
    if len(contracts) <= limit:
        return contracts

    exact_symbol = (contract_symbol or "").strip()
    target_strike = _target_strike_for_preference(spot, strike_preference, option_type)
    target_dte = _target_dte_for_preference(
        trade_date,
        expiry_preference,
        default_dte,
        target_dte_range,
    )

    def score(contract: dict) -> tuple[float, float, float, str]:
        symbol = str(contract.get("ticker") or "")
        exact_penalty = 0.0 if exact_symbol and symbol == exact_symbol else 1.0
        strike = _float_or_zero(contract.get("strike_price"))
        dte = _contract_dte(contract, trade_date)
        return (
            exact_penalty,
            abs(dte - target_dte),
            abs(strike - target_strike),
            symbol,
        )

    return sorted(contracts, key=score)[:limit]


def _polygon_contract_to_option_contract(
    contract: dict,
    *,
    api_key: str,
    underlying: str,
    option_type: str,
    trade_date: date,
    as_of: date | datetime,
    cache: PolygonCache | None,
    bar_minutes: int = 5,
) -> OptionContract | None:
    symbol = str(contract.get("ticker") or "").strip()
    strike = _float_or_zero(contract.get("strike_price"))
    expiry_raw = str(contract.get("expiration_date") or "").strip()
    if not symbol or strike <= 0 or not expiry_raw:
        return None
    try:
        expiration = date.fromisoformat(expiry_raw)
    except ValueError:
        return None
    dte = max((expiration - trade_date).days, 0)
    current_bar = (
        _current_option_bar(api_key, symbol, as_of, cache=cache, bar_minutes=bar_minutes)
        if cache is not None
        else None
    )
    observed_price = _option_bar_price(current_bar) if current_bar else 0.0
    observed_bid = _option_bar_bid(current_bar)
    observed_ask = _option_bar_ask(current_bar)
    bid, ask, mid = _historical_option_quote(
        contract,
        observed_price,
        observed_bid=observed_bid,
        observed_ask=observed_ask,
    )
    session_volume = _historical_option_session_volume(
        api_key,
        symbol,
        as_of,
        cache,
        bar_minutes=bar_minutes,
    )
    volume = max(_int_or_zero(contract.get("volume")), session_volume)
    quote_timestamp = None
    if current_bar and current_bar.get("t"):
        quote_timestamp = _bar_timestamp_eastern(current_bar)
    return OptionContract(
        symbol=symbol,
        underlying=underlying,
        option_type=option_type,
        strike=strike,
        expiration=expiration,
        bid=bid,
        ask=ask,
        mid=mid,
        volume=volume,
        open_interest=_int_or_zero(contract.get("open_interest")),
        dte=dte,
        quote_timestamp=quote_timestamp,
    )


def _entry_limit_price(contract: dict, observed_bar: dict | None) -> float:
    observed_price = _option_bar_price(observed_bar or {})
    bid, ask, mid = _historical_option_quote(
        contract,
        observed_price,
        observed_bid=_option_bar_bid(observed_bar),
        observed_ask=_option_bar_ask(observed_bar),
    )
    if ask <= 0:
        return 0.0
    if mid <= 0:
        mid = ask
    return min(
        round(mid + (ask - mid) * config.OPEN_ORDER_SPREAD_FRACTION, 2),
        ask,
    )


def _limit_buy_fill_price(limit_price: float, fill_bar: dict | None) -> float | None:
    if limit_price <= 0 or fill_bar is None:
        return None
    if _option_bar_quote_only(fill_bar):
        ask_price = _option_bar_ask(fill_bar)
        if ask_price <= 0 or ask_price > limit_price:
            return None
        return ask_price
    low_price = _option_bar_low_price(fill_bar)
    if low_price <= 0 or low_price > limit_price:
        return None
    open_price = _option_bar_open_price(fill_bar)
    if 0 < open_price <= limit_price:
        return open_price
    return limit_price


def _next_fill_option_bar(
    api_key: str,
    option_ticker: str,
    decision_time: datetime,
    cache: PolygonCache,
    bar_minutes: int = 5,
) -> dict | None:
    return _first_intraday_bar_at_or_after(
        api_key, option_ticker, decision_time, cache=cache, multiplier=bar_minutes,
    )


def _select_real_contract(
    api_key: str,
    underlying: str,
    option_type: str,
    spot: float,
    trade_date: date,
    strike_preference: str,
    expiry_preference: str,
    default_dte: int,
    decision_time: datetime | None = None,
    expression_profile: str | None = None,
    contract_symbol: str | None = None,
    target_delta_range: tuple[float, float] | None = None,
    target_dte_range: tuple[int, int] | None = None,
    max_spread_pct: float | None = None,
    cache: PolygonCache | None = None,
    bar_minutes: int = 5,
    reason_out: list[str] | None = None,
) -> dict | None:
    """Select a real Polygon option contract based on LLM preferences.

    Returns the contract dict from Polygon or None if nothing suitable found.
    """
    (
        strike_preference,
        expiry_preference,
        target_delta_range,
        target_dte_range,
    ) = resolve_expression_profile(
        strike_preference,
        expiry_preference,
        expression_profile=expression_profile,
        target_delta_range=target_delta_range,
        target_dte_range=target_dte_range,
    )

    # Determine expiry range
    if target_dte_range is not None:
        resolved_min_dte = max(min(target_dte_range), 1)
        resolved_max_dte = max(target_dte_range)
        resolved_max_dte = max(resolved_max_dte, resolved_min_dte)
        expiry_gte = trade_date + timedelta(days=resolved_min_dte)
        expiry_lte = trade_date + timedelta(days=resolved_max_dte)
    elif expiry_preference == "this_week":
        days_until_friday = (4 - trade_date.weekday()) % 7
        if days_until_friday < 2:
            days_until_friday += 7
        expiry_gte = trade_date + timedelta(days=2)
        expiry_lte = trade_date + timedelta(days=days_until_friday)
    elif expiry_preference == "next_week":
        days_until_friday = (4 - trade_date.weekday()) % 7 + 7
        expiry_gte = trade_date + timedelta(days=5)
        expiry_lte = trade_date + timedelta(days=days_until_friday + 2)
    else:  # monthly
        expiry_gte = trade_date + timedelta(days=max(7, default_dte - 7))
        expiry_lte = trade_date + timedelta(days=default_dte + 7)

    strike_band_pct = (
        0.35
        if (
            contract_symbol
            or target_delta_range is not None
            or expression_profile in {"time_cushion", "stock_proxy", "convex"}
        )
        else 0.15
    )
    strike_gte = round(spot * max(0.05, 1.0 - strike_band_pct), 2)
    strike_lte = round(spot * (1.0 + strike_band_pct), 2)

    contracts = fetch_polygon_option_contracts(
        api_key, underlying, option_type,
        expiry_gte, expiry_lte, strike_gte, strike_lte,
        as_of=trade_date, cache=cache,
    )
    if not contracts:
        if reason_out is not None:
            reason_out.append("no contracts returned")
        return None
    contracts = _rank_contracts_for_hydration(
        contracts,
        spot=spot,
        trade_date=trade_date,
        strike_preference=strike_preference,
        expiry_preference=expiry_preference,
        option_type=option_type,
        default_dte=default_dte,
        contract_symbol=contract_symbol,
        target_dte_range=target_dte_range,
        limit=getattr(config, "BACKTEST_CONTRACT_HYDRATION_LIMIT", 12),
    )
    as_of = decision_time or trade_date
    historical_contracts: list[OptionContract] = []
    raw_by_symbol: dict[str, dict] = {}
    hydrated_count = 0
    quoteable_count = 0
    for contract in contracts:
        hydrated_count += 1
        option_contract = _polygon_contract_to_option_contract(
            contract,
            api_key=api_key,
            underlying=underlying,
            option_type=option_type,
            trade_date=trade_date,
            as_of=as_of,
            cache=cache,
            bar_minutes=bar_minutes,
        )
        if option_contract is None or option_contract.ask <= 0:
            continue
        quoteable_count += 1
        theta_backed = _is_theta_option_ticker(option_contract.symbol)
        if (
            not theta_backed
            and option_contract.open_interest < config.MIN_OPEN_INTEREST
            and option_contract.volume < config.MIN_OPTION_VOLUME
        ):
            continue
        historical_contracts.append(option_contract)
        enriched_contract = dict(contract)
        enriched_contract.update(
            {
                "bid": option_contract.bid,
                "ask": option_contract.ask,
                "mid": option_contract.mid,
                "volume": option_contract.volume,
                "open_interest": option_contract.open_interest,
            }
        )
        raw_by_symbol[option_contract.symbol] = enriched_contract

    if not historical_contracts:
        if reason_out is not None:
            reason_out.append(
                "no contract quote data after hydration"
                if hydrated_count > 0 and quoteable_count == 0
                else "no liquid contracts after filters"
            )
        return None
    if not contract_symbol and target_dte_range is None:
        historical_contracts = filter_contracts_by_expiry_preference(
            historical_contracts,
            expiry_preference,
            as_of=trade_date,
        )
        if not historical_contracts:
            if reason_out is not None:
                reason_out.append("no contracts after expiry preference filter")
            return None

    selected = select_contract(
        historical_contracts,
        spot,
        strike_preference=strike_preference,
        expiry_preference=expiry_preference,
        expression_profile=expression_profile,
        contract_symbol=contract_symbol,
        target_delta_range=target_delta_range,
        target_dte_range=target_dte_range,
        max_spread_pct=max_spread_pct,
    )
    if selected is None:
        if reason_out is not None:
            reason_out.append("contract selector rejected all candidates")
        return None
    return raw_by_symbol.get(selected.symbol)


# ---------------------------------------------------------------------------
# Simulated position
# ---------------------------------------------------------------------------

@dataclass
class SimPosition:
    underlying: str
    option_type: str
    strike: float
    entry_date: date
    expiry_date: date
    entry_premium: float
    qty: int
    conviction: float
    reasoning: str
    expression_profile: str = ""
    polygon_ticker: str = ""
    risk_alert: str = ""

    @property
    def dte_from(self) -> int:
        return (self.expiry_date - self.entry_date).days


@dataclass
class SimTrade:
    entry_date: date
    exit_date: date
    underlying: str
    option_type: str
    strike: float
    entry_premium: float
    exit_premium: float
    qty: int
    pnl: float
    exit_reason: str
    conviction: float
    expression_profile: str = ""
    polygon_ticker: str = ""
    reasoning: str = ""


def _exit_premium_for_position(
    pos: SimPosition,
    decision_time: datetime,
    option_bar: dict | None,
) -> float | None:
    """Return the premium to use for auto-exit checks.

    Expired contracts must leave the book even if Polygon has no bar left for
    them; in that case we mark them at zero instead of keeping stale positions
    alive indefinitely.
    """
    if option_bar is not None:
        premium = _option_bar_exit_price(option_bar)
        if premium > 0:
            return premium
    if decision_time.date() >= pos.expiry_date:
        return 0.0
    return None


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

@dataclass
class BacktestConfig:
    start_date: date
    end_date: date
    initial_equity: float = 100_000.0
    max_risk_per_trade: float = config.MAX_RISK_PER_TRADE
    profit_target_pct: float = config.PROFIT_TARGET_PCT
    stop_loss_pct: float = config.STOP_LOSS_PCT
    time_stop_dte: int = config.TIME_STOP_DTE
    default_dte: int = 14
    max_positions: int = config.MAX_OPEN_POSITIONS
    llm_delay_seconds: float = 1.0
    decision_interval_minutes: int = config.SCAN_INTERVAL_MINUTES
    signal_bar_minutes: int = 5
    news_lookback_hours: int = config.NEWS_LOOKBACK_HOURS
    no_trade_minutes_after_open: int = config.NO_TRADE_MINUTES_AFTER_OPEN
    no_trade_minutes_before_close: int = config.NO_TRADE_MINUTES_BEFORE_CLOSE
    use_journal: bool = True
    journal_max_active: int = config.JOURNAL_MAX_ACTIVE
    journal_max_full_display: int = config.JOURNAL_MAX_FULL_DISPLAY
    journal_stale_cycles: int = config.JOURNAL_STALE_CYCLES
    journal_stale_conviction: float = config.JOURNAL_STALE_CONVICTION
    offline: bool = False
    prepare_only: bool = False
    cache_db_path: Path | None = None
    log_db_path: Path | None = None
    prepare_prefetch_symbols: int = config.PREPARE_PREFETCH_SYMBOLS
    prepare_prefetch_contracts_per_side: int = config.PREPARE_PREFETCH_CONTRACTS_PER_SIDE
    prepare_prefetch_strike_band_pct: float = config.PREPARE_PREFETCH_STRIKE_BAND_PCT
    max_consecutive_llm_errors: int = config.MAX_CONSECUTIVE_LLM_ERROR_CYCLES


@dataclass
class BacktestResult:
    trades: list[SimTrade] = field(default_factory=list)
    equity_curve: list[tuple[str, float]] = field(default_factory=list)
    initial_equity: float = 100_000.0
    final_equity: float = 100_000.0
    total_return_pct: float = 0.0
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float | None = None
    net_pnl: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float | None = None
    profit_factor: float | None = None
    avg_conviction: float = 0.0
    days_tested: int = 0
    llm_error_cycles: int = 0
    llm_failure_days: int = 0
    historical_options_provider: str = ""
    log_db_path: str | None = None
    decision_log: list[dict] = field(default_factory=list)


@dataclass
class PrepareBacktestResult:
    start_date: date
    end_date: date
    days_prepared: int
    decision_points: int
    cache_db_path: Path
    cache_entries: int
    cache_entries_before: int = 0
    cache_entries_added: int = 0
    option_contracts_warmed: int = 0
    option_contract_bars_prefetched: int = 0


def _trading_days(start: date, end: date) -> list[date]:
    """Generate NYSE trading days between start and end (inclusive)."""
    days = []
    current = start
    while current <= end:
        if _is_trading_day(current):
            days.append(current)
        current += timedelta(days=1)
    return days


def _previous_trading_day(trade_date: date) -> date:
    """Return the prior NYSE session for daily backtest signal generation."""
    previous = trade_date - timedelta(days=1)
    while not _is_trading_day(previous):
        previous -= timedelta(days=1)
    return previous


def _last_completed_trading_day(as_of: date | datetime) -> date:
    """Return the most recent fully completed session available at as_of."""
    as_of_dt = _coerce_eastern_datetime(as_of)
    session_day = as_of_dt.date()
    while not _is_trading_day(session_day):
        session_day = _previous_trading_day(session_day)
    if as_of_dt < _market_close_dt(session_day):
        return _previous_trading_day(session_day)
    return session_day


def _is_llm_error_message(message: str | None) -> bool:
    if not isinstance(message, str):
        return False
    return message.strip().lower().startswith("llm error:")


def _analysis_to_decisions_json(
    analysis: MarketAnalysis,
    llm_diagnostics: dict | None = None,
) -> str:
    return json.dumps(
        {
            "market_analysis": analysis.analysis,
            "thesis_updates": [
                {
                    "id": update.id,
                    "underlying": update.underlying,
                    "direction": update.direction,
                    "thesis": update.thesis,
                    "conviction": update.conviction,
                    "status": update.status,
                    "new_observation": update.new_observation,
                }
                for update in analysis.thesis_updates
            ],
            "trades": [
                {
                    "action": decision.action,
                    "underlying": decision.underlying,
                    "strike_preference": decision.strike_preference,
                    "expiry_preference": decision.expiry_preference,
                    "conviction": decision.conviction,
                    "risk_pct": decision.risk_pct,
                    "reasoning": decision.reasoning,
                    "target_symbol": decision.target_symbol,
                    "expression_profile": decision.expression_profile,
                    "contract_symbol": decision.contract_symbol,
                    "target_delta_range": decision.target_delta_range,
                    "target_dte_range": decision.target_dte_range,
                    "max_spread_pct": decision.max_spread_pct,
                }
                for decision in analysis.trades
            ],
            "dropped_trades": analysis.dropped_trades,
            "llm_diagnostics": llm_diagnostics or {},
        }
    )


def _log_backtest_open_trade(
    logger: AITradeLogger | None,
    *,
    decision_time: datetime,
    decision: TradeDecision,
    market_analysis: str,
    option_type: str,
    strike: float,
    expiry: date | None,
    polygon_ticker: str,
    qty: int,
    premium: float,
) -> None:
    if logger is None:
        return
    multiplier = 1 if option_type == "stock" else 100
    logger.log_trade(
        AITradeRecord(
            timestamp=decision_time,
            symbol=polygon_ticker,
            underlying=decision.underlying,
            option_type=option_type,
            strike=strike,
            expiration=expiry.isoformat() if expiry else "",
            action=decision.action,
            qty=qty,
            premium=premium,
            total_cost=premium * qty * multiplier,
            conviction=decision.conviction,
            reasoning=decision.reasoning,
            market_analysis=market_analysis,
            order_id=None,
            status="filled",
            expression_profile=decision.expression_profile or "",
        )
    )


def _log_backtest_close(
    logger: AITradeLogger | None,
    *,
    decision_time: datetime,
    position: SimPosition,
    exit_premium: float,
    trade_pnl: float,
    reason: str,
) -> None:
    if logger is None:
        return
    logger.log_position_close(
        PositionCloseRecord(
            timestamp=decision_time,
            symbol=position.polygon_ticker,
            underlying=position.underlying,
            qty=position.qty,
            entry_premium=position.entry_premium,
            exit_premium=exit_premium,
            pnl=trade_pnl,
            reason=reason,
            order_id=None,
            expression_profile=position.expression_profile,
            option_type=position.option_type,
            expiration=position.expiry_date.isoformat(),
            entry_date=position.entry_date.isoformat(),
        )
    )


# ---------------------------------------------------------------------------
# News quality filter — remove lawsuit spam & corporate housekeeping
# ---------------------------------------------------------------------------

_JUNK_TITLE_PATTERNS = [
    # Lawsuit/legal spam
    "securities fraud",
    "class action",
    "loss recovery",
    "reminds investors",
    "encourages.*investors",
    "investor rights",
    "deadline reminder",
    "deadline tuesday",
    "deadline alert",
    "recover your losses",
    "leading law firm",
    "have suffered losses",
    "securities class action",
    # Corporate housekeeping
    "inducement grant",
    "listing rule 5635",
    "repurchase of own shares",
    "omien osakkeiden",            # Nokia Finnish filings
]

_JUNK_RE = re.compile("|".join(_JUNK_TITLE_PATTERNS), re.IGNORECASE)


def _filter_news_quality(articles: list[dict]) -> list[dict]:
    """Remove junk articles (lawsuit spam, corporate housekeeping) by title."""
    return [a for a in articles if not _JUNK_RE.search(a.get("title", ""))]


def _format_news_for_backtest(
    articles: list[dict],
    focus_symbols: list[str] | None = None,
    reference_time: datetime | None = None,
) -> str:
    items = _news_items_from_backtest_articles(articles, reference_time=reference_time)
    return format_news_for_llm(
        items,
        focus_symbols=focus_symbols,
        reference_time=reference_time,
    )


def _news_items_from_backtest_articles(
    articles: list[dict],
    reference_time: datetime | None = None,
) -> list[NewsItem]:
    items: list[NewsItem] = []
    for article in articles:
        published_raw = article.get("published_utc", "")
        try:
            published_at = datetime.fromisoformat(str(published_raw).replace("Z", "+00:00")).astimezone(EASTERN_TZ)
        except ValueError:
            published_at = reference_time or now_eastern()
        items.append(
            NewsItem(
                headline=article.get("title", ""),
                summary=(article.get("description") or "")[:1000],
                source=article.get("publisher", {}).get("name", "") or "unknown",
                symbols=[str(t).upper() for t in article.get("tickers", []) if t],
                published_at=published_at,
                url=article.get("article_url") or "",
            )
        )
    return items


def _build_performance_summary(
    closed_trades: list[dict], equity: float, initial_equity: float,
) -> str:
    """Build a running performance summary from closed trades for LLM context."""
    if not closed_trades:
        return ""

    def _symbol_expiration(symbol: str) -> date | None:
        for index, char in enumerate(symbol):
            if not char.isdigit() or index + 6 > len(symbol):
                continue
            try:
                value = symbol[index:index + 6]
                year = 2000 + int(value[:2])
                month = int(value[2:4])
                day = int(value[4:6])
                return date(year, month, day)
            except ValueError:
                continue
        return None

    def _entry_dte(trade: dict) -> int | None:
        entry_raw = str(trade.get("entry_date", "") or "")[:10]
        symbol = str(trade.get("symbol") or trade.get("polygon_ticker") or "")
        if not entry_raw or not symbol:
            return None
        try:
            entry_day = date.fromisoformat(entry_raw)
        except ValueError:
            return None
        expiry = _symbol_expiration(symbol)
        if expiry is None:
            return None
        return max((expiry - entry_day).days, 0)

    wins = [t for t in closed_trades if (t.get("pnl") or 0) > 0]
    losses = [t for t in closed_trades if (t.get("pnl") or 0) < 0]
    total = len(closed_trades)
    n_wins = len(wins)
    n_losses = len(losses)
    win_rate = n_wins / total * 100 if total > 0 else 0
    net_pnl = sum(t.get("pnl", 0) for t in closed_trades)
    ret_pct = (equity - initial_equity) / initial_equity * 100 if initial_equity > 0 else 0

    avg_win = sum(t["pnl"] for t in wins) / n_wins if n_wins else 0
    avg_loss = sum(t["pnl"] for t in losses) / n_losses if n_losses else 0
    wl_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf") if avg_win > 0 else 0

    # Recent streak (last 5)
    recent = closed_trades[-5:]
    streak = ", ".join("W" if (t.get("pnl") or 0) > 0 else "L" for t in recent)

    # Per-ticker stats (top 5 most traded)
    ticker_stats: dict[str, dict] = {}
    for t in closed_trades:
        sym = t.get("underlying", "???")
        s = ticker_stats.setdefault(sym, {"wins": 0, "losses": 0, "pnl": 0.0, "trades": []})
        pnl = t.get("pnl", 0)
        if pnl > 0:
            s["wins"] += 1
        elif pnl < 0:
            s["losses"] += 1
        s["pnl"] += pnl
        s["trades"].append(t)
    top_tickers = sorted(ticker_stats.items(), key=lambda x: x[1]["wins"] + x[1]["losses"], reverse=True)[:5]

    lines = ["\n--- Running Performance ---"]

    # Repeat losers FIRST — most actionable signal, must not get buried
    repeat_losers = [
        (sym, s) for sym, s in ticker_stats.items()
        if s["losses"] >= 2 and s["wins"] == 0
    ]
    repeat_losers.sort(key=lambda x: x[1]["losses"], reverse=True)
    if repeat_losers:
        lines.append("WARNING — REPEAT LOSERS (you keep losing on these tickers):")
        for sym, s in repeat_losers:
            n_trades = s["losses"]
            total_lost = s["pnl"]
            avg_loss_val = total_lost / n_trades if n_trades else 0
            hold_days = []
            for t in s["trades"]:
                entry_str = t.get("entry_date", "")
                exit_str = t.get("timestamp", "")
                if entry_str and exit_str:
                    try:
                        ed = date.fromisoformat(str(entry_str)[:10])
                        xd = date.fromisoformat(str(exit_str)[:10])
                        hold_days.append((xd - ed).days)
                    except (ValueError, TypeError):
                        pass
            avg_hold = sum(hold_days) / len(hold_days) if hold_days else 0
            lines.append(
                f"  {sym}: {n_trades} trades, 0 wins, ${total_lost:+,.0f} total lost"
                f" — avg hold: {avg_hold:.1f} days, avg loss: ${avg_loss_val:+,.0f}"
            )

    # Equity momentum (last 5 trades)
    if recent:
        momentum = sum(t.get("pnl", 0) for t in recent)
        m_wins = sum(1 for t in recent if (t.get("pnl") or 0) > 0)
        m_losses = sum(1 for t in recent if (t.get("pnl") or 0) < 0)
        lines.append(
            f"Equity momentum (last {len(recent)} trades): ${momentum:+,.0f} "
            f"({m_wins}W/{m_losses}L)"
        )

    high_conviction = [t for t in closed_trades if float(t.get("conviction") or 0) >= 0.65]
    if high_conviction:
        high_conviction_wins = sum(1 for t in high_conviction if (t.get("pnl") or 0) > 0)
        high_conviction_net = sum(t.get("pnl", 0) for t in high_conviction)
        lines.append(
            "High-conviction review (>=0.65): "
            f"{high_conviction_wins}/{len(high_conviction)} wins net=${high_conviction_net:+,.0f}"
        )

    short_dte_trades = []
    longer_dte_trades = []
    for trade in closed_trades:
        entry_dte = _entry_dte(trade)
        if entry_dte is None:
            continue
        if entry_dte <= 10:
            short_dte_trades.append(trade)
        else:
            longer_dte_trades.append(trade)

    if short_dte_trades:
        short_wins = sum(1 for trade in short_dte_trades if (trade.get("pnl") or 0) > 0)
        short_net = sum(trade.get("pnl", 0) for trade in short_dte_trades)
        lines.append(
            "Fast-decay review (<=10 DTE at entry): "
            f"{short_wins}/{len(short_dte_trades)} wins net=${short_net:+,.0f}"
        )
    if longer_dte_trades:
        long_wins = sum(1 for trade in longer_dte_trades if (trade.get("pnl") or 0) > 0)
        long_net = sum(trade.get("pnl", 0) for trade in longer_dte_trades)
        lines.append(
            "More-time review (>10 DTE at entry): "
            f"{long_wins}/{len(longer_dte_trades)} wins net=${long_net:+,.0f}"
        )
    lines.extend(profile_calibration_lines(closed_trades))

    lines.extend([
        f"Total trades: {total} | Wins: {n_wins} | Losses: {n_losses} | Win rate: {win_rate:.1f}%",
        f"Net P&L: ${net_pnl:+,.2f} | Equity: ${equity:,.2f} ({ret_pct:+.1f}% vs start)",
        f"Avg winner: ${avg_win:+,.2f} | Avg loser: ${avg_loss:+,.2f} | W/L ratio: {wl_ratio:.2f}",
        f"Recent streak (last {len(recent)}): {streak}",
    ])
    if top_tickers:
        lines.append("Top tickers:")
        for sym, s in top_tickers:
            lines.append(f"  {sym}: {s['wins']}W/{s['losses']}L net=${s['pnl']:+,.2f}")

    return "\n".join(lines)


def _backtest_option_loss_streak_guard_reason(
    closed_trades: list[dict],
    conviction: float,
    *,
    lookback: int | None = None,
    min_conviction: float | None = None,
) -> str:
    lookback = (
        config.OPTION_LOSS_STREAK_GUARD_LOOKBACK
        if lookback is None else int(lookback)
    )
    min_conviction = (
        config.OPTION_LOSS_STREAK_GUARD_MIN_CONVICTION
        if min_conviction is None else float(min_conviction)
    )
    if lookback <= 0 or conviction >= min_conviction:
        return ""

    option_closes = [
        trade for trade in reversed(closed_trades)
        if str(trade.get("option_type") or "").lower() in {"call", "put"}
    ]
    if len(option_closes) < lookback:
        return ""

    if all(float(trade.get("pnl") or 0.0) < 0 for trade in option_closes[:lookback]):
        return (
            f"last {lookback} closed option trades were losses; "
            f"option entries require conviction >= {min_conviction:.2f}"
        )
    return ""


def _stock_symbol_notional_for_budget(
    positions: list[SimPosition],
    underlying: str,
    current_price: float,
) -> float:
    total = 0.0
    for pos in positions:
        if pos.option_type != "stock":
            continue
        if pos.underlying.upper() != underlying.upper():
            continue
        reference_price = current_price if current_price > 0 else pos.entry_premium
        total += abs(pos.qty) * reference_price
    return total


def _build_market_trend_context(
    api_key: str,
    as_of: date | datetime,
    lookback_days: int = 10,
    cache: PolygonCache | None = None,
    bar_minutes: int = 5,
    metrics_cache: dict[str, dict | None] | None = None,
) -> str:
    """Build market trend context using only information available at as_of."""
    is_intraday = isinstance(as_of, datetime)
    as_of_dt = _coerce_eastern_datetime(as_of)
    if is_intraday:
        lines = [
            f"Major Indices ({lookback_days}-day view) as of "
            f"{as_of_dt.strftime('%Y-%m-%d %H:%M %Z')}:"
        ]
        change_label = "now"
    else:
        lines = [f"Major Indices ({lookback_days}-day view):"]
        change_label = "today"

    for idx_sym in ["SPY", "QQQ", "IWM"]:
        metrics = _ticker_price_metrics_as_of(
            api_key,
            idx_sym,
            as_of_dt,
            lookback_days=lookback_days,
            cache=cache,
            bar_minutes=bar_minutes,
            metrics_cache=metrics_cache,
        )
        if not metrics:
            continue

        lines.append(
            f"  {idx_sym}: ${metrics['price']:.2f}"
            f" {change_label}({metrics['intraday_chg']:+.2f}%)"
            f" 5d({metrics['five_d_chg']:+.1f}%)"
            f" 10d({metrics['ten_d_chg']:+.1f}%)"
            f" trend={metrics['trend']}"
        )

    return "\n".join(lines)


def _build_ticker_price_context(
    api_key: str,
    ticker: str,
    as_of: date | datetime,
    lookback_days: int = 10,
    cache: PolygonCache | None = None,
    bar_minutes: int = 5,
    metrics_cache: dict[str, dict | None] | None = None,
) -> tuple[str | None, float]:
    """Build price trend context for a single ticker as of a timestamp."""
    metrics = _ticker_price_metrics_as_of(
        api_key,
        ticker,
        as_of,
        lookback_days=lookback_days,
        cache=cache,
        bar_minutes=bar_minutes,
        metrics_cache=metrics_cache,
    )
    if not metrics:
        return None, 0.0

    ctx = (
        f"spot=${metrics['price']:.2f}"
        f" {'now' if metrics['is_intraday'] else 'today'}({metrics['intraday_chg']:+.1f}%)"
        f" 5d({metrics['five_d_chg']:+.1f}%)"
        f" 10d({metrics['ten_d_chg']:+.1f}%)"
        f" {'session' if metrics['is_intraday'] else 'hi/lo'}=${metrics['session_low']:.0f}/${metrics['session_high']:.0f}"
        f" range={metrics['range_pos_pct']:.0f}% {metrics['range_label']}"
    )
    if metrics["is_intraday"]:
        ctx += (
            f" 10d_hi/lo=${metrics['recent_high']:.0f}/{metrics['recent_low']:.0f}"
        )
    return ctx, metrics["price"]


def _ticker_price_metrics_as_of(
    api_key: str,
    ticker: str,
    as_of: date | datetime,
    lookback_days: int = 10,
    cache: PolygonCache | None = None,
    bar_minutes: int = 5,
    metrics_cache: dict[str, dict | None] | None = None,
) -> dict | None:
    """Return price/trend metrics for a ticker using only info available at as_of."""
    is_intraday = isinstance(as_of, datetime)
    as_of_dt = _coerce_eastern_datetime(as_of)
    normalized_ticker = ticker.upper()
    if metrics_cache is not None and normalized_ticker in metrics_cache:
        return metrics_cache[normalized_ticker]
    cache_key = (normalized_ticker, as_of_dt.isoformat(), lookback_days, bar_minutes)
    if cache is not None and cache_key in cache.ticker_metrics:
        cached = cache.ticker_metrics[cache_key]
        if metrics_cache is not None:
            metrics_cache[normalized_ticker] = cached
        return cached
    last_completed_day = _last_completed_trading_day(as_of_dt)
    start = last_completed_day - timedelta(days=int(lookback_days * 1.8))
    bars = fetch_historical_daily_bars_range(
        api_key,
        ticker,
        start,
        last_completed_day,
        cache=cache,
    )
    if not bars:
        if cache is not None:
            cache.ticker_metrics[cache_key] = None
        if metrics_cache is not None:
            metrics_cache[normalized_ticker] = None
        return None

    current_price = float(bars[-1].get("c", 0))
    intraday_chg = 0.0
    session_high = current_price
    session_low = current_price

    if isinstance(as_of, datetime):
        intraday_bars = _session_intraday_bars_before(
            api_key, ticker, as_of_dt, cache=cache, multiplier=bar_minutes,
        )
        if intraday_bars:
            latest_bar = intraday_bars[-1]
            current_price = _equity_bar_price(latest_bar)
            session_open = float(intraday_bars[0].get("o", 0))
            intraday_chg = (
                (current_price - session_open) / session_open * 100
                if session_open > 0 and current_price > 0 else 0.0
            )
            session_high = max(float(b.get("h", 0)) for b in intraday_bars)
            lows = [float(b.get("l", 0)) for b in intraday_bars if float(b.get("l", 0)) > 0]
            session_low = min(lows) if lows else current_price
    else:
        today_bar = bars[-1]
        today_open = float(today_bar.get("o", 0))
        intraday_chg = (
            (current_price - today_open) / today_open * 100
            if today_open > 0 and current_price > 0 else 0.0
        )
        session_high = float(today_bar.get("h", current_price) or current_price)
        session_low = float(today_bar.get("l", current_price) or current_price)

    if current_price <= 0:
        return None

    five_d_chg = 0.0
    if len(bars) >= 6:
        ref = float(bars[-6].get("c", 0))
        if ref > 0:
            five_d_chg = (current_price - ref) / ref * 100

    ten_d_chg = 0.0
    if len(bars) >= 11:
        ref = float(bars[-11].get("c", 0))
        if ref > 0:
            ten_d_chg = (current_price - ref) / ref * 100

    recent_bars = bars[-lookback_days:] if len(bars) >= lookback_days else bars
    recent_high = max(float(b.get("h", 0)) for b in recent_bars)
    recent_low = min(float(b.get("l", float("inf"))) for b in recent_bars)
    if recent_high > recent_low:
        range_pos_pct = max(0.0, min(100.0, (current_price - recent_low) / (recent_high - recent_low) * 100))
    else:
        range_pos_pct = 50.0
    if range_pos_pct >= 85:
        range_label = "near_10d_high"
    elif range_pos_pct <= 15:
        range_label = "near_10d_low"
    else:
        range_label = "mid_range"
    if not is_intraday:
        session_high = recent_high
        session_low = recent_low
    trend = "up" if current_price >= (
        sum(float(b.get("c", 0)) for b in recent_bars) / len(recent_bars)
    ) else "down"
    result = {
        "price": current_price,
        "intraday_chg": intraday_chg,
        "five_d_chg": five_d_chg,
        "ten_d_chg": ten_d_chg,
        "session_high": session_high,
        "session_low": session_low,
        "recent_high": recent_high,
        "recent_low": recent_low,
        "range_pos_pct": range_pos_pct,
        "range_label": range_label,
        "trend": trend,
        "is_intraday": is_intraday,
    }
    if cache is not None:
        cache.ticker_metrics[cache_key] = result
    if metrics_cache is not None:
        metrics_cache[normalized_ticker] = result
    return result


def _build_catalyst_reaction_context(
    api_key: str,
    news_events: list,
    as_of: date | datetime,
    focus_symbols: list[str] | None = None,
    cache: PolygonCache | None = None,
    bar_minutes: int = 5,
    max_events: int = 5,
    metrics_cache: dict[str, dict | None] | None = None,
) -> str:
    lines = ["Catalyst Reaction Snapshot:"]
    seen: set[str] = set()
    focus_set = {symbol.upper() for symbol in (focus_symbols or [])}
    for event in news_events:
        symbol = next((sym.upper() for sym in event.symbols if sym), "")
        if not symbol or symbol in seen:
            continue
        if focus_set and symbol not in focus_set:
            continue
        metrics = _ticker_price_metrics_as_of(
            api_key,
            symbol,
            as_of,
            cache=cache,
            bar_minutes=bar_minutes,
            metrics_cache=metrics_cache,
        )
        if not metrics:
            continue
        reaction = classify_catalyst_reaction(
            event.age_minutes,
            metrics["intraday_chg"],
            metrics["five_d_chg"],
        )
        lines.append(
            f"  {symbol}: event={event.event_type}/{event.freshness}"
            f" age={event.age_minutes}m"
            f" now({metrics['intraday_chg']:+.1f}%)"
            f" 5d({metrics['five_d_chg']:+.1f}%)"
            f" trend={metrics['trend']}"
            f" reaction={reaction}"
        )
        seen.add(symbol)
        if len(seen) >= max_events:
            break
    return "\n".join(lines) if len(lines) > 1 else ""


def _annotate_closed_trades(
    closed_trades: list[dict],
    api_key: str,
    cache: PolygonCache,
) -> list[dict]:
    """Annotate closed trades with underlying price movement over holding period.

    Adds a 'context' field to all closed trades with valid entry/exit dates,
    showing what the underlying stock did while the position was held.
    Uses cached bars when available to minimize API calls.
    """
    annotated = []
    # Cache underlying bars per ticker to avoid redundant fetches
    underlying_bars_cache: dict[str, list[dict]] = {}

    for trade in closed_trades:
        trade = dict(trade)  # shallow copy to avoid mutating original
        underlying = trade.get("underlying", "")
        entry_date_str = trade.get("entry_date", "")
        exit_date_str = trade.get("timestamp", "")  # timestamp is exit date

        if underlying and entry_date_str and exit_date_str:
            try:
                entry_dt = (
                    date.fromisoformat(str(entry_date_str)[:10])
                    if isinstance(entry_date_str, str) else entry_date_str
                )
                exit_dt = (
                    date.fromisoformat(str(exit_date_str)[:10])
                    if isinstance(exit_date_str, str) else exit_date_str
                )
            except (ValueError, TypeError):
                annotated.append(trade)
                continue

            cache_key = underlying
            if cache_key not in underlying_bars_cache:
                # Fetch full range — usually already available from other queries
                bars = fetch_historical_daily_bars_range(
                    api_key,
                    underlying,
                    entry_dt,
                    exit_dt,
                    cache=cache,
                )
                underlying_bars_cache[cache_key] = bars

            bars = underlying_bars_cache[cache_key]
            if len(bars) >= 2:
                entry_price = float(bars[0].get("c", 0))
                exit_price = float(bars[-1].get("c", 0))
                if entry_price > 0 and exit_price > 0:
                    move_pct = (exit_price - entry_price) / entry_price * 100
                    option_type = trade.get("option_type", "")
                    direction = "bearish" if option_type == "put" else "bullish"
                    trade["context"] = (
                        f"Underlying moved: {underlying}"
                        f" ${entry_price:.0f}→${exit_price:.0f}"
                        f" ({move_pct:+.1f}%)"
                        f" while holding {direction} position"
                    )

        annotated.append(trade)

    return annotated


def _extract_top_news_tickers(
    news: list[dict],
    exclude: set[str] | None = None,
    max_tickers: int = 5,
) -> list[str]:
    """Count ticker mentions across news articles, return top N.

    Excludes broad index ETFs by default so the LLM focuses on
    individual stocks with actual options chains.
    """
    if exclude is None:
        exclude = {"SPY", "QQQ", "IWM", "DIA"}

    counts: Counter[str] = Counter()
    for article in news:
        for ticker in article.get("tickers", []):
            t = ticker.strip().upper()
            if t and t not in exclude and is_equity_candidate_symbol(t):
                counts[t] += 1

    return [t for t, _ in counts.most_common(max_tickers)]


def _build_focus_tickers(
    news_events: list,
    positions: list[SimPosition],
    journal: ThesisJournal | None,
    api_key: str,
    as_of: date | datetime,
    cache: PolygonCache,
    max_tickers: int = config.WATCHLIST_SIZE,
    metrics_cache: dict[str, dict | None] | None = None,
) -> tuple[str, list[str]]:
    """Build a candidate table and finalist symbols for deep-dive context."""
    source_tags_by_symbol: dict[str, set[str]] = {}
    position_symbols = [pos.underlying.upper() for pos in positions if pos.underlying]
    thesis_symbols = [
        entry.underlying.upper()
        for entry in (journal.active_entries() if journal else [])
        if entry.underlying
    ]
    direct_symbols = rank_symbols_from_events(
        news_events,
        focus_symbols=position_symbols + thesis_symbols,
        max_symbols=max_tickers,
    )
    if not direct_symbols and news_events:
        direct_symbols = list(dict.fromkeys(
            symbol.upper()
            for event in news_events
            for symbol in event.symbols
            if symbol
        ))[:max_tickers]
    spillover_symbols = expand_symbols_with_relationships(
        direct_symbols,
        events=news_events,
        max_symbols=max_tickers,
    )

    def _tag_symbols(symbols: list[str], tag: str) -> None:
        for symbol in symbols:
            normalized = symbol.upper()
            if not normalized or not is_equity_candidate_symbol(normalized):
                continue
            source_tags_by_symbol.setdefault(normalized, set()).add(tag)

    _tag_symbols(position_symbols, "position")
    _tag_symbols(thesis_symbols, "thesis")
    _tag_symbols(direct_symbols, "direct")
    direct_set = {symbol.upper() for symbol in direct_symbols}
    _tag_symbols(
        [symbol for symbol in spillover_symbols if symbol.upper() not in direct_set],
        "spillover",
    )
    _tag_symbols(["SPY", "QQQ", "IWM", "XLK", "XLF", "XLE", "XLV"], "macro")

    best_events = map_best_events_by_symbol(news_events)
    event_scores = {
        symbol: score_news_event(event)
        for symbol, event in best_events.items()
    }
    metrics_by_symbol: dict[str, dict] = {}
    for symbol in list(source_tags_by_symbol):
        metrics = _ticker_price_metrics_as_of(
            api_key,
            symbol,
            as_of,
            cache=cache,
            metrics_cache=metrics_cache,
        )
        if not metrics:
            continue
        event = best_events.get(symbol)
        reaction = ""
        if event is not None:
            reaction = classify_catalyst_reaction(
                event.age_minutes,
                metrics["intraday_chg"],
                metrics["five_d_chg"],
            )
        metrics_by_symbol[symbol] = {
            **metrics,
            "reaction": reaction,
        }

    candidates = build_candidate_ideas(
        source_tags_by_symbol,
        metrics_by_symbol,
        best_events,
        event_scores,
    )
    candidate_context = format_candidate_table(
        candidates,
        max_rows=config.CANDIDATE_TABLE_SIZE,
    )
    finalists = select_candidate_finalists(
        candidates,
        max_symbols=config.CANDIDATE_FINALISTS,
    )
    return candidate_context, finalists


def _build_options_context(
    api_key: str,
    tickers: list[str],
    as_of: date | datetime,
    cache: PolygonCache,
    best_events: dict[str, object] | None = None,
    closed_trades: list[dict] | None = None,
    default_dte: int = 14,
    bar_minutes: int = 5,
    metrics_cache: dict[str, dict | None] | None = None,
) -> str:
    """Build a real options context string for the most-discussed tickers.

    For each ticker: fetch spot price, build a call/put shortlist,
    and show the latest premium available before as_of.
    """
    if not tickers:
        return "(Backtest mode: real historical options data used for pricing)"

    as_of_dt = _coerce_eastern_datetime(as_of)
    trade_date = as_of_dt.date()
    lines: list[str] = [
        "Available options for the current focus tickers "
        f"(as of {as_of_dt.strftime('%H:%M %Z')}, detailed premium shown for the primary call/put only):"
    ]
    lines.extend(expression_guidance_lines(closed_trades or []))
    found_any = False

    expiry_gte = trade_date + timedelta(days=max(7, default_dte - 7))
    expiry_lte = trade_date + timedelta(days=default_dte + 7)

    for ticker in tickers:
        metrics = _ticker_price_metrics_as_of(
            api_key,
            ticker,
            as_of_dt,
            cache=cache,
            lookback_days=10,
            bar_minutes=bar_minutes,
            metrics_cache=metrics_cache,
        )
        # Fetch spot price + trend context (single range query, no extra call)
        trend_ctx, spot = _build_ticker_price_context(
            api_key,
            ticker,
            as_of_dt,
            cache=cache,
            bar_minutes=bar_minutes,
            metrics_cache=metrics_cache,
        )
        if spot <= 0:
            bar = _current_equity_bar(
                api_key, ticker, as_of_dt, cache=cache, bar_minutes=bar_minutes,
            )
            if bar is None:
                continue
            spot = _equity_bar_price(bar)
            if spot <= 0:
                continue

        if trend_ctx:
            lines.append(f"  {ticker} ({trend_ctx}):")
        else:
            lines.append(f"  {ticker} (spot=${spot:.2f}):")
        lines.append(
            format_symbol_setup_context(
                ticker,
                metrics,
                (best_events or {}).get(ticker.upper()),
            )
        )
        strike_gte = round(spot * 0.97, 2)
        strike_lte = round(spot * 1.03, 2)

        for opt_type in ("call", "put"):
            try:
                contracts = fetch_polygon_option_contracts(
                    api_key, ticker, opt_type,
                    expiry_gte, expiry_lte, strike_gte, strike_lte,
                    as_of=trade_date, cache=cache,
                )
            except RuntimeError as exc:
                if _is_offline_cache_miss(exc):
                    log(f"options context cache miss for {ticker} {opt_type}: {exc}")
                    continue
                raise
            if not contracts:
                continue
            def _shortlist_key(contract: dict) -> tuple[float, int]:
                strike = float(contract.get("strike_price", 0) or 0)
                expiry_raw = str(contract.get("expiration_date") or "")
                try:
                    expiry = date.fromisoformat(expiry_raw)
                    dte = max((expiry - trade_date).days, 0)
                except ValueError:
                    dte = default_dte
                return (
                    abs(strike - spot),
                    abs(dte - default_dte),
                )
            ranked_contracts = sorted(
                contracts,
                key=_shortlist_key,
            )[:3]
            if ranked_contracts:
                lines.append(f"    {opt_type.upper()} shortlist:")
            for idx, best in enumerate(ranked_contracts):
                opt_ticker = best.get("ticker", "")
                strike = float(best.get("strike_price", 0) or 0)
                expiry = str(best.get("expiration_date") or "")
                if not opt_ticker or strike <= 0:
                    continue

                dte = default_dte
                try:
                    dte = max((date.fromisoformat(expiry) - trade_date).days, 0)
                except ValueError:
                    pass
                delta = abs(approx_delta(strike, spot, max(dte, 1), opt_type))

                if idx > 0:
                    lines.append(
                        f"      {opt_ticker} {opt_type.upper()} ${strike:.2f} exp={expiry}"
                        f" delta~{delta:.2f} alt"
                    )
                    found_any = True
                    continue

                try:
                    session_bars = _session_intraday_bars_before(
                        api_key, opt_ticker, as_of_dt, cache=cache, multiplier=bar_minutes,
                    )
                except RuntimeError as exc:
                    if _is_offline_cache_miss(exc):
                        log(f"options context cache miss for {opt_ticker} bars: {exc}")
                        continue
                    raise
                if not session_bars:
                    continue

                opt_bar = session_bars[-1]
                premium = _option_bar_price(opt_bar)
                if premium <= 0:
                    continue
                vol = sum(int(b.get("v") or 0) for b in session_bars)
                lows = [float(b.get("l", 0)) for b in session_bars if float(b.get("l", 0)) > 0]
                highs = [float(b.get("h", 0)) for b in session_bars if float(b.get("h", 0)) > 0]
                session_range = (
                    f" session_range=${min(lows):.2f}-${max(highs):.2f}"
                    if lows and highs else ""
                )
                bar_time = _bar_timestamp_eastern(opt_bar)
                bar_label = f" asof={bar_time.strftime('%H:%M')}" if bar_time else ""
                lines.append(
                    f"      {opt_ticker} {opt_type.upper()} ${strike:.2f} exp={expiry}"
                    f" delta~{delta:.2f} premium=${premium:.2f} vol={vol}{session_range}{bar_label}"
                )
                found_any = True

    if not found_any:
        return "(Backtest mode: real historical options data used for pricing)"

    return "\n".join(lines)


def _contract_moneyness_bucket(option_type: str, strike: float, spot: float) -> str:
    if strike <= 0 or spot <= 0:
        return "atm"
    distance_pct = abs(strike - spot) / spot
    if distance_pct <= 0.02:
        return "atm"
    if option_type == "call":
        return "itm" if strike < spot else "otm"
    return "itm" if strike > spot else "otm"


def _rank_prefetch_contracts(
    contracts: list[dict],
    *,
    spot: float,
    trade_date: date,
    default_dte: int,
    option_type: str,
    limit: int,
) -> list[dict]:
    if limit <= 0 or not contracts:
        return []

    def _shortlist_key(contract: dict) -> tuple[float, int, str]:
        strike = float(contract.get("strike_price", 0) or 0)
        expiry_raw = str(contract.get("expiration_date") or "")
        try:
            expiry = date.fromisoformat(expiry_raw)
            dte = max((expiry - trade_date).days, 0)
        except ValueError:
            dte = default_dte
        return (
            abs(strike - spot),
            abs(dte - default_dte),
            str(contract.get("ticker") or ""),
        )

    buckets: dict[str, list[dict]] = {"atm": [], "itm": [], "otm": []}
    for contract in sorted(contracts, key=_shortlist_key):
        strike = float(contract.get("strike_price", 0) or 0)
        bucket = _contract_moneyness_bucket(option_type, strike, spot)
        buckets.setdefault(bucket, []).append(contract)

    ranked: list[dict] = []
    while len(ranked) < limit and any(buckets.values()):
        for bucket in ("atm", "itm", "otm"):
            queue = buckets.get(bucket, [])
            if not queue:
                continue
            ranked.append(queue.pop(0))
            if len(ranked) >= limit:
                break
    return ranked


def _prefetch_prepare_option_data(
    api_key: str,
    tickers: list[str],
    as_of: date | datetime,
    cache: PolygonCache,
    default_dte: int = 14,
    bar_minutes: int = 5,
    metrics_cache: dict[str, dict | None] | None = None,
    max_symbols: int = config.PREPARE_PREFETCH_SYMBOLS,
    contracts_per_side: int = config.PREPARE_PREFETCH_CONTRACTS_PER_SIDE,
    strike_band_pct: float = config.PREPARE_PREFETCH_STRIKE_BAND_PCT,
) -> int:
    """Warm a broader option universe for offline backtests.

    This is only used in prepare mode so the cold vendor cost is paid once,
    outside the actual model/backtest comparison loop.
    """
    if not tickers or max_symbols <= 0 or contracts_per_side <= 0:
        return 0

    as_of_dt = _coerce_eastern_datetime(as_of)
    trade_date = as_of_dt.date()
    expiry_gte = trade_date + timedelta(days=max(config.PREFERRED_DTE_MIN, default_dte - 10))
    expiry_lte = trade_date + timedelta(days=min(config.PREFERRED_DTE_MAX, default_dte + 21))
    prefetched: set[str] = set()

    for ticker in tickers[:max_symbols]:
        _, spot = _build_ticker_price_context(
            api_key,
            ticker,
            as_of_dt,
            cache=cache,
            bar_minutes=bar_minutes,
            metrics_cache=metrics_cache,
        )
        if spot <= 0:
            continue

        strike_gte = round(spot * max(0.05, 1.0 - strike_band_pct), 2)
        strike_lte = round(spot * (1.0 + strike_band_pct), 2)
        for option_type in ("call", "put"):
            contracts = fetch_polygon_option_contracts(
                api_key,
                ticker,
                option_type,
                expiry_gte,
                expiry_lte,
                strike_gte,
                strike_lte,
                as_of=trade_date,
                cache=cache,
            )
            ranked_contracts = _rank_prefetch_contracts(
                contracts,
                spot=spot,
                trade_date=trade_date,
                default_dte=default_dte,
                option_type=option_type,
                limit=contracts_per_side,
            )
            for contract in ranked_contracts:
                option_ticker = str(contract.get("ticker") or "")
                if not option_ticker or option_ticker in prefetched:
                    continue
                fetch_historical_intraday_bars(
                    api_key,
                    option_ticker,
                    trade_date,
                    multiplier=bar_minutes,
                    cache=cache,
                )
                fetch_option_daily_bar(
                    api_key,
                    option_ticker,
                    trade_date,
                    cache=cache,
                )
                prefetched.add(option_ticker)

    return len(prefetched)


def _warm_prepare_option_metadata(
    api_key: str,
    tickers: list[str],
    as_of: date | datetime,
    cache: PolygonCache,
    default_dte: int = 14,
    bar_minutes: int = 5,
    metrics_cache: dict[str, dict | None] | None = None,
    max_symbols: int = config.PREPARE_PREFETCH_SYMBOLS,
    strike_band_pct: float = config.PREPARE_PREFETCH_STRIKE_BAND_PCT,
) -> int:
    """Warm option chain metadata without hydrating contract bars.

    Prepare mode should accelerate later runs, not define the option surface
    the model is allowed to reason over.
    """
    if not tickers or max_symbols <= 0:
        return 0

    as_of_dt = _coerce_eastern_datetime(as_of)
    trade_date = as_of_dt.date()
    expiry_gte = trade_date + timedelta(days=max(config.PREFERRED_DTE_MIN, default_dte - 10))
    expiry_lte = trade_date + timedelta(days=min(config.PREFERRED_DTE_MAX, default_dte + 21))
    warmed_contracts: set[str] = set()

    for ticker in tickers[:max_symbols]:
        _, spot = _build_ticker_price_context(
            api_key,
            ticker,
            as_of_dt,
            cache=cache,
            bar_minutes=bar_minutes,
            metrics_cache=metrics_cache,
        )
        if spot <= 0:
            continue

        strike_gte = round(spot * max(0.05, 1.0 - strike_band_pct), 2)
        strike_lte = round(spot * (1.0 + strike_band_pct), 2)
        for option_type in ("call", "put"):
            contracts = fetch_polygon_option_contracts(
                api_key,
                ticker,
                option_type,
                expiry_gte,
                expiry_lte,
                strike_gte,
                strike_lte,
                as_of=trade_date,
                cache=cache,
            )
            for contract in contracts:
                option_ticker = str(contract.get("ticker") or "")
                if option_ticker:
                    warmed_contracts.add(option_ticker)

    return len(warmed_contracts)


def _build_enriched_portfolio_context(
    equity: float,
    initial_equity: float,
    positions: list[SimPosition],
    trade_date: date | datetime,
    api_key: str,
    cache: PolygonCache,
    profit_target_pct: float,
    stop_loss_pct: float,
    time_stop_dte: int,
    bar_minutes: int = 5,
) -> str:
    """Build mark-to-market portfolio context with unrealized P&L.

    Uses the latest option bar available before as_of.
    """
    as_of_dt = _coerce_eastern_datetime(trade_date)
    trade_day = as_of_dt.date()
    ret_pct = (equity - initial_equity) / initial_equity * 100 if initial_equity > 0 else 0
    lines = [
        f"Account Equity: ${equity:,.2f} (starting: ${initial_equity:,.2f}, return: {ret_pct:+.1f}%)",
        f"As of: {as_of_dt.strftime('%Y-%m-%d %H:%M %Z')}",
        f"Open Positions: {len(positions)}",
    ]

    total_unrealized = 0.0
    for pos in positions:
        if pos.option_type == "stock":
            line = f"  {pos.underlying} stock"
            lines.append(line)
            bar = _current_equity_bar(
                api_key,
                pos.underlying,
                trade_date,
                cache=cache,
                bar_minutes=bar_minutes,
            )
            current_price = _equity_bar_price(bar) if bar else 0.0
            if current_price > 0 and pos.entry_premium > 0:
                pnl_pct = (current_price - pos.entry_premium) / pos.entry_premium * 100
                pnl_dollar = (current_price - pos.entry_premium) * pos.qty
                total_unrealized += pnl_dollar
                lines.append(
                    f"    entry=${pos.entry_premium:.2f} current=${current_price:.2f} "
                    f"unrealized={pnl_pct:+.1f}% (${pnl_dollar:+,.2f}) qty={pos.qty}"
                )
            else:
                lines.append(f"    entry=${pos.entry_premium:.2f} qty={pos.qty}")
            if pos.risk_alert:
                lines.append(f"    ** RISK ALERT: {pos.risk_alert}")
            continue

        dte = (pos.expiry_date - trade_day).days
        line = (
            f"  {pos.polygon_ticker or pos.underlying} {pos.option_type} "
            f"${pos.strike:.2f} exp={pos.expiry_date} DTE={dte}"
        )
        lines.append(line)

        # Try to get current price (usually cached from exit check)
        current_premium = 0.0
        if pos.polygon_ticker:
            opt_bar = _current_option_bar(
                api_key, pos.polygon_ticker, trade_date, cache=cache, bar_minutes=bar_minutes,
            )
            if opt_bar:
                current_premium = _option_bar_price(opt_bar)

        if current_premium > 0 and pos.entry_premium > 0:
            pnl_pct = (current_premium - pos.entry_premium) / pos.entry_premium * 100
            pnl_dollar = (current_premium - pos.entry_premium) * pos.qty * 100
            total_unrealized += pnl_dollar
            lines.append(
                f"    entry=${pos.entry_premium:.2f} current=${current_premium:.2f} "
                f"unrealized={pnl_pct:+.1f}% (${pnl_dollar:+,.2f}) qty={pos.qty}"
            )
            if dte > 0:
                daily_decay = current_premium / dte
                total_daily = daily_decay * pos.qty * 100
                lines.append(f"    time decay ≈${total_daily:.0f}/day")
            # Flag proximity to exit triggers
            flags = []
            if pnl_pct >= profit_target_pct * 100 * 0.8:
                flags.append(f"approaching profit target of {profit_target_pct:.0%}")
            scaled_stop_loss_pct = stop_loss_for_dte(dte)
            if pnl_pct <= -scaled_stop_loss_pct * 100 * 0.8:
                flags.append(
                    f"approaching stop loss of {scaled_stop_loss_pct:.0%} ({dte} DTE)"
                )
            if dte <= time_stop_dte + 1:
                flags.append(f"approaching time stop ({time_stop_dte} DTE)")
            if flags:
                lines.append(f"    [{'; '.join(flags)}]")
            if pos.risk_alert:
                lines.append(f"    ** RISK ALERT: {pos.risk_alert}")
        else:
            lines.append(
                f"    entry=${pos.entry_premium:.2f} qty={pos.qty}"
            )
            if pos.risk_alert:
                lines.append(f"    ** RISK ALERT: {pos.risk_alert}")

    if positions:
        lines.append(f"Total unrealized P&L: ${total_unrealized:+,.2f}")

    return "\n".join(lines)


def _historical_cache_path(bt_config: BacktestConfig) -> Path:
    return Path(bt_config.cache_db_path or DEFAULT_HISTORICAL_CACHE_DB_PATH)


def _simulated_position_market_value(
    pos: SimPosition,
    as_of: date | datetime,
    api_key: str,
    cache: PolygonCache,
    *,
    bar_minutes: int = 5,
) -> float:
    premium = pos.entry_premium
    multiplier = 100
    if pos.option_type == "stock":
        multiplier = 1
        bar = _current_equity_bar(
            api_key,
            pos.underlying,
            as_of,
            cache=cache,
            bar_minutes=bar_minutes,
        )
        observed_price = _equity_bar_price(bar) if bar else 0.0
        if observed_price > 0:
            premium = observed_price
        return max(premium, 0.0) * pos.qty * multiplier
    if pos.polygon_ticker:
        option_bar = _current_option_bar(
            api_key,
            pos.polygon_ticker,
            as_of,
            cache=cache,
            bar_minutes=bar_minutes,
        )
        observed_premium = _option_bar_price(option_bar) if option_bar else 0.0
        if observed_premium > 0:
            premium = observed_premium
    return max(premium, 0.0) * pos.qty * multiplier


def _simulated_current_exposure(
    positions: list[SimPosition],
    as_of: date | datetime,
    api_key: str,
    cache: PolygonCache,
    *,
    bar_minutes: int = 5,
) -> float:
    return sum(
        _simulated_position_market_value(
            pos,
            as_of,
            api_key,
            cache,
            bar_minutes=bar_minutes,
        )
        for pos in positions
    )


def run_backtest(bt_config: BacktestConfig) -> BacktestResult:
    """Run the full backtest using real historical options data."""
    llm_provider = ""
    historical_options_provider = _historical_options_provider()
    brain: TradingBrain | None = None
    if not bt_config.prepare_only:
        llm_model = config.resolved_llm_model()
        llm_provider = infer_provider(
            model=llm_model,
            provider=os.environ.get("LLM_PROVIDER") or config.LLM_PROVIDER,
        )
        llm_api_key = resolve_api_key(llm_provider)
        if not llm_api_key:
            raise ValueError(f"{api_key_env_name(llm_provider)} required for backtesting")
        brain = TradingBrain(
            api_key=llm_api_key,
            provider=llm_provider,
            model=llm_model,
        )
    polygon_key = os.environ.get("POLYGON_API_KEY") or ""
    if not polygon_key and not bt_config.offline:
        raise ValueError("POLYGON_API_KEY required for backtesting")
    journal = ThesisJournal(
        db_path=bt_config.log_db_path if bt_config.log_db_path and not bt_config.prepare_only else None,
        max_active=bt_config.journal_max_active,
        max_full_display=bt_config.journal_max_full_display,
        stale_cycles=bt_config.journal_stale_cycles,
        stale_conviction=bt_config.journal_stale_conviction,
    ) if bt_config.use_journal else None
    trading_days = _trading_days(bt_config.start_date, bt_config.end_date)

    result = BacktestResult(
        initial_equity=bt_config.initial_equity,
        historical_options_provider=historical_options_provider,
    )
    result.log_db_path = str(bt_config.log_db_path) if bt_config.log_db_path else None
    equity = bt_config.initial_equity
    positions: list[SimPosition] = []
    closed_trades: list[dict] = []
    peak_equity = equity
    max_dd = 0.0
    daily_returns: list[float] = []
    llm_failure_dates: set[date] = set()
    consecutive_llm_errors = 0

    # Polygon cache — avoids redundant API calls in-process and across runs.
    cache = PolygonCache(
        store=PolygonResponseStore(_historical_cache_path(bt_config)),
        offline=bt_config.offline,
    )
    logger = AITradeLogger(bt_config.log_db_path) if bt_config.log_db_path and not bt_config.prepare_only else None

    log(f"backtest: {bt_config.start_date} to {bt_config.end_date} ({len(trading_days)} days)")
    log(f"initial equity: ${equity:,.2f}")
    log(f"historical options provider: {historical_options_provider}")
    if bt_config.prepare_only:
        log(f"mode: prepare-only cache warm ({cache.store.db_path})")
    elif bt_config.offline:
        log(f"mode: offline cache replay ({cache.store.db_path})")
    prior_day_equity = bt_config.initial_equity
    for day_idx, trade_date in enumerate(trading_days):
        log(
            f"\n=== {trade_date} (day {day_idx + 1}/{len(trading_days)}) "
            f"realized_equity=${equity:,.2f} ==="
        )
        decision_times = _decision_timestamps_for_day(
            trade_date,
            interval_minutes=bt_config.decision_interval_minutes,
            start_delay_minutes=bt_config.no_trade_minutes_after_open,
            end_buffer_minutes=bt_config.no_trade_minutes_before_close,
        )

        for decision_time in decision_times:
            remaining: list[SimPosition] = []
            for pos in positions:
                pos = replace(pos, risk_alert="")
                if pos.option_type == "stock":
                    remaining.append(pos)
                    continue
                dte = (pos.expiry_date - decision_time.date()).days
                if not pos.polygon_ticker:
                    remaining.append(pos)
                    continue

                option_bar = _current_option_bar(
                    polygon_key,
                    pos.polygon_ticker,
                    decision_time,
                    cache=cache,
                    bar_minutes=bt_config.signal_bar_minutes,
                )
                current_premium = _exit_premium_for_position(pos, decision_time, option_bar)
                if current_premium is None:
                    remaining.append(pos)
                    continue

                exit_reason = None
                if decision_time.date() >= pos.expiry_date:
                    exit_reason = "expiry"
                else:
                    pnl_pct = (
                        (current_premium - pos.entry_premium) / pos.entry_premium
                        if pos.entry_premium > 0
                        else 0.0
                    )
                    if pnl_pct <= -config.CATASTROPHIC_STOP_PCT:
                        exit_reason = "catastrophic_stop"
                    else:
                        risk_state = evaluate_position_risk(
                            entry_premium=pos.entry_premium,
                            current_premium=current_premium,
                            dte=dte,
                        )
                        if risk_state.should_close:
                            reason_text = risk_state.reason.lower()
                            if reason_text.startswith("profit target"):
                                exit_reason = "profit_target"
                            elif reason_text.startswith("stop loss"):
                                exit_reason = "stop_loss"
                            elif reason_text.startswith("time stop"):
                                exit_reason = "time_stop"
                            else:
                                exit_reason = "risk_exit"

                if exit_reason:
                    trade_pnl = (current_premium - pos.entry_premium) * pos.qty * 100
                    equity += trade_pnl
                    result.trades.append(
                        SimTrade(
                            entry_date=pos.entry_date,
                            exit_date=decision_time.date(),
                            underlying=pos.underlying,
                            option_type=pos.option_type,
                            strike=pos.strike,
                            entry_premium=pos.entry_premium,
                            exit_premium=current_premium,
                            qty=pos.qty,
                            pnl=trade_pnl,
                            exit_reason=exit_reason,
                            conviction=pos.conviction,
                            expression_profile=pos.expression_profile,
                            polygon_ticker=pos.polygon_ticker,
                            reasoning=pos.reasoning,
                        )
                    )
                    closed_trades.append({
                        "timestamp": decision_time.isoformat(),
                        "entry_date": pos.entry_date.isoformat(),
                        "symbol": pos.polygon_ticker,
                        "underlying": pos.underlying,
                        "option_type": pos.option_type,
                        "entry_premium": pos.entry_premium,
                        "exit_premium": current_premium,
                        "pnl": trade_pnl,
                        "reason": exit_reason,
                        "conviction": pos.conviction,
                        "expression_profile": pos.expression_profile,
                        "polygon_ticker": pos.polygon_ticker,
                    })
                    log(
                        f"  {decision_time.strftime('%H:%M')} AUTO EXIT {pos.polygon_ticker} "
                        f"reason={exit_reason} pnl=${trade_pnl:+,.2f}"
                    )
                    _log_backtest_close(
                        logger,
                        decision_time=decision_time,
                        position=pos,
                        exit_premium=current_premium,
                        trade_pnl=trade_pnl,
                        reason=exit_reason,
                    )
                else:
                    remaining.append(pos)
            positions = remaining

            account_equity, _ = _mark_to_market_equity(
                equity,
                positions,
                decision_time,
                polygon_key,
                cache,
                bar_minutes=bt_config.signal_bar_minutes,
            )

            news_window_start = decision_time - timedelta(hours=bt_config.news_lookback_hours)
            news_raw = fetch_historical_news_window(
                polygon_key,
                news_window_start,
                decision_time,
                limit=50,
                cache=cache,
            )
            news = _filter_news_quality(news_raw)
            log(
                f"  {decision_time.strftime('%H:%M')}: "
                f"news {len(news_raw)} raw → {len(news)}"
            )
            context_focus_symbols = [p.underlying for p in positions]
            if journal:
                context_focus_symbols.extend(entry.underlying for entry in journal.active_entries())
            news_items = _news_items_from_backtest_articles(
                news,
                reference_time=decision_time,
            )
            news_events = build_news_events(news_items, reference_time=decision_time)
            metrics_cache: dict[str, dict | None] = {}
            candidate_context, finalist_symbols = _build_focus_tickers(
                news_events,
                positions,
                journal,
                polygon_key,
                decision_time,
                cache,
                metrics_cache=metrics_cache,
            )
            deep_focus_symbols = list(dict.fromkeys(context_focus_symbols + finalist_symbols))
            news_context = _format_news_for_backtest(
                news,
                focus_symbols=deep_focus_symbols,
                reference_time=decision_time,
            )
            catalyst_reaction_context = _build_catalyst_reaction_context(
                polygon_key,
                news_events,
                decision_time,
                focus_symbols=deep_focus_symbols,
                cache=cache,
                bar_minutes=bt_config.signal_bar_minutes,
                metrics_cache=metrics_cache,
            )
            if catalyst_reaction_context:
                news_context = f"{catalyst_reaction_context}\n\n{news_context}"

            market_context = _build_market_trend_context(
                polygon_key,
                decision_time,
                cache=cache,
                bar_minutes=bt_config.signal_bar_minutes,
                metrics_cache=metrics_cache,
            )
            portfolio_context = _build_enriched_portfolio_context(
                account_equity,
                bt_config.initial_equity,
                positions,
                decision_time,
                polygon_key,
                cache,
                bt_config.profit_target_pct,
                bt_config.stop_loss_pct,
                bt_config.time_stop_dte,
                bar_minutes=bt_config.signal_bar_minutes,
            )

            journal_context = journal.to_context_str() if journal else ""
            trade_history_context = ""
            annotated_trades = list(closed_trades)
            if journal and closed_trades:
                annotated_trades = _annotate_closed_trades(closed_trades, polygon_key, cache)
                trade_history_context = format_trade_history([], annotated_trades[-10:])
            perf_summary = _build_performance_summary(
                closed_trades,
                account_equity,
                bt_config.initial_equity,
            )
            if perf_summary:
                trade_history_context += perf_summary

            position_symbols = [p.underlying.upper() for p in positions if p.underlying]
            thesis_symbols = [
                entry.underlying.upper()
                for entry in (journal.active_entries() if journal else [])
                if entry.underlying
            ]
            options_watchlist = prioritized_symbol_watchlist(
                position_symbols,
                thesis_symbols,
                finalist_symbols,
                limit=min(
                    config.WATCHLIST_SIZE,
                    len(position_symbols) + len(thesis_symbols) + config.CANDIDATE_FINALISTS,
                ),
            )

            if bt_config.prepare_only:
                metadata_symbols = prioritized_symbol_watchlist(
                    options_watchlist or deep_focus_symbols,
                    deep_focus_symbols,
                    limit=max(bt_config.prepare_prefetch_symbols, len(options_watchlist or deep_focus_symbols)),
                )
                cache_entries_before = cache.store.entry_count() if cache.store else 0
                warmed_contract_metadata = _warm_prepare_option_metadata(
                    polygon_key,
                    metadata_symbols,
                    decision_time,
                    cache,
                    default_dte=bt_config.default_dte,
                    bar_minutes=bt_config.signal_bar_minutes,
                    metrics_cache=metrics_cache,
                    max_symbols=bt_config.prepare_prefetch_symbols,
                    strike_band_pct=bt_config.prepare_prefetch_strike_band_pct,
                )
                prefetched_option_bars = _prefetch_prepare_option_data(
                    polygon_key,
                    metadata_symbols,
                    decision_time,
                    cache,
                    default_dte=bt_config.default_dte,
                    bar_minutes=bt_config.signal_bar_minutes,
                    metrics_cache=metrics_cache,
                    max_symbols=bt_config.prepare_prefetch_symbols,
                    contracts_per_side=bt_config.prepare_prefetch_contracts_per_side,
                    strike_band_pct=bt_config.prepare_prefetch_strike_band_pct,
                )
                cache_entries = cache.store.entry_count() if cache.store else 0
                cache_entries_added = max(cache_entries - cache_entries_before, 0)
                result.decision_log.append({
                    "date": trade_date.isoformat(),
                    "decision_time": decision_time.isoformat(),
                    "news_window_start": news_window_start.isoformat(),
                    "prepared_only": True,
                    "historical_options_provider": historical_options_provider,
                    "finalists": finalist_symbols,
                    "options_watchlist": options_watchlist or deep_focus_symbols,
                    "warmed_option_contract_metadata": warmed_contract_metadata,
                    "prefetched_option_contract_bars": prefetched_option_bars,
                    "cache_entries": cache_entries,
                    "cache_entries_added": cache_entries_added,
                    "open_positions": len(positions),
                })
                log(
                    f"  {decision_time.strftime('%H:%M')} PREPARED "
                    f"finalists={len(finalist_symbols)} "
                    f"warmed_contract_metadata={warmed_contract_metadata} "
                    f"prefetched_option_bars={prefetched_option_bars} "
                    f"cache_entries={cache_entries} (+{cache_entries_added})"
                )
                continue

            options_context = _build_options_context(
                polygon_key,
                options_watchlist or deep_focus_symbols,
                decision_time,
                cache,
                best_events=map_best_events_by_symbol(news_events),
                closed_trades=annotated_trades,
                default_dte=bt_config.default_dte,
                bar_minutes=bt_config.signal_bar_minutes,
                metrics_cache=metrics_cache,
            )

            trade_results: list[dict] = []
            if bt_config.llm_delay_seconds > 0:
                time_module.sleep(bt_config.llm_delay_seconds)

            run_result = brain.run(
                portfolio_context=portfolio_context,
                candidate_context=candidate_context,
                news_context=news_context,
                market_context=market_context,
                options_context=options_context,
                journal_context=journal_context,
                trade_history_context=trade_history_context,
            )
            analysis = run_result.analysis
            llm_error = _is_llm_error_message(analysis.analysis)
            if llm_error:
                result.llm_error_cycles += 1
                llm_failure_dates.add(trade_date)
                consecutive_llm_errors += 1
            else:
                consecutive_llm_errors = 0
            log(f"  {decision_time.strftime('%H:%M')} LLM: {analysis.analysis}")

            if journal and analysis.thesis_updates:
                journal.apply_updates(analysis.thesis_updates)
                log(
                    f"  Journal: {len(analysis.thesis_updates)} updates, "
                    f"{len(journal.active_entries())} active"
                )

            decision_id = 0
            if logger is not None:
                decision_id = logger.log_decision(
                    AIDecisionRecord(
                        timestamp=decision_time,
                        market_analysis=analysis.analysis,
                        news_summary=news_context[:1000],
                        portfolio_state=portfolio_context[:1000],
                        decisions_json=_analysis_to_decisions_json(
                            analysis,
                            run_result.diagnostics,
                        ),
                        trades_executed=0,
                        llm_provider=run_result.packet.provider,
                        llm_model=run_result.packet.model,
                        packet_json=json.dumps(run_result.packet.to_payload()),
                        response_json=json.dumps(run_result.completion.to_payload()),
                    )
                )

            trades_executed = 0
            for decision in analysis.trades:
                trade_record = {
                    "underlying": decision.underlying,
                    "action": decision.action,
                    "conviction": decision.conviction,
                    "risk_pct": decision.risk_pct,
                    "reasoning": decision.reasoning,
                    "status": "skipped",
                    "skip_reason": "",
                    "contract": "",
                    "qty": 0,
                    "premium": 0.0,
                }

                if decision.action == "close_position":
                    target = decision.target_symbol
                    matched_pos = None
                    if target:
                        matched_pos = next(
                            (
                                p for p in positions
                                if p.option_type != "stock" and p.polygon_ticker == target
                            ),
                            None,
                        )
                    if matched_pos is None:
                        matched_pos = next(
                            (
                                p for p in positions
                                if p.option_type != "stock"
                                and p.underlying.upper() == decision.underlying.upper()
                            ),
                            None,
                        )
                    if matched_pos is None:
                        trade_record["skip_reason"] = "no matching position"
                        trade_results.append(trade_record)
                        continue

                    exit_bar = _current_option_bar(
                        polygon_key,
                        matched_pos.polygon_ticker,
                        decision_time,
                        cache,
                        bar_minutes=bt_config.signal_bar_minutes,
                    )
                    if exit_bar is None:
                        trade_record["skip_reason"] = "no price data before close"
                        trade_results.append(trade_record)
                        continue

                    exit_premium = _option_bar_exit_price(exit_bar)
                    if exit_premium <= 0:
                        trade_record["skip_reason"] = "invalid exit price"
                        trade_results.append(trade_record)
                        continue

                    trade_pnl = (exit_premium - matched_pos.entry_premium) * matched_pos.qty * 100
                    equity += trade_pnl
                    result.trades.append(
                        SimTrade(
                            entry_date=matched_pos.entry_date,
                            exit_date=decision_time.date(),
                            underlying=matched_pos.underlying,
                            option_type=matched_pos.option_type,
                            strike=matched_pos.strike,
                            entry_premium=matched_pos.entry_premium,
                            exit_premium=exit_premium,
                            qty=matched_pos.qty,
                            pnl=trade_pnl,
                            exit_reason="manual_close",
                            conviction=matched_pos.conviction,
                            expression_profile=matched_pos.expression_profile,
                            polygon_ticker=matched_pos.polygon_ticker,
                            reasoning=matched_pos.reasoning,
                        )
                    )
                    closed_trades.append({
                        "timestamp": decision_time.isoformat(),
                        "entry_date": matched_pos.entry_date.isoformat(),
                        "symbol": matched_pos.polygon_ticker,
                        "underlying": matched_pos.underlying,
                        "option_type": matched_pos.option_type,
                        "entry_premium": matched_pos.entry_premium,
                        "exit_premium": exit_premium,
                        "pnl": trade_pnl,
                        "reason": "manual_close",
                        "conviction": matched_pos.conviction,
                        "expression_profile": matched_pos.expression_profile,
                        "polygon_ticker": matched_pos.polygon_ticker,
                    })
                    positions.remove(matched_pos)
                    trade_record["status"] = "executed"
                    trade_record["contract"] = matched_pos.polygon_ticker
                    trade_record["qty"] = matched_pos.qty
                    trade_record["premium"] = exit_premium
                    trade_record["pnl"] = trade_pnl
                    trade_results.append(trade_record)
                    trades_executed += 1
                    _log_backtest_close(
                        logger,
                        decision_time=decision_time,
                        position=matched_pos,
                        exit_premium=exit_premium,
                        trade_pnl=trade_pnl,
                        reason="manual_close",
                    )
                    account_equity, _ = _mark_to_market_equity(
                        equity,
                        positions,
                        decision_time,
                        polygon_key,
                        cache,
                        bar_minutes=bt_config.signal_bar_minutes,
                    )
                    continue

                if decision.action == "close_stock":
                    target = (decision.target_symbol or decision.underlying).upper()
                    matched_pos = next(
                        (
                            p for p in positions
                            if p.option_type == "stock"
                            and (p.underlying.upper() == target or p.polygon_ticker.upper() == target)
                        ),
                        None,
                    )
                    if matched_pos is None:
                        trade_record["skip_reason"] = "no matching stock position"
                        trade_results.append(trade_record)
                        continue

                    exit_price = _current_underlying_price(
                        polygon_key,
                        matched_pos.underlying,
                        decision_time,
                        cache,
                        bar_minutes=bt_config.signal_bar_minutes,
                    )
                    if exit_price <= 0:
                        trade_record["skip_reason"] = "no stock price data before close"
                        trade_results.append(trade_record)
                        continue

                    trade_pnl = (exit_price - matched_pos.entry_premium) * matched_pos.qty
                    equity += trade_pnl
                    result.trades.append(
                        SimTrade(
                            entry_date=matched_pos.entry_date,
                            exit_date=decision_time.date(),
                            underlying=matched_pos.underlying,
                            option_type="stock",
                            strike=0.0,
                            entry_premium=matched_pos.entry_premium,
                            exit_premium=exit_price,
                            qty=matched_pos.qty,
                            pnl=trade_pnl,
                            exit_reason="manual_close",
                            conviction=matched_pos.conviction,
                            expression_profile=matched_pos.expression_profile,
                            polygon_ticker=matched_pos.polygon_ticker,
                            reasoning=matched_pos.reasoning,
                        )
                    )
                    closed_trades.append({
                        "timestamp": decision_time.isoformat(),
                        "entry_date": matched_pos.entry_date.isoformat(),
                        "symbol": matched_pos.polygon_ticker,
                        "underlying": matched_pos.underlying,
                        "option_type": "stock",
                        "entry_premium": matched_pos.entry_premium,
                        "exit_premium": exit_price,
                        "pnl": trade_pnl,
                        "reason": "manual_close",
                        "conviction": matched_pos.conviction,
                        "expression_profile": matched_pos.expression_profile,
                        "polygon_ticker": matched_pos.polygon_ticker,
                    })
                    positions.remove(matched_pos)
                    trade_record["status"] = "executed"
                    trade_record["contract"] = matched_pos.polygon_ticker
                    trade_record["qty"] = matched_pos.qty
                    trade_record["premium"] = exit_price
                    trade_record["pnl"] = trade_pnl
                    trade_results.append(trade_record)
                    trades_executed += 1
                    _log_backtest_close(
                        logger,
                        decision_time=decision_time,
                        position=matched_pos,
                        exit_premium=exit_price,
                        trade_pnl=trade_pnl,
                        reason="manual_close",
                    )
                    account_equity, _ = _mark_to_market_equity(
                        equity,
                        positions,
                        decision_time,
                        polygon_key,
                        cache,
                        bar_minutes=bt_config.signal_bar_minutes,
                    )
                    continue

                if decision.action == "buy_stock":
                    if len(positions) >= bt_config.max_positions:
                        trade_record["skip_reason"] = "max positions reached"
                        trade_results.append(trade_record)
                        continue

                    underlying = decision.underlying
                    entry_price = _current_underlying_price(
                        polygon_key,
                        underlying,
                        decision_time,
                        cache,
                        bar_minutes=bt_config.signal_bar_minutes,
                    )
                    if entry_price <= 0:
                        trade_record["skip_reason"] = "no stock price data before decision time"
                        trade_results.append(trade_record)
                        continue

                    requested_risk_pct = min(
                        scale_risk_pct_for_conviction(decision.risk_pct, decision.conviction),
                        bt_config.max_risk_per_trade,
                    )
                    current_exposure = _simulated_current_exposure(
                        positions,
                        decision_time,
                        polygon_key,
                        cache,
                        bar_minutes=bt_config.signal_bar_minutes,
                    )
                    cash = max(account_equity - current_exposure, 0.0)
                    day_pl = account_equity - prior_day_equity
                    risk_check = evaluate_stock_trade_risk(
                        equity=account_equity,
                        cash=cash,
                        current_exposure=current_exposure,
                        open_positions=len(positions),
                        share_price=entry_price,
                        day_pl=day_pl,
                    )
                    if not risk_check.approved:
                        trade_record["skip_reason"] = f"risk: {risk_check.reason}"
                        trade_results.append(trade_record)
                        continue

                    requested_budget = account_equity * requested_risk_pct
                    existing_notional = _stock_symbol_notional_for_budget(
                        positions,
                        underlying,
                        entry_price,
                    )
                    remaining_symbol_budget = max(requested_budget - existing_notional, 0.0)
                    qty = min(
                        risk_check.max_shares,
                        size_for_risk_budget(remaining_symbol_budget, entry_price),
                    )
                    if qty <= 0:
                        trade_record["skip_reason"] = "stock risk budget too small for 1 share"
                        trade_results.append(trade_record)
                        continue

                    positions.append(
                        SimPosition(
                            underlying=underlying,
                            option_type="stock",
                            strike=0.0,
                            entry_date=trade_date,
                            expiry_date=bt_config.end_date,
                            entry_premium=entry_price,
                            qty=qty,
                            conviction=decision.conviction,
                            reasoning=decision.reasoning,
                            expression_profile=decision.expression_profile or "stock_proxy",
                            polygon_ticker=underlying,
                        )
                    )
                    trade_record["status"] = "executed"
                    trade_record["contract"] = underlying
                    trade_record["qty"] = qty
                    trade_record["premium"] = entry_price
                    trade_record["effective_risk_pct"] = requested_risk_pct
                    trade_record["expression_profile"] = decision.expression_profile or "stock_proxy"
                    trade_results.append(trade_record)
                    trades_executed += 1
                    _log_backtest_open_trade(
                        logger,
                        decision_time=decision_time,
                        decision=decision,
                        market_analysis=analysis.analysis,
                        option_type="stock",
                        strike=0.0,
                        expiry=None,
                        polygon_ticker=underlying,
                        qty=qty,
                        premium=entry_price,
                    )
                    account_equity, _ = _mark_to_market_equity(
                        equity,
                        positions,
                        decision_time,
                        polygon_key,
                        cache,
                        bar_minutes=bt_config.signal_bar_minutes,
                    )
                    continue

                if decision.action not in ("buy_call", "buy_put"):
                    trade_record["skip_reason"] = f"unsupported action in backtest: {decision.action}"
                    trade_results.append(trade_record)
                    continue

                if len(positions) >= bt_config.max_positions:
                    trade_record["skip_reason"] = "max positions reached"
                    trade_results.append(trade_record)
                    continue

                guard_reason = _backtest_option_loss_streak_guard_reason(
                    closed_trades,
                    decision.conviction,
                )
                if guard_reason:
                    trade_record["skip_reason"] = guard_reason
                    trade_results.append(trade_record)
                    continue

                option_type = "call" if decision.action == "buy_call" else "put"
                underlying = decision.underlying
                spot = _current_underlying_price(
                    polygon_key,
                    underlying,
                    decision_time,
                    cache,
                    bar_minutes=bt_config.signal_bar_minutes,
                )
                if spot <= 0:
                    trade_record["skip_reason"] = "no price data before decision time"
                    trade_results.append(trade_record)
                    continue

                contract_skip_reasons: list[str] = []
                contract = _select_real_contract(
                    polygon_key,
                    underlying,
                    option_type,
                    spot,
                    trade_date,
                    decision.strike_preference or "atm",
                    decision.expiry_preference or "next_week",
                    bt_config.default_dte,
                    decision_time=decision_time,
                    expression_profile=decision.expression_profile,
                    contract_symbol=decision.contract_symbol,
                    target_delta_range=decision.target_delta_range,
                    target_dte_range=decision.target_dte_range,
                    max_spread_pct=decision.max_spread_pct,
                    cache=cache,
                    bar_minutes=bt_config.signal_bar_minutes,
                    reason_out=contract_skip_reasons,
                )
                if contract is None:
                    trade_record["skip_reason"] = (
                        contract_skip_reasons[-1]
                        if contract_skip_reasons else "no contract found"
                    )
                    trade_results.append(trade_record)
                    continue

                polygon_ticker = contract.get("ticker", "")
                strike = float(contract.get("strike_price", 0))
                expiry_str = contract.get("expiration_date", "")
                if not polygon_ticker or strike <= 0 or not expiry_str:
                    trade_record["skip_reason"] = "invalid contract data"
                    trade_results.append(trade_record)
                    continue
                expiry = date.fromisoformat(expiry_str)
                quote_bar = _current_option_bar(
                    polygon_key,
                    polygon_ticker,
                    decision_time,
                    cache,
                    bar_minutes=bt_config.signal_bar_minutes,
                )
                limit_price = _entry_limit_price(contract, quote_bar)
                if limit_price < 0.01:
                    trade_record["skip_reason"] = "no tradable quote"
                    trade_results.append(trade_record)
                    continue

                entry_bar = _next_fill_option_bar(
                    polygon_key,
                    polygon_ticker,
                    decision_time,
                    cache,
                    bar_minutes=bt_config.signal_bar_minutes,
                )
                if entry_bar is None:
                    trade_record["skip_reason"] = "no fill bar after decision time"
                    trade_results.append(trade_record)
                    continue

                volume = int(entry_bar.get("v", 0) or 0)
                if volume <= 0 and not _option_bar_quote_only(entry_bar):
                    trade_record["skip_reason"] = "zero volume"
                    trade_results.append(trade_record)
                    continue

                premium = _limit_buy_fill_price(limit_price, entry_bar)
                if premium is None:
                    trade_record["skip_reason"] = "entry order not filled"
                    trade_results.append(trade_record)
                    continue

                requested_risk_pct = min(
                    scale_risk_pct_for_conviction(decision.risk_pct, decision.conviction),
                    bt_config.max_risk_per_trade,
                )
                current_exposure = _simulated_current_exposure(
                    positions,
                    decision_time,
                    polygon_key,
                    cache,
                    bar_minutes=bt_config.signal_bar_minutes,
                )
                cash = max(account_equity - current_exposure, 0.0)
                day_pl = account_equity - prior_day_equity
                risk_check = evaluate_trade_risk(
                    equity=account_equity,
                    cash=cash,
                    current_exposure=current_exposure,
                    open_positions=len(positions),
                    option_ask=limit_price,
                    day_pl=day_pl,
                )
                if not risk_check.approved:
                    trade_record["skip_reason"] = f"risk: {risk_check.reason}"
                    trade_results.append(trade_record)
                    continue
                max_cost = account_equity * requested_risk_pct
                cost_per_contract = limit_price * 100
                max_by_risk = size_for_risk_budget(max_cost, cost_per_contract)
                qty = min(risk_check.max_contracts, max_by_risk)
                if qty <= 0:
                    trade_record["skip_reason"] = "risk budget too small for 1 contract"
                    trade_results.append(trade_record)
                    continue

                positions.append(
                    SimPosition(
                        underlying=underlying,
                        option_type=option_type,
                        strike=strike,
                        entry_date=trade_date,
                        expiry_date=expiry,
                        entry_premium=premium,
                        qty=qty,
                        conviction=decision.conviction,
                        reasoning=decision.reasoning,
                        expression_profile=decision.expression_profile or "",
                        polygon_ticker=polygon_ticker,
                    )
                )
                trade_record["status"] = "executed"
                trade_record["contract"] = polygon_ticker
                trade_record["qty"] = qty
                trade_record["premium"] = premium
                trade_record["effective_risk_pct"] = requested_risk_pct
                if decision.expression_profile:
                    trade_record["expression_profile"] = decision.expression_profile
                trade_results.append(trade_record)
                trades_executed += 1
                _log_backtest_open_trade(
                    logger,
                    decision_time=decision_time,
                    decision=decision,
                    market_analysis=analysis.analysis,
                    option_type=option_type,
                    strike=strike,
                    expiry=expiry,
                    polygon_ticker=polygon_ticker,
                    qty=qty,
                    premium=premium,
                )
                account_equity, _ = _mark_to_market_equity(
                    equity,
                    positions,
                    decision_time,
                    polygon_key,
                    cache,
                    bar_minutes=bt_config.signal_bar_minutes,
                )

            thesis_updates_serialized = [
                {
                    "id": tu.id,
                    "underlying": tu.underlying,
                    "direction": tu.direction,
                    "thesis": tu.thesis,
                    "conviction": tu.conviction,
                    "status": tu.status,
                    "new_observation": tu.new_observation,
                }
                for tu in analysis.thesis_updates
            ]
            marked_equity, _ = _mark_to_market_equity(
                equity,
                positions,
                decision_time,
                polygon_key,
                cache,
                bar_minutes=bt_config.signal_bar_minutes,
            )
            result.decision_log.append({
                "date": trade_date.isoformat(),
                "decision_time": decision_time.isoformat(),
                "news_window_start": news_window_start.isoformat(),
                "historical_options_provider": historical_options_provider,
                "llm_provider": run_result.packet.provider,
                "llm_model": run_result.packet.model,
                "llm_error": llm_error,
                "market_analysis": analysis.analysis,
                "finalists": finalist_symbols,
                "options_watchlist": options_watchlist or deep_focus_symbols,
                "thesis_updates": thesis_updates_serialized,
                "trades_proposed": trade_results,
                "dropped_trades": analysis.dropped_trades,
                "llm_diagnostics": run_result.diagnostics,
                "trades_executed": sum(1 for t in trade_results if t["status"] == "executed"),
                "trades_skipped": sum(1 for t in trade_results if t["status"] == "skipped"),
                "equity": marked_equity,
                "open_positions": len(positions),
            })
            if logger is not None and decision_id > 0:
                logger.update_decision_trade_count(decision_id, trades_executed)
            if (
                bt_config.max_consecutive_llm_errors > 0
                and consecutive_llm_errors >= bt_config.max_consecutive_llm_errors
            ):
                raise RuntimeError(
                    "aborting backtest after "
                    f"{consecutive_llm_errors} consecutive LLM error cycles"
                )

        day_close_equity, _ = _mark_to_market_equity(
            equity,
            positions,
            _market_close_dt(trade_date),
            polygon_key,
            cache,
            bar_minutes=bt_config.signal_bar_minutes,
        )
        day_return = day_close_equity - prior_day_equity
        prior_day_equity = day_close_equity
        daily_returns.append(day_return)
        result.equity_curve.append((trade_date.isoformat(), day_close_equity))

        if day_close_equity > peak_equity:
            peak_equity = day_close_equity
        dd = peak_equity - day_close_equity
        if dd > max_dd:
            max_dd = dd

    # Close any remaining positions at last day
    last_date = trading_days[-1] if trading_days else bt_config.end_date
    for pos in positions:
        exit_premium = 0.0
        if pos.option_type == "stock":
            close_time = _market_close_dt(last_date)
            exit_premium = _current_underlying_price(
                polygon_key,
                pos.underlying,
                close_time,
                cache,
                bar_minutes=bt_config.signal_bar_minutes,
            )
            if exit_premium <= 0:
                for lookback in range(1, 4):
                    check_date = last_date - timedelta(days=lookback)
                    bar = fetch_historical_daily_bar(
                        polygon_key,
                        pos.underlying,
                        check_date,
                        cache=cache,
                    )
                    if bar is not None:
                        exit_premium = _equity_bar_price(bar)
                        if exit_premium > 0:
                            break
            if exit_premium <= 0:
                exit_premium = pos.entry_premium
        elif pos.polygon_ticker:
            close_time = _market_close_dt(last_date)
            option_bar = _current_option_bar(
                polygon_key,
                pos.polygon_ticker,
                close_time,
                cache=cache,
                bar_minutes=bt_config.signal_bar_minutes,
            )
            if option_bar is not None:
                exit_premium = _option_bar_exit_price(option_bar)
            if exit_premium <= 0:
                for lookback in range(1, 4):
                    check_date = last_date - timedelta(days=lookback)
                    option_bar = fetch_option_daily_bar(
                        polygon_key, pos.polygon_ticker, check_date, cache=cache,
                    )
                    if option_bar is not None:
                        exit_premium = _option_bar_exit_price(option_bar)
                        if exit_premium > 0:
                            break

        multiplier = 1 if pos.option_type == "stock" else 100
        trade_pnl = (exit_premium - pos.entry_premium) * pos.qty * multiplier
        equity += trade_pnl
        result.trades.append(
            SimTrade(
                entry_date=pos.entry_date,
                exit_date=last_date,
                underlying=pos.underlying,
                option_type=pos.option_type,
                strike=pos.strike,
                entry_premium=pos.entry_premium,
                exit_premium=exit_premium,
                qty=pos.qty,
                pnl=trade_pnl,
                exit_reason="backtest_end",
                conviction=pos.conviction,
                expression_profile=pos.expression_profile,
                polygon_ticker=pos.polygon_ticker,
                reasoning=pos.reasoning,
            )
        )
        _log_backtest_close(
            logger,
            decision_time=_market_close_dt(last_date),
            position=pos,
            exit_premium=exit_premium,
            trade_pnl=trade_pnl,
            reason="backtest_end",
        )

    # Compute results
    result.final_equity = equity
    result.total_return_pct = (equity - bt_config.initial_equity) / bt_config.initial_equity
    result.total_trades = len(result.trades)
    result.wins = sum(1 for t in result.trades if t.pnl > 0)
    result.losses = sum(1 for t in result.trades if t.pnl < 0)
    result.win_rate = (
        result.wins / (result.wins + result.losses)
        if (result.wins + result.losses) > 0 else None
    )
    result.net_pnl = sum(t.pnl for t in result.trades)
    result.max_drawdown = max_dd
    result.days_tested = len(trading_days)

    gross_profit = sum(t.pnl for t in result.trades if t.pnl > 0)
    gross_loss = abs(sum(t.pnl for t in result.trades if t.pnl < 0))
    result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else None

    convictions = [t.conviction for t in result.trades]
    result.avg_conviction = sum(convictions) / len(convictions) if convictions else 0
    result.llm_failure_days = len(llm_failure_dates)

    # Sharpe
    if len(daily_returns) >= 2:
        mean_r = sum(daily_returns) / len(daily_returns)
        var_r = sum((r - mean_r) ** 2 for r in daily_returns) / (len(daily_returns) - 1)
        std_r = math.sqrt(var_r) if var_r > 0 else 0
        if std_r > 0:
            result.sharpe_ratio = round((mean_r / std_r) * math.sqrt(252), 2)

    return result


def prepare_backtest_data(bt_config: BacktestConfig) -> PrepareBacktestResult:
    """Warm the persistent historical cache without invoking the LLM."""
    prepare_config = replace(
        bt_config,
        llm_delay_seconds=0.0,
        prepare_only=True,
    )
    store = PolygonResponseStore(_historical_cache_path(prepare_config))
    cache_entries_before = store.entry_count()
    result = run_backtest(prepare_config)
    cache_entries = store.entry_count()
    return PrepareBacktestResult(
        start_date=prepare_config.start_date,
        end_date=prepare_config.end_date,
        days_prepared=result.days_tested,
        decision_points=len(result.decision_log),
        cache_db_path=store.db_path,
        cache_entries=cache_entries,
        cache_entries_before=cache_entries_before,
        cache_entries_added=max(cache_entries - cache_entries_before, 0),
        option_contracts_warmed=sum(
            int(entry.get("warmed_option_contract_metadata") or 0)
            for entry in result.decision_log
        ),
        option_contract_bars_prefetched=sum(
            int(entry.get("prefetched_option_contract_bars") or 0)
            for entry in result.decision_log
        ),
    )


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def _summarize_decision_log(decision_log: list[dict]) -> dict:
    proposed = executed = skipped = 0
    skip_counter: Counter[str] = Counter()
    dropped = 0
    drop_counter: Counter[str] = Counter()
    provider_retry_cycles = 0
    provider_retries = 0
    provider_counter: Counter[str] = Counter()

    for entry in decision_log:
        proposed_trades = entry.get("trades_proposed") or []
        if isinstance(proposed_trades, list):
            proposed += len(proposed_trades)
            for trade in proposed_trades:
                if not isinstance(trade, dict):
                    continue
                status = str(trade.get("status") or "")
                if status == "executed":
                    executed += 1
                elif status == "skipped":
                    skipped += 1
                    reason = str(trade.get("skip_reason") or "unknown")
                    skip_counter[reason] += 1
        else:
            entry_executed = int(entry.get("trades_executed") or 0)
            entry_skipped = int(entry.get("trades_skipped") or 0)
            executed += entry_executed
            skipped += entry_skipped
            proposed += entry_executed + entry_skipped
        dropped_trades = entry.get("dropped_trades") or []
        if isinstance(dropped_trades, list):
            dropped += len(dropped_trades)
            for trade in dropped_trades:
                if not isinstance(trade, dict):
                    continue
                reason = str(trade.get("reason") or "unknown")
                drop_counter[reason] += 1
        diagnostics = entry.get("llm_diagnostics") or {}
        if isinstance(diagnostics, dict):
            retries = int(diagnostics.get("retries") or 0)
            if retries > 0:
                provider_retry_cycles += 1
                provider_retries += retries
            events = diagnostics.get("events") or []
            if isinstance(events, list):
                for event in events:
                    if not isinstance(event, dict):
                        continue
                    reason = str(event.get("reason") or "unknown")
                    provider_counter[reason] += 1

    guardrail_skips = sum(
        count for reason, count in skip_counter.items()
        if "option entries require conviction" in reason
        or "closed option trades were losses" in reason
    )
    return {
        "proposed": proposed,
        "executed": executed,
        "skipped": skipped,
        "dropped": dropped,
        "provider_retry_cycles": provider_retry_cycles,
        "provider_retries": provider_retries,
        "guardrail_skips": guardrail_skips,
        "skip_reasons": [
            {"reason": reason, "count": count}
            for reason, count in skip_counter.most_common()
        ],
        "drop_reasons": [
            {"reason": reason, "count": count}
            for reason, count in drop_counter.most_common()
        ],
        "provider_retry_reasons": [
            {"reason": reason, "count": count}
            for reason, count in provider_counter.most_common()
        ],
    }


def print_backtest_result(r: BacktestResult) -> None:
    print("=" * 60)
    print("  AI TRADER BACKTEST RESULTS")
    print("=" * 60)

    print(f"\n  Period:           {r.days_tested} trading days")
    print(f"  Initial equity:   ${r.initial_equity:,.2f}")
    print(f"  Final equity:     ${r.final_equity:,.2f}")
    print(f"  Total return:     {r.total_return_pct:+.2%}")
    print(f"  Net P&L:          ${r.net_pnl:+,.2f}")

    print(f"\n  Total trades:     {r.total_trades}")
    print(f"  Wins:             {r.wins}")
    print(f"  Losses:           {r.losses}")
    if r.win_rate is not None:
        print(f"  Win rate:         {r.win_rate:.1%}")

    decision_summary = _summarize_decision_log(r.decision_log)
    if decision_summary["proposed"] or decision_summary["dropped"]:
        print(f"\n  Proposed trades:  {decision_summary['proposed']}")
        print(f"  Executed props:   {decision_summary['executed']}")
        print(f"  Skipped props:    {decision_summary['skipped']}")
        print(f"  Dropped model:    {decision_summary['dropped']}")
        if decision_summary["provider_retries"]:
            print(f"  LLM retries:      {decision_summary['provider_retries']}")
        if decision_summary["guardrail_skips"]:
            print(f"  Guardrail skips:  {decision_summary['guardrail_skips']}")
        top_reasons = decision_summary["skip_reasons"][:3]
        if top_reasons:
            print("  Top skip reasons:")
            for row in top_reasons:
                print(f"    {row['count']}x {row['reason']}")
        top_drop_reasons = decision_summary["drop_reasons"][:3]
        if top_drop_reasons:
            print("  Top drop reasons:")
            for row in top_drop_reasons:
                print(f"    {row['count']}x {row['reason']}")
        top_retry_reasons = decision_summary["provider_retry_reasons"][:3]
        if top_retry_reasons:
            print("  Top LLM retry reasons:")
            for row in top_retry_reasons:
                print(f"    {row['count']}x {row['reason']}")

    print(f"\n  Max drawdown:     ${r.max_drawdown:,.2f}")
    if r.sharpe_ratio is not None:
        print(f"  Sharpe ratio:     {r.sharpe_ratio:.2f}")
    if r.profit_factor is not None:
        print(f"  Profit factor:    {r.profit_factor:.2f}")
    print(f"  LLM failure days: {r.llm_failure_days}")
    print(f"  LLM error cycles: {r.llm_error_cycles}")
    if r.log_db_path:
        print(f"  Log DB:           {r.log_db_path}")
    print(f"  Avg conviction:   {r.avg_conviction:.2f}")

    if r.trades:
        print(f"\n--- Trade Log ---")
        for t in r.trades:
            ticker_str = f" [{t.polygon_ticker}]" if t.polygon_ticker else ""
            print(
                f"  {t.entry_date} -> {t.exit_date} | "
                f"{t.underlying} {t.option_type} ${t.strike:.0f}{ticker_str} | "
                f"entry=${t.entry_premium:.2f} exit=${t.exit_premium:.2f} | "
                f"qty={t.qty} pnl=${t.pnl:+,.2f} | "
                f"{t.exit_reason} (conv={t.conviction:.2f})"
                + (f" | {t.reasoning[:80]}" if t.reasoning else "")
            )

    print("=" * 60)


def print_prepare_result(r: PrepareBacktestResult) -> None:
    print("=" * 60)
    print("  AI TRADER BACKTEST DATA PREP")
    print("=" * 60)
    print(f"\n  Period:           {r.start_date} to {r.end_date}")
    print(f"  Trading days:     {r.days_prepared}")
    print(f"  Decision points:  {r.decision_points}")
    print(f"  Cache DB:         {r.cache_db_path}")
    print(f"  Cache entries:    {r.cache_entries}")
    print(f"  New cache entries: {r.cache_entries_added}")
    print(f"  Contracts warmed: {r.option_contracts_warmed}")
    print(f"  Option bars:      {r.option_contract_bars_prefetched}")
    print("=" * 60)


def backtest_result_to_dict(r: BacktestResult) -> dict:
    return {
        "initial_equity": r.initial_equity,
        "final_equity": r.final_equity,
        "total_return_pct": r.total_return_pct,
        "net_pnl": r.net_pnl,
        "total_trades": r.total_trades,
        "wins": r.wins,
        "losses": r.losses,
        "win_rate": r.win_rate,
        "max_drawdown": r.max_drawdown,
        "sharpe_ratio": r.sharpe_ratio,
        "profit_factor": r.profit_factor,
        "avg_conviction": r.avg_conviction,
        "days_tested": r.days_tested,
        "llm_error_cycles": r.llm_error_cycles,
        "llm_failure_days": r.llm_failure_days,
        "historical_options_provider": r.historical_options_provider,
        "log_db_path": r.log_db_path,
        "decision_summary": _summarize_decision_log(r.decision_log),
        "equity_curve": [{"date": d, "equity": e} for d, e in r.equity_curve],
        "trades": [
            {
                "entry_date": t.entry_date.isoformat(),
                "exit_date": t.exit_date.isoformat(),
                "underlying": t.underlying,
                "option_type": t.option_type,
                "strike": t.strike,
                "entry_premium": t.entry_premium,
                "exit_premium": t.exit_premium,
                "qty": t.qty,
                "pnl": t.pnl,
                "exit_reason": t.exit_reason,
                "conviction": t.conviction,
                "expression_profile": t.expression_profile,
                "polygon_ticker": t.polygon_ticker,
                "reasoning": t.reasoning,
            }
            for t in r.trades
        ],
        "decision_log": r.decision_log,
    }


def save_backtest_result(r: BacktestResult, path: Path) -> None:
    """Save backtest results to a JSON file."""
    out = backtest_result_to_dict(r)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    log(f"results saved to {path}")


def save_prepare_result(r: PrepareBacktestResult, path: Path) -> None:
    out = {
        "start_date": r.start_date.isoformat(),
        "end_date": r.end_date.isoformat(),
        "days_prepared": r.days_prepared,
        "decision_points": r.decision_points,
        "cache_db_path": str(r.cache_db_path),
        "cache_entries": r.cache_entries,
        "cache_entries_before": r.cache_entries_before,
        "cache_entries_added": r.cache_entries_added,
        "option_contracts_warmed": r.option_contracts_warmed,
        "option_contract_bars_prefetched": r.option_contract_bars_prefetched,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    log(f"prepare summary saved to {path}")


def save_debug_log(r: BacktestResult, path: Path) -> None:
    """Write a human-readable markdown debug log of the backtest."""
    lines: list[str] = []

    # Header
    win_loss = f"{r.wins}W/{r.losses}L" if r.total_trades else "0 trades"
    lines.append("# Backtest Debug Log")
    lines.append(
        f"Period: {r.days_tested} trading days | "
        f"Final: ${r.final_equity:,.0f} ({r.total_return_pct:+.1%}) | {win_loss}"
    )
    decision_summary = _summarize_decision_log(r.decision_log)
    if (
        decision_summary["proposed"]
        or decision_summary["dropped"]
        or decision_summary["provider_retries"]
    ):
        lines.append(
            "Decision summary: "
            f"{decision_summary['proposed']} proposed | "
            f"{decision_summary['executed']} executed | "
            f"{decision_summary['skipped']} skipped | "
            f"{decision_summary['dropped']} dropped | "
            f"{decision_summary['provider_retries']} LLM retries | "
            f"{decision_summary['guardrail_skips']} guardrail skips"
        )
        if decision_summary["skip_reasons"]:
            lines.append("Top skip reasons:")
            for row in decision_summary["skip_reasons"][:5]:
                lines.append(f"- {row['count']}x {row['reason']}")
        if decision_summary["drop_reasons"]:
            lines.append("Top drop reasons:")
            for row in decision_summary["drop_reasons"][:5]:
                lines.append(f"- {row['count']}x {row['reason']}")
        if decision_summary["provider_retry_reasons"]:
            lines.append("Top LLM retry reasons:")
            for row in decision_summary["provider_retry_reasons"][:5]:
                lines.append(f"- {row['count']}x {row['reason']}")
    lines.append("")

    for entry in r.decision_log:
        lines.append("---")
        lines.append("")
        heading = entry.get("decision_time") or entry["date"]
        lines.append(f"## {heading}")
        lines.append("")

        # Market analysis
        analysis_text = entry.get("market_analysis", "")
        if analysis_text:
            lines.append(f"**Market Analysis:** {analysis_text}")
            lines.append("")

        llm_diagnostics = entry.get("llm_diagnostics") or {}
        if isinstance(llm_diagnostics, dict) and llm_diagnostics.get("retries"):
            reasons = []
            for event in llm_diagnostics.get("events") or []:
                if not isinstance(event, dict):
                    continue
                reason = event.get("reason") or "unknown"
                attempt = event.get("attempt") or "?"
                reasons.append(f"attempt {attempt}: {reason}")
            lines.append(
                f"**LLM Diagnostics:** attempts={llm_diagnostics.get('attempts')} "
                f"retries={llm_diagnostics.get('retries')}"
            )
            if reasons:
                lines.append(f"  {'; '.join(reasons)}")
            lines.append("")

        # Thesis updates
        thesis_updates = entry.get("thesis_updates", [])
        if thesis_updates:
            lines.append("**Thesis Updates:**")
            for tu in thesis_updates:
                tid = tu.get("id") or "new"
                status = tu.get("status", "")
                tag = "NEW" if tid == "new" or not tu.get("id") else "UPDATE"
                lines.append(
                    f"- [{tag}] {tid}: {tu.get('underlying', '?')} "
                    f"{tu.get('direction', '?')} — {tu.get('thesis', '')} "
                    f"(conviction: {tu.get('conviction', 0)}, {status})"
                )
            lines.append("")

        # Trades
        trades_proposed = entry.get("trades_proposed", [])
        if trades_proposed:
            lines.append("**Trades:**")
            for tr in trades_proposed:
                action = tr.get("action", "?").upper().replace("_", " ")
                underlying = tr.get("underlying", "?")
                conv = tr.get("conviction", 0)
                risk = tr.get("risk_pct", 0)
                reasoning = tr.get("reasoning", "")
                reasoning_short = reasoning[:100] if reasoning else ""
                lines.append(
                    f"- {action} {underlying} | conviction={conv:.2f} "
                    f"risk={risk:.0%} | \"{reasoning_short}\""
                )
                status = tr.get("status", "skipped")
                if status == "executed":
                    contract = tr.get("contract", "")
                    qty = tr.get("qty", 0)
                    premium = tr.get("premium", 0)
                    if tr.get("action") == "close_position":
                        pnl = tr.get("pnl", 0)
                        lines.append(
                            f"  → Closed: {contract} qty={qty} "
                            f"exit=${premium:.2f} pnl=${pnl:+,.2f}"
                        )
                    else:
                        lines.append(
                            f"  → Executed: {contract} qty={qty} premium=${premium:.2f}"
                        )
                else:
                    skip_reason = tr.get("skip_reason", "unknown")
                    lines.append(f"  → SKIPPED: {skip_reason}")
        else:
            lines.append("**Trades:** None")
        lines.append("")

        dropped_trades = entry.get("dropped_trades", [])
        if dropped_trades:
            lines.append("**Dropped Model Trades:**")
            for dropped in dropped_trades:
                if not isinstance(dropped, dict):
                    lines.append(f"- unknown | {dropped}")
                    continue
                raw = dropped.get("raw") if isinstance(dropped.get("raw"), dict) else {}
                action = (
                    dropped.get("action")
                    or raw.get("action")
                    or "?"
                )
                underlying = (
                    dropped.get("underlying")
                    or raw.get("underlying")
                    or raw.get("ticker")
                    or "?"
                )
                conv = dropped.get("conviction")
                conv_part = f" | conviction={conv:.2f}" if isinstance(conv, (int, float)) else ""
                lines.append(
                    f"- {action} {underlying}{conv_part} | "
                    f"{dropped.get('reason', 'unknown')}"
                )
            lines.append("")

        # Equity / positions
        eq = entry.get("equity", 0)
        pos_count = entry.get("open_positions", 0)
        lines.append(f"**Equity:** ${eq:,.0f} | Positions: {pos_count}")
        lines.append("")

    lines.append("---")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    log(f"debug log saved to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest the AI options trader on historical data."
    )
    parser.add_argument(
        "--start", required=True,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end", required=True,
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--equity", type=float, default=100_000,
        help="Starting equity (default: 100000)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Save results to JSON file",
    )
    parser.add_argument(
        "--delay", type=float, default=1.0,
        help="Seconds between LLM calls (cost control). Default: 1.0",
    )
    parser.add_argument(
        "--decision-interval", type=int, default=config.SCAN_INTERVAL_MINUTES,
        help="Minutes between simulated decision points (default: live scan interval)",
    )
    parser.add_argument(
        "--bar-minutes", type=int, default=5,
        help="Intraday bar size for state/fills (default: 5)",
    )
    parser.add_argument(
        "--news-lookback-hours", type=int, default=config.NEWS_LOOKBACK_HOURS,
        help="Hours of news to include before each simulated decision time",
    )
    parser.add_argument(
        "--no-journal", action="store_true",
        help="Disable thesis journal and trade history (stateless mode)",
    )
    parser.add_argument(
        "--journal-max-active", type=int, default=8,
        help="Max active theses in journal (0=unlimited, default=8)",
    )
    parser.add_argument(
        "--journal-max-display", type=int, default=5,
        help="Max theses shown in full detail (0=all, default=5)",
    )
    parser.add_argument(
        "--cache-db", type=str, default=None,
        help="SQLite DB path for persistent historical data cache",
    )
    parser.add_argument(
        "--log-db", type=str, default=None,
        help="SQLite DB path for streaming backtest trades/decisions during the run",
    )
    parser.add_argument(
        "--offline", action="store_true",
        help="Fail on historical cache misses instead of going to the network",
    )
    parser.add_argument(
        "--prepare-data", action="store_true",
        help="Warm the historical data cache without invoking the LLM",
    )
    parser.add_argument(
        "--prepare-prefetch-symbols", type=int, default=config.PREPARE_PREFETCH_SYMBOLS,
        help="In prepare-data mode, number of symbols to broaden option prefetch for",
    )
    parser.add_argument(
        "--prepare-prefetch-contracts", type=int, default=config.PREPARE_PREFETCH_CONTRACTS_PER_SIDE,
        help="In prepare-data mode, contracts per side to prefetch for each symbol",
    )
    parser.add_argument(
        "--max-consecutive-llm-errors",
        type=int,
        default=None,
        help="Abort after this many consecutive LLM error cycles (0 disables)",
    )
    args = parser.parse_args()

    env_path = Path(__file__).with_name(".env")
    if not env_path.exists():
        env_path = Path(__file__).parent.parent / "momentum_trader" / ".env"
    load_dotenv(env_path, override=True)

    bt_config = BacktestConfig(
        start_date=date.fromisoformat(args.start),
        end_date=date.fromisoformat(args.end),
        initial_equity=args.equity,
        llm_delay_seconds=0.0 if args.prepare_data else args.delay,
        decision_interval_minutes=args.decision_interval,
        signal_bar_minutes=args.bar_minutes,
        news_lookback_hours=args.news_lookback_hours,
        use_journal=not args.no_journal,
        journal_max_active=args.journal_max_active,
        journal_max_full_display=args.journal_max_display,
        offline=args.offline,
        prepare_only=args.prepare_data,
        cache_db_path=Path(args.cache_db) if args.cache_db else None,
        log_db_path=Path(args.log_db) if args.log_db else (
            Path(args.output).with_suffix(".db") if args.output else None
        ),
        prepare_prefetch_symbols=args.prepare_prefetch_symbols,
        prepare_prefetch_contracts_per_side=args.prepare_prefetch_contracts,
        max_consecutive_llm_errors=config.resolved_max_consecutive_llm_error_cycles(
            args.max_consecutive_llm_errors
        ),
    )

    if args.prepare_data:
        prepare_result = prepare_backtest_data(bt_config)
        print_prepare_result(prepare_result)
        if args.output:
            save_prepare_result(prepare_result, Path(args.output))
        return

    result = run_backtest(bt_config)
    print_backtest_result(result)

    if args.output:
        output_path = Path(args.output)
        save_backtest_result(result, output_path)
        save_debug_log(result, Path(f"{output_path}.debug.md"))


if __name__ == "__main__":
    run()
