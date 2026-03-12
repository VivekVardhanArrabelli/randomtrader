"""Options chain analysis and contract selection."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from . import config
from .alpaca_client import AlpacaClient
from .utils import log, now_eastern, parse_timestamp

EXPRESSION_PROFILE_DEFAULTS: dict[str, dict[str, object]] = {
    "balanced": {
        "label": "Balanced",
        "strike_preference": "atm",
        "expiry_preference": "next_week",
        "target_delta_range": (0.35, 0.60),
        "target_dte_range": (7, 21),
    },
    "time_cushion": {
        "label": "More time",
        "strike_preference": "atm",
        "expiry_preference": "monthly",
        "target_delta_range": (0.45, 0.70),
        "target_dte_range": (20, 45),
    },
    "stock_proxy": {
        "label": "Stock proxy",
        "strike_preference": "itm",
        "expiry_preference": "monthly",
        "target_delta_range": (0.65, 0.90),
        "target_dte_range": (20, 45),
    },
    "convex": {
        "label": "Convex",
        "strike_preference": "otm",
        "expiry_preference": "this_week",
        "target_delta_range": (0.15, 0.40),
        "target_dte_range": (5, 14),
    },
}


def approx_delta(strike: float, spot: float, dte: int, option_type: str) -> float:
    """Approximate option delta using a tanh-based model.

    Good enough for the LLM to distinguish lottery tickets (delta~0.10)
    from stock proxies (delta~0.90). Not for actual hedging.
    """
    if spot <= 0 or strike <= 0 or dte <= 0:
        return 0.0
    moneyness = (spot - strike) / spot  # positive = ITM for calls
    # Scale by time: shorter DTE = steeper delta curve
    time_scale = max(1.0, math.sqrt(dte / 30.0))
    scaling = 4.0 / time_scale
    call_delta = 0.5 + 0.5 * math.tanh(moneyness * scaling / 0.05)
    call_delta = max(0.01, min(0.99, call_delta))
    if option_type == "put":
        return round(call_delta - 1.0, 2)
    return round(call_delta, 2)


def target_strike_for_preference(
    underlying_price: float,
    option_type: str,
    strike_preference: str,
) -> float:
    if strike_preference == "otm":
        return underlying_price * (1.03 if option_type == "call" else 0.97)
    if strike_preference == "itm":
        return underlying_price * (0.97 if option_type == "call" else 1.03)
    return underlying_price


def target_dte_for_expiry_preference(expiry_preference: str) -> int:
    if expiry_preference == "this_week":
        return 4
    if expiry_preference == "monthly":
        return 25
    return 9


def absolute_delta(contract: "OptionContract", underlying_price: float) -> float:
    if contract.delta is not None:
        return abs(contract.delta)
    return abs(
        approx_delta(
            contract.strike,
            underlying_price,
            max(contract.dte, 1),
            contract.option_type,
        )
    )


def _normalize_float_range(
    value_range: tuple[float, float] | None,
) -> tuple[float, float] | None:
    if value_range is None:
        return None
    low, high = value_range
    return (min(low, high), max(low, high))


def _normalize_int_range(
    value_range: tuple[int, int] | None,
) -> tuple[int, int] | None:
    if value_range is None:
        return None
    low, high = value_range
    return (min(low, high), max(low, high))


def _distance_to_range(value: float, value_range: tuple[float, float] | None) -> float:
    if value_range is None:
        return 0.0
    low, high = value_range
    if value < low:
        return low - value
    if value > high:
        return value - high
    return 0.0


def resolve_expression_profile(
    strike_preference: str,
    expiry_preference: str,
    expression_profile: str | None = None,
    target_delta_range: tuple[float, float] | None = None,
    target_dte_range: tuple[int, int] | None = None,
) -> tuple[str, str, tuple[float, float] | None, tuple[int, int] | None]:
    profile = EXPRESSION_PROFILE_DEFAULTS.get((expression_profile or "").strip().lower())
    resolved_strike = strike_preference or config.DEFAULT_STRIKE_PREFERENCE
    resolved_expiry = expiry_preference or "next_week"
    resolved_delta_range = target_delta_range
    resolved_dte_range = target_dte_range
    if profile is None:
        return resolved_strike, resolved_expiry, resolved_delta_range, resolved_dte_range

    if strike_preference in ("", config.DEFAULT_STRIKE_PREFERENCE):
        resolved_strike = str(profile["strike_preference"])
    if expiry_preference in ("", "next_week"):
        resolved_expiry = str(profile["expiry_preference"])
    if resolved_delta_range is None:
        resolved_delta_range = profile["target_delta_range"]  # type: ignore[assignment]
    if resolved_dte_range is None:
        resolved_dte_range = profile["target_dte_range"]  # type: ignore[assignment]
    return resolved_strike, resolved_expiry, resolved_delta_range, resolved_dte_range


@dataclass(frozen=True)
class OptionContract:
    symbol: str               # OCC symbol e.g. AAPL250321C00150000
    underlying: str
    option_type: str           # "call" or "put"
    strike: float
    expiration: date
    bid: float
    ask: float
    mid: float
    volume: int
    open_interest: int
    dte: int                   # days to expiry
    delta: float | None = None
    implied_volatility: float | None = None
    quote_timestamp: datetime | None = None

    @property
    def spread_pct(self) -> float:
        if self.ask <= 0:
            return 1.0
        return (self.ask - self.bid) / self.ask

    def to_context_str(self, underlying_price: float = 0.0) -> str:
        parts = [
            f"{self.symbol} | {self.option_type.upper()} ${self.strike:.2f}",
        ]
        if underlying_price > 0:
            pct = abs(self.strike - underlying_price) / underlying_price * 100
            if self.option_type == "call":
                itm = self.strike < underlying_price
            else:
                itm = self.strike > underlying_price
            label = "ITM" if itm else "OTM"
            atm = pct < 0.5
            moneyness_str = "ATM" if atm else f"{pct:.1f}% {label}"
            parts.append(f"({moneyness_str})")
            d = self.delta if self.delta is not None else approx_delta(
                self.strike, underlying_price, self.dte, self.option_type,
            )
            parts.append(f"delta={d:.2f}" if self.delta is not None else f"delta~{d:.2f}")
            premium_pct = self.mid / underlying_price * 100 if self.mid > 0 else 0.0
            break_even_move_pct = (
                abs((self.strike + self.mid) - underlying_price) / underlying_price * 100
                if self.option_type == "call"
                else abs((self.strike - self.mid) - underlying_price) / underlying_price * 100
            )
            parts.append(f"premium={premium_pct:.1f}%spot")
            parts.append(f"be_move={break_even_move_pct:.1f}%")
        parts.append(f"spread={self.spread_pct * 100:.1f}%")
        if self.implied_volatility is not None:
            parts.append(f"iv={self.implied_volatility * 100:.1f}%")
        parts.append(
            f"exp={self.expiration} DTE={self.dte} "
            f"bid={self.bid:.2f} ask={self.ask:.2f} mid={self.mid:.2f} "
            f"vol={self.volume} OI={self.open_interest}"
        )
        if self.quote_timestamp is not None and self.quote_timestamp.tzinfo is not None:
            age_seconds = max((now_eastern() - self.quote_timestamp).total_seconds(), 0.0)
            if age_seconds < 60:
                parts.append(f"quote_age={int(age_seconds)}s")
            else:
                parts.append(f"quote_age={int(age_seconds // 60)}m")
        return " ".join(parts)


def fetch_option_chain(
    alpaca: AlpacaClient,
    underlying: str,
    underlying_price: float,
    option_type: str | None = None,
    min_dte: int | None = None,
    max_dte: int | None = None,
    strike_band_pct: float | None = None,
) -> list[OptionContract]:
    """Fetch and filter options contracts for an underlying."""
    today = now_eastern().date()
    resolved_min_dte = max(min_dte if min_dte is not None else config.PREFERRED_DTE_MIN, 1)
    resolved_max_dte = max_dte if max_dte is not None else config.PREFERRED_DTE_MAX
    resolved_max_dte = max(resolved_max_dte, resolved_min_dte)
    exp_gte = (today + timedelta(days=resolved_min_dte)).isoformat()
    exp_lte = (today + timedelta(days=resolved_max_dte)).isoformat()

    band_pct = strike_band_pct if strike_band_pct is not None else 0.15
    band_pct = max(band_pct, 0.05)
    strike_gte = round(underlying_price * max(0.05, 1.0 - band_pct), 2)
    strike_lte = round(underlying_price * (1.0 + band_pct), 2)

    try:
        raw = alpaca.get_option_contracts(
            underlying=underlying,
            option_type=option_type,
            expiration_date_gte=exp_gte,
            expiration_date_lte=exp_lte,
            strike_price_gte=strike_gte,
            strike_price_lte=strike_lte,
            limit=100,
        )
    except Exception as exc:
        log(f"option chain fetch error for {underlying}: {exc}")
        return []

    contracts: list[OptionContract] = []
    for item in raw:
        symbol = item.get("symbol") or ""
        strike = float(item.get("strike_price") or 0)
        exp_str = item.get("expiration_date") or ""
        opt_type = item.get("type") or ""
        if not symbol or strike <= 0 or not exp_str:
            continue
        try:
            exp_date = date.fromisoformat(exp_str)
        except ValueError:
            continue
        dte = (exp_date - today).days
        if dte < resolved_min_dte:
            continue

        contracts.append(
            OptionContract(
                symbol=symbol,
                underlying=underlying,
                option_type=opt_type,
                strike=strike,
                expiration=exp_date,
                bid=0.0,
                ask=0.0,
                mid=0.0,
                volume=0,
                open_interest=int(item.get("open_interest") or 0),
                dte=dte,
            )
        )

    # Try to get live snapshots for these contracts
    if contracts:
        contracts = _enrich_with_market_data(alpaca, underlying, contracts)

    # Filter by quality
    filtered = []
    for c in contracts:
        if c.ask <= 0:
            continue
        if c.spread_pct > config.MAX_BID_ASK_SPREAD_PCT:
            continue
        if c.open_interest < config.MIN_OPEN_INTEREST and c.volume < config.MIN_OPTION_VOLUME:
            continue
        filtered.append(c)

    filtered.sort(key=lambda c: (c.dte, abs(c.strike - underlying_price)))
    return filtered


def _extract_snapshot_map(data: dict | list | None) -> dict[str, dict]:
    if not isinstance(data, dict):
        return {}
    if "snapshots" in data and isinstance(data["snapshots"], dict):
        return {
            str(symbol): snapshot
            for symbol, snapshot in data["snapshots"].items()
            if isinstance(snapshot, dict)
        }
    return {
        str(symbol): snapshot
        for symbol, snapshot in data.items()
        if isinstance(snapshot, dict)
    }


def _snapshot_quote(snapshot: dict) -> dict:
    return (
        snapshot.get("latestQuote")
        or snapshot.get("latest_quote")
        or snapshot.get("quote")
        or {}
    )


def _enrich_with_market_data(
    alpaca: AlpacaClient,
    underlying: str,
    contracts: list[OptionContract],
) -> list[OptionContract]:
    """Add bid/ask plus snapshot greeks/IV when Alpaca exposes them."""
    snapshot_map: dict[str, dict] = {}
    try:
        snapshot_map = _extract_snapshot_map(alpaca.get_option_snapshots(underlying))
    except Exception as exc:
        log(f"option snapshots fetch error for {underlying}: {exc}")

    symbols = [c.symbol for c in contracts]
    # Fetch in chunks of 20
    enriched = []
    for i in range(0, len(symbols), 20):
        chunk_symbols = symbols[i : i + 20]
        chunk_contracts = contracts[i : i + 20]
        try:
            data = alpaca.get_option_latest_quotes(chunk_symbols)
        except Exception as exc:
            log(f"option quotes fetch error: {exc}")
            enriched.extend(chunk_contracts)
            continue

        quotes = data.get("quotes", data) if isinstance(data, dict) else {}
        for contract in chunk_contracts:
            snapshot = snapshot_map.get(contract.symbol, {})
            quote = _snapshot_quote(snapshot) if snapshot else {}
            if not quote:
                quote = quotes.get(contract.symbol, {})
            if not isinstance(quote, dict):
                enriched.append(contract)
                continue
            bid = float(quote.get("bp") or quote.get("bid_price") or 0.0)
            ask = float(quote.get("ap") or quote.get("ask_price") or 0.0)
            mid = (bid + ask) / 2 if bid > 0 and ask > 0 else 0.0
            greeks = snapshot.get("greeks") or {}
            delta = greeks.get("delta")
            iv = (
                snapshot.get("impliedVolatility")
                or snapshot.get("implied_volatility")
                or snapshot.get("iv")
            )
            day = snapshot.get("day") or {}
            quote_timestamp = parse_timestamp(
                quote.get("t")
                or quote.get("timestamp")
                or snapshot.get("updated_at")
                or snapshot.get("latest_trade", {}).get("t")
            )
            enriched.append(
                OptionContract(
                    symbol=contract.symbol,
                    underlying=contract.underlying,
                    option_type=contract.option_type,
                    strike=contract.strike,
                    expiration=contract.expiration,
                    bid=bid,
                    ask=ask,
                    mid=mid,
                    volume=int(
                        quote.get("volume")
                        or day.get("volume")
                        or contract.volume
                    ),
                    open_interest=int(
                        snapshot.get("open_interest")
                        or contract.open_interest
                    ),
                    dte=contract.dte,
                    delta=float(delta) if delta is not None else None,
                    implied_volatility=float(iv) if iv is not None else None,
                    quote_timestamp=quote_timestamp,
                )
            )
    return enriched


def select_contract(
    contracts: list[OptionContract],
    underlying_price: float,
    strike_preference: str = "atm",
    expiry_preference: str = "next_week",
    expression_profile: str | None = None,
    contract_symbol: str | None = None,
    target_delta_range: tuple[float, float] | None = None,
    target_dte_range: tuple[int, int] | None = None,
    max_spread_pct: float | None = None,
) -> OptionContract | None:
    ranked = rank_contracts(
        contracts,
        underlying_price,
        strike_preference=strike_preference,
        expiry_preference=expiry_preference,
        expression_profile=expression_profile,
        contract_symbol=contract_symbol,
        target_delta_range=target_delta_range,
        target_dte_range=target_dte_range,
        max_spread_pct=max_spread_pct,
    )
    return ranked[0] if ranked else None


def rank_contracts(
    contracts: list[OptionContract],
    underlying_price: float,
    strike_preference: str = "atm",
    expiry_preference: str = "next_week",
    expression_profile: str | None = None,
    contract_symbol: str | None = None,
    target_delta_range: tuple[float, float] | None = None,
    target_dte_range: tuple[int, int] | None = None,
    max_spread_pct: float | None = None,
) -> list[OptionContract]:
    """Rank contracts by fit and quality, preserving exact-symbol override."""
    if not contracts:
        return []

    if contract_symbol:
        exact_matches = [
            c for c in contracts if c.symbol.upper() == contract_symbol.upper()
        ]
        return exact_matches[:1]

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

    candidates = list(contracts)
    normalized_delta_range = _normalize_float_range(target_delta_range)
    normalized_dte_range = _normalize_int_range(target_dte_range)
    if normalized_dte_range is not None:
        min_dte, max_dte = normalized_dte_range
        bounded = [
            c for c in candidates
            if min_dte <= c.dte <= max_dte
        ]
        if not bounded:
            return []
        candidates = bounded

    if max_spread_pct is not None:
        spread_filtered = [c for c in candidates if c.spread_pct <= max_spread_pct]
        if not spread_filtered:
            return []
        candidates = spread_filtered

    if strike_preference == "itm":
        # For calls: strike < price. For puts: strike > price.
        # Sort by proximity to price, preferring ITM.
        calls = [c for c in candidates if c.option_type == "call" and c.strike <= underlying_price]
        puts = [c for c in candidates if c.option_type == "put" and c.strike >= underlying_price]
        candidates = calls + puts
    elif strike_preference == "otm":
        calls = [c for c in candidates if c.option_type == "call" and c.strike >= underlying_price]
        puts = [c for c in candidates if c.option_type == "put" and c.strike <= underlying_price]
        candidates = calls + puts

    if not candidates:
        return []

    if normalized_dte_range is not None:
        target_dte = (normalized_dte_range[0] + normalized_dte_range[1]) / 2
    else:
        target_dte = target_dte_for_expiry_preference(expiry_preference)

    def _selection_score(contract: OptionContract) -> float:
        target_strike = target_strike_for_preference(
            underlying_price,
            contract.option_type,
            strike_preference,
        )
        strike_penalty = (
            abs(contract.strike - target_strike) / underlying_price * 100
            if underlying_price > 0
            else 0.0
        )
        spread_penalty = contract.spread_pct * 4.0
        volume_penalty = 0.35 / (1.0 + min(contract.volume, 500) / 50.0)
        oi_penalty = 0.25 / (1.0 + min(contract.open_interest, 2000) / 200.0)
        dte_penalty = _distance_to_range(contract.dte, normalized_dte_range) / 10.0
        if normalized_dte_range is None:
            dte_penalty = abs(contract.dte - target_dte) / 20.0
        delta_penalty = 0.0
        if normalized_delta_range is not None and underlying_price > 0:
            delta_penalty = _distance_to_range(
                absolute_delta(contract, underlying_price),
                normalized_delta_range,
            ) * 4.0
        return (
            strike_penalty
            + spread_penalty
            + volume_penalty
            + oi_penalty
            + dte_penalty
            + delta_penalty
        )

    candidates.sort(
        key=lambda c: (
            _selection_score(c),
            c.spread_pct,
            -c.volume,
            -c.open_interest,
            c.dte,
        )
    )
    return candidates


def shortlist_contracts(
    contracts: list[OptionContract],
    underlying_price: float,
    per_type: int = 3,
) -> dict[str, list[OptionContract]]:
    shortlists: dict[str, list[OptionContract]] = {"call": [], "put": []}
    for option_type in ("call", "put"):
        typed_contracts = [c for c in contracts if c.option_type == option_type]
        ranked = rank_contracts(
            typed_contracts,
            underlying_price,
            strike_preference="atm",
            expiry_preference="next_week",
        )
        shortlists[option_type] = ranked[:per_type]
    return shortlists


def _expression_shortlist(
    contracts: list[OptionContract],
    underlying_price: float,
    option_type: str,
) -> list[tuple[str, OptionContract]]:
    typed_contracts = [c for c in contracts if c.option_type == option_type]
    if not typed_contracts:
        return []

    profiles = [
        (
            "Primary",
            dict(
                expression_profile="balanced",
                strike_preference="atm",
                expiry_preference="next_week",
            ),
        ),
        (
            "More time",
            dict(
                expression_profile="time_cushion",
            ),
        ),
        (
            "Stock proxy",
            dict(
                expression_profile="stock_proxy",
            ),
        ),
        (
            "Convex upside",
            dict(
                expression_profile="convex",
            ),
        ),
    ]

    labeled: list[tuple[str, OptionContract]] = []
    seen: set[str] = set()
    for label, params in profiles:
        ranked = rank_contracts(typed_contracts, underlying_price, **params)
        match = next((contract for contract in ranked if contract.symbol not in seen), None)
        if match is None:
            continue
        labeled.append((label, match))
        seen.add(match.symbol)

    if len(labeled) < 3:
        fallback = rank_contracts(
            typed_contracts,
            underlying_price,
            strike_preference="atm",
            expiry_preference="next_week",
        )
        for contract in fallback:
            if contract.symbol in seen:
                continue
            labeled.append(("Alternative", contract))
            seen.add(contract.symbol)
            if len(labeled) >= 3:
                break
    return labeled


def format_chain_for_llm(
    contracts: list[OptionContract],
    max_contracts: int = 15,
    underlying_price: float = 0.0,
    expression_guidance: list[str] | None = None,
) -> str:
    if not contracts:
        return "No options contracts available."
    lines: list[str] = []
    if contracts:
        if expression_guidance:
            lines.extend(expression_guidance)
        lines.append("Suggested contract shortlist:")
        lines.append(
            "  Primary = cleanest current expression | "
            "More time = slower decay / thesis room | "
            "Stock proxy = deeper delta / closer to shares | "
            "Convex = cheaper and faster but more fragile"
        )
        for option_type, label in (("call", "Calls"), ("put", "Puts")):
            ranked = _expression_shortlist(contracts, underlying_price, option_type)
            if not ranked:
                continue
            lines.append(f"  {label}:")
            for shortlist_label, contract in ranked:
                lines.append(
                    f"    {shortlist_label}: {contract.to_context_str(underlying_price)}"
                )
        lines.append("--- Full filtered chain ---")
    lines.extend(c.to_context_str(underlying_price) for c in contracts[:max_contracts])
    return "\n".join(lines)
