"""Options chain analysis and contract selection."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from . import config
from .alpaca_client import AlpacaClient
from .utils import log, now_eastern


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
    return abs(
        approx_delta(
            contract.strike,
            underlying_price,
            max(contract.dte, 1),
            contract.option_type,
        )
    )


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
            d = approx_delta(self.strike, underlying_price, self.dte, self.option_type)
            parts.append(f"delta~{d:.2f}")
        parts.append(
            f"exp={self.expiration} DTE={self.dte} "
            f"bid={self.bid:.2f} ask={self.ask:.2f} mid={self.mid:.2f} "
            f"vol={self.volume} OI={self.open_interest}"
        )
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

    # Try to get live quotes for these contracts
    if contracts:
        contracts = _enrich_with_quotes(alpaca, contracts)

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


def _enrich_with_quotes(
    alpaca: AlpacaClient, contracts: list[OptionContract]
) -> list[OptionContract]:
    """Add bid/ask/volume data from live quotes."""
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
            quote = quotes.get(contract.symbol, {})
            if not isinstance(quote, dict):
                enriched.append(contract)
                continue
            bid = float(quote.get("bp") or quote.get("bid_price") or 0.0)
            ask = float(quote.get("ap") or quote.get("ask_price") or 0.0)
            mid = (bid + ask) / 2 if bid > 0 and ask > 0 else 0.0
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
                    volume=int(quote.get("volume") or contract.volume),
                    open_interest=contract.open_interest,
                    dte=contract.dte,
                )
            )
    return enriched


def select_contract(
    contracts: list[OptionContract],
    underlying_price: float,
    strike_preference: str = "atm",
    expiry_preference: str = "next_week",
    contract_symbol: str | None = None,
    target_delta: float | None = None,
    min_dte: int | None = None,
    max_dte: int | None = None,
    max_spread_pct: float | None = None,
) -> OptionContract | None:
    """Select the best contract based on moneyness, expiry, and quality."""
    if not contracts:
        return None

    if contract_symbol:
        exact_matches = [
            c for c in contracts if c.symbol.upper() == contract_symbol.upper()
        ]
        return exact_matches[0] if exact_matches else None

    candidates = list(contracts)
    if min_dte is not None or max_dte is not None:
        bounded = [
            c for c in candidates
            if (min_dte is None or c.dte >= min_dte)
            and (max_dte is None or c.dte <= max_dte)
        ]
        if not bounded:
            return None
        candidates = bounded

    if max_spread_pct is not None:
        spread_filtered = [c for c in candidates if c.spread_pct <= max_spread_pct]
        if not spread_filtered:
            return None
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
        return None

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
        dte_penalty = abs(contract.dte - target_dte) / 20.0
        delta_penalty = 0.0
        if target_delta is not None and underlying_price > 0:
            delta_penalty = abs(absolute_delta(contract, underlying_price) - target_delta) * 4.0
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
    return candidates[0] if candidates else None


def format_chain_for_llm(
    contracts: list[OptionContract],
    max_contracts: int = 15,
    underlying_price: float = 0.0,
) -> str:
    if not contracts:
        return "No options contracts available."
    lines = [c.to_context_str(underlying_price) for c in contracts[:max_contracts]]
    return "\n".join(lines)
