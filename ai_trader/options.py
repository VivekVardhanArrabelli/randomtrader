"""Options chain analysis and contract selection."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
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
) -> list[OptionContract]:
    """Fetch and filter options contracts for an underlying."""
    today = now_eastern().date()
    exp_gte = (today + timedelta(days=config.PREFERRED_DTE_MIN)).isoformat()
    exp_lte = (today + timedelta(days=config.PREFERRED_DTE_MAX)).isoformat()

    # Strike range: +/- 15% from current price
    strike_gte = round(underlying_price * 0.85, 2)
    strike_lte = round(underlying_price * 1.15, 2)

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
        if dte < config.PREFERRED_DTE_MIN:
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
) -> OptionContract | None:
    """Select the best contract based on strike preference."""
    if not contracts:
        return None

    if strike_preference == "itm":
        # For calls: strike < price. For puts: strike > price.
        # Sort by proximity to price, preferring ITM.
        calls = [c for c in contracts if c.option_type == "call" and c.strike <= underlying_price]
        puts = [c for c in contracts if c.option_type == "put" and c.strike >= underlying_price]
        candidates = calls + puts
    elif strike_preference == "otm":
        calls = [c for c in contracts if c.option_type == "call" and c.strike >= underlying_price]
        puts = [c for c in contracts if c.option_type == "put" and c.strike <= underlying_price]
        candidates = calls + puts
    else:  # atm
        candidates = list(contracts)

    if not candidates:
        candidates = list(contracts)

    # Sort by distance from ATM, then by DTE (prefer sooner but not too soon)
    candidates.sort(
        key=lambda c: (abs(c.strike - underlying_price) / underlying_price, c.dte)
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
