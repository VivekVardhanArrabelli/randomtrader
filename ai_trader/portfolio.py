"""Portfolio state tracking."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from . import config
from .alpaca_client import AlpacaClient
from .options import approx_delta
from .risk import stop_loss_for_dte
from .utils import AccountSnapshot, log, now_eastern


@dataclass
class OptionPosition:
    symbol: str              # OCC option symbol
    underlying: str
    option_type: str         # call / put
    strike: float
    expiration: str
    qty: int
    avg_entry_price: float   # per share (premium)
    current_price: float     # per share (premium)
    market_value: float
    unrealized_pl: float
    cost_basis: float
    underlying_spot: float = 0.0    # current underlying price
    risk_alert: str = ""            # risk alert message for LLM

    @property
    def pnl_pct(self) -> float:
        if self.cost_basis <= 0:
            return 0.0
        return self.unrealized_pl / self.cost_basis

    @property
    def dte(self) -> int:
        try:
            exp = date.fromisoformat(self.expiration)
            return (exp - now_eastern().date()).days
        except (ValueError, TypeError):
            return 999

    def to_context_str(self) -> str:
        direction = "CALL" if self.option_type == "call" else "PUT"
        decay_info = ""
        if self.dte > 0 and self.dte != 999 and self.current_price > 0:
            daily_decay = self.current_price / self.dte
            total_daily = daily_decay * abs(self.qty) * 100
            decay_info = f" decay≈${total_daily:.0f}/day"

        # Underlying spot + moneyness
        spot_info = ""
        if self.underlying_spot > 0:
            pct = abs(self.strike - self.underlying_spot) / self.underlying_spot * 100
            if self.option_type == "call":
                itm = self.strike < self.underlying_spot
            else:
                itm = self.strike > self.underlying_spot
            label = "ITM" if itm else "OTM"
            atm = pct < 0.5
            moneyness_str = "ATM" if atm else f"{pct:.1f}% {label}"
            d = approx_delta(self.strike, self.underlying_spot, max(self.dte, 1), self.option_type)
            spot_info = f" underlying=${self.underlying_spot:.2f} ({moneyness_str}) delta~{d:.2f}"

        # Exit-trigger proximity flags
        flags: list[str] = []
        if self.cost_basis > 0 and self.current_price > 0:
            if self.pnl_pct >= config.PROFIT_TARGET_PCT * 0.8:
                flags.append(f"approaching profit target of {config.PROFIT_TARGET_PCT:.0%}")
            sl = stop_loss_for_dte(self.dte)
            if self.pnl_pct <= -sl * 0.8:
                flags.append(f"approaching stop loss of {sl:.0%} ({self.dte} DTE)")
        if self.dte != 999 and self.dte <= config.TIME_STOP_DTE + 1:
            flags.append(f"approaching time stop ({config.TIME_STOP_DTE} DTE)")
        flag_str = f" [{'; '.join(flags)}]" if flags else ""

        # Risk alert
        alert_str = f"\n    ** RISK ALERT: {self.risk_alert}" if self.risk_alert else ""

        return (
            f"{self.underlying} {direction} ${self.strike:.2f} exp={self.expiration} "
            f"qty={self.qty} entry=${self.avg_entry_price:.2f} "
            f"current=${self.current_price:.2f} "
            f"P&L=${self.unrealized_pl:.2f} ({self.pnl_pct:.1%}) "
            f"DTE={self.dte}{spot_info}{decay_info}{flag_str}{alert_str}"
        )


@dataclass
class EquityPosition:
    symbol: str
    qty: int
    avg_entry_price: float
    current_price: float
    market_value: float
    unrealized_pl: float
    cost_basis: float

    @property
    def pnl_pct(self) -> float:
        if self.cost_basis <= 0:
            return 0.0
        return self.unrealized_pl / self.cost_basis

    def to_context_str(self) -> str:
        side = "LONG" if self.qty >= 0 else "SHORT"
        return (
            f"{self.symbol} {side} qty={self.qty} "
            f"entry=${self.avg_entry_price:.2f} current=${self.current_price:.2f} "
            f"P&L=${self.unrealized_pl:.2f} ({self.pnl_pct:.1%})"
        )


@dataclass
class PortfolioState:
    account: AccountSnapshot
    option_positions: list[OptionPosition]
    equity_positions: list[EquityPosition]

    @property
    def total_options_exposure(self) -> float:
        return sum(abs(p.market_value) for p in self.option_positions)

    @property
    def total_equity_exposure(self) -> float:
        return sum(abs(p.market_value) for p in self.equity_positions)

    @property
    def total_exposure(self) -> float:
        return self.total_options_exposure + self.total_equity_exposure

    @property
    def open_option_count(self) -> int:
        return len(self.option_positions)

    @property
    def open_equity_count(self) -> int:
        return len(self.equity_positions)

    @property
    def open_position_count(self) -> int:
        return self.open_option_count + self.open_equity_count

    def to_context_str(self) -> str:
        lines = [
            f"Account Equity: ${self.account.equity:,.2f}",
            f"Cash: ${self.account.cash:,.2f}",
            f"Buying Power: ${self.account.buying_power:,.2f}",
            f"Day P&L: ${self.account.day_pl:,.2f}",
            f"Options Exposure: ${self.total_options_exposure:,.2f} "
            f"({self.total_options_exposure / self.account.equity * 100:.1f}% of equity)"
            if self.account.equity > 0 else "",
            f"Equity Exposure: ${self.total_equity_exposure:,.2f} "
            f"({self.total_equity_exposure / self.account.equity * 100:.1f}% of equity)"
            if self.account.equity > 0 else "",
            f"Total Exposure: ${self.total_exposure:,.2f} "
            f"({self.total_exposure / self.account.equity * 100:.1f}% of equity)"
            if self.account.equity > 0 else "",
            f"Open Option Positions: {self.open_option_count}",
            f"Open Equity Positions: {self.open_equity_count}",
        ]
        if self.option_positions or self.equity_positions:
            lines.append("\nCurrent Positions:")
            for pos in self.option_positions:
                lines.append(f"  {pos.to_context_str()}")
            for pos in self.equity_positions:
                lines.append(f"  {pos.to_context_str()}")
        else:
            lines.append("\nNo open positions.")
        return "\n".join(lines)


def get_portfolio_state(alpaca: AlpacaClient) -> PortfolioState:
    """Fetch current portfolio state from Alpaca."""
    account_data = alpaca.get_account()
    account = AccountSnapshot(
        equity=float(account_data.get("equity") or 0),
        cash=float(account_data.get("cash") or 0),
        buying_power=float(account_data.get("buying_power") or 0),
        day_pl=float(
            account_data.get("equity", 0)) - float(account_data.get("last_equity", 0)
        ),
        positions_value=float(account_data.get("long_market_value") or 0),
    )

    raw_positions = alpaca.get_positions()
    option_positions: list[OptionPosition] = []
    equity_positions: list[EquityPosition] = []

    for pos in raw_positions:
        asset_class = pos.get("asset_class", "").lower()
        symbol = pos.get("symbol", "")

        if asset_class == "options" or _looks_like_option_symbol(symbol):
            option_positions.append(
                OptionPosition(
                    symbol=symbol,
                    underlying=pos.get("underlying_symbol") or _extract_underlying(symbol),
                    option_type=_extract_option_type(symbol, pos),
                    strike=float(pos.get("strike_price") or _extract_strike(symbol)),
                    expiration=pos.get("expiration_date") or _extract_expiration(symbol),
                    qty=int(float(pos.get("qty") or 0)),
                    avg_entry_price=float(pos.get("avg_entry_price") or 0),
                    current_price=float(pos.get("current_price") or 0),
                    market_value=float(pos.get("market_value") or 0),
                    unrealized_pl=float(pos.get("unrealized_pl") or 0),
                    cost_basis=float(pos.get("cost_basis") or 0),
                )
            )
        else:
            equity_positions.append(
                EquityPosition(
                    symbol=symbol,
                    qty=int(float(pos.get("qty") or 0)),
                    avg_entry_price=float(pos.get("avg_entry_price") or 0),
                    current_price=float(pos.get("current_price") or 0),
                    market_value=float(pos.get("market_value") or 0),
                    unrealized_pl=float(pos.get("unrealized_pl") or 0),
                    cost_basis=float(pos.get("cost_basis") or 0),
                )
            )

    log(
        f"portfolio: equity=${account.equity:,.2f} cash=${account.cash:,.2f} "
        f"day_pl=${account.day_pl:,.2f} options={len(option_positions)} equities={len(equity_positions)}"
    )
    return PortfolioState(
        account=account,
        option_positions=option_positions,
        equity_positions=equity_positions,
    )


def _looks_like_option_symbol(symbol: str) -> bool:
    # OCC option symbols are typically 21 chars: AAPL  250321C00150000
    # But Alpaca may use condensed format: AAPL250321C00150000
    if len(symbol) >= 15 and any(c in symbol for c in ("C", "P")):
        # Check if it has a date-like pattern followed by C or P
        for i in range(len(symbol)):
            if symbol[i] in ("C", "P") and i >= 6:
                try:
                    int(symbol[i + 1 : i + 9])
                    return True
                except (ValueError, IndexError):
                    continue
    return False


def _extract_underlying(symbol: str) -> str:
    for i, c in enumerate(symbol):
        if c.isdigit():
            return symbol[:i].strip()
    return symbol[:4].strip()


def _extract_option_type(symbol: str, pos: dict) -> str:
    opt_type = pos.get("option_type") or pos.get("type") or ""
    if opt_type:
        normalized = str(opt_type).strip().lower()
        if normalized in ("call", "c"):
            return "call"
        if normalized in ("put", "p"):
            return "put"

    # OCC/condensed format: <UNDERLYING><YYMMDD><C|P><STRIKE*1000>
    # The option type is the char immediately after the 6-digit date.
    for i, c in enumerate(symbol):
        if c.isdigit():
            type_idx = i + 6
            if type_idx < len(symbol):
                cp = symbol[type_idx].upper()
                if cp == "C":
                    return "call"
                if cp == "P":
                    return "put"
            break

    # Fallback: scan from right for C/P followed by a numeric strike suffix.
    for i in range(len(symbol) - 9, -1, -1):
        cp = symbol[i].upper()
        if cp in ("C", "P") and symbol[i + 1 :].isdigit():
            return "call" if cp == "C" else "put"
    return "unknown"


def _extract_strike(symbol: str) -> float:
    for i, c in enumerate(symbol):
        if c in ("C", "P") and i > 0:
            try:
                raw = symbol[i + 1 :]
                return int(raw) / 1000
            except (ValueError, IndexError):
                pass
    return 0.0


def _extract_expiration(symbol: str) -> str:
    for i, c in enumerate(symbol):
        if c.isdigit() and i + 6 <= len(symbol):
            try:
                date_part = symbol[i : i + 6]
                int(date_part)
                year = 2000 + int(date_part[:2])
                month = int(date_part[2:4])
                day = int(date_part[4:6])
                return f"{year}-{month:02d}-{day:02d}"
            except (ValueError, IndexError):
                continue
    return ""
