"""Risk management for the AI trader.

Core rule: each trade risks at most 40% of portfolio equity.
For long options, max loss = premium paid, so:
  max_premium = equity * MAX_RISK_PER_TRADE
"""

from __future__ import annotations

from dataclasses import dataclass

from . import config
from .utils import log


@dataclass(frozen=True)
class RiskCheck:
    approved: bool
    reason: str
    max_contracts: int
    max_premium: float


def evaluate_trade_risk(
    equity: float,
    cash: float,
    current_exposure: float,
    open_positions: int,
    option_ask: float,
    day_pl: float,
) -> RiskCheck:
    """Check whether a proposed trade passes all risk rules."""

    # Daily loss limit
    if equity > 0 and day_pl <= -(equity * config.DAILY_LOSS_LIMIT):
        return RiskCheck(False, "daily loss limit reached", 0, 0.0)

    # Max open positions
    if open_positions >= config.MAX_OPEN_POSITIONS:
        return RiskCheck(False, "max open positions reached", 0, 0.0)

    # Max total exposure
    max_exposure = equity * config.MAX_TOTAL_EXPOSURE
    remaining_exposure = max_exposure - current_exposure
    if remaining_exposure <= 0:
        return RiskCheck(False, "max total exposure reached", 0, 0.0)

    # Max risk per trade: 40% of equity
    max_premium = equity * config.MAX_RISK_PER_TRADE

    # Also cap by remaining exposure budget and available cash
    max_premium = min(max_premium, remaining_exposure, cash)

    if max_premium <= 0:
        return RiskCheck(False, "insufficient buying power", 0, 0.0)

    if option_ask <= 0:
        return RiskCheck(False, "invalid option price", 0, 0.0)

    # Options contracts are 100 shares each
    cost_per_contract = option_ask * 100
    max_contracts = int(max_premium / cost_per_contract)

    if max_contracts <= 0:
        return RiskCheck(
            False,
            f"option too expensive (${cost_per_contract:.2f}/contract vs ${max_premium:.2f} budget)",
            0,
            0.0,
        )

    actual_premium = max_contracts * cost_per_contract
    log(
        f"risk check PASS: max_contracts={max_contracts} "
        f"premium=${actual_premium:.2f} "
        f"({actual_premium / equity * 100:.1f}% of equity)"
    )
    return RiskCheck(True, "approved", max_contracts, actual_premium)


@dataclass(frozen=True)
class PositionRiskState:
    should_close: bool
    reason: str


def evaluate_position_risk(
    entry_premium: float,
    current_premium: float,
    dte: int,
) -> PositionRiskState:
    """Check if an open position should be closed based on risk rules."""
    if entry_premium <= 0:
        return PositionRiskState(False, "hold")

    pnl_pct = (current_premium - entry_premium) / entry_premium

    # Take profit
    if pnl_pct >= config.PROFIT_TARGET_PCT:
        return PositionRiskState(True, f"profit target hit ({pnl_pct:.1%})")

    # Stop loss
    if pnl_pct <= -config.STOP_LOSS_PCT:
        return PositionRiskState(True, f"stop loss hit ({pnl_pct:.1%})")

    # Time-based exit
    if dte <= config.TIME_STOP_DTE:
        return PositionRiskState(True, f"time stop ({dte} DTE remaining)")

    return PositionRiskState(False, "hold")
