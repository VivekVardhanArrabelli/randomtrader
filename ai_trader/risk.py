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


@dataclass(frozen=True)
class StockRiskCheck:
    approved: bool
    reason: str
    max_shares: int
    max_notional: float


def size_for_risk_budget(risk_budget: float, unit_cost: float) -> int:
    """Return the largest whole-number position that fits within a hard budget."""
    if risk_budget <= 0 or unit_cost <= 0:
        return 0
    return max(int(risk_budget / unit_cost), 0)


def scale_risk_pct_for_conviction(risk_pct: float, conviction: float) -> float:
    """Scale marginal trades down without blocking the model's thesis.

    The execution threshold is deliberately permissive at 0.60 so the model can
    express fresh edges. Trades close to that threshold should be probes; full
    requested sizing is reserved for much stronger evidence.
    """
    if risk_pct <= 0:
        return 0.0

    risk_pct = max(0.0, float(risk_pct))
    if risk_pct <= 0.02:
        return risk_pct

    conviction = max(0.0, min(float(conviction), 1.0))
    if conviction >= 0.80:
        multiplier = 1.0
    elif conviction >= 0.75:
        multiplier = 0.85
    elif conviction >= 0.70:
        multiplier = 0.70
    elif conviction >= 0.65:
        multiplier = 0.50
    else:
        multiplier = 0.35
    return risk_pct * multiplier


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
        f"risk rails PASS: max_contracts={max_contracts} "
        f"max_premium_allowed=${actual_premium:.2f} "
        f"({actual_premium / equity * 100:.1f}% of equity cap)"
    )
    return RiskCheck(True, "approved", max_contracts, actual_premium)


def evaluate_stock_trade_risk(
    equity: float,
    cash: float,
    current_exposure: float,
    open_positions: int,
    share_price: float,
    day_pl: float,
) -> StockRiskCheck:
    """Check whether a proposed stock trade passes the same hard risk rails."""

    if equity > 0 and day_pl <= -(equity * config.DAILY_LOSS_LIMIT):
        return StockRiskCheck(False, "daily loss limit reached", 0, 0.0)

    if open_positions >= config.MAX_OPEN_POSITIONS:
        return StockRiskCheck(False, "max open positions reached", 0, 0.0)

    max_exposure = equity * config.MAX_TOTAL_EXPOSURE
    remaining_exposure = max_exposure - current_exposure
    if remaining_exposure <= 0:
        return StockRiskCheck(False, "max total exposure reached", 0, 0.0)

    max_notional = equity * config.MAX_RISK_PER_TRADE
    max_notional = min(max_notional, remaining_exposure, cash)

    if max_notional <= 0:
        return StockRiskCheck(False, "insufficient buying power", 0, 0.0)

    if share_price <= 0:
        return StockRiskCheck(False, "invalid stock price", 0, 0.0)

    max_shares = int(max_notional / share_price)
    if max_shares <= 0:
        return StockRiskCheck(
            False,
            f"stock too expensive (${share_price:.2f}/share vs ${max_notional:.2f} budget)",
            0,
            0.0,
        )

    actual_notional = max_shares * share_price
    log(
        f"stock risk check PASS: max_shares={max_shares} "
        f"notional=${actual_notional:.2f} "
        f"({actual_notional / equity * 100:.1f}% of equity)"
    )
    return StockRiskCheck(True, "approved", max_shares, actual_notional)


@dataclass(frozen=True)
class PositionRiskState:
    should_close: bool
    reason: str


@dataclass(frozen=True)
class PositionRiskAlert:
    alert_type: str       # "stop_loss", "profit_target", "time_stop"
    message: str
    severity: str         # "critical" or "warning"


def assess_position_risk(
    entry_premium: float,
    current_premium: float,
    dte: int,
) -> PositionRiskState | PositionRiskAlert | None:
    """Assess position risk, splitting hard stops from soft alerts.

    Returns:
      - PositionRiskState(should_close=True) for catastrophic loss (>= -85%).
        These are auto-closed immediately — no LLM override.
      - PositionRiskAlert for DTE-scaled stop losses, profit targets, and
        time stops. These are shown to the LLM as alerts so it can decide.
      - None if no action needed.
    """
    if entry_premium <= 0:
        return None

    pnl_pct = (current_premium - entry_premium) / entry_premium

    # Hard stop: catastrophic loss — auto-close, no override
    if pnl_pct <= -config.CATASTROPHIC_STOP_PCT:
        return PositionRiskState(
            True,
            f"catastrophic stop ({pnl_pct:.1%} loss, limit {config.CATASTROPHIC_STOP_PCT:.0%})",
        )

    # Soft alert: profit target
    if pnl_pct >= config.PROFIT_TARGET_PCT:
        return PositionRiskAlert(
            alert_type="profit_target",
            message=f"PROFIT TARGET: +{pnl_pct:.1%} gain (target {config.PROFIT_TARGET_PCT:.0%}). Consider taking profits.",
            severity="warning",
        )

    # Soft alert: DTE-scaled stop loss
    sl = stop_loss_for_dte(dte)
    if pnl_pct <= -sl:
        return PositionRiskAlert(
            alert_type="stop_loss",
            message=f"STOP LOSS: {pnl_pct:.1%} loss (limit {sl:.0%} for {dte} DTE). Consider closing.",
            severity="critical",
        )

    # Soft alert: time stop
    if dte <= config.TIME_STOP_DTE:
        return PositionRiskAlert(
            alert_type="time_stop",
            message=f"TIME STOP: only {dte} DTE remaining. Close unless strong conviction.",
            severity="critical",
        )

    return None


def stop_loss_for_dte(dte: int) -> float:
    """Return the stop-loss threshold scaled by days to expiration.

    Wider stops for longer-dated positions (more time to recover),
    tighter stops for short-dated positions (theta is aggressive).
    """
    if dte <= config.STOP_LOSS_SHORT_DTE_THRESHOLD:
        return config.STOP_LOSS_PCT
    if dte >= config.STOP_LOSS_LONG_DTE_THRESHOLD:
        return config.STOP_LOSS_PCT_LONG_DTE
    return config.STOP_LOSS_PCT_MID_DTE


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

    # Stop loss — scaled by DTE
    sl = stop_loss_for_dte(dte)
    if pnl_pct <= -sl:
        return PositionRiskState(True, f"stop loss hit ({pnl_pct:.1%}, limit {sl:.0%} for {dte} DTE)")

    # Time-based exit
    if dte <= config.TIME_STOP_DTE:
        return PositionRiskState(True, f"time stop ({dte} DTE remaining)")

    return PositionRiskState(False, "hold")
