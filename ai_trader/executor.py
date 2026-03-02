"""Trade execution - turns LLM decisions into actual orders."""

from __future__ import annotations

import time as time_module
from dataclasses import dataclass
from datetime import date, timedelta

from . import config
from .alpaca_client import AlpacaClient
from .brain import TradeDecision
from .db import AITradeLogger, AITradeRecord, PositionCloseRecord
from .options import OptionContract, fetch_option_chain, select_contract
from .portfolio import OptionPosition, PortfolioState
from .risk import evaluate_trade_risk
from .utils import log, now_eastern


@dataclass(frozen=True)
class ExecutionResult:
    success: bool
    symbol: str
    order_id: str | None
    qty: int
    premium: float
    message: str


def execute_trade(
    alpaca: AlpacaClient,
    decision: TradeDecision,
    portfolio: PortfolioState,
    logger: AITradeLogger,
    market_analysis: str,
) -> ExecutionResult:
    """Execute a single trade decision."""

    if decision.action == "close_position":
        return _execute_close(alpaca, decision, portfolio, logger, market_analysis)

    return _execute_open(alpaca, decision, portfolio, logger, market_analysis)


def _execute_open(
    alpaca: AlpacaClient,
    decision: TradeDecision,
    portfolio: PortfolioState,
    logger: AITradeLogger,
    market_analysis: str,
) -> ExecutionResult:
    """Open a new options position."""

    underlying = decision.underlying
    option_type = "call" if decision.action == "buy_call" else "put"

    # Get underlying price
    underlying_price = _get_underlying_price(alpaca, underlying)
    if underlying_price <= 0:
        log(f"cannot get price for {underlying}")
        return ExecutionResult(False, underlying, None, 0, 0.0, "no price data")

    # Fetch options chain
    chain = fetch_option_chain(
        alpaca, underlying, underlying_price, option_type=option_type
    )
    if not chain:
        log(f"no suitable options for {underlying} {option_type}")
        return ExecutionResult(
            False, underlying, None, 0, 0.0, "no suitable contracts"
        )

    # Filter by expiry preference
    chain = _filter_by_expiry(chain, decision.expiry_preference)
    if not chain:
        log(f"no contracts matching expiry preference for {underlying}")
        return ExecutionResult(
            False, underlying, None, 0, 0.0, "no matching expiry"
        )

    # Select best contract
    contract = select_contract(
        chain, underlying_price, decision.strike_preference
    )
    if contract is None:
        log(f"no suitable contract selected for {underlying}")
        return ExecutionResult(
            False, underlying, None, 0, 0.0, "no suitable contract"
        )

    if contract.ask <= 0:
        log(f"contract {contract.symbol} has no ask price")
        return ExecutionResult(
            False, contract.symbol, None, 0, 0.0, "no ask price"
        )

    # Risk check
    risk = evaluate_trade_risk(
        equity=portfolio.account.equity,
        cash=portfolio.account.cash,
        current_exposure=portfolio.total_options_exposure,
        open_positions=portfolio.open_option_count,
        option_ask=contract.ask,
        day_pl=portfolio.account.day_pl,
    )
    if not risk.approved:
        log(f"risk check FAILED for {contract.symbol}: {risk.reason}")
        _log_trade(
            logger, contract, decision, 0, 0.0, market_analysis,
            None, "risk_rejected",
        )
        return ExecutionResult(
            False, contract.symbol, None, 0, 0.0, f"risk: {risk.reason}"
        )

    # Cap contracts by decision's risk_pct
    max_by_risk = int(
        (portfolio.account.equity * decision.risk_pct) / (contract.ask * 100)
    )
    qty = min(risk.max_contracts, max(1, max_by_risk))
    total_premium = qty * contract.ask * 100

    log(
        f"executing: BUY {qty}x {contract.symbol} "
        f"@ ${contract.ask:.2f} (${total_premium:.2f} total) "
        f"conviction={decision.conviction:.2f}"
    )

    # Submit order
    try:
        order = alpaca.submit_order(
            symbol=contract.symbol,
            qty=qty,
            side="buy",
            order_type="limit",
            time_in_force="day",
            limit_price=contract.ask,
        )
    except Exception as exc:
        log(f"order submission error: {exc}")
        _log_trade(
            logger, contract, decision, qty, total_premium, market_analysis,
            None, "error",
        )
        return ExecutionResult(
            False, contract.symbol, None, qty, total_premium, f"order error: {exc}"
        )

    order_id = order.get("id", "")
    status = order.get("status", "unknown")
    log(f"order submitted: {order_id} status={status}")

    # Wait briefly for fill
    filled_order = _await_fill(alpaca, order_id, timeout_seconds=30)
    fill_status = filled_order.get("status", status)

    _log_trade(
        logger, contract, decision, qty, total_premium, market_analysis,
        order_id, fill_status,
    )

    return ExecutionResult(
        success=fill_status == "filled",
        symbol=contract.symbol,
        order_id=order_id,
        qty=qty,
        premium=total_premium,
        message=f"order {fill_status}",
    )


def _execute_close(
    alpaca: AlpacaClient,
    decision: TradeDecision,
    portfolio: PortfolioState,
    logger: AITradeLogger,
    market_analysis: str,
) -> ExecutionResult:
    """Close an existing options position."""

    target = decision.target_symbol
    if not target:
        # Find position by underlying
        matching = [
            p for p in portfolio.option_positions
            if p.underlying.upper() == decision.underlying.upper()
        ]
        if not matching:
            return ExecutionResult(
                False, decision.underlying, None, 0, 0.0, "no matching position"
            )
        target = matching[0].symbol

    position = next(
        (p for p in portfolio.option_positions if p.symbol == target), None
    )
    if position is None:
        return ExecutionResult(
            False, target, None, 0, 0.0, "position not found"
        )

    log(
        f"closing position: SELL {position.qty}x {position.symbol} "
        f"current=${position.current_price:.2f}"
    )

    try:
        order = alpaca.submit_order(
            symbol=position.symbol,
            qty=position.qty,
            side="sell",
            order_type="market",
            time_in_force="day",
        )
    except Exception as exc:
        log(f"close order error: {exc}")
        return ExecutionResult(
            False, position.symbol, None, 0, 0.0, f"close error: {exc}"
        )

    order_id = order.get("id", "")
    filled_order = _await_fill(alpaca, order_id, timeout_seconds=30)
    fill_status = filled_order.get("status", "unknown")

    if fill_status == "filled":
        exit_price = float(filled_order.get("filled_avg_price") or position.current_price)
        pnl = (exit_price - position.avg_entry_price) * position.qty * 100
        logger.log_position_close(
            PositionCloseRecord(
                timestamp=now_eastern(),
                symbol=position.symbol,
                underlying=position.underlying,
                qty=position.qty,
                entry_premium=position.avg_entry_price,
                exit_premium=exit_price,
                pnl=pnl,
                reason=decision.reasoning,
            )
        )
        log(f"position closed: {position.symbol} pnl=${pnl:.2f}")

    return ExecutionResult(
        success=fill_status == "filled",
        symbol=position.symbol,
        order_id=order_id,
        qty=position.qty,
        premium=0.0,
        message=f"close {fill_status}",
    )


def check_and_close_risk_exits(
    alpaca: AlpacaClient,
    portfolio: PortfolioState,
    logger: AITradeLogger,
) -> list[ExecutionResult]:
    """Check all positions for risk-based exits (stop loss, take profit, time stop)."""
    from .risk import evaluate_position_risk

    results: list[ExecutionResult] = []
    for position in portfolio.option_positions:
        risk_state = evaluate_position_risk(
            entry_premium=position.avg_entry_price,
            current_premium=position.current_price,
            dte=position.dte,
        )
        if not risk_state.should_close:
            continue

        log(f"risk exit triggered for {position.symbol}: {risk_state.reason}")
        decision = TradeDecision(
            action="close_position",
            underlying=position.underlying,
            strike_preference="",
            expiry_preference="",
            conviction=1.0,
            risk_pct=0.0,
            reasoning=f"auto-exit: {risk_state.reason}",
            target_symbol=position.symbol,
        )
        result = _execute_close(alpaca, decision, portfolio, logger, "auto risk exit")
        results.append(result)

    return results


def _get_underlying_price(alpaca: AlpacaClient, symbol: str) -> float:
    try:
        data = alpaca.get_snapshots([symbol])
        snap = data.get(symbol, {})
        trade = snap.get("latestTrade") or snap.get("latest_trade") or {}
        price = float(trade.get("p") or trade.get("price") or 0.0)
        if price > 0:
            return price
        bar = snap.get("dailyBar") or snap.get("daily_bar") or {}
        return float(bar.get("c") or bar.get("close") or 0.0)
    except Exception as exc:
        log(f"price fetch error for {symbol}: {exc}")
        return 0.0


def _filter_by_expiry(
    contracts: list[OptionContract], preference: str
) -> list[OptionContract]:
    today = now_eastern().date()
    if preference == "this_week":
        # This Friday or next 5 trading days
        cutoff = today + timedelta(days=7)
        return [c for c in contracts if c.expiration <= cutoff]
    elif preference == "next_week":
        start = today + timedelta(days=5)
        cutoff = today + timedelta(days=14)
        return [c for c in contracts if start <= c.expiration <= cutoff] or [
            c for c in contracts if c.expiration <= cutoff
        ]
    else:  # monthly
        return [c for c in contracts if c.dte >= 14]


def _await_fill(alpaca: AlpacaClient, order_id: str, timeout_seconds: int = 30) -> dict:
    deadline = now_eastern().timestamp() + timeout_seconds
    while now_eastern().timestamp() < deadline:
        try:
            order = alpaca.get_order(order_id)
            status = order.get("status", "")
            if status in ("filled", "canceled", "expired", "rejected"):
                return order
        except Exception:
            pass
        time_module.sleep(2)
    # Try to get final state
    try:
        return alpaca.get_order(order_id)
    except Exception:
        return {"status": "unknown", "id": order_id}


def _log_trade(
    logger: AITradeLogger,
    contract: OptionContract,
    decision: TradeDecision,
    qty: int,
    total_premium: float,
    market_analysis: str,
    order_id: str | None,
    status: str,
) -> None:
    logger.log_trade(
        AITradeRecord(
            timestamp=now_eastern(),
            symbol=contract.symbol,
            underlying=contract.underlying,
            option_type=contract.option_type,
            strike=contract.strike,
            expiration=contract.expiration.isoformat(),
            action=decision.action,
            qty=qty,
            premium=contract.ask,
            total_cost=total_premium,
            conviction=decision.conviction,
            reasoning=decision.reasoning,
            market_analysis=market_analysis,
            order_id=order_id,
            status=status,
        )
    )
