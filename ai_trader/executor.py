"""Trade execution - turns LLM decisions into actual orders."""

from __future__ import annotations

import re
import time as time_module
from dataclasses import dataclass
from datetime import timedelta

from . import config
from .alpaca_client import AlpacaClient
from .brain import TradeDecision
from .db import AITradeLogger, AITradeRecord, PositionCloseRecord
from .options import OptionContract, fetch_option_chain, select_contract
from .portfolio import EquityPosition, PortfolioState
from .risk import evaluate_stock_trade_risk, evaluate_trade_risk, size_for_risk_budget
from .utils import log, now_eastern

_TERMINAL_ORDER_STATUSES = {
    "filled",
    "canceled",
    "expired",
    "rejected",
    "done_for_day",
    "stopped",
    "suspended",
}

_PENDING_ORDER_STATUSES = {
    "new",
    "accepted",
    "pending_new",
    "accepted_for_bidding",
    "partially_filled",
    "pending_replace",
    "replaced",
    "calculated",
    "held",
}


@dataclass(frozen=True)
class ExecutionResult:
    success: bool
    symbol: str
    order_id: str | None
    qty: int
    premium: float
    message: str
    filled: bool = False


def execute_trade(
    alpaca: AlpacaClient,
    decision: TradeDecision,
    portfolio: PortfolioState,
    logger: AITradeLogger,
    market_analysis: str,
) -> ExecutionResult:
    """Execute a single trade decision."""
    action = decision.action

    if action == "close_position":
        return _execute_close_option(alpaca, decision, portfolio, logger, market_analysis)
    if action in ("buy_call", "buy_put"):
        return _execute_open_option(alpaca, decision, portfolio, logger, market_analysis)
    if action == "buy_stock":
        return _execute_open_stock(alpaca, decision, portfolio, logger, market_analysis)
    if action in ("close_stock", "sell_stock"):
        return _execute_close_stock(alpaca, decision, portfolio, logger, market_analysis)

    return ExecutionResult(
        False,
        decision.underlying,
        None,
        0,
        0.0,
        f"unsupported action: {action}",
    )


def _execute_open_option(
    alpaca: AlpacaClient,
    decision: TradeDecision,
    portfolio: PortfolioState,
    logger: AITradeLogger,
    market_analysis: str,
) -> ExecutionResult:
    """Open a new options position."""
    underlying = decision.underlying
    option_type = "call" if decision.action == "buy_call" else "put"

    underlying_price = _get_underlying_price(alpaca, underlying)
    if underlying_price <= 0:
        log(f"cannot get price for {underlying}")
        return ExecutionResult(False, underlying, None, 0, 0.0, "no price data")

    chain = fetch_option_chain(alpaca, underlying, underlying_price, option_type=option_type)
    if not chain:
        log(f"no suitable options for {underlying} {option_type}")
        return ExecutionResult(False, underlying, None, 0, 0.0, "no suitable contracts")

    chain = _filter_by_expiry(chain, decision.expiry_preference)
    if not chain:
        log(f"no contracts matching expiry preference for {underlying}")
        return ExecutionResult(False, underlying, None, 0, 0.0, "no matching expiry")

    contract = select_contract(chain, underlying_price, decision.strike_preference)
    if contract is None:
        log(f"no suitable contract selected for {underlying}")
        return ExecutionResult(False, underlying, None, 0, 0.0, "no suitable contract")

    if contract.ask <= 0:
        log(f"contract {contract.symbol} has no ask price")
        return ExecutionResult(False, contract.symbol, None, 0, 0.0, "no ask price")

    existing_buy = _find_open_order(alpaca, contract.symbol, side="buy")
    if existing_buy:
        existing_status = _normalize_order_status(existing_buy)
        existing_order_id = str(existing_buy.get("id") or "")
        existing_qty = _as_int(existing_buy.get("qty"), default=0)
        log(
            f"open order already pending: {contract.symbol} "
            f"order_id={existing_order_id} status={existing_status}"
        )
        return ExecutionResult(
            True,
            contract.symbol,
            existing_order_id or None,
            existing_qty,
            0.0,
            f"order already pending ({existing_status})",
        )

    risk = evaluate_trade_risk(
        equity=portfolio.account.equity,
        cash=portfolio.account.cash,
        current_exposure=portfolio.total_exposure,
        open_positions=portfolio.open_position_count,
        option_ask=contract.ask,
        day_pl=portfolio.account.day_pl,
    )
    if not risk.approved:
        log(f"risk check FAILED for {contract.symbol}: {risk.reason}")
        _log_option_trade(
            logger, contract, decision, 0, 0.0, market_analysis, None, "risk_rejected"
        )
        return ExecutionResult(False, contract.symbol, None, 0, 0.0, f"risk: {risk.reason}")

    limit_price = min(
        round(contract.mid + (contract.ask - contract.mid) * config.OPEN_ORDER_SPREAD_FRACTION, 2),
        contract.ask,
    )
    requested_budget = portfolio.account.equity * max(decision.risk_pct, 0.0)
    max_by_risk = size_for_risk_budget(requested_budget, limit_price * 100)
    if max_by_risk <= 0:
        reason = (
            f"requested risk budget too small for 1 contract "
            f"(${requested_budget:.2f} vs ${limit_price * 100:.2f})"
        )
        log(f"risk check FAILED for {contract.symbol}: {reason}")
        _log_option_trade(
            logger, contract, decision, 0, 0.0, market_analysis, None, "risk_rejected"
        )
        return ExecutionResult(False, contract.symbol, None, 0, 0.0, f"risk: {reason}")

    qty = min(risk.max_contracts, max_by_risk)
    total_premium = qty * limit_price * 100

    log(
        f"executing: BUY {qty}x {contract.symbol} "
        f"@ ${limit_price:.2f} (mid=${contract.mid:.2f} ask=${contract.ask:.2f}) "
        f"(${total_premium:.2f} total) conviction={decision.conviction:.2f}"
    )

    try:
        order = alpaca.submit_order(
            symbol=contract.symbol,
            qty=qty,
            side="buy",
            order_type="limit",
            time_in_force="day",
            limit_price=limit_price,
        )
    except Exception as exc:
        log(f"order submission error: {exc}")
        _log_option_trade(
            logger, contract, decision, qty, total_premium, market_analysis, None, "error"
        )
        return ExecutionResult(
            False, contract.symbol, None, qty, total_premium, f"order error: {exc}"
        )

    order_id = order.get("id", "")
    status = order.get("status", "unknown")
    log(f"order submitted: {order_id} status={status}")

    filled_order = _await_fill(alpaca, order_id, timeout_seconds=30)
    fill_status = _normalize_order_status(filled_order, fallback=status)

    _log_option_trade(
        logger, contract, decision, qty, total_premium, market_analysis, order_id, fill_status
    )

    accepted = _is_order_accepted(fill_status)
    return ExecutionResult(
        success=accepted,
        symbol=contract.symbol,
        order_id=order_id,
        qty=qty,
        premium=total_premium,
        message=f"order {fill_status}",
        filled=fill_status == "filled",
    )


def _execute_open_stock(
    alpaca: AlpacaClient,
    decision: TradeDecision,
    portfolio: PortfolioState,
    logger: AITradeLogger,
    market_analysis: str,
) -> ExecutionResult:
    """Open a new equity position."""
    symbol = decision.underlying.upper()
    quote = _get_stock_quote(alpaca, symbol)
    ask = quote["ask"]
    mid = quote["mid"]
    fallback = quote["price"]
    if fallback <= 0 and ask <= 0:
        return ExecutionResult(False, symbol, None, 0, 0.0, "no stock price data")

    existing_buy = _find_open_order(alpaca, symbol, side="buy")
    if existing_buy:
        existing_status = _normalize_order_status(existing_buy)
        existing_order_id = str(existing_buy.get("id") or "")
        existing_qty = _as_int(existing_buy.get("qty"), default=0)
        log(
            f"stock order already pending: {symbol} "
            f"order_id={existing_order_id} status={existing_status}"
        )
        return ExecutionResult(
            True,
            symbol,
            existing_order_id or None,
            existing_qty,
            0.0,
            f"order already pending ({existing_status})",
        )

    if ask > 0 and mid > 0:
        limit_price = min(
            round(mid + (ask - mid) * config.OPEN_ORDER_SPREAD_FRACTION, 2),
            ask,
        )
    elif ask > 0:
        limit_price = round(ask, 2)
    else:
        limit_price = round(fallback, 2)

    risk = evaluate_stock_trade_risk(
        equity=portfolio.account.equity,
        cash=portfolio.account.cash,
        current_exposure=portfolio.total_exposure,
        open_positions=portfolio.open_position_count,
        share_price=limit_price,
        day_pl=portfolio.account.day_pl,
    )
    if not risk.approved:
        log(f"stock risk check FAILED for {symbol}: {risk.reason}")
        _log_stock_trade(
            logger=logger,
            symbol=symbol,
            decision=decision,
            qty=0,
            price=limit_price,
            total_cost=0.0,
            market_analysis=market_analysis,
            order_id=None,
            status="risk_rejected",
        )
        return ExecutionResult(False, symbol, None, 0, 0.0, f"risk: {risk.reason}")

    requested_budget = portfolio.account.equity * max(decision.risk_pct, 0.0)
    max_by_risk = size_for_risk_budget(requested_budget, limit_price)
    if max_by_risk <= 0:
        reason = (
            f"requested risk budget too small for 1 share "
            f"(${requested_budget:.2f} vs ${limit_price:.2f})"
        )
        log(f"stock risk check FAILED for {symbol}: {reason}")
        _log_stock_trade(
            logger=logger,
            symbol=symbol,
            decision=decision,
            qty=0,
            price=limit_price,
            total_cost=0.0,
            market_analysis=market_analysis,
            order_id=None,
            status="risk_rejected",
        )
        return ExecutionResult(False, symbol, None, 0, 0.0, f"risk: {reason}")

    qty = min(risk.max_shares, max_by_risk)
    total_notional = qty * limit_price

    log(
        f"executing: BUY {qty}x {symbol} @ ${limit_price:.2f} "
        f"(${total_notional:.2f} total) conviction={decision.conviction:.2f}"
    )

    try:
        order = alpaca.submit_order(
            symbol=symbol,
            qty=qty,
            side="buy",
            order_type="limit",
            time_in_force="day",
            limit_price=limit_price,
        )
    except Exception as exc:
        log(f"stock order submission error: {exc}")
        _log_stock_trade(
            logger=logger,
            symbol=symbol,
            decision=decision,
            qty=qty,
            price=limit_price,
            total_cost=total_notional,
            market_analysis=market_analysis,
            order_id=None,
            status="error",
        )
        return ExecutionResult(
            False, symbol, None, qty, total_notional, f"order error: {exc}"
        )

    order_id = order.get("id", "")
    status = order.get("status", "unknown")
    log(f"stock order submitted: {order_id} status={status}")

    filled_order = _await_fill(alpaca, order_id, timeout_seconds=30)
    fill_status = _normalize_order_status(filled_order, fallback=status)

    _log_stock_trade(
        logger=logger,
        symbol=symbol,
        decision=decision,
        qty=qty,
        price=limit_price,
        total_cost=total_notional,
        market_analysis=market_analysis,
        order_id=order_id,
        status=fill_status,
    )

    accepted = _is_order_accepted(fill_status)
    return ExecutionResult(
        success=accepted,
        symbol=symbol,
        order_id=order_id,
        qty=qty,
        premium=total_notional,
        message=f"order {fill_status}",
        filled=fill_status == "filled",
    )


def _execute_close_option(
    alpaca: AlpacaClient,
    decision: TradeDecision,
    portfolio: PortfolioState,
    logger: AITradeLogger,
    market_analysis: str,
) -> ExecutionResult:
    """Close an existing options position."""
    target = decision.target_symbol
    position = None
    if target:
        position = next((p for p in portfolio.option_positions if p.symbol == target), None)
    else:
        matching = [
            p for p in portfolio.option_positions
            if p.underlying.upper() == decision.underlying.upper()
        ]
        if not matching:
            return ExecutionResult(False, decision.underlying, None, 0, 0.0, "no matching position")
        if len(matching) == 1:
            position = matching[0]
        else:
            position = _select_close_candidate_from_reasoning(matching, decision.reasoning)
            if position is None:
                candidates = ", ".join(p.symbol for p in matching)
                return ExecutionResult(
                    False,
                    decision.underlying,
                    None,
                    0,
                    0.0,
                    f"ambiguous close target for {decision.underlying}; candidates: {candidates}",
                )

    if position is None:
        return ExecutionResult(False, target or decision.underlying, None, 0, 0.0, "position not found")

    existing_close = _find_open_order(alpaca, position.symbol, side="sell")
    if existing_close:
        existing_status = _normalize_order_status(existing_close)
        existing_order_id = str(existing_close.get("id") or "")
        log(
            f"close already pending: {position.symbol} "
            f"order_id={existing_order_id} status={existing_status}"
        )
        return ExecutionResult(
            True,
            position.symbol,
            existing_order_id or None,
            position.qty,
            0.0,
            f"close pending ({existing_status})",
        )

    try:
        quote_data = alpaca.get_option_latest_quotes([position.symbol])
        quotes = quote_data.get("quotes", quote_data) if isinstance(quote_data, dict) else {}
        quote = quotes.get(position.symbol, {})
        bid = float(quote.get("bp") or quote.get("bid_price") or 0.0)
        ask = float(quote.get("ap") or quote.get("ask_price") or 0.0)
        mid_price = round((bid + ask) / 2, 2) if bid > 0 and ask > 0 else 0.0
    except Exception:
        mid_price = 0.0

    log(
        f"closing position: SELL {position.qty}x {position.symbol} "
        f"current=${position.current_price:.2f} mid=${mid_price:.2f}"
    )

    order_id, fill_status, filled_order, err = _submit_close_order(
        alpaca=alpaca,
        symbol=position.symbol,
        qty=position.qty,
        mid_price=mid_price,
    )
    if err:
        return ExecutionResult(False, position.symbol, None, 0, 0.0, f"close error: {err}")

    _log_close_option_trade(
        logger=logger,
        position=position,
        decision=decision,
        market_analysis=market_analysis,
        order_id=order_id,
        status=fill_status,
    )

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
                order_id=order_id,
            )
        )
        log(f"position closed: {position.symbol} pnl=${pnl:.2f}")

    close_accepted = _is_order_accepted(fill_status)
    return ExecutionResult(
        success=close_accepted,
        symbol=position.symbol,
        order_id=order_id,
        qty=position.qty,
        premium=0.0,
        message=f"close {fill_status}",
        filled=fill_status == "filled",
    )


def _execute_close(
    alpaca: AlpacaClient,
    decision: TradeDecision,
    portfolio: PortfolioState,
    logger: AITradeLogger,
    market_analysis: str,
) -> ExecutionResult:
    """Backward-compatible alias for option close."""
    return _execute_close_option(alpaca, decision, portfolio, logger, market_analysis)


def _execute_close_stock(
    alpaca: AlpacaClient,
    decision: TradeDecision,
    portfolio: PortfolioState,
    logger: AITradeLogger,
    market_analysis: str,
) -> ExecutionResult:
    """Close an existing equity position."""
    target = (decision.target_symbol or decision.underlying).upper()
    position = _find_equity_position(portfolio.equity_positions, target)
    if position is None:
        return ExecutionResult(False, target, None, 0, 0.0, "stock position not found")

    qty_to_close = abs(position.qty)
    if qty_to_close <= 0:
        return ExecutionResult(False, position.symbol, None, 0, 0.0, "invalid position qty")

    existing_close = _find_open_order(alpaca, position.symbol, side="sell")
    if existing_close:
        existing_status = _normalize_order_status(existing_close)
        existing_order_id = str(existing_close.get("id") or "")
        log(
            f"stock close already pending: {position.symbol} "
            f"order_id={existing_order_id} status={existing_status}"
        )
        return ExecutionResult(
            True,
            position.symbol,
            existing_order_id or None,
            qty_to_close,
            0.0,
            f"close pending ({existing_status})",
        )

    quote = _get_stock_quote(alpaca, position.symbol)
    mid_price = quote["mid"]
    log(
        f"closing stock: SELL {qty_to_close}x {position.symbol} "
        f"current=${position.current_price:.2f} mid=${mid_price:.2f}"
    )

    order_id, fill_status, filled_order, err = _submit_close_order(
        alpaca=alpaca,
        symbol=position.symbol,
        qty=qty_to_close,
        mid_price=mid_price,
    )
    if err:
        return ExecutionResult(False, position.symbol, None, 0, 0.0, f"close error: {err}")

    if fill_status == "filled":
        exit_price = float(filled_order.get("filled_avg_price") or position.current_price)
        pnl = (exit_price - position.avg_entry_price) * position.qty
        _log_stock_trade(
            logger=logger,
            symbol=position.symbol,
            decision=decision,
            qty=qty_to_close,
            price=position.avg_entry_price,
            total_cost=0.0,
            market_analysis=market_analysis,
            order_id=order_id,
            status=fill_status,
        )
        logger.log_position_close(
            PositionCloseRecord(
                timestamp=now_eastern(),
                symbol=position.symbol,
                underlying=position.symbol,
                qty=qty_to_close,
                entry_premium=position.avg_entry_price,
                exit_premium=exit_price,
                pnl=pnl,
                reason=decision.reasoning,
                order_id=order_id,
            )
        )
        log(f"stock position closed: {position.symbol} pnl=${pnl:.2f}")
    else:
        _log_stock_trade(
            logger=logger,
            symbol=position.symbol,
            decision=decision,
            qty=qty_to_close,
            price=position.avg_entry_price,
            total_cost=0.0,
            market_analysis=market_analysis,
            order_id=order_id,
            status=fill_status,
        )

    close_accepted = _is_order_accepted(fill_status)
    return ExecutionResult(
        success=close_accepted,
        symbol=position.symbol,
        order_id=order_id,
        qty=qty_to_close,
        premium=0.0,
        message=f"close {fill_status}",
        filled=fill_status == "filled",
    )


def _submit_close_order(
    alpaca: AlpacaClient,
    symbol: str,
    qty: int,
    mid_price: float,
) -> tuple[str, str, dict, str]:
    """Submit a close order with limit->market fallback."""
    if mid_price > 0:
        try:
            order = alpaca.submit_order(
                symbol=symbol,
                qty=qty,
                side="sell",
                order_type="limit",
                time_in_force="day",
                limit_price=mid_price,
            )
        except Exception as exc:
            return "", "error", {}, str(exc)

        order_id = order.get("id", "")
        filled_order = _await_fill(alpaca, order_id, timeout_seconds=config.CLOSE_LIMIT_TIMEOUT_SECONDS)
        fill_status = _normalize_order_status(filled_order)
        filled_qty = _as_int(filled_order.get("filled_qty"), default=0)
        remaining_qty = max(qty - filled_qty, 0)

        if fill_status == "filled":
            return order_id, fill_status, filled_order, ""

        if remaining_qty <= 0:
            return order_id, "filled", filled_order, ""

        log(
            f"limit close not filled after {config.CLOSE_LIMIT_TIMEOUT_SECONDS}s "
            f"(status={fill_status}), falling back to market for remaining {remaining_qty}"
        )
        try:
            alpaca.cancel_order(order_id)
        except Exception:
            pass

    try:
        order = alpaca.submit_order(
            symbol=symbol,
            qty=remaining_qty if mid_price > 0 else qty,
            side="sell",
            order_type="market",
            time_in_force="day",
        )
    except Exception as exc:
        return "", "error", {}, str(exc)

    order_id = order.get("id", "")
    filled_order = _await_fill(alpaca, order_id, timeout_seconds=120)
    fill_status = _normalize_order_status(filled_order)
    return order_id, fill_status, filled_order, ""


def _select_close_candidate_from_reasoning(
    positions: list,
    reasoning: str,
):
    if not positions:
        return None
    if len(positions) == 1:
        return positions[0]

    text = (reasoning or "").lower()
    candidates = positions
    if " put" in f" {text}" and " call" not in f" {text}":
        filtered = [p for p in candidates if p.option_type == "put"]
        if filtered:
            candidates = filtered
    elif " call" in f" {text}" and " put" not in f" {text}":
        filtered = [p for p in candidates if p.option_type == "call"]
        if filtered:
            candidates = filtered

    if len(candidates) == 1:
        return candidates[0]

    strike_hint = _extract_strike_hint(text)
    if strike_hint is not None:
        ranked = sorted(candidates, key=lambda p: abs(p.strike - strike_hint))
        if ranked:
            return ranked[0]

    return None


def _extract_strike_hint(text: str) -> float | None:
    patterns = (
        r"\$([0-9]+(?:\.[0-9]+)?)\s*(?:call|put)s?\b",
        r"(?:call|put)s?\s*\$([0-9]+(?:\.[0-9]+)?)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except (TypeError, ValueError):
                pass
    return None


def _log_close_option_trade(
    logger: AITradeLogger,
    position,
    decision: TradeDecision,
    market_analysis: str,
    order_id: str | None,
    status: str,
) -> None:
    # For close rows, store entry premium so async reconciliation can compute
    # realized P&L accurately when an order fills after timeout.
    logger.log_trade(
        AITradeRecord(
            timestamp=now_eastern(),
            symbol=position.symbol,
            underlying=position.underlying,
            option_type=position.option_type,
            strike=position.strike,
            expiration=position.expiration,
            action=decision.action,
            qty=position.qty,
            premium=position.avg_entry_price,
            total_cost=0.0,
            conviction=decision.conviction,
            reasoning=decision.reasoning,
            market_analysis=market_analysis,
            order_id=order_id,
            status=status,
        )
    )


def reconcile_pending_orders(
    alpaca: AlpacaClient,
    logger: AITradeLogger,
    limit: int = 100,
) -> tuple[int, int]:
    """Reconcile pending order statuses and backfill async close fills."""
    pending = logger.get_pending_trades(limit=limit)
    status_updates = 0
    closes_backfilled = 0
    seen_orders: set[str] = set()

    for trade in pending:
        order_id = str(trade.get("order_id") or "")
        if not order_id or order_id in seen_orders:
            continue
        seen_orders.add(order_id)

        try:
            order = alpaca.get_order(order_id)
        except Exception:
            continue

        old_status = str(trade.get("status") or "unknown").lower()
        new_status = _normalize_order_status(order, fallback=old_status)
        if new_status != old_status:
            status_updates += logger.update_trade_status(order_id, new_status)

        if trade.get("action") not in ("close_position", "close_stock", "sell_stock"):
            continue
        if new_status != "filled":
            continue
        if logger.has_position_close_for_order(order_id):
            continue

        qty = _as_int(order.get("filled_qty"), default=_as_int(trade.get("qty"), default=0))
        if qty <= 0:
            continue

        entry_premium = _as_float(trade.get("premium"), default=0.0)
        exit_premium = _as_float(order.get("filled_avg_price"), default=0.0)
        if exit_premium <= 0:
            exit_premium = entry_premium

        option_type = str(trade.get("option_type") or "")
        multiplier = 1 if option_type == "stock" else 100
        pnl = (exit_premium - entry_premium) * qty * multiplier

        symbol = str(trade.get("symbol") or "")
        underlying = str(trade.get("underlying") or symbol)
        logger.log_position_close(
            PositionCloseRecord(
                timestamp=now_eastern(),
                symbol=symbol,
                underlying=underlying,
                qty=qty,
                entry_premium=entry_premium,
                exit_premium=exit_premium,
                pnl=pnl,
                reason=str(trade.get("reasoning") or "async close fill reconciliation"),
                order_id=order_id,
            )
        )
        closes_backfilled += 1

    return status_updates, closes_backfilled


def _find_equity_position(
    positions: list[EquityPosition],
    target: str,
) -> EquityPosition | None:
    target = target.upper()
    for position in positions:
        if position.symbol.upper() == target:
            return position
    return None


def check_and_close_risk_exits(
    alpaca: AlpacaClient,
    portfolio: PortfolioState,
    logger: AITradeLogger,
) -> list[ExecutionResult]:
    """Check all option positions for risk-based exits."""
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
        result = _execute_close_option(alpaca, decision, portfolio, logger, "auto risk exit")
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


def _get_stock_quote(alpaca: AlpacaClient, symbol: str) -> dict[str, float]:
    try:
        data = alpaca.get_snapshots([symbol])
        snap = data.get(symbol, {})
        quote = snap.get("latestQuote") or snap.get("latest_quote") or {}
        trade = snap.get("latestTrade") or snap.get("latest_trade") or {}
        bar = snap.get("dailyBar") or snap.get("daily_bar") or {}
        bid = float(quote.get("bp") or quote.get("bid_price") or 0.0)
        ask = float(quote.get("ap") or quote.get("ask_price") or 0.0)
        mid = round((bid + ask) / 2, 2) if bid > 0 and ask > 0 else 0.0
        price = float(trade.get("p") or trade.get("price") or bar.get("c") or bar.get("close") or 0.0)
        return {"bid": bid, "ask": ask, "mid": mid, "price": price}
    except Exception:
        return {"bid": 0.0, "ask": 0.0, "mid": 0.0, "price": 0.0}


def _filter_by_expiry(
    contracts: list[OptionContract], preference: str
) -> list[OptionContract]:
    today = now_eastern().date()
    if preference == "this_week":
        cutoff = today + timedelta(days=7)
        return [c for c in contracts if c.expiration <= cutoff]
    if preference == "next_week":
        start = today + timedelta(days=5)
        cutoff = today + timedelta(days=14)
        return [c for c in contracts if start <= c.expiration <= cutoff] or [
            c for c in contracts if c.expiration <= cutoff
        ]
    return [c for c in contracts if c.dte >= 14]


def _await_fill(alpaca: AlpacaClient, order_id: str, timeout_seconds: int = 30) -> dict:
    deadline = now_eastern().timestamp() + timeout_seconds
    while now_eastern().timestamp() < deadline:
        try:
            order = alpaca.get_order(order_id)
            status = _normalize_order_status(order)
            if _is_terminal_order_status(status):
                return order
        except Exception:
            pass
        time_module.sleep(2)
    try:
        return alpaca.get_order(order_id)
    except Exception:
        return {"status": "unknown", "id": order_id}


def _find_open_order(
    alpaca: AlpacaClient,
    symbol: str,
    side: str | None = None,
) -> dict | None:
    """Return an open order for symbol (+optional side) if one exists."""
    try:
        open_orders = alpaca.get_orders(status="open", limit=200)
    except Exception as exc:
        log(f"open order lookup failed for {symbol}: {exc}")
        return None

    target_symbol = symbol.upper()
    target_side = side.lower() if side else None
    for order in open_orders:
        if str(order.get("symbol", "")).upper() != target_symbol:
            continue
        order_side = str(order.get("side", "")).lower()
        if target_side and order_side != target_side:
            continue
        status = _normalize_order_status(order)
        if _is_order_pending(status):
            return order
    return None


def _is_terminal_order_status(status: str) -> bool:
    return status in _TERMINAL_ORDER_STATUSES


def _is_order_pending(status: str) -> bool:
    return status in _PENDING_ORDER_STATUSES


def _is_order_accepted(status: str) -> bool:
    return status == "filled" or _is_order_pending(status)


def _normalize_order_status(order: dict | None, fallback: str = "unknown") -> str:
    if not order:
        return str(fallback or "unknown").lower()
    raw = order.get("status", fallback)
    return str(raw or fallback or "unknown").lower()


def _as_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _log_option_trade(
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


def _log_stock_trade(
    logger: AITradeLogger,
    symbol: str,
    decision: TradeDecision,
    qty: int,
    price: float,
    total_cost: float,
    market_analysis: str,
    order_id: str | None,
    status: str,
) -> None:
    logger.log_trade(
        AITradeRecord(
            timestamp=now_eastern(),
            symbol=symbol,
            underlying=symbol,
            option_type="stock",
            strike=0.0,
            expiration="",
            action=decision.action,
            qty=qty,
            premium=price,
            total_cost=total_cost,
            conviction=decision.conviction,
            reasoning=decision.reasoning,
            market_analysis=market_analysis,
            order_id=order_id,
            status=status,
        )
    )
