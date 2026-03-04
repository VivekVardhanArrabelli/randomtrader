"""Trade execution - turns LLM decisions into actual orders."""

from __future__ import annotations

import re
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

    # If we already have an open buy order for this exact contract, don't
    # submit a duplicate. Treat this as accepted/pending execution.
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
    # Better pricing: bid at spread_fraction above mid instead of at ask
    limit_price = min(
        round(contract.mid + (contract.ask - contract.mid) * config.OPEN_ORDER_SPREAD_FRACTION, 2),
        contract.ask,
    )
    max_by_risk = int(
        (portfolio.account.equity * decision.risk_pct) / (limit_price * 100)
    )
    qty = min(risk.max_contracts, max(1, max_by_risk))
    total_premium = qty * limit_price * 100

    log(
        f"executing: BUY {qty}x {contract.symbol} "
        f"@ ${limit_price:.2f} (mid=${contract.mid:.2f} ask=${contract.ask:.2f}) "
        f"(${total_premium:.2f} total) conviction={decision.conviction:.2f}"
    )

    # Submit order
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
    fill_status = _normalize_order_status(filled_order, fallback=status)

    _log_trade(
        logger, contract, decision, qty, total_premium, market_analysis,
        order_id, fill_status,
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


def _execute_close(
    alpaca: AlpacaClient,
    decision: TradeDecision,
    portfolio: PortfolioState,
    logger: AITradeLogger,
    market_analysis: str,
) -> ExecutionResult:
    """Close an existing options position."""

    target = decision.target_symbol
    position: OptionPosition | None = None
    if target:
        position = next(
            (p for p in portfolio.option_positions if p.symbol == target), None
        )
    else:
        matching = [
            p for p in portfolio.option_positions
            if p.underlying.upper() == decision.underlying.upper()
        ]
        if not matching:
            return ExecutionResult(
                False, decision.underlying, None, 0, 0.0, "no matching position"
            )
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
        return ExecutionResult(
            False, target or decision.underlying, None, 0, 0.0, "position not found"
        )

    # Avoid duplicate close submissions when an open sell order already exists.
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

    # Get fresh quote for mid-price limit order
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

    # Try limit at mid first, fallback to market after timeout
    if mid_price > 0:
        try:
            order = alpaca.submit_order(
                symbol=position.symbol,
                qty=position.qty,
                side="sell",
                order_type="limit",
                time_in_force="day",
                limit_price=mid_price,
            )
        except Exception as exc:
            log(f"close limit order error: {exc}")
            return ExecutionResult(
                False, position.symbol, None, 0, 0.0, f"close error: {exc}"
            )

        order_id = order.get("id", "")
        filled_order = _await_fill(alpaca, order_id, timeout_seconds=config.CLOSE_LIMIT_TIMEOUT_SECONDS)
        fill_status = _normalize_order_status(filled_order)
        filled_qty = _as_int(filled_order.get("filled_qty"), default=0)
        remaining_qty = max(position.qty - filled_qty, 0)

        if fill_status != "filled":
            # Cancel unfilled limit and resubmit as market
            if remaining_qty <= 0:
                fill_status = "filled"
            else:
                log(
                    f"limit close not filled after {config.CLOSE_LIMIT_TIMEOUT_SECONDS}s "
                    f"(status={fill_status}), falling back to market for remaining {remaining_qty}"
                )
            try:
                alpaca.cancel_order(order_id)
            except Exception:
                pass
            if remaining_qty > 0:
                try:
                    order = alpaca.submit_order(
                        symbol=position.symbol,
                        qty=remaining_qty,
                        side="sell",
                        order_type="market",
                        time_in_force="day",
                    )
                except Exception as exc:
                    log(f"close market fallback error: {exc}")
                    return ExecutionResult(
                        False, position.symbol, None, 0, 0.0, f"close error: {exc}"
                    )
                order_id = order.get("id", "")
                # Market options orders can still settle asynchronously.
                filled_order = _await_fill(alpaca, order_id, timeout_seconds=120)
                fill_status = _normalize_order_status(filled_order)
    else:
        # No mid price available — go straight to market
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
        filled_order = _await_fill(alpaca, order_id, timeout_seconds=120)
        fill_status = _normalize_order_status(filled_order)

    _log_close_trade(
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
            status = _normalize_order_status(order)
            if _is_terminal_order_status(status):
                return order
        except Exception:
            pass
        time_module.sleep(2)
    # Try to get final state
    try:
        return alpaca.get_order(order_id)
    except Exception:
        return {"status": "unknown", "id": order_id}


def _select_close_candidate_from_reasoning(
    positions: list[OptionPosition],
    reasoning: str,
) -> OptionPosition | None:
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


def _log_close_trade(
    logger: AITradeLogger,
    position: OptionPosition,
    decision: TradeDecision,
    market_analysis: str,
    order_id: str | None,
    status: str,
) -> None:
    # For close rows, store the position's entry premium so asynchronous
    # reconciliation can compute realized P&L accurately once the order fills.
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

        if trade.get("action") != "close_position":
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
        pnl = (exit_premium - entry_premium) * qty * 100

        logger.log_position_close(
            PositionCloseRecord(
                timestamp=now_eastern(),
                symbol=str(trade.get("symbol") or ""),
                underlying=str(trade.get("underlying") or ""),
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
