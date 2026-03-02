"""Manual test trade runner for Alpaca paper trading."""

from __future__ import annotations

import argparse
import math
import time as time_module
from datetime import datetime, time, timedelta
from pathlib import Path

from dotenv import load_dotenv

from . import config
from .alpaca_client import AlpacaClient
from .db import EntryAttemptRecord, TEST_TRADE_DB_PATH, TradeLogger, TradeRecord
from .utils import EASTERN_TZ, now_eastern, parse_timestamp


def _log(message: str) -> None:
    timestamp = now_eastern().strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"[{timestamp}] {message}")


def _await_order_fill(alpaca: AlpacaClient, order_id: str) -> dict:
    deadline = now_eastern() + timedelta(seconds=config.ENTRY_FILL_TIMEOUT_SECONDS)
    last_order: dict = {}
    while now_eastern() < deadline:
        order = alpaca.get_order(order_id)
        last_order = order
        status = order.get("status")
        if status in {"filled", "canceled", "expired", "rejected"}:
            return order
        time_module.sleep(config.ENTRY_POLL_INTERVAL_SECONDS)
    try:
        alpaca.cancel_order(order_id)
    except RuntimeError:
        pass
    time_module.sleep(config.ENTRY_POLL_INTERVAL_SECONDS)
    return alpaca.get_order(order_id)


def _latest_filled_sell_order(orders: list[dict], symbol: str) -> dict | None:
    latest_order = None
    latest_time = None
    for order in orders:
        if order.get("symbol") != symbol or order.get("side") != "sell":
            continue
        filled_at = parse_timestamp(order.get("filled_at"))
        if filled_at is None:
            continue
        if latest_time is None or filled_at > latest_time:
            latest_time = filled_at
            latest_order = order
    return latest_order


def _get_snapshot(alpaca: AlpacaClient, symbol: str) -> dict | None:
    data = alpaca.get_snapshots([symbol])
    return data.get(symbol)


def _get_limit_price(snapshot: dict) -> float:
    latest_quote = snapshot.get("latestQuote") or {}
    latest_trade = snapshot.get("latestTrade") or {}
    daily_bar = snapshot.get("dailyBar") or {}
    ask = float(latest_quote.get("ap") or 0.0)
    last_price = float(latest_trade.get("p") or daily_bar.get("c") or 0.0)
    return ask if ask > 0 else last_price


def _log_entry_attempt(
    logger: TradeLogger,
    signal_time: datetime,
    signal_price: float,
    order: dict,
    rejection_reason: str | None = None,
) -> None:
    filled_qty = float(order.get("filled_qty") or 0.0)
    filled_avg_price = order.get("filled_avg_price")
    avg_price = float(filled_avg_price) if filled_avg_price else None
    filled_at = parse_timestamp(order.get("filled_at"))
    status = order.get("status", "unknown")
    if filled_qty > 0 and status != "filled":
        status = "partial"
    slippage_pct = None
    if avg_price is not None and signal_price > 0:
        slippage_pct = (avg_price - signal_price) / signal_price
    logger.log_entry_attempt(
        EntryAttemptRecord(
            signal_time=signal_time,
            symbol=order.get("symbol", ""),
            signal_price=signal_price,
            order_id=order.get("id"),
            status=status,
            filled_qty=filled_qty,
            filled_avg_price=avg_price,
            fill_time=filled_at,
            slippage_pct=slippage_pct,
            rejection_reason=rejection_reason,
        )
    )


def _submit_exit_oco(
    alpaca: AlpacaClient,
    symbol: str,
    qty: float,
    entry_price: float,
    take_profit_pct: float,
    stop_loss_pct: float,
) -> dict:
    take_profit_price = entry_price * (1 + take_profit_pct)
    stop_price = entry_price * (1 - stop_loss_pct)
    return alpaca.submit_order(
        symbol=symbol,
        qty=qty,
        side="sell",
        order_type="limit",
        time_in_force=config.ORDER_TIME_IN_FORCE,
        order_class="oco",
        take_profit={"limit_price": str(round(take_profit_price, 2))},
        stop_loss={"stop_price": str(round(stop_price, 2))},
    )


def _force_time_exit(alpaca: AlpacaClient, symbol: str, qty: float, open_orders: list[dict]) -> dict:
    for order in open_orders:
        try:
            alpaca.cancel_order(order["id"])
        except RuntimeError:
            continue
    order = alpaca.submit_order(
        symbol=symbol,
        qty=qty,
        side="sell",
        order_type="market",
        time_in_force=config.ORDER_TIME_IN_FORCE,
    )
    return _await_order_fill(alpaca, order["id"])


def _wait_for_exit(
    alpaca: AlpacaClient,
    logger: TradeLogger,
    symbol: str,
    entry_price: float,
    qty: float,
    max_minutes: int,
    force_exit_on_timeout: bool,
) -> None:
    deadline = None
    if max_minutes > 0:
        deadline = now_eastern() + timedelta(minutes=max_minutes)
    time_stop = time(config.TIME_STOP_HOUR, config.TIME_STOP_MINUTE, tzinfo=EASTERN_TZ)
    while True:
        if deadline and now_eastern() >= deadline:
            if not force_exit_on_timeout:
                _log("exit monitor timeout")
                return
            open_orders = alpaca.get_orders(status="open")
            open_sell_orders = [
                order
                for order in open_orders
                if order.get("symbol") == symbol and order.get("side") == "sell"
            ]
            order = _force_time_exit(alpaca, symbol, qty, open_sell_orders)
            exit_price = float(order.get("filled_avg_price") or 0.0)
            exit_time = parse_timestamp(order.get("filled_at")) or now_eastern()
            pnl = (exit_price - entry_price) * qty
            logger.log_trade_exit(
                symbol=symbol,
                exit_price=exit_price,
                exit_time=exit_time,
                pnl=pnl,
                exit_reason="timeout_force_exit",
            )
            _log(
                f"timeout force exit {symbol} entry={entry_price:.2f} exit={exit_price:.2f} pnl={pnl:.2f}"
            )
            return
        positions = alpaca.get_positions()
        if symbol not in {position["symbol"] for position in positions}:
            closed = alpaca.get_orders(status="closed", limit=100)
            order = _latest_filled_sell_order(closed, symbol)
            if order:
                exit_price = float(order.get("filled_avg_price") or 0.0)
                exit_time = parse_timestamp(order.get("filled_at")) or now_eastern()
                pnl = (exit_price - entry_price) * qty
                logger.log_trade_exit(
                    symbol=symbol,
                    exit_price=exit_price,
                    exit_time=exit_time,
                    pnl=pnl,
                    exit_reason="broker_exit",
                )
                _log(
                    f"exit filled {symbol} entry={entry_price:.2f} exit={exit_price:.2f} pnl={pnl:.2f}"
                )
            return
        if now_eastern().timetz() >= time_stop:
            open_orders = alpaca.get_orders(status="open")
            open_sell_orders = [order for order in open_orders if order.get("symbol") == symbol and order.get("side") == "sell"]
            order = _force_time_exit(alpaca, symbol, qty, open_sell_orders)
            exit_price = float(order.get("filled_avg_price") or 0.0)
            exit_time = parse_timestamp(order.get("filled_at")) or now_eastern()
            pnl = (exit_price - entry_price) * qty
            logger.log_trade_exit(
                symbol=symbol,
                exit_price=exit_price,
                exit_time=exit_time,
                pnl=pnl,
                exit_reason="time_stop",
            )
            _log(
                f"time stop exit {symbol} entry={entry_price:.2f} exit={exit_price:.2f} pnl={pnl:.2f}"
            )
            return
        time_module.sleep(config.POLL_INTERVAL_SECONDS)


def run() -> None:
    parser = argparse.ArgumentParser(description="Submit a manual test trade (paper).")
    parser.add_argument("--symbol", required=True, help="Ticker symbol to trade.")
    parser.add_argument("--qty", type=float, default=1.0, help="Quantity to trade.")
    parser.add_argument("--limit", type=float, default=None, help="Limit price override.")
    parser.add_argument(
        "--take-profit-pct",
        type=float,
        default=config.PROFIT_TARGET_PCT,
        help="Override take-profit percent (e.g. 0.07 for 7%%).",
    )
    parser.add_argument(
        "--stop-loss-pct",
        type=float,
        default=config.STOP_LOSS_PCT,
        help="Override stop-loss percent (e.g. 0.03 for 3%%).",
    )
    parser.add_argument(
        "--wait-exit",
        action="store_true",
        help="Wait for the exit to fill and log PnL.",
    )
    parser.add_argument(
        "--max-minutes",
        type=int,
        default=120,
        help="Max minutes to wait for exit before stopping.",
    )
    parser.add_argument(
        "--force-exit-on-timeout",
        action="store_true",
        help="If --wait-exit times out, cancel open sells and market-close the position.",
    )
    args = parser.parse_args()

    env_path = Path(__file__).with_name(".env")
    load_dotenv(env_path, override=True)
    alpaca = AlpacaClient.from_env()
    logger = TradeLogger(TEST_TRADE_DB_PATH)

    snapshot = _get_snapshot(alpaca, args.symbol)
    if not snapshot:
        _log(f"no snapshot for {args.symbol}")
        return

    limit_price = args.limit
    if limit_price is None:
        limit_price = _get_limit_price(snapshot)
    if limit_price <= 0:
        _log("invalid limit price")
        return

    qty = math.floor(args.qty)
    if qty <= 0:
        _log("invalid qty")
        return

    signal_time = now_eastern()
    order = alpaca.submit_order(
        symbol=args.symbol,
        qty=qty,
        side="buy",
        order_type="limit",
        time_in_force=config.ORDER_TIME_IN_FORCE,
        limit_price=limit_price,
    )
    rejection_reason = order.get("reject_reason") or order.get("rejected_reason")
    if order.get("status") in {"rejected", "canceled"}:
        _log_entry_attempt(logger, signal_time, limit_price, order, rejection_reason)
        _log(f"entry rejected: {order.get('status')}")
        return

    order = _await_order_fill(alpaca, order["id"])
    _log_entry_attempt(logger, signal_time, limit_price, order, rejection_reason)
    filled_qty = float(order.get("filled_qty") or 0.0)
    avg_price = order.get("filled_avg_price")
    filled_avg_price = float(avg_price) if avg_price else None

    if filled_qty <= 0 or filled_avg_price is None:
        _log("entry not filled")
        return

    fill_time = parse_timestamp(order.get("filled_at")) or now_eastern()
    logger.log_trades(
        [
            TradeRecord(
                entry_time=fill_time,
                symbol=args.symbol,
                entry_price=filled_avg_price,
                qty=filled_qty,
                exit_price=None,
                exit_time=None,
                pnl=None,
                exit_reason=None,
            )
        ]
    )
    _log(f"entry filled {args.symbol} qty={filled_qty} price={filled_avg_price:.2f}")

    try:
        _submit_exit_oco(
            alpaca,
            args.symbol,
            filled_qty,
            filled_avg_price,
            take_profit_pct=float(args.take_profit_pct),
            stop_loss_pct=float(args.stop_loss_pct),
        )
        _log("exit OCO submitted")
    except RuntimeError as exc:
        _log(f"exit OCO submit failed: {exc}")

    if args.wait_exit:
        _wait_for_exit(
            alpaca,
            logger,
            args.symbol,
            filled_avg_price,
            filled_qty,
            args.max_minutes,
            force_exit_on_timeout=bool(args.force_exit_on_timeout),
        )


if __name__ == "__main__":
    run()
