"""Live (paper) trading runner using Polygon + Alpaca."""

from __future__ import annotations

from dataclasses import dataclass

from .alpaca_client import AlpacaClient
from .db import TradeLogger
from .env import get_env
from .main import TraderEngine
from .polygon_client import PolygonClient
from .risk import evaluate_daily_risk, within_trading_window
from .utils import AccountSnapshot, now_eastern


@dataclass(frozen=True)
class LiveConfig:
    alpaca_key: str
    alpaca_secret: str
    alpaca_base_url: str
    polygon_key: str


def load_live_config() -> LiveConfig:
    return LiveConfig(
        alpaca_key=get_env("ALPACA_API_KEY"),
        alpaca_secret=get_env("ALPACA_API_SECRET"),
        alpaca_base_url=get_env("ALPACA_BASE_URL"),
        polygon_key=get_env("POLYGON_API_KEY"),
    )


def build_account_snapshot(client: AlpacaClient) -> AccountSnapshot:
    account = client.get_account()
    equity = float(account.equity)
    last_equity = float(account.last_equity) if account.last_equity is not None else equity
    day_pl = equity - last_equity
    return AccountSnapshot(
        equity=equity,
        cash=float(account.cash),
        buying_power=float(account.buying_power),
        day_pl=day_pl,
    )


def run_paper_trade_once() -> None:
    live_config = load_live_config()
    alpaca = AlpacaClient(live_config.alpaca_key, live_config.alpaca_secret, live_config.alpaca_base_url)
    polygon = PolygonClient(live_config.polygon_key)

    account = build_account_snapshot(alpaca)
    if not within_trading_window(now_eastern()):
        return

    risk_state = evaluate_daily_risk(account.equity, account.day_pl)
    if risk_state.daily_loss_limit_hit:
        return

    snapshots = polygon.fetch_gainers()
    if not snapshots:
        return

    logger = TradeLogger()
    engine = TraderEngine(logger)

    market_snapshots = [snapshot.market for snapshot in snapshots]
    quote_snapshots = {snapshot.quote.symbol: snapshot.quote for snapshot in snapshots}

    candidates = engine.run_scan(market_snapshots)
    decisions = engine.evaluate_entries(candidates, quote_snapshots, account)

    for decision in decisions:
        if not decision.should_enter:
            continue
        quote = quote_snapshots[decision.symbol]
        qty = round(decision.position_size, 2)
        if qty <= 0:
            continue
        alpaca.submit_limit_buy(decision.symbol, qty, quote.ask)
        engine.open_position(decision.symbol, quote.ask, qty)


if __name__ == "__main__":
    run_paper_trade_once()
