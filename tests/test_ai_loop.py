import json

from ai_trader.loop import _portfolio_snapshot_record
from ai_trader.portfolio import EquityPosition, OptionPosition, PortfolioState
from ai_trader.utils import AccountSnapshot


def test_portfolio_snapshot_record_serializes_open_exposure() -> None:
    portfolio = PortfolioState(
        account=AccountSnapshot(
            equity=100_000.0,
            cash=80_000.0,
            buying_power=160_000.0,
            day_pl=250.0,
            positions_value=7_500.0,
        ),
        option_positions=[
            OptionPosition(
                symbol="AAPL260515C00150000",
                underlying="AAPL",
                option_type="call",
                strike=150.0,
                expiration="2026-05-15",
                qty=1,
                avg_entry_price=5.0,
                current_price=6.0,
                market_value=600.0,
                unrealized_pl=100.0,
                cost_basis=500.0,
            )
        ],
        equity_positions=[
            EquityPosition(
                symbol="NVDA",
                qty=12,
                avg_entry_price=200.0,
                current_price=205.0,
                market_value=2_460.0,
                unrealized_pl=60.0,
                cost_basis=2_400.0,
            )
        ],
    )

    record = _portfolio_snapshot_record(portfolio)
    positions = json.loads(record.positions_json)

    assert record.total_options_exposure == 600.0
    assert record.total_equity_exposure == 2_460.0
    assert record.total_exposure == 3_060.0
    assert record.open_option_count == 1
    assert record.open_equity_count == 1
    assert [position["asset_type"] for position in positions] == ["option", "stock"]
