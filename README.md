# randomtrader

A Python-based momentum scanner and auto-trader design for small-cap day trading strategies. This repository currently contains the system specification and operating guidelines. Use it as the blueprint for building the scanner, entry logic, risk controls, and reporting pipeline.

## What this project is

The system is designed to:

1. Scan for small-cap stocks up **50%+** by late morning with high relative volume.
2. Enter trades only when price action and liquidity checks are favorable.
3. Exit via profit targets, stop losses, or time-based stops.
4. Run in paper trading mode by default.

> ⚠️ **Disclaimer**: This project is for educational purposes. Trading is risky; nothing here is financial advice.

## Technical stack

- Python 3.11+
- Alpaca API (paper trading + market data)
- SQLite for trade logging
- Discord/Telegram webhook for alerts (optional)

## Core components

### 1. Market Scanner (`scanner.py`)
- Run at **10:30 AM EST** daily
- Criteria:
  - Price change from open: **+50% to +150%**
  - Price: **$1 - $50**
  - Relative volume: **≥ 5x** average daily volume
  - Market cap: **$50M - $2B**
  - Shares outstanding / float: **< 50M** preferred
  - Must be tradeable on Alpaca (no OTC)
- Ranking score: **(% gain × relative volume)**

### 2. Entry Logic (`entry.py`)
- Only consider stocks:
  - Within **10%** of the day’s high
  - Bid-ask spread **< 1%**
  - Not halted
- Position size:
  - Risk no more than **2%** of account per trade
  - Max **3 positions** open simultaneously
  - **LIMIT** orders at ask price (avoid market orders)

### 3. Position Manager (`positions.py`)
- Poll open positions every **1 second**
- Exit conditions:
  - **Profit target**: +5% from entry → sell 100%
  - **Stop loss**: -3% from entry → sell 100%
  - **Time stop**: exit at 3:45 PM EST if neither hit
- Use **LIMIT** orders for exits when possible, **MARKET** for urgent stops
- Track partial fills and log outcomes

### 4. Risk Management (`risk.py`)
- Daily loss limit: **-5% of account** → stop trading
- Per-trade risk: **max 2%**
- No trading in first 15 minutes or last 15 minutes of market
- Blacklist trading on:
  - FOMC days
  - Monthly options expiration (OPEX)
- Track win rate, average win, and average loss in real time

### 5. Database & Logging (`db.py`)
- SQLite database tables:
  - `scans`: timestamp, symbols found, criteria met
  - `trades`: entry_time, symbol, entry_price, exit_price, exit_time, pnl, exit_reason
  - `daily_summary`: date, trades, wins, losses, net_pnl
- Log every decision with reasoning for post-analysis

### 6. Main Orchestrator (`main.py`)
- Scheduler using `schedule` library:
  - **9:30 AM**: Initialize, check account status
  - **11:00 AM**: Run scanner, evaluate entries
  - **11:01 AM onward**: Monitor positions every second
  - **4:00 PM**: Generate daily report
- Paper trading by default (`PAPER_TRADING=True`)
- Command line args: `--live` for real trading (requires confirmation)

### 7. Configuration (`config.py`)
```python
PAPER_TRADING = True
PROFIT_TARGET_PCT = 0.05  # 5%
STOP_LOSS_PCT = 0.03      # 3%
MAX_POSITIONS = 2
MAX_RISK_PER_TRADE = 0.02 # 2% of account
DAILY_LOSS_LIMIT = 0.05   # 5% of account
SCAN_TIME = "11:00"       # EST
MIN_PRICE = 1.0
MAX_PRICE = 50.0
MIN_GAIN_PCT = 0.50       # 50% up from open
MIN_REL_VOLUME = 5        # 5x average volume
```

## Alpaca setup

1. Create an Alpaca account at <https://alpaca.markets>.
2. Generate paper trading API keys.
3. Generate a Polygon API key at <https://polygon.io>.
4. Store keys in a `.env` file (never commit real keys).

Example `.env`:
```
ALPACA_API_KEY=your_key
ALPACA_API_SECRET=your_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets
POLYGON_API_KEY=your_polygon_key
```

## Suggested repo structure

```
momentum_trader/
├── main.py
├── scanner.py
├── entry.py
├── positions.py
├── risk.py
├── db.py
├── config.py
├── utils.py
├── .env.example
├── requirements.txt
├── README.md
└── logs/
    └── trades.db
```

## Quick start (paper trading)

1. Install dependencies:
   ```bash
   pip install -r momentum_trader/requirements.txt
   ```
2. Export environment variables (or load from `.env`):
   ```bash
   export ALPACA_API_KEY=your_key
   export ALPACA_API_SECRET=your_secret
   export ALPACA_BASE_URL=https://paper-api.alpaca.markets
   export POLYGON_API_KEY=your_polygon_key
   ```
3. Run a single scan + entry cycle:
   ```bash
   python -m momentum_trader.live
   ```

## Implementation notes

- Use **Eastern Time** for all market times.
- Handle market halts (common on big movers).
- Implement exponential backoff for API rate limits.
- Log **everything** for post-analysis.
- Consider a `--backtest` mode using historical data when possible.

## Testing checklist

- Dry-run scanner on historical big-mover days
- Paper trade for at least 2 weeks
- Generate daily P&L reports

## Outputs

- Console output showing scan results, entries, and exits
- Daily summary email/Discord notification
- Trade log exportable to CSV

## Next steps

If you plan to implement the system in this repo, start with:

1. `config.py` for configurable thresholds
2. `scanner.py` for market filtering logic
3. `entry.py` and `positions.py` for trade lifecycle management
4. `db.py` and logging utilities for observability

From there, integrate Alpaca and wire up `main.py` to the scheduler.
