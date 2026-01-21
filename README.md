# randomtrader

# Momentum Scanner & Auto-Trader System

## Overview
Build a Python-based day trading system that:
1. Scans for stocks up 50%+ by 10:30 AM EST with high relative volume
2. Enters positions based on specific criteria
3. Manages exits via profit target or stop loss
4. Runs in paper trading mode by default

## Technical Stack
- Python 3.11+
- Alpaca API (free paper trading + market data)
- SQLite for trade logging
- Discord/Telegram webhook for alerts (optional)

## Core Components

### 1. Market Scanner (`scanner.py`)
- Run at 10:30 AM EST daily
- Query for stocks meeting criteria:
  - Price change from open: +20% to +150%
  - Current price: $1 - $50 (avoid penny stocks under $1)
  - Volume: At least 5x average daily volume (relative volume > 5)
  - Market cap: $50M - $2B (small caps, not micro-caps)
  - Shares outstanding / float: Under 50M shares preferred
  - Must be tradeable on Alpaca (no OTC)
- Return ranked list by: (% gain × relative volume)

### 2. Entry Logic (`entry.py`)
- For each candidate from scanner:
  - Check if price is within 10% of day's high (don't buy on a dump)
  - Check bid-ask spread < 1% (liquidity filter)
  - Check that it's not halted
  - Position size: Risk no more than 2% of account per trade
  - Max 3 positions open simultaneously
  - Use LIMIT orders at ask price (don't market order)

### 3. Position Manager (`positions.py`)
- Monitor all open positions in real-time (1-second polling)
- Exit conditions:
  - PROFIT TARGET: +5% from entry → sell 100%
  - STOP LOSS: -3% from entry → sell 100%
  - TIME STOP: If neither hit by 3:45 PM EST → sell 100%
- Use LIMIT orders for exits when possible, MARKET if urgent (stop loss)
- Track partial fills

### 4. Risk Management (`risk.py`)
- Daily loss limit: -5% of account → stop trading for day
- Per-trade risk: Max 2% of account
- No trading in first 15 min or last 15 min of market
- Blacklist: No trading on FOMC days, monthly options expiry
- Track win rate, average win, average loss in real-time

### 5. Database & Logging (`db.py`)
- SQLite database with tables:
  - `scans`: timestamp, symbols found, criteria met
  - `trades`: entry_time, symbol, entry_price, exit_price, exit_time, pnl, exit_reason
  - `daily_summary`: date, trades, wins, losses, net_pnl
- Log every decision with reasoning

### 6. Main Orchestrator (`main.py`)
- Scheduler using `schedule` library:
  - 9:30 AM: Initialize, check account status
  - 11:00 AM: Run scanner, evaluate entries
  - 11:01 AM onwards: Monitor positions every second
  - 4:00 PM: Generate daily report
- Paper trading mode by default (set `PAPER_TRADING=True`)
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

## Alpaca Setup
- Use Alpaca's free paper trading API
- Endpoint: `https://paper-api.alpaca.markets`
- Get API keys from alpaca.markets
- Store keys in `.env` file (never commit)

## File Structure
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

## Key Implementation Notes
1. Always use Eastern Time for market hours
2. Handle market halts gracefully (common on big movers)
3. Implement exponential backoff for API rate limits
4. Log EVERYTHING for post-analysis
5. Include a `--backtest` mode using historical data if time permits

## Testing
- Dry run scanner on historical "big mover" days
- Paper trade for minimum 2 weeks before considering live
- Generate daily P&L reports

## Output
- Console output showing scan results, entries, exits
- Daily summary email/Discord notification
- Trade log exportable to CSV for analysis
