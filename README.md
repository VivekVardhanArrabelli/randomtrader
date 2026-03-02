# randomtrader

AI-driven autonomous options trading system. An LLM reads real-time market news,
analyzes conditions, determines option trades, and executes them without human
intervention.

The core thesis: as scaling laws hold, LLMs get better at reasoning, continual
learning, and computer use — making faster, more disciplined decisions without
emotional baggage.

## Architecture

```
News (Alpaca API)  ──┐
Market Data (Alpaca) ─┤──▶ Claude LLM Brain ──▶ Trade Decisions ──▶ Alpaca Execution
Options Chain ────────┘         │                      │
                          Risk Manager          Position Manager
                         (40% max/trade)       (auto stop/profit)
```

### ai_trader/ — autonomous AI options trader
- `brain.py` — Claude LLM integration; analyzes news + market data, returns
  structured trade decisions via tool use
- `loop.py` — main autonomous loop; runs every 15 min during market hours
- `executor.py` — turns LLM decisions into Alpaca option orders
- `options.py` — options chain fetching, filtering, contract selection
- `risk.py` — 40% max risk per trade, daily loss limits, position stops
- `portfolio.py` — portfolio state tracking
- `news.py` — news aggregation from Alpaca
- `alpaca_client.py` — Alpaca REST client with options support
- `db.py` — SQLite logging for all decisions and trades
- `config.py` — all configuration constants

### momentum_trader/ — original momentum equity scanner (legacy)

## Setup

```bash
# 1. Install dependencies
pip install anthropic requests python-dotenv

# 2. Configure API keys
cp ai_trader/.env.example ai_trader/.env
# Edit ai_trader/.env with your keys:
#   ALPACA_API_KEY, ALPACA_API_SECRET (paper trading)
#   ANTHROPIC_API_KEY (Claude API)

# 3. Run
python -m ai_trader
```

## Risk Rules

- **Max 40% of equity** per single trade (premium paid)
- **Max 80% total exposure** across all open positions
- **Max 5 open positions** at once
- **Daily loss limit**: halt if day P&L drops below -10%
- **Auto stop loss**: close at -40% on premium
- **Auto take profit**: close at +50% on premium
- **Time stop**: close positions at <= 2 DTE
- **Minimum conviction**: LLM must score >= 0.60 to execute

## How It Works

Every 15 minutes during market hours:

1. Fetch portfolio state from Alpaca
2. Check existing positions for risk exits (stop loss, take profit, time decay)
3. Fetch latest news from Alpaca news API
4. Get market overview (SPY, QQQ, IWM, DIA) and top movers
5. Build watchlist from news mentions + movers
6. Fetch options chains for watchlist symbols
7. Send everything to Claude with structured tool use
8. Claude analyzes and returns: market thesis + trade decisions
9. Validate each decision against risk rules
10. Execute approved trades via Alpaca options API
11. Log everything to SQLite

## Running Tests

```bash
python -m pytest tests/ -v
```
