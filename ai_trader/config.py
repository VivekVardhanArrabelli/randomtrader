"""Configuration for the AI autonomous options trader."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Trading mode
# ---------------------------------------------------------------------------
PAPER_TRADING = True

# ---------------------------------------------------------------------------
# Risk management
# ---------------------------------------------------------------------------
MAX_RISK_PER_TRADE = 0.40          # max 40 % of equity per single trade
MAX_TOTAL_EXPOSURE = 0.80          # max 80 % of equity in open positions
MAX_OPEN_POSITIONS = 5
DAILY_LOSS_LIMIT = 0.10            # halt trading if day P&L <= -10 %
MIN_TRADE_CONVICTION = 0.60        # LLM must have >= 60 % conviction

# ---------------------------------------------------------------------------
# Options preferences
# ---------------------------------------------------------------------------
MIN_OPTION_VOLUME = 10
MIN_OPEN_INTEREST = 50
MAX_BID_ASK_SPREAD_PCT = 0.15      # max 15 % spread on the option
PREFERRED_DTE_MIN = 2              # minimum days to expiry
PREFERRED_DTE_MAX = 45             # maximum days to expiry
DEFAULT_STRIKE_PREFERENCE = "atm"  # itm / atm / otm

# ---------------------------------------------------------------------------
# Position management
# ---------------------------------------------------------------------------
PROFIT_TARGET_PCT = 0.50           # take profit at 50 % gain on premium
STOP_LOSS_PCT = 0.40               # stop loss at 40 % loss on premium (short DTE ≤5)
STOP_LOSS_PCT_MID_DTE = 0.55       # stop loss at 55 % loss (6-14 DTE)
STOP_LOSS_PCT_LONG_DTE = 0.70      # stop loss at 70 % loss (15+ DTE)
STOP_LOSS_SHORT_DTE_THRESHOLD = 5  # DTE boundary: short vs mid
STOP_LOSS_LONG_DTE_THRESHOLD = 15  # DTE boundary: mid vs long
TIME_STOP_DTE = 2                  # close if <= 2 DTE remaining
CATASTROPHIC_STOP_PCT = 0.85       # auto-close at -85% loss, no LLM override

# ---------------------------------------------------------------------------
# Order pricing
# ---------------------------------------------------------------------------
OPEN_ORDER_SPREAD_FRACTION = 0.3   # bid 30% above mid (vs at ask)
CLOSE_LIMIT_TIMEOUT_SECONDS = 15   # wait for limit fill before market fallback

# ---------------------------------------------------------------------------
# Scanning & loop
# ---------------------------------------------------------------------------
SCAN_INTERVAL_MINUTES = 15         # how often to run the full scan cycle
NEWS_LOOKBACK_HOURS = 12           # how far back to fetch news (extended for multi-cycle context)
WATCHLIST_SIZE = 20                # top N tickers to watch
CANDIDATE_TABLE_SIZE = 12          # broad triage rows shown to the LLM
CANDIDATE_FINALISTS = 6            # finalists that get rich news/options context
POSITION_CHECK_INTERVAL_SECONDS = 60

# ---------------------------------------------------------------------------
# Thesis journal limits
# ---------------------------------------------------------------------------
JOURNAL_MAX_ACTIVE = 8
JOURNAL_MAX_FULL_DISPLAY = 5
JOURNAL_STALE_CYCLES = 8
JOURNAL_STALE_CONVICTION = 0.4

# ---------------------------------------------------------------------------
# Market hours (Eastern)
# ---------------------------------------------------------------------------
NO_TRADE_MINUTES_AFTER_OPEN = 5    # wait 5 min after open
NO_TRADE_MINUTES_BEFORE_CLOSE = 15 # stop 15 min before close
TIME_STOP_HOUR = 15
TIME_STOP_MINUTE = 45

# ---------------------------------------------------------------------------
# Alpaca API
# ---------------------------------------------------------------------------
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"
ALPACA_DATA_URL = "https://data.alpaca.markets"
HTTP_TIMEOUT_SECONDS = 30
ALPACA_RATE_LIMIT_BACKOFF_INITIAL_SECONDS = 1
ALPACA_RATE_LIMIT_BACKOFF_MAX_SECONDS = 16
ALPACA_RATE_LIMIT_MAX_RETRIES = 3

# ---------------------------------------------------------------------------
# Polygon API
# ---------------------------------------------------------------------------
POLYGON_BASE_URL = "https://api.polygon.io"

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
LLM_PROVIDER = ""                 # infer from model unless overridden
LLM_MODEL = "claude-opus-4-6"
LLM_MAX_TOKENS = 4096
LLM_TEMPERATURE = 0.3              # lower = more deterministic trading

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = "INFO"
