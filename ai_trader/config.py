"""Configuration for the AI autonomous options trader."""

from __future__ import annotations

import os

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
OPTION_LOSS_STREAK_GUARD_LOOKBACK = 3
OPTION_LOSS_STREAK_GUARD_MIN_CONVICTION = 0.80

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
PREPARE_PREFETCH_SYMBOLS = 8       # broader option-prefetch symbol cap for offline backtests
PREPARE_PREFETCH_CONTRACTS_PER_SIDE = 8
PREPARE_PREFETCH_STRIKE_BAND_PCT = 0.12

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
POLYGON_MIN_REQUEST_INTERVAL_SECONDS = 1.1
POLYGON_MAX_REQUEST_INTERVAL_SECONDS = 20.0
POLYGON_429_RETRY_ATTEMPTS = 5
BACKTEST_CONTRACT_HYDRATION_LIMIT = int(
    os.environ.get("BACKTEST_CONTRACT_HYDRATION_LIMIT", "12") or "12"
)

# ---------------------------------------------------------------------------
# Theta Data API (historical options backtests)
# ---------------------------------------------------------------------------
THETA_BASE_URL = "http://127.0.0.1:25510"
HISTORICAL_OPTIONS_PROVIDER = "polygon"  # theta / polygon

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
LLM_PROVIDER = ""                 # infer from model unless overridden
LLM_MODEL = "gpt-5.4"
LLM_MAX_TOKENS = 4096
DEEPSEEK_LLM_MAX_TOKENS = 8192    # DeepSeek Pro uses completion budget for reasoning + JSON
LLM_TEMPERATURE = 0.3              # lower = more deterministic trading
DEEPSEEK_LLM_TEMPERATURE = 0.0     # backtest/replay reproducibility matters more than variety
LLM_MAX_ATTEMPTS = 3
MAX_CONSECUTIVE_LLM_ERROR_CYCLES = int(
    os.environ.get("MAX_CONSECUTIVE_LLM_ERROR_CYCLES", "3") or "0"
)


def resolved_llm_model(model: str | None = None) -> str:
    """Resolve the active model at runtime so env/CLI overrides are honored."""
    if model and model.strip():
        return model.strip()
    return os.environ.get("LLM_MODEL", "").strip() or LLM_MODEL


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(int(raw), minimum)
    except ValueError:
        return default


def _env_float(
    name: str,
    default: float,
    *,
    minimum: float = 0.0,
    maximum: float = 2.0,
) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return min(max(float(raw), minimum), maximum)
    except ValueError:
        return default


def resolved_llm_max_tokens(
    *,
    model: str | None = None,
    provider: str | None = None,
) -> int:
    """Resolve structured-output token budget after provider/model selection."""
    explicit = os.environ.get("LLM_MAX_TOKENS", "").strip()
    if explicit:
        return _env_int("LLM_MAX_TOKENS", LLM_MAX_TOKENS)

    resolved_model = resolved_llm_model(model).lower()
    resolved_provider = (provider or os.environ.get("LLM_PROVIDER", "")).strip().lower()
    if resolved_provider == "deepseek" or resolved_model.startswith("deepseek"):
        return _env_int(
            "DEEPSEEK_LLM_MAX_TOKENS",
            DEEPSEEK_LLM_MAX_TOKENS,
            minimum=LLM_MAX_TOKENS,
        )
    return LLM_MAX_TOKENS


def resolved_llm_temperature(
    *,
    model: str | None = None,
    provider: str | None = None,
) -> float:
    """Resolve model temperature after provider/model selection."""
    explicit = os.environ.get("LLM_TEMPERATURE", "").strip()
    if explicit:
        return _env_float("LLM_TEMPERATURE", LLM_TEMPERATURE)

    resolved_model = resolved_llm_model(model).lower()
    resolved_provider = (provider or os.environ.get("LLM_PROVIDER", "")).strip().lower()
    if resolved_provider == "deepseek" or resolved_model.startswith("deepseek"):
        return _env_float(
            "DEEPSEEK_LLM_TEMPERATURE",
            DEEPSEEK_LLM_TEMPERATURE,
        )
    return LLM_TEMPERATURE


def resolved_max_consecutive_llm_error_cycles(value: int | None = None) -> int:
    """Resolve the LLM error fail-fast threshold after .env loading."""
    if value is not None:
        return max(int(value), 0)
    raw = os.environ.get("MAX_CONSECUTIVE_LLM_ERROR_CYCLES", "").strip()
    if raw:
        return max(int(raw), 0)
    return max(int(MAX_CONSECUTIVE_LLM_ERROR_CYCLES), 0)


def resolved_historical_options_provider(provider: str | None = None) -> str:
    """Resolve the historical options provider at runtime."""
    if provider and provider.strip():
        return provider.strip().lower()
    return (
        os.environ.get("HISTORICAL_OPTIONS_PROVIDER", "").strip().lower()
        or HISTORICAL_OPTIONS_PROVIDER
    )


def resolved_theta_base_url(base_url: str | None = None) -> str:
    """Resolve the Theta Terminal base URL at runtime."""
    if base_url and base_url.strip():
        return base_url.strip().rstrip("/")
    return (
        os.environ.get("THETA_BASE_URL", "").strip().rstrip("/")
        or THETA_BASE_URL
    )

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = "INFO"
