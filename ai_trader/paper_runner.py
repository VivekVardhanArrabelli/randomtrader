"""Paper-trading launch wrapper with hard safety checks.

This is intended for launchd. It refuses to start if the trader is not pointed
at Alpaca paper trading or if the expected LLM configuration is not present.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from . import config

EASTERN_TZ = ZoneInfo("America/New_York")


def _redirect_to_dated_logs() -> None:
    run_date = datetime.now(tz=EASTERN_TZ).date().isoformat()
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"morning_{run_date}_continuous.out"
    stderr_path = log_dir / f"morning_{run_date}_continuous.err"

    stdout = stdout_path.open("a", buffering=1)
    stderr = stderr_path.open("a", buffering=1)
    os.dup2(stdout.fileno(), sys.stdout.fileno())
    os.dup2(stderr.fileno(), sys.stderr.fileno())


def _log(message: str) -> None:
    ts = datetime.now(tz=EASTERN_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"[{ts}] {message}", flush=True)


def _normalized_alpaca_base_url() -> str:
    base_url = os.environ.get("ALPACA_BASE_URL", config.ALPACA_BASE_URL).rstrip("/")
    if base_url.endswith("/v2"):
        base_url = base_url[:-3]
    return base_url


def preflight() -> None:
    env_path = Path(__file__).with_name(".env")
    load_dotenv(env_path, override=True)

    required = ["ALPACA_API_KEY", "ALPACA_API_SECRET", "POLYGON_API_KEY"]
    provider = os.environ.get("LLM_PROVIDER", config.LLM_PROVIDER).strip().lower()
    model = os.environ.get("LLM_MODEL", config.LLM_MODEL).strip()

    if provider == "deepseek":
        required.append("DEEPSEEK_API_KEY")
    elif provider == "anthropic":
        required.append("ANTHROPIC_API_KEY")
    else:
        raise SystemExit(f"unsupported LLM_PROVIDER={provider}")

    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise SystemExit(f"missing required env vars: {', '.join(missing)}")

    if config.PAPER_TRADING is not True:
        raise SystemExit("refusing to start: config.PAPER_TRADING is not True")

    base_url = _normalized_alpaca_base_url()
    if "paper-api.alpaca.markets" not in base_url:
        raise SystemExit(
            f"refusing to start: ALPACA_BASE_URL is not paper trading ({base_url})"
        )

    if provider != "deepseek":
        raise SystemExit(f"refusing to start: LLM_PROVIDER={provider}, expected deepseek")

    allow_nonstandard_model = os.environ.get("ALLOW_NONSTANDARD_LLM_MODEL", "").lower()
    if model != "deepseek-v4-pro" and allow_nonstandard_model not in {
        "1",
        "true",
        "yes",
    }:
        raise SystemExit(
            f"refusing to start: LLM_MODEL={model}, expected deepseek-v4-pro"
        )

    _log(
        "paper trader preflight passed | "
        f"paper_trading={config.PAPER_TRADING} "
        f"alpaca_base_url={base_url} "
        f"llm_provider={provider} "
        f"llm_model={model}"
    )


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    _redirect_to_dated_logs()
    _log("paper trader launch wrapper starting")
    preflight()
    if "--preflight-only" in argv:
        _log("preflight-only complete")
        return 0

    from .loop import run

    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
