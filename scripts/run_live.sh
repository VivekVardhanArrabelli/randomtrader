#!/bin/zsh
set -euo pipefail

ROOT="/Users/vivekvardhanarrabelli/randomtrader"
PYTHON_BIN="/opt/homebrew/Caskroom/miniconda/base/bin/python"
LOG_DIR="$ROOT/momentum_trader/logs"

mkdir -p "$LOG_DIR"
cd "$ROOT"

exec "$PYTHON_BIN" -u -m momentum_trader.live >> "$LOG_DIR/live.out" 2>&1

