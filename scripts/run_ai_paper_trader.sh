#!/usr/bin/env bash
# Run the AI trader in paper mode with hard safety checks.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/homebrew/Caskroom/miniconda/base/bin/python}"

cd "$ROOT"
exec "$PYTHON_BIN" -u -m ai_trader.paper_runner "$@"
