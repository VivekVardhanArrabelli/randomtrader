#!/usr/bin/env bash
# Run the AI autonomous options trader
set -euo pipefail
cd "$(dirname "$0")/.."
python -m ai_trader
