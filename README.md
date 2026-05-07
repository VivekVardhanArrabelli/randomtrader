# randomtrader

LLM-driven options research and trading system.

This repo is not trying to prove that an LLM can make money by free-associating
over headlines. The working thesis is narrower:

- stronger models should make better trade decisions if they are given richer,
  causal, high-signal context
- the system should avoid dumb deterministic gating that suppresses genuine
  model edge
- execution, realism, and risk controls must still stay deterministic
- every claimed improvement must survive honest backtests and replay, not just
  sound smarter

## Current Philosophy

The project is now built around a hybrid design:

- deterministic systems handle data timing, retrieval, caching, execution
  hygiene, and risk rails
- the model handles thesis formation, cross-symbol reasoning, setup selection,
  and trade expression

This distinction matters. The goal is not to hand-code the strategy around the
model. The goal is to remove low-value bookkeeping from the model so more of its
intelligence shows up in the trades it chooses.

### What We Want

- richer context, not noisier context
- better model freedom, not more hard vetoes
- repeatable positive edge across multiple windows, not one lucky backtest
- cleaner option expression and timing
- easy model/provider swaps without redesigning the whole stack

### What We Do Not Want

- fake backtests with lookahead leakage
- deterministic rules quietly overriding model judgment
- optimizing one short window until it looks amazing
- pivoting the whole strategy because of one noisy sample
- chasing raw return without drawdown, reliability, or realism

## Current State

The repo has moved materially beyond the original prototype.

- live trading uses Alpaca
- historical options backtesting now defaults to Theta Data
- historical news and equity backfill still use Polygon
- LLM access is provider-agnostic
- decision packets can be logged and replayed
- the backtester is causal and timestamp-aware
- options and news context are richer than the original headline dump

One recent reference result on the current path:

- 3-month backtest window: `2025-11-13` to `2026-02-13`
- result: `+12.26%`
- net PnL: `+$12,255.80`
- trades: `11`
- win rate: `45.5%`
- Sharpe: `0.87`
- max drawdown: `$13,930`
- LLM failure days: `0`

This is a credible positive baseline, not proof of robust production alpha.

## System Overview

```text
Alpaca News / Market Data / Options Chain
Theta Historical Options Data (backtests)
Polygon Historical News / Equity Data
        │
        ▼
Deterministic Retrieval + Enrichment + Candidate Triage
        │
        ▼
Provider-Agnostic LLM Brain
        │
        ├── Thesis / trade decision
        ├── Option expression choice
        └── Journal / replay packet logging
        │
        ▼
Execution + Risk Rails
        │
        ▼
SQLite logs / backtest outputs / replay analysis
```

## Repo Structure

### `ai_trader/`

- `brain.py` - prompt building, structured parsing, and model-facing decision
  logic
- `llm/` - provider adapters and packet abstractions
- `loop.py` - live trading loop
- `backtest.py` - causal historical backtester
- `historical_cache.py` - persistent historical response cache for repeatable backtests
- `news.py` - news/event normalization, enrichment, and setup context
- `candidates.py` - candidate triage and finalist selection
- `options.py` - option menu construction and contract ranking
- `executor.py` - live order execution
- `risk.py` - portfolio and position risk constraints
- `db.py` - decision, packet, and trade logging
- `replay.py` - rerun logged decision packets against a selected model/provider

### `tests/`

AI-specific tests should stay green before any experiment is taken seriously.

## Guardrails For Future Agents

If you are changing this repo in a future context window, follow these rules.

### 1. Protect Backtest Integrity

Never reintroduce:

- same-day lookahead
- full-day news leakage into same-day fills
- using future bars to choose contracts
- optimistic fill assumptions hidden behind daily data

If you touch `ai_trader/backtest.py`, assume realism is the first thing that can
break.

### 2. Distinguish Good Determinism From Bad Gating

Good determinism:

- timestamp gating
- caching
- quote/liquidity/spread checks
- risk caps
- causal replay
- execution validation

Bad gating:

- hiding legitimate candidates from the model without reason
- hard-coding trade vetoes because a setup "feels wrong"
- forcing contract substitutions that the model did not choose
- parser-level filtering that quietly drops model intent

The default bias should be:

- improve context quality
- improve menu quality
- improve calibration visibility
- preserve model agency

### 3. Do Not Optimize For A Single Window

Any change that looks great on one 3-week or 3-month sample can still be bad.

The target is:

- positive expectancy across multiple rolling 3-month windows
- acceptable drawdown
- stable LLM reliability
- better expression quality, especially on historically weak buckets

### 4. Prefer Advisory Improvements Over Hard Rules

If you want to improve decisions, first try:

- better setup quality labels
- better catalyst context
- better expression alternatives
- clearer recent outcome calibration
- clearer range / trend / reaction-state context

Only add hard rules when they are execution realism or risk protection.

### 5. Do Not Pivot Strategy Prematurely

This repo currently tests whether better model reasoning can improve a
directional options system. Do not casually pivot the whole project toward short
premium, 0DTE, or another regime because of one noisy run.

If a new strategy family is explored, treat it as a separate sleeve or branch,
not an in-place replacement.

## Current Targets

The next meaningful targets are:

1. Make the positive edge repeatable across more rolling 3-month windows.
2. Improve contract expression and timing.
3. Keep model freedom high while improving option menu quality and calibration.
4. Build research discipline so future changes are promoted only when they earn
   it.

The biggest current weakness is not raw model intelligence alone. It is how
that intelligence gets expressed through option choice, timing, and setup
selection under noisy conditions.

## Research Discipline

The right model for future iteration is a lightweight trading version of an
`autoresearch` loop.

That means:

- fixed baseline
- fixed evaluation windows
- fixed metrics
- explicit keep/discard decisions
- experiment logging

That does **not** mean:

- unrestricted self-modifying code changes
- blind optimization of one scalar metric
- letting the loop mutate risk rules or backtest realism freely

### Recommended Experiment Ladder

For any meaningful change:

1. `python -m pytest tests/test_ai_*.py -q`
2. replay on a fixed packet sample
3. 3-week smoke backtest
4. multiple rolling 3-month backtests
5. compare against baseline on return, Sharpe, drawdown, profit factor, and
   reliability

Promotion rule:

- keep only changes that improve robustness, not just a single lucky sample

## Data Sources

### Live / Paper Trading

- Alpaca is the live dependency for market data, news, options chains, and
  order execution

### Historical Backtesting

- Polygon is the default historical provider for unattended backtests
- Theta Data can still be selected with `HISTORICAL_OPTIONS_PROVIDER=theta`
- the repo supports persistent historical caching and offline backtest mode
- the first cold historical materialization can still be slow because unique
  option requests must be fetched once

## Logging And Replay

The system logs decision packets, visible reasoning, trade choices, and trade
results. This is important for two reasons:

- future model comparisons should be apples-to-apples
- we need to understand not just PnL, but why certain setup and expression
  families work or fail

The system does **not** log hidden chain-of-thought that the model provider does
not expose.

## Setup

```bash
pip install anthropic openai requests python-dotenv
cp ai_trader/.env.example ai_trader/.env
```

Configure the keys you actually use:

- `ALPACA_API_KEY`
- `ALPACA_API_SECRET`
- `ANTHROPIC_API_KEY` and/or `OPENAI_API_KEY`
- `POLYGON_API_KEY` for historical news/equity backfills used by backtests
- a local Theta Terminal for historical options backtests

The default provider/model path in this repo is now `openai/gpt-5.4`. Override
`LLM_MODEL` or `LLM_PROVIDER` only when you intentionally want a different
comparison target.

## Basic Commands

Run AI tests:

```bash
python -m pytest tests/test_ai_*.py -q
```

Run live trader:

```bash
python -m ai_trader
```

Run a backtest:

```bash
python -m ai_trader.backtest --start 2026-01-20 --end 2026-02-13
```

Add `--log-db path/to/run.db` if you want streaming SQLite trade/decision logs
during the backtest. If `--output` is set, the CLI will default the log DB to a
matching `.db` path.

Backtests abort after 3 consecutive LLM error cycles by default. This prevents
quota, DNS, or provider outage runs from silently producing contaminated
research artifacts. Use `--max-consecutive-llm-errors 0` only when deliberately
debugging provider failures.

Run the rolling 3-month experiment harness:

```bash
python -m ai_trader.experiments --label current-idea
```

Prepare historical cache:

```bash
python -m ai_trader.backtest --start 2026-01-20 --end 2026-02-13 --prepare-data
```

Replay logged packets:

```bash
python -m ai_trader.replay --limit 25 --provider openai --model gpt-5.4 --json
```

Experiment suite outputs land under `ai_trader/logs/experiments/` by default and
record the rolling-window metrics plus per-window backtest artifacts. Each
window now also gets a dedicated SQLite run log next to its JSON/debug files so
you can inspect decisions, fills, and closes before the suite finishes.

## Final Note For Future Iterations

Do not confuse "more engineering" with "more edge."

The right path is:

- cleaner causal data
- richer, more faithful context
- fewer bad gates
- stronger experiment discipline
- better conversion of model reasoning into option expression

If a change makes the system sound smarter but does not survive honest replay
and backtest evaluation, discard it.
