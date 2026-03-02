"""LLM trading brain - Claude analyzes markets and makes trade decisions.

Now with thesis journal (memory across cycles) and trade history awareness.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import anthropic

from . import config
from .journal import ThesisUpdate, parse_thesis_updates
from .utils import log


@dataclass(frozen=True)
class TradeDecision:
    action: str                    # buy_call, buy_put, close_position
    underlying: str
    strike_preference: str         # itm, atm, otm
    expiry_preference: str         # this_week, next_week, monthly
    conviction: float              # 0.0 - 1.0
    risk_pct: float                # fraction of equity to risk (max 0.40)
    reasoning: str
    target_symbol: str | None      # specific option symbol for closes


@dataclass(frozen=True)
class MarketAnalysis:
    analysis: str
    trades: list[TradeDecision]
    thesis_updates: list[ThesisUpdate]


SYSTEM_PROMPT = """\
You are an autonomous AI options trader. Your job is to analyze real-time market \
news and data, then make profitable options trading decisions.

CORE RULES:
1. You trade OPTIONS (calls and puts), not stocks directly.
2. Each trade risks premium (the amount paid). Size your risk_pct based on \
conviction and how much of your equity you can afford to lose on THIS trade \
if you're wrong. A 0.40 risk_pct means you're betting 40% of your equity — \
reserve that for your highest-conviction, most asymmetric setups. Most trades \
should risk 5-15% of equity.
3. You buy calls when you're bullish, puts when you're bearish.
4. Only trade when you have genuine conviction from news catalysts or clear \
technical setups. If nothing looks compelling, return no trades.
5. Prefer liquid options with tight bid-ask spreads.
6. Consider the current portfolio - don't double down on the same thesis.
7. Be disciplined. No emotional trading. Cut losses, let winners run.
8. For expiry, prefer weeklies (this_week/next_week) for momentum plays and \
monthlies for thesis-driven trades.
9. Your conviction score (0-1) should reflect how strongly the news/data \
supports the trade. Only trades with conviction >= 0.6 will be executed.
10. Think about risk/reward. A great trade has asymmetric upside.

THESIS JOURNAL:
You maintain a thesis journal across trading cycles. This is your memory.
- Create "developing" theses when you see early signals worth watching.
- Update existing theses with new observations each cycle.
- Raise conviction and mark "ready" when evidence is strong enough to trade.
- Mark "acted_on" after you trade on a thesis.
- Mark "invalidated" when the thesis is disproven.
- A good thesis builds over 2-3 cycles before becoming a trade.
- Not every thesis needs to become a trade — invalidating is fine.
- Reference thesis IDs when making trade decisions so there's a clear link.

SELF-ASSESSMENT:
Your trade history shows your actual results. Study it carefully each cycle:
- Which tickers have you traded repeatedly? What were the results?
- If you see "Repeat losers" in your performance summary, you have a pattern \
problem. Ask yourself: is your thesis genuinely different this time, or are \
you making the same bet hoping for a different outcome?
- Look at the underlying price movement on your closed trades. Did the stock \
move the way you expected? If yes but the option lost money, it may be a \
strike/expiry problem. If no, your thesis was wrong.
- Your win rate and average winner vs average loser tell you whether your \
edge is in accuracy (high win rate) or asymmetry (big winners, small losers). \
Trade accordingly.

MARKET REGIME AWARENESS:
Before making any trade, assess the broader environment from the market data:
- Are major indices trending up or down over 5 and 10 days?
- Is your trade fighting the trend or riding it?
- In a broad selloff, even the best bullish thesis can lose. In a strong rally, \
even weak names get lifted. Factor this into your conviction.
- If indices are down >2% over 5 days, raise the bar for bullish trades. \
If indices are up >2% over 5 days, raise the bar for bearish trades.

WHEN TO TRADE:
- Breaking news with clear directional impact (earnings beats/misses, FDA \
approvals, major contracts, geopolitical events)
- Strong momentum with volume confirmation
- Sector rotation signals
- Macro catalysts (Fed decisions, economic data)
- A thesis that has matured with confirming evidence over multiple cycles

WHEN NOT TO TRADE:
- Unclear or mixed signals
- Low-volume, illiquid names
- Already heavily exposed to the same sector/thesis
- Near market close with no clear catalyst
- A thesis is still "developing" with insufficient evidence
- You've already lost money on this exact ticker recently — your edge on it \
is clearly not working. Move on to fresh setups unless you have genuinely \
new information (not the same thesis repackaged).
- The option premium is very cheap (<$0.50) — this often means far OTM with \
poor odds. Cheap options are cheap for a reason.

You will receive the current portfolio state, thesis journal, recent trade \
history, news, and market data. Analyze everything, update your journal, \
and decide whether to make any trades.\
"""

TRADE_TOOL = {
    "name": "submit_trade_decisions",
    "description": (
        "Submit your thesis journal updates AND trading decisions. "
        "Always update the journal — even if you make no trades, "
        "record what you're watching and thinking."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "market_analysis": {
                "type": "string",
                "description": (
                    "Your concise market analysis (2-4 sentences). "
                    "What's the overall market doing? Any key themes?"
                ),
            },
            "thesis_updates": {
                "type": "array",
                "description": (
                    "Updates to your thesis journal. Create new theses, "
                    "update existing ones with new observations, or "
                    "invalidate theses that are no longer valid. "
                    "Always include at least one update to show you're thinking."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": (
                                "Existing thesis ID to update (e.g. 'thesis-1'). "
                                "Omit or leave empty to create a new thesis."
                            ),
                        },
                        "underlying": {
                            "type": "string",
                            "description": "The stock/sector this thesis is about.",
                        },
                        "direction": {
                            "type": "string",
                            "enum": ["bullish", "bearish", "neutral"],
                            "description": "Your directional view.",
                        },
                        "thesis": {
                            "type": "string",
                            "description": "The core thesis (1-2 sentences).",
                        },
                        "conviction": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Current conviction level for this thesis.",
                        },
                        "status": {
                            "type": "string",
                            "enum": ["developing", "ready", "acted_on", "invalidated"],
                            "description": (
                                "developing = watching, gathering evidence. "
                                "ready = conviction high enough to trade. "
                                "acted_on = you've traded on this thesis. "
                                "invalidated = thesis disproven, remove it."
                            ),
                        },
                        "new_observation": {
                            "type": "string",
                            "description": "What new evidence did you see this cycle?",
                        },
                    },
                    "required": ["underlying", "direction", "thesis", "conviction", "status"],
                },
            },
            "trades": {
                "type": "array",
                "description": (
                    "Trade decisions. Empty array if no trades warranted. "
                    "Prefer trading on theses that have matured to 'ready' status."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["buy_call", "buy_put", "close_position"],
                            "description": "The trade action to take.",
                        },
                        "underlying": {
                            "type": "string",
                            "description": "The underlying stock ticker (e.g. AAPL, SPY).",
                        },
                        "strike_preference": {
                            "type": "string",
                            "enum": ["itm", "atm", "otm"],
                            "description": "Strike price preference relative to current price.",
                        },
                        "expiry_preference": {
                            "type": "string",
                            "enum": ["this_week", "next_week", "monthly"],
                            "description": "Expiration preference.",
                        },
                        "conviction": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "How confident you are (0-1). Only >= 0.6 will execute.",
                        },
                        "risk_pct": {
                            "type": "number",
                            "minimum": 0.01,
                            "maximum": 0.40,
                            "description": "Fraction of equity to risk on this trade (max 0.40).",
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Why you're making this trade. Reference a thesis ID if applicable.",
                        },
                        "target_symbol": {
                            "type": "string",
                            "description": (
                                "For close_position: the option symbol to close. "
                                "Leave empty for new trades."
                            ),
                        },
                    },
                    "required": [
                        "action",
                        "underlying",
                        "conviction",
                        "risk_pct",
                        "reasoning",
                    ],
                },
            },
        },
        "required": ["market_analysis", "thesis_updates", "trades"],
    },
}


class TradingBrain:
    def __init__(self, api_key: str) -> None:
        self.client = anthropic.Anthropic(api_key=api_key)

    def analyze(
        self,
        portfolio_context: str,
        news_context: str,
        market_context: str,
        options_context: str = "",
        journal_context: str = "",
        trade_history_context: str = "",
    ) -> MarketAnalysis:
        """Send all context to Claude and get trading decisions."""

        user_message = self._build_prompt(
            portfolio_context, news_context, market_context,
            options_context, journal_context, trade_history_context,
        )

        log("sending market data to LLM for analysis...")
        try:
            response = self.client.messages.create(
                model=config.LLM_MODEL,
                max_tokens=config.LLM_MAX_TOKENS,
                temperature=config.LLM_TEMPERATURE,
                system=SYSTEM_PROMPT,
                tools=[TRADE_TOOL],
                tool_choice={"type": "tool", "name": "submit_trade_decisions"},
                messages=[{"role": "user", "content": user_message}],
            )
        except Exception as exc:
            log(f"LLM API error: {exc}")
            return MarketAnalysis(analysis=f"LLM error: {exc}", trades=[], thesis_updates=[])

        return self._parse_response(response)

    def _build_prompt(
        self,
        portfolio_context: str,
        news_context: str,
        market_context: str,
        options_context: str,
        journal_context: str,
        trade_history_context: str,
    ) -> str:
        sections = [
            "=== CURRENT PORTFOLIO ===",
            portfolio_context,
        ]
        if journal_context:
            sections.extend(["", "=== YOUR THESIS JOURNAL ===", journal_context])
        if trade_history_context:
            sections.extend(["", "=== YOUR RECENT TRADE HISTORY ===", trade_history_context])
        sections.extend([
            "", "=== RECENT NEWS ===", news_context,
            "", "=== MARKET DATA ===", market_context,
        ])
        if options_context:
            sections.extend(["", "=== AVAILABLE OPTIONS ===", options_context])
        sections.extend([
            "",
            "Analyze the above. First update your thesis journal with what you're "
            "seeing — create new theses, update existing ones, or invalidate stale ones. "
            "Then decide whether to make any trades. Prefer trading on mature theses "
            "over snap decisions.",
        ])
        return "\n".join(sections)

    def _parse_response(self, response) -> MarketAnalysis:
        """Extract structured trade decisions and journal updates."""
        for block in response.content:
            if block.type == "tool_use" and block.name == "submit_trade_decisions":
                data = block.input
                analysis = data.get("market_analysis", "")

                # Parse thesis updates
                raw_theses = data.get("thesis_updates", [])
                thesis_updates = parse_thesis_updates(raw_theses)
                if thesis_updates:
                    log(f"journal updates: {len(thesis_updates)} theses")

                # Parse trades
                raw_trades = data.get("trades", [])
                trades = []
                for t in raw_trades:
                    conviction = float(t.get("conviction", 0))
                    if conviction < config.MIN_TRADE_CONVICTION:
                        log(
                            f"skipping {t.get('underlying', '?')} trade: "
                            f"conviction {conviction:.2f} < {config.MIN_TRADE_CONVICTION}"
                        )
                        continue

                    risk_pct = min(float(t.get("risk_pct", 0.10)), config.MAX_RISK_PER_TRADE)

                    trades.append(
                        TradeDecision(
                            action=t.get("action", ""),
                            underlying=t.get("underlying", "").upper(),
                            strike_preference=t.get("strike_preference", "atm"),
                            expiry_preference=t.get("expiry_preference", "next_week"),
                            conviction=conviction,
                            risk_pct=risk_pct,
                            reasoning=t.get("reasoning", ""),
                            target_symbol=t.get("target_symbol"),
                        )
                    )

                log(f"LLM analysis complete: {len(trades)} actionable trades")
                return MarketAnalysis(
                    analysis=analysis, trades=trades, thesis_updates=thesis_updates,
                )

        # Fallback: no tool use in response
        text_parts = [
            block.text for block in response.content if hasattr(block, "text")
        ]
        analysis = " ".join(text_parts) if text_parts else "No analysis returned."
        log("LLM returned no structured trades")
        return MarketAnalysis(analysis=analysis, trades=[], thesis_updates=[])
