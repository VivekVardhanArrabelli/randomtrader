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
    contract_symbol: str | None = None
    target_delta: float | None = None
    min_dte: int | None = None
    max_dte: int | None = None
    max_spread_pct: float | None = None


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
- Many thesis-driven setups should build over 2-3 cycles before becoming a trade.
- Do not artificially delay clean, breaking catalysts just to force extra cycles.
- Not every thesis needs to become a trade — invalidating is fine.
- Reference thesis IDs when making trade decisions so there's a clear link.
- Your journal has limited active slots. If it's full, the system will auto-prune \
your lowest-conviction developing theses. Proactively invalidate weak theses to \
make room for stronger ones. Quality over quantity — 5 strong theses beat 15 mediocre ones.
- Developing theses you don't update or raise conviction on for many cycles will \
be automatically pruned. Keep your journal fresh.

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

VOLATILITY AWARENESS:
You receive VIXY (VIX proxy) data in the market context. Use it:
- When VIXY is elevated (>$25), option premiums are expensive. You're paying \
more for the same exposure. Raise the bar for entries — you need a stronger \
catalyst to justify the inflated premium.
- When VIXY is low (<$20), premiums are cheap. Good entries if you have a thesis.
- Rising VIXY = fear is increasing. Puts get more expensive, calls cheaper. \
Consider this when sizing and timing.
- Falling VIXY = fear is fading. If you hold puts, your premium is decaying \
from both theta AND falling IV (double headwind).

RISK ALERTS:
You now control most position exits. The system attaches risk alerts to your \
positions when they hit certain thresholds. You must address each alert:
- STOP LOSS alerts: The position has hit or exceeded the DTE-scaled stop loss. \
Close the position unless you have a strong, specific thesis justifying holding \
(not just "it might come back"). If you hold, you MUST update your thesis with \
a justification.
- PROFIT TARGET alerts: The position has hit the profit target. Take profits \
unless you have clear evidence of continued momentum (e.g., strong catalyst \
still developing). If you hold, update your thesis explaining why.
- TIME STOP alerts: Very few DTE remaining. Close unless you expect a specific \
catalyst before expiration.
- CATASTROPHIC stops (-85% loss) are auto-closed by the system. You cannot \
override these.
When you see "RISK ALERT" on a position, you must either close it or explicitly \
justify holding it in a thesis update. Ignoring alerts is not allowed.

WHEN TO TRADE:
- Breaking news with clear directional impact (earnings beats/misses, FDA \
approvals, major contracts, geopolitical events)
- Strong momentum with volume confirmation
- Sector rotation signals
- Macro catalysts (Fed decisions, economic data)
- A thesis that has matured with confirming evidence over multiple cycles
- Immediate entries are allowed when the catalyst is truly fresh, clear, and \
not obviously already priced in. Multi-cycle buildup is for ambiguous or \
slower-developing setups, not a blanket rule.

NEWS INTERPRETATION:
You receive both raw headlines and a structured event map.
- The event map is a machine-generated compression layer: event_type, freshness,
  source_count, and grouped related headlines. Use it to scan faster and spot
  corroborated catalysts.
- You may also receive a relationship map: peers, ecosystem links, and sector
  ETFs connected to the catalyst names. Use it to reason about second-order
  beneficiaries, losers, and basket expressions.
- Do not treat the event map as ground truth. Verify nuance in the raw headlines
  before trading.
- Do not treat sympathy moves as automatic. A relationship map is a causal clue,
  not a trade signal by itself.
- Fresh, multi-source events deserve more attention than stale echoes of the
  same story.

ENTRY TIMING:
You may also receive a catalyst reaction snapshot that links event age to the
current price move.
- Fresh catalyst + small move can still be early.
- Older catalyst + large same-day move can already be priced in.
- When a move already looks extended, raise the bar for entry. Prefer a laggard,
  a sector expression, a better expiry/strike, or no trade at all.

TRADE EXPRESSION:
You are not limited to coarse contract buckets if the context supports a more
precise view.
- You may submit an exact `contract_symbol` for new trades when a specific
  contract is clearly best.
- You may also guide selection with `target_delta`, `min_dte`, `max_dte`, and
  `max_spread_pct`.
- You may trade the best expression of a thesis, including a related stock or
  sector ETF, when it is cleaner than the headline ticker itself.
- Use those fields only when they improve the trade thesis. Otherwise the
  coarse strike/expiry preferences are fine.
- If timing is not right yet, do not force a trade. Keep the thesis developing
  or ready and state what you are waiting for.

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
                        "contract_symbol": {
                            "type": "string",
                            "description": (
                                "Optional for new trades: exact option symbol to buy "
                                "when you want a specific contract."
                            ),
                        },
                        "target_delta": {
                            "type": "number",
                            "minimum": 0.05,
                            "maximum": 0.95,
                            "description": (
                                "Optional for new trades: desired absolute delta "
                                "(for example 0.35 or 0.60)."
                            ),
                        },
                        "min_dte": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 90,
                            "description": (
                                "Optional for new trades: minimum days to expiry."
                            ),
                        },
                        "max_dte": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 120,
                            "description": (
                                "Optional for new trades: maximum days to expiry."
                            ),
                        },
                        "max_spread_pct": {
                            "type": "number",
                            "minimum": 0.01,
                            "maximum": 0.50,
                            "description": (
                                "Optional for new trades: maximum acceptable bid-ask "
                                "spread as a fraction of ask, such as 0.12 for 12%."
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
        instruction = (
            "Analyze the above. First update your thesis journal with what you're "
            "seeing — create new theses, update existing ones, or invalidate stale ones. "
            "If your journal is near capacity, prioritize: invalidate your weakest theses "
            "before creating new ones. "
        )
        if "RISK ALERT" in portfolio_context:
            instruction += (
                "IMPORTANT: You have positions with RISK ALERTS. For each alerted position, "
                "you must either close it (submit a close_position trade) or explicitly "
                "justify holding it in a thesis update with specific reasoning. "
                "Do not ignore any risk alerts. "
            )
        if "REPEAT LOSERS" in trade_history_context:
            instruction += (
                "You have repeat-loser tickers in your history. Before trading them "
                "again, state what is genuinely different this time — a new catalyst, "
                "not the same thesis repackaged. "
            )
        instruction += (
            "Then decide whether to make any trades. Prefer trading on mature theses "
            "for ambiguous setups, but do not delay a clean breaking catalyst "
            "just to force extra cycles."
        )
        sections.extend(["", instruction])
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
                            contract_symbol=t.get("contract_symbol"),
                            target_delta=(
                                float(t.get("target_delta"))
                                if t.get("target_delta") is not None
                                else None
                            ),
                            min_dte=(
                                int(t.get("min_dte"))
                                if t.get("min_dte") is not None
                                else None
                            ),
                            max_dte=(
                                int(t.get("max_dte"))
                                if t.get("max_dte") is not None
                                else None
                            ),
                            max_spread_pct=(
                                float(t.get("max_spread_pct"))
                                if t.get("max_spread_pct") is not None
                                else None
                            ),
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
