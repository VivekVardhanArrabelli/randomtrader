"""LLM trading brain - provider-agnostic market analysis and trade decisions.

Now with thesis journal (memory across cycles) and trade history awareness.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from . import config
from .journal import ThesisUpdate, parse_thesis_updates
from .llm import LLMAdapter, LLMCompletion, LLMDecisionPacket, ToolCall, create_adapter, infer_provider
from .utils import log


_VALID_ACTIONS = {
    "buy_call",
    "buy_put",
    "close_position",
    "buy_stock",
    "close_stock",
}
_ACTION_ALIASES = {
    "call": "buy_call",
    "long_call": "buy_call",
    "buy_call_option": "buy_call",
    "put": "buy_put",
    "long_put": "buy_put",
    "buy_put_option": "buy_put",
    "long": "buy_stock",
    "buy": "buy_stock",
    "stock": "buy_stock",
    "long_stock": "buy_stock",
}
_VALID_STRIKE_PREFERENCES = {"itm", "atm", "otm"}
_VALID_EXPIRY_PREFERENCES = {"this_week", "next_week", "monthly"}


@dataclass(frozen=True)
class TradeDecision:
    action: str                    # buy_call, buy_put, close_position, buy_stock, close_stock
    underlying: str
    strike_preference: str         # itm, atm, otm
    expiry_preference: str         # this_week, next_week, monthly
    conviction: float              # 0.0 - 1.0
    risk_pct: float                # fraction of equity to risk (max 0.40)
    reasoning: str
    target_symbol: str | None      # specific option/equity symbol for closes
    expression_profile: str | None = None
    contract_symbol: str | None = None
    target_delta_range: tuple[float, float] | None = None
    target_dte_range: tuple[int, int] | None = None
    max_spread_pct: float | None = None


@dataclass(frozen=True)
class MarketAnalysis:
    analysis: str
    trades: list[TradeDecision]
    thesis_updates: list[ThesisUpdate]
    dropped_trades: list[dict] = field(default_factory=list)


@dataclass(frozen=True)
class AnalysisRun:
    packet: LLMDecisionPacket
    completion: LLMCompletion
    analysis: MarketAnalysis


SYSTEM_PROMPT = """\
You are an autonomous AI trader. Your job is to analyze real-time market \
news and data, then make profitable trading decisions.

CORE RULES:
1. You can trade OPTIONS (calls and puts) and STOCKS (long only).
2. Each trade uses risk_pct as a hard sizing budget from account equity. \
A 0.40 risk_pct means you're allocating up to 40% of your equity — \
reserve that for your highest-conviction, most asymmetric setups. Most trades \
should risk 5-15% of equity.
3. You buy calls or stocks when you're bullish; buy puts when you're bearish.
4. Options are preferred for catalyst plays; stocks are valid when you want \
clean directional exposure without theta decay.
5. Only trade when you have genuine conviction from news catalysts or clear \
technical setups. If nothing looks compelling, return no trades.
6. Prefer liquid options with tight bid-ask spreads.
7. Consider the current portfolio - don't double down on the same thesis.
8. Be disciplined. No emotional trading. Cut losses, let winners run.
9. For expiry, prefer weeklies (this_week/next_week) for momentum plays and \
monthlies for thesis-driven trades.
10. Your conviction score (0-1) should reflect how strongly the news/data \
supports the trade. Only trades with conviction >= 0.6 will be executed.
11. Think about risk/reward. A great trade has asymmetric upside.
12. Do not let one risk flag become a blanket veto. Elevated VIXY, mixed breadth, \
or near-term uncertainty should change the expression, sizing, or required \
evidence — not automatically force no trade when a genuine edge exists.
13. If options are expensive but the directional thesis is strong, prefer \
buy_stock with smaller risk instead of discarding the trade. Use options only \
when the catalyst/asymmetry justifies paying premium.
14. Avoid chasing extended long entries. If the underlying is described as \
overbought, near all-time highs, surging/gapping, or up more than 20% over \
10 days, only buy if you can name the pullback/reclaim/consolidation level \
that makes the entry valid. Otherwise keep the thesis developing.
15. Every buy reasoning must include a concrete invalidation/stop condition \
and profit-taking condition. If an existing long breaks its invalidation level \
or breaks below the consolidation you bought, close it instead of hoping.

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

CANDIDATE TRIAGE:
You may receive a candidate table before the detailed packets.
- Treat it as a broad scan across a larger universe.
- Use it to decide which names deserve the deepest attention.
- Detailed raw news and option chains are usually only shown for the finalists.

NEWS INTERPRETATION:
You receive both raw headlines and a structured event map.
- The event map is a machine-generated compression layer: event_type, freshness,
  source_count, and grouped related headlines. Use it to scan faster and spot
  corroborated catalysts.
- You may also receive a relationship map: peers, ecosystem links, and sector
  ETFs connected to the catalyst names. Use it to reason about second-order
  beneficiaries, losers, and basket expressions.
- You may also receive catalyst-quality tags. Treat `hard_catalyst` as a
  reported event, `soft_catalyst` as weaker or more interpretive signal, and
  `opinion_or_recap` as commentary that usually needs stronger price
  confirmation.
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
- You may also choose an `expression_profile` such as balanced, time_cushion,
  stock_proxy, or convex when you want the selector to express the thesis in a
  particular style without naming an exact contract.
- You may also guide selection with `target_delta_range`, `target_dte_range`,
  and `max_spread_pct`.
- The options context may include a top-3 shortlist for calls and puts. When one
  of those contracts is clearly the best expression, choose it directly with
  `contract_symbol`.
- The shortlist may also show safer and more aggressive alternatives. On softer
  catalysts or mixed setups, a longer-DTE alternative can be a cleaner
  expression than a fast-decay weekly.
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
and decide whether to make any trades.

FORMAT DISCIPLINE:
- Every returned string value must be short plain text.
- Do not use markdown bullets, code fences, or embedded double quotes inside string values.
- Keep `market_analysis` to 2-4 short sentences.
- Keep thesis/reasoning fields to one short sentence each when possible.
- For trades, use exact action values: buy_call, buy_put, buy_stock,
  close_position, or close_stock. Use `underlying` for the ticker symbol.\
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
                    "Your concise market analysis (2-4 short plain-text sentences). "
                    "No markdown or embedded quotes."
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
                            "description": "The core thesis in 1 short plain-text sentence. No embedded quotes.",
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
                            "description": "What new evidence did you see this cycle? One short plain-text sentence.",
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
                            "enum": [
                                "buy_call",
                                "buy_put",
                                "close_position",
                                "buy_stock",
                                "close_stock",
                            ],
                            "description": "The trade action to take.",
                        },
                        "underlying": {
                            "type": "string",
                            "description": "The underlying stock ticker (e.g. AAPL, SPY).",
                        },
                        "strike_preference": {
                            "type": "string",
                            "enum": ["itm", "atm", "otm"],
                            "description": "For options only: strike price preference relative to current price.",
                        },
                        "expiry_preference": {
                            "type": "string",
                            "enum": ["this_week", "next_week", "monthly"],
                            "description": "For options only: expiration preference.",
                        },
                        "conviction": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "How confident you are (0-1). Only >= 0.6 will execute.",
                        },
                        "risk_pct": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 0.40,
                            "description": "Fraction of equity to risk on this trade (max 0.40). Use 0 for closes.",
                        },
                        "reasoning": {
                            "type": "string",
                            "description": (
                                "Why you're making this trade. One short plain-text sentence; "
                                "reference a thesis ID if applicable and avoid embedded quotes."
                            ),
                        },
                        "target_symbol": {
                            "type": "string",
                            "description": (
                                "For close_position/close_stock: symbol to close. "
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
                        "expression_profile": {
                            "type": "string",
                            "enum": ["balanced", "time_cushion", "stock_proxy", "convex"],
                            "description": (
                                "Optional for new trades: preferred contract style "
                                "when you want the selector to favor more time, "
                                "deeper delta, or more convexity."
                            ),
                        },
                        "target_delta_range": {
                            "type": "object",
                            "description": (
                                "Optional for new trades: desired absolute delta range "
                                "for the option."
                            ),
                            "properties": {
                                "min": {
                                    "type": "number",
                                    "minimum": 0.05,
                                    "maximum": 0.95,
                                },
                                "max": {
                                    "type": "number",
                                    "minimum": 0.05,
                                    "maximum": 0.95,
                                },
                            },
                            "required": ["min", "max"],
                        },
                        "target_dte_range": {
                            "type": "object",
                            "description": (
                                "Optional for new trades: desired days-to-expiry range."
                            ),
                            "properties": {
                                "min": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": 90,
                                },
                                "max": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": 120,
                                },
                            },
                            "required": ["min", "max"],
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


def _parse_range(
    raw: object,
    caster,
) -> tuple | None:
    if not isinstance(raw, dict):
        return None
    if "min" not in raw or "max" not in raw:
        return None
    try:
        low = caster(raw["min"])
        high = caster(raw["max"])
    except (TypeError, ValueError):
        return None
    return (min(low, high), max(low, high))


def _coerce_float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def _coerce_text(value: object, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def _normalize_trade_action(raw_trade: dict) -> str:
    raw_action = _coerce_text(raw_trade.get("action")).strip().lower()
    direction = _coerce_text(raw_trade.get("direction")).strip().lower()
    instrument = _coerce_text(raw_trade.get("instrument")).strip().lower()
    option_type = _coerce_text(
        raw_trade.get("option_type") or raw_trade.get("type")
    ).strip().lower()

    option_direction = direction or option_type
    if raw_action in {"buy", "long"}:
        if option_direction in {"call", "calls", "bullish_call"}:
            return "buy_call"
        if option_direction in {"put", "puts", "bearish_put"}:
            return "buy_put"
        if instrument in {"stock", "stocks", "equity", "shares"}:
            return "buy_stock"

    if raw_action in {"buy_option", "option"}:
        if option_direction in {"call", "calls", "bullish_call"}:
            return "buy_call"
        if option_direction in {"put", "puts", "bearish_put"}:
            return "buy_put"

    return _ACTION_ALIASES.get(raw_action, raw_action)


def _normalize_underlying(raw_trade: dict) -> str:
    for key in ("underlying", "ticker", "target_underlying"):
        value = _coerce_text(raw_trade.get(key)).strip().upper()
        if value and not value.startswith("O:"):
            return value

    tickers = raw_trade.get("tickers")
    if isinstance(tickers, list) and tickers:
        value = _coerce_text(tickers[0]).strip().upper()
        if value and not value.startswith("O:"):
            return value

    symbol = _coerce_text(raw_trade.get("symbol")).strip().upper()
    if symbol and not symbol.startswith("O:"):
        return symbol

    contract = _coerce_text(
        raw_trade.get("contract_symbol") or raw_trade.get("option_symbol") or symbol
    ).strip().upper()
    if contract.startswith("O:"):
        payload = contract[2:]
        underlying = []
        for char in payload:
            if char.isdigit():
                break
            underlying.append(char)
        return "".join(underlying)
    return ""


def _trade_direction(action: str) -> str | None:
    if action in {"buy_call", "buy_stock"}:
        return "bullish"
    if action == "buy_put":
        return "bearish"
    return None


def _matching_thesis_conviction(
    thesis_updates: list[ThesisUpdate],
    *,
    underlying: str,
    action: str,
) -> float | None:
    direction = _trade_direction(action)
    if not underlying or direction is None:
        return None
    matches = [
        update.conviction
        for update in thesis_updates
        if update.underlying.upper() == underlying.upper()
        and update.direction == direction
        and update.status in {"ready", "acted_on", "developing"}
    ]
    return max(matches) if matches else None


def _dropped_trade_record(
    raw_trade: object,
    *,
    reason: str,
    action: str = "",
    underlying: str = "",
    conviction: float | None = None,
) -> dict:
    record = {
        "reason": reason,
        "action": action,
        "underlying": underlying,
    }
    if conviction is not None:
        record["conviction"] = conviction
    if isinstance(raw_trade, dict):
        record["raw"] = raw_trade
    else:
        record["raw"] = repr(raw_trade)
    return record


def _parse_json_object_from_text(text: str) -> dict | None:
    candidate = text.strip()
    if not candidate:
        return None
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()

    for raw in (
        candidate,
        candidate[candidate.find("{"):candidate.rfind("}") + 1]
        if "{" in candidate and "}" in candidate else "",
    ):
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _completion_with_text_tool_call(completion: LLMCompletion) -> LLMCompletion:
    if completion.tool_calls:
        return completion
    for text in completion.text_blocks:
        parsed = _parse_json_object_from_text(text)
        if not isinstance(parsed, dict):
            continue
        if "market_analysis" not in parsed:
            continue
        parsed.setdefault("thesis_updates", [])
        parsed.setdefault("trades", [])
        if not isinstance(parsed.get("thesis_updates"), list):
            parsed["thesis_updates"] = []
        if not isinstance(parsed.get("trades"), list):
            parsed["trades"] = []
        log("recovered structured trade payload from LLM text block")
        return LLMCompletion(
            provider=completion.provider,
            model=completion.model,
            text_blocks=completion.text_blocks,
            tool_calls=[
                ToolCall(
                    name="submit_trade_decisions",
                    input=parsed,
                )
            ],
            raw_response=completion.raw_response,
        )
    return completion


class TradingBrain:
    def __init__(
        self,
        api_key: str | None = None,
        *,
        provider: str | None = None,
        model: str | None = None,
        adapter: LLMAdapter | None = None,
    ) -> None:
        self.model = config.resolved_llm_model(model)
        self.provider = (
            adapter.provider
            if adapter is not None
            else infer_provider(model=self.model, provider=provider)
        )
        self.adapter = adapter or create_adapter(
            provider=self.provider,
            model=self.model,
            api_key=api_key,
        )

    def analyze(
        self,
        portfolio_context: str,
        candidate_context: str,
        news_context: str,
        market_context: str,
        options_context: str = "",
        journal_context: str = "",
        trade_history_context: str = "",
    ) -> MarketAnalysis:
        """Send all context to the configured LLM and get trading decisions."""

        return self.run(
            portfolio_context=portfolio_context,
            candidate_context=candidate_context,
            news_context=news_context,
            market_context=market_context,
            options_context=options_context,
            journal_context=journal_context,
            trade_history_context=trade_history_context,
        ).analysis

    def run(
        self,
        *,
        portfolio_context: str,
        candidate_context: str,
        news_context: str,
        market_context: str,
        options_context: str = "",
        journal_context: str = "",
        trade_history_context: str = "",
    ) -> AnalysisRun:
        """Build an exact request packet, execute it, and parse the result."""

        packet = self.build_packet(
            portfolio_context, candidate_context, news_context, market_context,
            options_context, journal_context, trade_history_context,
        )
        return self.run_packet(packet)

    def build_packet(
        self,
        portfolio_context: str,
        candidate_context: str,
        news_context: str,
        market_context: str,
        options_context: str = "",
        journal_context: str = "",
        trade_history_context: str = "",
    ) -> LLMDecisionPacket:
        user_message = self._build_prompt(
            portfolio_context, candidate_context, news_context, market_context,
            options_context, journal_context, trade_history_context,
        )
        return LLMDecisionPacket(
            provider=self.provider,
            model=self.model,
            system_prompt=SYSTEM_PROMPT,
            user_message=user_message,
            tool=TRADE_TOOL,
            max_tokens=config.LLM_MAX_TOKENS,
            temperature=config.LLM_TEMPERATURE,
            contexts={
                "portfolio_context": portfolio_context,
                "candidate_context": candidate_context,
                "news_context": news_context,
                "market_context": market_context,
                "options_context": options_context,
                "journal_context": journal_context,
                "trade_history_context": trade_history_context,
            },
        )

    def run_packet(self, packet: LLMDecisionPacket) -> AnalysisRun:
        request_packet = packet.with_target(provider=self.provider, model=self.model)

        log("sending market data to LLM for analysis...")
        last_exc: Exception | None = None
        completion: LLMCompletion | None = None
        max_attempts = max(config.LLM_MAX_ATTEMPTS, 1)
        for attempt in range(max_attempts):
            try:
                completion = self.adapter.complete_structured(
                    model=request_packet.model,
                    max_tokens=request_packet.max_tokens,
                    temperature=request_packet.temperature,
                    system_prompt=request_packet.system_prompt,
                    user_message=request_packet.user_message,
                    tool=request_packet.tool,
                )
                completion = _completion_with_text_tool_call(completion)
                if not completion.tool_calls and attempt < max_attempts - 1:
                    log(
                        f"LLM returned no structured tool call "
                        f"(attempt {attempt + 1}/{max_attempts})"
                    )
                    continue
                break
            except Exception as exc:
                last_exc = exc
                if attempt < max_attempts - 1:
                    log(f"LLM API error (attempt {attempt + 1}/{max_attempts}): {exc}")
                    continue
                log(f"LLM API error: {exc}")
                analysis = MarketAnalysis(analysis=f"LLM error: {exc}", trades=[], thesis_updates=[])
                return AnalysisRun(
                    packet=request_packet,
                    completion=LLMCompletion(
                        provider=self.provider,
                        model=self.model,
                        text_blocks=[str(exc)],
                        raw_response={"error": str(exc)},
                    ),
                    analysis=analysis,
                )

        if completion is None:
            exc = last_exc or RuntimeError("LLM request failed")
            analysis = MarketAnalysis(analysis=f"LLM error: {exc}", trades=[], thesis_updates=[])
            return AnalysisRun(
                packet=request_packet,
                completion=LLMCompletion(
                    provider=self.provider,
                    model=self.model,
                    text_blocks=[str(exc)],
                    raw_response={"error": str(exc)},
                ),
                analysis=analysis,
            )

        if not completion.tool_calls:
            message = f"LLM returned no structured tool call after {max_attempts} attempts"
            log(message)
            return AnalysisRun(
                packet=request_packet,
                completion=completion,
                analysis=MarketAnalysis(
                    analysis=f"LLM error: {message}",
                    trades=[],
                    thesis_updates=[],
                ),
            )

        return AnalysisRun(
            packet=request_packet,
            completion=completion,
            analysis=self._parse_response(completion),
        )

    def _build_prompt(
        self,
        portfolio_context: str,
        candidate_context: str,
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
        if candidate_context:
            sections.extend(["", "=== BROAD CANDIDATE TRIAGE ===", candidate_context])
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
                "you must either close it (submit close_position or close_stock as appropriate) or explicitly "
                "justify holding it in a thesis update with specific reasoning. "
                "Do not ignore any risk alerts. "
            )
        if "REPEAT LOSERS" in trade_history_context:
            instruction += (
                "You have repeat-loser tickers in your history. Before trading them "
                "again, state what is genuinely different this time — a new catalyst, "
                "not the same thesis repackaged. "
            )
        if "High-conviction review" in trade_history_context:
            instruction += (
                "Calibrate conviction to actual edge, not narrative quality. If recent "
                "high-conviction trades underperformed, lower conviction or choose a "
                "cleaner expression. "
            )
        if "Fast-decay review" in trade_history_context:
            instruction += (
                "If fast-decay entries are underperforming, prefer a longer-DTE or "
                "higher-quality expression on softer setups. "
            )
        if (
            "Short-dated calls (<=14 DTE)" in trade_history_context
            or "Stop-loss cluster" in trade_history_context
        ):
            instruction += (
                "If one contract style keeps stopping out, change the expression "
                "before changing nothing else; use more time, a cleaner trigger, "
                "or smaller size rather than repeating the same fast entry. "
            )
        instruction += (
            "Then decide whether to make any trades. Prefer trading on mature theses "
            "for ambiguous setups, but do not delay a clean breaking catalyst "
            "just to force extra cycles."
        )
        sections.extend(["", instruction])
        return "\n".join(sections)

    def _parse_response(self, response: LLMCompletion) -> MarketAnalysis:
        """Extract structured trade decisions and journal updates."""
        for tool_call in response.tool_calls:
            if tool_call.name == "submit_trade_decisions":
                data = tool_call.input
                analysis = data.get("market_analysis", "")

                # Parse thesis updates
                raw_theses = data.get("thesis_updates", [])
                thesis_updates = parse_thesis_updates(raw_theses)
                if thesis_updates:
                    log(f"journal updates: {len(thesis_updates)} theses")

                # Parse trades
                raw_trades = data.get("trades", [])
                trades = []
                dropped_trades = []
                for t in raw_trades:
                    if not isinstance(t, dict):
                        log("skipping malformed trade: not an object")
                        dropped_trades.append(
                            _dropped_trade_record(
                                t,
                                reason="malformed trade: not an object",
                            )
                        )
                        continue

                    raw_action = _coerce_text(t.get("action")).strip().lower()
                    action = _normalize_trade_action(t)
                    if action not in _VALID_ACTIONS:
                        log(f"skipping malformed trade: unsupported action {raw_action or '?'}")
                        dropped_trades.append(
                            _dropped_trade_record(
                                t,
                                reason=f"unsupported action {raw_action or '?'}",
                                action=action,
                                underlying=_normalize_underlying(t),
                            )
                        )
                        continue

                    underlying = _normalize_underlying(t)
                    target_symbol = _coerce_text(
                        t.get("target_symbol") or t.get("close_symbol")
                    ).strip().upper() or None
                    if target_symbol is None and action in {"close_position", "close_stock"}:
                        target_symbol = _coerce_text(
                            t.get("symbol") or t.get("contract_symbol") or t.get("option_symbol")
                        ).strip().upper() or None
                    if not underlying and target_symbol and not target_symbol.startswith("O:"):
                        underlying = target_symbol
                    if not underlying and action not in {"close_position", "close_stock"}:
                        log(f"skipping malformed trade: missing underlying for {action}")
                        dropped_trades.append(
                            _dropped_trade_record(
                                t,
                                reason="missing underlying",
                                action=action,
                                underlying=underlying,
                            )
                        )
                        continue

                    is_close_action = action in ("close_position", "close_stock")
                    default_conviction = 1.0 if is_close_action else 0.0
                    raw_conviction = (
                        t.get("conviction")
                        if t.get("conviction") is not None
                        else t.get("confidence")
                    )
                    if raw_conviction is None:
                        raw_conviction = t.get("confidence_score")
                    thesis_conviction = _matching_thesis_conviction(
                        thesis_updates,
                        underlying=underlying,
                        action=action,
                    )
                    if raw_conviction is None and thesis_conviction is not None:
                        raw_conviction = thesis_conviction
                    conviction = _clamp(
                        _coerce_float(raw_conviction, default_conviction),
                        0.0,
                        1.0,
                    )
                    if not is_close_action and conviction < config.MIN_TRADE_CONVICTION:
                        log(
                            f"skipping {underlying or '?'} trade: "
                            f"conviction {conviction:.2f} < {config.MIN_TRADE_CONVICTION}"
                        )
                        dropped_trades.append(
                            _dropped_trade_record(
                                t,
                                reason=(
                                    f"conviction {conviction:.2f} < "
                                    f"{config.MIN_TRADE_CONVICTION}"
                                ),
                                action=action,
                                underlying=underlying,
                                conviction=conviction,
                            )
                        )
                        continue

                    default_risk_pct = 0.0 if is_close_action else 0.10
                    risk_pct = _clamp(
                        _coerce_float(t.get("risk_pct"), default_risk_pct),
                        0.0,
                        config.MAX_RISK_PER_TRADE,
                    )
                    strike_preference = _coerce_text(
                        t.get("strike_preference"),
                        config.DEFAULT_STRIKE_PREFERENCE,
                    ).strip().lower()
                    if strike_preference not in _VALID_STRIKE_PREFERENCES:
                        strike_preference = config.DEFAULT_STRIKE_PREFERENCE

                    expiry_preference = _coerce_text(
                        t.get("expiry_preference"),
                        "next_week",
                    ).strip().lower()
                    if expiry_preference not in _VALID_EXPIRY_PREFERENCES:
                        expiry_preference = "next_week"

                    target_delta_range = _parse_range(t.get("target_delta_range"), float)
                    legacy_target_delta = t.get("target_delta")
                    if target_delta_range is None and legacy_target_delta is not None:
                        try:
                            delta_value = float(legacy_target_delta)
                            target_delta_range = (delta_value, delta_value)
                        except (TypeError, ValueError):
                            target_delta_range = None

                    target_dte_range = _parse_range(t.get("target_dte_range"), int)
                    legacy_min_dte = t.get("min_dte")
                    legacy_max_dte = t.get("max_dte")
                    if target_dte_range is None and (
                        legacy_min_dte is not None or legacy_max_dte is not None
                    ):
                        try:
                            low = int(legacy_min_dte if legacy_min_dte is not None else legacy_max_dte)
                            high = int(legacy_max_dte if legacy_max_dte is not None else legacy_min_dte)
                            target_dte_range = (min(low, high), max(low, high))
                        except (TypeError, ValueError):
                            target_dte_range = None

                    trades.append(
                        TradeDecision(
                            action=action,
                            underlying=underlying,
                            strike_preference=strike_preference,
                            expiry_preference=expiry_preference,
                            conviction=conviction,
                            risk_pct=risk_pct,
                            reasoning=_coerce_text(t.get("reasoning")),
                            target_symbol=target_symbol,
                            expression_profile=t.get("expression_profile"),
                            contract_symbol=_coerce_text(
                                t.get("contract_symbol") or t.get("option_symbol")
                            ).strip() or None,
                            target_delta_range=target_delta_range,
                            target_dte_range=target_dte_range,
                            max_spread_pct=(
                                _coerce_float(t.get("max_spread_pct"), 0.0)
                                if t.get("max_spread_pct") is not None
                                else None
                            ),
                        )
                    )

                log(
                    f"LLM analysis complete: {len(trades)} actionable trades, "
                    f"{len(dropped_trades)} dropped"
                )
                return MarketAnalysis(
                    analysis=analysis,
                    trades=trades,
                    thesis_updates=thesis_updates,
                    dropped_trades=dropped_trades,
                )

        # Fallback: no tool use in response
        analysis = " ".join(response.text_blocks) if response.text_blocks else "No analysis returned."
        log("LLM returned no structured trades")
        return MarketAnalysis(analysis=analysis, trades=[], thesis_updates=[], dropped_trades=[])
