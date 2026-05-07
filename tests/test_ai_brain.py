"""Tests for the AI trader brain module."""

from ai_trader.llm.types import LLMCompletion, ToolCall
from ai_trader.brain import TradingBrain, TradeDecision, MarketAnalysis, SYSTEM_PROMPT, TRADE_TOOL
from ai_trader.journal import parse_thesis_updates


def test_trade_decision_fields():
    d = TradeDecision(
        action="buy_call",
        underlying="AAPL",
        strike_preference="atm",
        expiry_preference="next_week",
        conviction=0.85,
        risk_pct=0.20,
        reasoning="Strong earnings beat",
        target_symbol=None,
        expression_profile="time_cushion",
        contract_symbol="AAPL250321C00150000",
        target_delta_range=(0.45, 0.60),
        target_dte_range=(7, 21),
        max_spread_pct=0.12,
    )
    assert d.action == "buy_call"
    assert d.conviction == 0.85
    assert d.risk_pct <= 0.40
    assert d.expression_profile == "time_cushion"
    assert d.contract_symbol == "AAPL250321C00150000"
    assert d.target_delta_range == (0.45, 0.60)
    assert d.target_dte_range == (7, 21)


def test_market_analysis_structure():
    a = MarketAnalysis(
        analysis="Market is bullish",
        trades=[
            TradeDecision(
                action="buy_call",
                underlying="AAPL",
                strike_preference="atm",
                expiry_preference="next_week",
                conviction=0.80,
                risk_pct=0.15,
                reasoning="test",
                target_symbol=None,
            )
        ],
        thesis_updates=[],
    )
    assert len(a.trades) == 1
    assert a.trades[0].underlying == "AAPL"
    assert a.thesis_updates == []
    assert a.dropped_trades == []


def test_system_prompt_contains_key_rules():
    assert "40%" in SYSTEM_PROMPT
    assert "options" in SYSTEM_PROMPT.lower()
    assert "stocks" in SYSTEM_PROMPT.lower()
    assert "buy_stock" in SYSTEM_PROMPT
    assert "conviction" in SYSTEM_PROMPT.lower()
    assert "avoid chasing extended" in SYSTEM_PROMPT.lower()
    assert "concrete invalidation" in SYSTEM_PROMPT.lower()


def test_trade_tool_schema():
    assert TRADE_TOOL["name"] == "submit_trade_decisions"
    schema = TRADE_TOOL["input_schema"]
    assert "market_analysis" in schema["properties"]
    assert "trades" in schema["properties"]
    assert "thesis_updates" in schema["properties"]
    trade_props = schema["properties"]["trades"]["items"]["properties"]
    assert "buy_stock" in trade_props["action"]["enum"]
    assert "close_stock" in trade_props["action"]["enum"]
    assert trade_props["risk_pct"]["minimum"] == 0.0
    assert trade_props["risk_pct"]["maximum"] == 0.40
    assert "contract_symbol" in trade_props
    assert "expression_profile" in trade_props
    assert "target_delta_range" in trade_props
    assert "target_dte_range" in trade_props
    assert "max_spread_pct" in trade_props


def test_system_prompt_contains_thesis_rules():
    assert "thesis journal" in SYSTEM_PROMPT.lower()
    assert "developing" in SYSTEM_PROMPT.lower()
    assert "invalidated" in SYSTEM_PROMPT.lower()
    assert "trade history" in SYSTEM_PROMPT.lower()
    assert "clean, breaking catalysts" in SYSTEM_PROMPT.lower()
    assert "hard_catalyst" in SYSTEM_PROMPT
    assert "safer and more aggressive alternatives" in SYSTEM_PROMPT.lower()


def test_system_prompt_contains_format_discipline():
    assert "format discipline" in SYSTEM_PROMPT.lower()
    assert "short plain text" in SYSTEM_PROMPT.lower()
    assert "embedded double quotes" in SYSTEM_PROMPT.lower()
    assert "use exact action values" in SYSTEM_PROMPT.lower()
    assert "every thesis update must include" in SYSTEM_PROMPT.lower()
    assert "underlying" in SYSTEM_PROMPT.lower()


def test_build_prompt_repeat_loser_instruction():
    """Final instruction includes repeat-loser accountability when present."""
    brain = TradingBrain(api_key="fake")
    history_with = "WARNING — REPEAT LOSERS\n  AAPL: 3 trades, 0 wins"
    history_without = "Recent stats: 3W/1L net=$+500"

    prompt_with = brain._build_prompt("port", "candidates", "news", "mkt", "", "", history_with)
    assert "genuinely different" in prompt_with
    assert "=== BROAD CANDIDATE TRIAGE ===" in prompt_with

    prompt_without = brain._build_prompt("port", "", "news", "mkt", "", "", history_without)
    assert "genuinely different" not in prompt_without


def test_build_prompt_adds_calibration_and_fast_decay_instructions():
    brain = TradingBrain(api_key="fake")
    history = (
        "High-conviction review (>=0.65): 1/5 wins net=$-6000\n"
        "Fast-decay review (<=10 DTE at entry): 0/4 wins net=$-5500\n"
        "Stop-loss cluster: 4 trades net=$-9000 | calls=4 puts=0"
    )

    prompt = brain._build_prompt("port", "", "news", "mkt", "", "", history)
    assert "Calibrate conviction to actual edge" in prompt
    assert "prefer a longer-DTE" in prompt
    assert "change the expression before changing nothing else" in prompt


def test_run_packet_retries_once_on_adapter_error():
    class FlakyAdapter:
        provider = "openai"

        def __init__(self):
            self.calls = 0

        def complete_structured(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("OpenAI returned invalid tool JSON")
            return LLMCompletion(
                provider="openai",
                model="gpt-5.4",
                tool_calls=[
                    ToolCall(
                        name="submit_trade_decisions",
                        input={
                            "market_analysis": "ok",
                            "thesis_updates": [],
                            "trades": [],
                        },
                    )
                ],
                raw_response={},
            )

    adapter = FlakyAdapter()
    brain = TradingBrain(adapter=adapter, model="gpt-5.4")

    analysis = brain.analyze(
        portfolio_context="port",
        candidate_context="candidates",
        news_context="news",
        market_context="market",
        options_context="options",
    )

    assert adapter.calls == 2
    assert analysis.analysis == "ok"


def test_parse_response_accepts_stock_actions():
    completion = LLMCompletion(
        provider="openai",
        model="gpt-5.4",
        tool_calls=[
            ToolCall(
                name="submit_trade_decisions",
                input={
                    "market_analysis": "stock expression is cleaner",
                    "thesis_updates": [],
                    "trades": [
                        {
                            "action": "buy_stock",
                            "underlying": "NVDA",
                            "conviction": 0.70,
                            "risk_pct": 0.05,
                            "reasoning": "Use stock to avoid elevated option premium.",
                        }
                    ],
                },
            )
        ],
        raw_response={},
    )
    brain = TradingBrain(api_key="fake")

    analysis = brain._parse_response(completion)

    assert len(analysis.trades) == 1
    assert analysis.trades[0].action == "buy_stock"
    assert analysis.trades[0].underlying == "NVDA"
    assert analysis.trades[0].risk_pct == 0.05


def test_parse_response_allows_close_stock_without_conviction_or_risk():
    completion = LLMCompletion(
        provider="openai",
        model="gpt-5.4",
        tool_calls=[
            ToolCall(
                name="submit_trade_decisions",
                input={
                    "market_analysis": "close unsupported thesis",
                    "thesis_updates": [],
                    "trades": [
                        {
                            "action": "close_stock",
                            "underlying": "AVGO",
                            "target_symbol": "AVGO",
                            "reasoning": "No active thesis remains.",
                        }
                    ],
                },
            )
        ],
        raw_response={},
    )
    brain = TradingBrain(api_key="fake")

    analysis = brain._parse_response(completion)

    assert len(analysis.trades) == 1
    assert analysis.trades[0].action == "close_stock"
    assert analysis.trades[0].conviction == 1.0
    assert analysis.trades[0].risk_pct == 0.0
    assert analysis.trades[0].target_symbol == "AVGO"


def test_parse_response_normalizes_deepseek_loose_option_trade():
    completion = LLMCompletion(
        provider="deepseek",
        model="deepseek-v4-pro",
        tool_calls=[
            ToolCall(
                name="submit_trade_decisions",
                input={
                    "market_analysis": "ASTS bearish catalyst is fresh.",
                    "thesis_updates": [
                        {
                            "action": "create",
                            "thesis_id": 1,
                            "ticker": "ASTS",
                            "direction": "bearish",
                            "conviction": 0.7,
                            "status": "ready",
                            "reasoning": "Large insider sale supports downside.",
                        }
                    ],
                    "trades": [
                        {
                            "action": "buy",
                            "ticker": "ASTS",
                            "direction": "put",
                            "instrument": "option",
                            "risk_pct": 0.05,
                            "contract_symbol": "O:ASTS260522P00069000",
                            "reasoning": "Buy puts after insider sale.",
                        }
                    ],
                },
            )
        ],
        raw_response={},
    )
    brain = TradingBrain(api_key="fake")

    analysis = brain._parse_response(completion)

    assert len(analysis.thesis_updates) == 1
    assert analysis.thesis_updates[0].underlying == "ASTS"
    assert analysis.thesis_updates[0].thesis == "Large insider sale supports downside."
    assert len(analysis.trades) == 1
    assert analysis.trades[0].action == "buy_put"
    assert analysis.trades[0].underlying == "ASTS"
    assert analysis.trades[0].conviction == 0.7
    assert analysis.trades[0].risk_pct == 0.05
    assert analysis.trades[0].contract_symbol == "O:ASTS260522P00069000"
    assert analysis.dropped_trades == []


def test_parse_response_records_dropped_low_conviction_trade():
    completion = LLMCompletion(
        provider="deepseek",
        model="deepseek-v4-pro",
        tool_calls=[
            ToolCall(
                name="submit_trade_decisions",
                input={
                    "market_analysis": "Not enough edge.",
                    "thesis_updates": [],
                    "trades": [
                        {
                            "action": "buy",
                            "ticker": "AAPL",
                            "direction": "call",
                            "instrument": "option",
                            "conviction": 0.4,
                            "risk_pct": 0.05,
                            "reasoning": "Weak setup.",
                        }
                    ],
                },
            )
        ],
        raw_response={},
    )
    brain = TradingBrain(api_key="fake")

    analysis = brain._parse_response(completion)

    assert analysis.trades == []
    assert analysis.dropped_trades[0]["action"] == "buy_call"
    assert analysis.dropped_trades[0]["underlying"] == "AAPL"
    assert "conviction 0.40" in analysis.dropped_trades[0]["reason"]


def test_parse_thesis_updates_accepts_ticker_and_reasoning_aliases():
    updates = parse_thesis_updates([
        {
            "action": "create",
            "tickers": ["ASTS"],
            "direction": "put",
            "conviction": "0.68",
            "status": "ready",
            "reasoning": "Insider selling confirms the bearish setup.",
        },
        {
            "action": "invalidate",
            "thesis_id": "thesis-2",
            "reason": "Ticker unclear and setup is no longer actionable.",
        },
        {
            "thesis_id": "TH-001",
            "ticker": "AMZN",
            "status": "developing",
            "conviction": 0.4,
            "summary": "Strong earnings, waiting for pullback.",
        },
    ])

    assert len(updates) == 3
    assert updates[0].id is None
    assert updates[0].underlying == "ASTS"
    assert updates[0].direction == "bearish"
    assert updates[0].thesis == "Insider selling confirms the bearish setup."
    assert updates[0].new_observation == "Insider selling confirms the bearish setup."
    assert updates[1].id == "thesis-2"
    assert updates[1].status == "invalidated"
    assert updates[1].direction == ""
    assert updates[1].new_observation == "Ticker unclear and setup is no longer actionable."
    assert updates[2].id is None
    assert updates[2].underlying == "AMZN"
    assert updates[2].direction == "neutral"
    assert updates[2].thesis == "Strong earnings, waiting for pullback."
    assert updates[2].new_observation == "Strong earnings, waiting for pullback."


def test_run_packet_marks_missing_tool_call_after_retries_as_llm_error():
    class AlwaysEmptyAdapter:
        provider = "deepseek"

        def __init__(self):
            self.calls = 0

        def complete_structured(self, **kwargs):
            self.calls += 1
            return LLMCompletion(
                provider="deepseek",
                model="deepseek-v4-pro",
                text_blocks=["{}"],
                tool_calls=[],
                raw_response={},
            )

    adapter = AlwaysEmptyAdapter()
    brain = TradingBrain(adapter=adapter, model="deepseek-v4-pro")

    analysis = brain.analyze(
        portfolio_context="port",
        candidate_context="candidates",
        news_context="news",
        market_context="market",
        options_context="options",
    )

    assert adapter.calls >= 1
    assert analysis.analysis.startswith("LLM error: LLM returned no structured tool call")
    assert analysis.trades == []


def test_run_packet_recovers_json_text_block_without_tool_call():
    class TextOnlyAdapter:
        provider = "deepseek"

        def __init__(self):
            self.calls = 0

        def complete_structured(self, **kwargs):
            self.calls += 1
            return LLMCompletion(
                provider="deepseek",
                model="deepseek-v4-pro",
                text_blocks=[
                    """```json
{"market_analysis":"Recovered JSON","thesis_updates":[],"trades":[{"action":"buy","ticker":"AAPL","direction":"call","instrument":"option","conviction":0.7,"risk_pct":0.02,"reasoning":"Fresh catalyst with stop and target."}]}
```"""
                ],
                tool_calls=[],
                raw_response={},
            )

    adapter = TextOnlyAdapter()
    brain = TradingBrain(adapter=adapter, model="deepseek-v4-pro")

    analysis = brain.analyze(
        portfolio_context="port",
        candidate_context="candidates",
        news_context="news",
        market_context="market",
        options_context="options",
    )

    assert adapter.calls == 1
    assert analysis.analysis == "Recovered JSON"
    assert len(analysis.trades) == 1
    assert analysis.trades[0].action == "buy_call"
    assert analysis.trades[0].underlying == "AAPL"


def test_run_packet_retries_once_on_missing_tool_call():
    class EmptyThenValidAdapter:
        provider = "openai"

        def __init__(self):
            self.calls = 0

        def complete_structured(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return LLMCompletion(
                    provider="openai",
                    model="gpt-5.4",
                    text_blocks=["plain text only"],
                    tool_calls=[],
                    raw_response={},
                )
            return LLMCompletion(
                provider="openai",
                model="gpt-5.4",
                tool_calls=[
                    ToolCall(
                        name="submit_trade_decisions",
                        input={
                            "market_analysis": "ok",
                            "thesis_updates": [],
                            "trades": [],
                        },
                    )
                ],
                raw_response={},
            )

    adapter = EmptyThenValidAdapter()
    brain = TradingBrain(adapter=adapter, model="gpt-5.4")

    analysis = brain.analyze(
        portfolio_context="port",
        candidate_context="candidates",
        news_context="news",
        market_context="market",
        options_context="options",
    )

    assert adapter.calls == 2
    assert analysis.analysis == "ok"
