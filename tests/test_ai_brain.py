"""Tests for the AI trader brain module."""

from ai_trader.llm.types import LLMCompletion, ToolCall
from ai_trader.brain import TradingBrain, TradeDecision, MarketAnalysis, SYSTEM_PROMPT, TRADE_TOOL


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


def test_system_prompt_contains_key_rules():
    assert "40%" in SYSTEM_PROMPT
    assert "options" in SYSTEM_PROMPT.lower()
    assert "conviction" in SYSTEM_PROMPT.lower()


def test_trade_tool_schema():
    assert TRADE_TOOL["name"] == "submit_trade_decisions"
    schema = TRADE_TOOL["input_schema"]
    assert "market_analysis" in schema["properties"]
    assert "trades" in schema["properties"]
    assert "thesis_updates" in schema["properties"]
    trade_props = schema["properties"]["trades"]["items"]["properties"]
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
