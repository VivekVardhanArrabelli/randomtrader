"""Tests for the AI trader brain module."""

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
    )
    assert d.action == "buy_call"
    assert d.conviction == 0.85
    assert d.risk_pct <= 0.40


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


def test_system_prompt_contains_thesis_rules():
    assert "thesis journal" in SYSTEM_PROMPT.lower()
    assert "developing" in SYSTEM_PROMPT.lower()
    assert "invalidated" in SYSTEM_PROMPT.lower()
    assert "trade history" in SYSTEM_PROMPT.lower()


def test_build_prompt_repeat_loser_instruction():
    """Final instruction includes repeat-loser accountability when present."""
    brain = TradingBrain(api_key="fake")
    history_with = "WARNING — REPEAT LOSERS\n  AAPL: 3 trades, 0 wins"
    history_without = "Recent stats: 3W/1L net=$+500"

    prompt_with = brain._build_prompt("port", "news", "mkt", "", "", history_with)
    assert "genuinely different" in prompt_with

    prompt_without = brain._build_prompt("port", "news", "mkt", "", "", history_without)
    assert "genuinely different" not in prompt_without
