"""Tests for replay packet loading and parsing."""

from __future__ import annotations

import json
from datetime import datetime

from ai_trader.db import AIDecisionRecord, AITradeLogger
from ai_trader.llm import LLMDecisionPacket, LLMCompletion, ToolCall
from ai_trader.replay import load_replay_records, parse_replay_record


def _logged_packet() -> LLMDecisionPacket:
    return LLMDecisionPacket(
        provider="anthropic",
        model="claude-opus-4-6",
        system_prompt="system",
        user_message="exact replay prompt",
        tool={
            "name": "submit_trade_decisions",
            "description": "Submit decisions",
            "input_schema": {"type": "object"},
        },
        max_tokens=1024,
        temperature=0.2,
        contexts={
            "portfolio_context": "portfolio",
            "market_context": "market",
        },
    )


def _logged_completion() -> LLMCompletion:
    return LLMCompletion(
        provider="anthropic",
        model="claude-opus-4-6",
        tool_calls=[
            ToolCall(
                name="submit_trade_decisions",
                input={
                    "market_analysis": "Momentum improving",
                    "thesis_updates": [],
                    "trades": [],
                },
            )
        ],
        raw_response={"id": "resp_1"},
    )


def test_load_replay_records_parses_logged_packets(tmp_path):
    db_path = tmp_path / "replay.db"
    logger = AITradeLogger(db_path)
    packet = _logged_packet()
    completion = _logged_completion()

    logger.log_decision(
        AIDecisionRecord(
            timestamp=datetime(2025, 3, 1, 10, 0, 0),
            market_analysis="Momentum improving",
            news_summary="summary",
            portfolio_state="portfolio",
            decisions_json=json.dumps(
                {
                    "market_analysis": "Momentum improving",
                    "thesis_updates": [],
                    "trades": [{"underlying": "AAPL", "action": "buy_call"}],
                }
            ),
            trades_executed=1,
            llm_provider=packet.provider,
            llm_model=packet.model,
            packet_json=json.dumps(packet.to_payload()),
            response_json=json.dumps(completion.to_payload()),
        )
    )

    records = load_replay_records(db_path, limit=1)

    assert len(records) == 1
    record = records[0]
    assert record.packet.user_message == "exact replay prompt"
    assert record.packet.contexts["market_context"] == "market"
    assert record.recorded_decisions["trades"][0]["underlying"] == "AAPL"
    assert record.recorded_response is not None
    assert record.recorded_response["tool_calls"][0]["name"] == "submit_trade_decisions"


def test_parse_replay_record_falls_back_to_row_provider_and_model():
    row = {
        "decision_id": 7,
        "timestamp": "2025-03-01T10:00:00",
        "market_analysis": "Flat open",
        "decisions_json": json.dumps({"market_analysis": "Flat open", "trades": []}),
        "response_json": json.dumps({"tool_calls": []}),
        "llm_provider": "openai",
        "llm_model": "gpt-5",
        "packet_json": json.dumps(
            {
                "schema_version": 1,
                "system_prompt": "system",
                "user_message": "prompt",
                "tool": {"name": "submit_trade_decisions", "input_schema": {"type": "object"}},
                "max_tokens": 256,
                "temperature": 0.1,
                "contexts": {},
            }
        ),
    }

    record = parse_replay_record(row)

    assert record.packet.provider == "openai"
    assert record.packet.model == "gpt-5"
    assert record.timestamp == datetime(2025, 3, 1, 10, 0, 0)
