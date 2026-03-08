"""Tests for AI trader SQLite packet logging."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime

from ai_trader.db import AIDecisionRecord, AITradeLogger
from ai_trader.llm import LLMDecisionPacket, LLMCompletion, ToolCall


def _sample_packet() -> LLMDecisionPacket:
    return LLMDecisionPacket(
        provider="anthropic",
        model="claude-opus-4-6",
        system_prompt="system prompt",
        user_message="exact prompt body",
        tool={
            "name": "submit_trade_decisions",
            "description": "Submit trades",
            "input_schema": {"type": "object"},
        },
        max_tokens=4096,
        temperature=0.3,
        contexts={
            "portfolio_context": "Portfolio context",
            "news_context": "News context",
        },
    )


def _sample_completion() -> LLMCompletion:
    return LLMCompletion(
        provider="anthropic",
        model="claude-opus-4-6",
        tool_calls=[
            ToolCall(
                name="submit_trade_decisions",
                input={
                    "market_analysis": "Bullish tape",
                    "thesis_updates": [],
                    "trades": [],
                },
            )
        ],
        raw_response={"id": "msg_123"},
    )


def test_ai_decisions_schema_migration_adds_packet_columns(tmp_path):
    db_path = tmp_path / "legacy.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE ai_decisions (
            timestamp TEXT,
            market_analysis TEXT,
            news_summary TEXT,
            portfolio_state TEXT,
            decisions_json TEXT,
            trades_executed INTEGER
        )
        """
    )
    conn.execute(
        "INSERT INTO ai_decisions VALUES (?, ?, ?, ?, ?, ?)",
        (
            "2025-03-01T10:00:00",
            "legacy analysis",
            "legacy news",
            "legacy portfolio",
            "[]",
            1,
        ),
    )
    conn.commit()
    conn.close()

    AITradeLogger(db_path)

    conn = sqlite3.connect(db_path)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(ai_decisions)")}
    row = conn.execute(
        "SELECT market_analysis, trades_executed, packet_json FROM ai_decisions"
    ).fetchone()
    conn.close()

    assert {"llm_provider", "llm_model", "packet_json", "response_json"} <= cols
    assert row == ("legacy analysis", 1, None)


def test_log_decision_persists_full_packet_and_updates_trade_count(tmp_path):
    db_path = tmp_path / "packets.db"
    logger = AITradeLogger(db_path)
    packet = _sample_packet()
    completion = _sample_completion()

    decision_id = logger.log_decision(
        AIDecisionRecord(
            timestamp=datetime(2025, 3, 1, 10, 0, 0),
            market_analysis="Bullish tape",
            news_summary="summary",
            portfolio_state="portfolio",
            decisions_json=json.dumps(
                {
                    "market_analysis": "Bullish tape",
                    "thesis_updates": [],
                    "trades": [],
                }
            ),
            trades_executed=0,
            llm_provider=packet.provider,
            llm_model=packet.model,
            packet_json=json.dumps(packet.to_payload()),
            response_json=json.dumps(completion.to_payload()),
        )
    )

    rows = logger.get_recent_decisions(limit=1)
    assert len(rows) == 1
    assert rows[0]["decision_id"] == decision_id
    assert rows[0]["llm_provider"] == "anthropic"
    assert rows[0]["llm_model"] == "claude-opus-4-6"
    assert json.loads(rows[0]["packet_json"])["user_message"] == "exact prompt body"
    assert json.loads(rows[0]["response_json"])["tool_calls"][0]["name"] == "submit_trade_decisions"

    assert logger.update_decision_trade_count(decision_id, 2) == 1
    updated = logger.get_recent_decisions(limit=1)[0]
    assert updated["trades_executed"] == 2
