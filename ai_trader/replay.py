"""Replay logged live decision packets through a chosen provider/model."""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from dotenv import load_dotenv

from . import config
from .brain import TradingBrain
from .db import DEFAULT_DB_PATH
from .llm import LLMDecisionPacket, api_key_env_name, infer_provider, resolve_api_key


@dataclass(frozen=True)
class ReplayRecord:
    decision_id: int
    timestamp: datetime
    recorded_provider: str
    recorded_model: str
    packet: LLMDecisionPacket
    recorded_market_analysis: str
    recorded_decisions: dict[str, Any]
    recorded_response: dict[str, Any] | None


def _parse_json_blob(raw: str | None, *, field_name: str, default: Any) -> Any:
    if raw in (None, ""):
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {field_name}: {exc}") from exc


def parse_replay_record(row: Mapping[str, Any]) -> ReplayRecord:
    decision_id = int(row.get("decision_id") or 0)
    if decision_id <= 0:
        raise ValueError("Replay row missing decision_id")

    timestamp_raw = row.get("timestamp")
    if not timestamp_raw:
        raise ValueError(f"Replay row {decision_id} missing timestamp")

    packet_payload = _parse_json_blob(
        row.get("packet_json"),
        field_name=f"packet_json for decision {decision_id}",
        default=None,
    )
    if packet_payload is None:
        raise ValueError(f"Replay row {decision_id} has no packet_json")

    if not isinstance(packet_payload, dict):
        raise ValueError(f"Replay packet {decision_id} must decode to an object")

    if not packet_payload.get("provider") and row.get("llm_provider"):
        packet_payload["provider"] = row["llm_provider"]
    if not packet_payload.get("model") and row.get("llm_model"):
        packet_payload["model"] = row["llm_model"]

    packet = LLMDecisionPacket.from_payload(packet_payload)
    recorded_decisions = _parse_json_blob(
        row.get("decisions_json"),
        field_name=f"decisions_json for decision {decision_id}",
        default={},
    )
    if not isinstance(recorded_decisions, dict):
        raise ValueError(f"Replay decisions {decision_id} must decode to an object")

    recorded_response = _parse_json_blob(
        row.get("response_json"),
        field_name=f"response_json for decision {decision_id}",
        default=None,
    )
    if recorded_response is not None and not isinstance(recorded_response, dict):
        raise ValueError(f"Replay response {decision_id} must decode to an object")

    return ReplayRecord(
        decision_id=decision_id,
        timestamp=datetime.fromisoformat(str(timestamp_raw)),
        recorded_provider=str(row.get("llm_provider") or packet.provider),
        recorded_model=str(row.get("llm_model") or packet.model),
        packet=packet,
        recorded_market_analysis=str(row.get("market_analysis") or ""),
        recorded_decisions=recorded_decisions,
        recorded_response=recorded_response,
    )


def load_replay_records(
    db_path: Path = DEFAULT_DB_PATH,
    *,
    limit: int = 10,
    decision_id: int | None = None,
) -> list[ReplayRecord]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        if decision_id is not None:
            rows = conn.execute(
                """
                SELECT rowid AS decision_id, *
                FROM ai_decisions
                WHERE rowid = ? AND packet_json IS NOT NULL AND packet_json != ''
                """,
                (decision_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT rowid AS decision_id, *
                FROM ai_decisions
                WHERE packet_json IS NOT NULL AND packet_json != ''
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
    finally:
        conn.close()

    return [parse_replay_record(dict(row)) for row in rows]


def replay_records(
    records: list[ReplayRecord],
    *,
    provider: str,
    model: str,
    api_key: str,
) -> list[dict[str, Any]]:
    brain = TradingBrain(api_key=api_key, provider=provider, model=model)
    results: list[dict[str, Any]] = []
    for record in records:
        run_result = brain.run_packet(record.packet)
        analysis = run_result.analysis
        replay_trades = [
            {
                "action": trade.action,
                "underlying": trade.underlying,
                "conviction": trade.conviction,
                "risk_pct": trade.risk_pct,
                "reasoning": trade.reasoning,
            }
            for trade in analysis.trades
        ]
        results.append(
            {
                "decision_id": record.decision_id,
                "timestamp": record.timestamp.isoformat(),
                "recorded_provider": record.recorded_provider,
                "recorded_model": record.recorded_model,
                "replay_provider": provider,
                "replay_model": model,
                "recorded_market_analysis": record.recorded_market_analysis,
                "replay_market_analysis": analysis.analysis,
                "recorded_trades": record.recorded_decisions.get("trades", []),
                "replay_trades": replay_trades,
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay logged AI trader decision packets.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="Path to ai_trades.db")
    parser.add_argument("--limit", type=int, default=5, help="Number of recent packets to replay")
    parser.add_argument("--decision-id", type=int, help="Replay a specific ai_decisions rowid")
    parser.add_argument("--provider", help="Provider to replay through (anthropic/openai)")
    parser.add_argument("--model", default=None, help="Model to replay through")
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    args = parser.parse_args()

    env_path = Path(__file__).with_name(".env")
    if not env_path.exists():
        env_path = Path(__file__).parent.parent / "momentum_trader" / ".env"
    load_dotenv(env_path, override=True)

    model = config.resolved_llm_model(args.model)
    provider = infer_provider(model=model, provider=args.provider or config.LLM_PROVIDER)
    api_key = resolve_api_key(provider)
    if not api_key:
        raise SystemExit(f"{api_key_env_name(provider)} not set")

    records = load_replay_records(
        args.db,
        limit=args.limit,
        decision_id=args.decision_id,
    )
    if not records:
        raise SystemExit("No replayable decision packets found")

    results = replay_records(
        records,
        provider=provider,
        model=model,
        api_key=api_key,
    )
    if args.json:
        print(json.dumps(results, indent=2))
        return

    for item in results:
        print(
            f"decision_id={item['decision_id']} "
            f"recorded={item['recorded_provider']}/{item['recorded_model']} "
            f"replay={item['replay_provider']}/{item['replay_model']} "
            f"recorded_trades={len(item['recorded_trades'])} "
            f"replay_trades={len(item['replay_trades'])}"
        )
        print(f"  recorded: {item['recorded_market_analysis']}")
        print(f"  replay:   {item['replay_market_analysis']}")


if __name__ == "__main__":
    main()
