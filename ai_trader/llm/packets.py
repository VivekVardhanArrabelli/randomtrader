"""Serializable LLM request packets for logging and replay."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping


@dataclass(frozen=True)
class LLMDecisionPacket:
    """Exact logical packet sent to a provider for one trading decision cycle."""

    provider: str
    model: str
    system_prompt: str
    user_message: str
    tool: dict[str, Any]
    max_tokens: int
    temperature: float
    contexts: dict[str, str] = field(default_factory=dict)

    def with_target(self, *, provider: str | None = None, model: str | None = None) -> "LLMDecisionPacket":
        return replace(
            self,
            provider=provider or self.provider,
            model=model or self.model,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "provider": self.provider,
            "model": self.model,
            "system_prompt": self.system_prompt,
            "user_message": self.user_message,
            "tool": self.tool,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "contexts": dict(self.contexts),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "LLMDecisionPacket":
        if not isinstance(payload, Mapping):
            raise ValueError("Decision packet payload must be a mapping")

        tool = payload.get("tool")
        if not isinstance(tool, Mapping):
            raise ValueError("Decision packet missing tool definition")

        contexts_raw = payload.get("contexts") or {}
        if not isinstance(contexts_raw, Mapping):
            raise ValueError("Decision packet contexts must be a mapping")

        provider = str(payload.get("provider") or "").strip()
        model = str(payload.get("model") or "").strip()
        if not provider:
            raise ValueError("Decision packet missing provider")
        if not model:
            raise ValueError("Decision packet missing model")

        return cls(
            provider=provider,
            model=model,
            system_prompt=str(payload.get("system_prompt") or ""),
            user_message=str(payload.get("user_message") or ""),
            tool=dict(tool),
            max_tokens=int(payload.get("max_tokens") or 0),
            temperature=float(payload.get("temperature") or 0.0),
            contexts={
                str(key): str(value or "")
                for key, value in contexts_raw.items()
            },
        )
