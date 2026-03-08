"""Shared provider-agnostic LLM types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolCall:
    """Normalized tool invocation returned by an LLM provider."""

    name: str
    input: dict[str, Any]

    def to_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "input": self.input,
        }


@dataclass(frozen=True)
class LLMCompletion:
    """Provider-agnostic response payload used by the trading brain."""

    provider: str
    model: str
    text_blocks: list[str] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw_response: dict[str, Any] | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "text_blocks": list(self.text_blocks),
            "tool_calls": [call.to_payload() for call in self.tool_calls],
            "raw_response": self.raw_response,
        }


class LLMAdapter(Protocol):
    """Minimal protocol that provider adapters must satisfy."""

    provider: str

    def complete_structured(
        self,
        *,
        model: str,
        system_prompt: str,
        user_message: str,
        tool: dict[str, Any],
        max_tokens: int,
        temperature: float,
    ) -> LLMCompletion:
        ...
