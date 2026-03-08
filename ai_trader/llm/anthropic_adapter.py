"""Anthropic implementation of the provider-agnostic LLM adapter."""

from __future__ import annotations

from typing import Any

import anthropic

from .types import LLMCompletion, ToolCall


def _serialize_anthropic_block(block: Any) -> dict[str, Any]:
    block_type = getattr(block, "type", "")
    payload: dict[str, Any] = {"type": block_type}
    if hasattr(block, "text"):
        payload["text"] = getattr(block, "text")
    if hasattr(block, "name"):
        payload["name"] = getattr(block, "name")
    if hasattr(block, "input"):
        payload["input"] = getattr(block, "input")
    if hasattr(block, "id"):
        payload["id"] = getattr(block, "id")
    return payload


class AnthropicAdapter:
    provider = "anthropic"

    def __init__(self, api_key: str, client: anthropic.Anthropic | None = None) -> None:
        self.client = client or anthropic.Anthropic(api_key=api_key)

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
        response = self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            tools=[tool],
            tool_choice={"type": "tool", "name": tool["name"]},
            messages=[{"role": "user", "content": user_message}],
        )

        text_blocks: list[str] = []
        tool_calls: list[ToolCall] = []
        content_blocks = getattr(response, "content", []) or []
        for block in content_blocks:
            block_type = getattr(block, "type", "")
            if block_type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        name=str(getattr(block, "name", "")),
                        input=dict(getattr(block, "input", {}) or {}),
                    )
                )
            elif hasattr(block, "text") and getattr(block, "text"):
                text_blocks.append(str(getattr(block, "text")))

        usage = getattr(response, "usage", None)
        raw_response = {
            "id": getattr(response, "id", None),
            "model": getattr(response, "model", model),
            "role": getattr(response, "role", None),
            "stop_reason": getattr(response, "stop_reason", None),
            "stop_sequence": getattr(response, "stop_sequence", None),
            "usage": {
                "input_tokens": getattr(usage, "input_tokens", None),
                "output_tokens": getattr(usage, "output_tokens", None),
            } if usage is not None else None,
            "content": [_serialize_anthropic_block(block) for block in content_blocks],
        }
        return LLMCompletion(
            provider=self.provider,
            model=str(getattr(response, "model", model)),
            text_blocks=text_blocks,
            tool_calls=tool_calls,
            raw_response=raw_response,
        )
