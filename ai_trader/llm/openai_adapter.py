"""OpenAI chat-completions implementation of the provider-agnostic adapter."""

from __future__ import annotations

import json
from typing import Any

import requests

from .types import LLMCompletion, ToolCall


def _to_openai_tool(tool: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema", {}),
        },
    }


def _extract_text_parts(content: Any) -> list[str]:
    if isinstance(content, str):
        return [content] if content else []
    if not isinstance(content, list):
        return []
    text_parts: list[str] = []
    for part in content:
        if isinstance(part, dict):
            if part.get("type") == "text" and part.get("text"):
                text_parts.append(str(part["text"]))
            elif "text" in part and part.get("text"):
                text_parts.append(str(part["text"]))
    return text_parts


class OpenAIAdapter:
    provider = "openai"

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.openai.com/v1",
        session: requests.Session | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()

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
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "tools": [_to_openai_tool(tool)],
            "tool_choice": {
                "type": "function",
                "function": {"name": tool["name"]},
            },
        }
        response = self.session.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"OpenAI {response.status_code}: {response.text[:300]}"
            )

        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("OpenAI returned no choices")

        message = dict((choices[0] or {}).get("message") or {})
        tool_calls: list[ToolCall] = []
        for call in message.get("tool_calls") or []:
            function = dict(call.get("function") or {})
            arguments = function.get("arguments") or "{}"
            try:
                parsed_arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"OpenAI returned invalid tool JSON: {exc}") from exc
            tool_calls.append(
                ToolCall(
                    name=str(function.get("name") or ""),
                    input=dict(parsed_arguments or {}),
                )
            )

        return LLMCompletion(
            provider=self.provider,
            model=str(data.get("model") or model),
            text_blocks=_extract_text_parts(message.get("content")),
            tool_calls=tool_calls,
            raw_response={
                "id": data.get("id"),
                "model": data.get("model") or model,
                "usage": data.get("usage"),
                "choices": [
                    {
                        "finish_reason": choice.get("finish_reason"),
                        "message": choice.get("message"),
                    }
                    for choice in choices
                ],
            },
        )
