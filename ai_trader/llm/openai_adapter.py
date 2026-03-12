"""OpenAI chat-completions implementation of the provider-agnostic adapter."""

from __future__ import annotations

from copy import deepcopy
import json
import os
import re
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


def _uses_max_completion_tokens(model: str) -> bool:
    normalized = (model or "").strip().lower()
    return normalized.startswith(("gpt-5", "o1", "o3", "o4"))


def _uses_responses_api(model: str, reasoning_effort: str) -> bool:
    return _uses_max_completion_tokens(model) or bool(reasoning_effort)


def _to_responses_tool(tool: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "name": tool["name"],
        "description": tool.get("description", ""),
        "parameters": tool.get("input_schema", {}),
    }


def _nullable_schema(schema: dict[str, Any]) -> dict[str, Any]:
    nullable = deepcopy(schema)
    schema_type = nullable.get("type")
    if isinstance(schema_type, str):
        if schema_type != "null":
            nullable["type"] = [schema_type, "null"]
        return nullable
    if isinstance(schema_type, list):
        if "null" not in schema_type:
            nullable["type"] = [*schema_type, "null"]
        return nullable
    enum_values = nullable.get("enum")
    if isinstance(enum_values, list):
        if None not in enum_values:
            nullable["enum"] = [*enum_values, None]
        nullable.setdefault("type", ["null"])
        return nullable
    return {"anyOf": [nullable, {"type": "null"}]}


def _to_openai_response_schema(
    schema: dict[str, Any],
    *,
    required: bool = True,
) -> dict[str, Any]:
    normalized = deepcopy(schema)
    schema_type = normalized.get("type")

    if schema_type == "object" or "properties" in normalized:
        properties = normalized.get("properties") or {}
        original_required = list(normalized.get("required") or [])
        normalized_properties: dict[str, Any] = {}
        for key, value in properties.items():
            if not isinstance(value, dict):
                continue
            normalized_properties[key] = _to_openai_response_schema(
                value,
                required=key in original_required,
            )
        normalized["type"] = "object"
        normalized["properties"] = normalized_properties
        normalized["required"] = list(normalized_properties.keys())
        normalized["additionalProperties"] = False
    elif schema_type == "array":
        items = normalized.get("items")
        if isinstance(items, dict):
            normalized["items"] = _to_openai_response_schema(items)

    if not required:
        return _nullable_schema(normalized)
    return normalized


def _to_responses_text_format(tool: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "name": tool["name"],
        "schema": _to_openai_response_schema(tool.get("input_schema", {})),
        "strict": True,
    }


def _coerce_response_text(part: dict[str, Any]) -> str | None:
    text = part.get("text")
    if isinstance(text, str) and text:
        return text
    if isinstance(text, dict):
        value = text.get("value")
        if isinstance(value, str) and value:
            return value
    return None


def _extract_response_text(output: Any, raw_response: dict[str, Any] | None = None) -> list[str]:
    if not isinstance(output, list):
        fallback_text = raw_response.get("output_text") if isinstance(raw_response, dict) else None
        return [fallback_text] if isinstance(fallback_text, str) and fallback_text else []
    text_parts: list[str] = []
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") not in {"output_text", "text"}:
                continue
            text = _coerce_response_text(part)
            if text:
                text_parts.append(text)
    if text_parts:
        return text_parts
    fallback_text = raw_response.get("output_text") if isinstance(raw_response, dict) else None
    if isinstance(fallback_text, str) and fallback_text:
        return [fallback_text]
    return text_parts


def _extract_response_tool_calls(output: Any) -> list[ToolCall]:
    if not isinstance(output, list):
        return []
    tool_calls: list[ToolCall] = []
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "function_call":
            continue
        arguments = item.get("arguments") or "{}"
        try:
            parsed_arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"OpenAI returned invalid tool JSON: {exc}") from exc
        tool_calls.append(
            ToolCall(
                name=str(item.get("name") or ""),
                input=dict(parsed_arguments or {}),
            )
        )
    return tool_calls


def _extract_response_text_tool_call(
    output: Any,
    tool_name: str,
    raw_response: dict[str, Any] | None = None,
) -> list[ToolCall]:
    text_parts = _extract_response_text(output, raw_response)
    if not text_parts:
        return []
    raw_text = "".join(text_parts).strip()
    candidates = [raw_text]
    fenced = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text, flags=re.IGNORECASE | re.DOTALL).strip()
    if fenced and fenced != raw_text:
        candidates.append(fenced)
    start = fenced.find("{")
    end = fenced.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(fenced[start : end + 1].strip())
    decoder = json.JSONDecoder()
    decode_error: json.JSONDecodeError | None = None
    parsed_arguments: Any = None
    try:
        for candidate in candidates:
            if not candidate:
                continue
            try:
                parsed_arguments = json.loads(candidate)
                break
            except json.JSONDecodeError as exc:
                decode_error = exc
                if "{" in candidate:
                    try:
                        parsed_arguments, _ = decoder.raw_decode(candidate[candidate.find("{") :])
                        break
                    except json.JSONDecodeError:
                        pass
        else:
            raise decode_error or json.JSONDecodeError("invalid JSON", raw_text, 0)
    except json.JSONDecodeError as exc:
        preview = raw_text[:200].replace("\n", "\\n")
        raise RuntimeError(
            f"OpenAI returned invalid structured JSON: {exc}; preview={preview}"
        ) from exc
    if not isinstance(parsed_arguments, dict):
        raise RuntimeError("OpenAI structured output was not a JSON object")
    return [ToolCall(name=tool_name, input=dict(parsed_arguments))]


def _request_timeout_seconds() -> float:
    raw_value = (
        os.environ.get("LLM_HTTP_TIMEOUT_SECONDS")
        or os.environ.get("OPENAI_TIMEOUT_SECONDS")
        or "180"
    ).strip()
    try:
        return max(float(raw_value), 1.0)
    except ValueError:
        return 180.0


def _request_retries() -> int:
    raw_value = (os.environ.get("LLM_HTTP_RETRIES") or "2").strip()
    try:
        return max(int(raw_value), 1)
    except ValueError:
        return 2


def _should_retry_status(status_code: int) -> bool:
    return status_code in {408, 429, 500, 502, 503, 504}


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

    def _post_json(self, endpoint: str, payload: dict[str, Any]) -> requests.Response:
        timeout = _request_timeout_seconds()
        last_error: requests.RequestException | None = None
        last_response: requests.Response | None = None
        retries = _request_retries()
        for attempt in range(retries):
            try:
                response = self.session.post(
                    f"{self.base_url}/{endpoint.lstrip('/')}",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=timeout,
                )
                if _should_retry_status(response.status_code) and attempt < retries - 1:
                    last_response = response
                    continue
                return response
            except requests.RequestException as exc:
                last_error = exc
        if last_response is not None:
            return last_response
        raise RuntimeError(f"OpenAI request failed after retries: {last_error}") from last_error

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
            "tools": [_to_openai_tool(tool)],
            "tool_choice": {
                "type": "function",
                "function": {"name": tool["name"]},
            },
        }
        if _uses_max_completion_tokens(model):
            payload["max_completion_tokens"] = max_tokens
        else:
            payload["max_tokens"] = max_tokens
        reasoning_effort = (
            os.environ.get("OPENAI_REASONING_EFFORT")
            or os.environ.get("LLM_REASONING_EFFORT")
            or ""
        ).strip()
        if _uses_responses_api(model, reasoning_effort):
            output_budget = max_tokens
            if reasoning_effort:
                output_budget = max(output_budget, 8192)
            payload = {
                "model": model,
                "instructions": system_prompt,
                "input": user_message,
                "max_output_tokens": output_budget,
                "text": {"format": _to_responses_text_format(tool)},
            }
            if reasoning_effort:
                payload["reasoning"] = {"effort": reasoning_effort}
            response = self._post_json("/responses", payload)
            if response.status_code >= 400:
                raise RuntimeError(
                    f"OpenAI {response.status_code}: {response.text[:300]}"
                )

            data = response.json()
            output = data.get("output") or []
            text_blocks = _extract_response_text(output, data)
            return LLMCompletion(
                provider=self.provider,
                model=str(data.get("model") or model),
                text_blocks=text_blocks,
                tool_calls=_extract_response_text_tool_call(output, tool["name"], data),
                raw_response=data,
            )

        response = self._post_json("/chat/completions", payload)
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
