"""Tests for provider-agnostic LLM adapter selection."""

import json

import requests

from ai_trader import config
from ai_trader.brain import TradingBrain
from ai_trader.llm import create_adapter, infer_provider
from ai_trader.llm.anthropic_adapter import AnthropicAdapter
from ai_trader.llm.openai_adapter import OpenAIAdapter


def test_infer_provider_from_model_prefixes(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)

    assert infer_provider(model="claude-opus-4-6") == "anthropic"
    assert infer_provider(model="deepseek-v4-pro") == "deepseek"
    assert infer_provider(model="gpt-5") == "openai"
    assert infer_provider(model="o4-mini") == "openai"


def test_create_adapter_selects_openai_from_model(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)

    adapter = create_adapter(
        model="gpt-5",
        api_key="test-openai-key",
        base_url="https://example.com/v1",
    )

    assert isinstance(adapter, OpenAIAdapter)
    assert adapter.provider == "openai"
    assert adapter.base_url == "https://example.com/v1"


def test_create_adapter_selects_deepseek_from_provider():
    adapter = create_adapter(
        provider="deepseek",
        api_key="test-deepseek-key",
        base_url="https://deepseek.example",
    )

    assert isinstance(adapter, OpenAIAdapter)
    assert adapter.provider == "deepseek"
    assert adapter.base_url == "https://deepseek.example"


def test_create_adapter_selects_anthropic_from_provider():
    adapter = create_adapter(
        provider="anthropic",
        api_key="test-anthropic-key",
    )

    assert isinstance(adapter, AnthropicAdapter)
    assert adapter.provider == "anthropic"


def test_resolved_llm_model_prefers_env(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "gpt-5.4")

    assert config.resolved_llm_model() == "gpt-5.4"
    assert config.resolved_llm_model("o4-mini") == "o4-mini"


def test_default_llm_model_is_openai_family(monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)

    assert config.resolved_llm_model() == "gpt-5.4"


def test_resolved_llm_max_tokens_uses_deepseek_budget(monkeypatch):
    monkeypatch.delenv("LLM_MAX_TOKENS", raising=False)
    monkeypatch.delenv("DEEPSEEK_LLM_MAX_TOKENS", raising=False)

    assert config.resolved_llm_max_tokens(
        model="deepseek-v4-pro",
        provider="deepseek",
    ) == 8192
    assert config.resolved_llm_max_tokens(
        model="gpt-5.4",
        provider="openai",
    ) == 4096


def test_resolved_llm_max_tokens_allows_explicit_override(monkeypatch):
    monkeypatch.setenv("LLM_MAX_TOKENS", "6000")

    assert config.resolved_llm_max_tokens(
        model="deepseek-v4-pro",
        provider="deepseek",
    ) == 6000


def test_resolved_llm_temperature_uses_deepseek_replay_default(monkeypatch):
    monkeypatch.delenv("LLM_TEMPERATURE", raising=False)
    monkeypatch.delenv("DEEPSEEK_LLM_TEMPERATURE", raising=False)

    assert config.resolved_llm_temperature(
        model="deepseek-v4-pro",
        provider="deepseek",
    ) == 0.0
    assert config.resolved_llm_temperature(
        model="gpt-5.4",
        provider="openai",
    ) == 0.3


def test_resolved_llm_temperature_allows_provider_and_global_override(monkeypatch):
    monkeypatch.delenv("LLM_TEMPERATURE", raising=False)
    monkeypatch.setenv("DEEPSEEK_LLM_TEMPERATURE", "0.15")

    assert config.resolved_llm_temperature(
        model="deepseek-v4-pro",
        provider="deepseek",
    ) == 0.15

    monkeypatch.setenv("LLM_TEMPERATURE", "0.25")
    assert config.resolved_llm_temperature(
        model="deepseek-v4-pro",
        provider="deepseek",
    ) == 0.25


def test_default_historical_options_provider_is_polygon(monkeypatch):
    monkeypatch.delenv("HISTORICAL_OPTIONS_PROVIDER", raising=False)

    assert config.resolved_historical_options_provider() == "polygon"


def test_historical_options_provider_env_override(monkeypatch):
    monkeypatch.setenv("HISTORICAL_OPTIONS_PROVIDER", "theta")

    assert config.resolved_historical_options_provider() == "theta"


def test_trading_brain_defaults_to_resolved_model(monkeypatch):
    class DummyAdapter:
        provider = "openai"

    monkeypatch.setenv("LLM_MODEL", "gpt-5.4")

    brain = TradingBrain(adapter=DummyAdapter())

    assert brain.model == "gpt-5.4"
    assert brain.provider == "openai"


def test_trading_brain_uses_resolved_provider_temperature(monkeypatch):
    class DummyAdapter:
        provider = "deepseek"

    monkeypatch.delenv("LLM_TEMPERATURE", raising=False)
    monkeypatch.delenv("DEEPSEEK_LLM_TEMPERATURE", raising=False)

    brain = TradingBrain(adapter=DummyAdapter(), model="deepseek-v4-pro")
    packet = brain.build_packet("portfolio", "candidates", "news", "market")

    assert packet.temperature == 0.0


def test_openai_adapter_passes_reasoning_effort_from_env(monkeypatch):
    class DummyResponse:
        status_code = 200

        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

    class DummySession:
        def __init__(self):
            self.last_json = None
            self.last_url = None
            self.last_timeout = None

        def post(self, url, headers, json, timeout):
            self.last_url = url
            self.last_json = json
            self.last_timeout = timeout
            return DummyResponse(
                {
                    "id": "resp_123",
                    "model": json["model"],
                    "usage": {"total_tokens": 123},
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": json_module.dumps(
                                        {
                                            "market_analysis": "ok",
                                            "thesis_updates": [],
                                            "trades": [],
                                        }
                                    ),
                                }
                            ],
                        }
                    ],
                }
            )

    json_module = json
    monkeypatch.setenv("LLM_REASONING_EFFORT", "high")
    session = DummySession()
    adapter = OpenAIAdapter(
        api_key="test-openai-key",
        base_url="https://example.com/v1",
        session=session,
    )

    completion = adapter.complete_structured(
        model="gpt-5.4",
        system_prompt="system",
        user_message="user",
        tool={
            "name": "submit_trade_decisions",
            "description": "desc",
            "input_schema": {"type": "object"},
        },
        max_tokens=100,
        temperature=0.2,
    )

    assert session.last_url == "https://example.com/v1/responses"
    assert session.last_json["reasoning"] == {"effort": "high"}
    assert session.last_json["max_output_tokens"] == 8192
    assert session.last_json["text"]["format"]["type"] == "json_schema"
    assert session.last_json["text"]["format"]["name"] == "submit_trade_decisions"
    assert session.last_json["text"]["format"]["strict"] is True
    assert "tool_choice" not in session.last_json
    assert session.last_timeout == 180.0
    assert completion.model == "gpt-5.4"


def test_openai_adapter_retries_request_exceptions(monkeypatch):
    json_module = json

    class DummyResponse:
        status_code = 200

        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

    class FlakySession:
        def __init__(self):
            self.calls = 0

        def post(self, url, headers, json, timeout):
            self.calls += 1
            if self.calls == 1:
                raise requests.Timeout("slow response")
            return DummyResponse(
                {
                    "id": "resp_123",
                    "model": json["model"],
                    "usage": {"total_tokens": 123},
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": json_module.dumps(
                                        {
                                            "market_analysis": "ok",
                                            "thesis_updates": [],
                                            "trades": [],
                                        }
                                    ),
                                }
                            ],
                        }
                    ],
                }
            )

    monkeypatch.setenv("LLM_HTTP_RETRIES", "2")
    session = FlakySession()
    adapter = OpenAIAdapter(
        api_key="test-openai-key",
        base_url="https://example.com/v1",
        session=session,
    )

    completion = adapter.complete_structured(
        model="gpt-5.4",
        system_prompt="system",
        user_message="user",
        tool={
            "name": "submit_trade_decisions",
            "description": "desc",
            "input_schema": {"type": "object"},
        },
        max_tokens=100,
        temperature=0.2,
    )

    assert session.calls == 2
    assert completion.model == "gpt-5.4"


def test_deepseek_adapter_uses_json_content_mode_by_default(monkeypatch):
    class DummyResponse:
        status_code = 200

        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

    class DummySession:
        def __init__(self):
            self.last_json = None

        def post(self, url, headers, json, timeout):
            self.last_json = json
            return DummyResponse(
                {
                    "id": "chatcmpl_123",
                    "model": json["model"],
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": (
                                    '{"market_analysis":"ok",'
                                    '"thesis_updates":[],"trades":[]}'
                                )
                            },
                        }
                    ],
                }
            )

    monkeypatch.delenv("DEEPSEEK_JSON_MODE", raising=False)
    session = DummySession()
    adapter = OpenAIAdapter(
        api_key="test-deepseek-key",
        base_url="https://deepseek.example",
        provider="deepseek",
        session=session,
    )

    completion = adapter.complete_structured(
        model="deepseek-v4-pro",
        system_prompt="system",
        user_message="user",
        tool={
            "name": "submit_trade_decisions",
            "description": "desc",
            "input_schema": {"type": "object", "properties": {"market_analysis": {"type": "string"}}},
        },
        max_tokens=100,
        temperature=0.2,
    )

    assert session.last_json["response_format"] == {"type": "json_object"}
    assert "tools" not in session.last_json
    assert "tool_choice" not in session.last_json
    assert "Return ONLY a valid JSON object" in session.last_json["messages"][1]["content"]
    assert completion.provider == "deepseek"
    assert completion.tool_calls[0].name == "submit_trade_decisions"
    assert completion.tool_calls[0].input["market_analysis"] == "ok"


def test_deepseek_json_content_mode_can_be_disabled(monkeypatch):
    class DummyResponse:
        status_code = 200

        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

    class DummySession:
        def __init__(self):
            self.last_json = None

        def post(self, url, headers, json, timeout):
            self.last_json = json
            return DummyResponse(
                {
                    "id": "chatcmpl_123",
                    "model": json["model"],
                    "choices": [
                        {
                            "finish_reason": "tool_calls",
                            "message": {
                                "tool_calls": [
                                    {
                                        "function": {
                                            "name": "submit_trade_decisions",
                                            "arguments": (
                                                '{"market_analysis":"ok",'
                                                '"thesis_updates":[],"trades":[]}'
                                            ),
                                        }
                                    }
                                ]
                            },
                        }
                    ],
                }
            )

    monkeypatch.setenv("DEEPSEEK_JSON_MODE", "0")
    session = DummySession()
    adapter = OpenAIAdapter(
        api_key="test-deepseek-key",
        base_url="https://deepseek.example",
        provider="deepseek",
        session=session,
    )

    completion = adapter.complete_structured(
        model="deepseek-v4-pro",
        system_prompt="system",
        user_message="user",
        tool={
            "name": "submit_trade_decisions",
            "description": "desc",
            "input_schema": {"type": "object"},
        },
        max_tokens=100,
        temperature=0.2,
    )

    assert "response_format" not in session.last_json
    assert "tools" in session.last_json
    assert session.last_json["tool_choice"]["function"]["name"] == "submit_trade_decisions"
    assert completion.tool_calls[0].input["market_analysis"] == "ok"


def test_openai_adapter_retries_retryable_status_codes(monkeypatch):
    json_module = json

    class DummyResponse:
        def __init__(self, status_code, payload=None, text=""):
            self.status_code = status_code
            self._payload = payload or {}
            self.text = text

        def json(self):
            return self._payload

    class FlakySession:
        def __init__(self):
            self.calls = 0

        def post(self, url, headers, json, timeout):
            self.calls += 1
            if self.calls == 1:
                return DummyResponse(502, text="bad gateway")
            return DummyResponse(
                200,
                {
                    "id": "resp_123",
                    "model": json["model"],
                    "usage": {"total_tokens": 123},
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": json_module.dumps(
                                        {
                                            "market_analysis": "ok",
                                            "thesis_updates": [],
                                            "trades": [],
                                        }
                                    ),
                                }
                            ],
                        }
                    ],
                },
            )

    monkeypatch.setenv("LLM_HTTP_RETRIES", "2")
    session = FlakySession()
    adapter = OpenAIAdapter(
        api_key="test-openai-key",
        base_url="https://example.com/v1",
        session=session,
    )

    completion = adapter.complete_structured(
        model="gpt-5.4",
        system_prompt="system",
        user_message="user",
        tool={
            "name": "submit_trade_decisions",
            "description": "desc",
            "input_schema": {"type": "object"},
        },
        max_tokens=100,
        temperature=0.2,
    )

    assert session.calls == 2
    assert completion.model == "gpt-5.4"


def test_openai_adapter_normalizes_schema_for_strict_structured_output():
    class DummyResponse:
        status_code = 200

        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

    class DummySession:
        def __init__(self):
            self.last_json = None

        def post(self, url, headers, json, timeout):
            self.last_json = json
            return DummyResponse(
                {
                    "id": "resp_123",
                    "model": json["model"],
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": '{"market_analysis":"ok","thesis_updates":[],"trades":[]}',
                                }
                            ],
                        }
                    ],
                }
            )

    session = DummySession()
    adapter = OpenAIAdapter(
        api_key="test-openai-key",
        base_url="https://example.com/v1",
        session=session,
    )

    adapter.complete_structured(
        model="gpt-5.4",
        system_prompt="system",
        user_message="user",
        tool={
            "name": "submit_trade_decisions",
            "description": "desc",
            "input_schema": {
                "type": "object",
                "properties": {
                    "required_text": {"type": "string"},
                    "optional_object": {
                        "type": "object",
                        "properties": {
                            "min": {"type": "number"},
                            "max": {"type": "number"},
                        },
                        "required": ["min", "max"],
                    },
                },
                "required": ["required_text"],
            },
        },
        max_tokens=100,
        temperature=0.2,
    )

    schema = session.last_json["text"]["format"]["schema"]
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["required_text", "optional_object"]
    optional_object = schema["properties"]["optional_object"]
    assert optional_object["type"] == ["object", "null"]
    assert optional_object["properties"]["min"]["type"] == "number"
    assert optional_object["additionalProperties"] is False


def test_openai_adapter_joins_structured_output_chunks_without_newlines():
    class DummyResponse:
        status_code = 200

        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

    class DummySession:
        def post(self, url, headers, json, timeout):
            return DummyResponse(
                {
                    "id": "resp_123",
                    "model": json["model"],
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": '{"market_analysis":"split',
                                },
                                {
                                    "type": "output_text",
                                    "text": ' value","thesis_updates":[],"trades":[]}',
                                },
                            ],
                        }
                    ],
                }
            )

    adapter = OpenAIAdapter(
        api_key="test-openai-key",
        base_url="https://example.com/v1",
        session=DummySession(),
    )

    completion = adapter.complete_structured(
        model="gpt-5.4",
        system_prompt="system",
        user_message="user",
        tool={
            "name": "submit_trade_decisions",
            "description": "desc",
            "input_schema": {"type": "object"},
        },
        max_tokens=100,
        temperature=0.2,
    )

    assert completion.tool_calls[0].input["market_analysis"] == "split value"


def test_openai_adapter_recovers_json_inside_code_fence():
    class DummyResponse:
        status_code = 200

        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

    class DummySession:
        def post(self, url, headers, json, timeout):
            return DummyResponse(
                {
                    "id": "resp_123",
                    "model": json["model"],
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        "```json\n"
                                        '{"market_analysis":"ok","thesis_updates":[],"trades":[]}'
                                        "\n```"
                                    ),
                                }
                            ],
                        }
                    ],
                }
            )

    adapter = OpenAIAdapter(
        api_key="test-openai-key",
        base_url="https://example.com/v1",
        session=DummySession(),
    )

    completion = adapter.complete_structured(
        model="gpt-5.4",
        system_prompt="system",
        user_message="user",
        tool={
            "name": "submit_trade_decisions",
            "description": "desc",
            "input_schema": {"type": "object"},
        },
        max_tokens=100,
        temperature=0.2,
    )

    assert completion.tool_calls[0].input["market_analysis"] == "ok"
