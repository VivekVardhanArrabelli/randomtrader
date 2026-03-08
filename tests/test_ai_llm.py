"""Tests for provider-agnostic LLM adapter selection."""

from ai_trader.llm import create_adapter, infer_provider
from ai_trader.llm.anthropic_adapter import AnthropicAdapter
from ai_trader.llm.openai_adapter import OpenAIAdapter


def test_infer_provider_from_model_prefixes(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)

    assert infer_provider(model="claude-opus-4-6") == "anthropic"
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


def test_create_adapter_selects_anthropic_from_provider():
    adapter = create_adapter(
        provider="anthropic",
        api_key="test-anthropic-key",
    )

    assert isinstance(adapter, AnthropicAdapter)
    assert adapter.provider == "anthropic"
