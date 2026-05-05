"""Adapter selection and provider resolution helpers."""

from __future__ import annotations

import os

from .anthropic_adapter import AnthropicAdapter
from .openai_adapter import OpenAIAdapter
from .types import LLMAdapter

_PROVIDER_API_KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "openai": "OPENAI_API_KEY",
}


def infer_provider(model: str | None = None, provider: str | None = None) -> str:
    if provider:
        normalized = provider.strip().lower()
        if normalized:
            return normalized

    env_provider = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if env_provider:
        return env_provider

    model_name = (model or "").strip().lower()
    if model_name.startswith("claude"):
        return "anthropic"
    if model_name.startswith("deepseek"):
        return "deepseek"
    if model_name.startswith(("gpt-", "chatgpt-", "o1", "o3", "o4")):
        return "openai"

    raise ValueError(
        "Unable to infer LLM provider. Set LLM_PROVIDER explicitly or use a known model prefix."
    )


def api_key_env_name(provider: str) -> str:
    normalized = provider.strip().lower()
    if normalized not in _PROVIDER_API_KEYS:
        raise ValueError(f"Unsupported LLM provider: {provider}")
    return _PROVIDER_API_KEYS[normalized]


def resolve_api_key(provider: str, api_key: str | None = None) -> str | None:
    if api_key:
        return api_key

    env_name = api_key_env_name(provider)
    return os.environ.get(env_name) or os.environ.get("LLM_API_KEY")


def create_adapter(
    *,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> LLMAdapter:
    resolved_provider = infer_provider(model=model, provider=provider)
    resolved_api_key = resolve_api_key(resolved_provider, api_key=api_key)
    if not resolved_api_key:
        raise ValueError(
            f"{api_key_env_name(resolved_provider)} not set for provider {resolved_provider}"
        )

    if resolved_provider == "anthropic":
        return AnthropicAdapter(api_key=resolved_api_key)
    if resolved_provider == "deepseek":
        return OpenAIAdapter(
            api_key=resolved_api_key,
            base_url=(
                base_url
                or os.environ.get("DEEPSEEK_BASE_URL")
                or os.environ.get("LLM_BASE_URL")
                or "https://api.deepseek.com"
            ),
            provider="deepseek",
        )
    if resolved_provider == "openai":
        return OpenAIAdapter(
            api_key=resolved_api_key,
            base_url=base_url or os.environ.get("OPENAI_BASE_URL") or os.environ.get("LLM_BASE_URL") or "https://api.openai.com/v1",
        )

    raise ValueError(f"Unsupported LLM provider: {resolved_provider}")
