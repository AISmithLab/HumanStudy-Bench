"""
Factory to get the appropriate LLM client by provider.
"""

import os
from typing import Optional

from src.llm.base import BaseLLMClient
from src.llm.openai_client import OpenAIClient
from src.llm.anthropic_client import AnthropicClient
from src.llm.xai_client import XAIClient


def get_client(
    provider: str,
    model: str,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
) -> BaseLLMClient:
    """
    Return a unified LLM client for the given provider.

    Args:
        provider: One of "openai", "anthropic", "xai", "openrouter", "gemini"
        model: Model name (e.g. "gpt-4o", "claude-3-5-sonnet-20241022", "grok-2", "mistralai/mistral-nemo")
        api_key: Optional; otherwise read from OPENAI_API_KEY, ANTHROPIC_API_KEY, XAI_API_KEY, OPENROUTER_API_KEY, GOOGLE_API_KEY
        api_base: Optional; for openrouter default is https://openrouter.ai/api/v1

    Returns:
        BaseLLMClient implementation
    """
    provider = (provider or "").lower().strip()
    key = api_key
    base = api_base

    if provider == "openai":
        key = key or os.getenv("OPENAI_API_KEY")
        return OpenAIClient(model=model, api_key=key, api_base=base)
    if provider == "anthropic":
        key = key or os.getenv("ANTHROPIC_API_KEY")
        return AnthropicClient(model=model, api_key=key, api_base=base)
    if provider == "xai":
        key = key or os.getenv("XAI_API_KEY")
        return XAIClient(model=model, api_key=key, api_base=base)
    if provider == "openrouter":
        key = key or os.getenv("OPENROUTER_API_KEY")
        base = base or "https://openrouter.ai/api/v1"
        return OpenAIClient(model=model, api_key=key, api_base=base)
    if provider == "gemini":
        # Defer import so google-genai is optional until gemini is used
        from src.llm.gemini_client import GeminiLLMClient
        key = key or os.getenv("GOOGLE_API_KEY")
        return GeminiLLMClient(model=model, api_key=key, api_base=base)

    raise ValueError(f"Unknown provider: {provider}. Use one of: openai, anthropic, xai, openrouter, gemini")


def infer_provider_from_model(model: str) -> str:
    """
    Infer provider from model string for backward compatibility.

    - "anthropic/..." or model starting with "claude" -> anthropic (via openrouter if has /)
    - "x-ai/..." or "xai/..." or "grok" in model -> xai (or openrouter)
    - Contains "/" and not anthropic/xai -> openrouter
    - Else -> openai
    """
    m = (model or "").lower()
    if m.startswith("anthropic/") or ("/" not in model and "claude" in m):
        return "anthropic"
    if m.startswith("x-ai/") or m.startswith("xai/") or "grok" in m:
        return "xai"
    if "/" in m:
        return "openrouter"
    return "openai"
