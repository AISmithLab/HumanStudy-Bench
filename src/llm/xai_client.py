"""
xAI Grok API client (OpenAI-compatible base_url).
"""

import os
from typing import Any, Dict, List, Optional

from src.llm.base import BaseLLMClient
from src.llm.openai_client import OpenAIClient


class XAIClient(OpenAIClient):
    """xAI Grok via OpenAI-compatible API (api.x.ai/v1)."""

    DEFAULT_BASE = "https://api.x.ai/v1"

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ):
        api_key = api_key or os.getenv("XAI_API_KEY")
        if not api_key:
            raise ValueError("XAI_API_KEY not set and api_key not provided")
        base = api_base or self.DEFAULT_BASE
        super().__init__(model=model, api_key=api_key, api_base=base)
