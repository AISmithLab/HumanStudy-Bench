"""
Gemini API client wrapper implementing BaseLLMClient.
Uses legacy GeminiClient for actual calls (supports upload_file for PDF when used from pipeline).
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.llm.base import BaseLLMClient

# Legacy GeminiClient lives under legacy/validation_pipeline
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

try:
    from legacy.validation_pipeline.utils.gemini_client import GeminiClient as LegacyGeminiClient
except ImportError:
    LegacyGeminiClient = None


def _messages_to_prompt_and_system(messages: List[Dict[str, Any]], system: Optional[str] = None) -> tuple[str, Optional[str]]:
    """Extract single prompt string and system from messages."""
    sys_text = system
    user_parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            sys_text = content if isinstance(content, str) else ""
            if isinstance(content, list) and content:
                for c in content:
                    if c.get("type") == "text":
                        sys_text = c.get("text", "")
                        break
            continue
        if role == "user":
            if isinstance(content, str):
                user_parts.append(content)
            elif isinstance(content, list):
                for c in content:
                    if c.get("type") == "text":
                        user_parts.append(c.get("text", ""))
    prompt = "\n\n".join(user_parts) if user_parts else ""
    return prompt, sys_text


class GeminiLLMClient(BaseLLMClient):
    """Unified client wrapping legacy GeminiClient (text-only in this path; PDF via pipeline uses legacy directly)."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ):
        if LegacyGeminiClient is None:
            raise ImportError("Google Gemini SDK not installed. Install with: pip install google-genai")
        super().__init__(model=model, api_key=api_key, api_base=api_base)
        self._gemini = LegacyGeminiClient(model=model, api_key=api_key)

    def generate_text(
        self,
        messages: List[Dict[str, Any]],
        *,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        prompt, sys_text = _messages_to_prompt_and_system(messages, system=system)
        return self._gemini.generate_content(
            prompt=prompt,
            system_instruction=sys_text,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    @property
    def upload_file(self):
        """Expose legacy upload_file for pipelines that still pass PDF refs (e.g. before PDF→text migration)."""
        return self._gemini.upload_file

    def generate_content(self, prompt, system_instruction=None, temperature=0.7, max_tokens=None):
        """Legacy-style call for backward compatibility."""
        return self._gemini.generate_content(
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def generate_structured(self, prompt, system_instruction=None, response_format="json", temperature=0.3):
        """Legacy-style structured (JSON) call."""
        return self._gemini.generate_structured(
            prompt=prompt,
            system_instruction=system_instruction,
            response_format=response_format,
            temperature=temperature,
        )
