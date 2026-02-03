"""
Helper functions: generate_text and generate_json using any BaseLLMClient.
"""

import json
import re
from typing import Any, Dict, List, Optional

from src.llm.base import BaseLLMClient


def generate_text(
    client: BaseLLMClient,
    messages: List[Dict[str, Any]],
    *,
    system: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
) -> str:
    """Convenience wrapper around client.generate_text."""
    return client.generate_text(
        messages=messages,
        system=system,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _strip_json_fence(text: str) -> str:
    """Remove markdown code fence around JSON."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def generate_json(
    client: BaseLLMClient,
    messages: List[Dict[str, Any]],
    *,
    system: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: Optional[int] = None,
    json_suffix: str = "\n\nRespond with valid JSON only, no markdown code blocks.",
    retries: int = 1,
) -> Dict[str, Any]:
    """
    Generate text then parse as JSON. Appends json_suffix to the last user message.
    Strips markdown fences and retries on parse error up to retries times.
    """
    # Append JSON instruction to last user message
    msgs = []
    last_user_idx = -1
    for i, m in enumerate(messages):
        if m.get("role") == "user":
            last_user_idx = i
    for i, m in enumerate(messages):
        if i == last_user_idx and last_user_idx >= 0:
            content = m.get("content", "")
            if isinstance(content, str):
                content = content + json_suffix
            else:
                content = list(content) + [{"type": "text", "text": json_suffix}]
            msgs.append({**m, "content": content})
        else:
            msgs.append(dict(m))
    text = ""
    last_error = None
    for attempt in range(retries + 1):
        try:
            text = client.generate_text(
                messages=msgs,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            cleaned = _strip_json_fence(text)
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            last_error = e
            if attempt < retries and last_user_idx >= 0:
                extra = "\n\nFix the JSON and output again."
                c = msgs[last_user_idx].get("content", "")
                msgs[last_user_idx]["content"] = c + extra if isinstance(c, str) else c
    raise ValueError(
        f"Failed to parse JSON after {retries + 1} attempt(s): {last_error}. "
        f"Response preview: {text[:500] if text else 'empty'}"
    )
