"""LLM provider abstraction layer."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.config import Config


@dataclass
class LLMResponse:
    content: str = ""
    tokens_used: int = 0
    parsed_json: dict | None = None
    jurisdiction: str = ""
    tool_calls: list[dict] = field(default_factory=list)


class ContextTooLongError(Exception):
    pass


async def call_llm(
    config: Config,
    system: str,
    user: str,
    model_key: str = "default",
    json_mode: bool = False,
    tools: list[dict] | None = None,
) -> LLMResponse:
    """Call the LLM provider configured for this agent role.

    Each agent call starts from a fresh context -- no conversation history.
    """
    model = config.model_for(model_key)

    if config.default_provider == "anthropic" or model.startswith("claude"):
        return await _call_anthropic(config, system, user, model, json_mode, tools)
    elif config.default_provider == "openai" or model.startswith("gpt"):
        return await _call_openai(config, system, user, model, json_mode, tools)
    else:
        raise ValueError(f"Unknown provider: {config.default_provider}")


async def _call_anthropic(
    config: Config,
    system: str,
    user: str,
    model: str,
    json_mode: bool,
    tools: list[dict] | None,
) -> LLMResponse:
    import anthropic

    client = anthropic.AsyncAnthropic(api_key=config.anthropic_api_key)

    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": 8192,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }

    if json_mode:
        kwargs["messages"][0]["content"] += "\n\nВідповідай ТІЛЬКИ валідним JSON."

    response = await client.messages.create(**kwargs)

    content = ""
    for block in response.content:
        if block.type == "text":
            content += block.text

    parsed = None
    if json_mode:
        try:
            # Try to extract JSON from the response
            text = content.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()
            parsed = json.loads(text)
        except (json.JSONDecodeError, IndexError):
            parsed = None

    tokens = response.usage.input_tokens + response.usage.output_tokens

    return LLMResponse(
        content=content,
        tokens_used=tokens,
        parsed_json=parsed,
    )


async def _call_openai(
    config: Config,
    system: str,
    user: str,
    model: str,
    json_mode: bool,
    tools: list[dict] | None,
) -> LLMResponse:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=config.openai_api_key)

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": 8192,
    }

    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = await client.chat.completions.create(**kwargs)

    content = response.choices[0].message.content or ""

    parsed = None
    if json_mode:
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            parsed = None

    tokens = (response.usage.prompt_tokens + response.usage.completion_tokens) if response.usage else 0

    return LLMResponse(
        content=content,
        tokens_used=tokens,
        parsed_json=parsed,
    )
