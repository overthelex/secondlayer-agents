"""LLM provider abstraction layer -- AWS Bedrock only."""

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
    """Call Anthropic via AWS Bedrock. No direct API calls."""
    model = config.model_for(model_key)
    return await _call_bedrock(config, system, user, model, json_mode, tools)


async def _call_bedrock(
    config: Config,
    system: str,
    user: str,
    model: str,
    json_mode: bool,
    tools: list[dict] | None,
) -> LLMResponse:
    import anthropic

    client = anthropic.AsyncAnthropicBedrock(
        aws_region=config.aws_region,
        aws_access_key=config.aws_access_key_id or None,
        aws_secret_key=config.aws_secret_access_key or None,
    )

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
