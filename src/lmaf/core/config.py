"""Configuration for the LMAF system."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    """System configuration, loaded from YAML."""

    # LLM provider
    default_provider: str = "anthropic"
    default_model: str = "us.anthropic.claude-sonnet-4-6-20250514-v1:0"

    # Per-agent model overrides
    model_overrides: dict[str, str] = field(default_factory=dict)

    # Loop control
    max_iterations: int = 15
    critic_every_n: int = 3  # run critic every N iterations
    max_consecutive_failures: int = 3

    # SecondLayer API
    secondlayer_api_url: str = "https://legal.org.ua/api"
    secondlayer_api_key: str = ""

    # Workspace
    workspace_dir: str = "workspaces"
    logs_dir: str = ""

    # API keys (loaded from env)
    anthropic_api_key: str = ""
    openai_api_key: str = ""

    # AWS Bedrock
    aws_region: str = ""
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""

    def model_for(self, agent_role: str) -> str:
        return self.model_overrides.get(agent_role, self.default_model)

    @classmethod
    def from_yaml(cls, path: Path) -> Config:
        data = yaml.safe_load(path.read_text()) or {}
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_env(cls) -> Config:
        import os

        config = cls()
        config.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        config.openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        config.secondlayer_api_key = os.environ.get("SECONDLAYER_API_KEY", "")
        config.secondlayer_api_url = os.environ.get(
            "SECONDLAYER_API_URL", config.secondlayer_api_url
        )
        config.aws_region = os.environ.get("AWS_REGION", "")
        config.aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID", "")
        config.aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
        return config
