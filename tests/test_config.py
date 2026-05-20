"""Tests for Config."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from lmaf.core.config import Config


class TestConfig:
    def test_defaults(self):
        config = Config()
        assert config.max_iterations == 15
        assert config.critic_every_n == 3
        assert config.max_consecutive_failures == 3
        assert "eu.anthropic" in config.default_model

    def test_model_for_default(self):
        config = Config()
        assert config.model_for("surveyor") == config.default_model

    def test_model_for_override(self):
        config = Config(model_overrides={"surveyor": "custom-model"})
        assert config.model_for("surveyor") == "custom-model"
        assert config.model_for("planner") == config.default_model

    def test_from_yaml(self):
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write("max_iterations: 5\ncritic_every_n: 2\n")
            path = Path(f.name)

        config = Config.from_yaml(path)
        assert config.max_iterations == 5
        assert config.critic_every_n == 2
        path.unlink()

    def test_from_yaml_ignores_unknown_fields(self):
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write("max_iterations: 10\nunknown_field: true\n")
            path = Path(f.name)

        config = Config.from_yaml(path)
        assert config.max_iterations == 10
        path.unlink()

    def test_from_env(self):
        env = {
            "AWS_REGION": "us-east-1",
            "AWS_ACCESS_KEY_ID": "test-key",
            "AWS_SECRET_ACCESS_KEY": "test-secret",
            "SECONDLAYER_API_KEY": "sl-key",
        }
        with patch.dict(os.environ, env, clear=False):
            config = Config.from_env()
            assert config.aws_region == "us-east-1"
            assert config.aws_access_key_id == "test-key"
            assert config.secondlayer_api_key == "sl-key"

    def test_from_env_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            config = Config.from_env()
            assert config.aws_region == ""
            assert config.secondlayer_api_url == "https://legal.org.ua/api"
