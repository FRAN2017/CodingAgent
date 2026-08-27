import pytest

from coding_agent.config import Config, ConfigurationError


def test_config_requires_api_key(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    with pytest.raises(ConfigurationError, match="DEEPSEEK_API_KEY"):
        Config.from_env()


def test_config_reads_deepseek_environment(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("DEEPSEEK_MODEL", "test-model")
    monkeypatch.setenv("DEEPSEEK_THINKING", "disabled")

    config = Config.from_env()

    assert config.api_key == "test-key"
    assert config.model == "test-model"
    assert config.thinking_enabled is False
