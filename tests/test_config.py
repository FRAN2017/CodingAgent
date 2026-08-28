import pytest

from coding_agent.config import (
    QIANWEN_DEFAULT_BASE_URL,
    Config,
    ConfigurationError,
    DeepseekConfig,
    QianwenConfig,
)


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


def test_config_keeps_original_deepseek_alias():
    assert Config is DeepseekConfig


def test_qianwen_config_requires_api_key(monkeypatch):
    monkeypatch.delenv("QIANWEN_API_KEY", raising=False)

    with pytest.raises(ConfigurationError, match="QIANWEN_API_KEY"):
        QianwenConfig.from_env()


def test_config_reads_qianwen_environment(monkeypatch):
    monkeypatch.setenv("QIANWEN_API_KEY", "qianwen-test-key")
    monkeypatch.setenv("QIANWEN_BASE_URL", "https://example.test/v1/")
    monkeypatch.setenv("QIANWEN_MODEL", "qwen-test-model")
    monkeypatch.setenv("QIANWEN_THINKING", "off")
    monkeypatch.setenv("QIANWEN_REASONING_EFFORT", "low")

    config = QianwenConfig.from_env()

    assert config.api_key == "qianwen-test-key"
    assert config.base_url == "https://example.test/v1"
    assert config.model == "qwen-test-model"
    assert config.thinking_enabled is False
    assert config.reasoning_effort == "low"


def test_qianwen_config_uses_portable_default_base_url():
    config = QianwenConfig(api_key="test-key")

    assert config.base_url == QIANWEN_DEFAULT_BASE_URL
