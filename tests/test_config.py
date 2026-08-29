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
    monkeypatch.setenv("DEEPSEEK_REQUEST_TIMEOUT_SECONDS", "240.5")
    monkeypatch.setenv("DEEPSEEK_MAX_RETRIES", "4")

    config = Config.from_env()

    assert config.api_key == "test-key"
    assert config.model == "test-model"
    assert config.thinking_enabled is False
    assert config.request_timeout_seconds == 240.5
    assert config.max_retries == 4


def test_config_keeps_original_deepseek_alias():
    assert Config is DeepseekConfig


def test_direct_config_rejects_invalid_request_settings():
    with pytest.raises(ConfigurationError, match="request_timeout_seconds"):
        DeepseekConfig(api_key="test-key", request_timeout_seconds=0)
    with pytest.raises(ConfigurationError, match="max_retries"):
        QianwenConfig(api_key="test-key", max_retries=-1)


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
    monkeypatch.setenv("QIANWEN_REQUEST_TIMEOUT_SECONDS", "300")
    monkeypatch.setenv("QIANWEN_MAX_RETRIES", "1")

    config = QianwenConfig.from_env()

    assert config.api_key == "qianwen-test-key"
    assert config.base_url == "https://example.test/v1"
    assert config.model == "qwen-test-model"
    assert config.thinking_enabled is False
    assert config.reasoning_effort == "low"
    assert config.request_timeout_seconds == 300
    assert config.max_retries == 1


def test_qianwen_config_uses_portable_default_base_url():
    config = QianwenConfig(api_key="test-key")

    assert config.base_url == QIANWEN_DEFAULT_BASE_URL


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("QIANWEN_REQUEST_TIMEOUT_SECONDS", "slow", "must be a number"),
        ("QIANWEN_REQUEST_TIMEOUT_SECONDS", "0", "must be between"),
        ("QIANWEN_MAX_RETRIES", "many", "must be an integer"),
        ("QIANWEN_MAX_RETRIES", "11", "must be between"),
    ],
)
def test_qianwen_config_rejects_invalid_request_settings(
    monkeypatch,
    name,
    value,
    message,
):
    monkeypatch.setenv("QIANWEN_API_KEY", "test-key")
    monkeypatch.setenv(name, value)

    with pytest.raises(ConfigurationError, match=message):
        QianwenConfig.from_env()
