from coding_agent.cli import Provider, create_client
from coding_agent.llm_client import DeepSeekClient, QianwenClient


def test_create_client_selects_deepseek(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-test-key")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-test-model")

    client, model = create_client(Provider.deepseek)

    assert isinstance(client, DeepSeekClient)
    assert model == "deepseek-test-model"


def test_create_client_selects_qianwen(monkeypatch):
    monkeypatch.setenv("QIANWEN_API_KEY", "qianwen-test-key")
    monkeypatch.setenv("QIANWEN_MODEL", "qianwen-test-model")

    client, model = create_client(Provider.qianwen)

    assert isinstance(client, QianwenClient)
    assert model == "qianwen-test-model"
