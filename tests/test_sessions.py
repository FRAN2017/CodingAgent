import json

import pytest

from coding_agent.agent import Agent
from coding_agent.context import ConversationHistory
from coding_agent.protocol import ModelTurn
from coding_agent.sessions import JsonSessionStore, SessionDocument, SessionError
from coding_agent.tools import ToolRegistry


def _completed_history(task="first task", answer="first answer"):
    history = ConversationHistory.for_task("system", task)
    history.append_assistant({"role": "assistant", "content": answer})
    return history


def _document(store, history, session_id="1"):
    return SessionDocument.create(
        session_id=session_id,
        workspace=str(store.workspace),
        provider="deepseek",
        model="test-model",
        messages=history.messages,
    )


def test_json_session_round_trip_and_append_preserves_history(tmp_path):
    store = JsonSessionStore(tmp_path)
    original = _completed_history()
    path = store.save(_document(store, original))

    loaded = store.load("1")
    assert loaded is not None
    restored = ConversationHistory.from_messages(loaded.messages)
    restored.append_user("second task")
    restored.append_assistant({"role": "assistant", "content": "second answer"})
    store.save(loaded.with_messages(restored.messages, model="new-model"))

    reloaded = store.load("1")
    assert reloaded is not None
    assert reloaded.model == "new-model"
    assert reloaded.messages[:3] == original.messages
    assert reloaded.messages[-2:] == [
        {"role": "user", "content": "second task"},
        {"role": "assistant", "content": "second answer"},
    ]
    assert path == tmp_path / ".coding-agent" / "sessions" / "session-1.json"


@pytest.mark.parametrize("session_id", ["", "../escape", "bad/id", "a" * 65])
def test_session_id_validation_rejects_unsafe_values(tmp_path, session_id):
    store = JsonSessionStore(tmp_path)

    with pytest.raises(SessionError, match="Session id"):
        store.session_path(session_id)


def test_corrupt_session_is_rejected_without_being_overwritten(tmp_path):
    store = JsonSessionStore(tmp_path)
    path = store.session_path("broken")
    path.parent.mkdir(parents=True)
    path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(SessionError, match="invalid JSON"):
        store.load("broken")

    assert path.read_text(encoding="utf-8") == "{not-json"


def test_session_with_different_workspace_is_rejected(tmp_path):
    store = JsonSessionStore(tmp_path)
    document = _document(store, _completed_history())
    data = document.to_dict()
    data["workspace"] = str(tmp_path / "another-workspace")
    path = store.session_path("1")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(SessionError, match="different workspace"):
        store.load("1")


def test_failed_atomic_replace_keeps_previous_session(tmp_path, monkeypatch):
    store = JsonSessionStore(tmp_path)
    original = _document(store, _completed_history())
    path = store.save(original)
    previous = path.read_bytes()
    updated_history = ConversationHistory.from_messages(original.messages)
    updated_history.append_user("new task")
    updated_history.append_assistant({"role": "assistant", "content": "new answer"})

    def fail_replace(source, destination):
        raise OSError("replace failed")

    monkeypatch.setattr("coding_agent.sessions.store.os.replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        store.save(original.with_messages(updated_history.messages, model="test-model"))

    assert path.read_bytes() == previous
    assert list(path.parent.glob("*.tmp")) == []


class FinalAnswerClient:
    def __init__(self, expected_latest_task, answer, expected_old_task=None):
        self.expected_latest_task = expected_latest_task
        self.expected_old_task = expected_old_task
        self.answer = answer

    def complete(self, messages, tools):
        user_messages = [
            message["content"] for message in messages if message["role"] == "user"
        ]
        assert user_messages[-1] == self.expected_latest_task
        if self.expected_old_task is not None:
            assert self.expected_old_task in user_messages
        return ModelTurn(content=self.answer, finish_reason="stop")


def test_agent_resumes_complete_history_from_the_same_session(tmp_path):
    store = JsonSessionStore(tmp_path)
    first = Agent(
        FinalAnswerClient("create calculator", "created"),
        tmp_path,
        session_store=store,
        session_id="calculator",
        provider="deepseek",
        model="test-model",
    )
    first.run("create calculator")

    second = Agent(
        FinalAnswerClient(
            "modify the previous code",
            "modified",
            expected_old_task="create calculator",
        ),
        tmp_path,
        session_store=JsonSessionStore(tmp_path),
        session_id="calculator",
        provider="deepseek",
        model="test-model",
    )
    result = second.run("modify the previous code")

    assert result.final_answer == "modified"
    loaded = store.load("calculator")
    assert loaded is not None
    assert [message["role"] for message in loaded.messages] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
    ]


def test_model_tools_cannot_access_session_state(tmp_path):
    store = JsonSessionStore(tmp_path)
    path = store.save(_document(store, _completed_history()))
    registry = ToolRegistry(tmp_path)

    read_result = registry.execute(
        "read_file",
        json.dumps({"path": ".coding-agent/sessions/session-1.json"}),
    )
    write_result = registry.execute(
        "write_file",
        json.dumps({"path": ".coding-agent/forged.json", "content": "{}"}),
    )
    listed = registry.execute("list_files", json.dumps({"path": "."}))

    assert path.is_file()
    assert read_result["ok"] is False
    assert write_result["ok"] is False
    assert listed["ok"] is True
    assert all(
        not entry["path"].startswith(".coding-agent")
        for entry in listed["entries"]
    )
