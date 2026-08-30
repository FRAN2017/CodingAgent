import json

import pytest

from coding_agent.agent import Agent
from coding_agent.checkpoints import RestoreResult
from coding_agent.context import ConversationHistory
from coding_agent.protocol import ModelTurn
from coding_agent.sessions import (
    JsonSessionStore,
    ProviderSegment,
    SessionDocument,
    SessionError,
    adapt_messages_for_provider,
)
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


def test_version_one_session_is_migrated_without_changing_messages(tmp_path):
    store = JsonSessionStore(tmp_path)
    original = _document(store, _completed_history())
    legacy_data = original.to_dict()
    legacy_data["format_version"] = 1
    legacy_data.pop("provider_segments")
    legacy_data.pop("workspace_events")
    path = store.session_path("1")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(legacy_data), encoding="utf-8")

    loaded = store.load("1")

    assert loaded is not None
    assert loaded.format_version == 3
    assert loaded.messages == original.messages
    assert loaded.provider_segments == (ProviderSegment(0, "deepseek"),)
    store.save(loaded)
    saved_data = json.loads(path.read_text(encoding="utf-8"))
    assert saved_data["format_version"] == 3
    assert saved_data["messages"] == legacy_data["messages"]
    assert saved_data["provider_segments"] == [
        {"start_index": 0, "provider": "deepseek"}
    ]
    assert saved_data["workspace_events"] == []


def test_version_two_session_is_migrated_with_empty_workspace_events(tmp_path):
    store = JsonSessionStore(tmp_path)
    data = _document(store, _completed_history()).to_dict()
    data["format_version"] = 2
    data.pop("workspace_events")
    path = store.session_path("version-two")
    data["session_id"] = "version-two"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(data), encoding="utf-8")

    loaded = store.load("version-two")

    assert loaded is not None
    assert loaded.format_version == 3
    assert loaded.workspace_events == ()


def test_provider_adapter_removes_only_foreign_reasoning_without_mutation():
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "deepseek task"},
        {
            "role": "assistant",
            "content": "deepseek answer",
            "reasoning_content": "deepseek private reasoning",
        },
        {"role": "user", "content": "qianwen task"},
        {
            "role": "assistant",
            "content": "qianwen answer",
            "reasoning_content": "qianwen private reasoning",
        },
    ]
    segments = (
        ProviderSegment(0, "deepseek"),
        ProviderSegment(3, "qianwen"),
    )

    adapted = adapt_messages_for_provider(
        messages,
        segments,
        target_provider="qianwen",
    )

    assert "reasoning_content" not in adapted[2]
    assert adapted[4]["reasoning_content"] == "qianwen private reasoning"
    assert messages[2]["reasoning_content"] == "deepseek private reasoning"


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


class ReasoningAnswerClient:
    def __init__(self, answer, reasoning, inspect_request=None):
        self.answer = answer
        self.reasoning = reasoning
        self.inspect_request = inspect_request

    def complete(self, messages, tools):
        if self.inspect_request is not None:
            self.inspect_request(messages)
        return ModelTurn(
            content=self.answer,
            reasoning_content=self.reasoning,
            finish_reason="stop",
        )


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


def test_agent_switches_provider_using_a_converted_request_copy(tmp_path):
    store = JsonSessionStore(tmp_path)
    first = Agent(
        ReasoningAnswerClient("deepseek answer", "deepseek reasoning"),
        tmp_path,
        session_store=store,
        session_id="switch-provider",
        provider="deepseek",
        model="deepseek-model",
    )
    first.run("first task")

    def inspect_qianwen_request(messages):
        old_answer = next(
            message
            for message in messages
            if message.get("content") == "deepseek answer"
        )
        assert "reasoning_content" not in old_answer

    second = Agent(
        ReasoningAnswerClient(
            "qianwen answer",
            "qianwen reasoning",
            inspect_request=inspect_qianwen_request,
        ),
        tmp_path,
        session_store=JsonSessionStore(tmp_path),
        session_id="switch-provider",
        provider="qianwen",
        model="qianwen-model",
    )
    second.run("second task")

    loaded = store.load("switch-provider")
    assert loaded is not None
    assert loaded.provider == "qianwen"
    assert loaded.model == "qianwen-model"
    assert loaded.provider_segments == (
        ProviderSegment(0, "deepseek"),
        ProviderSegment(3, "qianwen"),
    )
    assistant_messages = [
        message for message in loaded.messages if message["role"] == "assistant"
    ]
    assert assistant_messages[0]["reasoning_content"] == "deepseek reasoning"
    assert assistant_messages[1]["reasoning_content"] == "qianwen reasoning"


def test_workspace_restore_event_is_persisted_and_injected_into_request(tmp_path):
    store = JsonSessionStore(tmp_path)
    first = Agent(
        FinalAnswerClient("first task", "first answer"),
        tmp_path,
        session_store=store,
        session_id="restore-event",
        provider="deepseek",
        model="test-model",
    )
    first.run("first task")
    first.record_workspace_restore(
        RestoreResult(
            checkpoint_id="cp-original",
            safety_checkpoint_id="cp-safety",
            restored_files=2,
            removed_files=1,
        )
    )

    after_event = store.load("restore-event")
    assert after_event is not None
    assert len(after_event.workspace_events) == 1
    assert after_event.workspace_events[0].checkpoint_id == "cp-original"
    assert "Trusted local workspace event" not in after_event.messages[0]["content"]

    def inspect_request(messages):
        system_content = messages[0]["content"]
        assert "Trusted local workspace event" in system_content
        assert "cp-original" in system_content
        assert "re-read relevant files" in system_content

    second = Agent(
        ReasoningAnswerClient(
            "second answer",
            "second reasoning",
            inspect_request=inspect_request,
        ),
        tmp_path,
        session_store=JsonSessionStore(tmp_path),
        session_id="restore-event",
        provider="deepseek",
        model="test-model",
    )
    second.run("second task")

    reloaded = store.load("restore-event")
    assert reloaded is not None
    assert len(reloaded.workspace_events) == 1
    assert "Trusted local workspace event" not in reloaded.messages[0]["content"]


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
