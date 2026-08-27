import json

from coding_agent import tools
from coding_agent.tools import ToolRegistry


def test_read_file_returns_numbered_lines(tmp_path):
    (tmp_path / "sample.txt").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    registry = ToolRegistry(tmp_path)

    result = registry.execute(
        "read_file",
        json.dumps({"path": "sample.txt", "start_line": 2, "end_line": 3}),
    )

    assert result["ok"] is True
    assert result["total_lines"] == 3
    assert "2 | beta" in result["content"]
    assert "3 | gamma" in result["content"]


def test_read_file_rejects_path_escape(tmp_path):
    registry = ToolRegistry(tmp_path)

    result = registry.execute("read_file", json.dumps({"path": "../secret.txt"}))

    assert result["ok"] is False
    assert "escapes" in result["error"]


def test_tool_arguments_are_validated(tmp_path):
    registry = ToolRegistry(tmp_path)

    invalid_json = registry.execute("read_file", "{not-json")
    extra_argument = registry.execute(
        "read_file", json.dumps({"path": "x", "unexpected": True})
    )

    assert invalid_json["ok"] is False
    assert extra_argument["ok"] is False


def test_registry_exposes_read_and_list_tools(tmp_path):
    registry = ToolRegistry(tmp_path)

    names = [schema["function"]["name"] for schema in registry.schemas]

    assert names == ["read_file", "list_files", "write_file"]


def test_list_files_recurses_in_stable_order_and_returns_metadata(tmp_path):
    (tmp_path / "src" / "nested").mkdir(parents=True)
    (tmp_path / "README.md").write_text("docs", encoding="utf-8")
    (tmp_path / "src" / "a.py").write_text("print('a')", encoding="utf-8")
    (tmp_path / "src" / "nested" / "b.py").write_text(
        "print('b')", encoding="utf-8"
    )
    registry = ToolRegistry(tmp_path)

    result = registry.execute(
        "list_files", json.dumps({"path": ".", "max_depth": 3, "limit": 20})
    )

    assert result["ok"] is True
    assert result["path"] == "."
    assert result["count"] == 5
    assert result["truncated"] is False
    assert [entry["path"] for entry in result["entries"]] == [
        "README.md",
        "src",
        "src/a.py",
        "src/nested",
        "src/nested/b.py",
    ]
    assert result["entries"][0] == {
        "path": "README.md",
        "type": "file",
        "size": 4,
    }


def test_list_files_respects_max_depth(tmp_path):
    (tmp_path / "src" / "nested").mkdir(parents=True)
    (tmp_path / "src" / "module.py").write_text("", encoding="utf-8")
    (tmp_path / "src" / "nested" / "deep.py").write_text("", encoding="utf-8")
    registry = ToolRegistry(tmp_path)

    result = registry.execute(
        "list_files", json.dumps({"path": ".", "max_depth": 1})
    )

    assert result["ok"] is True
    assert [entry["path"] for entry in result["entries"]] == ["src"]


def test_list_files_ignores_secrets_generated_and_dependency_paths(tmp_path):
    ignored_paths = [
        tmp_path / ".git" / "config",
        tmp_path / ".venv" / "pyvenv.cfg",
        tmp_path / "__pycache__" / "cached.pyc",
        tmp_path / "node_modules" / "package" / "index.js",
        tmp_path / "sample.egg-info" / "PKG-INFO",
    ]
    for ignored_path in ignored_paths:
        ignored_path.parent.mkdir(parents=True, exist_ok=True)
        ignored_path.write_text("ignored", encoding="utf-8")
    (tmp_path / ".env").write_text("SECRET=value", encoding="utf-8")
    (tmp_path / ".env.example").write_text("SECRET=placeholder", encoding="utf-8")
    (tmp_path / "visible.py").write_text("", encoding="utf-8")
    registry = ToolRegistry(tmp_path)

    result = registry.execute("list_files", "{}")
    paths = [entry["path"] for entry in result["entries"]]

    assert result["ok"] is True
    assert paths == [".env.example", "visible.py"]


def test_list_files_reports_truncation(tmp_path):
    for name in ["c.txt", "a.txt", "b.txt"]:
        (tmp_path / name).write_text(name, encoding="utf-8")
    registry = ToolRegistry(tmp_path)

    result = registry.execute("list_files", json.dumps({"limit": 2}))

    assert result["ok"] is True
    assert result["count"] == 2
    assert result["truncated"] is True
    assert [entry["path"] for entry in result["entries"]] == ["a.txt", "b.txt"]


def test_list_files_rejects_escape_ignored_path_and_file_target(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / "file.txt").write_text("content", encoding="utf-8")
    registry = ToolRegistry(tmp_path)

    escaped = registry.execute("list_files", json.dumps({"path": ".."}))
    ignored = registry.execute("list_files", json.dumps({"path": ".git"}))
    file_target = registry.execute("list_files", json.dumps({"path": "file.txt"}))

    assert escaped["ok"] is False
    assert "escapes" in escaped["error"]
    assert ignored["ok"] is False
    assert "ignored path" in ignored["error"]
    assert file_target == {"ok": False, "error": "Not a directory: file.txt"}


def test_list_files_arguments_are_validated(tmp_path):
    registry = ToolRegistry(tmp_path)

    zero_depth = registry.execute("list_files", json.dumps({"max_depth": 0}))
    excessive_limit = registry.execute("list_files", json.dumps({"limit": 1001}))
    extra_argument = registry.execute(
        "list_files", json.dumps({"unexpected": True})
    )

    assert zero_depth["ok"] is False
    assert excessive_limit["ok"] is False
    assert extra_argument["ok"] is False


def test_write_file_creates_utf8_file_and_returns_metadata(tmp_path):
    registry = ToolRegistry(tmp_path)

    result = registry.execute(
        "write_file",
        json.dumps({"path": "hello.txt", "content": "你好"}, ensure_ascii=False),
    )

    assert result["ok"] is True
    assert result["path"] == "hello.txt"
    assert result["action"] == "created"
    assert result["bytes_written"] == 6
    assert result["previous_sha256"] is None
    assert result["changed"] is True
    assert result["parent_dirs_created"] is False
    assert len(result["sha256"]) == 64
    assert (tmp_path / "hello.txt").read_text(encoding="utf-8") == "你好"


def test_write_file_requires_explicit_overwrite_and_preserves_original(tmp_path):
    target = tmp_path / "existing.txt"
    target.write_text("original", encoding="utf-8")
    registry = ToolRegistry(tmp_path)

    refused = registry.execute(
        "write_file", json.dumps({"path": "existing.txt", "content": "replacement"})
    )

    assert refused["ok"] is False
    assert "overwrite=true" in refused["error"]
    assert target.read_text(encoding="utf-8") == "original"

    overwritten = registry.execute(
        "write_file",
        json.dumps(
            {"path": "existing.txt", "content": "replacement", "overwrite": True}
        ),
    )

    assert overwritten["ok"] is True
    assert overwritten["action"] == "overwritten"
    assert overwritten["previous_sha256"] is not None
    assert overwritten["previous_sha256"] != overwritten["sha256"]
    assert overwritten["changed"] is True
    assert target.read_text(encoding="utf-8") == "replacement"


def test_write_file_reports_unchanged_overwrite(tmp_path):
    target = tmp_path / "same.txt"
    target.write_text("same", encoding="utf-8")
    registry = ToolRegistry(tmp_path)

    result = registry.execute(
        "write_file",
        json.dumps({"path": "same.txt", "content": "same", "overwrite": True}),
    )

    assert result["ok"] is True
    assert result["action"] == "overwritten"
    assert result["changed"] is False
    assert result["previous_sha256"] == result["sha256"]


def test_write_file_can_create_parent_directories_explicitly(tmp_path):
    registry = ToolRegistry(tmp_path)

    refused = registry.execute(
        "write_file", json.dumps({"path": "new/nested/file.py", "content": "pass\n"})
    )
    created = registry.execute(
        "write_file",
        json.dumps(
            {
                "path": "new/nested/file.py",
                "content": "pass\n",
                "create_parent_dirs": True,
            }
        ),
    )

    assert refused["ok"] is False
    assert "Parent directory does not exist" in refused["error"]
    assert created["ok"] is True
    assert created["parent_dirs_created"] is True
    assert (tmp_path / "new" / "nested" / "file.py").read_text(
        encoding="utf-8"
    ) == "pass\n"


def test_write_file_rejects_escape_sensitive_paths_and_directory_target(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / "folder").mkdir()
    registry = ToolRegistry(tmp_path)

    escaped = registry.execute(
        "write_file", json.dumps({"path": "../outside.txt", "content": "blocked"})
    )
    secret = registry.execute(
        "write_file", json.dumps({"path": ".env", "content": "SECRET=value"})
    )
    git_file = registry.execute(
        "write_file", json.dumps({"path": ".git/config", "content": "blocked"})
    )
    directory = registry.execute(
        "write_file",
        json.dumps({"path": "folder", "content": "blocked", "overwrite": True}),
    )

    assert escaped["ok"] is False
    assert "escapes" in escaped["error"]
    assert secret["ok"] is False
    assert git_file["ok"] is False
    assert "sensitive path" in git_file["error"]
    assert directory == {"ok": False, "error": "Not a file: folder"}


def test_write_file_enforces_utf8_byte_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "MAX_WRITE_BYTES", 5)
    registry = ToolRegistry(tmp_path)

    result = registry.execute(
        "write_file",
        json.dumps({"path": "large.txt", "content": "你好"}, ensure_ascii=False),
    )

    assert result["ok"] is False
    assert "exceeds 5 UTF-8 bytes" in result["error"]
    assert not (tmp_path / "large.txt").exists()


def test_write_file_arguments_are_validated(tmp_path):
    registry = ToolRegistry(tmp_path)

    missing_content = registry.execute(
        "write_file", json.dumps({"path": "missing.txt"})
    )
    extra_argument = registry.execute(
        "write_file",
        json.dumps({"path": "x.txt", "content": "x", "unexpected": True}),
    )

    assert missing_content["ok"] is False
    assert extra_argument["ok"] is False
