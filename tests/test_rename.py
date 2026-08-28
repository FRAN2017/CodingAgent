import json

from coding_agent.tools import ToolRegistry
from coding_agent.tools import rename as rename_tools


def test_rename_file_moves_file_and_preserves_content(tmp_path):
    source = tmp_path / "bubble_sort.py"
    destination = tmp_path / "selection_sort.py"
    source.write_text("def selection_sort(values):\n    return values\n", encoding="utf-8")
    source_size = source.stat().st_size
    registry = ToolRegistry(tmp_path)

    result = registry.execute(
        "rename_file",
        json.dumps(
            {
                "source": "bubble_sort.py",
                "destination": "selection_sort.py",
            }
        ),
    )

    assert result == {
        "ok": True,
        "source": "bubble_sort.py",
        "destination": "selection_sort.py",
        "action": "renamed",
        "bytes_moved": source_size,
        "replaced_bytes": None,
        "parent_dirs_created": False,
    }
    assert not source.exists()
    assert destination.read_text(encoding="utf-8") == (
        "def selection_sort(values):\n    return values\n"
    )


def test_rename_file_requires_explicit_overwrite(tmp_path):
    source = tmp_path / "source.txt"
    destination = tmp_path / "destination.txt"
    source.write_text("new", encoding="utf-8")
    destination.write_text("old content", encoding="utf-8")
    registry = ToolRegistry(tmp_path)

    refused = registry.execute(
        "rename_file",
        json.dumps({"source": "source.txt", "destination": "destination.txt"}),
    )

    assert refused["ok"] is False
    assert "overwrite=true" in refused["error"]
    assert source.read_text(encoding="utf-8") == "new"
    assert destination.read_text(encoding="utf-8") == "old content"

    replaced = registry.execute(
        "rename_file",
        json.dumps(
            {
                "source": "source.txt",
                "destination": "destination.txt",
                "overwrite": True,
            }
        ),
    )

    assert replaced["ok"] is True
    assert replaced["action"] == "replaced"
    assert replaced["bytes_moved"] == 3
    assert replaced["replaced_bytes"] == 11
    assert not source.exists()
    assert destination.read_text(encoding="utf-8") == "new"


def test_rename_file_can_create_destination_parent_directories(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("pass\n", encoding="utf-8")
    registry = ToolRegistry(tmp_path)

    refused = registry.execute(
        "rename_file",
        json.dumps({"source": "source.py", "destination": "src/new_name.py"}),
    )
    created = registry.execute(
        "rename_file",
        json.dumps(
            {
                "source": "source.py",
                "destination": "src/new_name.py",
                "create_parent_dirs": True,
            }
        ),
    )

    assert refused["ok"] is False
    assert "parent directory does not exist" in refused["error"]
    assert created["ok"] is True
    assert created["parent_dirs_created"] is True
    assert not source.exists()
    assert (tmp_path / "src" / "new_name.py").read_text(encoding="utf-8") == "pass\n"


def test_rename_file_rejects_same_path_missing_source_and_directory(tmp_path):
    (tmp_path / "file.txt").write_text("content", encoding="utf-8")
    (tmp_path / "folder").mkdir()
    registry = ToolRegistry(tmp_path)

    same = registry.execute(
        "rename_file",
        json.dumps({"source": "file.txt", "destination": "./file.txt"}),
    )
    missing = registry.execute(
        "rename_file",
        json.dumps({"source": "missing.txt", "destination": "new.txt"}),
    )
    directory = registry.execute(
        "rename_file",
        json.dumps({"source": "folder", "destination": "renamed-folder"}),
    )

    assert same["ok"] is False
    assert "must be different" in same["error"]
    assert missing["ok"] is False
    assert "does not exist" in missing["error"]
    assert directory["ok"] is False
    assert "not a file" in directory["error"]


def test_rename_file_rejects_escape_and_ignored_paths(tmp_path):
    (tmp_path / "source.txt").write_text("content", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("config", encoding="utf-8")
    registry = ToolRegistry(tmp_path)

    source_escape = registry.execute(
        "rename_file",
        json.dumps({"source": "../outside.txt", "destination": "new.txt"}),
    )
    destination_escape = registry.execute(
        "rename_file",
        json.dumps({"source": "source.txt", "destination": "../outside.txt"}),
    )
    ignored_source = registry.execute(
        "rename_file",
        json.dumps({"source": ".git/config", "destination": "config.txt"}),
    )
    ignored_destination = registry.execute(
        "rename_file",
        json.dumps({"source": "source.txt", "destination": ".git/new.txt"}),
    )

    assert source_escape["ok"] is False
    assert "escapes" in source_escape["error"]
    assert destination_escape["ok"] is False
    assert "escapes" in destination_escape["error"]
    assert ignored_source["ok"] is False
    assert "ignored or sensitive" in ignored_source["error"]
    assert ignored_destination["ok"] is False
    assert "ignored or sensitive" in ignored_destination["error"]
    assert (tmp_path / "source.txt").exists()


def test_rename_file_keeps_source_when_atomic_move_fails(tmp_path, monkeypatch):
    source = tmp_path / "source.txt"
    source.write_text("content", encoding="utf-8")
    registry = ToolRegistry(tmp_path)

    def fail_replace(source_path, destination_path):
        raise OSError("simulated move failure")

    monkeypatch.setattr(rename_tools.os, "replace", fail_replace)
    result = registry.execute(
        "rename_file",
        json.dumps({"source": "source.txt", "destination": "destination.txt"}),
    )

    assert result["ok"] is False
    assert "simulated move failure" in result["error"]
    assert source.read_text(encoding="utf-8") == "content"
    assert not (tmp_path / "destination.txt").exists()


def test_rename_file_arguments_are_validated(tmp_path):
    registry = ToolRegistry(tmp_path)

    missing_destination = registry.execute(
        "rename_file", json.dumps({"source": "source.txt"})
    )
    extra_argument = registry.execute(
        "rename_file",
        json.dumps(
            {
                "source": "source.txt",
                "destination": "destination.txt",
                "unexpected": True,
            }
        ),
    )

    assert missing_destination["ok"] is False
    assert extra_argument["ok"] is False
