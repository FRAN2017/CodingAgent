import json

from coding_agent.tools import ToolRegistry
from coding_agent.tools import search as search_tools


def test_search_text_recurses_in_stable_order_and_returns_locations(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / ".git").mkdir()
    (tmp_path / "README.md").write_text("find needle here\n", encoding="utf-8")
    (tmp_path / "src" / "module.py").write_text(
        "first line\n    needle = True\n", encoding="utf-8"
    )
    (tmp_path / ".git" / "config").write_text("needle\n", encoding="utf-8")
    (tmp_path / "binary.bin").write_bytes(b"\xff\xfeneedle")
    registry = ToolRegistry(tmp_path)

    result = registry.execute("search_text", json.dumps({"query": "needle"}))

    assert result["ok"] is True
    assert result["match_count"] == 2
    assert result["truncated"] is False
    assert [match["path"] for match in result["matches"]] == [
        "README.md",
        "src/module.py",
    ]
    assert result["matches"][0] == {
        "path": "README.md",
        "line": 1,
        "column": 6,
        "text": "find needle here",
        "line_truncated": False,
    }
    assert result["matches"][1]["line"] == 2
    assert result["matches"][1]["column"] == 5
    assert result["files_considered"] == 3
    assert result["files_searched"] == 2
    assert result["skipped"]["non_text"] == 1


def test_search_text_supports_case_insensitive_search_and_file_glob(tmp_path):
    (tmp_path / "module.py").write_text("TARGET value\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("target value\n", encoding="utf-8")
    registry = ToolRegistry(tmp_path)

    result = registry.execute(
        "search_text",
        json.dumps(
            {
                "query": "target",
                "case_sensitive": False,
                "file_pattern": "*.py",
            }
        ),
    )

    assert result["ok"] is True
    assert result["file_pattern"] == "*.py"
    assert result["match_count"] == 1
    assert result["matches"][0]["path"] == "module.py"
    assert result["matches"][0]["column"] == 1


def test_search_text_is_case_sensitive_by_default(tmp_path):
    (tmp_path / "sample.txt").write_text("Needle\nneedle\n", encoding="utf-8")
    registry = ToolRegistry(tmp_path)

    result = registry.execute("search_text", json.dumps({"query": "needle"}))

    assert result["ok"] is True
    assert result["case_sensitive"] is True
    assert result["match_count"] == 1
    assert result["matches"][0]["line"] == 2


def test_search_text_can_search_one_file(tmp_path):
    (tmp_path / "one.py").write_text("alpha\nbeta\n", encoding="utf-8")
    (tmp_path / "two.py").write_text("beta\n", encoding="utf-8")
    registry = ToolRegistry(tmp_path)

    result = registry.execute(
        "search_text", json.dumps({"query": "beta", "path": "one.py"})
    )

    assert result["ok"] is True
    assert result["path"] == "one.py"
    assert result["files_considered"] == 1
    assert result["files_searched"] == 1
    assert [match["path"] for match in result["matches"]] == ["one.py"]


def test_search_text_reports_result_and_file_limit_truncation(tmp_path):
    (tmp_path / "a.txt").write_text("match\nmatch\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("match\n", encoding="utf-8")
    registry = ToolRegistry(tmp_path)

    result_limited = registry.execute(
        "search_text", json.dumps({"query": "match", "max_results": 1})
    )
    file_limited = registry.execute(
        "search_text",
        json.dumps({"query": "absent", "max_files": 1}),
    )

    assert result_limited["ok"] is True
    assert result_limited["match_count"] == 1
    assert result_limited["truncated"] is True
    assert file_limited["ok"] is True
    assert file_limited["files_considered"] == 1
    assert file_limited["truncated"] is True


def test_search_text_skips_files_over_size_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(search_tools, "MAX_SEARCH_FILE_BYTES", 4)
    (tmp_path / "large.txt").write_text("needle", encoding="utf-8")
    registry = ToolRegistry(tmp_path)

    result = registry.execute("search_text", json.dumps({"query": "needle"}))

    assert result["ok"] is True
    assert result["match_count"] == 0
    assert result["files_considered"] == 1
    assert result["files_searched"] == 0
    assert result["skipped"]["too_large"] == 1


def test_search_text_truncates_long_matching_line(tmp_path, monkeypatch):
    monkeypatch.setattr(search_tools, "MAX_MATCH_LINE_CHARS", 20)
    (tmp_path / "long.txt").write_text(
        "abcdefghijneedleklmnopqrst\n", encoding="utf-8"
    )
    registry = ToolRegistry(tmp_path)

    result = registry.execute("search_text", json.dumps({"query": "needle"}))

    assert result["ok"] is True
    assert result["matches"][0]["column"] == 11
    assert result["matches"][0]["line_truncated"] is True
    assert len(result["matches"][0]["text"]) <= 20


def test_search_text_rejects_escape_ignored_path_and_invalid_arguments(tmp_path):
    (tmp_path / ".git").mkdir()
    registry = ToolRegistry(tmp_path)

    escaped = registry.execute(
        "search_text", json.dumps({"query": "x", "path": ".."})
    )
    ignored = registry.execute(
        "search_text", json.dumps({"query": "x", "path": ".git"})
    )
    multiline = registry.execute(
        "search_text", json.dumps({"query": "first\nsecond"})
    )
    zero_results = registry.execute(
        "search_text", json.dumps({"query": "x", "max_results": 0})
    )
    extra_argument = registry.execute(
        "search_text", json.dumps({"query": "x", "unexpected": True})
    )

    assert escaped["ok"] is False
    assert "escapes" in escaped["error"]
    assert ignored["ok"] is False
    assert "ignored path" in ignored["error"]
    assert multiline["ok"] is False
    assert zero_results["ok"] is False
    assert extra_argument["ok"] is False
