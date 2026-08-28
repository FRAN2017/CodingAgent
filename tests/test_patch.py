import json
import os

from coding_agent.tools import ToolRegistry
from coding_agent.tools import patch as patch_tools


def test_apply_patch_replaces_exact_context_and_returns_metadata(tmp_path):
    target = tmp_path / "sample.py"
    target.write_text("alpha\nold value\nomega\n", encoding="utf-8", newline="")
    registry = ToolRegistry(tmp_path)

    result = registry.execute(
        "apply_patch",
        json.dumps(
            {
                "path": "sample.py",
                "patch": "@@\n alpha\n-old value\n+new value\n omega",
            }
        ),
    )

    assert result["ok"] is True
    assert result["path"] == "sample.py"
    assert result["action"] == "patched"
    assert result["hunks_applied"] == 1
    assert result["changed"] is True
    assert result["lines_removed"] == 1
    assert result["lines_added"] == 1
    assert result["matches"] == [
        {"hunk": 1, "line": 1, "removed": 1, "added": 1}
    ]
    assert len(result["sha256"]) == 64
    assert len(result["previous_sha256"]) == 64
    assert target.read_bytes() == b"alpha\nnew value\nomega\n"


def test_apply_patch_accepts_wrapper_and_preserves_crlf(tmp_path):
    target = tmp_path / "sample.txt"
    target.write_bytes(b"alpha\r\nold\r\nomega\r\n")
    registry = ToolRegistry(tmp_path)
    patch = """*** Begin Patch
*** Update File: sample.txt
@@ -1,3 +1,3 @@
 alpha
-old
+new
 omega
*** End Patch"""

    result = registry.execute(
        "apply_patch", json.dumps({"path": "sample.txt", "patch": patch})
    )

    assert result["ok"] is True
    assert target.read_bytes() == b"alpha\r\nnew\r\nomega\r\n"


def test_apply_patch_applies_multiple_non_overlapping_hunks(tmp_path):
    target = tmp_path / "sample.txt"
    target.write_text("one\ntwo\nthree\nfour\n", encoding="utf-8", newline="")
    registry = ToolRegistry(tmp_path)
    patch = """@@
-one
+ONE
 two
@@
 three
-four
+FOUR"""

    result = registry.execute(
        "apply_patch", json.dumps({"path": "sample.txt", "patch": patch})
    )

    assert result["ok"] is True
    assert result["hunks_applied"] == 2
    assert target.read_text(encoding="utf-8") == "ONE\ntwo\nthree\nFOUR\n"


def test_apply_patch_rejects_ambiguous_context_without_modifying_file(tmp_path):
    target = tmp_path / "sample.txt"
    original = "target\nmiddle\ntarget\n"
    target.write_text(original, encoding="utf-8", newline="")
    registry = ToolRegistry(tmp_path)

    result = registry.execute(
        "apply_patch",
        json.dumps({"path": "sample.txt", "patch": "@@\n-target\n+changed"}),
    )

    assert result["ok"] is False
    assert "matched 2 locations" in result["error"]
    assert target.read_text(encoding="utf-8") == original


def test_apply_patch_line_header_can_disambiguate_repeated_context(tmp_path):
    target = tmp_path / "sample.txt"
    target.write_text("target\ntarget\n", encoding="utf-8", newline="")
    registry = ToolRegistry(tmp_path)

    result = registry.execute(
        "apply_patch",
        json.dumps(
            {
                "path": "sample.txt",
                "patch": "@@ -2,1 +2,1 @@\n-target\n+changed",
            }
        ),
    )

    assert result["ok"] is True
    assert target.read_text(encoding="utf-8") == "target\nchanged\n"


def test_apply_patch_rejects_malformed_count_overlap_and_wrong_wrapper_path(
    tmp_path,
):
    target = tmp_path / "sample.txt"
    target.write_text("one\ntwo\nthree\n", encoding="utf-8", newline="")
    registry = ToolRegistry(tmp_path)

    wrong_count = registry.execute(
        "apply_patch",
        json.dumps(
            {
                "path": "sample.txt",
                "patch": "@@ -1,2 +1,1 @@\n-one\n+ONE",
            }
        ),
    )
    overlap = registry.execute(
        "apply_patch",
        json.dumps(
            {
                "path": "sample.txt",
                "patch": (
                    "@@\n-one\n+ONE\n two\n"
                    "@@\n two\n-three\n+THREE"
                ),
            }
        ),
    )
    wrong_path = registry.execute(
        "apply_patch",
        json.dumps(
            {
                "path": "sample.txt",
                "patch": (
                    "*** Begin Patch\n*** Update File: other.txt\n"
                    "@@\n-one\n+ONE\n*** End Patch"
                ),
            }
        ),
    )

    assert wrong_count["ok"] is False
    assert "old_count=2" in wrong_count["error"]
    assert overlap["ok"] is False
    assert "overlaps" in overlap["error"]
    assert wrong_path["ok"] is False
    assert "does not match" in wrong_path["error"]
    assert target.read_text(encoding="utf-8") == "one\ntwo\nthree\n"


def test_apply_patch_rejects_escape_ignored_missing_and_non_text_files(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("old\n", encoding="utf-8")
    (tmp_path / "binary.bin").write_bytes(b"\xff\xfe")
    registry = ToolRegistry(tmp_path)
    patch = "@@\n-old\n+new"

    escaped = registry.execute(
        "apply_patch", json.dumps({"path": "../outside.txt", "patch": patch})
    )
    ignored = registry.execute(
        "apply_patch", json.dumps({"path": ".git/config", "patch": patch})
    )
    missing = registry.execute(
        "apply_patch", json.dumps({"path": "missing.txt", "patch": patch})
    )
    non_text = registry.execute(
        "apply_patch", json.dumps({"path": "binary.bin", "patch": patch})
    )

    assert escaped["ok"] is False
    assert "escapes" in escaped["error"]
    assert ignored["ok"] is False
    assert "ignored or sensitive" in ignored["error"]
    assert missing["ok"] is False
    assert "does not exist" in missing["error"]
    assert non_text["ok"] is False
    assert "not UTF-8" in non_text["error"]


def test_apply_patch_keeps_original_when_atomic_replace_fails(tmp_path, monkeypatch):
    target = tmp_path / "sample.txt"
    target.write_text("old\n", encoding="utf-8", newline="")
    registry = ToolRegistry(tmp_path)

    def fail_replace(source_path, destination_path):
        raise OSError("simulated patch failure")

    monkeypatch.setattr(patch_tools.os, "replace", fail_replace)
    result = registry.execute(
        "apply_patch",
        json.dumps({"path": "sample.txt", "patch": "@@\n-old\n+new"}),
    )

    assert result["ok"] is False
    assert "simulated patch failure" in result["error"]
    assert target.read_text(encoding="utf-8") == "old\n"
    assert not any(path.suffix == ".tmp" for path in tmp_path.iterdir())


def test_apply_patch_preserves_file_mode(tmp_path):
    target = tmp_path / "script.py"
    target.write_text("old\n", encoding="utf-8", newline="")
    os.chmod(target, 0o744)
    original_mode = target.stat().st_mode
    registry = ToolRegistry(tmp_path)

    result = registry.execute(
        "apply_patch",
        json.dumps({"path": "script.py", "patch": "@@\n-old\n+new"}),
    )

    assert result["ok"] is True
    assert target.stat().st_mode == original_mode


def test_apply_patch_arguments_are_validated(tmp_path):
    registry = ToolRegistry(tmp_path)

    missing_patch = registry.execute(
        "apply_patch", json.dumps({"path": "sample.txt"})
    )
    null_byte = registry.execute(
        "apply_patch",
        json.dumps({"path": "sample.txt", "patch": "@@\x00"}),
    )
    extra_argument = registry.execute(
        "apply_patch",
        json.dumps({"path": "sample.txt", "patch": "@@", "unexpected": True}),
    )

    assert missing_patch["ok"] is False
    assert null_byte["ok"] is False
    assert extra_argument["ok"] is False
