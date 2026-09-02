import pytest

from coding_agent.checkpoints import CheckpointError, CheckpointManager


def test_checkpoint_diff_detects_add_modify_delete_and_rename(tmp_path):
    (tmp_path / "modified.py").write_text("value = 1\n", encoding="utf-8")
    (tmp_path / "deleted.py").write_text("delete me\n", encoding="utf-8")
    (tmp_path / "old_name.py").write_text("same content\n", encoding="utf-8")
    manager = CheckpointManager(tmp_path)
    checkpoint = manager.create("change files", session_id="demo")

    (tmp_path / "modified.py").write_text("value = 2\n", encoding="utf-8")
    (tmp_path / "deleted.py").unlink()
    (tmp_path / "old_name.py").rename(tmp_path / "new_name.py")
    (tmp_path / "added.py").write_text("new file\n", encoding="utf-8")

    changes = manager.diff(checkpoint.checkpoint_id).changes
    by_status = {change.status: change for change in changes}

    assert set(by_status) == {"added", "modified", "deleted", "renamed"}
    assert by_status["added"].path == "added.py"
    assert "+new file" in (by_status["added"].patch or "")
    assert by_status["modified"].path == "modified.py"
    assert "-value = 1" in (by_status["modified"].patch or "")
    assert "+value = 2" in (by_status["modified"].patch or "")
    assert by_status["deleted"].path == "deleted.py"
    assert by_status["renamed"].old_path == "old_name.py"
    assert by_status["renamed"].path == "new_name.py"


def test_restore_recovers_original_files_and_creates_safety_checkpoint(tmp_path):
    (tmp_path / "keep.txt").write_text("original\n", encoding="utf-8")
    (tmp_path / "deleted.txt").write_text("restore me\n", encoding="utf-8")
    manager = CheckpointManager(tmp_path)
    checkpoint = manager.create("before changes")

    (tmp_path / "keep.txt").write_text("changed\n", encoding="utf-8")
    (tmp_path / "deleted.txt").unlink()
    (tmp_path / "added.txt").write_text("remove me\n", encoding="utf-8")

    result = manager.restore(checkpoint.checkpoint_id)

    assert (tmp_path / "keep.txt").read_text(encoding="utf-8") == "original\n"
    assert (tmp_path / "deleted.txt").read_text(encoding="utf-8") == "restore me\n"
    assert not (tmp_path / "added.txt").exists()
    assert result.checkpoint_id == checkpoint.checkpoint_id
    assert result.restored_files == 2
    assert result.removed_files == 1
    assert manager.get(result.safety_checkpoint_id).kind == "pre_undo"
    assert manager.diff(checkpoint.checkpoint_id).changes == ()


def test_checkpoint_ignores_protected_agent_state_and_env(tmp_path):
    (tmp_path / ".env").write_text("SECRET=old\n", encoding="utf-8")
    manager = CheckpointManager(tmp_path)
    checkpoint = manager.create("protected files")

    (tmp_path / ".env").write_text("SECRET=new\n", encoding="utf-8")

    assert checkpoint.files == ()
    assert manager.diff(checkpoint.checkpoint_id).changes == ()
    manager.restore(checkpoint.checkpoint_id)
    assert (tmp_path / ".env").read_text(encoding="utf-8") == "SECRET=new\n"


def test_restore_rejects_corrupt_object_before_changing_workspace(tmp_path):
    target = tmp_path / "code.py"
    target.write_text("before\n", encoding="utf-8")
    manager = CheckpointManager(tmp_path)
    checkpoint = manager.create("corrupt object")
    entry = checkpoint.files[0]
    object_path = (
        tmp_path
        / ".coding-agent"
        / "checkpoints"
        / "objects"
        / entry.sha256[:2]
        / entry.sha256[2:]
    )
    object_path.write_bytes(b"corrupt")
    target.write_text("after\n", encoding="utf-8")

    with pytest.raises(CheckpointError, match="corrupt"):
        manager.restore(checkpoint.checkpoint_id)

    assert target.read_text(encoding="utf-8") == "after\n"


def test_checkpoint_rejects_workspace_symlink(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("content\n", encoding="utf-8")
    link = tmp_path / "link.txt"
    try:
        link.symlink_to(source)
    except OSError:
        pytest.skip("Symbolic links are not available on this platform")

    with pytest.raises(CheckpointError, match="symbolic links"):
        CheckpointManager(tmp_path).create("scan symlink")


def test_checkpoint_operations_are_scoped_to_session(tmp_path):
    manager = CheckpointManager(tmp_path)
    alpha = manager.create("alpha task", session_id="alpha")
    beta = manager.create("beta task", session_id="beta")
    anonymous = manager.create("anonymous task")

    assert [item.checkpoint_id for item in manager.list(session_id="alpha")] == [
        alpha.checkpoint_id
    ]
    assert [item.checkpoint_id for item in manager.list(session_id="beta")] == [
        beta.checkpoint_id
    ]
    assert [item.checkpoint_id for item in manager.list(session_id=None)] == [
        anonymous.checkpoint_id
    ]
    assert manager.latest(session_id="alpha").checkpoint_id == alpha.checkpoint_id

    with pytest.raises(CheckpointError, match="current session"):
        manager.get(beta.checkpoint_id, session_id="alpha")
    with pytest.raises(CheckpointError, match="current session"):
        manager.diff(beta.checkpoint_id, session_id="alpha")
    with pytest.raises(CheckpointError, match="current session"):
        manager.restore(beta.checkpoint_id, session_id="alpha")


def test_checkpoint_session_scope_reports_when_no_checkpoint_exists(tmp_path):
    manager = CheckpointManager(tmp_path)
    manager.create("other task", session_id="other")

    with pytest.raises(CheckpointError, match="current session"):
        manager.latest(session_id="current")
