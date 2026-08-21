from __future__ import annotations

from pathlib import Path

import pytest

from tend.orchestrator.code_snapshot import (
    CODE_SUBDIR_NAME,
    OrchestratorCodeSnapshotError,
    code_dir_for_root,
    code_snapshot_is_present,
    create_code_snapshot,
    repoint_uv_project_prefix,
    require_code_snapshot,
)


def _make_checkout(path: Path) -> Path:
    """Create a fake tend working checkout with committed and ignored content."""

    (path / "src" / "tend").mkdir(parents=True)
    (path / "src" / "tend" / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (path / "pyproject.toml").write_text("[project]\nname = 'tend'\n", encoding="utf-8")
    # Ignored artifacts that must not be copied.
    (path / ".git").mkdir()
    (path / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (path / ".venv").mkdir()
    (path / ".venv" / "junk").write_text("nope\n", encoding="utf-8")
    (path / ".lake").mkdir()
    (path / ".lake" / "build").write_text("artifact\n", encoding="utf-8")
    (path / "src" / "tend" / "__pycache__").mkdir()
    (path / "src" / "tend" / "__pycache__" / "module.cpython-313.pyc").write_text(
        "bytecode\n", encoding="utf-8"
    )
    (path / "src" / "tend" / "stale.pyc").write_text("bytecode\n", encoding="utf-8")
    return path


def test_code_dir_for_root_uses_standard_subdir(tmp_path: Path) -> None:
    assert code_dir_for_root(tmp_path) == tmp_path / CODE_SUBDIR_NAME


def test_create_copies_working_tree_including_uncommitted_edits(tmp_path: Path) -> None:
    checkout = _make_checkout(tmp_path / "tend")
    # Uncommitted edit: the copy must capture the live working tree as-is.
    (checkout / "src" / "tend" / "module.py").write_text("VALUE = 999\n", encoding="utf-8")
    (checkout / "src" / "tend" / "new.py").write_text("NEW = 1\n", encoding="utf-8")
    code_dir = code_dir_for_root(tmp_path / "run")

    result = create_code_snapshot(source_checkout=checkout, code_dir=code_dir)

    assert result == code_dir
    assert (code_dir / "src" / "tend" / "module.py").read_text(
        encoding="utf-8"
    ) == "VALUE = 999\n"
    assert (code_dir / "src" / "tend" / "new.py").read_text(encoding="utf-8") == "NEW = 1\n"
    assert (code_dir / "pyproject.toml").is_file()


def test_create_skips_ignored_paths(tmp_path: Path) -> None:
    checkout = _make_checkout(tmp_path / "tend")
    code_dir = code_dir_for_root(tmp_path / "run")

    create_code_snapshot(source_checkout=checkout, code_dir=code_dir)

    assert not (code_dir / ".git").exists()
    assert not (code_dir / ".venv").exists()
    assert not (code_dir / ".lake").exists()
    assert not (code_dir / "src" / "tend" / "__pycache__").exists()
    assert not (code_dir / "src" / "tend" / "stale.pyc").exists()


def test_create_rejects_nonexistent_source(tmp_path: Path) -> None:
    with pytest.raises(OrchestratorCodeSnapshotError, match="not a directory"):
        create_code_snapshot(
            source_checkout=tmp_path / "missing",
            code_dir=code_dir_for_root(tmp_path / "run"),
        )


def test_create_refuses_to_overwrite_existing_snapshot(tmp_path: Path) -> None:
    checkout = _make_checkout(tmp_path / "tend")
    code_dir = code_dir_for_root(tmp_path / "run")
    create_code_snapshot(source_checkout=checkout, code_dir=code_dir)

    with pytest.raises(OrchestratorCodeSnapshotError, match="refusing to overwrite"):
        create_code_snapshot(source_checkout=checkout, code_dir=code_dir)


def test_create_rejects_source_inside_snapshot_dir(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    code_dir = code_dir_for_root(run_root)
    inner = code_dir / "tend"
    _make_checkout(inner)

    with pytest.raises(OrchestratorCodeSnapshotError, match="must not live inside"):
        create_code_snapshot(source_checkout=inner, code_dir=code_dir)


def test_create_rejects_snapshot_dir_inside_source_checkout(tmp_path: Path) -> None:
    checkout = _make_checkout(tmp_path / "tend")
    code_dir = checkout / "runs" / "run-1" / "code"

    with pytest.raises(OrchestratorCodeSnapshotError, match="must not live inside"):
        create_code_snapshot(source_checkout=checkout, code_dir=code_dir)


def test_present_helpers_reflect_state(tmp_path: Path) -> None:
    checkout = _make_checkout(tmp_path / "tend")
    code_dir = code_dir_for_root(tmp_path / "run")
    assert code_snapshot_is_present(code_dir) is False
    create_code_snapshot(source_checkout=checkout, code_dir=code_dir)
    assert code_snapshot_is_present(code_dir) is True


def test_require_returns_resolved_dir_when_present(tmp_path: Path) -> None:
    checkout = _make_checkout(tmp_path / "tend")
    code_dir = code_dir_for_root(tmp_path / "run")
    create_code_snapshot(source_checkout=checkout, code_dir=code_dir)

    assert require_code_snapshot(code_dir) == code_dir.resolve()


def test_require_hard_fails_when_missing(tmp_path: Path) -> None:
    with pytest.raises(OrchestratorCodeSnapshotError, match="no orchestrator code snapshot"):
        require_code_snapshot(code_dir_for_root(tmp_path / "run"))


def test_require_hard_fails_when_empty(tmp_path: Path) -> None:
    code_dir = code_dir_for_root(tmp_path / "run")
    code_dir.mkdir(parents=True)
    with pytest.raises(OrchestratorCodeSnapshotError, match="no orchestrator code snapshot"):
        require_code_snapshot(code_dir)


def test_repoint_rewrites_first_project_value(tmp_path: Path) -> None:
    argv = [
        "uv",
        "run",
        "--project",
        "/old/tend",
        "tend-agent",
        "--agent",
        "{worker_agent_config_path}",
    ]
    code_dir = tmp_path / "code"

    result = repoint_uv_project_prefix(argv, project=code_dir)

    assert result == (
        "uv",
        "run",
        "--project",
        str(code_dir.resolve()),
        "tend-agent",
        "--agent",
        "{worker_agent_config_path}",
    )


def test_repoint_leaves_non_uv_argv_unchanged(tmp_path: Path) -> None:
    argv = ["tend-agent", "--agent", "x"]
    assert repoint_uv_project_prefix(argv, project=tmp_path / "code") == tuple(argv)


def test_repoint_leaves_uv_without_project_unchanged(tmp_path: Path) -> None:
    argv = ["uv", "run", "tend-agent", "--agent", "x"]
    assert repoint_uv_project_prefix(argv, project=tmp_path / "code") == tuple(argv)
