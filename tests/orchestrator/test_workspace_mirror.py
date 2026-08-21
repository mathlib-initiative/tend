"""Workspace mirror integration tests for the orchestrator.

These tests cover the first-class ``workspace_mirror`` step inserted between
``git worktree add`` (and the orchestrator-metadata gitignore step) and the
existing ``worktree_setup_command``.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from tests.orchestrator.test_orchestrator import WorktreeTestingOrchestrator

from tend.orchestrator.config import (
    AsyncOrchestratorConfig,
    AsyncOrchestratorWorkspaceMirrorConfig,
)
from tend.orchestrator.state import AsyncOrchestratorWorktree


def _initialize_git_repo(repo: Path) -> None:
    repo.mkdir()
    _run_git(repo, "init")
    _run_git(repo, "config", "user.email", "test@example.com")
    _run_git(repo, "config", "user.name", "Test User")
    _run_git(repo, "checkout", "-b", "main")
    (repo / "README.md").write_text("initial\n", encoding="utf-8")
    _run_git(repo, "add", "README.md")
    _run_git(repo, "commit", "-m", "initial")

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not available")


async def _create_worktree(
    orchestrator: WorktreeTestingOrchestrator,
    *,
    name: str,
) -> AsyncOrchestratorWorktree:
    return await orchestrator.create_fresh_worktree_for_test(name=name)


async def test_workspace_mirror_disabled_by_default_skips_mirroring(tmp_path: Path) -> None:
    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    _initialize_git_repo(entrypoint)
    # Entrypoint-local cache that should *not* leak into the worktree by default.
    (entrypoint / ".lake").mkdir()
    (entrypoint / ".lake" / "stamp").write_text("entrypoint\n", encoding="utf-8")

    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(root=root, entrypoint=entrypoint),
    )

    worktree = await _create_worktree(orchestrator, name="entrypoint-copy")

    assert not (worktree.path / ".lake").exists(), (
        "workspace_mirror is disabled by default; no entrypoint state should be copied"
    )


async def test_workspace_mirror_symlinks_configured_lake_cache(tmp_path: Path) -> None:
    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    _initialize_git_repo(entrypoint)
    (entrypoint / ".lake").mkdir()
    (entrypoint / ".lake" / "cache.bin").write_bytes(b"cached")

    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=root,
            entrypoint=entrypoint,
            workspace_mirror=AsyncOrchestratorWorkspaceMirrorConfig(
                enabled=True,
                symlink_paths=[".lake"],
            ),
        ),
    )

    worktree = await _create_worktree(orchestrator, name="entrypoint-copy")

    lake = worktree.path / ".lake"
    assert lake.is_symlink(), "configured symlink_paths entries must be linked, not copied"
    # The mirror resolves symlink targets to absolute paths.
    assert Path(lake.readlink()) == (entrypoint / ".lake").resolve()
    assert (lake / "cache.bin").read_bytes() == b"cached"


async def test_workspace_mirror_symlinks_nested_path_and_creates_intermediate_dirs(
    tmp_path: Path,
) -> None:
    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    _initialize_git_repo(entrypoint)
    nested = entrypoint / ".lake" / "packages" / "mathlib"
    nested.mkdir(parents=True)
    (nested / "Mathlib.lean").write_text("-- mathlib\n", encoding="utf-8")
    # Sibling content under ``.lake/packages/`` must be copied normally, not
    # masked away by the nested symlink rule.
    sibling = entrypoint / ".lake" / "packages" / "other"
    sibling.mkdir()
    (sibling / "Other.lean").write_text("-- other\n", encoding="utf-8")

    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=root,
            entrypoint=entrypoint,
            workspace_mirror=AsyncOrchestratorWorkspaceMirrorConfig(
                enabled=True,
                symlink_paths=[".lake/packages/mathlib"],
            ),
        ),
    )

    worktree = await _create_worktree(orchestrator, name="entrypoint-copy")

    packages = worktree.path / ".lake" / "packages"
    assert packages.is_dir() and not packages.is_symlink(), (
        "intermediate directories must be created in the worktree, not linked"
    )
    mathlib = packages / "mathlib"
    assert mathlib.is_symlink(), "nested symlink_paths entry must be linked at the leaf"
    assert Path(mathlib.readlink()) == nested.resolve()
    # Sibling content under the same intermediate directory still mirrors.
    assert (packages / "other" / "Other.lean").read_text(encoding="utf-8") == "-- other\n"


async def test_workspace_mirror_failure_cleans_up_worktree(tmp_path: Path) -> None:
    """A failing mirror step must trigger the failed-worktree cleanup path."""

    from tend.workspace.mirror import MirrorExistingPathPolicy

    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    worktree_path = root / "worktrees" / "entrypoint-copy"
    _initialize_git_repo(entrypoint)
    # ``existing_path_policy=ERROR`` + a conflicting file in the entrypoint /
    # worktree that ``git worktree add`` itself materializes (README.md) causes
    # the mirror to raise ``MirrorConflictError`` when it tries to overwrite
    # the worktree's checked-out README.md.
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=root,
            entrypoint=entrypoint,
            workspace_mirror=AsyncOrchestratorWorkspaceMirrorConfig(
                enabled=True,
                existing_path_policy=MirrorExistingPathPolicy.ERROR,
            ),
        ),
    )

    from tend.workspace.mirror import MirrorConflictError

    with pytest.raises(MirrorConflictError):
        await _create_worktree(orchestrator, name="entrypoint-copy")

    # Failed creation must roll back: worktree dir gone, no registered git worktree,
    # no orchestrator state.
    assert not worktree_path.exists()
    assert str(worktree_path) not in _run_git(entrypoint, "worktree", "list", "--porcelain")
    assert orchestrator.worktree_ids == ()


async def test_workspace_mirror_runs_before_worktree_setup_command(tmp_path: Path) -> None:
    """The mirror must populate the worktree *before* the setup command runs."""

    if shutil.which("sh") is None:
        pytest.skip("sh executable is required")

    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    _initialize_git_repo(entrypoint)
    (entrypoint / ".lake").mkdir()
    (entrypoint / ".lake" / "stamp").write_text("from-entrypoint\n", encoding="utf-8")

    from tend.orchestrator.config import (
        AsyncOrchestratorWorktreeSetupCommandConfig,
    )

    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=root,
            entrypoint=entrypoint,
            workspace_mirror=AsyncOrchestratorWorkspaceMirrorConfig(
                enabled=True,
                symlink_paths=[".lake"],
            ),
            # The setup command reads the mirrored ``.lake/stamp`` and writes a
            # marker file: if the mirror hadn't already run, the read would fail.
            worktree_setup_command=AsyncOrchestratorWorktreeSetupCommandConfig(
                argv=(
                    "sh",
                    "-c",
                    'cat "$1/.lake/stamp" > "$1/post-mirror.txt"',
                    "setup",
                    "{worktree}",
                ),
            ),
        ),
    )

    worktree = await _create_worktree(orchestrator, name="entrypoint-copy")

    assert (worktree.path / ".lake").is_symlink()
    assert (
        (worktree.path / "post-mirror.txt").read_text(encoding="utf-8") == "from-entrypoint\n"
    )


def _run_git(cwd: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()
