"""Tests for the staging-worktree merge path (``merge_validation_worktree``).

When enabled, the merge pipeline trial-merges and runs validation in a
long-lived ``<root>/staging`` worktree and only fast-forwards the pristine
entrypoint to an already-validated commit. The entrypoint is never
reset/reverted, and the validation build does not hold ``entrypoint_lock`` (so
ready-task worktree creation is not starved while it runs).
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from pydantic import ValidationError
from tests.orchestrator import test_orchestrator as _test_orchestrator

from tend.orchestrator import orchestrator as _orchestrator_module
from tend.orchestrator.config import (
    AsyncOrchestratorConfig,
    AsyncOrchestratorValidationCommandConfig,
    AsyncOrchestratorWorkspaceMirrorConfig,
    AsyncOrchestratorWorktreeSetupCommandConfig,
)
from tend.orchestrator.state import (
    AsyncOrchestratorAgentRole,
    AsyncOrchestratorWorktree,
    WorktreeState,
)
from tend.orchestrator.task_io import task_directory, write_task
from tend.orchestrator.task_manager import TaskManager
from tend.orchestrator.task_validation import TaskValidationFailure
from tend.orchestrator.tasks import Task, TaskStatus

WorktreeTestingOrchestrator = _test_orchestrator.WorktreeTestingOrchestrator
_initialize_git_repo = _test_orchestrator._initialize_git_repo  # pyright: ignore[reportPrivateUsage]
_run_git = _test_orchestrator._run_git  # pyright: ignore[reportPrivateUsage]
_commit_worktree = _test_orchestrator._commit_worktree  # pyright: ignore[reportPrivateUsage]


def _task_manager_with_tasks(*tasks: Task) -> TaskManager:
    return TaskManager(tasks=list(tasks))


def _seed_tasks_dir(entrypoint: Path) -> Task:
    seed = Task(
        id="task-seed",
        title="Seed",
        summary="Seed task",
        description="Seed task to keep tasks/ non-empty.",
    )
    write_task(task_directory(entrypoint) / "001-seed.yaml", seed)
    _run_git(entrypoint, "add", "tasks")
    _run_git(entrypoint, "commit", "-m", "seed tasks dir")
    return seed


def _follow_up(seed: Task) -> Task:
    return Task(
        id="task-follow-up",
        title="Follow up",
        summary="Follow-up task",
        description="A follow-up task.",
        depends_on=[seed.id],
    )


def _validation_command(script: str, *args: str) -> AsyncOrchestratorValidationCommandConfig:
    return AsyncOrchestratorValidationCommandConfig(argv=("sh", "-c", script, "validator", *args))


def _require_git_and_sh() -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable is not available")
    if shutil.which("sh") is None:
        pytest.skip("sh executable is required")


def test_entrypoint_guard_lock_selects_lock_by_flag(tmp_path: Path) -> None:
    """Creation shares ``merge_lock`` legacy, but only ``entrypoint_lock`` with staging."""

    legacy = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=tmp_path / "a",
            entrypoint=tmp_path / "ep",
            merge_validation_worktree=False,
        ),
        task_manager=_task_manager_with_tasks(),
    )
    staging = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=tmp_path / "b",
            entrypoint=tmp_path / "ep",
            merge_validation_worktree=True,
        ),
        task_manager=_task_manager_with_tasks(),
    )
    assert legacy._entrypoint_guard_lock is legacy.runtime.merge_lock  # pyright: ignore[reportPrivateUsage]
    assert staging._entrypoint_guard_lock is staging.runtime.entrypoint_lock  # pyright: ignore[reportPrivateUsage]


async def test_staging_merge_success_fast_forwards_entrypoint_and_builds_in_staging(
    tmp_path: Path,
) -> None:
    _require_git_and_sh()

    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    _initialize_git_repo(entrypoint)
    seed = _seed_tasks_dir(entrypoint)
    original_head = _run_git(entrypoint, "rev-parse", "HEAD")
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=root,
            entrypoint=entrypoint,
            merge_validation_worktree=True,
            # Writes a marker into the build cwd so we can prove validation ran
            # in the staging worktree, not the entrypoint.
            pre_merge_validation_commands=(_validation_command("printf ran > validation-ran.txt"),),
        ),
        task_manager=_task_manager_with_tasks(seed),
    )
    worktree = await orchestrator.create_fresh_worktree_for_test(name="adds-task", task=seed)
    write_task(task_directory(worktree.path) / "002-follow-up.yaml", _follow_up(seed))
    _commit_worktree(worktree.path)
    await orchestrator.transition_worktree_for_test(worktree.worktree_id, WorktreeState.MERGE)

    await orchestrator.process_merge_queue_once_for_test()

    # The worktree closed and the entrypoint advanced (a new commit) to include
    # the follow-up task.
    assert orchestrator.worktrees_by_id[worktree.worktree_id].state is WorktreeState.CLOSED
    new_head = _run_git(entrypoint, "rev-parse", "HEAD")
    assert new_head != original_head
    assert (task_directory(entrypoint) / "002-follow-up.yaml").exists()
    # Validation ran in the staging worktree, not the entrypoint.
    staging = root / "staging"
    assert staging.is_dir()
    assert (staging / "validation-ran.txt").exists()
    assert not (entrypoint / "validation-ran.txt").exists()


async def test_staging_merge_recovers_already_published_worktree_as_closed(
    tmp_path: Path,
) -> None:
    _require_git_and_sh()

    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    _initialize_git_repo(entrypoint)
    seed = _seed_tasks_dir(entrypoint)
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=root,
            entrypoint=entrypoint,
            merge_validation_worktree=True,
            batched_merge=False,
            # Would fail if the retry tried to re-stage/re-validate already landed work.
            pre_merge_validation_commands=(_validation_command("exit 99"),),
        ),
        task_manager=_task_manager_with_tasks(seed),
    )
    worktree = await orchestrator.create_fresh_worktree_for_test(
        name="already-published",
        task=seed,
    )
    write_task(task_directory(worktree.path) / "002-follow-up.yaml", _follow_up(seed))
    _commit_worktree(worktree.path)
    worker_head = _run_git(worktree.path, "rev-parse", "HEAD")
    await orchestrator.transition_worktree_for_test(worktree.worktree_id, WorktreeState.MERGE)

    # Simulate crash after publish but before MERGE -> CLOSED persisted.
    _run_git(entrypoint, "merge", "--no-edit", worker_head)

    await orchestrator.process_merge_queue_once_for_test()

    updated = orchestrator.store.get_worktree(worktree.worktree_id)
    assert updated is not None
    assert updated.state is WorktreeState.CLOSED
    assert updated.discussion == ()
    assert orchestrator.worker_queue == ()
    assert (task_directory(entrypoint) / "002-follow-up.yaml").exists()


async def test_batched_staging_merge_recovers_already_published_worktree_as_closed(
    tmp_path: Path,
) -> None:
    _require_git_and_sh()

    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    _initialize_git_repo(entrypoint)
    seed = _seed_tasks_dir(entrypoint)
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=root,
            entrypoint=entrypoint,
            merge_validation_worktree=True,
            batched_merge=True,
            # Would fail if the retry tried to build a non-empty batch.
            pre_merge_validation_commands=(_validation_command("exit 99"),),
        ),
        task_manager=_task_manager_with_tasks(seed),
    )
    worktree = await orchestrator.create_fresh_worktree_for_test(
        name="already-published-batch",
        task=seed,
    )
    write_task(task_directory(worktree.path) / "002-follow-up.yaml", _follow_up(seed))
    _commit_worktree(worktree.path)
    worker_head = _run_git(worktree.path, "rev-parse", "HEAD")
    await orchestrator.transition_worktree_for_test(worktree.worktree_id, WorktreeState.MERGE)

    # Simulate crash after batched publish but before MERGE -> CLOSED persisted.
    _run_git(entrypoint, "merge", "--no-edit", worker_head)

    await orchestrator.process_merge_queue_once_for_test()

    updated = orchestrator.store.get_worktree(worktree.worktree_id)
    assert updated is not None
    assert updated.state is WorktreeState.CLOSED
    assert updated.discussion == ()
    assert orchestrator.worker_queue == ()
    assert (task_directory(entrypoint) / "002-follow-up.yaml").exists()


async def test_staging_pre_merge_validation_gets_agent_oom_score_adj(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_git_and_sh()

    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    _initialize_git_repo(entrypoint)
    seed = _seed_tasks_dir(entrypoint)
    seen: dict[str, object] = {}

    async def _record_validation_commands(
        _commands: object,
        cwd: Path,
        oom_score_adj: int | None = None,
    ) -> None:
        seen["cwd"] = cwd
        seen["oom_score_adj"] = oom_score_adj
        return None

    monkeypatch.setattr(
        _orchestrator_module,
        "_run_validation_commands_async",
        _record_validation_commands,
    )
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=root,
            entrypoint=entrypoint,
            merge_validation_worktree=True,
            agent_oom_score_adj=321,
            pre_merge_validation_commands=(_validation_command("true"),),
        ),
        task_manager=_task_manager_with_tasks(seed),
    )
    worktree = await orchestrator.create_fresh_worktree_for_test(name="oom-score", task=seed)
    write_task(task_directory(worktree.path) / "002-follow-up.yaml", _follow_up(seed))
    _commit_worktree(worktree.path)
    await orchestrator.transition_worktree_for_test(worktree.worktree_id, WorktreeState.MERGE)

    await orchestrator.process_merge_queue_once_for_test()

    assert seen == {"cwd": root / "staging", "oom_score_adj": 321}
    assert orchestrator.worktrees_by_id[worktree.worktree_id].state is WorktreeState.CLOSED


async def test_staging_validation_failure_leaves_entrypoint_pristine(
    tmp_path: Path,
) -> None:
    """A failing build never touches the entrypoint — no merge commit, no revert."""

    _require_git_and_sh()

    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    _initialize_git_repo(entrypoint)
    seed = _seed_tasks_dir(entrypoint)
    original_head = _run_git(entrypoint, "rev-parse", "HEAD")
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=root,
            entrypoint=entrypoint,
            merge_validation_worktree=True,
            pre_merge_validation_commands=(_validation_command("exit 1"),),
        ),
        task_manager=_task_manager_with_tasks(seed),
    )
    worktree = await orchestrator.create_fresh_worktree_for_test(name="build-fails", task=seed)
    write_task(task_directory(worktree.path) / "002-follow-up.yaml", _follow_up(seed))
    _commit_worktree(worktree.path)
    await orchestrator.transition_worktree_for_test(worktree.worktree_id, WorktreeState.MERGE)

    await orchestrator.process_merge_queue_once_for_test()

    # Entrypoint HEAD is byte-for-byte the pre-merge HEAD: never advanced, so
    # never reverted. The follow-up task did not land.
    assert _run_git(entrypoint, "rev-parse", "HEAD") == original_head
    assert not (task_directory(entrypoint) / "002-follow-up.yaml").exists()
    assert _run_git(entrypoint, "status", "--porcelain") == ""
    # The worktree returns to PENDING for another attempt, with staging-specific
    # feedback that does not claim the untouched entrypoint was reset.
    updated = orchestrator.worktrees_by_id[worktree.worktree_id]
    assert updated.state is WorktreeState.PENDING
    assert updated.discussion[-1].role is AsyncOrchestratorAgentRole.ORCHESTRATOR
    message = updated.discussion[-1].message
    assert "The staged trial merge was discarded" in message
    assert "entrypoint repository was left untouched" in message
    assert "entrypoint repository was reset" not in message


async def test_staging_malformed_task_leaves_entrypoint_pristine(tmp_path: Path) -> None:
    _require_git_and_sh()

    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    _initialize_git_repo(entrypoint)
    seed = _seed_tasks_dir(entrypoint)
    original_head = _run_git(entrypoint, "rev-parse", "HEAD")
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=root,
            entrypoint=entrypoint,
            merge_validation_worktree=True,
        ),
        task_manager=_task_manager_with_tasks(seed),
    )
    worktree = await orchestrator.create_fresh_worktree_for_test(name="bad-yaml", task=seed)
    bad_file = task_directory(worktree.path) / "002-malformed.yaml"
    bad_file.parent.mkdir(parents=True, exist_ok=True)
    bad_file.write_text("id: task-broken\n  bad: : indentation\n", encoding="utf-8")
    _commit_worktree(worktree.path)
    await orchestrator.transition_worktree_for_test(worktree.worktree_id, WorktreeState.MERGE)

    await orchestrator.process_merge_queue_once_for_test()

    assert _run_git(entrypoint, "rev-parse", "HEAD") == original_head
    assert _run_git(entrypoint, "status", "--porcelain") == ""
    updated = orchestrator.worktrees_by_id[worktree.worktree_id]
    assert updated.state is WorktreeState.PENDING
    assert updated.discussion[-1].role is AsyncOrchestratorAgentRole.ORCHESTRATOR
    message = updated.discussion[-1].message
    assert "The staged trial merge was discarded" in message
    assert "entrypoint repository was left untouched" in message
    assert "entrypoint repository was reset" not in message


async def test_staging_non_utf8_task_filename_bounce_persists_escaped_path(
    tmp_path: Path,
) -> None:
    """Raw filename bytes stay matchable but never enter durable text as surrogates."""

    _require_git_and_sh()
    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    _initialize_git_repo(entrypoint)
    seed = _seed_tasks_dir(entrypoint)
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=root,
            entrypoint=entrypoint,
            merge_validation_worktree=True,
        ),
        task_manager=_task_manager_with_tasks(seed),
    )
    worktree = await orchestrator.create_fresh_worktree_for_test(
        name="non-utf8-task-path", task=seed
    )
    filename = os.fsdecode(b"bad-\xff.yaml")
    cyclic = Task(
        id="task-non-utf8-cycle",
        title="Non-UTF-8 filename cycle",
        summary="Non-UTF-8 filename cycle",
        description="Self-cyclic task used to exercise bounce persistence.",
        depends_on=["task-non-utf8-cycle"],
    )
    write_task(task_directory(worktree.path) / filename, cyclic)
    # Keep this ordinary path in the same contribution so pre-fix code arms the
    # task gate before it learns to parse NUL-delimited unusual paths itself.
    write_task(task_directory(worktree.path) / "002-marker.yaml", _follow_up(seed))
    _commit_worktree(worktree.path)
    await orchestrator.transition_worktree_for_test(worktree.worktree_id, WorktreeState.MERGE)

    await orchestrator.process_merge_queue_once_for_test()

    updated = orchestrator.worktrees_by_id[worktree.worktree_id]
    assert updated.state is WorktreeState.PENDING
    message = updated.discussion[-1].message
    message.encode("utf-8")
    assert "bad-\\udcff.yaml" in message
    persisted = (worktree.path / ".tend" / "discussion.md").read_text(encoding="utf-8")
    assert "bad-\\udcff.yaml" in persisted


async def test_staging_worktree_reused_across_two_merges(tmp_path: Path) -> None:
    """The staging worktree is created once and reused for subsequent merges."""

    _require_git_and_sh()

    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    _initialize_git_repo(entrypoint)
    seed = _seed_tasks_dir(entrypoint)
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=root,
            entrypoint=entrypoint,
            merge_validation_worktree=True,
            pre_merge_validation_commands=(_validation_command("true"),),
        ),
        task_manager=_task_manager_with_tasks(seed),
    )

    for index in range(2):
        worktree = await orchestrator.create_fresh_worktree_for_test(
            name=f"adds-task-{index}", task=seed
        )
        task = Task(
            id=f"task-extra-{index}",
            title="Extra",
            summary="Extra task",
            description="Extra task.",
            depends_on=[seed.id],
        )
        write_task(task_directory(worktree.path) / f"01{index}-extra.yaml", task)
        _commit_worktree(worktree.path)
        await orchestrator.transition_worktree_for_test(worktree.worktree_id, WorktreeState.MERGE)
        await orchestrator.process_merge_queue_once_for_test()
        assert (
            orchestrator.worktrees_by_id[worktree.worktree_id].state
            is WorktreeState.CLOSED
        )

    # Exactly one staging worktree registered for the entrypoint repo.
    worktree_list = _run_git(entrypoint, "worktree", "list")
    assert worktree_list.count(str(root / "staging")) == 1
    assert (task_directory(entrypoint) / "010-extra.yaml").exists()
    assert (task_directory(entrypoint) / "011-extra.yaml").exists()


async def test_staging_reset_cleans_untracked_validation_scratch_files(
    tmp_path: Path,
) -> None:
    """Validation scratch files must not poison the next trial merge."""

    _require_git_and_sh()

    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    _initialize_git_repo(entrypoint)
    seed = _seed_tasks_dir(entrypoint)
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=root,
            entrypoint=entrypoint,
            merge_validation_worktree=True,
            pre_merge_validation_commands=(
                _validation_command("printf stale > Scratch.lean"),
            ),
        ),
        task_manager=_task_manager_with_tasks(seed),
    )

    first = await orchestrator.create_fresh_worktree_for_test(
        name="first-merge",
        task=seed,
    )
    write_task(task_directory(first.path) / "002-follow-up.yaml", _follow_up(seed))
    _commit_worktree(first.path)
    await orchestrator.transition_worktree_for_test(first.worktree_id, WorktreeState.MERGE)
    await orchestrator.process_merge_queue_once_for_test()
    assert orchestrator.worktrees_by_id[first.worktree_id].state is WorktreeState.CLOSED
    assert (root / "staging" / "Scratch.lean").exists()

    second = await orchestrator.create_fresh_worktree_for_test(
        name="second-merge",
        task=seed,
    )
    (second.path / "Scratch.lean").write_text("tracked scratch\n", encoding="utf-8")
    _commit_worktree(second.path)
    await orchestrator.transition_worktree_for_test(second.worktree_id, WorktreeState.MERGE)

    await orchestrator.process_merge_queue_once_for_test()

    # If staging reset leaves the stale untracked file in place, Git refuses this
    # merge with "untracked working tree files would be overwritten by merge".
    assert orchestrator.worktrees_by_id[second.worktree_id].state is WorktreeState.CLOSED
    assert (entrypoint / "Scratch.lean").read_text(encoding="utf-8") == "tracked scratch\n"


async def test_tracked_in_tree_sentinel_does_not_make_staging_provisioned(
    tmp_path: Path,
) -> None:
    """A tracked legacy-style sentinel cannot substitute for external readiness."""

    _require_git_and_sh()
    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    _initialize_git_repo(entrypoint)
    (entrypoint / ".gitignore").write_text(".setup-marker\n", encoding="utf-8")
    tracked_sentinel = entrypoint / ".tend" / "provisioned"
    tracked_sentinel.parent.mkdir()
    tracked_sentinel.write_text("tracked, not ready\n", encoding="utf-8")
    _run_git(entrypoint, "add", "-f", ".gitignore", ".tend/provisioned")
    _run_git(entrypoint, "commit", "-m", "track legacy sentinel path")
    head = _run_git(entrypoint, "rev-parse", "HEAD")
    config = AsyncOrchestratorConfig(
        root=root,
        entrypoint=entrypoint,
        merge_validation_worktree=True,
        worktree_setup_command=AsyncOrchestratorWorktreeSetupCommandConfig(
            argv=("sh", "-c", ': > "$1/.setup-marker"', "setup", "{worktree}"),
        ),
    )
    first = WorktreeTestingOrchestrator(
        config,
        task_manager=_task_manager_with_tasks(),
    )
    staging = await first._ensure_validation_worktree(  # pyright: ignore[reportPrivateUsage]
        head=head
    )
    sentinel = root / "staging.provisioned"
    assert sentinel.is_file()
    assert (staging / ".tend" / "provisioned").is_file()
    assert (staging / ".setup-marker").is_file()

    # Simulate an interrupted provisioning attempt and a process restart. Git
    # reset restores the tracked in-tree file, but only the absent external
    # sentinel controls readiness, so setup must run again.
    sentinel.unlink()
    (staging / ".setup-marker").unlink()
    resumed = WorktreeTestingOrchestrator(config)

    reused = await resumed._ensure_validation_worktree(  # pyright: ignore[reportPrivateUsage]
        head=head
    )

    assert reused == staging
    assert (staging / ".setup-marker").is_file()
    assert sentinel.is_file()
    assert resumed._validation_worktree_ready  # pyright: ignore[reportPrivateUsage]


async def test_workspace_mirror_cannot_publish_external_staging_sentinel(
    tmp_path: Path,
) -> None:
    """Mirrored entrypoint metadata remains inside staging and never marks it ready."""

    _require_git_and_sh()
    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    staging = root / "staging"
    _initialize_git_repo(entrypoint)
    mirrored = entrypoint / ".tend" / "provisioned"
    mirrored.parent.mkdir()
    mirrored.write_text("entrypoint artifact\n", encoding="utf-8")
    head = _run_git(entrypoint, "rev-parse", "HEAD")
    config = AsyncOrchestratorConfig(
        root=root,
        entrypoint=entrypoint,
        merge_validation_worktree=True,
        workspace_mirror=AsyncOrchestratorWorkspaceMirrorConfig(enabled=True),
        worktree_setup_command=AsyncOrchestratorWorktreeSetupCommandConfig(
            argv=("sh", "-c", ': > "$1/.setup-marker"', "setup", "{worktree}"),
        ),
    )

    # Simulate a kill after mirror but before setup/sentinel publication.
    _run_git(entrypoint, "worktree", "add", "--detach", str(staging), head)
    await asyncio.to_thread(
        _orchestrator_module.mirror_workspace,
        entrypoint,
        staging,
        config=config.workspace_mirror.to_workspace_mirror_config(),
    )
    assert (staging / ".tend" / "provisioned").is_file()
    assert not (staging / ".setup-marker").exists()
    assert not (root / "staging.provisioned").exists()

    orchestrator = WorktreeTestingOrchestrator(
        config,
        task_manager=_task_manager_with_tasks(),
    )
    reused = await orchestrator._ensure_validation_worktree(  # pyright: ignore[reportPrivateUsage]
        head=head
    )

    assert reused == staging
    assert (staging / ".setup-marker").is_file()
    assert (root / "staging.provisioned").is_file()
    assert orchestrator._validation_worktree_ready  # pyright: ignore[reportPrivateUsage]


async def test_external_staging_sentinel_symlink_is_not_honored(tmp_path: Path) -> None:
    """Readiness checks and publication never follow a sentinel symlink."""

    _require_git_and_sh()
    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    _initialize_git_repo(entrypoint)
    (entrypoint / ".gitignore").write_text(".setup-marker\n", encoding="utf-8")
    _run_git(entrypoint, "add", ".gitignore")
    _run_git(entrypoint, "commit", "-m", "ignore setup artifact")
    head = _run_git(entrypoint, "rev-parse", "HEAD")
    config = AsyncOrchestratorConfig(
        root=root,
        entrypoint=entrypoint,
        merge_validation_worktree=True,
        worktree_setup_command=AsyncOrchestratorWorktreeSetupCommandConfig(
            argv=("sh", "-c", ': > "$1/.setup-marker"', "setup", "{worktree}"),
        ),
    )
    first = WorktreeTestingOrchestrator(config)
    staging = await first._ensure_validation_worktree(  # pyright: ignore[reportPrivateUsage]
        head=head
    )
    sentinel = root / "staging.provisioned"
    symlink_target = root / "sentinel-target"
    symlink_target.write_text("do not overwrite\n", encoding="utf-8")
    sentinel.unlink()
    sentinel.symlink_to(symlink_target)
    (staging / ".setup-marker").unlink()

    resumed = WorktreeTestingOrchestrator(config)
    reused = await resumed._ensure_validation_worktree(  # pyright: ignore[reportPrivateUsage]
        head=head
    )

    assert reused == staging
    assert (staging / ".setup-marker").is_file()
    assert sentinel.is_file() and not sentinel.is_symlink()
    assert symlink_target.read_text(encoding="utf-8") == "do not overwrite\n"


@pytest.mark.parametrize("sentinel_present", [False, True])
async def test_unexpected_staging_symlink_is_quarantined_without_touching_victim(
    tmp_path: Path,
    sentinel_present: bool,
) -> None:
    """A staging-path symlink is moved aside, never cleaned through."""

    _require_git_and_sh()
    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    staging = root / "staging"
    _initialize_git_repo(entrypoint)
    (entrypoint / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")
    _run_git(entrypoint, "add", ".gitignore")
    _run_git(entrypoint, "commit", "-m", "add ignore rule")
    head = _run_git(entrypoint, "rev-parse", "HEAD")

    # Dirty every class that reset/clean would mutate in the symlink target.
    (entrypoint / "README.md").write_text("dirty tracked\n", encoding="utf-8")
    (entrypoint / "untracked.txt").write_text("untracked\n", encoding="utf-8")
    (entrypoint / "ignored.txt").write_text("ignored\n", encoding="utf-8")
    victim_status = _run_git(entrypoint, "status", "--porcelain", "--ignored")
    root.mkdir()
    staging.symlink_to(entrypoint, target_is_directory=True)
    if sentinel_present:
        (root / "staging.provisioned").write_text("provisioned\n", encoding="utf-8")

    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=root,
            entrypoint=entrypoint,
            merge_validation_worktree=True,
        )
    )
    provisioned = await orchestrator._ensure_validation_worktree(  # pyright: ignore[reportPrivateUsage]
        head=head
    )

    assert (entrypoint / "README.md").read_text(encoding="utf-8") == "dirty tracked\n"
    assert (entrypoint / "untracked.txt").read_text(encoding="utf-8") == "untracked\n"
    assert (entrypoint / "ignored.txt").read_text(encoding="utf-8") == "ignored\n"
    assert _run_git(entrypoint, "status", "--porcelain", "--ignored") == victim_status
    quarantined = list(root.glob("staging.invalid-*"))
    assert len(quarantined) == 1
    assert quarantined[0].is_symlink()
    assert quarantined[0].readlink() == entrypoint
    assert provisioned == staging
    assert staging.is_dir() and not staging.is_symlink()
    assert _run_git(staging, "rev-parse", "HEAD") == head
    assert str(staging) in _run_git(entrypoint, "worktree", "list", "--porcelain")
    assert (root / "staging.provisioned").is_file()


async def test_mismatched_staging_gitdir_is_quarantined_without_touching_victim(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Staging cannot borrow another registered worktree's administrative dir."""

    _require_git_and_sh()
    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    staging = root / "staging"
    victim = tmp_path / "victim"
    _initialize_git_repo(entrypoint)
    old_head = _run_git(entrypoint, "rev-parse", "HEAD")
    (entrypoint / "README.md").write_text("newer\n", encoding="utf-8")
    _run_git(entrypoint, "commit", "-am", "newer revision")
    current_head = _run_git(entrypoint, "rev-parse", "HEAD")
    _run_git(entrypoint, "worktree", "add", "--detach", str(staging), current_head)
    _run_git(entrypoint, "worktree", "add", "--detach", str(victim), current_head)
    (root / "staging.provisioned").write_text("provisioned\n", encoding="utf-8")

    victim_gitfile = victim / ".git"
    victim_gitfile_before = victim_gitfile.read_bytes()
    victim_gitdir = Path(
        victim_gitfile.read_text(encoding="utf-8").strip().removeprefix("gitdir: ")
    ).resolve()
    victim_metadata_before = {
        path.relative_to(victim_gitdir): path.read_bytes()
        for path in victim_gitdir.rglob("*")
        if path.is_file()
    }
    victim_files_before = {
        path.relative_to(victim): path.read_bytes()
        for path in victim.rglob("*")
        if path.is_file() and path != victim_gitfile
    }

    # Keep staging's own registration, but make Git commands from staging use
    # the victim's administrative directory.  The old one-way identity gate
    # accepted both facts independently and reset the victim metadata to old_head.
    (staging / ".git").write_bytes(victim_gitfile_before)
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=root,
            entrypoint=entrypoint,
            merge_validation_worktree=True,
        )
    )
    caplog.set_level(logging.WARNING)

    provisioned = await orchestrator._ensure_validation_worktree(  # pyright: ignore[reportPrivateUsage]
        head=old_head
    )

    assert victim_gitfile.read_bytes() == victim_gitfile_before
    assert {
        path.relative_to(victim_gitdir): path.read_bytes()
        for path in victim_gitdir.rglob("*")
        if path.is_file()
    } == victim_metadata_before
    assert {
        path.relative_to(victim): path.read_bytes()
        for path in victim.rglob("*")
        if path.is_file() and path != victim_gitfile
    } == victim_files_before
    assert _run_git(victim, "rev-parse", "HEAD") == current_head
    assert _run_git(victim, "status", "--porcelain") == ""

    quarantined = list(root.glob("staging.invalid-*"))
    assert len(quarantined) == 1
    assert (quarantined[0] / ".git").read_bytes() == victim_gitfile_before
    assert (quarantined[0] / "README.md").read_text(encoding="utf-8") == "newer\n"
    assert "quarantined unexpected async worktree path" in caplog.text
    assert provisioned == staging
    assert _run_git(staging, "rev-parse", "HEAD") == old_head
    assert (root / "staging.provisioned").is_file()


async def test_unregistered_staging_directory_is_quarantined_and_reprovisioned(
    tmp_path: Path,
) -> None:
    """An interrupted pre-registration directory cannot poison every resume."""

    _require_git_and_sh()
    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    staging = root / "staging"
    _initialize_git_repo(entrypoint)
    head = _run_git(entrypoint, "rev-parse", "HEAD")
    staging.mkdir(parents=True)
    (staging / "junk.txt").write_text("preserve for inspection\n", encoding="utf-8")
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=root,
            entrypoint=entrypoint,
            merge_validation_worktree=True,
        )
    )

    provisioned = await orchestrator._ensure_validation_worktree(  # pyright: ignore[reportPrivateUsage]
        head=head
    )
    reused = await orchestrator._ensure_validation_worktree(  # pyright: ignore[reportPrivateUsage]
        head=head
    )

    quarantined = list(root.glob("staging.invalid-*"))
    assert len(quarantined) == 1
    assert (quarantined[0] / "junk.txt").read_text(encoding="utf-8") == (
        "preserve for inspection\n"
    )
    assert provisioned == reused == staging
    assert _run_git(staging, "rev-parse", "HEAD") == head
    assert (root / "staging.provisioned").is_file()


async def test_resume_reclaims_missing_registered_staging_worktree(
    tmp_path: Path,
) -> None:
    """A missing checkout with a stale Git registration is rebuilt on resume."""

    _require_git_and_sh()
    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    staging = root / "staging"
    _initialize_git_repo(entrypoint)
    head = _run_git(entrypoint, "rev-parse", "HEAD")
    _run_git(entrypoint, "worktree", "add", "--detach", str(staging), head)
    _run_git(entrypoint, "worktree", "lock", str(staging))
    shutil.rmtree(staging)
    assert str(staging) in _run_git(entrypoint, "worktree", "list", "--porcelain")
    assert not (root / "staging.provisioned").exists()

    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=root,
            entrypoint=entrypoint,
            merge_validation_worktree=True,
        )
    )
    rebuilt = await orchestrator._ensure_validation_worktree(  # pyright: ignore[reportPrivateUsage]
        head=head
    )

    assert rebuilt == staging
    assert staging.is_dir()
    assert (root / "staging.provisioned").is_file()
    assert _run_git(staging, "rev-parse", "HEAD") == head
    assert _run_git(entrypoint, "worktree", "list", "--porcelain").count(str(staging)) == 1


async def test_setup_cancellation_without_sigkill_uses_sigterm_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Windows-like signal namespace selects SIGTERM twice and propagates cancellation.

    The fake runner verifies signal selection only, not Windows process-tree
    termination (the production fallback can terminate only the direct child).
    """

    started = threading.Event()
    stopped = threading.Event()
    delivered_signals: list[int] = []

    class _WindowsLikeSignals:
        SIGTERM = 15

    class _FakeSetupRunner:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def run(self) -> None:
            started.set()
            assert stopped.wait(timeout=1)

        def signal_process_group(self, signum: int) -> None:
            delivered_signals.append(signum)
            if len(delivered_signals) == 2:
                stopped.set()

    monkeypatch.setattr(_orchestrator_module, "signal", _WindowsLikeSignals)
    monkeypatch.setattr(
        _orchestrator_module,
        "_WorktreeSetupCommandRunner",
        _FakeSetupRunner,
    )
    monkeypatch.setattr(
        _orchestrator_module,
        "_VALIDATION_PROVISION_CANCEL_SETTLE_SECONDS",
        0.01,
    )
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(root=tmp_path / "orch", entrypoint=tmp_path / "entrypoint")
    )
    setup_task = asyncio.create_task(
        orchestrator._run_cancellable_worktree_setup_command(  # pyright: ignore[reportPrivateUsage]
            AsyncOrchestratorWorktreeSetupCommandConfig(argv=("setup",)),
            entrypoint=tmp_path / "entrypoint",
            worktree=tmp_path,
        )
    )
    for _ in range(100):
        if started.is_set():
            break
        await asyncio.sleep(0.01)
    assert started.is_set()

    setup_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(setup_task, timeout=1)

    assert stopped.is_set()
    assert delivered_signals == [
        _WindowsLikeSignals.SIGTERM,
        _WindowsLikeSignals.SIGTERM,
    ]


async def test_setup_cancellation_bounds_wait_for_blocked_popen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Cancellation cannot hang while Popen is blocked before process publication."""

    popen_started = threading.Event()
    release_popen = threading.Event()
    runner_finished = threading.Event()

    class _FakeSetupProcess:
        pid = 2_000_000_000
        returncode = 0

        def poll(self) -> int:
            return self.returncode

        def send_signal(self, signum: int) -> None:
            pass

        def communicate(self) -> tuple[str, str]:
            runner_finished.set()
            return "", ""

    def _blocked_popen(*args: object, **kwargs: object) -> _FakeSetupProcess:
        popen_started.set()
        assert release_popen.wait(timeout=2)
        return _FakeSetupProcess()

    monkeypatch.setattr(subprocess, "Popen", _blocked_popen)
    monkeypatch.setattr(
        _orchestrator_module,
        "_VALIDATION_PROVISION_CANCEL_SETTLE_SECONDS",
        0.01,
    )
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(root=tmp_path / "orch", entrypoint=tmp_path / "entrypoint")
    )
    setup_task = asyncio.create_task(
        orchestrator._run_cancellable_worktree_setup_command(  # pyright: ignore[reportPrivateUsage]
            AsyncOrchestratorWorktreeSetupCommandConfig(argv=("setup",)),
            entrypoint=tmp_path / "entrypoint",
            worktree=tmp_path,
        )
    )
    for _ in range(100):
        if popen_started.is_set():
            break
        await asyncio.sleep(0.01)
    assert popen_started.is_set()

    async def _release_blocked_popen_later() -> None:
        await asyncio.sleep(0.2)
        release_popen.set()

    release_task = asyncio.create_task(_release_blocked_popen_later())
    started_at = asyncio.get_running_loop().time()
    setup_task.cancel()
    with caplog.at_level(logging.WARNING):
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(setup_task, timeout=1)
    elapsed = asyncio.get_running_loop().time() - started_at

    assert elapsed < 0.1
    assert "setup command is unresponsive and will be abandoned" in caplog.text
    await release_task
    for _ in range(100):
        if runner_finished.is_set():
            break
        await asyncio.sleep(0.01)
    assert runner_finished.is_set()


async def test_cancelling_stuck_crash_reprovision_kills_setup_process_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancellation kills a SIGTERM-ignoring setup without test-side release."""

    _require_git_and_sh()
    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    block_setup = root / "block-setup"
    setup_pid = root / "setup.pid"
    _initialize_git_repo(entrypoint)
    head = _run_git(entrypoint, "rev-parse", "HEAD")
    config = AsyncOrchestratorConfig(
        root=root,
        entrypoint=entrypoint,
        merge_validation_worktree=True,
        worktree_setup_command=AsyncOrchestratorWorktreeSetupCommandConfig(
            argv=(
                "sh",
                "-c",
                'if [ -e "$1" ]; then trap "" TERM; printf "%s" "$$" > "$2"; '
                "while :; do sleep 1; done; fi",
                "setup",
                str(block_setup),
                str(setup_pid),
            ),
        ),
    )
    orchestrator = WorktreeTestingOrchestrator(
        config,
        task_manager=_task_manager_with_tasks(),
    )
    staging = await orchestrator._ensure_validation_worktree(  # pyright: ignore[reportPrivateUsage]
        head=head
    )
    block_setup.touch()
    monkeypatch.setattr(
        _orchestrator_module,
        "_VALIDATION_PROVISION_CANCEL_SETTLE_SECONDS",
        0.05,
        raising=False,
    )

    purge_task = asyncio.create_task(
        orchestrator._purge_staging_after_crash(  # pyright: ignore[reportPrivateUsage]
            staging,
            head,
            signal.SIGSEGV,
        )
    )
    for _ in range(200):
        if setup_pid.exists():
            break
        await asyncio.sleep(0.01)
    assert setup_pid.exists(), "staging setup command did not start"
    pid = int(setup_pid.read_text(encoding="utf-8"))

    purge_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(purge_task, timeout=1)

    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    assert not orchestrator._validation_worktree_ready  # pyright: ignore[reportPrivateUsage]
    assert not (root / "staging.provisioned").exists()
    assert not staging.exists()


async def test_crash_purge_clears_sentinel_before_waiting_for_creation_lock(
    tmp_path: Path,
) -> None:
    """An interruption while purge is lock-blocked must force cold resume."""

    _require_git_and_sh()
    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    _initialize_git_repo(entrypoint)
    head = _run_git(entrypoint, "rev-parse", "HEAD")
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=root,
            entrypoint=entrypoint,
            merge_validation_worktree=True,
        )
    )
    staging = await orchestrator._ensure_validation_worktree(  # pyright: ignore[reportPrivateUsage]
        head=head
    )
    sentinel = root / "staging.provisioned"
    assert sentinel.is_file()

    await orchestrator.runtime.worktree_creation_lock.acquire()
    purge_task = asyncio.create_task(
        orchestrator._purge_staging_after_crash(  # pyright: ignore[reportPrivateUsage]
            staging,
            head,
            signal.SIGSEGV,
        )
    )
    try:
        await asyncio.sleep(0)
        assert not purge_task.done()
        assert not sentinel.exists()
    finally:
        purge_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await purge_task
        orchestrator.runtime.worktree_creation_lock.release()


async def test_cancel_after_successful_staging_add_removes_owned_worktree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancellation after git add returns cannot leak its registered worktree."""

    _require_git_and_sh()
    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    staging = root / "staging"
    _initialize_git_repo(entrypoint)
    head = _run_git(entrypoint, "rev-parse", "HEAD")
    original_add = _orchestrator_module._add_detached_worktree  # pyright: ignore[reportPrivateUsage]
    loop = asyncio.get_running_loop()
    provisioning_task: asyncio.Task[Path]

    def _successful_add_then_cancel(repo: Path, *, path: Path, head: str) -> None:
        original_add(repo, path=path, head=head)
        loop.call_soon_threadsafe(provisioning_task.cancel)

    monkeypatch.setattr(
        _orchestrator_module,
        "_add_detached_worktree",
        _successful_add_then_cancel,
    )
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=root,
            entrypoint=entrypoint,
            merge_validation_worktree=True,
        )
    )

    provisioning_task = asyncio.create_task(
        orchestrator._ensure_validation_worktree(  # pyright: ignore[reportPrivateUsage]
            head=head
        )
    )
    with pytest.raises(asyncio.CancelledError):
        await provisioning_task

    assert not staging.exists()
    assert str(staging) not in _run_git(entrypoint, "worktree", "list", "--porcelain")
    assert not orchestrator._validation_worktree_ready  # pyright: ignore[reportPrivateUsage]
    assert not (root / "staging.provisioned").exists()


async def test_failed_staging_add_does_not_remove_competing_registered_worktree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed add must not clean a worktree concurrently created at its path."""

    _require_git_and_sh()
    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    staging = root / "staging"
    _initialize_git_repo(entrypoint)
    head = _run_git(entrypoint, "rev-parse", "HEAD")
    original_add = _orchestrator_module._add_detached_worktree  # pyright: ignore[reportPrivateUsage]

    def _race_with_competing_add(repo: Path, *, path: Path, head: str) -> None:
        _run_git(repo, "worktree", "add", "--detach", str(path), head)
        (path / "uncommitted-by-competitor.txt").write_text(
            "must survive\n",
            encoding="utf-8",
        )
        original_add(repo, path=path, head=head)

    monkeypatch.setattr(
        _orchestrator_module,
        "_add_detached_worktree",
        _race_with_competing_add,
    )
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=root,
            entrypoint=entrypoint,
            merge_validation_worktree=True,
        )
    )

    with pytest.raises(subprocess.CalledProcessError):
        await orchestrator._ensure_validation_worktree(  # pyright: ignore[reportPrivateUsage]
            head=head
        )

    assert (staging / "uncommitted-by-competitor.txt").read_text(encoding="utf-8") == (
        "must survive\n"
    )
    assert str(staging) in _run_git(entrypoint, "worktree", "list", "--porcelain")
    assert not orchestrator._validation_worktree_ready  # pyright: ignore[reportPrivateUsage]
    assert not (root / "staging.provisioned").exists()


async def test_failed_worktree_cleanup_quarantines_unregistered_directory(
    tmp_path: Path,
) -> None:
    """Failed-add cleanup preserves an unregistered partial tree for inspection."""

    _require_git_and_sh()
    entrypoint = tmp_path / "entrypoint"
    staging = tmp_path / "orch" / "staging"
    _initialize_git_repo(entrypoint)
    staging.mkdir(parents=True)
    (staging / "junk.txt").write_text("partial add\n", encoding="utf-8")

    await asyncio.to_thread(
        _orchestrator_module._cleanup_failed_worktree_creation,  # pyright: ignore[reportPrivateUsage]
        entrypoint,
        staging,
        worktree_id="staging",
    )

    quarantined = list(staging.parent.glob("staging.invalid-*"))
    assert not staging.exists()
    assert len(quarantined) == 1
    assert (quarantined[0] / "junk.txt").read_text(encoding="utf-8") == "partial add\n"


async def test_failed_worktree_cleanup_unlocks_before_removing(tmp_path: Path) -> None:
    """Best-effort failed-provision cleanup also removes a locked staging tree."""

    _require_git_and_sh()
    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    _initialize_git_repo(entrypoint)
    head = _run_git(entrypoint, "rev-parse", "HEAD")
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=root,
            entrypoint=entrypoint,
            merge_validation_worktree=True,
        ),
        task_manager=_task_manager_with_tasks(),
    )
    staging = await orchestrator._ensure_validation_worktree(  # pyright: ignore[reportPrivateUsage]
        head=head
    )
    _run_git(entrypoint, "worktree", "lock", str(staging))

    await asyncio.to_thread(
        _orchestrator_module._cleanup_failed_worktree_creation,  # pyright: ignore[reportPrivateUsage]
        entrypoint,
        staging,
        worktree_id="staging",
    )

    assert not staging.exists()


async def test_creation_blocks_on_entrypoint_lock_not_merge_lock(tmp_path: Path) -> None:
    """The throughput fix: under staging, creation waits for the brief publish
    (entrypoint_lock), and is NOT blocked by a held merge_lock (under which the
    slow validation build runs)."""

    if shutil.which("git") is None:
        pytest.skip("git executable is not available")

    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    ready = Task(id="task-1", title="Ready", summary="Ready", description="Ready to run.")
    _initialize_git_repo(entrypoint)
    write_task(task_directory(entrypoint) / "001-ready.yaml", ready)
    _run_git(entrypoint, "add", "tasks")
    _run_git(entrypoint, "commit", "-m", "seed")
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=root, entrypoint=entrypoint, merge_validation_worktree=True
        ),
        task_manager=_task_manager_with_tasks(ready),
    )

    # Holding merge_lock (where a staging validation build would run) must NOT
    # block worktree creation.
    await orchestrator.runtime.merge_lock.acquire()
    try:
        worktree = await asyncio.wait_for(
            orchestrator.ensure_worktree_for_ready_task_id_for_test(ready.id),
            timeout=2.0,
        )
        assert worktree is not None
    finally:
        orchestrator.runtime.merge_lock.release()

    # Holding entrypoint_lock (the publish window) DOES block creation.
    other = Task(id="task-2", title="Other", summary="Other", description="Other.")
    write_task(task_directory(entrypoint) / "002-other.yaml", other)
    _run_git(entrypoint, "add", "tasks")
    _run_git(entrypoint, "commit", "-m", "second")
    await orchestrator.sync_task_manager_once_for_test()

    await orchestrator.runtime.entrypoint_lock.acquire()
    ensure_task = asyncio.create_task(
        orchestrator.ensure_worktree_for_ready_task_id_for_test(other.id)
    )
    try:
        await asyncio.sleep(0.05)
        assert not ensure_task.done()
    finally:
        orchestrator.runtime.entrypoint_lock.release()
    created = await asyncio.wait_for(ensure_task, timeout=2.0)
    assert created is not None


async def test_staging_merge_prep_git_failure_returns_to_pending(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A git failure while preparing the merge must not tear the run down.

    The committed-worktree reads + publish-target read run before the trial merge.
    If one raises (e.g. a broken branch ref), the worktree returns to PENDING —
    matching the legacy in-entrypoint path — instead of the exception escaping the
    merge service and cancelling the whole orchestrator. Staging creation itself
    stays unguarded (a setup/disk failure there is allowed to stop the run).
    """

    _require_git_and_sh()

    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    _initialize_git_repo(entrypoint)
    seed = _seed_tasks_dir(entrypoint)
    original_head = _run_git(entrypoint, "rev-parse", "HEAD")
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=root,
            entrypoint=entrypoint,
            merge_validation_worktree=True,
        ),
        task_manager=_task_manager_with_tasks(seed),
    )
    worktree = await orchestrator.create_fresh_worktree_for_test(name="prep-fails", task=seed)
    write_task(task_directory(worktree.path) / "002-follow-up.yaml", _follow_up(seed))
    _commit_worktree(worktree.path)
    await orchestrator.transition_worktree_for_test(worktree.worktree_id, WorktreeState.MERGE)

    def _raise_branch_head(*_args: object, **_kwargs: object) -> str:
        raise subprocess.CalledProcessError(1, ["git", "rev-parse"], output="", stderr="boom")

    monkeypatch.setattr(_orchestrator_module, "_branch_head", _raise_branch_head)

    # Must return normally (not raise) — the merge service survives the failure.
    await orchestrator.process_merge_queue_once_for_test()

    assert orchestrator.worktrees_by_id[worktree.worktree_id].state is WorktreeState.PENDING
    # The entrypoint was never touched.
    assert _run_git(entrypoint, "rev-parse", "HEAD") == original_head
    assert _run_git(entrypoint, "status", "--porcelain") == ""


async def test_staging_setup_uses_target_branch_tip_when_entrypoint_head_is_detached(
    tmp_path: Path,
) -> None:
    """Initial staging setup must observe the branch tip, not detached HEAD."""

    _require_git_and_sh()

    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    _initialize_git_repo(entrypoint)
    seed = _seed_tasks_dir(entrypoint)
    worktree_base = _run_git(entrypoint, "rev-parse", "HEAD")
    (entrypoint / "main-tip.txt").write_text("main tip\n", encoding="utf-8")
    _run_git(entrypoint, "add", "main-tip.txt")
    _run_git(entrypoint, "commit", "-m", "advance main with setup marker")
    _run_git(entrypoint, "checkout", "--detach", worktree_base)

    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=root,
            entrypoint=entrypoint,
            merge_validation_worktree=True,
            worktree_setup_command=AsyncOrchestratorWorktreeSetupCommandConfig(
                argv=(
                    "sh",
                    "-c",
                    'if [ "$(basename "$1")" = "staging" ]; then test -f "$1/main-tip.txt"; fi',
                    "setup",
                    "{worktree}",
                ),
            ),
        ),
        task_manager=_task_manager_with_tasks(seed),
    )
    worktree = await orchestrator.create_fresh_worktree_for_test(
        name="setup-target-tip",
        task=seed,
    )
    write_task(task_directory(worktree.path) / "002-follow-up.yaml", _follow_up(seed))
    _commit_worktree(worktree.path)
    await orchestrator.transition_worktree_for_test(worktree.worktree_id, WorktreeState.MERGE)

    await orchestrator.process_merge_queue_once_for_test()

    assert orchestrator.worktrees_by_id[worktree.worktree_id].state is WorktreeState.CLOSED
    assert (root / "staging" / "main-tip.txt").read_text(encoding="utf-8") == "main tip\n"


async def test_staging_publish_targets_branch_not_detached_head(tmp_path: Path) -> None:
    """Publish fast-forwards ``merge_target_branch`` even if ``HEAD`` is detached.

    The trial merge is built on the ``merge_target_branch`` tip (not the
    entrypoint's ``HEAD``), so ``git merge --ff-only`` stays valid when the
    entrypoint sits on a detached/older ``HEAD``. With the old ``HEAD``-based
    read the trial merge would branch off the older commit and the ff-only
    publish would fail, bouncing the worktree to PENDING — so reaching CLOSED
    here is the discriminating assertion.
    """

    _require_git_and_sh()

    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    _initialize_git_repo(entrypoint)
    seed = _seed_tasks_dir(entrypoint)
    # Advance ``main`` one commit past the worktree base, then detach ``HEAD`` to
    # the earlier commit: now ``HEAD`` (seed) != ``main`` tip (ahead-commit).
    worktree_base = _run_git(entrypoint, "rev-parse", "HEAD")
    (entrypoint / "README.md").write_text("advanced\n", encoding="utf-8")
    _run_git(entrypoint, "add", "README.md")
    _run_git(entrypoint, "commit", "-m", "advance main past the worktree base")
    main_tip = _run_git(entrypoint, "rev-parse", "HEAD")
    _run_git(entrypoint, "checkout", "--detach", worktree_base)

    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=root,
            entrypoint=entrypoint,
            merge_validation_worktree=True,
        ),
        task_manager=_task_manager_with_tasks(seed),
    )
    worktree = await orchestrator.create_fresh_worktree_for_test(name="tend-target", task=seed)
    write_task(task_directory(worktree.path) / "002-follow-up.yaml", _follow_up(seed))
    _commit_worktree(worktree.path)
    await orchestrator.transition_worktree_for_test(worktree.worktree_id, WorktreeState.MERGE)

    await orchestrator.process_merge_queue_once_for_test()

    assert orchestrator.worktrees_by_id[worktree.worktree_id].state is WorktreeState.CLOSED
    # ``main`` advanced past its prior tip and carries both the follow-up task and
    # the earlier ``advance`` commit (the trial merge built on the branch tip).
    assert _run_git(entrypoint, "rev-parse", "main") != main_tip
    assert (task_directory(entrypoint) / "002-follow-up.yaml").exists()


async def test_seed_worktree_build_snapshots_staging_and_seeds_new_worktree(
    tmp_path: Path,
) -> None:
    """With ``seed_worktree_build`` on, a validated merge snapshots the staging
    ``.lake/build`` into ``<root>/.build-cache`` and a newly-created task worktree
    is seeded from it (so its first build is incremental, not from scratch)."""

    _require_git_and_sh()

    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    _initialize_git_repo(entrypoint)
    seed = _seed_tasks_dir(entrypoint)
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=root,
            entrypoint=entrypoint,
            merge_validation_worktree=True,
            seed_worktree_build=True,
            # seed_worktree_build requires the mirror (it supplies the packages
            # the seeded .lake/build references).
            workspace_mirror=AsyncOrchestratorWorkspaceMirrorConfig(enabled=True),
            # Simulate a Lean build writing an olean into the staging .lake/build.
            pre_merge_validation_commands=(
                _validation_command(
                    "mkdir -p .lake/build/lib && printf x > .lake/build/lib/Seed.olean"
                ),
            ),
        ),
        task_manager=_task_manager_with_tasks(seed),
    )
    worktree = await orchestrator.create_fresh_worktree_for_test(name="adds-task", task=seed)
    write_task(task_directory(worktree.path) / "002-follow-up.yaml", _follow_up(seed))
    _commit_worktree(worktree.path)
    await orchestrator.transition_worktree_for_test(worktree.worktree_id, WorktreeState.MERGE)

    await orchestrator.process_merge_queue_once_for_test()

    # The staging build was snapshotted into the cache after the validated merge.
    cached = root / ".build-cache" / "build" / "lib" / "Seed.olean"
    assert cached.is_file()
    assert cached.read_text() == "x"
    # No swap leftovers.
    assert not (root / ".build-cache" / "build.incoming").exists()

    # A newly-created task worktree is seeded from the cache.
    fresh = await orchestrator.create_fresh_worktree_for_test(name="seeded", task=seed)
    seeded = fresh.path / ".lake" / "build" / "lib" / "Seed.olean"
    assert seeded.is_file()
    assert seeded.read_text() == "x"


async def test_seed_worktree_build_disabled_creates_no_cache(tmp_path: Path) -> None:
    """Default (``seed_worktree_build`` off): no cache, worktrees not seeded."""

    _require_git_and_sh()

    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    _initialize_git_repo(entrypoint)
    seed = _seed_tasks_dir(entrypoint)
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=root,
            entrypoint=entrypoint,
            merge_validation_worktree=True,
            seed_worktree_build=False,
            pre_merge_validation_commands=(
                _validation_command(
                    "mkdir -p .lake/build/lib && printf x > .lake/build/lib/Seed.olean"
                ),
            ),
        ),
        task_manager=_task_manager_with_tasks(seed),
    )
    worktree = await orchestrator.create_fresh_worktree_for_test(name="adds-task", task=seed)
    write_task(task_directory(worktree.path) / "002-follow-up.yaml", _follow_up(seed))
    _commit_worktree(worktree.path)
    await orchestrator.transition_worktree_for_test(worktree.worktree_id, WorktreeState.MERGE)
    await orchestrator.process_merge_queue_once_for_test()

    assert not (root / ".build-cache").exists()
    fresh = await orchestrator.create_fresh_worktree_for_test(name="not-seeded", task=seed)
    assert not (fresh.path / ".lake" / "build").exists()


def test_replace_dir_copy_roundtrip(tmp_path: Path) -> None:
    """The reflink-or-copy helper replaces snapshots atomically."""

    src = tmp_path / "src"
    (src / "lib").mkdir(parents=True)
    (src / "lib" / "A.olean").write_text("a")
    (src / "lib" / "B.olean").write_text("b")
    cache = tmp_path / "cache" / "build"
    cache.mkdir(parents=True)
    (cache / "stale.txt").write_text("old")
    _orchestrator_module._replace_dir_copy(src, cache)  # pyright: ignore[reportPrivateUsage]
    assert (cache / "lib" / "A.olean").read_text() == "a"
    assert (cache / "lib" / "B.olean").read_text() == "b"
    assert not (cache / "stale.txt").exists()
    assert not (cache.parent / "build.incoming").exists()
    assert not (cache.parent / "build.old").exists()


def test_seed_build_dir_copy_replaces_stale_destination(tmp_path: Path) -> None:
    """Seeding must replace the destination build tree, not overlay it."""

    src = tmp_path / "src"
    (src / "lib").mkdir(parents=True)
    (src / "lib" / "Fresh.olean").write_text("fresh")
    destination = tmp_path / "worktree" / ".lake" / "build"
    (destination / "lib").mkdir(parents=True)
    (destination / "lib" / "Stale.olean").write_text("stale")

    _orchestrator_module._seed_build_dir_copy(src, destination)  # pyright: ignore[reportPrivateUsage]

    assert (destination / "lib" / "Fresh.olean").read_text() == "fresh"
    assert not (destination / "lib" / "Stale.olean").exists()


@pytest.mark.parametrize("symlink_path", [".lake", ".lake/build"])
async def test_seed_worktree_build_skips_symlinked_lake_mirror_paths(
    tmp_path: Path,
    symlink_path: str,
) -> None:
    """Seeding must not write through symlinked ``.lake`` or ``.lake/build``."""

    _require_git_and_sh()

    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    _initialize_git_repo(entrypoint)
    seed = _seed_tasks_dir(entrypoint)
    # Ensure the configured symlink source exists in the entrypoint workspace.
    (entrypoint / ".lake" / "build").mkdir(parents=True)
    (root / ".build-cache" / "build" / "lib").mkdir(parents=True)
    (root / ".build-cache" / "build" / "lib" / "Fresh.olean").write_text("fresh")
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=root,
            entrypoint=entrypoint,
            merge_validation_worktree=True,
            seed_worktree_build=True,
            workspace_mirror=AsyncOrchestratorWorkspaceMirrorConfig(
                enabled=True,
                symlink_paths=[symlink_path],
            ),
        ),
        task_manager=_task_manager_with_tasks(seed),
    )

    fresh = await orchestrator.create_fresh_worktree_for_test(
        name=f"seeded-{symlink_path.replace('/', '-')}",
        task=seed,
    )

    assert (fresh.path / symlink_path).is_symlink()
    assert not (entrypoint / ".lake" / "build" / "lib" / "Fresh.olean").exists()


def test_seed_worktree_build_requires_staging_and_mirror(tmp_path: Path) -> None:
    """``seed_worktree_build`` is rejected without its prerequisites."""

    root = tmp_path / "r"
    entrypoint = tmp_path / "e"
    enabled_mirror = AsyncOrchestratorWorkspaceMirrorConfig(enabled=True)

    # Requires the staging validation worktree.
    with pytest.raises(ValidationError, match="merge_validation_worktree"):
        AsyncOrchestratorConfig(
            root=root,
            entrypoint=entrypoint,
            seed_worktree_build=True,
            merge_validation_worktree=False,
            workspace_mirror=enabled_mirror,
        )

    # Requires the workspace mirror (which supplies the packages).
    with pytest.raises(ValidationError, match="workspace_mirror"):
        AsyncOrchestratorConfig(
            root=root,
            entrypoint=entrypoint,
            seed_worktree_build=True,
            merge_validation_worktree=True,
            workspace_mirror=AsyncOrchestratorWorkspaceMirrorConfig(enabled=False),
        )

    # Both prerequisites present: accepted.
    config = AsyncOrchestratorConfig(
        root=root,
        entrypoint=entrypoint,
        seed_worktree_build=True,
        merge_validation_worktree=True,
        workspace_mirror=enabled_mirror,
    )
    assert config.seed_worktree_build is True


def _named_task(seed: Task, name: str) -> Task:
    return Task(
        id=f"task-{name}",
        title=name,
        summary=f"{name} task",
        description=f"{name} task.",
        depends_on=[seed.id],
    )


async def _ready_worktree_adding(
    orchestrator: WorktreeTestingOrchestrator,
    seed: Task,
    *,
    name: str,
    task_file: str | None = None,
    task: Task | None = None,
    plain_file: str | None = None,
    raw_task_file: tuple[str, str] | None = None,
    delete_task_file: str | None = None,
) -> AsyncOrchestratorWorktree:
    """Create a worktree, write (or delete) a file in it, and transition it to MERGE."""

    wt = await orchestrator.create_fresh_worktree_for_test(name=name, task=seed)
    if task_file is not None and task is not None:
        write_task(task_directory(wt.path) / task_file, task)
    if raw_task_file is not None:
        (task_directory(wt.path) / raw_task_file[0]).write_text(raw_task_file[1])
    if plain_file is not None:
        (wt.path / plain_file).write_text("x")
    if delete_task_file is not None:
        (task_directory(wt.path) / delete_task_file).unlink()
    _commit_worktree(wt.path)
    await orchestrator.transition_worktree_for_test(wt.worktree_id, WorktreeState.MERGE)
    return wt


async def _committed_plain_file_worktree(
    orchestrator: WorktreeTestingOrchestrator,
    seed: Task,
    *,
    name: str,
    filename: str,
) -> AsyncOrchestratorWorktree:
    wt = await orchestrator.create_fresh_worktree_for_test(name=name, task=seed)
    (wt.path / filename).write_text(f"{filename}\n", encoding="utf-8")
    _commit_worktree(wt.path)
    return wt


async def test_batched_merge_publishes_all_good_in_one_round(tmp_path: Path) -> None:
    """Several ready worktrees are validated together and all published."""

    _require_git_and_sh()
    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    _initialize_git_repo(entrypoint)
    seed = _seed_tasks_dir(entrypoint)
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=root,
            entrypoint=entrypoint,
            merge_validation_worktree=True,
            batched_merge=True,
            # passes as long as no worktree dropped a `BAD` marker
            pre_merge_validation_commands=(_validation_command("test ! -e BAD"),),
        ),
        task_manager=_task_manager_with_tasks(seed),
    )
    a = await _ready_worktree_adding(
        orchestrator, seed, name="a", task_file="002-a.yaml", task=_named_task(seed, "a")
    )
    b = await _ready_worktree_adding(
        orchestrator, seed, name="b", task_file="003-b.yaml", task=_named_task(seed, "b")
    )

    await orchestrator.process_merge_queue_once_for_test()

    # Both landed in one round.
    assert orchestrator.worktrees_by_id[a.worktree_id].state is WorktreeState.CLOSED
    assert orchestrator.worktrees_by_id[b.worktree_id].state is WorktreeState.CLOSED
    assert (task_directory(entrypoint) / "002-a.yaml").exists()
    assert (task_directory(entrypoint) / "003-b.yaml").exists()
    # One staging build for the batch: exactly one merge commit advanced main.
    assert _run_git(entrypoint, "status", "--porcelain") == ""


async def test_batched_merge_without_cap_drains_all_visible_worktrees(
    tmp_path: Path,
) -> None:
    """The default batch size is uncapped, preserving the drain-all behavior."""

    _require_git_and_sh()
    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    counter = tmp_path / "build-batches.txt"
    _initialize_git_repo(entrypoint)
    seed = _seed_tasks_dir(entrypoint)
    record_present_files = (
        "for f in A B C; do "
        "if [ -e \"$f\" ]; then printf '%s' \"$f\" >> \"$1\"; fi; "
        "done; printf '\\n' >> \"$1\""
    )
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=root,
            entrypoint=entrypoint,
            merge_validation_worktree=True,
            batched_merge=True,
            pre_merge_validation_commands=(
                _validation_command(record_present_files, str(counter)),
            ),
        ),
        task_manager=_task_manager_with_tasks(seed),
    )
    a = await _committed_plain_file_worktree(orchestrator, seed, name="a", filename="A")
    b = await _committed_plain_file_worktree(orchestrator, seed, name="b", filename="B")
    c = await _committed_plain_file_worktree(orchestrator, seed, name="c", filename="C")
    await orchestrator.transition_worktree_for_test(a.worktree_id, WorktreeState.MERGE)
    await orchestrator.transition_worktree_for_test(b.worktree_id, WorktreeState.MERGE)
    await orchestrator.transition_worktree_for_test(c.worktree_id, WorktreeState.MERGE)

    await orchestrator.process_merge_queue_once_for_test()

    assert counter.read_text(encoding="utf-8") == "ABC\n"
    assert orchestrator.worktrees_by_id[a.worktree_id].state is WorktreeState.CLOSED
    assert orchestrator.worktrees_by_id[b.worktree_id].state is WorktreeState.CLOSED
    assert orchestrator.worktrees_by_id[c.worktree_id].state is WorktreeState.CLOSED


async def test_batched_merge_respects_max_batch_size_in_fifo_order(tmp_path: Path) -> None:
    """A configured cap limits each validation batch and leaves later work queued."""

    _require_git_and_sh()
    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    counter = tmp_path / "build-batches.txt"
    _initialize_git_repo(entrypoint)
    seed = _seed_tasks_dir(entrypoint)
    record_present_files = (
        "for f in A B C; do "
        "if [ -e \"$f\" ]; then printf '%s' \"$f\" >> \"$1\"; fi; "
        "done; printf '\\n' >> \"$1\""
    )
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=root,
            entrypoint=entrypoint,
            merge_validation_worktree=True,
            batched_merge=True,
            max_merge_batch_size=2,
            pre_merge_validation_commands=(
                _validation_command(record_present_files, str(counter)),
            ),
        ),
        task_manager=_task_manager_with_tasks(seed),
    )
    a = await _committed_plain_file_worktree(orchestrator, seed, name="a", filename="A")
    b = await _committed_plain_file_worktree(orchestrator, seed, name="b", filename="B")
    c = await _committed_plain_file_worktree(orchestrator, seed, name="c", filename="C")
    # Enqueue C before B to prove the cap follows runtime FIFO order rather than
    # worktree creation/durable insertion order.
    await orchestrator.transition_worktree_for_test(a.worktree_id, WorktreeState.MERGE)
    await orchestrator.transition_worktree_for_test(c.worktree_id, WorktreeState.MERGE)
    await orchestrator.transition_worktree_for_test(b.worktree_id, WorktreeState.MERGE)

    await orchestrator.process_merge_queue_once_for_test()

    assert counter.read_text(encoding="utf-8") == "AC\nABC\n"
    assert orchestrator.worktrees_by_id[a.worktree_id].state is WorktreeState.CLOSED
    assert orchestrator.worktrees_by_id[b.worktree_id].state is WorktreeState.CLOSED
    assert orchestrator.worktrees_by_id[c.worktree_id].state is WorktreeState.CLOSED


async def test_batched_merge_uses_merge_queue_fifo_order_for_conflicts(
    tmp_path: Path,
) -> None:
    """Batch assembly preserves the merge queue order, not worktree creation order."""

    _require_git_and_sh()
    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    _initialize_git_repo(entrypoint)
    seed = _seed_tasks_dir(entrypoint)
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=root,
            entrypoint=entrypoint,
            merge_validation_worktree=True,
            batched_merge=True,
        ),
        task_manager=_task_manager_with_tasks(seed),
    )

    # Create A before B, but enqueue B for merge first. If batching drains durable
    # state insertion order, A lands and B conflicts; FIFO queue order lands B and
    # bounces A, matching the serial merge path.
    a = await orchestrator.create_fresh_worktree_for_test(name="a-conflict", task=seed)
    (a.path / "conflict.txt").write_text("a\n", encoding="utf-8")
    _commit_worktree(a.path)
    b = await orchestrator.create_fresh_worktree_for_test(name="b-conflict", task=seed)
    (b.path / "conflict.txt").write_text("b\n", encoding="utf-8")
    _commit_worktree(b.path)
    await orchestrator.transition_worktree_for_test(b.worktree_id, WorktreeState.MERGE)
    await orchestrator.transition_worktree_for_test(a.worktree_id, WorktreeState.MERGE)

    await orchestrator.process_merge_queue_once_for_test()

    assert orchestrator.worktrees_by_id[b.worktree_id].state is WorktreeState.CLOSED
    assert orchestrator.worktrees_by_id[a.worktree_id].state is WorktreeState.PENDING
    assert (entrypoint / "conflict.txt").read_text(encoding="utf-8") == "b\n"


async def test_batched_merge_non_utf8_filename_conflict_bounces_with_escaped_message(
    tmp_path: Path,
) -> None:
    """A surrogateescaped Git conflict path remains persistable when bounced."""

    _require_git_and_sh()
    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    _initialize_git_repo(entrypoint)
    seed = _seed_tasks_dir(entrypoint)
    filename = os.fsdecode(b"bad-\xff.txt")
    (entrypoint / filename).write_text("base\n", encoding="utf-8")
    _commit_worktree(entrypoint)
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=root,
            entrypoint=entrypoint,
            merge_validation_worktree=True,
            batched_merge=True,
        ),
        task_manager=_task_manager_with_tasks(seed),
    )

    first = await orchestrator.create_fresh_worktree_for_test(
        name="first-raw-conflict", task=seed
    )
    (first.path / filename).write_text("first\n", encoding="utf-8")
    _commit_worktree(first.path)
    second = await orchestrator.create_fresh_worktree_for_test(
        name="second-raw-conflict", task=seed
    )
    (second.path / filename).write_text("second\n", encoding="utf-8")
    _commit_worktree(second.path)
    await orchestrator.transition_worktree_for_test(first.worktree_id, WorktreeState.MERGE)
    await orchestrator.transition_worktree_for_test(second.worktree_id, WorktreeState.MERGE)

    await orchestrator.process_merge_queue_once_for_test()

    assert orchestrator.worktrees_by_id[first.worktree_id].state is WorktreeState.CLOSED
    conflicting = orchestrator.worktrees_by_id[second.worktree_id]
    assert conflicting.state is WorktreeState.PENDING
    message = conflicting.discussion[-1].message
    message.encode("utf-8")
    assert "bad-\\udcff.txt" in message
    assert "bad-\\\\udcff.txt" not in message
    persisted = (second.path / ".tend" / "discussion.md").read_text(encoding="utf-8")
    assert "bad-\\udcff.txt" in persisted
    assert (entrypoint / filename).read_text(encoding="utf-8") == "first\n"


async def test_batched_merge_bisects_to_isolate_culprit(tmp_path: Path) -> None:
    """A failing batch bisects: innocent worktrees land, the culprit bounces."""

    _require_git_and_sh()
    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    _initialize_git_repo(entrypoint)
    seed = _seed_tasks_dir(entrypoint)
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=root,
            entrypoint=entrypoint,
            merge_validation_worktree=True,
            batched_merge=True,
            pre_merge_validation_commands=(_validation_command("test ! -e BAD"),),
        ),
        task_manager=_task_manager_with_tasks(seed),
    )
    good1 = await _ready_worktree_adding(
        orchestrator, seed, name="good1", task_file="002-g1.yaml", task=_named_task(seed, "g1")
    )
    bad = await _ready_worktree_adding(orchestrator, seed, name="bad", plain_file="BAD")
    good2 = await _ready_worktree_adding(
        orchestrator, seed, name="good2", task_file="003-g2.yaml", task=_named_task(seed, "g2")
    )

    await orchestrator.process_merge_queue_once_for_test()

    # Innocent worktrees published; the culprit bounced back to PENDING.
    assert orchestrator.worktrees_by_id[good1.worktree_id].state is WorktreeState.CLOSED
    assert orchestrator.worktrees_by_id[good2.worktree_id].state is WorktreeState.CLOSED
    assert orchestrator.worktrees_by_id[bad.worktree_id].state is WorktreeState.PENDING
    assert (task_directory(entrypoint) / "002-g1.yaml").exists()
    assert (task_directory(entrypoint) / "003-g2.yaml").exists()
    assert not (entrypoint / "BAD").exists()


async def test_malformed_task_file_isolated_without_building(tmp_path: Path) -> None:
    """A worktree with a malformed task file is isolated by the build-free task
    gate and bounced; the good worktrees build once and land.

    The task gate (gate 1) attributes the parse failure to the offending file's
    worktree, which is probed alone, fails the build-free gate on its own
    contribution, and bounces before the expensive build gate ever runs for it —
    so the build runs a *single* time, on the survivors only, rather than via
    build-bearing bisection. ``build-count.txt`` (outside staging, so it
    survives the staging reset) records one run.
    """

    _require_git_and_sh()
    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    counter = tmp_path / "build-count.txt"
    _initialize_git_repo(entrypoint)
    seed = _seed_tasks_dir(entrypoint)
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=root,
            entrypoint=entrypoint,
            merge_validation_worktree=True,
            batched_merge=True,
            pre_merge_validation_commands=(_validation_command(f"printf x >> {counter}"),),
        ),
        task_manager=_task_manager_with_tasks(seed),
    )
    good1 = await _ready_worktree_adding(
        orchestrator, seed, name="g1", task_file="002-g1.yaml", task=_named_task(seed, "g1")
    )
    # Unparseable YAML (unclosed flow sequence) -> task-tree parse failure.
    bad = await _ready_worktree_adding(
        orchestrator, seed, name="bad", raw_task_file=("002-bad.yaml", "[unclosed\n")
    )
    good2 = await _ready_worktree_adding(
        orchestrator, seed, name="g2", task_file="003-g2.yaml", task=_named_task(seed, "g2")
    )

    await orchestrator.process_merge_queue_once_for_test()

    assert orchestrator.worktrees_by_id[good1.worktree_id].state is WorktreeState.CLOSED
    assert orchestrator.worktrees_by_id[good2.worktree_id].state is WorktreeState.CLOSED
    assert orchestrator.worktrees_by_id[bad.worktree_id].state is WorktreeState.PENDING
    assert not (task_directory(entrypoint) / "002-bad.yaml").exists()
    # Build-free isolation: exactly one build ran (the two survivors together).
    assert counter.read_text() == "x"


async def test_build_failure_attributed_then_remainder_revalidated(tmp_path: Path) -> None:
    """A build failure naming a worktree's file probes that worktree alone; it
    fails its own confirming build, bounces, and the remainder RE-BUILDS before
    merging.

    Invariant guards: a member is bounced only after failing a validation of
    exactly its own contribution (verify-before-bounce), and survivors are
    never merged "by elimination" — the remaining set must pass *its own* fresh
    build. The failing batch + the culprit's confirming solo build + the
    survivors' passing re-validation = exactly three builds; a direct bounce of
    the attributed member or a merge-by-elimination shortcut would record
    fewer and fail this test.
    """

    _require_git_and_sh()
    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    counter = tmp_path / "build-count.txt"
    _initialize_git_repo(entrypoint)
    seed = _seed_tasks_dir(entrypoint)
    # Count every build; fail (emitting a Lean-style error for Bad.lean) iff present.
    script = (
        f"printf x >> {counter}; "
        "if [ -e Bad.lean ]; then echo 'Bad.lean:1:1: error: boom'; exit 1; fi"
    )
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=root,
            entrypoint=entrypoint,
            merge_validation_worktree=True,
            batched_merge=True,
            pre_merge_validation_commands=(_validation_command(script),),
        ),
        task_manager=_task_manager_with_tasks(seed),
    )
    good1 = await _ready_worktree_adding(
        orchestrator, seed, name="g1", task_file="002-g1.yaml", task=_named_task(seed, "g1")
    )
    bad = await _ready_worktree_adding(orchestrator, seed, name="bad", plain_file="Bad.lean")
    good2 = await _ready_worktree_adding(
        orchestrator, seed, name="g2", task_file="003-g2.yaml", task=_named_task(seed, "g2")
    )

    await orchestrator.process_merge_queue_once_for_test()

    assert orchestrator.worktrees_by_id[good1.worktree_id].state is WorktreeState.CLOSED
    assert orchestrator.worktrees_by_id[good2.worktree_id].state is WorktreeState.CLOSED
    assert orchestrator.worktrees_by_id[bad.worktree_id].state is WorktreeState.PENDING
    assert not (entrypoint / "Bad.lean").exists()
    # Three builds: the failing full batch, the culprit's confirming solo build
    # (fails on its own contribution — verify-before-bounce), then the
    # survivors' passing re-validation.
    assert counter.read_text() == "xxx"


async def test_cycle_introduced_by_one_member_bounced_in_one_round(tmp_path: Path) -> None:
    """A member whose task files form a ``depends_on`` cycle is attributed,
    fails the build-free task gate alone, and bounces; the rest of the batch
    builds once and lands.

    The cycle error names every task id on the cycle path (issue #128), which
    the task gate maps to the declaring yaml files; both files belong to the
    culprit's diff, so isolation probes it alone (a build-free confirming
    validation) instead of build-per-round bisection. ``build-count.txt``
    proves it: exactly one build (the survivors') ran.
    """

    _require_git_and_sh()
    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    counter = tmp_path / "build-count.txt"
    _initialize_git_repo(entrypoint)
    seed = _seed_tasks_dir(entrypoint)
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=root,
            entrypoint=entrypoint,
            merge_validation_worktree=True,
            batched_merge=True,
            pre_merge_validation_commands=(_validation_command(f"printf x >> {counter}"),),
        ),
        task_manager=_task_manager_with_tasks(seed),
    )
    good1 = await _ready_worktree_adding(
        orchestrator, seed, name="g1", task_file="002-g1.yaml", task=_named_task(seed, "g1")
    )
    bad = await orchestrator.create_fresh_worktree_for_test(name="bad", task=seed)
    cyc_a = Task(
        id="task-cyc-a",
        title="Cyc A",
        summary="Cyc A",
        description="Cyc A.",
        depends_on=["task-cyc-b"],
    )
    cyc_b = Task(
        id="task-cyc-b",
        title="Cyc B",
        summary="Cyc B",
        description="Cyc B.",
        depends_on=["task-cyc-a"],
    )
    write_task(task_directory(bad.path) / "010-cyc-a.yaml", cyc_a)
    write_task(task_directory(bad.path) / "011-cyc-b.yaml", cyc_b)
    _commit_worktree(bad.path)
    await orchestrator.transition_worktree_for_test(bad.worktree_id, WorktreeState.MERGE)
    good2 = await _ready_worktree_adding(
        orchestrator, seed, name="g2", task_file="003-g2.yaml", task=_named_task(seed, "g2")
    )

    await orchestrator.process_merge_queue_once_for_test()

    assert orchestrator.worktrees_by_id[good1.worktree_id].state is WorktreeState.CLOSED
    assert orchestrator.worktrees_by_id[good2.worktree_id].state is WorktreeState.CLOSED
    updated_bad = orchestrator.worktrees_by_id[bad.worktree_id]
    assert updated_bad.state is WorktreeState.PENDING
    assert not (task_directory(entrypoint) / "010-cyc-a.yaml").exists()
    assert not (task_directory(entrypoint) / "011-cyc-b.yaml").exists()
    # Attribution, not bisection: exactly one build ran (the two survivors).
    assert counter.read_text() == "x"
    # The bounce message names the cycle and both offending files.
    message = updated_bad.discussion[-1].message
    assert "cycle" in message
    assert "010-cyc-a.yaml" in message
    assert "011-cyc-b.yaml" in message


async def test_complete_depends_on_open_attributed_despite_preexisting_file(
    tmp_path: Path,
) -> None:
    """Both ids of a complete-depends-on-open error map to files, one of which
    (the open dependency) pre-exists on base and is touched by nobody; the
    member that added the complete task is still the sole attributed member —
    it fails the build-free gate alone and bounces with no extra build.
    """

    _require_git_and_sh()
    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    counter = tmp_path / "build-count.txt"
    _initialize_git_repo(entrypoint)
    seed = _seed_tasks_dir(entrypoint)
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=root,
            entrypoint=entrypoint,
            merge_validation_worktree=True,
            batched_merge=True,
            pre_merge_validation_commands=(_validation_command(f"printf x >> {counter}"),),
        ),
        task_manager=_task_manager_with_tasks(seed),
    )
    good1 = await _ready_worktree_adding(
        orchestrator, seed, name="g1", task_file="002-g1.yaml", task=_named_task(seed, "g1")
    )
    # A complete task depending on the (open, pre-existing) seed task.
    invalid_complete = Task(
        id="task-done",
        title="Done",
        summary="Done",
        description="Invalidly complete.",
        status=TaskStatus.COMPLETE,
        depends_on=[seed.id],
    )
    bad = await _ready_worktree_adding(
        orchestrator, seed, name="bad", task_file="012-done.yaml", task=invalid_complete
    )
    good2 = await _ready_worktree_adding(
        orchestrator, seed, name="g2", task_file="003-g2.yaml", task=_named_task(seed, "g2")
    )

    await orchestrator.process_merge_queue_once_for_test()

    assert orchestrator.worktrees_by_id[good1.worktree_id].state is WorktreeState.CLOSED
    assert orchestrator.worktrees_by_id[good2.worktree_id].state is WorktreeState.CLOSED
    assert orchestrator.worktrees_by_id[bad.worktree_id].state is WorktreeState.PENDING
    assert not (task_directory(entrypoint) / "012-done.yaml").exists()
    # One-round attribution: a single build for the surviving pair.
    assert counter.read_text() == "x"


async def test_batched_merge_retries_cancelled_validation_instead_of_bisecting(
    tmp_path: Path,
) -> None:
    """A signal-terminated batch validation is retried in place, not bisected.

    A cancelled (signal-killed) build says nothing about batch validity. The
    first build self-terminates with SIGTERM — standing in for an external
    kill (operator, container shutdown, OOM killer) — and the retry passes:
    every member lands, nobody is bounced, and exactly two builds run (the
    cancelled one plus its retry; a bisection of the batch would run more and
    bounce a member). Guards the issue #132 regression where a `-15` exit was
    booked as a batch failure and sent a healthy 27-member batch into
    bisection.
    """

    _require_git_and_sh()
    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    counter = tmp_path / "build-count.txt"
    marker = tmp_path / "already-cancelled-once"
    _initialize_git_repo(entrypoint)
    seed = _seed_tasks_dir(entrypoint)
    script = 'printf x >> "$1"; if [ ! -e "$2" ]; then : > "$2"; kill -TERM $$; fi'
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=root,
            entrypoint=entrypoint,
            merge_validation_worktree=True,
            batched_merge=True,
            pre_merge_validation_commands=(
                _validation_command(script, str(counter), str(marker)),
            ),
        ),
        task_manager=_task_manager_with_tasks(seed),
    )
    a = await _ready_worktree_adding(
        orchestrator, seed, name="a", task_file="002-a.yaml", task=_named_task(seed, "a")
    )
    b = await _ready_worktree_adding(
        orchestrator, seed, name="b", task_file="003-b.yaml", task=_named_task(seed, "b")
    )

    await orchestrator.process_merge_queue_once_for_test()

    assert orchestrator.worktrees_by_id[a.worktree_id].state is WorktreeState.CLOSED
    assert orchestrator.worktrees_by_id[b.worktree_id].state is WorktreeState.CLOSED
    assert (task_directory(entrypoint) / "002-a.yaml").exists()
    assert (task_directory(entrypoint) / "003-b.yaml").exists()
    # Exactly two builds: the cancelled one and its passing retry.
    assert counter.read_text() == "xx"


async def test_batched_merge_cancellation_retry_budget_is_per_command(
    tmp_path: Path,
) -> None:
    """Independent cancellations of distinct commands each get their own retry.

    Command A is killed on sequence attempt 1 (then passes), command B on
    attempt 2 (then passes); attempt 3 passes end-to-end and the batch
    publishes. Each retry restarts from the first command (later commands may
    depend on earlier ones), so A runs three times and B twice. A single
    global retry budget would have booked B's kill as a batch failure and
    bounced a healthy batch.
    """

    _require_git_and_sh()
    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    counter_a = tmp_path / "build-count-a.txt"
    counter_b = tmp_path / "build-count-b.txt"
    marker_a = tmp_path / "a-already-cancelled-once"
    marker_b = tmp_path / "b-already-cancelled-once"
    _initialize_git_repo(entrypoint)
    seed = _seed_tasks_dir(entrypoint)
    script = 'printf x >> "$1"; if [ ! -e "$2" ]; then : > "$2"; kill -TERM $$; fi'
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=root,
            entrypoint=entrypoint,
            merge_validation_worktree=True,
            batched_merge=True,
            pre_merge_validation_commands=(
                _validation_command(script, str(counter_a), str(marker_a)),
                _validation_command(script, str(counter_b), str(marker_b)),
            ),
        ),
        task_manager=_task_manager_with_tasks(seed),
    )
    a = await _ready_worktree_adding(
        orchestrator, seed, name="a", task_file="002-a.yaml", task=_named_task(seed, "a")
    )
    b = await _ready_worktree_adding(
        orchestrator, seed, name="b", task_file="003-b.yaml", task=_named_task(seed, "b")
    )

    await orchestrator.process_merge_queue_once_for_test()

    assert orchestrator.worktrees_by_id[a.worktree_id].state is WorktreeState.CLOSED
    assert orchestrator.worktrees_by_id[b.worktree_id].state is WorktreeState.CLOSED
    assert (task_directory(entrypoint) / "002-a.yaml").exists()
    assert (task_directory(entrypoint) / "003-b.yaml").exists()
    # A: cancelled attempt + two passes; B: cancelled attempt + one pass.
    assert counter_a.read_text() == "xxx"
    assert counter_b.read_text() == "xx"


async def test_batched_merge_persistent_cancellation_falls_through_to_failure(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A command cancelled past its retry budget stops retrying, books a failure.

    Every build self-terminates with SIGTERM. The single command has a budget
    of one cancellation retry (two builds total — the retry guard, not an
    infinite loop), a warning names the cancellation, and the ordinary failure
    handling then bounces the member without publishing anything.
    """

    _require_git_and_sh()
    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    counter = tmp_path / "build-count.txt"
    _initialize_git_repo(entrypoint)
    seed = _seed_tasks_dir(entrypoint)
    original_head = _run_git(entrypoint, "rev-parse", "HEAD")
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=root,
            entrypoint=entrypoint,
            merge_validation_worktree=True,
            batched_merge=True,
            pre_merge_validation_commands=(
                _validation_command('printf x >> "$1"; kill -TERM $$', str(counter)),
            ),
        ),
        task_manager=_task_manager_with_tasks(seed),
    )
    wt = await _ready_worktree_adding(
        orchestrator,
        seed,
        name="always-killed",
        task_file="002-a.yaml",
        task=_named_task(seed, "a"),
    )

    caplog.set_level(logging.INFO, logger=_orchestrator_module.__name__)
    await orchestrator.process_merge_queue_once_for_test()

    # Retried once, then treated as an ordinary failure: bounced, nothing landed.
    assert counter.read_text() == "xx"
    assert orchestrator.worktrees_by_id[wt.worktree_id].state is WorktreeState.PENDING
    assert _run_git(entrypoint, "rev-parse", "HEAD") == original_head
    assert not (task_directory(entrypoint) / "002-a.yaml").exists()
    warning_messages = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING
    ]
    assert any(
        f"validation cancelled (signal {signal.SIGTERM})" in message
        and "retry budget exhausted" in message
        for message in warning_messages
    )


async def test_batched_merge_cancellation_retry_budget_is_shared_across_bisection(
    tmp_path: Path,
) -> None:
    """Bisection nodes share one episode-wide cancellation-retry budget.

    Every build self-terminates with SIGTERM, and the batch has two members so
    the booked cancellation failure bisects. The single command's one retry is
    spent at the top-level node (2 builds); each half then sees its budget
    already exhausted and books the cancellation as a failure after a single
    build (1 + 1). Four builds total — a per-node budget would run six and
    hand a persistent killer O(2N) extra kills to amplify — and nothing lands.
    """

    _require_git_and_sh()
    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    counter = tmp_path / "build-count.txt"
    _initialize_git_repo(entrypoint)
    seed = _seed_tasks_dir(entrypoint)
    original_head = _run_git(entrypoint, "rev-parse", "HEAD")
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=root,
            entrypoint=entrypoint,
            merge_validation_worktree=True,
            batched_merge=True,
            pre_merge_validation_commands=(
                _validation_command('printf x >> "$1"; kill -TERM $$', str(counter)),
            ),
        ),
        task_manager=_task_manager_with_tasks(seed),
    )
    a = await _ready_worktree_adding(
        orchestrator, seed, name="a", task_file="002-a.yaml", task=_named_task(seed, "a")
    )
    b = await _ready_worktree_adding(
        orchestrator, seed, name="b", task_file="003-b.yaml", task=_named_task(seed, "b")
    )

    await orchestrator.process_merge_queue_once_for_test()

    # 2 (batch: cancelled + spent retry) + 1 (half a) + 1 (half b).
    assert counter.read_text() == "xxxx"
    assert orchestrator.worktrees_by_id[a.worktree_id].state is WorktreeState.PENDING
    assert orchestrator.worktrees_by_id[b.worktree_id].state is WorktreeState.PENDING
    assert _run_git(entrypoint, "rev-parse", "HEAD") == original_head
    assert not (task_directory(entrypoint) / "002-a.yaml").exists()
    assert not (task_directory(entrypoint) / "003-b.yaml").exists()


async def test_batched_merge_timeout_is_a_failure_not_a_cancellation(
    tmp_path: Path,
) -> None:
    """A timed-out batch validation keeps ordinary failure handling — no retry.

    The orchestrator SIGTERM/SIGKILLs a timed-out build itself, but the timeout
    path records its own distinct failure and must not be mistaken for a
    cancellation: the single build runs once and the member bounces.
    """

    _require_git_and_sh()
    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    counter = tmp_path / "build-count.txt"
    _initialize_git_repo(entrypoint)
    seed = _seed_tasks_dir(entrypoint)
    original_head = _run_git(entrypoint, "rev-parse", "HEAD")
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=root,
            entrypoint=entrypoint,
            merge_validation_worktree=True,
            batched_merge=True,
            pre_merge_validation_commands=(
                AsyncOrchestratorValidationCommandConfig(
                    argv=("sh", "-c", 'printf x >> "$1"; sleep 30', "validator", str(counter)),
                    timeout_seconds=0.2,
                ),
            ),
        ),
        task_manager=_task_manager_with_tasks(seed),
    )
    wt = await _ready_worktree_adding(
        orchestrator, seed, name="hangs", task_file="002-a.yaml", task=_named_task(seed, "a")
    )

    await orchestrator.process_merge_queue_once_for_test()

    # One build only: the timeout was not retried as a cancellation.
    assert counter.read_text() == "x"
    assert orchestrator.worktrees_by_id[wt.worktree_id].state is WorktreeState.PENDING
    assert _run_git(entrypoint, "rev-parse", "HEAD") == original_head


async def test_batched_merge_crash_signal_is_a_failure_not_a_cancellation(
    tmp_path: Path,
) -> None:
    """A validator crash (SIGSEGV) keeps ordinary failure handling — no retry.

    Only cancellation signals (SIGTERM, SIGKILL, ...) are retried in place. A
    deterministic validator crash is evidence of a real failure, and a retry
    would convert that failure signal into a pass: the build runs exactly
    once, the member bounces, and nothing is published.
    """

    _require_git_and_sh()
    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    counter = tmp_path / "build-count.txt"
    _initialize_git_repo(entrypoint)
    seed = _seed_tasks_dir(entrypoint)
    original_head = _run_git(entrypoint, "rev-parse", "HEAD")
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=root,
            entrypoint=entrypoint,
            merge_validation_worktree=True,
            batched_merge=True,
            pre_merge_validation_commands=(
                _validation_command('printf x >> "$1"; kill -SEGV $$', str(counter)),
            ),
        ),
        task_manager=_task_manager_with_tasks(seed),
    )
    wt = await _ready_worktree_adding(
        orchestrator,
        seed,
        name="crashes",
        task_file="002-a.yaml",
        task=_named_task(seed, "a"),
    )

    await orchestrator.process_merge_queue_once_for_test()

    # One build only: the crash was not retried as a cancellation.
    assert counter.read_text() == "x"
    assert orchestrator.worktrees_by_id[wt.worktree_id].state is WorktreeState.PENDING
    assert _run_git(entrypoint, "rev-parse", "HEAD") == original_head
    assert not (task_directory(entrypoint) / "002-a.yaml").exists()


async def test_batched_merge_crash_purges_ignored_state_before_bisection(
    tmp_path: Path,
) -> None:
    """Bisection after a validator crash must not trust crash-left ignored state,
    and the purge must re-provision staging's infrastructure.

    The validator crashes with SIGSEGV after planting a gitignored marker in
    staging, and passes whenever the marker is present. ``git clean -fd``
    (the normal staging sync) preserves ignored files, so without the
    post-crash purge each one-member bisection half would see the marker,
    "pass", and publish a batch whose clean validation still crashes. With the
    purge, both halves rebuild cold, crash the same way, and bounce: three
    builds (batch + two halves), nothing lands.

    The purge (``git clean -ffdx``) also deletes staging's *provisioned*
    gitignored infrastructure — the workspace-mirror ``.lake`` symlink and the
    setup command's artifact here — which must then be fully re-provisioned.
    Every validation records whether that infrastructure is present: the
    post-crash halves must run with it restored, while the crash-planted marker
    and an ignored nested Git repository are both gone.
    """

    _require_git_and_sh()
    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    counter = tmp_path / "build-count.txt"
    infra_counter = tmp_path / "infra-count.txt"
    _initialize_git_repo(entrypoint)
    # No trailing slashes: staging's .lake is a symlink, which a "dir/"
    # pattern would not match (and plain `clean -fd` would then delete it).
    (entrypoint / ".gitignore").write_text(
        ".cache\n.lake\n.setup-marker\n", encoding="utf-8"
    )
    _run_git(entrypoint, "add", ".gitignore")
    _run_git(entrypoint, "commit", "-m", "ignore build state and infra")
    # Mirror symlink source (stands in for the shared dependency tree).
    (entrypoint / ".lake" / "packages").mkdir(parents=True)
    (entrypoint / ".lake" / "packages" / "dep.txt").write_text("dep\n", encoding="utf-8")
    seed = _seed_tasks_dir(entrypoint)
    original_head = _run_git(entrypoint, "rev-parse", "HEAD")
    # Runs with cwd=staging: the marker lives in staging's ignored .cache/,
    # and each run records whether the provisioned infra is present.
    script = (
        'printf x >> "$1"; '
        'if [ -L .lake ] && [ -e .setup-marker ]; then printf y >> "$2"; fi; '
        "mkdir -p .cache/nested-repo; "
        "if [ ! -d .cache/nested-repo/.git ]; then "
        "git init -q .cache/nested-repo; fi; "
        ": > .cache/nested-repo/crash-tainted; "
        "if [ -e .cache/crashed ]; then exit 0; fi; "
        ": > .cache/crashed; kill -SEGV $$"
    )
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=root,
            entrypoint=entrypoint,
            merge_validation_worktree=True,
            batched_merge=True,
            workspace_mirror=AsyncOrchestratorWorkspaceMirrorConfig(
                enabled=True,
                symlink_paths=[".lake"],
            ),
            worktree_setup_command=AsyncOrchestratorWorktreeSetupCommandConfig(
                argv=("sh", "-c", ': > "$1/.setup-marker"', "setup", "{worktree}"),
            ),
            pre_merge_validation_commands=(
                _validation_command(script, str(counter), str(infra_counter)),
            ),
        ),
        task_manager=_task_manager_with_tasks(seed),
    )
    a = await _ready_worktree_adding(
        orchestrator, seed, name="a", task_file="002-a.yaml", task=_named_task(seed, "a")
    )
    b = await _ready_worktree_adding(
        orchestrator, seed, name="b", task_file="003-b.yaml", task=_named_task(seed, "b")
    )

    await orchestrator.process_merge_queue_once_for_test()

    # Three builds, all crashing cold: batch, then each purged half.
    assert counter.read_text() == "xxx"
    # The mirror symlink and setup artifact were present for every build,
    # including both post-purge halves (re-provisioned, not just cleaned).
    assert infra_counter.read_text() == "yyy"
    assert orchestrator.worktrees_by_id[a.worktree_id].state is WorktreeState.PENDING
    assert orchestrator.worktrees_by_id[b.worktree_id].state is WorktreeState.PENDING
    assert _run_git(entrypoint, "rev-parse", "HEAD") == original_head
    assert not (task_directory(entrypoint) / "002-a.yaml").exists()
    assert not (task_directory(entrypoint) / "003-b.yaml").exists()
    # Post-purge staging is indistinguishable from freshly provisioned: infra
    # restored, crash-planted marker gone.
    staging = root / "staging"
    assert (staging / ".lake").is_symlink()
    assert (staging / ".setup-marker").exists()
    assert not (staging / ".cache" / "nested-repo" / "crash-tainted").exists()
    assert not (staging / ".cache" / "nested-repo").exists()
    assert not (staging / ".cache").exists()


async def test_cross_worktree_cycle_caught_by_cumulative_revalidation(tmp_path: Path) -> None:
    """Two worktrees that each edit a *pre-existing* task so that together they
    form a ``depends_on`` cycle.

    Neither edit is cyclic on its own, so isolation must halve — and the second
    half must then be re-validated against the *advancing* main (with the first
    half's merge applied) to catch the cycle. One worktree lands; the other is
    bounced; the cycle never reaches main. Guards the regression where a
    base-relative task pre-screen passed both halves independently and merged the
    cycle.
    """

    _require_git_and_sh()
    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    _initialize_git_repo(entrypoint)
    seed = _seed_tasks_dir(entrypoint)
    t1 = Task(id="task-t1", title="T1", summary="T1", description="T1.", depends_on=[seed.id])
    t2 = Task(id="task-t2", title="T2", summary="T2", description="T2.", depends_on=[seed.id])
    write_task(task_directory(entrypoint) / "002-t1.yaml", t1)
    write_task(task_directory(entrypoint) / "003-t2.yaml", t2)
    _run_git(entrypoint, "add", "tasks")
    _run_git(entrypoint, "commit", "-m", "seed t1 and t2")
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=root,
            entrypoint=entrypoint,
            merge_validation_worktree=True,
            batched_merge=True,
            pre_merge_validation_commands=(_validation_command("true"),),  # build always green
        ),
        task_manager=_task_manager_with_tasks(seed, t1, t2),
    )
    # A: t1 -> t2 ; B: t2 -> t1. Each acyclic alone; together a cycle.
    t1_cyc = Task(
        id="task-t1", title="T1", summary="T1", description="T1.", depends_on=[seed.id, t2.id]
    )
    t2_cyc = Task(
        id="task-t2", title="T2", summary="T2", description="T2.", depends_on=[seed.id, t1.id]
    )
    a = await _ready_worktree_adding(
        orchestrator, seed, name="a", task_file="002-t1.yaml", task=t1_cyc
    )
    b = await _ready_worktree_adding(
        orchestrator, seed, name="b", task_file="003-t2.yaml", task=t2_cyc
    )

    await orchestrator.process_merge_queue_once_for_test()

    states = {
        orchestrator.worktrees_by_id[a.worktree_id].state,
        orchestrator.worktrees_by_id[b.worktree_id].state,
    }
    # Exactly one landed; the other bounced — the cycle was prevented.
    assert states == {WorktreeState.CLOSED, WorktreeState.PENDING}
    # Published main is acyclic: the two cross-edges are not both present.
    t1_has_t2 = "task-t2" in (task_directory(entrypoint) / "002-t1.yaml").read_text()
    t2_has_t1 = "task-t1" in (task_directory(entrypoint) / "003-t2.yaml").read_text()
    assert not (t1_has_t2 and t2_has_t1)


async def test_harmless_editor_of_duplicated_task_file_publishes(tmp_path: Path) -> None:
    """Attribution selects the bisection partition; it never bounces directly.

    A duplicate-id failure maps to every file declaring the id: the member that
    harmlessly *edited* the pre-existing declaring file is attributed alongside
    the member that *added* the duplicate. With an unrelated third member in the
    batch the culprit set (2) is smaller than the batch (3) — a direct bounce of
    the attributed set would punish the harmless editor for its batch-mate,
    worse than bisection, under which the edit lands. Instead the attributed
    pair is validated on its own first and bisected within: the editor lands,
    only the duplicator bounces (after failing the build-free task gate alone),
    and the unrelated member publishes (adversarial review of issue #128).
    """

    _require_git_and_sh()
    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    counter = tmp_path / "build-count.txt"
    _initialize_git_repo(entrypoint)
    seed = _seed_tasks_dir(entrypoint)
    existing = Task(
        id="task-x",
        title="X",
        summary="X",
        description="X.",
        depends_on=[seed.id],
    )
    write_task(task_directory(entrypoint) / "002-x.yaml", existing)
    _run_git(entrypoint, "add", "tasks")
    _run_git(entrypoint, "commit", "-m", "seed task-x")
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=root,
            entrypoint=entrypoint,
            merge_validation_worktree=True,
            batched_merge=True,
            pre_merge_validation_commands=(_validation_command(f"printf x >> {counter}"),),
        ),
        task_manager=_task_manager_with_tasks(seed, existing),
    )
    # editor harmlessly amends the existing declaration of task-x in place.
    edited = existing.model_copy(update={"summary": "X, clarified"})
    editor = await _ready_worktree_adding(
        orchestrator, seed, name="editor", task_file="002-x.yaml", task=edited
    )
    # duplicator re-declares task-x in a new file.
    duplicate = Task(
        id="task-x",
        title="X again",
        summary="X again",
        description="X again.",
        depends_on=[seed.id],
    )
    duplicator = await _ready_worktree_adding(
        orchestrator, seed, name="duplicator", task_file="010-dup.yaml", task=duplicate
    )
    unrelated = await _ready_worktree_adding(
        orchestrator, seed, name="unrelated", task_file="003-c.yaml", task=_named_task(seed, "c")
    )

    await orchestrator.process_merge_queue_once_for_test()

    assert orchestrator.worktrees_by_id[editor.worktree_id].state is WorktreeState.CLOSED
    assert orchestrator.worktrees_by_id[unrelated.worktree_id].state is WorktreeState.CLOSED
    updated_duplicator = orchestrator.worktrees_by_id[duplicator.worktree_id]
    assert updated_duplicator.state is WorktreeState.PENDING
    assert "duplicate task id" in updated_duplicator.discussion[-1].message
    # The harmless edit landed; the duplicate declaration never did.
    assert "X, clarified" in (task_directory(entrypoint) / "002-x.yaml").read_text()
    assert not (task_directory(entrypoint) / "010-dup.yaml").exists()
    # Two builds: the editor's, then the unrelated member's — the duplicator is
    # isolated by the build-free task gate in every failing round.
    assert counter.read_text() == "xx"


async def test_cross_worktree_cycle_with_unrelated_third_lands_one_cycle_former(
    tmp_path: Path,
) -> None:
    """A two-member cycle plus an unrelated third bounces exactly ONE cycle former.

    The cycle error names both edited files, attributing both cycle formers;
    with the unrelated third member the culprit set (2) is smaller than the
    batch (3), and a direct bounce of the attributed set would evict both cycle
    formers even though either edit is valid alone (batch composition would
    decide what publishes: with only the two formers queued, one landed).
    Instead the attributed pair is validated on its own first and bisected
    within, against the advancing main — one former lands, the other bounces,
    matching the two-member behavior (adversarial review of issue #128).
    """

    _require_git_and_sh()
    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    counter = tmp_path / "build-count.txt"
    _initialize_git_repo(entrypoint)
    seed = _seed_tasks_dir(entrypoint)
    t1 = Task(id="task-t1", title="T1", summary="T1", description="T1.", depends_on=[seed.id])
    t2 = Task(id="task-t2", title="T2", summary="T2", description="T2.", depends_on=[seed.id])
    write_task(task_directory(entrypoint) / "002-t1.yaml", t1)
    write_task(task_directory(entrypoint) / "003-t2.yaml", t2)
    _run_git(entrypoint, "add", "tasks")
    _run_git(entrypoint, "commit", "-m", "seed t1 and t2")
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=root,
            entrypoint=entrypoint,
            merge_validation_worktree=True,
            batched_merge=True,
            pre_merge_validation_commands=(_validation_command(f"printf x >> {counter}"),),
        ),
        task_manager=_task_manager_with_tasks(seed, t1, t2),
    )
    t1_cyc = Task(
        id="task-t1", title="T1", summary="T1", description="T1.", depends_on=[seed.id, t2.id]
    )
    t2_cyc = Task(
        id="task-t2", title="T2", summary="T2", description="T2.", depends_on=[seed.id, t1.id]
    )
    a = await _ready_worktree_adding(
        orchestrator, seed, name="a", task_file="002-t1.yaml", task=t1_cyc
    )
    b = await _ready_worktree_adding(
        orchestrator, seed, name="b", task_file="003-t2.yaml", task=t2_cyc
    )
    unrelated = await _ready_worktree_adding(
        orchestrator, seed, name="unrelated", task_file="004-c.yaml", task=_named_task(seed, "c")
    )

    await orchestrator.process_merge_queue_once_for_test()

    assert orchestrator.worktrees_by_id[unrelated.worktree_id].state is WorktreeState.CLOSED
    # Bisection semantics among the attributed pair: the first former lands,
    # the second (re-validated against the advancing main) bounces.
    assert orchestrator.worktrees_by_id[a.worktree_id].state is WorktreeState.CLOSED
    assert orchestrator.worktrees_by_id[b.worktree_id].state is WorktreeState.PENDING
    # Published main is acyclic: the two cross-edges are not both present.
    t1_has_t2 = "task-t2" in (task_directory(entrypoint) / "002-t1.yaml").read_text()
    t2_has_t1 = "task-t1" in (task_directory(entrypoint) / "003-t2.yaml").read_text()
    assert not (t1_has_t2 and t2_has_t1)
    # Two builds: the surviving cycle former's, then the unrelated member's.
    assert counter.read_text() == "xx"


async def test_deleted_dependency_file_attributed_to_deleting_member(tmp_path: Path) -> None:
    """Deleting a task file out from under a pre-existing depender is attributed
    to the deleting member, which is probed alone and bounces.

    The unknown-dependency failure names the depender's file (touched by
    nobody) *and* — resolved from the pre-merge tree — the deleted declaring
    file, which the deleter's diff touched (deletions appear in
    ``git diff --name-only``). The deleter is validated alone, fails the
    build-free task gate on its own contribution, and bounces; the innocent
    members build once and land (adversarial review of issue #128).
    """

    _require_git_and_sh()
    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    counter = tmp_path / "build-count.txt"
    _initialize_git_repo(entrypoint)
    seed = _seed_tasks_dir(entrypoint)
    dep_task = Task(id="task-dep", title="Dep", summary="Dep", description="Dep.")
    depender = Task(
        id="task-a",
        title="A",
        summary="A",
        description="A.",
        depends_on=["task-dep"],
    )
    write_task(task_directory(entrypoint) / "004-a.yaml", depender)
    write_task(task_directory(entrypoint) / "005-dep.yaml", dep_task)
    _run_git(entrypoint, "add", "tasks")
    _run_git(entrypoint, "commit", "-m", "seed depender and dependency")
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=root,
            entrypoint=entrypoint,
            merge_validation_worktree=True,
            batched_merge=True,
            pre_merge_validation_commands=(_validation_command(f"printf x >> {counter}"),),
        ),
        task_manager=_task_manager_with_tasks(seed, dep_task, depender),
    )
    good1 = await _ready_worktree_adding(
        orchestrator, seed, name="g1", task_file="002-g1.yaml", task=_named_task(seed, "g1")
    )
    deleter = await _ready_worktree_adding(
        orchestrator, seed, name="deleter", delete_task_file="005-dep.yaml"
    )
    good2 = await _ready_worktree_adding(
        orchestrator, seed, name="g2", task_file="003-g2.yaml", task=_named_task(seed, "g2")
    )

    await orchestrator.process_merge_queue_once_for_test()

    assert orchestrator.worktrees_by_id[good1.worktree_id].state is WorktreeState.CLOSED
    assert orchestrator.worktrees_by_id[good2.worktree_id].state is WorktreeState.CLOSED
    updated_deleter = orchestrator.worktrees_by_id[deleter.worktree_id]
    assert updated_deleter.state is WorktreeState.PENDING
    assert "unknown task id" in updated_deleter.discussion[-1].message
    # The deletion never landed and the dependency graph on main stays intact.
    assert (task_directory(entrypoint) / "005-dep.yaml").exists()
    # One-round attribution: a single build for the surviving pair.
    assert counter.read_text() == "x"


async def test_renamed_dependency_with_changed_id_attributes_renaming_member(
    tmp_path: Path,
) -> None:
    """A rename plus id change reports the deleted source path for attribution."""

    _require_git_and_sh()
    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    counter = tmp_path / "build-count.txt"
    _initialize_git_repo(entrypoint)
    seed = _seed_tasks_dir(entrypoint)
    dependency = Task(id="task-dep", title="Dep", summary="Dep", description="Dep.")
    depender = Task(
        id="task-a",
        title="A",
        summary="A",
        description="A.",
        depends_on=[dependency.id],
    )
    write_task(task_directory(entrypoint) / "004-a.yaml", depender)
    write_task(task_directory(entrypoint) / "005-dep.yaml", dependency)
    _run_git(entrypoint, "add", "tasks")
    _run_git(entrypoint, "commit", "-m", "seed depender and dependency")
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=root,
            entrypoint=entrypoint,
            merge_validation_worktree=True,
            batched_merge=True,
            pre_merge_validation_commands=(_validation_command(f"printf x >> {counter}"),),
        ),
        task_manager=_task_manager_with_tasks(seed, dependency, depender),
    )
    good1 = await _ready_worktree_adding(
        orchestrator, seed, name="g1", task_file="002-g1.yaml", task=_named_task(seed, "g1")
    )
    renamer = await orchestrator.create_fresh_worktree_for_test(name="renamer", task=seed)
    old_path = task_directory(renamer.path) / "005-dep.yaml"
    new_path = task_directory(renamer.path) / "006-renamed.yaml"
    old_path.rename(new_path)
    write_task(new_path, dependency.model_copy(update={"id": "task-renamed"}))
    _commit_worktree(renamer.path)
    await orchestrator.transition_worktree_for_test(renamer.worktree_id, WorktreeState.MERGE)
    good2 = await _ready_worktree_adding(
        orchestrator, seed, name="g2", task_file="003-g2.yaml", task=_named_task(seed, "g2")
    )

    await orchestrator.process_merge_queue_once_for_test()

    assert orchestrator.worktrees_by_id[good1.worktree_id].state is WorktreeState.CLOSED
    assert orchestrator.worktrees_by_id[good2.worktree_id].state is WorktreeState.CLOSED
    updated_renamer = orchestrator.worktrees_by_id[renamer.worktree_id]
    assert updated_renamer.state is WorktreeState.PENDING
    assert "unknown task id" in updated_renamer.discussion[-1].message
    assert (task_directory(entrypoint) / "005-dep.yaml").exists()
    assert not (task_directory(entrypoint) / "006-renamed.yaml").exists()
    # The renamer is attributed from the deleted old path in the first round;
    # only the surviving pair reaches the build gate.
    assert counter.read_text() == "x"


async def test_attributed_early_members_publish_when_later_member_broke_them(
    tmp_path: Path,
) -> None:
    """Attribution never bounces — attributed members are validated alone FIRST.

    FIFO-early members a and b add modules that the build reports errors in,
    but only because the FIFO-later member c changed the API they use: the
    diagnostics attribute a and b while the true culprit c is unattributed.
    Publishing the unattributed rest first would land c and then evict both a
    and b (round-2 adversarial review, finding 1) — worse than pure FIFO
    bisection, which lands a and b. Attributed-first ordering validates {a, b}
    alone against the original base: they pass and publish, then c fails its
    own confirming build against the advanced head and bounces.
    """

    _require_git_and_sh()
    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    counter = tmp_path / "build-count.txt"
    _initialize_git_repo(entrypoint)
    seed = _seed_tasks_dir(entrypoint)
    # The "API break": errors in A.lean/B.lean appear only when NewApi.marker
    # (c's change) is present alongside them.
    script = (
        f"printf x >> {counter}; "
        "if [ -e NewApi.marker ]; then "
        "if [ -e A.lean ]; then echo 'A.lean:1:1: error: broken by new api'; fi; "
        "if [ -e B.lean ]; then echo 'B.lean:1:1: error: broken by new api'; fi; "
        "if [ -e A.lean ] || [ -e B.lean ]; then exit 1; fi; "
        "fi"
    )
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=root,
            entrypoint=entrypoint,
            merge_validation_worktree=True,
            batched_merge=True,
            pre_merge_validation_commands=(_validation_command(script),),
        ),
        task_manager=_task_manager_with_tasks(seed),
    )
    a = await _ready_worktree_adding(orchestrator, seed, name="a", plain_file="A.lean")
    b = await _ready_worktree_adding(orchestrator, seed, name="b", plain_file="B.lean")
    c = await _ready_worktree_adding(orchestrator, seed, name="c", plain_file="NewApi.marker")

    await orchestrator.process_merge_queue_once_for_test()

    # FIFO blame preserved: the earlier members whose files carried the
    # diagnostics land untouched; the later member that broke them bounces.
    assert orchestrator.worktrees_by_id[a.worktree_id].state is WorktreeState.CLOSED
    assert orchestrator.worktrees_by_id[b.worktree_id].state is WorktreeState.CLOSED
    assert orchestrator.worktrees_by_id[c.worktree_id].state is WorktreeState.PENDING
    assert (entrypoint / "A.lean").exists()
    assert (entrypoint / "B.lean").exists()
    assert not (entrypoint / "NewApi.marker").exists()
    # Three builds: the failing batch, {a, b} alone (passes, publishes), then
    # c's confirming solo build (fails, bounces).
    assert counter.read_text() == "xxx"


async def test_transitive_complete_depends_on_open_attributes_intermediate_editor(
    tmp_path: Path,
) -> None:
    """The member that edited the *intermediate* edge of a complete→open path
    is attributed and bounces; unrelated members build once and land.

    Base holds complete ``task-parent`` -> complete ``task-middle`` and open
    ``task-open``; the culprit edits *middle* to depend on *open*, so the
    violation's endpoints (parent, open) live in files nobody touched. The
    graph error must name every task along the path — endpoints and
    intermediates — for the file mapping to reach the culprit (round-2
    adversarial review, finding 3); endpoint-only reporting would find no
    toucher and fall back to build-per-round bisection.
    """

    _require_git_and_sh()
    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    counter = tmp_path / "build-count.txt"
    _initialize_git_repo(entrypoint)
    seed = _seed_tasks_dir(entrypoint)
    open_task = Task(id="task-open", title="Open", summary="Open", description="Open.")
    middle = Task(
        id="task-middle",
        title="Middle",
        summary="Middle",
        description="Middle.",
        status=TaskStatus.COMPLETE,
    )
    parent = Task(
        id="task-parent",
        title="Parent",
        summary="Parent",
        description="Parent.",
        status=TaskStatus.COMPLETE,
        depends_on=["task-middle"],
    )
    write_task(task_directory(entrypoint) / "010-parent.yaml", parent)
    write_task(task_directory(entrypoint) / "011-middle.yaml", middle)
    write_task(task_directory(entrypoint) / "012-open.yaml", open_task)
    _run_git(entrypoint, "add", "tasks")
    _run_git(entrypoint, "commit", "-m", "seed parent -> middle chain and open task")
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=root,
            entrypoint=entrypoint,
            merge_validation_worktree=True,
            batched_merge=True,
            pre_merge_validation_commands=(_validation_command(f"printf x >> {counter}"),),
        ),
        task_manager=_task_manager_with_tasks(seed, open_task, middle, parent),
    )
    good1 = await _ready_worktree_adding(
        orchestrator, seed, name="g1", task_file="002-g1.yaml", task=_named_task(seed, "g1")
    )
    # The culprit edits only the intermediate task: middle -> open.
    middle_edited = middle.model_copy(update={"dependencies": ["task-open"]})
    m_editor = await _ready_worktree_adding(
        orchestrator, seed, name="m-editor", task_file="011-middle.yaml", task=middle_edited
    )
    good2 = await _ready_worktree_adding(
        orchestrator, seed, name="g2", task_file="003-g2.yaml", task=_named_task(seed, "g2")
    )

    await orchestrator.process_merge_queue_once_for_test()

    assert orchestrator.worktrees_by_id[good1.worktree_id].state is WorktreeState.CLOSED
    assert orchestrator.worktrees_by_id[good2.worktree_id].state is WorktreeState.CLOSED
    updated_editor = orchestrator.worktrees_by_id[m_editor.worktree_id]
    assert updated_editor.state is WorktreeState.PENDING
    message = updated_editor.discussion[-1].message
    assert "complete task cannot depend on open task" in message
    # The bounce message names the intermediate file (the culprit's edit), not
    # just the untouched endpoint files.
    assert "011-middle.yaml" in message
    # The invalid edge never landed.
    assert "task-open" not in (task_directory(entrypoint) / "011-middle.yaml").read_text()
    # Full-path attribution: the culprit fails the build-free gate alone, so a
    # single build (the surviving pair's) ran; endpoint-only reporting would
    # have needed a second build round to isolate it.
    assert counter.read_text() == "x"


async def test_isolation_worklist_bounded_for_degenerate_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The isolation driver is an explicit worklist, not Python recursion.

    Reproduces the round-3 reviewer's degenerate probe: every validation
    attributes exactly one member, then reprocesses the shrinking remainder.
    All git/task-validation boundaries are mocked, so no repositories or builds
    run. Even at N=600 and a modest recursion limit, only one
    ``_publish_or_bisect`` invocation may be active at a time.
    """

    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=tmp_path / "orch",
            entrypoint=tmp_path / "ep",
            merge_validation_worktree=True,
            batched_merge=True,
        ),
    )

    def member(index: int) -> tuple[AsyncOrchestratorWorktree, str]:
        name = f"worktree_{index:06d}"
        worktree = AsyncOrchestratorWorktree(
            worktree_id=name, path=tmp_path / name, head="feedface"
        )
        return (worktree, f"head-{name}")

    members = [member(index) for index in range(600)]
    member_ids = [worktree.worktree_id for (worktree, _) in members]
    assembled_ids: list[str] = []
    bounced: list[str] = []

    async def fake_ensure_validation_worktree(
        self: WorktreeTestingOrchestrator, *, head: str
    ) -> Path:
        return tmp_path / "unused-staging"

    def fake_assemble_batch(
        staging: Path,
        base_head: str,
        member_heads: list[tuple[str, str]],
    ) -> tuple[list[tuple[str, str]], list[object], str]:
        assembled_ids[:] = [worktree_id for worktree_id, _ in member_heads]
        return list(member_heads), [], f"staged-{base_head}"

    def fake_check_post_merge_task_tree(
        *, entrypoint: Path, original_head: str
    ) -> TaskValidationFailure:
        culprit = assembled_ids[0]
        return TaskValidationFailure(
            summary=f"validation failed for {culprit}",
            detail=f"validation failed for {culprit}",
            offending_paths=(f"{culprit}.lean",),
        )

    def fake_members_touching(
        self: WorktreeTestingOrchestrator,
        group: list[tuple[AsyncOrchestratorWorktree, str]],
        reported_paths: set[str],
        base_head: str,
    ) -> list[tuple[AsyncOrchestratorWorktree, str]]:
        return [m for m in group if f"{m[0].worktree_id}.lean" in reported_paths]

    async def fake_bounce_worktrees(
        self: WorktreeTestingOrchestrator,
        worktrees: list[AsyncOrchestratorWorktree],
        message: str,
    ) -> None:
        bounced.extend(worktree.worktree_id for worktree in worktrees)

    def fake_sync_staging_to_head(staging: Path, head: str) -> None:
        return None

    original_publish = WorktreeTestingOrchestrator._publish_or_bisect  # pyright: ignore[reportPrivateUsage]
    active_calls = 0
    max_active_calls = 0

    async def tracked_publish(
        self: WorktreeTestingOrchestrator,
        group: list[tuple[AsyncOrchestratorWorktree, str]],
        base_head: str,
        retry_budget: _orchestrator_module._CancellationRetryBudget,  # pyright: ignore[reportPrivateUsage]
    ) -> str:
        nonlocal active_calls, max_active_calls
        active_calls += 1
        max_active_calls = max(max_active_calls, active_calls)
        try:
            return await original_publish(self, group, base_head, retry_budget)
        finally:
            active_calls -= 1

    monkeypatch.setattr(
        WorktreeTestingOrchestrator,
        "_ensure_validation_worktree",
        fake_ensure_validation_worktree,
    )
    monkeypatch.setattr(_orchestrator_module, "_assemble_batch", fake_assemble_batch)
    monkeypatch.setattr(
        _orchestrator_module, "_check_post_merge_task_tree", fake_check_post_merge_task_tree
    )
    monkeypatch.setattr(
        _orchestrator_module, "_sync_staging_to_head", fake_sync_staging_to_head
    )
    monkeypatch.setattr(WorktreeTestingOrchestrator, "_members_touching", fake_members_touching)
    monkeypatch.setattr(WorktreeTestingOrchestrator, "_bounce_worktrees", fake_bounce_worktrees)
    monkeypatch.setattr(WorktreeTestingOrchestrator, "_publish_or_bisect", tracked_publish)

    old_recursion_limit = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(250)
        head = await orchestrator._publish_or_bisect(  # pyright: ignore[reportPrivateUsage]
            members,
            "base",
            _orchestrator_module._CancellationRetryBudget(),  # pyright: ignore[reportPrivateUsage]
        )
    finally:
        sys.setrecursionlimit(old_recursion_limit)

    assert bounced == member_ids
    assert head == "base"
    assert max_active_calls == 1


async def test_batch_of_malformed_task_files_attributed_in_one_round(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every independently malformed file is attributed by the first gate failure."""

    _require_git_and_sh()
    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    _initialize_git_repo(entrypoint)
    seed = _seed_tasks_dir(entrypoint)
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=root,
            entrypoint=entrypoint,
            merge_validation_worktree=True,
            batched_merge=True,
        ),
        task_manager=_task_manager_with_tasks(seed),
    )
    malformed_count = 8
    malformed = [
        await _ready_worktree_adding(
            orchestrator,
            seed,
            name=f"malformed-{index}",
            raw_task_file=(f"05{index}-malformed.yaml", "[not a mapping]\n"),
        )
        for index in range(malformed_count)
    ]

    original_assemble = _orchestrator_module._assemble_batch  # pyright: ignore[reportPrivateUsage]
    original_check = _orchestrator_module._check_post_merge_task_tree  # pyright: ignore[reportPrivateUsage]
    assembly_work = 0
    validation_calls = 0
    first_failure: TaskValidationFailure | None = None

    def tracked_assemble(
        staging: Path,
        base_head: str,
        members: list[tuple[str, str]],
    ) -> tuple[list[tuple[str, str]], list[tuple[str, subprocess.CalledProcessError]], str]:
        nonlocal assembly_work
        assembly_work += len(members)
        return original_assemble(staging, base_head, members)

    def tracked_check(
        *, entrypoint: Path, original_head: str
    ) -> TaskValidationFailure | None:
        nonlocal validation_calls, first_failure
        validation_calls += 1
        failure = original_check(entrypoint=entrypoint, original_head=original_head)
        if first_failure is None and failure is not None:
            first_failure = failure
        return failure

    monkeypatch.setattr(_orchestrator_module, "_assemble_batch", tracked_assemble)
    monkeypatch.setattr(_orchestrator_module, "_check_post_merge_task_tree", tracked_check)

    await orchestrator.process_merge_queue_once_for_test()

    assert first_failure is not None
    assert {Path(path).name for path in first_failure.offending_paths} == {
        f"05{index}-malformed.yaml" for index in range(malformed_count)
    }
    # A single balanced isolation tree is 8 + 2*4 + 4*2 + 8*1 = 32 member
    # assemblies. First-file-only attribution peels shrinking remainders and
    # exceeds this bound (43 for N=8).
    assert assembly_work <= malformed_count * 4
    assert validation_calls <= 2 * malformed_count
    assert all(
        orchestrator.worktrees_by_id[worktree.worktree_id].state is WorktreeState.PENDING
        for worktree in malformed
    )


async def test_batch_of_independently_cyclic_members_attributed_in_one_round(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """N members each adding its own self-cyclic task are ALL attributed by the
    first gate-1 failure and resolved build-free.

    ``validate_task_graph`` reports every violation at once (round-3
    adversarial review), so the batch failure names all N cyclic files, the
    attributed set fails alone, and halving within it bounces each culprit off
    its own build-free solo failure — the expensive build runs exactly once,
    for the surviving good members.
    """

    _require_git_and_sh()
    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    counter = tmp_path / "build-count.txt"
    _initialize_git_repo(entrypoint)
    seed = _seed_tasks_dir(entrypoint)
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=root,
            entrypoint=entrypoint,
            merge_validation_worktree=True,
            batched_merge=True,
            pre_merge_validation_commands=(_validation_command(f"printf x >> {counter}"),),
        ),
        task_manager=_task_manager_with_tasks(seed),
    )
    cyclic: list[AsyncOrchestratorWorktree] = []
    cyclic_count = 12
    for index in range(cyclic_count):
        self_cycle = Task(
            id=f"task-cyc-{index}",
            title=f"Cyc {index}",
            summary=f"Cyc {index}",
            description=f"Cyc {index}.",
            depends_on=[f"task-cyc-{index}"],
        )
        cyclic.append(
            await _ready_worktree_adding(
                orchestrator,
                seed,
                name=f"cyc-{index}",
                task_file=f"05{index}-cyc.yaml",
                task=self_cycle,
            )
        )
    good1 = await _ready_worktree_adding(
        orchestrator, seed, name="g1", task_file="002-g1.yaml", task=_named_task(seed, "g1")
    )
    good2 = await _ready_worktree_adding(
        orchestrator, seed, name="g2", task_file="003-g2.yaml", task=_named_task(seed, "g2")
    )
    original_check = _orchestrator_module._check_post_merge_task_tree  # pyright: ignore[reportPrivateUsage]
    task_failures: list[TaskValidationFailure] = []

    def tracked_check_post_merge_task_tree(
        *, entrypoint: Path, original_head: str
    ) -> TaskValidationFailure | None:
        failure = original_check(entrypoint=entrypoint, original_head=original_head)
        if failure is not None:
            task_failures.append(failure)
        return failure

    monkeypatch.setattr(
        _orchestrator_module,
        "_check_post_merge_task_tree",
        tracked_check_post_merge_task_tree,
    )

    await orchestrator.process_merge_queue_once_for_test()

    # The initial whole-batch gate failure attributes every independently
    # cyclic contribution at once; isolation does not peel one member from a
    # repeatedly rebuilt shrinking remainder.
    assert {Path(path).name for path in task_failures[0].offending_paths} == {
        f"05{index}-cyc.yaml" for index in range(cyclic_count)
    }
    assert orchestrator.worktrees_by_id[good1.worktree_id].state is WorktreeState.CLOSED
    assert orchestrator.worktrees_by_id[good2.worktree_id].state is WorktreeState.CLOSED
    for index, worktree in enumerate(cyclic):
        updated = orchestrator.worktrees_by_id[worktree.worktree_id]
        assert updated.state is WorktreeState.PENDING
        # Each culprit's bounce message comes from its own solo failure and
        # names its own cycle file.
        assert f"05{index}-cyc.yaml" in updated.discussion[-1].message
        assert not (task_directory(entrypoint) / f"05{index}-cyc.yaml").exists()
    # All culprits were isolated by build-free gate-1 halving: the build ran
    # exactly once, for the two survivors.
    assert counter.read_text() == "x"


async def test_batch_with_distinct_open_dependencies_attributes_all_in_first_round(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """K members opening distinct dependencies are isolated by one bisection tree.

    The initial complete parent has a separate edge to each dependency. Reporting
    only its first reachable open dependency peels one member at a time and
    repeatedly reassembles the shrinking remainder, yielding quadratic assembly
    work. Reporting every open path attributes all K members immediately.
    """

    _require_git_and_sh()
    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    _initialize_git_repo(entrypoint)
    seed = _seed_tasks_dir(entrypoint)
    dependency_count = 8
    dependencies = [
        Task(
            id=f"task-dep-{index}",
            title=f"Dep {index}",
            summary=f"Dep {index}",
            description=f"Dep {index}.",
            status=TaskStatus.COMPLETE,
        )
        for index in range(dependency_count)
    ]
    parent = Task(
        id="task-parent",
        title="Parent",
        summary="Parent",
        description="Parent.",
        status=TaskStatus.COMPLETE,
        depends_on=[task.id for task in dependencies],
    )
    write_task(task_directory(entrypoint) / "010-parent.yaml", parent)
    for index, task in enumerate(dependencies):
        write_task(task_directory(entrypoint) / f"02{index}-dep.yaml", task)
    _run_git(entrypoint, "add", "tasks")
    _run_git(entrypoint, "commit", "-m", "seed complete parent and dependencies")
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=root,
            entrypoint=entrypoint,
            merge_validation_worktree=True,
            batched_merge=True,
        ),
        task_manager=_task_manager_with_tasks(seed, parent, *dependencies),
    )
    bad_members: list[AsyncOrchestratorWorktree] = []
    for index, dependency in enumerate(dependencies):
        bad_members.append(
            await _ready_worktree_adding(
                orchestrator,
                seed,
                name=f"opens-{index}",
                task_file=f"02{index}-dep.yaml",
                task=dependency.model_copy(update={"status": TaskStatus.OPEN}),
            )
        )

    original_assemble = _orchestrator_module._assemble_batch  # pyright: ignore[reportPrivateUsage]
    original_check = _orchestrator_module._check_post_merge_task_tree  # pyright: ignore[reportPrivateUsage]
    assembly_work = 0
    validation_calls = 0
    first_failure: TaskValidationFailure | None = None

    def tracked_assemble(
        staging: Path,
        base_head: str,
        members: list[tuple[str, str]],
    ) -> tuple[list[tuple[str, str]], list[tuple[str, subprocess.CalledProcessError]], str]:
        nonlocal assembly_work
        assembly_work += len(members)
        return original_assemble(staging, base_head, members)

    def tracked_check(
        *, entrypoint: Path, original_head: str
    ) -> TaskValidationFailure | None:
        nonlocal validation_calls, first_failure
        validation_calls += 1
        failure = original_check(entrypoint=entrypoint, original_head=original_head)
        if first_failure is None and failure is not None:
            first_failure = failure
        return failure

    monkeypatch.setattr(_orchestrator_module, "_assemble_batch", tracked_assemble)
    monkeypatch.setattr(_orchestrator_module, "_check_post_merge_task_tree", tracked_check)

    await orchestrator.process_merge_queue_once_for_test()

    assert first_failure is not None
    assert {Path(path).name for path in first_failure.offending_paths} >= {
        f"02{index}-dep.yaml" for index in range(dependency_count)
    }
    assert validation_calls <= 2 * dependency_count
    # A balanced isolation tree assembles each member once per tree level:
    # 8 + 8 + 8 + 8 = 32. First-path-only attribution costs 44 here and grows
    # quadratically as K increases.
    assert assembly_work <= dependency_count * 4
    assert all(
        orchestrator.worktrees_by_id[worktree.worktree_id].state is WorktreeState.PENDING
        for worktree in bad_members
    )


async def test_deleted_non_utf8_task_path_is_attributed_before_bisection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A raw pre-merge task path resolves and immediately identifies its deleter."""

    _require_git_and_sh()
    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    _initialize_git_repo(entrypoint)
    seed = _seed_tasks_dir(entrypoint)
    filename = os.fsdecode(b"dep-\xff.yaml")
    dependency = Task(
        id="task-raw-dependency",
        title="Raw dependency",
        summary="Raw dependency",
        description="Dependency stored under a non-UTF-8 filename.",
    )
    depender = Task(
        id="task-raw-depender",
        title="Raw depender",
        summary="Raw depender",
        description="Task that retains the dependency reference.",
        depends_on=[dependency.id],
    )
    write_task(task_directory(entrypoint) / filename, dependency)
    write_task(task_directory(entrypoint) / "010-depender.yaml", depender)
    _run_git(entrypoint, "add", "tasks")
    _run_git(entrypoint, "commit", "-m", "seed raw-path dependency")
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=root,
            entrypoint=entrypoint,
            merge_validation_worktree=True,
            batched_merge=True,
        ),
        task_manager=_task_manager_with_tasks(seed, dependency, depender),
    )
    deleter = await _ready_worktree_adding(
        orchestrator,
        seed,
        name="raw-path-deleter",
        delete_task_file=filename,
    )
    good = await _ready_worktree_adding(
        orchestrator,
        seed,
        name="unrelated-good",
        task_file="020-good.yaml",
        task=_named_task(seed, "unrelated-good"),
    )

    original_members_touching = (
        WorktreeTestingOrchestrator._members_touching  # pyright: ignore[reportPrivateUsage]
    )
    initial_reported_paths: set[str] | None = None
    initial_matched_ids: list[str] | None = None

    def tracked_members_touching(
        self: WorktreeTestingOrchestrator,
        members: list[tuple[AsyncOrchestratorWorktree, str]],
        reported_paths: set[str],
        base_head: str,
    ) -> list[tuple[AsyncOrchestratorWorktree, str]]:
        nonlocal initial_reported_paths, initial_matched_ids
        matched = original_members_touching(self, members, reported_paths, base_head)
        if len(members) == 2 and initial_reported_paths is None:
            initial_reported_paths = set(reported_paths)
            initial_matched_ids = [worktree.worktree_id for worktree, _ in matched]
        return matched

    monkeypatch.setattr(
        WorktreeTestingOrchestrator, "_members_touching", tracked_members_touching
    )

    await orchestrator.process_merge_queue_once_for_test()

    raw_old_path = f"tasks/{filename}"
    assert initial_reported_paths is not None
    assert raw_old_path in initial_reported_paths
    assert initial_matched_ids == [deleter.worktree_id]
    assert orchestrator.worktrees_by_id[deleter.worktree_id].state is WorktreeState.PENDING
    assert orchestrator.worktrees_by_id[good.worktree_id].state is WorktreeState.CLOSED


def test_non_utf8_git_filename_round_trips_through_attribution(tmp_path: Path) -> None:
    """Raw ``git diff -z`` path bytes cannot crash member attribution."""

    _require_git_and_sh()
    entrypoint = tmp_path / "entrypoint"
    _initialize_git_repo(entrypoint)
    base_head = _run_git(entrypoint, "rev-parse", "HEAD")
    filename = os.fsdecode(b"bad-\xff.lean")
    (entrypoint / filename).write_text("example\n", encoding="utf-8")
    _commit_worktree(entrypoint)
    head = _run_git(entrypoint, "rev-parse", "HEAD")

    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(root=tmp_path / "orch", entrypoint=entrypoint)
    )
    member = AsyncOrchestratorWorktree(
        worktree_id="worktree_000001", path=entrypoint, head=head
    )

    matched = orchestrator._members_touching(  # pyright: ignore[reportPrivateUsage]
        [(member, head)], {filename}, base_head
    )

    assert matched == [(member, head)]


def _timeout_validation_command(
    script: str, timeout_seconds: float
) -> AsyncOrchestratorValidationCommandConfig:
    return AsyncOrchestratorValidationCommandConfig(
        argv=("sh", "-c", script, "validator"),
        timeout_seconds=timeout_seconds,
    )


def _fake_lake_timeout_validation_command(
    tmp_path: Path, script: str, timeout_seconds: float
) -> AsyncOrchestratorValidationCommandConfig:
    """A hang-capable fake ``lake`` executable.

    ``script`` is installed as ``<tmp>/fake-bin/lake`` and invoked directly, so
    ``argv[0]``'s basename is literally ``lake`` and the command legitimately
    passes the ``_is_lake_invocation`` gate on the timeout-attribution
    heuristics."""

    lake = tmp_path / "fake-bin" / "lake"
    lake.parent.mkdir(parents=True, exist_ok=True)
    lake.write_text(f"#!/bin/sh\n{script}\n", encoding="utf-8")
    lake.chmod(0o755)
    return AsyncOrchestratorValidationCommandConfig(
        argv=(str(lake), "build"),
        timeout_seconds=timeout_seconds,
    )


async def test_timed_out_build_attributed_from_in_flight_module_marker(
    tmp_path: Path,
) -> None:
    """A timed-out batch build is attributed from lake's in-flight job header
    (``✖ [i/n] Building X``) in the output buffered at kill time; the member
    that touched the in-flight module is probed first — it fails its own
    confirming solo build (verify-before-bounce) and bounces — and the
    remainder re-validates and lands, with no per-halving re-timeouts
    (issue #133).

    The fake build prints lake-shaped progress, then hangs only while the
    culprit's ``Slow.lean`` is present, so the survivors' re-validation passes.
    ``build-count.txt`` records exactly three runs: the timed-out batch, the
    culprit's confirming solo build (times out again), and the survivors'
    passing build. Plain halving over three members would have re-timed-out
    once more.
    """

    _require_git_and_sh()
    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    counter = tmp_path / "build-count.txt"
    _initialize_git_repo(entrypoint)
    seed = _seed_tasks_dir(entrypoint)
    # \342\234\224 / \342\234\226 are the UTF-8 octal escapes for lake's
    # ``✔``/``✖`` status marks (octal because dash's printf lacks \xHH).
    script = (
        f"printf x >> {counter}; "
        "printf '\\342\\234\\224 [1/3] Built Probe.Base (10ms)\\n'; "
        "if [ -e Slow.lean ]; then "
        "printf '\\342\\234\\226 [2/3] Building Slow (999ms)\\n'; sleep 30; "
        "fi"
    )
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=root,
            entrypoint=entrypoint,
            merge_validation_worktree=True,
            batched_merge=True,
            pre_merge_validation_commands=(
                _fake_lake_timeout_validation_command(tmp_path, script, 1.0),
            ),
        ),
        task_manager=_task_manager_with_tasks(seed),
    )
    good1 = await _ready_worktree_adding(
        orchestrator, seed, name="g1", task_file="002-g1.yaml", task=_named_task(seed, "g1")
    )
    bad = await _ready_worktree_adding(orchestrator, seed, name="bad", plain_file="Slow.lean")
    good2 = await _ready_worktree_adding(
        orchestrator, seed, name="g2", task_file="003-g2.yaml", task=_named_task(seed, "g2")
    )

    await orchestrator.process_merge_queue_once_for_test()

    assert orchestrator.worktrees_by_id[good1.worktree_id].state is WorktreeState.CLOSED
    assert orchestrator.worktrees_by_id[good2.worktree_id].state is WorktreeState.CLOSED
    updated_bad = orchestrator.worktrees_by_id[bad.worktree_id]
    assert updated_bad.state is WorktreeState.PENDING
    assert not (entrypoint / "Slow.lean").exists()
    # The timed-out batch, the culprit's confirming solo build (times out —
    # verify-before-bounce), then the survivors' passing re-validation.
    assert counter.read_text() == "xxx"
    assert "timed out" in updated_bad.discussion[-1].message


async def test_timed_out_build_attributed_by_exonerating_completed_modules(
    tmp_path: Path,
) -> None:
    """When the kill-time output holds no in-flight header (the common shape:
    lake dies with the build, having reported only completed jobs), attribution
    falls back to the members' touched ``.lean`` files minus the modules lake
    reported ``Built`` — deprioritizing the member whose module finished, and
    probing the remaining suspect first. Here the suspect really is the
    culprit: its confirming solo build times out and it bounces.
    """

    _require_git_and_sh()
    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    counter = tmp_path / "build-count.txt"
    _initialize_git_repo(entrypoint)
    seed = _seed_tasks_dir(entrypoint)
    script = (
        f"printf x >> {counter}; "
        "if [ -e Good.lean ]; then printf '\\342\\234\\224 [1/2] Built Good (12ms)\\n'; fi; "
        "if [ -e Slow.lean ]; then sleep 30; fi"
    )
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=root,
            entrypoint=entrypoint,
            merge_validation_worktree=True,
            batched_merge=True,
            pre_merge_validation_commands=(
                _fake_lake_timeout_validation_command(tmp_path, script, 1.0),
            ),
        ),
        task_manager=_task_manager_with_tasks(seed),
    )
    # good adds a .lean module that lake reports Built before the hang.
    good = await _ready_worktree_adding(orchestrator, seed, name="good", plain_file="Good.lean")
    bad = await _ready_worktree_adding(orchestrator, seed, name="bad", plain_file="Slow.lean")
    task_only = await _ready_worktree_adding(
        orchestrator, seed, name="task-only", task_file="002-t.yaml", task=_named_task(seed, "t")
    )

    await orchestrator.process_merge_queue_once_for_test()

    assert orchestrator.worktrees_by_id[good.worktree_id].state is WorktreeState.CLOSED
    assert orchestrator.worktrees_by_id[task_only.worktree_id].state is WorktreeState.CLOSED
    assert orchestrator.worktrees_by_id[bad.worktree_id].state is WorktreeState.PENDING
    assert not (entrypoint / "Slow.lean").exists()
    assert (entrypoint / "Good.lean").exists()
    # Three runs: the timed-out batch, the suspect's confirming solo build
    # (times out — verify-before-bounce), then the survivors' passing build.
    assert counter.read_text() == "xxx"


async def test_timed_out_built_module_does_not_exonerate_its_author(
    tmp_path: Path,
) -> None:
    """``Built`` is not innocence: the implicated bystander publishes, the
    "exonerated" author bounces.

    The culprit's ``Base.lean`` compiles fine (lake reports ``Built Base``) but
    its change hangs the rest of the build, and no in-flight header is
    buffered. Completed-module subtraction then implicates only the bystander's
    ``Unrelated.lean`` — the round-2 adversarial review's finding 2, where the
    old direct bounce evicted the bystander. Attribution is now only a probe
    order: the bystander is validated alone, passes, and publishes; the culprit
    then times out on its own confirming build and bounces.
    """

    _require_git_and_sh()
    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    counter = tmp_path / "build-count.txt"
    _initialize_git_repo(entrypoint)
    seed = _seed_tasks_dir(entrypoint)
    # Base.lean itself builds (and is reported Built) — then its change hangs
    # the rest of the build, downstream of it.
    script = (
        f"printf x >> {counter}; "
        "if [ -e Base.lean ]; then "
        "printf '\\342\\234\\224 [1/3] Built Base (10ms)\\n'; sleep 30; "
        "fi"
    )
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=root,
            entrypoint=entrypoint,
            merge_validation_worktree=True,
            batched_merge=True,
            pre_merge_validation_commands=(
                _fake_lake_timeout_validation_command(tmp_path, script, 1.0),
            ),
        ),
        task_manager=_task_manager_with_tasks(seed),
    )
    culprit = await _ready_worktree_adding(
        orchestrator, seed, name="culprit", plain_file="Base.lean"
    )
    bystander = await _ready_worktree_adding(
        orchestrator, seed, name="bystander", plain_file="Unrelated.lean"
    )

    await orchestrator.process_merge_queue_once_for_test()

    # The wrongly implicated bystander survives its confirming solo build and
    # lands; the "exonerated" culprit fails its own and bounces.
    assert orchestrator.worktrees_by_id[bystander.worktree_id].state is WorktreeState.CLOSED
    updated_culprit = orchestrator.worktrees_by_id[culprit.worktree_id]
    assert updated_culprit.state is WorktreeState.PENDING
    assert "timed out" in updated_culprit.discussion[-1].message
    assert (entrypoint / "Unrelated.lean").exists()
    assert not (entrypoint / "Base.lean").exists()
    # Three runs: the timed-out batch, the bystander's passing solo build, the
    # culprit's confirming timeout.
    assert counter.read_text() == "xxx"


async def test_timed_out_non_lake_validator_ignores_lake_shaped_output(
    tmp_path: Path,
) -> None:
    """A timed-out NON-lake validator never activates the lake output heuristics,
    even when its buffered output contains lake-shaped ``Built`` lines.

    ``pre_merge_validation_commands`` are arbitrary; here a custom validator
    prints ``Built Good`` and then hangs — both caused by the same member's
    ``Good.lean``. Completed-module subtraction would deprioritize that member
    and probe the innocent one (whose file was never "Built") first, as the
    adversarial review reproduced. With the heuristics gated on the command's
    executable being lake, attribution stays empty and plain halving isolates
    the real culprit: the innocent member publishes (issue #133 review).
    """

    _require_git_and_sh()
    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    counter = tmp_path / "build-count.txt"
    _initialize_git_repo(entrypoint)
    seed = _seed_tasks_dir(entrypoint)
    script = (
        f"printf x >> {counter}; "
        "if [ -e Good.lean ]; then printf 'Built Good\\n'; sleep 30; fi"
    )
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=root,
            entrypoint=entrypoint,
            merge_validation_worktree=True,
            batched_merge=True,
            # NOT lake: no argv token has basename ``lake``.
            pre_merge_validation_commands=(_timeout_validation_command(script, 1.0),),
        ),
        task_manager=_task_manager_with_tasks(seed),
    )
    bad = await _ready_worktree_adding(orchestrator, seed, name="bad", plain_file="Good.lean")
    innocent = await _ready_worktree_adding(
        orchestrator, seed, name="innocent", plain_file="Innocent.lean"
    )

    await orchestrator.process_merge_queue_once_for_test()

    # Halving isolates the hang's author; the innocent member (whom
    # completed-module subtraction would have probed first) lands.
    assert orchestrator.worktrees_by_id[innocent.worktree_id].state is WorktreeState.CLOSED
    updated_bad = orchestrator.worktrees_by_id[bad.worktree_id]
    assert updated_bad.state is WorktreeState.PENDING
    assert "timed out" in updated_bad.discussion[-1].message
    assert (entrypoint / "Innocent.lean").exists()
    assert not (entrypoint / "Good.lean").exists()
    # Three runs — plain bisection: the timed-out batch, the culprit half
    # (times out again), the innocent half (passes).
    assert counter.read_text() == "xxx"


def test_is_lake_invocation_requires_lake_as_the_executable() -> None:
    """The gate passes iff the command's executable is lake — directly, by
    absolute path, or behind known wrapper executables. A mere *argument* named
    ``lake`` does not match, and neither does shell-wrapped lake (which safely
    degrades to bisection)."""

    is_lake = _orchestrator_module._is_lake_invocation  # pyright: ignore[reportPrivateUsage]
    assert is_lake(("lake", "build"))
    assert is_lake(("/usr/local/bin/lake", "build", "CFT"))
    assert is_lake(("taskset", "-c", "0-31", "lake", "build"))
    assert is_lake(("env", "FOO=1", "lake", "build"))
    assert is_lake(("nice", "-n", "10", "taskset", "-c", "0-31", "/opt/elan/bin/lake", "build"))
    assert not is_lake(())
    assert not is_lake(("make", "check"))
    assert not is_lake(("flake8", "src"))
    # An argument named lake is not an executable.
    assert not is_lake(("validator", "--config", "lake"))
    # Shell-wrapped lake is not recognized; the miss means plain bisection.
    assert not is_lake(("sh", "-c", "lake build"))


def test_lake_module_progress_separates_in_flight_from_completed() -> None:
    """The parser tolerates every observed lake job-line shape (status marks,
    ``[i/n]`` counters, timings, old counter-first start lines) and ignores
    non-job output; a module both Building and Built is not in flight.
    """

    failure = _orchestrator_module._ValidationCommandFailure(  # pyright: ignore[reportPrivateUsage]
        argv=("lake", "build"),
        returncode=None,
        stdout=(
            "info: toolchain not updated; already up-to-date\n"
            "✔ [0/9] Ran job computation\n"
            "✔ [2/9] Built Probe.Fast (208ms)\n"
            "⚠ [3/9] Replayed RiemannSurface.PartII.Foo\n"
            # Failed/killed job header (current lake) — in flight.
            "✖ [4/9] Building Probe.Slow (186ms)\n"
            "error: Lean exited with code -15\n"
            "Some required targets logged failures:\n"
            "- Probe.Slow\n"
            # Old-style counter-first job-start line — in flight.
            "[123/456] Building Mathlib.Data.List.Basic\n"
            # Started then completed — NOT in flight.
            "Building CFT.Cup.X\n"
            "✔ [5/9] Built CFT.Cup.X (1.2s)\n"
            "trace: .> LEAN_PATH=x lean Probe/Fast.lean\n"
        ),
        error="validation command timed out after 60s",
        timed_out=True,
    )

    in_flight, completed = _orchestrator_module._lake_module_progress(failure)  # pyright: ignore[reportPrivateUsage]

    assert in_flight == {"Probe/Slow.lean", "Mathlib/Data/List/Basic.lean"}
    assert completed == {
        "Probe/Fast.lean",
        "RiemannSurface/PartII/Foo.lean",
        "CFT/Cup/X.lean",
    }


def test_lake_module_progress_yields_nothing_for_non_lake_output() -> None:
    """No lake job lines -> both sets empty; the caller falls back to halving."""

    failure = _orchestrator_module._ValidationCommandFailure(  # pyright: ignore[reportPrivateUsage]
        argv=("make", "check"),
        returncode=None,
        stdout="checking...\nstill checking...\n",
        stderr="make: *** wait: Interrupted system call.\n",
        error="validation command timed out after 60s",
        timed_out=True,
    )

    in_flight, completed = _orchestrator_module._lake_module_progress(failure)  # pyright: ignore[reportPrivateUsage]

    assert in_flight == set()
    assert completed == set()


def test_failed_lean_files_extracts_error_paths() -> None:
    failure = _orchestrator_module._ValidationCommandFailure(  # pyright: ignore[reportPrivateUsage]
        argv=("lake", "build"),
        returncode=1,
        stdout=(
            "⚠ [8477/8618] Replayed RiemannSurface.PartII.Foo\n"
            # warning line (severity-first) — must be IGNORED
            "warning: RiemannSurface/PartII/Foo.lean:79:32: try 'simp' instead of 'simpa'\n"
            # lake's severity-first error form
            "error: RiemannSurface/PartV/Bar.lean:12:7: unsolved goals\n"
        ),
        # lean's position-first error form
        stderr="./Baz.lean:1:1: error: unknown identifier 'x'\n",
    )
    assert _orchestrator_module._failed_lean_files(failure) == {  # pyright: ignore[reportPrivateUsage]
        "RiemannSurface/PartV/Bar.lean",
        "Baz.lean",
    }


def test_paths_match_tolerates_prefixes_but_not_different_files() -> None:
    match = _orchestrator_module._paths_match  # pyright: ignore[reportPrivateUsage]
    assert match("tasks/x.yaml", "/abs/staging/tasks/x.yaml")
    assert match("./RiemannSurface/A.lean", "RiemannSurface/A.lean")
    assert match("A.lean", "RiemannSurface/A.lean")
    assert not match("A.lean", "B.lean")
    assert not match("foo/A.lean", "bar/A.lean")
