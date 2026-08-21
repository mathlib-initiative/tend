"""Tests for the optional task-only build skip (issue #118).

``skip_build_validation_for_task_only_merges`` lets an approved merge (or
assembled staging batch) whose entire diff stays under ``tasks/`` skip the
expensive ``pre_merge_validation_commands`` gate. The build-free post-merge
task-tree gate still runs first and must pass; a merge touching any non-task
path builds exactly as before, and the option is off by default.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

import pytest
from tests.orchestrator import test_orchestrator as _test_orchestrator

from tend.orchestrator import orchestrator as _orchestrator_module
from tend.orchestrator.config import (
    AsyncOrchestratorConfig,
    AsyncOrchestratorValidationCommandConfig,
)
from tend.orchestrator.state import AsyncOrchestratorWorktree, WorktreeState
from tend.orchestrator.task_io import task_directory, write_task
from tend.orchestrator.task_manager import TaskManager
from tend.orchestrator.tasks import Task

WorktreeTestingOrchestrator = _test_orchestrator.WorktreeTestingOrchestrator
_initialize_git_repo = _test_orchestrator._initialize_git_repo  # pyright: ignore[reportPrivateUsage]
_run_git = _test_orchestrator._run_git  # pyright: ignore[reportPrivateUsage]
_commit_worktree = _test_orchestrator._commit_worktree  # pyright: ignore[reportPrivateUsage]
_merge_changed_only_task_paths = (
    _orchestrator_module._merge_changed_only_task_paths  # pyright: ignore[reportPrivateUsage]
)
_merge_touched_task_directory = (
    _orchestrator_module._merge_touched_task_directory  # pyright: ignore[reportPrivateUsage]
)


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


def _named_task(seed: Task, name: str) -> Task:
    return Task(
        id=f"task-{name}",
        title=name,
        summary=f"{name} task",
        description=f"{name} task.",
        depends_on=[seed.id],
    )


def _validation_command(script: str, *args: str) -> AsyncOrchestratorValidationCommandConfig:
    return AsyncOrchestratorValidationCommandConfig(argv=("sh", "-c", script, "validator", *args))


def _build_counter_command(counter: Path) -> AsyncOrchestratorValidationCommandConfig:
    """Build-gate stand-in that appends one ``x`` to ``counter`` per run."""

    return _validation_command(f"printf x >> {counter}")


def _require_git_and_sh() -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable is not available")
    if shutil.which("sh") is None:
        pytest.skip("sh executable is required")


async def _ready_worktree_adding(
    orchestrator: WorktreeTestingOrchestrator,
    seed: Task,
    *,
    name: str,
    task_file: str | None = None,
    task: Task | None = None,
    plain_file: str | None = None,
    raw_task_file: tuple[str, str] | None = None,
) -> AsyncOrchestratorWorktree:
    """Create a worktree, write a file into it, and transition it to MERGE."""

    wt = await orchestrator.create_fresh_worktree_for_test(name=name, task=seed)
    if task_file is not None and task is not None:
        write_task(task_directory(wt.path) / task_file, task)
    if raw_task_file is not None:
        (task_directory(wt.path) / raw_task_file[0]).write_text(raw_task_file[1])
    if plain_file is not None:
        (wt.path / plain_file).write_text("x")
    _commit_worktree(wt.path)
    await orchestrator.transition_worktree_for_test(wt.worktree_id, WorktreeState.MERGE)
    return wt


async def test_batched_task_only_merge_skips_build_when_enabled(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A batch whose combined diff stays under tasks/ publishes without a build."""

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
            skip_build_validation_for_task_only_merges=True,
            pre_merge_validation_commands=(_build_counter_command(counter),),
        ),
        task_manager=_task_manager_with_tasks(seed),
    )
    a = await _ready_worktree_adding(
        orchestrator, seed, name="a", task_file="002-a.yaml", task=_named_task(seed, "a")
    )
    b = await _ready_worktree_adding(
        orchestrator, seed, name="b", task_file="003-b.yaml", task=_named_task(seed, "b")
    )

    with caplog.at_level(logging.INFO, logger="tend.orchestrator.orchestrator"):
        await orchestrator.process_merge_queue_once_for_test()

    # Both landed without any build: the task-tree gate validated the batch and
    # the configured command never ran.
    assert orchestrator.worktrees_by_id[a.worktree_id].state is WorktreeState.CLOSED
    assert orchestrator.worktrees_by_id[b.worktree_id].state is WorktreeState.CLOSED
    assert (task_directory(entrypoint) / "002-a.yaml").exists()
    assert (task_directory(entrypoint) / "003-b.yaml").exists()
    assert not counter.exists()
    # The skip is stated at INFO so merge logs show why no build ran.
    assert "skipping pre-merge build validation" in caplog.text


async def test_batched_mixed_merge_still_runs_build_when_enabled(tmp_path: Path) -> None:
    """A batch that also touches a non-task path runs the build gate as before."""

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
            skip_build_validation_for_task_only_merges=True,
            pre_merge_validation_commands=(_build_counter_command(counter),),
        ),
        task_manager=_task_manager_with_tasks(seed),
    )
    a = await _ready_worktree_adding(
        orchestrator, seed, name="a", task_file="002-a.yaml", task=_named_task(seed, "a")
    )
    b = await _ready_worktree_adding(orchestrator, seed, name="b", plain_file="Source.lean")

    await orchestrator.process_merge_queue_once_for_test()

    assert orchestrator.worktrees_by_id[a.worktree_id].state is WorktreeState.CLOSED
    assert orchestrator.worktrees_by_id[b.worktree_id].state is WorktreeState.CLOSED
    assert (entrypoint / "Source.lean").exists()
    # Exactly one build for the mixed batch.
    assert counter.read_text() == "x"


async def test_task_only_merge_runs_build_by_default(tmp_path: Path) -> None:
    """Without the opt-in, a task-only batch still runs the build gate."""

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
            pre_merge_validation_commands=(_build_counter_command(counter),),
        ),
        task_manager=_task_manager_with_tasks(seed),
    )
    a = await _ready_worktree_adding(
        orchestrator, seed, name="a", task_file="002-a.yaml", task=_named_task(seed, "a")
    )

    await orchestrator.process_merge_queue_once_for_test()

    assert orchestrator.worktrees_by_id[a.worktree_id].state is WorktreeState.CLOSED
    assert counter.read_text() == "x"


async def test_task_tree_failure_still_bounces_with_skip_enabled(tmp_path: Path) -> None:
    """The skip never bypasses gate 1: a bad task tree still bounces its worktree.

    With the opt-in enabled and an all-task-only batch, no build runs at any
    point: the malformed file is caught by the build-free task-tree gate and
    attributed, and the surviving task-only worktree then publishes under the
    skip. The build counter therefore stays untouched throughout.
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
            skip_build_validation_for_task_only_merges=True,
            pre_merge_validation_commands=(_build_counter_command(counter),),
        ),
        task_manager=_task_manager_with_tasks(seed),
    )
    good = await _ready_worktree_adding(
        orchestrator, seed, name="good", task_file="002-good.yaml", task=_named_task(seed, "good")
    )
    # Unparseable YAML (unclosed flow sequence) -> task-tree parse failure.
    bad = await _ready_worktree_adding(
        orchestrator, seed, name="bad", raw_task_file=("003-bad.yaml", "[unclosed\n")
    )

    await orchestrator.process_merge_queue_once_for_test()

    assert orchestrator.worktrees_by_id[good.worktree_id].state is WorktreeState.CLOSED
    updated_bad = orchestrator.worktrees_by_id[bad.worktree_id]
    assert updated_bad.state is WorktreeState.PENDING
    assert "Post-merge task validation failed" in updated_bad.discussion[-1].message
    assert (task_directory(entrypoint) / "002-good.yaml").exists()
    assert not (task_directory(entrypoint) / "003-bad.yaml").exists()
    assert not counter.exists()


async def test_non_ascii_malformed_task_bounces_with_skip_enabled(tmp_path: Path) -> None:
    """A non-ASCII task path must arm gate 1 before gate 2 skips the build."""

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
            skip_build_validation_for_task_only_merges=True,
            pre_merge_validation_commands=(_build_counter_command(counter),),
        ),
        task_manager=_task_manager_with_tasks(seed),
    )
    bad = await _ready_worktree_adding(
        orchestrator,
        seed,
        name="bad-non-ascii",
        raw_task_file=("bäd.yaml", "[unclosed\n"),
    )

    await orchestrator.process_merge_queue_once_for_test()

    updated_bad = orchestrator.worktrees_by_id[bad.worktree_id]
    assert updated_bad.state is WorktreeState.PENDING
    assert "Post-merge task validation failed" in updated_bad.discussion[-1].message
    assert _run_git(entrypoint, "rev-parse", "HEAD") == original_head
    assert not (task_directory(entrypoint) / "bäd.yaml").exists()
    assert not counter.exists()


@pytest.mark.skipif(os.name != "posix", reason="raw-byte filenames require POSIX")
async def test_raw_byte_malformed_task_bounces_and_batch_continues(
    tmp_path: Path,
) -> None:
    """A raw task path is escaped at persistence and cannot abort the batch."""

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
            skip_build_validation_for_task_only_merges=True,
            pre_merge_validation_commands=(_build_counter_command(counter),),
        ),
        task_manager=_task_manager_with_tasks(seed),
    )
    raw_filename = os.fsdecode(b"\xff.yaml")
    bad = await _ready_worktree_adding(
        orchestrator,
        seed,
        name="bad-raw-byte",
        raw_task_file=(raw_filename, "[unclosed\n"),
    )
    good = await _ready_worktree_adding(
        orchestrator,
        seed,
        name="good-after-raw-byte",
        task_file="002-good.yaml",
        task=_named_task(seed, "good"),
    )

    # The malformed member is isolated and bounced without allowing its
    # surrogate-bearing validation message to escape merge processing. The
    # same queue pass must continue on to publish the healthy member.
    await orchestrator.process_merge_queue_once_for_test()

    updated_bad = orchestrator.worktrees_by_id[bad.worktree_id]
    assert updated_bad.state is WorktreeState.PENDING
    message = updated_bad.discussion[-1].message
    message.encode("utf-8")
    assert "Post-merge task validation failed" in message
    assert "\\udcff.yaml" in message
    persisted = (bad.path / ".tend" / "discussion.md").read_text(encoding="utf-8")
    assert "\\udcff.yaml" in persisted

    assert orchestrator.worktrees_by_id[good.worktree_id].state is WorktreeState.CLOSED
    assert not (task_directory(entrypoint) / raw_filename).exists()
    assert (task_directory(entrypoint) / "002-good.yaml").exists()
    assert not counter.exists()


async def test_serial_staging_task_only_merge_skips_build_when_enabled(
    tmp_path: Path,
) -> None:
    """The single-worktree staging path honors the skip too."""

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
            batched_merge=False,
            skip_build_validation_for_task_only_merges=True,
            pre_merge_validation_commands=(_build_counter_command(counter),),
        ),
        task_manager=_task_manager_with_tasks(seed),
    )
    worktree = await _ready_worktree_adding(
        orchestrator, seed, name="a", task_file="002-a.yaml", task=_named_task(seed, "a")
    )

    await orchestrator.process_merge_queue_once_for_test()

    assert orchestrator.worktrees_by_id[worktree.worktree_id].state is WorktreeState.CLOSED
    assert (task_directory(entrypoint) / "002-a.yaml").exists()
    assert not counter.exists()


async def test_legacy_entrypoint_task_only_merge_skips_build_when_enabled(
    tmp_path: Path,
) -> None:
    """The legacy in-entrypoint path honors the skip too."""

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
            merge_validation_worktree=False,
            skip_build_validation_for_task_only_merges=True,
            pre_merge_validation_commands=(_build_counter_command(counter),),
        ),
        task_manager=_task_manager_with_tasks(seed),
    )
    worktree = await _ready_worktree_adding(
        orchestrator, seed, name="a", task_file="002-a.yaml", task=_named_task(seed, "a")
    )

    await orchestrator.process_merge_queue_once_for_test()

    assert orchestrator.worktrees_by_id[worktree.worktree_id].state is WorktreeState.CLOSED
    assert (task_directory(entrypoint) / "002-a.yaml").exists()
    assert not counter.exists()


async def test_legacy_entrypoint_mixed_merge_still_runs_build_when_enabled(
    tmp_path: Path,
) -> None:
    """The legacy path still builds when the merge touches a non-task path."""

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
            merge_validation_worktree=False,
            skip_build_validation_for_task_only_merges=True,
            pre_merge_validation_commands=(_build_counter_command(counter),),
        ),
        task_manager=_task_manager_with_tasks(seed),
    )
    worktree = await _ready_worktree_adding(
        orchestrator,
        seed,
        name="a",
        task_file="002-a.yaml",
        task=_named_task(seed, "a"),
        plain_file="Source.lean",
    )

    await orchestrator.process_merge_queue_once_for_test()

    assert orchestrator.worktrees_by_id[worktree.worktree_id].state is WorktreeState.CLOSED
    assert counter.read_text() == "x"


def test_merge_changed_only_task_paths_true_for_task_only_diff(tmp_path: Path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable is not available")

    entrypoint = tmp_path / "repo"
    _initialize_git_repo(entrypoint)
    original_head = _run_git(entrypoint, "rev-parse", "HEAD")
    nested = task_directory(entrypoint) / "nested"
    nested.mkdir(parents=True)
    (nested / "deep.yaml").write_text("placeholder\n", encoding="utf-8")
    _run_git(entrypoint, "add", "tasks")
    _run_git(entrypoint, "commit", "-m", "task-only change")

    assert _merge_changed_only_task_paths(
        entrypoint=entrypoint,
        original_head=original_head,
    )


def test_merge_changed_only_task_paths_false_for_mixed_diff(tmp_path: Path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable is not available")

    entrypoint = tmp_path / "repo"
    _initialize_git_repo(entrypoint)
    original_head = _run_git(entrypoint, "rev-parse", "HEAD")
    task_directory(entrypoint).mkdir(parents=True)
    (task_directory(entrypoint) / "one.yaml").write_text("placeholder\n", encoding="utf-8")
    (entrypoint / "Source.lean").write_text("def x := 1\n", encoding="utf-8")
    _run_git(entrypoint, "add", "-A")
    _run_git(entrypoint, "commit", "-m", "mixed change")

    assert not _merge_changed_only_task_paths(
        entrypoint=entrypoint,
        original_head=original_head,
    )


def test_merge_changed_only_task_paths_false_for_sibling_prefix(tmp_path: Path) -> None:
    """A directory sharing the ``tasks`` name prefix must not count as task-only."""

    if shutil.which("git") is None:
        pytest.skip("git executable is not available")

    entrypoint = tmp_path / "repo"
    _initialize_git_repo(entrypoint)
    original_head = _run_git(entrypoint, "rev-parse", "HEAD")
    sibling = entrypoint / "tasks-extra"
    sibling.mkdir()
    (sibling / "note.txt").write_text("hi\n", encoding="utf-8")
    _run_git(entrypoint, "add", "tasks-extra")
    _run_git(entrypoint, "commit", "-m", "sibling change")

    assert not _merge_changed_only_task_paths(
        entrypoint=entrypoint,
        original_head=original_head,
    )


def test_merge_changed_only_task_paths_false_for_leading_whitespace_path(
    tmp_path: Path,
) -> None:
    """A legal non-task path like " tasks/Evil.lean" must not be reclassified.

    Regression for the adversarial-review finding: stripping git's path output
    turned the non-task path " tasks/Evil.lean" (leading space, outside the
    task directory) into "tasks/Evil.lean" and skipped the build for it.
    """

    if shutil.which("git") is None:
        pytest.skip("git executable is not available")

    entrypoint = tmp_path / "repo"
    _initialize_git_repo(entrypoint)
    original_head = _run_git(entrypoint, "rev-parse", "HEAD")
    evil = entrypoint / " tasks" / "Evil.lean"
    evil.parent.mkdir(parents=True)
    evil.write_text("-- not a task file\n", encoding="utf-8")
    _run_git(entrypoint, "add", "-f", " tasks")
    _run_git(entrypoint, "commit", "-m", "non-task change in whitespace dir")

    assert not _merge_changed_only_task_paths(
        entrypoint=entrypoint,
        original_head=original_head,
    )


@pytest.mark.parametrize(
    ("relative_path", "expected"),
    [
        (b"tasks/\xff.yaml", True),
        (b"not-tasks/\xff.yaml", False),
    ],
)
def test_merge_path_detectors_handle_raw_non_utf8_paths(
    tmp_path: Path,
    relative_path: bytes,
    expected: bool,
) -> None:
    """Raw Git paths cannot crash classification or make the gates disagree."""

    if shutil.which("git") is None:
        pytest.skip("git executable is not available")

    entrypoint = tmp_path / "repo"
    _initialize_git_repo(entrypoint)
    original_head = _run_git(entrypoint, "rev-parse", "HEAD")
    changed_path = entrypoint / os.fsdecode(relative_path)
    changed_path.parent.mkdir(parents=True)
    changed_path.write_text("placeholder\n", encoding="utf-8")
    _run_git(entrypoint, "add", "-A")
    _run_git(entrypoint, "commit", "-m", "raw path change")

    task_only = _merge_changed_only_task_paths(
        entrypoint=entrypoint,
        original_head=original_head,
    )
    touches_tasks = _merge_touched_task_directory(
        entrypoint=entrypoint,
        original_head=original_head,
    )

    assert isinstance(task_only, bool)
    assert task_only is expected
    assert touches_tasks is expected


def test_merge_path_detectors_fail_in_conservative_directions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unexpected raw-path parsing failures run both validation gates."""

    def fail_to_parse_paths(*, entrypoint: Path, original_head: str) -> tuple[str, ...]:
        del entrypoint, original_head
        raise ValueError("malformed raw diff")

    monkeypatch.setattr(_orchestrator_module, "_merge_diff_paths", fail_to_parse_paths)

    assert not _merge_changed_only_task_paths(
        entrypoint=tmp_path,
        original_head="HEAD^",
    )
    assert _merge_touched_task_directory(
        entrypoint=tmp_path,
        original_head="HEAD^",
    )


def test_merge_path_detectors_agree_for_non_ascii_task_path(tmp_path: Path) -> None:
    """NUL-delimited raw parsing recognizes a Unicode task path in both gates."""

    if shutil.which("git") is None:
        pytest.skip("git executable is not available")

    entrypoint = tmp_path / "repo"
    _initialize_git_repo(entrypoint)
    original_head = _run_git(entrypoint, "rev-parse", "HEAD")
    task_directory(entrypoint).mkdir(parents=True)
    (task_directory(entrypoint) / "bäd.yaml").write_text("placeholder\n", encoding="utf-8")
    _run_git(entrypoint, "add", "tasks")
    _run_git(entrypoint, "commit", "-m", "non-ASCII task path")

    assert _merge_touched_task_directory(
        entrypoint=entrypoint,
        original_head=original_head,
    )
    assert _merge_changed_only_task_paths(
        entrypoint=entrypoint,
        original_head=original_head,
    )


def test_merge_changed_only_task_paths_false_for_empty_diff(tmp_path: Path) -> None:
    """An empty endpoint diff proves nothing task-only: the build must run.

    Regression for the adversarial-review finding: ``all`` over an empty path
    list classified a no-op merge (e.g. an empty commit) as task-only.
    """

    if shutil.which("git") is None:
        pytest.skip("git executable is not available")

    entrypoint = tmp_path / "repo"
    _initialize_git_repo(entrypoint)
    original_head = _run_git(entrypoint, "rev-parse", "HEAD")
    _run_git(entrypoint, "commit", "--allow-empty", "-m", "no-op commit")

    assert not _merge_changed_only_task_paths(
        entrypoint=entrypoint,
        original_head=original_head,
    )
