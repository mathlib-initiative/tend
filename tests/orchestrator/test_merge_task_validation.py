"""Tests for the post-merge task-tree validation gate.

A merge whose post-merge ``tasks/`` tree fails to parse strictly or whose
dependency graph is malformed (cycle, unknown dep, complete-depends-on-open)
is rejected exactly like a post-merge build failure: the entrypoint is reset
to the pre-merge HEAD and the worktree returns to PENDING with a discussion
message naming the offending file.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pytest
from tests.orchestrator import test_orchestrator as _test_orchestrator

import tend.orchestrator.orchestrator as orchestrator_module
from tend.orchestrator.config import (
    AsyncOrchestratorConfig,
    AsyncOrchestratorValidationCommandConfig,
)
from tend.orchestrator.state import (
    AsyncOrchestratorAgentRole,
    WorktreeState,
)
from tend.orchestrator.task_io import task_directory, write_task
from tend.orchestrator.task_manager import TaskManager
from tend.orchestrator.task_validation import (
    TaskValidationFailure,
    validate_task_directory,
)
from tend.orchestrator.tasks import Task, TaskStatus

WorktreeTestingOrchestrator = _test_orchestrator.WorktreeTestingOrchestrator
_initialize_git_repo = _test_orchestrator._initialize_git_repo  # pyright: ignore[reportPrivateUsage]
_run_git = _test_orchestrator._run_git  # pyright: ignore[reportPrivateUsage]
_commit_worktree = _test_orchestrator._commit_worktree  # pyright: ignore[reportPrivateUsage]
_check_post_merge_task_tree = (
    orchestrator_module._check_post_merge_task_tree  # pyright: ignore[reportPrivateUsage]
)
_merge_touched_task_directory = (
    orchestrator_module._merge_touched_task_directory  # pyright: ignore[reportPrivateUsage]
)
_rollback_entrypoint_to_head = (
    orchestrator_module._rollback_entrypoint_to_head  # pyright: ignore[reportPrivateUsage]
)


def _task_manager_with_tasks(*tasks: Task) -> TaskManager:
    return TaskManager(tasks=list(tasks))


def _initial_tasks_dir_with_task(entrypoint: Path) -> Task:
    """Seed the entrypoint's ``tasks/`` directory with one valid task.

    Returns the seed task so callers can reference its id when constructing
    follow-up tasks in worktree branches.
    """

    seed = Task(
        id="task-seed",
        title="Seed",
        summary="Seed task",
        description="Seed task to keep tasks/ non-empty.",
    )
    tasks_dir = task_directory(entrypoint)
    write_task(tasks_dir / "001-seed.yaml", seed)
    _run_git(entrypoint, "add", "tasks")
    _run_git(entrypoint, "commit", "-m", "seed tasks dir")
    return seed


async def test_merge_with_valid_post_merge_tasks_succeeds(tmp_path: Path) -> None:
    """Regression: a merge that adds well-formed task YAML still closes cleanly."""

    if shutil.which("git") is None:
        pytest.skip("git executable is not available")

    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    _initialize_git_repo(entrypoint)
    seed = _initial_tasks_dir_with_task(entrypoint)
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(root=root, entrypoint=entrypoint),
        task_manager=_task_manager_with_tasks(seed),
    )
    worktree = await orchestrator.create_fresh_worktree_for_test(name="adds-task", task=seed)
    # The worker added a new well-formed task that depends on the seed.
    follow_up = Task(
        id="task-follow-up",
        title="Follow up",
        summary="Follow-up task",
        description="A follow-up task.",
        depends_on=[seed.id],
    )
    write_task(task_directory(worktree.path) / "002-follow-up.yaml", follow_up)
    _commit_worktree(worktree.path)
    await orchestrator.transition_worktree_for_test(worktree.worktree_id, WorktreeState.MERGE)

    await orchestrator.process_merge_queue_once_for_test()

    assert orchestrator.merge_queue == ()
    assert orchestrator.worker_queue == ()
    assert orchestrator.worktrees_by_id[worktree.worktree_id].state is WorktreeState.CLOSED
    # The follow-up task is now in the entrypoint tasks/ directory.
    follow_up_path = task_directory(entrypoint) / "002-follow-up.yaml"
    assert follow_up_path.exists()


def test_invalid_utf8_task_file_is_returned_as_validation_failure(
    tmp_path: Path,
) -> None:
    """Undecodable task contents are a per-file failure, not a gate crash."""

    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()
    bad_file = tasks_dir / "bad.yaml"
    bad_file.write_bytes(b"\xff")

    failure = validate_task_directory(tasks_dir)

    assert failure is not None
    assert failure.offending_paths == (str(bad_file),)
    assert "failed to parse" in failure.summary


async def test_merge_with_malformed_yaml_task_rolls_back_and_requeues(
    tmp_path: Path,
) -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable is not available")

    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    _initialize_git_repo(entrypoint)
    seed = _initial_tasks_dir_with_task(entrypoint)
    original_head = _run_git(entrypoint, "rev-parse", "HEAD")
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(root=root, entrypoint=entrypoint),
        task_manager=_task_manager_with_tasks(seed),
    )
    worktree = await orchestrator.create_fresh_worktree_for_test(name="bad-yaml", task=seed)
    # A worker wrote a syntactically broken YAML task file.
    bad_file = task_directory(worktree.path) / "002-malformed.yaml"
    bad_file.parent.mkdir(parents=True, exist_ok=True)
    bad_file.write_text("id: task-broken\n  bad: : indentation\n", encoding="utf-8")
    _commit_worktree(worktree.path)
    await orchestrator.transition_worktree_for_test(worktree.worktree_id, WorktreeState.MERGE)

    await orchestrator.process_merge_queue_once_for_test()

    # The entrypoint is rolled back to exactly the pre-merge HEAD (merge --abort
    # + reset --hard), so the bad file is gone and the tree is clean.
    assert _run_git(entrypoint, "rev-parse", "HEAD") == original_head
    assert _run_git(entrypoint, "status", "--porcelain") == ""
    assert not (task_directory(entrypoint) / "002-malformed.yaml").exists()
    # The worktree is returned to PENDING with a task-validation discussion message.
    assert orchestrator.merge_queue == ()
    assert orchestrator.worker_queue == (worktree.worktree_id,)
    updated = orchestrator.worktrees_by_id[worktree.worktree_id]
    assert updated.state is WorktreeState.PENDING
    assert updated.discussion[-1].role is AsyncOrchestratorAgentRole.ORCHESTRATOR
    message = updated.discussion[-1].message
    assert "Post-merge task validation failed" in message
    # The discussion message names the offending file so the worker knows what to fix.
    assert "002-malformed.yaml" in message
    discussion_log = worktree.path / ".tend" / "discussion.md"
    assert "Post-merge task validation failed" in discussion_log.read_text(encoding="utf-8")


async def test_merge_with_dependency_cycle_rolls_back_and_requeues(tmp_path: Path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable is not available")

    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    _initialize_git_repo(entrypoint)
    seed = _initial_tasks_dir_with_task(entrypoint)
    original_head = _run_git(entrypoint, "rev-parse", "HEAD")
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(root=root, entrypoint=entrypoint),
        task_manager=_task_manager_with_tasks(seed),
    )
    worktree = await orchestrator.create_fresh_worktree_for_test(name="cyclic-tasks", task=seed)
    # The worker added two new tasks that depend on each other (a cycle).
    cyclic_a = Task(
        id="task-cyc-a",
        title="Cyclic A",
        summary="Cyclic A",
        description="Cyclic A.",
        depends_on=["task-cyc-b"],
    )
    cyclic_b = Task(
        id="task-cyc-b",
        title="Cyclic B",
        summary="Cyclic B",
        description="Cyclic B.",
        depends_on=["task-cyc-a"],
    )
    write_task(task_directory(worktree.path) / "010-cyclic-a.yaml", cyclic_a)
    write_task(task_directory(worktree.path) / "011-cyclic-b.yaml", cyclic_b)
    _commit_worktree(worktree.path)
    await orchestrator.transition_worktree_for_test(worktree.worktree_id, WorktreeState.MERGE)

    await orchestrator.process_merge_queue_once_for_test()

    # Rollback was clean: entrypoint HEAD is byte-identical to pre-merge.
    assert _run_git(entrypoint, "rev-parse", "HEAD") == original_head
    assert _run_git(entrypoint, "status", "--porcelain") == ""
    assert not (task_directory(entrypoint) / "010-cyclic-a.yaml").exists()
    assert not (task_directory(entrypoint) / "011-cyclic-b.yaml").exists()
    # And the worktree is back in PENDING with a task-validation discussion message
    # that names the cycle.
    assert orchestrator.worker_queue == (worktree.worktree_id,)
    updated = orchestrator.worktrees_by_id[worktree.worktree_id]
    assert updated.state is WorktreeState.PENDING
    message = updated.discussion[-1].message
    assert "Post-merge task validation failed" in message
    assert "cycle" in message


async def test_merge_not_touching_tasks_skips_validation(tmp_path: Path) -> None:
    """A pre-existing cycle in tasks/ must not block a merge that ignores tasks/.

    Touch detection mirrors the sync PR: only merges that touch the task
    directory arm the gate. This keeps an unrelated contribution from being
    blamed for a pre-existing problem under ``tasks/``.
    """

    if shutil.which("git") is None:
        pytest.skip("git executable is not available")

    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    _initialize_git_repo(entrypoint)
    seed = _initial_tasks_dir_with_task(entrypoint)
    # Commit a pre-existing cycle into the entrypoint's tasks/ directory before
    # the worktree branches off, so both branches share it and the merge does
    # not touch tasks/.
    cyclic_a = Task(
        id="pre-cyc-a",
        title="Pre Cyc A",
        summary="Pre Cyc A",
        description="Pre cyclic A.",
        depends_on=["pre-cyc-b"],
    )
    cyclic_b = Task(
        id="pre-cyc-b",
        title="Pre Cyc B",
        summary="Pre Cyc B",
        description="Pre cyclic B.",
        depends_on=["pre-cyc-a"],
    )
    write_task(task_directory(entrypoint) / "020-pre-cyc-a.yaml", cyclic_a)
    write_task(task_directory(entrypoint) / "021-pre-cyc-b.yaml", cyclic_b)
    _run_git(entrypoint, "add", "tasks")
    _run_git(entrypoint, "commit", "-m", "introduce pre-existing cycle")

    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(root=root, entrypoint=entrypoint),
        task_manager=_task_manager_with_tasks(seed),
    )
    worktree = await orchestrator.create_fresh_worktree_for_test(name="no-tasks-touch", task=seed)
    # The worktree only modifies source files; it does not touch tasks/.
    (worktree.path / "worker-output.txt").write_text("done\n", encoding="utf-8")
    _commit_worktree(worktree.path)
    await orchestrator.transition_worktree_for_test(worktree.worktree_id, WorktreeState.MERGE)

    await orchestrator.process_merge_queue_once_for_test()

    # The pre-existing cycle did not block the merge: the gate skipped because
    # tasks/ was not touched.
    assert orchestrator.merge_queue == ()
    assert orchestrator.worker_queue == ()
    assert orchestrator.worktrees_by_id[worktree.worktree_id].state is WorktreeState.CLOSED
    assert (entrypoint / "worker-output.txt").read_text(encoding="utf-8") == "done\n"


async def test_merge_with_bad_tasks_rolls_back_before_pre_merge_validation_runs(
    tmp_path: Path,
) -> None:
    """The task gate fires before configured pre_merge_validation_commands.

    A merge that introduces a bad task tree must be rolled back before the
    project's configured pre-merge validation
    runs, so a failing task gate does not waste CI time and the validation
    command never sees a half-merged tree.
    """

    if shutil.which("git") is None:
        pytest.skip("git executable is not available")
    if shutil.which("sh") is None:
        pytest.skip("sh executable is required")

    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    validation_marker = tmp_path / "validation-ran.txt"
    _initialize_git_repo(entrypoint)
    seed = _initial_tasks_dir_with_task(entrypoint)
    original_head = _run_git(entrypoint, "rev-parse", "HEAD")
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=root,
            entrypoint=entrypoint,
            pre_merge_validation_commands=(
                AsyncOrchestratorValidationCommandConfig(
                    argv=(
                        "sh",
                        "-c",
                        'printf ran > "$1"',
                        "validator",
                        str(validation_marker),
                    ),
                ),
            ),
        ),
        task_manager=_task_manager_with_tasks(seed),
    )
    worktree = await orchestrator.create_fresh_worktree_for_test(
        name="bad-then-validate",
        task=seed,
    )
    # A worker added two new tasks that depend on each other (a cycle).
    cyclic_a = Task(
        id="ord-cyc-a",
        title="Ord Cyc A",
        summary="Ord Cyc A",
        description="Ord cyc A.",
        depends_on=["ord-cyc-b"],
    )
    cyclic_b = Task(
        id="ord-cyc-b",
        title="Ord Cyc B",
        summary="Ord Cyc B",
        description="Ord cyc B.",
        depends_on=["ord-cyc-a"],
    )
    write_task(task_directory(worktree.path) / "030-ord-cyc-a.yaml", cyclic_a)
    write_task(task_directory(worktree.path) / "031-ord-cyc-b.yaml", cyclic_b)
    _commit_worktree(worktree.path)
    await orchestrator.transition_worktree_for_test(worktree.worktree_id, WorktreeState.MERGE)

    await orchestrator.process_merge_queue_once_for_test()

    # The merge was rolled back by the task gate; the validation command never ran.
    assert _run_git(entrypoint, "rev-parse", "HEAD") == original_head
    assert not validation_marker.exists()
    updated = orchestrator.worktrees_by_id[worktree.worktree_id]
    assert updated.state is WorktreeState.PENDING
    assert "Post-merge task validation failed" in updated.discussion[-1].message


async def test_merge_with_invalid_task_field_rolls_back_and_names_file(
    tmp_path: Path,
) -> None:
    """A YAML-parsable task file whose fields fail validation is also rejected."""

    if shutil.which("git") is None:
        pytest.skip("git executable is not available")

    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    _initialize_git_repo(entrypoint)
    seed = _initial_tasks_dir_with_task(entrypoint)
    original_head = _run_git(entrypoint, "rev-parse", "HEAD")
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(root=root, entrypoint=entrypoint),
        task_manager=_task_manager_with_tasks(seed),
    )
    worktree = await orchestrator.create_fresh_worktree_for_test(name="bad-fields", task=seed)
    # Parses as YAML but is missing required Task fields (title, summary, etc.).
    incomplete = task_directory(worktree.path) / "040-incomplete.yaml"
    incomplete.parent.mkdir(parents=True, exist_ok=True)
    incomplete.write_text("id: task-incomplete\n", encoding="utf-8")
    _commit_worktree(worktree.path)
    await orchestrator.transition_worktree_for_test(worktree.worktree_id, WorktreeState.MERGE)

    await orchestrator.process_merge_queue_once_for_test()

    assert _run_git(entrypoint, "rev-parse", "HEAD") == original_head
    assert _run_git(entrypoint, "status", "--porcelain") == ""
    updated = orchestrator.worktrees_by_id[worktree.worktree_id]
    assert updated.state is WorktreeState.PENDING
    assert "Post-merge task validation failed" in updated.discussion[-1].message
    assert "040-incomplete.yaml" in updated.discussion[-1].message


def test_task_failure_discussion_bounds_offending_path_rendering() -> None:
    """Worker discussions enumerate only a bounded sample of large path sets."""

    class GuardedPaths:
        yielded = 0

        def __len__(self) -> int:
            return 100

        def __iter__(self) -> Iterator[str]:
            for index in range(100):
                self.yielded += 1
                if self.yielded > 31:
                    raise AssertionError("discussion enumerated past cap + 1")
                yield f"tasks/{index:03d}.yaml"

    guarded_paths = GuardedPaths()
    failure = TaskValidationFailure(
        summary="large graph",
        detail="large graph detail",
        offending_paths=cast(tuple[str, ...], guarded_paths),
    )

    message = orchestrator_module._task_validation_failure_discussion_message(  # pyright: ignore[reportPrivateUsage]
        failure,
        rollback_failure=None,
        original_head="deadbeef",
        staged=True,
    )

    assert guarded_paths.yielded == 31
    assert len(message) < 10_000
    assert "tasks/029.yaml" in message
    assert "tasks/030.yaml" not in message
    assert "... and 70 more" in message


def test_rollback_is_byte_identical_to_pre_merge_state(tmp_path: Path) -> None:
    """The rollback helper used by the gate is ``merge --abort`` + ``reset --hard``."""

    if shutil.which("git") is None:
        pytest.skip("git executable is not available")

    entrypoint = tmp_path / "repo"
    _initialize_git_repo(entrypoint)
    # Capture a clean pre-merge snapshot.
    pre_merge_head = _run_git(entrypoint, "rev-parse", "HEAD")
    pre_merge_tree = _run_git(entrypoint, "rev-parse", "HEAD^{tree}")
    # Make a commit that "would have been" merged.
    (entrypoint / "extra.txt").write_text("post-merge\n", encoding="utf-8")
    _run_git(entrypoint, "add", "extra.txt")
    _run_git(entrypoint, "commit", "-m", "would-be merge result")
    # And leave the worktree dirty too, to mimic a half-applied merge.
    (entrypoint / "extra.txt").write_text("dirty\n", encoding="utf-8")

    _rollback_entrypoint_to_head(entrypoint, pre_merge_head)

    # HEAD and tree match the pre-merge snapshot exactly.
    assert _run_git(entrypoint, "rev-parse", "HEAD") == pre_merge_head
    assert _run_git(entrypoint, "rev-parse", "HEAD^{tree}") == pre_merge_tree
    assert _run_git(entrypoint, "status", "--porcelain") == ""
    assert not (entrypoint / "extra.txt").exists()


def test_check_post_merge_task_tree_returns_failure_on_cycle(tmp_path: Path) -> None:
    """Unit-level: the gate function returns a typed failure for a cyclic tree."""

    if shutil.which("git") is None:
        pytest.skip("git executable is not available")

    entrypoint = tmp_path / "repo"
    _initialize_git_repo(entrypoint)
    # Pre-merge state has no tasks; the merge "introduces" a cyclic pair.
    original_head = _run_git(entrypoint, "rev-parse", "HEAD")
    cyclic_a = Task(
        id="u-cyc-a",
        title="U Cyc A",
        summary="U Cyc A",
        description="U cyc A.",
        depends_on=["u-cyc-b"],
    )
    cyclic_b = Task(
        id="u-cyc-b",
        title="U Cyc B",
        summary="U Cyc B",
        description="U cyc B.",
        depends_on=["u-cyc-a"],
    )
    write_task(task_directory(entrypoint) / "050-u-cyc-a.yaml", cyclic_a)
    write_task(task_directory(entrypoint) / "051-u-cyc-b.yaml", cyclic_b)
    _run_git(entrypoint, "add", "tasks")
    _run_git(entrypoint, "commit", "-m", "introduce cycle")

    failure = _check_post_merge_task_tree(
        entrypoint=entrypoint,
        original_head=original_head,
    )

    assert failure is not None
    assert "cycle" in failure.summary
    assert "cycle" in failure.detail
    # The failure names every file in the cycle so the batched merge can
    # attribute it to the contributing worktree(s) instead of bisecting (#128).
    assert tuple(Path(p).name for p in failure.offending_paths) == (
        "050-u-cyc-a.yaml",
        "051-u-cyc-b.yaml",
    )


def test_check_post_merge_task_tree_names_complete_depends_on_open_files(
    tmp_path: Path,
) -> None:
    """Both files of a complete-depends-on-open pair are named as offending."""

    if shutil.which("git") is None:
        pytest.skip("git executable is not available")

    entrypoint = tmp_path / "repo"
    _initialize_git_repo(entrypoint)
    original_head = _run_git(entrypoint, "rev-parse", "HEAD")
    open_task = Task(
        id="u-open",
        title="U Open",
        summary="U Open",
        description="U open.",
    )
    complete_task = Task(
        id="u-complete",
        title="U Complete",
        summary="U Complete",
        description="U complete.",
        status=TaskStatus.COMPLETE,
        depends_on=["u-open"],
    )
    write_task(task_directory(entrypoint) / "070-u-open.yaml", open_task)
    write_task(task_directory(entrypoint) / "071-u-complete.yaml", complete_task)
    _run_git(entrypoint, "add", "tasks")
    _run_git(entrypoint, "commit", "-m", "introduce complete-depends-on-open")

    failure = _check_post_merge_task_tree(
        entrypoint=entrypoint,
        original_head=original_head,
    )

    assert failure is not None
    assert "complete task cannot depend on open task" in failure.summary
    # Ordered as the error names them: the complete task first, then its open
    # dependency.
    assert tuple(Path(p).name for p in failure.offending_paths) == (
        "071-u-complete.yaml",
        "070-u-open.yaml",
    )


def test_check_post_merge_task_tree_names_every_file_on_transitive_open_path(
    tmp_path: Path,
) -> None:
    """A transitive complete→open violation names the intermediate file too.

    The causative edit may be the middle edge of ``parent -> middle -> open``
    (round-2 adversarial review of issue #128): endpoint-only reporting would
    map to two files nobody touched and leave the merge gate unable to
    attribute the failure.
    """

    if shutil.which("git") is None:
        pytest.skip("git executable is not available")

    entrypoint = tmp_path / "repo"
    _initialize_git_repo(entrypoint)
    original_head = _run_git(entrypoint, "rev-parse", "HEAD")
    open_task = Task(id="u-open", title="U Open", summary="U Open", description="U open.")
    middle = Task(
        id="u-middle",
        title="U Middle",
        summary="U Middle",
        description="U middle.",
        status=TaskStatus.COMPLETE,
        depends_on=["u-open"],
    )
    parent = Task(
        id="u-parent",
        title="U Parent",
        summary="U Parent",
        description="U parent.",
        status=TaskStatus.COMPLETE,
        depends_on=["u-middle"],
    )
    write_task(task_directory(entrypoint) / "080-u-parent.yaml", parent)
    write_task(task_directory(entrypoint) / "081-u-middle.yaml", middle)
    write_task(task_directory(entrypoint) / "082-u-open.yaml", open_task)
    _run_git(entrypoint, "add", "tasks")
    _run_git(entrypoint, "commit", "-m", "introduce transitive complete-depends-on-open")

    failure = _check_post_merge_task_tree(
        entrypoint=entrypoint,
        original_head=original_head,
    )

    assert failure is not None
    assert "complete task cannot depend on open task" in failure.summary
    # Every file along the dependency path, in path order.
    assert tuple(Path(p).name for p in failure.offending_paths) == (
        "080-u-parent.yaml",
        "081-u-middle.yaml",
        "082-u-open.yaml",
    )


def test_check_post_merge_task_tree_names_deleted_dependency_file(tmp_path: Path) -> None:
    """An unknown-dependency failure caused by *deleting* the declaring file
    names both the depender's (untouched) file and the deleted pre-merge path.

    The missing id has no declaring file in the post-merge tree, so the gate
    resolves it against the pre-merge tree at ``original_head`` — the batched
    merge can then attribute the failure to the member whose diff deleted that
    path instead of bisecting (adversarial review of issue #128).
    """

    if shutil.which("git") is None:
        pytest.skip("git executable is not available")

    entrypoint = tmp_path / "repo"
    _initialize_git_repo(entrypoint)
    dep_task = Task(
        id="u-dep",
        title="U Dep",
        summary="U Dep",
        description="U dep.",
    )
    depender = Task(
        id="u-a",
        title="U A",
        summary="U A",
        description="U a.",
        depends_on=["u-dep"],
    )
    write_task(task_directory(entrypoint) / "060-u-dep.yaml", dep_task)
    write_task(task_directory(entrypoint) / "061-u-a.yaml", depender)
    _run_git(entrypoint, "add", "tasks")
    _run_git(entrypoint, "commit", "-m", "seed depender and dependency")
    original_head = _run_git(entrypoint, "rev-parse", "HEAD")
    # The "merge" deletes the dependency's declaring file out from under u-a.
    _run_git(entrypoint, "rm", "tasks/060-u-dep.yaml")
    _run_git(entrypoint, "commit", "-m", "delete dependency task file")

    failure = _check_post_merge_task_tree(
        entrypoint=entrypoint,
        original_head=original_head,
    )

    assert failure is not None
    assert "unknown task id" in failure.summary
    # The depender's file (from the post-merge tree), then the deleted file
    # (resolved from the pre-merge tree).
    assert tuple(Path(p).name for p in failure.offending_paths) == (
        "061-u-a.yaml",
        "060-u-dep.yaml",
    )


def test_check_post_merge_task_tree_returns_none_when_tasks_untouched(
    tmp_path: Path,
) -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable is not available")

    entrypoint = tmp_path / "repo"
    _initialize_git_repo(entrypoint)
    # Seed a pre-existing cyclic tasks/ tree before the "pre-merge" snapshot.
    cyclic_a = Task(
        id="s-cyc-a",
        title="S Cyc A",
        summary="S Cyc A",
        description="S cyc A.",
        depends_on=["s-cyc-b"],
    )
    cyclic_b = Task(
        id="s-cyc-b",
        title="S Cyc B",
        summary="S Cyc B",
        description="S cyc B.",
        depends_on=["s-cyc-a"],
    )
    write_task(task_directory(entrypoint) / "060-s-cyc-a.yaml", cyclic_a)
    write_task(task_directory(entrypoint) / "061-s-cyc-b.yaml", cyclic_b)
    _run_git(entrypoint, "add", "tasks")
    _run_git(entrypoint, "commit", "-m", "seed cycle")
    original_head = _run_git(entrypoint, "rev-parse", "HEAD")
    # The "merge" only changed source files, not tasks/.
    (entrypoint / "src.txt").write_text("hello\n", encoding="utf-8")
    _run_git(entrypoint, "add", "src.txt")
    _run_git(entrypoint, "commit", "-m", "non-task change")

    failure = _check_post_merge_task_tree(
        entrypoint=entrypoint,
        original_head=original_head,
    )

    # The pre-existing cycle is ignored because the merge did not touch tasks/.
    assert failure is None


def test_merge_touched_task_directory_detects_nested_changes(tmp_path: Path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable is not available")

    entrypoint = tmp_path / "repo"
    _initialize_git_repo(entrypoint)
    original_head = _run_git(entrypoint, "rev-parse", "HEAD")
    nested = task_directory(entrypoint) / "nested"
    nested.mkdir(parents=True)
    (nested / "deep.yaml").write_text("placeholder\n", encoding="utf-8")
    _run_git(entrypoint, "add", "tasks")
    _run_git(entrypoint, "commit", "-m", "nested task change")

    assert _merge_touched_task_directory(
        entrypoint=entrypoint,
        original_head=original_head,
    )


def test_merge_touched_task_directory_ignores_sibling_prefix(tmp_path: Path) -> None:
    """A directory that shares a name prefix with tasks/ must not arm the gate."""

    if shutil.which("git") is None:
        pytest.skip("git executable is not available")

    entrypoint = tmp_path / "repo"
    _initialize_git_repo(entrypoint)
    original_head = _run_git(entrypoint, "rev-parse", "HEAD")
    # A sibling that starts with "tasks" must NOT trigger the gate.
    sibling = entrypoint / "tasks-extra"
    sibling.mkdir()
    (sibling / "note.txt").write_text("hi\n", encoding="utf-8")
    _run_git(entrypoint, "add", "tasks-extra")
    _run_git(entrypoint, "commit", "-m", "sibling change")

    assert not _merge_touched_task_directory(
        entrypoint=entrypoint,
        original_head=original_head,
    )
