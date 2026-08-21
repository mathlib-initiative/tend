from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import signal
import subprocess
import threading
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import JsonValue
from tests.orchestrator.store_helpers import seed_store_state, worktree_ids_for_task

import tend.orchestrator.orchestrator as orchestrator_module
from tend.llm.usage import Cost, TokenUsage, Usage
from tend.orchestrator.config import (
    AsyncOrchestratorAgentCommandConfig,
    AsyncOrchestratorBudgetConfig,
    AsyncOrchestratorConfig,
    AsyncOrchestratorValidationCommandConfig,
    AsyncOrchestratorWorktreeSetupCommandConfig,
)
from tend.orchestrator.control_store import (
    AsyncOrchestratorControlStoreIOError,
    ControlActiveAgentSnapshot,
    ControlCommandName,
    ControlCommandRecord,
    ControlRunRecord,
    ControlRunStatus,
    SQLiteAsyncOrchestratorStore,
)
from tend.orchestrator.orchestrator import (
    AsyncCostBudgetCurrencyMismatchError,
    AsyncOrchestrator,
    AsyncOrchestratorBudgetStop,
    AsyncOrchestratorRunStop,
)
from tend.orchestrator.state import (
    AsyncOrchestratorAgentRole,
    AsyncOrchestratorWorktree,
    WorktreeState,
)
from tend.orchestrator.task_io import write_task
from tend.orchestrator.task_manager import TaskManager
from tend.orchestrator.tasks import Task, TaskPriority, TaskStatus


class WorktreeTestingOrchestrator(AsyncOrchestrator):
    def __init__(
        self,
        config: AsyncOrchestratorConfig,
        *,
        task_manager: TaskManager | None = None,
        worktrees: Sequence[AsyncOrchestratorWorktree] = (),
        store: SQLiteAsyncOrchestratorStore | None = None,
        check_resume_health: bool = False,
    ) -> None:
        seeded_store = SQLiteAsyncOrchestratorStore(config.root) if store is None else store
        if task_manager is not None or worktrees:
            seed_store_state(
                seeded_store,
                task_manager=task_manager,
                worktrees=worktrees,
            )
        super().__init__(
            config,
            store=seeded_store,
            check_resume_health=check_resume_health,
        )

    async def create_fresh_worktree_for_test(
        self,
        *,
        name: str | None = None,
        task: Task | str | None = None,
    ) -> AsyncOrchestratorWorktree:
        return await self._create_fresh_worktree(name=name, task=task)

    async def sync_task_manager_once_for_test(self) -> None:
        await self._sync_task_manager_once()

    async def ensure_ready_task_worktrees_once_for_test(self) -> None:
        await self._ensure_ready_task_worktrees_once()

    async def enqueue_ready_tasks_once_for_test(self) -> None:
        await self._enqueue_ready_tasks_once()

    async def process_task_queue_once_for_test(self) -> None:
        await self._process_task_queue_once()

    async def spawn_worker_agents_once_for_test(self) -> None:
        await self._spawn_worker_agents_once()

    async def spawn_reviewer_agents_once_for_test(self) -> None:
        await self._spawn_reviewer_agents_once()

    async def process_merge_queue_once_for_test(self) -> None:
        await self._process_merge_queue_once()

    async def merge_worktree_id_for_test(self, worktree_id: str) -> None:
        await self._merge_worktree_id(worktree_id)

    async def transition_worktree_for_test(
        self,
        worktree_id: str,
        state: WorktreeState,
    ) -> None:
        await self._transition_worktree(worktree_id, state)

    async def wait_for_worker_agents_for_test(self) -> None:
        await asyncio.gather(*self.runtime.worker_agent_tasks.values())

    async def wait_for_reviewer_agents_for_test(self) -> None:
        await asyncio.gather(*self.runtime.reviewer_agent_tasks.values())

    async def record_budget_stop_if_exceeded_for_test(self) -> bool:
        return await self._record_budget_stop_if_exceeded()

    async def refresh_budget_stop_accumulated_cost_for_test(self) -> None:
        await self._refresh_budget_stop_accumulated_cost()

    async def in_flight_work_settled_for_test(self) -> bool:
        return await self._in_flight_work_settled()

    def pause_scheduling_for_test(self) -> None:
        self.runtime.set_paused(True)

    def resume_scheduling_for_test(self) -> None:
        self.runtime.set_paused(False)

    def request_drain_for_test(self) -> None:
        self.runtime.request_drain()

    @property
    def run_stop_for_test(self) -> AsyncOrchestratorRunStop | None:
        return self._run_stop

    async def all_tasks_complete_once_for_test(self) -> bool:
        return await self._all_tasks_complete_once()

    async def all_tasks_complete_locked_for_test(self) -> bool:
        return await self._all_tasks_complete_locked()

    async def backfill_session_usage_once_for_test(self) -> None:
        await self._backfill_session_usage_once()

    async def ensure_worktree_for_ready_task_id_for_test(
        self, task_id: str
    ) -> AsyncOrchestratorWorktree | None:
        return await self._ensure_worktree_for_ready_task_id(task_id)

    def spawn_worker_agent_task_for_test(self, worktree_id: str) -> None:
        self._spawn_worker_agent_task(worktree_id)

    def spawn_reviewer_agent_task_for_test(self, worktree_id: str) -> None:
        self._spawn_reviewer_agent_task(worktree_id)

    @property
    def budget_stop_for_test(self) -> AsyncOrchestratorBudgetStop | None:
        return self._budget_stop

    def force_budget_stop_for_test(self, stop: AsyncOrchestratorBudgetStop) -> None:
        self._budget_stop = stop

    def set_control_run_id_for_test(self, run_id: str) -> None:
        self._control_run_id = run_id

    async def apply_control_command_for_test(
        self,
        command: ControlCommandRecord,
    ) -> None:
        await self._apply_control_command(command)

    async def record_control_heartbeat_once_for_test(self) -> None:
        await self._record_control_heartbeat_once()


async def test_create_fresh_worktree_adds_detached_worktree_under_root(tmp_path: Path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable is not available")

    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    _initialize_git_repo(entrypoint)
    source_head = _run_git(entrypoint, "rev-parse", "--verify", "HEAD")

    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(root=root, entrypoint=entrypoint),
    )

    worktree = await orchestrator.create_fresh_worktree_for_test(name="entrypoint-copy")

    assert worktree.worktree_id == "worktree_000001"
    assert worktree.path == root / "worktrees" / "entrypoint-copy"
    assert worktree.head == source_head
    assert worktree.state is WorktreeState.PENDING
    assert orchestrator.worktree_ids == (worktree.worktree_id,)
    assert orchestrator.worktree_ids == (worktree.worktree_id,)
    assert orchestrator.worker_queue == ()
    assert orchestrator.worktrees_by_id[worktree.worktree_id] == worktree
    assert orchestrator.worktrees_by_id[worktree.worktree_id] == worktree
    assert worktree.path.is_dir()
    assert _run_git(worktree.path, "rev-parse", "--show-toplevel") == str(worktree.path)
    assert _run_git(worktree.path, "rev-parse", "--verify", "HEAD") == source_head
    assert _run_git(worktree.path, "branch", "--show-current") == ""


async def test_create_fresh_worktree_runs_setup_command_with_path_placeholders(
    tmp_path: Path,
) -> None:
    if shutil.which("git") is None or shutil.which("sh") is None:
        pytest.skip("git and sh executables are required")

    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    _initialize_git_repo(entrypoint)
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=root,
            entrypoint=entrypoint,
            worktree_setup_command=AsyncOrchestratorWorktreeSetupCommandConfig(
                argv=(
                    "sh",
                    "-c",
                    'printf "%s\\n%s\\n" "$1" "$2" > "$2/setup-paths.txt"',
                    "setup",
                    "{entrypoint}",
                    "{worktree}",
                ),
            ),
        ),
    )

    worktree = await orchestrator.create_fresh_worktree_for_test(name="entrypoint-copy")

    assert (worktree.path / "setup-paths.txt").read_text(encoding="utf-8").splitlines() == [
        str(entrypoint.resolve()),
        str(worktree.path),
    ]


async def test_create_fresh_worktree_skips_stale_dirs_after_state_clear(
    tmp_path: Path,
) -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable is not available")

    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    _initialize_git_repo(entrypoint)
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(root=root, entrypoint=entrypoint),
    )

    first = await orchestrator.create_fresh_worktree_for_test()
    assert first.worktree_id == "worktree_000001"
    assert first.path.exists()

    orchestrator.store.clear_state()
    second = await orchestrator.create_fresh_worktree_for_test()

    assert second.worktree_id == "worktree_000002"
    assert second.path == root / "worktrees" / "worktree_000002"
    assert [worktree.worktree_id for worktree in orchestrator.store.list_worktrees()] == [
        "worktree_000002",
    ]
    assert (root / "worktrees" / "worktree_000001").exists()


async def test_create_fresh_worktree_cleans_up_failed_setup_command(
    tmp_path: Path,
) -> None:
    if shutil.which("git") is None or shutil.which("sh") is None:
        pytest.skip("git and sh executables are required")

    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    worktree_path = root / "worktrees" / "entrypoint-copy"
    _initialize_git_repo(entrypoint)
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=root,
            entrypoint=entrypoint,
            worktree_setup_command=AsyncOrchestratorWorktreeSetupCommandConfig(
                argv=("sh", "-c", "echo setup failed >&2; exit 3"),
            ),
        ),
    )

    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        await orchestrator.create_fresh_worktree_for_test(name="entrypoint-copy")

    assert exc_info.value.returncode == 3
    assert "setup failed" in exc_info.value.stderr
    assert not worktree_path.exists()
    assert str(worktree_path) not in _run_git(entrypoint, "worktree", "list", "--porcelain")
    assert orchestrator.worktree_ids == ()
    assert orchestrator.worker_queue == ()
    assert orchestrator.store.list_worktrees() == ()


async def test_create_fresh_worktree_cleans_up_if_store_allocation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable is not available")

    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    worktree_path = root / "worktrees" / "worktree_000001"
    _initialize_git_repo(entrypoint)
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(root=root, entrypoint=entrypoint),
    )

    def fail_allocate(
        self: SQLiteAsyncOrchestratorStore,
        *,
        task_id: str | None,
        path: object,
        head: str,
        worktree_id: str | None = None,
    ) -> str:
        del self, task_id, path, head, worktree_id
        raise RuntimeError("forced allocation failure")

    monkeypatch.setattr(SQLiteAsyncOrchestratorStore, "allocate_worktree", fail_allocate)

    with pytest.raises(RuntimeError, match="forced allocation failure"):
        await orchestrator.create_fresh_worktree_for_test()

    assert not worktree_path.exists()
    assert str(worktree_path) not in _run_git(entrypoint, "worktree", "list", "--porcelain")
    assert orchestrator.store.list_worktrees() == ()


async def test_sync_task_manager_once_loads_entrypoint_tasks(tmp_path: Path) -> None:
    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    task = Task(id="task-1", title="Task", summary="Task", description="Do it.")
    write_task(entrypoint / "tasks" / "task-1.yaml", task)
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(root=root, entrypoint=entrypoint),
    )

    await orchestrator.sync_task_manager_once_for_test()

    assert orchestrator.store.load_task_snapshot().tasks == [task]


async def test_sync_task_manager_once_does_not_hold_entrypoint_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Discovery sync parses the task tree OFF the entrypoint guard lock.

    The full-tree reload must not block on (or hold) the lock that worktree
    creation / merge publishes use, so a growing task tree can't starve
    creation. We prove it by holding both ``merge_lock`` and ``entrypoint_lock``
    and showing the discovery sync still completes — it acquires neither. The
    authoritative reload stays under the guard lock in the creation path (see
    ``test_ready_task_queue_reloads_tasks_before_creating_worktree``), preserving
    the consistent committed-task-view invariant.
    """

    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    load_called = False
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(root=root, entrypoint=entrypoint),
    )

    def fake_load_entrypoint_task_manager(path: Path) -> TaskManager:
        nonlocal load_called
        assert path == entrypoint.resolve()
        load_called = True
        return TaskManager()

    monkeypatch.setattr(
        orchestrator_module,
        "load_entrypoint_task_manager",
        fake_load_entrypoint_task_manager,
    )

    # Hold both candidate guard locks; discovery sync must still complete.
    await orchestrator.runtime.merge_lock.acquire()
    await orchestrator.runtime.entrypoint_lock.acquire()
    try:
        await asyncio.wait_for(orchestrator.sync_task_manager_once_for_test(), timeout=1.0)
    finally:
        orchestrator.runtime.entrypoint_lock.release()
        orchestrator.runtime.merge_lock.release()

    assert load_called


async def test_sync_task_manager_once_skips_transient_reload_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A torn/partial read (e.g. mid merge-publish) is swallowed; retry next tick."""

    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(root=root, entrypoint=entrypoint),
        task_manager=task_manager_with_tasks(
            Task(id="task-1", title="T", summary="T", description="T.")
        ),
    )
    before = orchestrator.store.load_task_snapshot().task_ids

    def boom(path: Path) -> TaskManager:
        raise ValueError("torn task yaml read")

    monkeypatch.setattr(orchestrator_module, "load_entrypoint_task_manager", boom)

    # Must not raise; the prior task manager state is left untouched.
    await orchestrator.sync_task_manager_once_for_test()
    assert orchestrator.store.load_task_snapshot().task_ids == before


async def test_all_tasks_complete_is_reconfirmed_against_committed_tasks(
    tmp_path: Path,
) -> None:
    """Terminal completion is decided from the committed ``tasks/``, not the
    advisory discovery view.

    A stale/torn discovery read can momentarily make ``state.task_manager`` look
    all-complete while an open task still exists on disk. The advisory gate would
    fire, but the authoritative guard-locked re-check must reload the real
    ``tasks/`` and see the open task, preventing a premature ``Complete``.
    """

    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    # On disk (the committed, authoritative view): one OPEN task.
    open_task = Task(id="task-open", title="Open", summary="Open", description="Still open.")
    write_task(entrypoint / "tasks" / "001-open.yaml", open_task)
    # Advisory state: a single COMPLETE task — the stale view that "looks done".
    done_task = Task(
        id="task-done",
        title="Done",
        summary="Done",
        description="Done.",
        status=TaskStatus.COMPLETE,
    )
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(root=root, entrypoint=entrypoint),
        task_manager=task_manager_with_tasks(done_task),
    )

    # The advisory gate is fooled by the stale all-complete view...
    assert await orchestrator.all_tasks_complete_once_for_test() is True
    # ...but the authoritative re-check reloads the committed tasks/ and refuses.
    assert await orchestrator.all_tasks_complete_locked_for_test() is False
    # The locked reload also refreshed state to the committed view.
    assert orchestrator.store.load_task_snapshot().task_ids == ("task-open",)

    # When the committed tree is genuinely all-complete, the authoritative check
    # agrees and the run is allowed to stop.
    write_task(
        entrypoint / "tasks" / "001-open.yaml",
        open_task.model_copy(update={"status": TaskStatus.COMPLETE}),
    )
    assert await orchestrator.all_tasks_complete_locked_for_test() is True


async def test_backfill_does_not_clobber_a_snapshot_written_during_the_read_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TOCTOU guard for the usage backfill.

    The backfill reads each terminal session's log off the lock, then applies the
    snapshots under the lock. Terminal-ness is derived from the live ``state``, so
    a worktree eligible at read time can be re-run and re-snapshotted by
    ``_record_session_usage`` before the apply. The apply must re-check under the
    lock and never overwrite that fresher snapshot with its stale read-pass value.
    """

    def _usage(amount: str) -> Usage:
        return Usage(
            tokens=TokenUsage(input_tokens=10, output_tokens=5),
            cost=Cost(amount=Decimal(amount), currency="USD"),
            model_requests=1,
        )

    stale = _usage("1.00")  # what the lock-free read pass sees on disk
    fresh = _usage("3.00")  # what a concurrent re-run snapshots before the apply

    worktree = AsyncOrchestratorWorktree(
        worktree_id="worktree_000001",
        path=tmp_path / "wt",
        head="abc",
        state=WorktreeState.CLOSED,  # terminal for the worker role
        worker_session_started=True,
    )
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(root=tmp_path / "orch", entrypoint=tmp_path / "entrypoint"),
        worktrees=(worktree,),
    )

    # Simulate the concurrent _record_session_usage: while the read pass is
    # reading the worker log it returns the stale value AND stores a fresher
    # snapshot into the store (as a real re-run+stop would, between read and apply).
    def read_then_concurrently_snapshot(
        root: Path, worktree_id: str, role: AsyncOrchestratorAgentRole
    ) -> Usage | None:
        if worktree_id == worktree.worktree_id and role is AsyncOrchestratorAgentRole.WORKER:
            orchestrator.store.set_agent_session_usage(worktree_id, role, fresh)
            return stale
        return None

    monkeypatch.setattr(
        orchestrator_module, "load_agent_session_usage", read_then_concurrently_snapshot
    )

    await orchestrator.backfill_session_usage_once_for_test()

    # The fresher snapshot survives; the stale read-pass value did not clobber it.
    stored = orchestrator.worktrees_by_id[worktree.worktree_id].agent_session_usage(
        AsyncOrchestratorAgentRole.WORKER
    )
    assert stored == fresh


async def test_backfill_guarded_write_does_not_clobber_after_recheck_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _usage(amount: str) -> Usage:
        return Usage(
            tokens=TokenUsage(input_tokens=10, output_tokens=5),
            cost=Cost(amount=Decimal(amount), currency="USD"),
            model_requests=1,
        )

    stale = _usage("1.00")
    fresh = _usage("4.00")

    class RacingUsageStore(SQLiteAsyncOrchestratorStore):
        __slots__ = ("fresh", "raced")

        fresh: Usage
        raced: bool

        def __init__(self, root: Path) -> None:
            super().__init__(root)
            self.fresh = fresh
            self.raced = False

        def set_agent_session_usage_if_missing_and_inactive(
            self,
            worktree_id: str,
            role: AsyncOrchestratorAgentRole,
            usage: Usage,
            *,
            expected_state: WorktreeState,
        ) -> bool:
            if not self.raced:
                self.raced = True
                self.set_agent_session_usage(worktree_id, role, self.fresh)
            return super().set_agent_session_usage_if_missing_and_inactive(
                worktree_id,
                role,
                usage,
                expected_state=expected_state,
            )

    worktree = AsyncOrchestratorWorktree(
        worktree_id="worktree_000001",
        path=tmp_path / "wt",
        head="abc",
        state=WorktreeState.CLOSED,
        worker_session_started=True,
    )
    store = RacingUsageStore(tmp_path / "orch")
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(root=tmp_path / "orch", entrypoint=tmp_path / "entrypoint"),
        worktrees=(worktree,),
        store=store,
    )

    def read_stale(
        root: Path, worktree_id: str, role: AsyncOrchestratorAgentRole
    ) -> Usage | None:
        del root
        if worktree_id == worktree.worktree_id and role is AsyncOrchestratorAgentRole.WORKER:
            return stale
        return None

    monkeypatch.setattr(orchestrator_module, "load_agent_session_usage", read_stale)

    await orchestrator.backfill_session_usage_once_for_test()

    stored = orchestrator.worktrees_by_id[worktree.worktree_id].agent_session_usage(
        AsyncOrchestratorAgentRole.WORKER
    )
    assert stored == fresh
    assert store.raced is True


async def test_ready_task_queue_uses_task_priority_order(tmp_path: Path) -> None:
    root = tmp_path / "orch"
    entrypoint = tmp_path / "entrypoint"
    first_default = Task(
        id="task-default-1",
        title="Default one",
        summary="Default one",
        description="Default priority.",
    )
    high = Task(
        id="task-high",
        title="High",
        summary="High",
        description="High priority.",
        priority=TaskPriority.HIGH,
    )
    second_default = Task(
        id="task-default-2",
        title="Default two",
        summary="Default two",
        description="Default priority.",
    )
    max_priority = Task(
        id="task-max",
        title="Max",
        summary="Max",
        description="Max priority.",
        priority=TaskPriority.MAX,
    )
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(root=root, entrypoint=entrypoint),
        task_manager=task_manager_with_tasks(
            first_default,
            high,
            second_default,
            max_priority,
        ),
    )

    await orchestrator.enqueue_ready_tasks_once_for_test()

    assert orchestrator.task_queue == (
        max_priority.id,
        high.id,
        first_default.id,
        second_default.id,
    )

    orchestrator.pause_scheduling_for_test()
    promoted_default = second_default.model_copy(update={"priority": TaskPriority.MAX})
    orchestrator.store.replace_task_snapshot(
        TaskManager(tasks=[first_default, high, promoted_default, max_priority])
    )

    await orchestrator.enqueue_ready_tasks_once_for_test()

    assert orchestrator.task_queue == (
        second_default.id,
        max_priority.id,
        high.id,
        first_default.id,
    )


async def test_stale_ready_task_queue_cannot_create_lower_priority_worktree(
    tmp_path: Path,
) -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable is not available")

    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    first = Task(
        id="task-1",
        title="First",
        summary="First",
        description="Queued first at default priority.",
    )
    second = Task(
        id="task-2",
        title="Second",
        summary="Second",
        description="Initially queued second at default priority.",
    )
    _initialize_git_repo(entrypoint)
    _write_and_commit_tasks(entrypoint, first, second)
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(root=root, entrypoint=entrypoint),
        task_manager=task_manager_with_tasks(first, second),
    )

    await orchestrator.enqueue_ready_tasks_once_for_test()
    assert orchestrator.task_queue == (first.id, second.id)

    promoted_second = second.model_copy(update={"priority": TaskPriority.MAX})
    _write_and_commit_tasks(entrypoint, first, promoted_second)

    await orchestrator.process_task_queue_once_for_test()

    assert orchestrator.task_queue == ()
    assert orchestrator.worker_queue == ("worktree_000001",)
    assert worktree_ids_for_task(orchestrator.store, first) == ()
    assert worktree_ids_for_task(orchestrator.store, promoted_second) == ("worktree_000001",)


async def test_ready_task_queue_feeds_worktree_creation(tmp_path: Path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable is not available")

    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    ready = Task(id="task-1", title="Ready", summary="Ready", description="Ready to run.")
    _initialize_git_repo(entrypoint)
    _write_and_commit_tasks(entrypoint, ready)
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(root=root, entrypoint=entrypoint),
        task_manager=task_manager_with_tasks(ready),
    )

    await orchestrator.enqueue_ready_tasks_once_for_test()

    assert orchestrator.task_queue == (ready.id,)
    assert orchestrator.worker_queue == ()
    assert orchestrator.worktree_ids == ()

    await orchestrator.process_task_queue_once_for_test()

    assert orchestrator.task_queue == ()
    assert orchestrator.worker_queue == ("worktree_000001",)
    assert worktree_ids_for_task(orchestrator.store, ready) == ("worktree_000001",)
    assert worktree_ids_for_task(orchestrator.store, ready) == ("worktree_000001",)
    assert orchestrator.store.aggregate_usage(root) == Usage()


async def test_ready_task_queue_reloads_tasks_before_creating_worktree(
    tmp_path: Path,
) -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable is not available")

    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    ready = Task(id="task-1", title="Ready", summary="Ready", description="Ready to run.")
    _initialize_git_repo(entrypoint)
    _write_and_commit_tasks(entrypoint, ready)
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(root=root, entrypoint=entrypoint),
        task_manager=task_manager_with_tasks(ready),
    )

    await orchestrator.enqueue_ready_tasks_once_for_test()
    assert orchestrator.task_queue == (ready.id,)

    completed = ready.model_copy(update={"status": TaskStatus.COMPLETE})
    _write_and_commit_tasks(entrypoint, completed)

    await orchestrator.process_task_queue_once_for_test()

    assert orchestrator.task_queue == ()
    assert orchestrator.worker_queue == ()
    assert orchestrator.worktree_ids == ()
    assert orchestrator.store.load_task_snapshot().tasks == [completed]


async def test_ready_task_worktree_creation_waits_for_merge_lock(
    tmp_path: Path,
) -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable is not available")

    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    ready = Task(id="task-1", title="Ready", summary="Ready", description="Ready to run.")
    _initialize_git_repo(entrypoint)
    _write_and_commit_tasks(entrypoint, ready)
    # Legacy in-entrypoint merge path pins creation to merge_lock. The default
    # staging path moves it to entrypoint_lock (decoupled from the build) — see
    # test_creation_blocks_on_entrypoint_lock_not_merge_lock.
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=root, entrypoint=entrypoint, merge_validation_worktree=False
        ),
        task_manager=task_manager_with_tasks(ready),
    )

    await orchestrator.runtime.merge_lock.acquire()
    ensure_task = asyncio.create_task(
        orchestrator.ensure_worktree_for_ready_task_id_for_test(ready.id)
    )
    try:
        await asyncio.sleep(0.05)
        assert not ensure_task.done()
    finally:
        orchestrator.runtime.merge_lock.release()

    worktree = await asyncio.wait_for(ensure_task, timeout=1.0)
    assert worktree is not None
    assert orchestrator.worker_queue == (worktree.worktree_id,)


async def test_ensure_ready_task_worktrees_once_creates_missing_association(
    tmp_path: Path,
) -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable is not available")

    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    ready = Task(id="task-1", title="Ready", summary="Ready", description="Ready to run.")
    blocked_dependency = Task(
        id="task-2", title="Blocked dep", summary="Blocked dep", description="Still open."
    )
    blocked = Task(
        id="task-3",
        title="Blocked", summary="Blocked",
        description="Not ready.",
        depends_on=[blocked_dependency.id],
    )
    _initialize_git_repo(entrypoint)
    _write_and_commit_tasks(entrypoint, ready, blocked_dependency, blocked)
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(root=root, entrypoint=entrypoint),
        task_manager=task_manager_with_tasks(ready, blocked_dependency, blocked),
    )

    await orchestrator.ensure_ready_task_worktrees_once_for_test()

    assert worktree_ids_for_task(orchestrator.store, ready) == ("worktree_000001",)
    assert worktree_ids_for_task(orchestrator.store, blocked_dependency) == (
        "worktree_000002",
    )
    assert worktree_ids_for_task(orchestrator.store, blocked) == ()
    assert orchestrator.worker_queue == ("worktree_000001", "worktree_000002")
    assert all(
        worktree.state is WorktreeState.PENDING
        for worktree in orchestrator.worktrees_by_id.values()
    )


async def test_ready_task_creation_rechecks_admission_after_task_reload(
    tmp_path: Path,
) -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable is not available")

    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    ready = Task(id="task-1", title="Ready", summary="Ready", description="Ready to run.")
    _initialize_git_repo(entrypoint)
    _write_and_commit_tasks(entrypoint, ready)

    class ReloadBlockedOrchestrator(WorktreeTestingOrchestrator):
        reload_completed: asyncio.Event
        continue_after_reload: asyncio.Event

        async def _reload_task_manager_locked(self) -> TaskManager:
            task_manager = await super()._reload_task_manager_locked()
            self.reload_completed.set()
            await self.continue_after_reload.wait()
            return task_manager

    orchestrator = ReloadBlockedOrchestrator(
        AsyncOrchestratorConfig(root=root, entrypoint=entrypoint),
        task_manager=task_manager_with_tasks(ready),
    )
    orchestrator.reload_completed = asyncio.Event()
    orchestrator.continue_after_reload = asyncio.Event()
    await orchestrator.enqueue_ready_tasks_once_for_test()

    process_task = asyncio.create_task(orchestrator.process_task_queue_once_for_test())
    try:
        await asyncio.wait_for(orchestrator.reload_completed.wait(), timeout=1.0)
        await asyncio.sleep(0)
        orchestrator.pause_scheduling_for_test()
        orchestrator.continue_after_reload.set()
        await asyncio.wait_for(process_task, timeout=3.0)
    finally:
        orchestrator.continue_after_reload.set()
        if not process_task.done():
            process_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await process_task

    assert orchestrator.task_queue == (ready.id,)
    assert orchestrator.worker_queue == ()
    assert orchestrator.worktree_ids == ()
    assert not (root / "worktrees").exists()


async def test_ready_task_with_only_closed_worktree_gets_fresh_worktree(
    tmp_path: Path,
) -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable is not available")

    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    ready = Task(id="task-1", title="Ready", summary="Ready", description="Ready to run.")
    _initialize_git_repo(entrypoint)
    _write_and_commit_tasks(entrypoint, ready)
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(root=root, entrypoint=entrypoint),
        task_manager=task_manager_with_tasks(ready),
    )
    closed_worktree = await orchestrator.create_fresh_worktree_for_test(task=ready)
    await orchestrator.transition_worktree_for_test(
        closed_worktree.worktree_id,
        WorktreeState.CLOSED,
    )

    await orchestrator.ensure_ready_task_worktrees_once_for_test()

    assert orchestrator.task_queue == ()
    assert orchestrator.worker_queue == ("worktree_000002",)
    assert worktree_ids_for_task(orchestrator.store, ready) == (
        "worktree_000001",
        "worktree_000002",
    )
    assert (
        orchestrator.worktrees_by_id[closed_worktree.worktree_id].state
        is WorktreeState.CLOSED
    )
    assert orchestrator.worktrees_by_id["worktree_000002"].state is WorktreeState.PENDING


@pytest.mark.parametrize(
    "worktree_state",
    [
        WorktreeState.PENDING,
        WorktreeState.WORKER_RUNNING,
        WorktreeState.REVIEW,
        WorktreeState.MERGE,
    ],
)
async def test_ready_task_with_non_closed_worktree_is_not_duplicated(
    tmp_path: Path,
    worktree_state: WorktreeState,
) -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable is not available")

    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    ready = Task(id="task-1", title="Ready", summary="Ready", description="Ready to run.")
    _initialize_git_repo(entrypoint)
    _write_and_commit_tasks(entrypoint, ready)
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(root=root, entrypoint=entrypoint),
        task_manager=task_manager_with_tasks(ready),
    )
    existing_worktree = await orchestrator.create_fresh_worktree_for_test(task=ready)
    await orchestrator.transition_worktree_for_test(
        existing_worktree.worktree_id,
        worktree_state,
    )

    await orchestrator.ensure_ready_task_worktrees_once_for_test()

    assert orchestrator.task_queue == ()
    assert worktree_ids_for_task(orchestrator.store, ready) == ("worktree_000001",)
    assert orchestrator.worktree_ids == ("worktree_000001",)
    assert (
        orchestrator.worktrees_by_id[existing_worktree.worktree_id].state
        is worktree_state
    )


async def test_constructor_resume_resets_worker_running_worktrees(
    tmp_path: Path,
) -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable is not available")

    task = Task(id="task-1", title="Ready", summary="Ready", description="Ready to run.")
    worktree_path = tmp_path / "worktree"
    _initialize_git_repo(worktree_path)
    worktree = AsyncOrchestratorWorktree(
        worktree_id="worktree_000001",
        task_id=task.id,
        path=worktree_path,
        head="abc123",
        state=WorktreeState.WORKER_RUNNING,
        worker_session_started=True,
    )

    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(root=tmp_path / "orch", entrypoint=tmp_path / "entrypoint"),
        task_manager=task_manager_with_tasks(task),
        worktrees=(worktree,),
        check_resume_health=True,
    )

    resumed = orchestrator.store.get_worktree("worktree_000001")
    assert resumed is not None
    assert resumed.state is WorktreeState.PENDING
    assert resumed.worker_session_started is True
    assert orchestrator.worker_queue == ("worktree_000001",)


async def test_resume_warns_and_skips_unhealthy_worktrees(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable is not available")

    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    valid_worktree_path = tmp_path / "valid-worktree"
    missing_worktree_path = tmp_path / "missing-worktree"
    invalid_worktree_path = tmp_path / "not-a-git-worktree"
    invalid_worktree_path.mkdir()
    _initialize_git_repo(entrypoint)
    _run_git(entrypoint, "worktree", "add", "--detach", str(valid_worktree_path), "HEAD")
    task = Task(id="task-1", title="Ready", summary="Ready", description="Ready to run.")
    task_manager = task_manager_with_tasks(task)
    worktrees = (
        AsyncOrchestratorWorktree(
            worktree_id="worktree_000001",
            task_id=task.id,
            path=valid_worktree_path,
            head="abc123",
            state=WorktreeState.PENDING,
        ),
        AsyncOrchestratorWorktree(
            worktree_id="worktree_000002",
            task_id=task.id,
            path=missing_worktree_path,
            head="abc123",
            state=WorktreeState.PENDING,
        ),
        AsyncOrchestratorWorktree(
            worktree_id="worktree_000003",
            task_id=task.id,
            path=invalid_worktree_path,
            head="abc123",
            state=WorktreeState.REVIEW,
        ),
        AsyncOrchestratorWorktree(
            worktree_id="worktree_000004",
            task_id=task.id,
            path=tmp_path / "closed-missing-worktree",
            head="abc123",
            state=WorktreeState.CLOSED,
        ),
    )
    caplog.set_level(logging.WARNING, logger=orchestrator_module.__name__)

    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(root=root, entrypoint=entrypoint),
        task_manager=task_manager,
        worktrees=worktrees,
        check_resume_health=True,
    )

    assert orchestrator.store.load_task_snapshot() == task_manager
    expected_worktrees = {worktree.worktree_id: worktree for worktree in worktrees}
    assert orchestrator.worktrees_by_id == expected_worktrees
    assert orchestrator.worker_queue == ("worktree_000001",)
    assert orchestrator.review_queue == ()
    assert orchestrator.merge_queue == ()
    log_text = caplog.text
    assert "worktree_000002" in log_text
    assert "path does not exist" in log_text
    assert "worktree_000003" in log_text
    assert "path is missing .git metadata" in log_text
    assert "worktree_000004" not in log_text


async def test_concurrent_ready_worktree_ensure_is_atomic(tmp_path: Path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable is not available")

    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    ready = Task(id="task-1", title="Ready", summary="Ready", description="Ready to run.")
    _initialize_git_repo(entrypoint)
    _write_and_commit_tasks(entrypoint, ready)
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(root=root, entrypoint=entrypoint),
        task_manager=task_manager_with_tasks(ready),
    )

    await asyncio.gather(
        *(orchestrator.ensure_ready_task_worktrees_once_for_test() for _ in range(5))
    )

    assert worktree_ids_for_task(orchestrator.store, ready) == ("worktree_000001",)
    assert orchestrator.worktree_ids == ("worktree_000001",)
    assert orchestrator.worker_queue == ("worktree_000001",)


async def test_worker_loop_spawns_up_to_concurrency_and_queues_review(
    tmp_path: Path,
) -> None:
    if shutil.which("git") is None or shutil.which("sh") is None:
        pytest.skip("git and sh executables are required")

    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    first = Task(id="task-1", title="First", summary="First", description="First task.")
    second = Task(id="task-2", title="Second", summary="Second", description="Second task.")
    _initialize_git_repo(entrypoint)
    _write_and_commit_tasks(entrypoint, first, second)
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=root,
            entrypoint=entrypoint,
            worker_agent_command=AsyncOrchestratorAgentCommandConfig(
                argv=(
                    "sh",
                    "-c",
                    "printf \"$TEND_TASK_ID\" > worker-task.txt; "
                    "while [ ! -f release-worker ]; do sleep 0.05; done; "
                    "rm release-worker; git add worker-task.txt; "
                    "git commit -q -m worker-output; "
                    "printf '{\"schema_version\":1,\"status\":\"completed\","
                    "\"summary\":\"Implemented worker changes.\"}'",
                ),
            ),
            max_concurrent_worker_agents=1,
        ),
        task_manager=task_manager_with_tasks(first, second),
    )

    await orchestrator.ensure_ready_task_worktrees_once_for_test()
    await orchestrator.spawn_worker_agents_once_for_test()
    first_worktree = orchestrator.worktrees_by_id["worktree_000001"]
    second_worktree = orchestrator.worktrees_by_id["worktree_000002"]
    await _wait_for_path(first_worktree.path / "worker-task.txt")

    assert orchestrator.worker_queue == ("worktree_000002",)
    assert orchestrator.worktrees_by_id[first_worktree.worktree_id].state is (
        WorktreeState.WORKER_RUNNING
    )
    assert orchestrator.worktrees_by_id[second_worktree.worktree_id].state is (
        WorktreeState.PENDING
    )

    (first_worktree.path / "release-worker").write_text("done\n", encoding="utf-8")
    await orchestrator.wait_for_worker_agents_for_test()
    assert orchestrator.review_queue == ("worktree_000001",)
    updated_first = orchestrator.worktrees_by_id[first_worktree.worktree_id]
    assert updated_first.state is WorktreeState.REVIEW
    assert updated_first.discussion[-1].message == "Implemented worker changes."
    assert "Implemented worker changes." in (
        first_worktree.path / ".tend" / "discussion.md"
    ).read_text(encoding="utf-8")

    await orchestrator.spawn_worker_agents_once_for_test()
    await _wait_for_path(second_worktree.path / "worker-task.txt")
    assert orchestrator.worker_queue == ()
    assert orchestrator.worktrees_by_id[second_worktree.worktree_id].state is (
        WorktreeState.WORKER_RUNNING
    )
    (second_worktree.path / "release-worker").write_text("done\n", encoding="utf-8")
    await orchestrator.wait_for_worker_agents_for_test()


async def test_worker_success_runs_validation_commands_before_review(
    tmp_path: Path,
) -> None:
    if shutil.which("sh") is None or shutil.which("git") is None:
        pytest.skip("sh and git executables are required")

    worktree_path = tmp_path / "worktree"
    _initialize_git_repo(worktree_path)
    task = Task(id="task-1", title="Task", summary="Task", description="Do it.")
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=tmp_path / "orch",
            entrypoint=tmp_path / "entrypoint",
            worker_agent_command=AsyncOrchestratorAgentCommandConfig(
                argv=(
                    "sh",
                    "-c",
                    "printf worker > worker-made.txt; "
                    "git add worker-made.txt; git commit -q -m worker-made; "
                    "printf '{\"schema_version\":1,\"status\":\"completed\","
                    "\"summary\":\"Worker completed.\"}'",
                ),
            ),
            validation_commands=(
                AsyncOrchestratorValidationCommandConfig(
                    argv=("sh", "-c", "test -f worker-made.txt")
                ),
                AsyncOrchestratorValidationCommandConfig(
                    argv=("sh", "-c", "printf validated > .tend/validation-ran.txt")
                ),
            ),
        ),
        task_manager=task_manager_with_tasks(task),
        worktrees=(
            AsyncOrchestratorWorktree(
                worktree_id="worktree_000001",
                task_id=task.id,
                path=worktree_path,
                head="abc123",
            ),
        ),
    )

    await orchestrator.spawn_worker_agents_once_for_test()
    await orchestrator.wait_for_worker_agents_for_test()

    assert (worktree_path / ".tend" / "validation-ran.txt").read_text(
        encoding="utf-8"
    ) == "validated"
    assert orchestrator.worker_queue == ()
    assert orchestrator.review_queue == ("worktree_000001",)
    updated = orchestrator.worktrees_by_id["worktree_000001"]
    assert updated.state is WorktreeState.REVIEW
    assert updated.discussion[-1].role is AsyncOrchestratorAgentRole.WORKER
    assert updated.discussion[-1].message == "Worker completed."


async def test_worker_validation_failure_requeues_with_feedback(
    tmp_path: Path,
) -> None:
    if shutil.which("sh") is None or shutil.which("git") is None:
        pytest.skip("sh and git executables are required")

    worktree_path = tmp_path / "worktree"
    _initialize_git_repo(worktree_path)
    task = Task(id="task-1", title="Task", summary="Task", description="Do it.")
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=tmp_path / "orch",
            entrypoint=tmp_path / "entrypoint",
            worker_agent_command=AsyncOrchestratorAgentCommandConfig(
                argv=(
                    "sh",
                    "-c",
                    "printf '{\"schema_version\":1,\"status\":\"completed\","
                    "\"summary\":\"Worker completed.\"}'",
                ),
            ),
            validation_commands=(
                AsyncOrchestratorValidationCommandConfig(
                    argv=("sh", "-c", "printf first > first-validation.txt")
                ),
                AsyncOrchestratorValidationCommandConfig(
                    argv=(
                        "sh",
                        "-c",
                        "printf validation-out; printf validation-err >&2; exit 7",
                    )
                ),
                AsyncOrchestratorValidationCommandConfig(
                    argv=("sh", "-c", "printf should-not-run > later-validation.txt")
                ),
            ),
        ),
        task_manager=task_manager_with_tasks(task),
        worktrees=(
            AsyncOrchestratorWorktree(
                worktree_id="worktree_000001",
                task_id=task.id,
                path=worktree_path,
                head="abc123",
            ),
        ),
    )

    await orchestrator.spawn_worker_agents_once_for_test()
    await orchestrator.wait_for_worker_agents_for_test()

    assert (worktree_path / "first-validation.txt").read_text(encoding="utf-8") == "first"
    assert not (worktree_path / "later-validation.txt").exists()
    assert orchestrator.review_queue == ()
    assert orchestrator.worker_queue == ("worktree_000001",)
    updated = orchestrator.worktrees_by_id["worktree_000001"]
    assert updated.state is WorktreeState.PENDING
    assert [message.role for message in updated.discussion[-2:]] == [
        AsyncOrchestratorAgentRole.WORKER,
        AsyncOrchestratorAgentRole.ORCHESTRATOR,
    ]
    assert updated.discussion[-2].message == "Worker completed."
    feedback = updated.discussion[-1].message
    assert "Validation failed" in feedback
    assert "Exit code: 7" in feedback
    assert "validation-out" in feedback
    assert "validation-err" in feedback
    discussion_log = worktree_path / ".tend" / "discussion.md"
    assert "Validation failed" in discussion_log.read_text(encoding="utf-8")


async def test_worker_dirty_worktree_requeues_before_validation_or_review(
    tmp_path: Path,
) -> None:
    if shutil.which("sh") is None or shutil.which("git") is None:
        pytest.skip("sh and git executables are required")

    entrypoint = tmp_path / "entrypoint"
    _initialize_git_repo(entrypoint)
    head = _run_git(entrypoint, "rev-parse", "HEAD")
    worktree_path = tmp_path / "worktree"
    _run_git(entrypoint, "worktree", "add", "--detach", str(worktree_path), "HEAD")
    task = Task(id="task-1", title="Task", summary="Task", description="Do it.")
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=tmp_path / "orch",
            entrypoint=entrypoint,
            worker_agent_command=AsyncOrchestratorAgentCommandConfig(
                argv=(
                    "sh",
                    "-c",
                    "printf keep > committed.txt; git add committed.txt; "
                    "git commit -q -m committed; "
                    "printf wip > uncommitted.txt; "
                    "printf '{\"schema_version\":1,\"status\":\"completed\","
                    "\"summary\":\"Worker completed.\"}'",
                ),
            ),
            validation_commands=(
                AsyncOrchestratorValidationCommandConfig(
                    argv=("sh", "-c", "printf validated > .tend/validation-ran.txt")
                ),
            ),
        ),
        task_manager=task_manager_with_tasks(task),
        worktrees=(
            AsyncOrchestratorWorktree(
                worktree_id="worktree_000001",
                task_id=task.id,
                path=worktree_path,
                head=head,
            ),
        ),
    )

    await orchestrator.spawn_worker_agents_once_for_test()
    await orchestrator.wait_for_worker_agents_for_test()

    updated = orchestrator.worktrees_by_id["worktree_000001"]
    assert updated.state is WorktreeState.PENDING
    assert orchestrator.worker_queue == ("worktree_000001",)
    assert orchestrator.review_queue == ()
    assert not (worktree_path / ".tend" / "validation-ran.txt").exists()
    assert [message.role for message in updated.discussion[-2:]] == [
        AsyncOrchestratorAgentRole.WORKER,
        AsyncOrchestratorAgentRole.ORCHESTRATOR,
    ]
    assert updated.discussion[-2].message == "Worker completed."
    feedback = updated.discussion[-1].message
    assert "Uncommitted worktree changes detected" in feedback
    assert "uncommitted.txt" in feedback
    assert "committed tree" in feedback
    discussion_log = worktree_path / ".tend" / "discussion.md"
    assert "Uncommitted worktree changes detected" in discussion_log.read_text(
        encoding="utf-8"
    )


async def test_worker_validation_dirty_output_requeues_before_review(
    tmp_path: Path,
) -> None:
    if shutil.which("sh") is None or shutil.which("git") is None:
        pytest.skip("sh and git executables are required")

    worktree_path = tmp_path / "worktree"
    _initialize_git_repo(worktree_path)
    task = Task(id="task-1", title="Task", summary="Task", description="Do it.")
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=tmp_path / "orch",
            entrypoint=tmp_path / "entrypoint",
            worker_agent_command=AsyncOrchestratorAgentCommandConfig(
                argv=(
                    "sh",
                    "-c",
                    "printf keep > committed.txt; git add committed.txt; "
                    "git commit -q -m committed; "
                    "printf '{\"schema_version\":1,\"status\":\"completed\","
                    "\"summary\":\"Worker completed.\"}'",
                ),
            ),
            validation_commands=(
                AsyncOrchestratorValidationCommandConfig(
                    argv=("sh", "-c", "printf generated > validation-output.txt")
                ),
            ),
        ),
        task_manager=task_manager_with_tasks(task),
        worktrees=(
            AsyncOrchestratorWorktree(
                worktree_id="worktree_000001",
                task_id=task.id,
                path=worktree_path,
                head="abc123",
            ),
        ),
    )

    await orchestrator.spawn_worker_agents_once_for_test()
    await orchestrator.wait_for_worker_agents_for_test()

    updated = orchestrator.worktrees_by_id["worktree_000001"]
    assert updated.state is WorktreeState.PENDING
    assert orchestrator.worker_queue == ("worktree_000001",)
    assert orchestrator.review_queue == ()
    assert (worktree_path / "validation-output.txt").read_text(
        encoding="utf-8"
    ) == "generated"
    feedback = updated.discussion[-1].message
    assert "Uncommitted worktree changes detected" in feedback
    assert "validation-output.txt" in feedback


async def test_worker_dirty_tend_metadata_does_not_block_review(
    tmp_path: Path,
) -> None:
    if shutil.which("sh") is None or shutil.which("git") is None:
        pytest.skip("sh and git executables are required")

    entrypoint = tmp_path / "entrypoint"
    _initialize_git_repo(entrypoint)
    head = _run_git(entrypoint, "rev-parse", "HEAD")
    worktree_path = tmp_path / "worktree"
    _run_git(entrypoint, "worktree", "add", "--detach", str(worktree_path), "HEAD")
    task = Task(id="task-1", title="Task", summary="Task", description="Do it.")
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=tmp_path / "orch",
            entrypoint=entrypoint,
            worker_agent_command=AsyncOrchestratorAgentCommandConfig(
                argv=(
                    "sh",
                    "-c",
                    "printf keep > committed.txt; git add committed.txt; "
                    "git commit -q -m committed; "
                    "mkdir -p .tend; printf scratch > .tend/agent-scratch.txt; "
                    "printf '{\"schema_version\":1,\"status\":\"completed\","
                    "\"summary\":\"Worker completed.\"}'",
                ),
            ),
        ),
        task_manager=task_manager_with_tasks(task),
        worktrees=(
            AsyncOrchestratorWorktree(
                worktree_id="worktree_000001",
                task_id=task.id,
                path=worktree_path,
                head=head,
            ),
        ),
    )

    await orchestrator.spawn_worker_agents_once_for_test()
    await orchestrator.wait_for_worker_agents_for_test()

    updated = orchestrator.worktrees_by_id["worktree_000001"]
    assert updated.state is WorktreeState.REVIEW
    assert orchestrator.review_queue == ("worktree_000001",)
    assert orchestrator.worker_queue == ()
    assert updated.discussion[-1].role is AsyncOrchestratorAgentRole.WORKER
    assert updated.discussion[-1].message == "Worker completed."


async def test_blocked_worker_with_no_commits_closes_for_respawn(
    tmp_path: Path,
) -> None:
    if shutil.which("sh") is None or shutil.which("git") is None:
        pytest.skip("sh and git executables are required")

    entrypoint = tmp_path / "entrypoint"
    _initialize_git_repo(entrypoint)
    head = _run_git(entrypoint, "rev-parse", "HEAD")
    worktree_path = tmp_path / "worktree"
    _run_git(entrypoint, "worktree", "add", "--detach", str(worktree_path), "HEAD")
    task = Task(id="task-1", title="Task", summary="Task", description="Do it.")
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=tmp_path / "orch",
            entrypoint=entrypoint,
            worker_agent_command=AsyncOrchestratorAgentCommandConfig(
                argv=(
                    "sh",
                    "-c",
                    "printf '{\"schema_version\":1,\"status\":\"blocked\","
                    "\"summary\":\"Blocked; committed nothing.\"}'",
                ),
            ),
        ),
        task_manager=task_manager_with_tasks(task),
        worktrees=(
            AsyncOrchestratorWorktree(
                worktree_id="worktree_000001",
                task_id=task.id,
                path=worktree_path,
                head=head,
            ),
        ),
    )

    await orchestrator.spawn_worker_agents_once_for_test()
    await orchestrator.wait_for_worker_agents_for_test()

    # Nothing committed -> close the worktree so the still-open task re-enqueues a
    # fresh worker, rather than queueing it for review or resuming in place.
    updated = orchestrator.worktrees_by_id["worktree_000001"]
    assert updated.state is WorktreeState.CLOSED
    assert updated.discussion[-1].message == "Blocked; committed nothing."
    assert orchestrator.review_queue == ()
    assert orchestrator.worker_queue == ()
    # This close path records a discussion message instead of going through the
    # merge transition helper; clean CLOSED worktrees are still reclaimed.
    assert not worktree_path.exists()
    listed = _run_git(entrypoint, "worktree", "list", "--porcelain")
    assert str(worktree_path) not in listed


async def test_blocked_worker_with_committed_task_edits_routes_to_review(
    tmp_path: Path,
) -> None:
    if shutil.which("sh") is None or shutil.which("git") is None:
        pytest.skip("sh and git executables are required")

    entrypoint = tmp_path / "entrypoint"
    _initialize_git_repo(entrypoint)
    head = _run_git(entrypoint, "rev-parse", "HEAD")
    worktree_path = tmp_path / "worktree"
    _run_git(entrypoint, "worktree", "add", "--detach", str(worktree_path), "HEAD")
    task = Task(id="task-1", title="Task", summary="Task", description="Do it.")
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=tmp_path / "orch",
            entrypoint=entrypoint,
            worker_agent_command=AsyncOrchestratorAgentCommandConfig(
                argv=(
                    "sh",
                    "-c",
                    "mkdir -p tasks; printf 'prereq' > tasks/task-1-1.md; "
                    "git add -A; git commit -q -m 'file prerequisite'; "
                    "printf '{\"schema_version\":1,\"status\":\"blocked\","
                    "\"summary\":\"Filed prerequisite task-1-1; blocked on it.\"}'",
                ),
            ),
        ),
        task_manager=task_manager_with_tasks(task),
        worktrees=(
            AsyncOrchestratorWorktree(
                worktree_id="worktree_000001",
                task_id=task.id,
                path=worktree_path,
                head=head,
            ),
        ),
    )

    await orchestrator.spawn_worker_agents_once_for_test()
    await orchestrator.wait_for_worker_agents_for_test()

    # Committed task-graph edits -> route through review so they merge (the task stays
    # open and is rescheduled once unblocked).
    updated = orchestrator.worktrees_by_id["worktree_000001"]
    assert updated.state is WorktreeState.REVIEW
    assert orchestrator.review_queue == ("worktree_000001",)
    assert orchestrator.worker_queue == ()
    assert updated.discussion[-1].role is AsyncOrchestratorAgentRole.WORKER
    assert (worktree_path / "tasks" / "task-1-1.md").read_text(encoding="utf-8") == "prereq"


async def test_blocked_worker_with_non_task_edits_flags_review_context(
    tmp_path: Path,
) -> None:
    if shutil.which("sh") is None or shutil.which("git") is None:
        pytest.skip("sh and git executables are required")

    entrypoint = tmp_path / "entrypoint"
    _initialize_git_repo(entrypoint)
    head = _run_git(entrypoint, "rev-parse", "HEAD")
    worktree_path = tmp_path / "worktree"
    _run_git(entrypoint, "worktree", "add", "--detach", str(worktree_path), "HEAD")
    task = Task(id="task-1", title="Task", summary="Task", description="Do it.")
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=tmp_path / "orch",
            entrypoint=entrypoint,
            worker_agent_command=AsyncOrchestratorAgentCommandConfig(
                argv=(
                    "sh",
                    "-c",
                    "mkdir -p tasks src; printf 'prereq' > tasks/task-1-1.md; "
                    "printf 'non-task work' > src/Foo.lean; "
                    "git add -A; git commit -q -m 'file prerequisite and wip'; "
                    "printf '{\"schema_version\":1,\"status\":\"blocked\","
                    "\"summary\":\"Filed prerequisite task-1-1; blocked on it.\"}'",
                ),
            ),
        ),
        task_manager=task_manager_with_tasks(task),
        worktrees=(
            AsyncOrchestratorWorktree(
                worktree_id="worktree_000001",
                task_id=task.id,
                path=worktree_path,
                head=head,
            ),
        ),
    )

    await orchestrator.spawn_worker_agents_once_for_test()
    await orchestrator.wait_for_worker_agents_for_test()

    updated = orchestrator.worktrees_by_id["worktree_000001"]
    assert updated.state is WorktreeState.REVIEW
    assert orchestrator.review_queue == ("worktree_000001",)
    assert orchestrator.worker_queue == ()
    assert [message.role for message in updated.discussion[-2:]] == [
        AsyncOrchestratorAgentRole.WORKER,
        AsyncOrchestratorAgentRole.ORCHESTRATOR,
    ]
    feedback = updated.discussion[-1].message
    assert "Blocked contribution touched non-task files." in feedback
    assert "src/Foo.lean" in feedback
    assert "diff --git" in feedback
    assert "+non-task work" in feedback
    assert "tasks/task-1-1.md" not in feedback
    discussion_log = worktree_path / ".tend" / "discussion.md"
    assert "Blocked contribution touched non-task files." in discussion_log.read_text(
        encoding="utf-8"
    )


async def test_run_validation_commands_times_out_a_hung_command(tmp_path: Path) -> None:
    if shutil.which("sh") is None:
        pytest.skip("sh executable is required")

    failure = await orchestrator_module._run_validation_commands_async(  # pyright: ignore[reportPrivateUsage]
        (
            AsyncOrchestratorValidationCommandConfig(
                argv=("sh", "-c", "sleep 30"),
                timeout_seconds=0.2,
            ),
        ),
        tmp_path,
    )

    # A command that exceeds its timeout is reported as a validation failure (no
    # returncode) rather than hanging the merge thread forever. The kill
    # escalation that follows a timeout must NOT reclassify it as cancelled.
    assert failure is not None
    assert failure.returncode is None
    assert "timed out" in failure.error
    assert failure.cancelled is False


async def test_run_validation_commands_respects_no_timeout(tmp_path: Path) -> None:
    if shutil.which("sh") is None:
        pytest.skip("sh executable is required")

    # With timeout_seconds=None a quick command completes normally (no timeout set).
    failure = await orchestrator_module._run_validation_commands_async(  # pyright: ignore[reportPrivateUsage]
        (AsyncOrchestratorValidationCommandConfig(argv=("sh", "-c", "true")),),
        tmp_path,
    )

    assert failure is None


async def test_run_validation_commands_classifies_signal_exit_as_cancelled(
    tmp_path: Path,
) -> None:
    if shutil.which("sh") is None:
        pytest.skip("sh executable is required")

    failure = await orchestrator_module._run_validation_commands_async(  # pyright: ignore[reportPrivateUsage]
        (AsyncOrchestratorValidationCommandConfig(argv=("sh", "-c", "kill -TERM $$")),),
        tmp_path,
    )

    # A cancellation-signal exit outside the timeout path (SIGTERM here — an
    # external kill such as an operator kill or a container shutdown; the
    # orchestrator's own cancellation raises CancelledError and never produces
    # a classified result) is classified as cancelled, not as an ordinary
    # validation failure (issue #132).
    assert failure is not None
    assert failure.returncode == -signal.SIGTERM
    assert failure.cancelled is True
    assert failure.signal_number == signal.SIGTERM


# SIGQUIT is deliberately off the cancellation allowlist: it is traditionally
# a core-dump/fatal signal, weak evidence of infrastructure cancellation.
@pytest.mark.parametrize("signal_name", ["SEGV", "QUIT"])
async def test_run_validation_commands_classifies_crash_signal_as_failure(
    signal_name: str,
    tmp_path: Path,
) -> None:
    if shutil.which("sh") is None:
        pytest.skip("sh executable is required")

    signum = int(getattr(signal, f"SIG{signal_name}"))
    failure = await orchestrator_module._run_validation_commands_async(  # pyright: ignore[reportPrivateUsage]
        (
            AsyncOrchestratorValidationCommandConfig(
                argv=("sh", "-c", f"kill -{signal_name} $$")
            ),
        ),
        tmp_path,
    )

    # A signal outside the cancellation allowlist means the validator itself
    # crashed. A deterministic crash is evidence of a real failure, so it is
    # an ordinary validation failure — never retried, since a retry would
    # convert that failure signal into a pass — with the signal recorded in
    # the error.
    assert failure is not None
    assert failure.returncode == -signum
    assert failure.cancelled is False
    assert failure.crashed is True
    assert failure.signal_number == signum
    assert failure.error == f"validation command crashed (signal {signum})"


def test_cancellation_signal_allowlist_filters_by_availability() -> None:
    # TERM and INT exist on every supported platform and must always classify
    # as cancellation.
    assert signal.SIGTERM in orchestrator_module._VALIDATION_CANCELLATION_SIGNALS  # pyright: ignore[reportPrivateUsage]
    assert signal.SIGINT in orchestrator_module._VALIDATION_CANCELLATION_SIGNALS  # pyright: ignore[reportPrivateUsage]

    # Windows' signal module exposes no SIGKILL/SIGHUP; the allowlist filters
    # by availability instead of raising AttributeError at import time.
    class _WindowsLikeSignals:
        SIGTERM = 15
        SIGINT = 2

    assert orchestrator_module._cancellation_signals_available_in(  # pyright: ignore[reportPrivateUsage]
        _WindowsLikeSignals
    ) == frozenset({15, 2})


async def test_run_validation_commands_signal_exit_kills_surviving_descendants(
    tmp_path: Path,
) -> None:
    if os.name == "nt" or shutil.which("sh") is None:
        pytest.skip("POSIX sh executable is required")

    pidfile = tmp_path / "child.pid"
    # A descendant that redirects its stdio survives the leader's signal exit
    # (communicate() sees EOF as soon as the leader dies). The classification
    # path must kill the process group before returning, or the caller's retry
    # would run concurrently with the survivor in the same worktree.
    script = f'sleep 30 >/dev/null 2>&1 & echo $! > "{pidfile}"; kill -TERM $$'
    failure = await orchestrator_module._run_validation_commands_async(  # pyright: ignore[reportPrivateUsage]
        (AsyncOrchestratorValidationCommandConfig(argv=("sh", "-c", script)),),
        tmp_path,
    )

    assert failure is not None
    assert failure.cancelled is True
    child_pid = int(pidfile.read_text(encoding="utf-8"))
    try:
        await _wait_for_pid_to_stop(child_pid)
    finally:
        try:
            os.kill(child_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


async def test_run_validation_commands_signal_exit_with_inherited_pipes_does_not_hang(
    tmp_path: Path,
) -> None:
    if os.name == "nt" or shutil.which("sh") is None:
        pytest.skip("POSIX sh executable is required")

    pidfile = tmp_path / "child.pid"
    # Unlike the redirected-stdio variant above, the descendant here INHERITS
    # the captured stdout/stderr pipes, so communicate() alone would only
    # return when the child exits — with no timeout configured, a long-lived
    # child hangs the validation indefinitely and the group kill in the
    # classification path is never reached. Leader-exit monitoring must kill
    # the group and return promptly instead.
    script = f'sleep 30 & echo $! > "{pidfile}"; kill -TERM $$'
    started = asyncio.get_running_loop().time()
    failure = await orchestrator_module._run_validation_commands_async(  # pyright: ignore[reportPrivateUsage]
        (AsyncOrchestratorValidationCommandConfig(argv=("sh", "-c", script)),),
        tmp_path,
    )
    elapsed = asyncio.get_running_loop().time() - started

    assert failure is not None
    assert failure.cancelled is True
    assert failure.returncode == -signal.SIGTERM
    # Far below the child's 30s lifetime: the return must not wait for it.
    assert elapsed < 5.0
    child_pid = int(pidfile.read_text(encoding="utf-8"))
    try:
        await _wait_for_pid_to_stop(child_pid)
    finally:
        try:
            os.kill(child_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


async def test_run_validation_commands_ordinary_failure_is_not_cancelled(
    tmp_path: Path,
) -> None:
    if shutil.which("sh") is None:
        pytest.skip("sh executable is required")

    failure = await orchestrator_module._run_validation_commands_async(  # pyright: ignore[reportPrivateUsage]
        (AsyncOrchestratorValidationCommandConfig(argv=("sh", "-c", "exit 3")),),
        tmp_path,
    )

    # A plain non-zero exit keeps the ordinary-failure classification.
    assert failure is not None
    assert failure.returncode == 3
    assert failure.cancelled is False
    assert failure.signal_number is None


async def test_worker_resumes_previous_session_and_receives_discussion_log(
    tmp_path: Path,
) -> None:
    if shutil.which("sh") is None or shutil.which("git") is None:
        pytest.skip("sh and git executables are required")

    worktree_path = tmp_path / "worktree"
    _initialize_git_repo(worktree_path)
    task = Task(id="task-1", title="Task", summary="Task", description="Do it.")
    script = (
        "printf \"$TEND_AGENT_RESUME\" > .tend/resume-$TEND_AGENT_RESUME.txt; "
        "printf \"%s\" \"$*\" > .tend/args-$TEND_AGENT_RESUME.txt; "
        "cat \"$TEND_AGENT_DISCUSSION_PATH\" > .tend/seen-$TEND_AGENT_RESUME.md; "
        "printf '{\"schema_version\":1,\"status\":\"completed\","
        "\"summary\":\"Worker completed a pass.\"}'"
    )
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=tmp_path / "orch",
            entrypoint=tmp_path / "entrypoint",
            worker_agent_command=AsyncOrchestratorAgentCommandConfig(
                argv=("sh", "-c", script, "agent"),
                resume_argv=("--resume-session",),
            ),
        ),
        task_manager=task_manager_with_tasks(task),
        worktrees=(
            AsyncOrchestratorWorktree(
                worktree_id="worktree_000001",
                task_id=task.id,
                path=worktree_path,
                head="abc123",
            ),
        ),
    )

    await orchestrator.spawn_worker_agents_once_for_test()
    await orchestrator.wait_for_worker_agents_for_test()
    assert (worktree_path / ".tend" / "resume-0.txt").read_text(encoding="utf-8") == "0"
    assert (worktree_path / ".tend" / "args-0.txt").read_text(encoding="utf-8") == ""

    await orchestrator.transition_worktree_for_test("worktree_000001", WorktreeState.PENDING)
    await orchestrator.spawn_worker_agents_once_for_test()
    await orchestrator.wait_for_worker_agents_for_test()

    assert (worktree_path / ".tend" / "resume-1.txt").read_text(encoding="utf-8") == "1"
    assert (
        worktree_path / ".tend" / "args-1.txt"
    ).read_text(encoding="utf-8") == "--resume-session"
    assert "Worker completed a pass." in (
        worktree_path / ".tend" / "seen-1.md"
    ).read_text(encoding="utf-8")
    assert orchestrator.worktrees_by_id["worktree_000001"].worker_session_started


async def test_reviewer_loop_routes_approved_worktree_to_merge_queue(tmp_path: Path) -> None:
    if shutil.which("sh") is None:
        pytest.skip("sh executable is required")

    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()
    worktree = AsyncOrchestratorWorktree(
        worktree_id="worktree_000001",
        path=worktree_path,
        head="abc123",
        state=WorktreeState.REVIEW,
    )
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=tmp_path / "orch",
            entrypoint=tmp_path / "entrypoint",
            reviewer_agent_command=AsyncOrchestratorAgentCommandConfig(
                argv=(
                    "sh",
                    "-c",
                    "printf '{\"schema_version\":1,\"verdict\":\"approve\","
                    "\"notes\":\"Looks good.\"}'",
                ),
            ),
        ),
        worktrees=(worktree,),
    )

    await orchestrator.spawn_reviewer_agents_once_for_test()
    await orchestrator.wait_for_reviewer_agents_for_test()

    assert orchestrator.review_queue == ()
    assert orchestrator.merge_queue == ("worktree_000001",)
    updated = orchestrator.worktrees_by_id[worktree.worktree_id]
    assert updated.state is WorktreeState.MERGE
    assert updated.discussion[-1].message == "Looks good."
    # The structured verdict is persisted on state and as a JSON artifact.
    assert len(updated.review_verdicts) == 1
    assert updated.review_verdicts[0].verdict == "approve"
    assert updated.review_verdicts[0].notes == "Looks good."
    artifact = worktree_path / ".tend" / "reviews" / "001-review-verdict.json"
    assert artifact.is_file()
    persisted = json.loads(artifact.read_text(encoding="utf-8"))
    assert persisted["verdict"] == "approve"
    assert persisted["notes"] == "Looks good."


async def test_reviewer_request_changes_persists_structured_comments(tmp_path: Path) -> None:
    if shutil.which("sh") is None:
        pytest.skip("sh executable is required")

    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()
    worktree = AsyncOrchestratorWorktree(
        worktree_id="worktree_000001",
        path=worktree_path,
        head="abc123",
        state=WorktreeState.REVIEW,
    )
    # A request_changes verdict carries a per-comment array that the discussion
    # message flattens away; it must survive in the structured artifact.
    verdict_json = (
        '{"schema_version":1,"verdict":"request_changes",'
        '"notes":"Criterion 2 FAIL.","feedback_text":"Fix the proof.",'
        '"comments":[{"message":"sorry placeholder remains","path":"src/foo.lean",'
        '"line_start":10,"severity":"error"}]}'
    )
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=tmp_path / "orch",
            entrypoint=tmp_path / "entrypoint",
            reviewer_agent_command=AsyncOrchestratorAgentCommandConfig(
                argv=("sh", "-c", f"printf '{verdict_json}'"),
            ),
        ),
        worktrees=(worktree,),
    )

    await orchestrator.spawn_reviewer_agents_once_for_test()
    await orchestrator.wait_for_reviewer_agents_for_test()

    updated = orchestrator.worktrees_by_id[worktree.worktree_id]
    assert updated.state is WorktreeState.PENDING
    assert len(updated.review_verdicts) == 1
    assert [comment.message for comment in updated.review_verdicts[0].comments] == [
        "sorry placeholder remains",
    ]
    artifact = worktree_path / ".tend" / "reviews" / "001-review-verdict.json"
    persisted = json.loads(artifact.read_text(encoding="utf-8"))
    assert persisted["comments"][0]["path"] == "src/foo.lean"
    assert persisted["comments"][0]["line_start"] == 10


async def test_reviewer_resumes_previous_session_and_receives_discussion_log(
    tmp_path: Path,
) -> None:
    if shutil.which("sh") is None:
        pytest.skip("sh executable is required")

    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()
    script = (
        "printf \"$TEND_AGENT_RESUME\" > reviewer-resume-$TEND_AGENT_RESUME.txt; "
        "printf \"%s\" \"$*\" > reviewer-args-$TEND_AGENT_RESUME.txt; "
        "cat \"$TEND_AGENT_DISCUSSION_PATH\" > reviewer-seen-$TEND_AGENT_RESUME.md; "
        "printf '{\"schema_version\":1,\"verdict\":\"approve\",\"notes\":\"Reviewer approves.\"}'"
    )
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=tmp_path / "orch",
            entrypoint=tmp_path / "entrypoint",
            reviewer_agent_command=AsyncOrchestratorAgentCommandConfig(
                argv=("sh", "-c", script, "agent"),
                resume_argv=("--resume-session",),
            ),
        ),
        worktrees=(
            AsyncOrchestratorWorktree(
                worktree_id="worktree_000001",
                path=worktree_path,
                head="abc123",
                state=WorktreeState.REVIEW,
            ),
        ),
    )

    await orchestrator.spawn_reviewer_agents_once_for_test()
    await orchestrator.wait_for_reviewer_agents_for_test()
    assert (worktree_path / "reviewer-resume-0.txt").read_text(encoding="utf-8") == "0"
    assert (worktree_path / "reviewer-args-0.txt").read_text(encoding="utf-8") == ""

    await orchestrator.transition_worktree_for_test("worktree_000001", WorktreeState.REVIEW)
    await orchestrator.spawn_reviewer_agents_once_for_test()
    await orchestrator.wait_for_reviewer_agents_for_test()

    assert (worktree_path / "reviewer-resume-1.txt").read_text(encoding="utf-8") == "1"
    assert (worktree_path / "reviewer-args-1.txt").read_text(encoding="utf-8") == (
        "--resume-session"
    )
    assert "Reviewer approves." in (worktree_path / "reviewer-seen-1.md").read_text(
        encoding="utf-8"
    )
    assert orchestrator.worktrees_by_id["worktree_000001"].reviewer_session_started


async def test_merge_loop_closes_successfully_merged_worktree(tmp_path: Path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable is not available")

    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    _initialize_git_repo(entrypoint)
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(root=root, entrypoint=entrypoint),
    )
    worktree = await orchestrator.create_fresh_worktree_for_test(name="mergeable")
    (worktree.path / "worker-output.txt").write_text("done\n", encoding="utf-8")
    (worktree.path / ".tend").mkdir()
    (worktree.path / ".tend" / "discussion.md").write_text("metadata\n", encoding="utf-8")
    _commit_worktree(worktree.path)
    await orchestrator.transition_worktree_for_test(worktree.worktree_id, WorktreeState.MERGE)

    await orchestrator.process_merge_queue_once_for_test()

    assert orchestrator.merge_queue == ()
    assert orchestrator.worker_queue == ()
    assert orchestrator.worktrees_by_id[worktree.worktree_id].state is WorktreeState.CLOSED
    assert (entrypoint / "worker-output.txt").read_text(encoding="utf-8") == "done\n"
    assert not (entrypoint / ".tend" / "discussion.md").exists()


async def test_merge_lands_only_committed_changes_and_keeps_uncommitted_work(
    tmp_path: Path,
) -> None:
    """The merge honors the worker-owns-commits contract.

    Only what the worker committed is merged; uncommitted work never lands on
    the target branch and is left untouched in the worktree (so a resumed
    session can still commit it).
    """

    if shutil.which("git") is None:
        pytest.skip("git executable is not available")

    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    _initialize_git_repo(entrypoint)
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(root=root, entrypoint=entrypoint),
    )
    worktree = await orchestrator.create_fresh_worktree_for_test(name="partial")
    # The worker commits only the file it intends to land...
    (worktree.path / "committed.txt").write_text("keep\n", encoding="utf-8")
    _commit_worktree(worktree.path, "committed work")
    # ...and leaves unrelated work-in-progress uncommitted in the worktree.
    (worktree.path / "uncommitted.txt").write_text("wip\n", encoding="utf-8")
    await orchestrator.transition_worktree_for_test(worktree.worktree_id, WorktreeState.MERGE)

    await orchestrator.process_merge_queue_once_for_test()

    assert orchestrator.worktrees_by_id[worktree.worktree_id].state is WorktreeState.CLOSED
    # The committed file lands; the uncommitted file is not swept in by add -A.
    assert (entrypoint / "committed.txt").read_text(encoding="utf-8") == "keep\n"
    assert not (entrypoint / "uncommitted.txt").exists()
    # The uncommitted work survives in the worktree rather than being destroyed.
    assert (worktree.path / "uncommitted.txt").read_text(encoding="utf-8") == "wip\n"


async def test_merge_with_nothing_committed_returns_to_worker_with_message(
    tmp_path: Path,
) -> None:
    """A contribution that committed nothing lands nothing and is retried.

    The worktree (and any uncommitted work in it) is left untouched, and the
    worker is told why so it can commit and finish again.
    """

    if shutil.which("git") is None:
        pytest.skip("git executable is not available")

    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    _initialize_git_repo(entrypoint)
    entrypoint_head = _run_git(entrypoint, "rev-parse", "HEAD")
    task = Task(id="task-1", title="Task", summary="Task", description="Do the task.")
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(root=root, entrypoint=entrypoint),
        task_manager=task_manager_with_tasks(task),
    )
    worktree = await orchestrator.create_fresh_worktree_for_test(name="empty", task=task)
    # The worker left work uncommitted and committed nothing.
    (worktree.path / "uncommitted.txt").write_text("wip\n", encoding="utf-8")
    await orchestrator.transition_worktree_for_test(worktree.worktree_id, WorktreeState.MERGE)

    await orchestrator.process_merge_queue_once_for_test()

    updated = orchestrator.worktrees_by_id[worktree.worktree_id]
    # Returned to the worker queue; the entrypoint never advanced.
    assert updated.state is WorktreeState.PENDING
    assert orchestrator.worker_queue == (worktree.worktree_id,)
    assert orchestrator.merge_queue == ()
    assert _run_git(entrypoint, "rev-parse", "HEAD") == entrypoint_head
    # The explanatory message names the target branch and the cause.
    assert updated.discussion[-1].role is AsyncOrchestratorAgentRole.ORCHESTRATOR
    message = updated.discussion[-1].message
    assert "nothing to merge" in message.lower()
    assert "`main`" in message
    # The uncommitted work is preserved for the resumed session.
    assert (worktree.path / "uncommitted.txt").read_text(encoding="utf-8") == "wip\n"


async def test_merge_loop_runs_pre_merge_validation_before_closing(
    tmp_path: Path,
) -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable is not available")
    if shutil.which("sh") is None:
        pytest.skip("sh executable is required")

    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    validation_marker = tmp_path / "pre-merge-validation-ran.txt"
    _initialize_git_repo(entrypoint)
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=root,
            entrypoint=entrypoint,
            pre_merge_validation_commands=(
                AsyncOrchestratorValidationCommandConfig(
                    argv=(
                        "sh",
                        "-c",
                        'test -f worker-output.txt && printf validated > "$1"',
                        "validator",
                        str(validation_marker),
                    ),
                ),
            ),
        ),
    )
    worktree = await orchestrator.create_fresh_worktree_for_test(name="pre-merge-valid")
    (worktree.path / "worker-output.txt").write_text("done\n", encoding="utf-8")
    _commit_worktree(worktree.path)
    await orchestrator.transition_worktree_for_test(worktree.worktree_id, WorktreeState.MERGE)

    await orchestrator.process_merge_queue_once_for_test()

    assert validation_marker.read_text(encoding="utf-8") == "validated"
    assert orchestrator.merge_queue == ()
    assert orchestrator.worker_queue == ()
    assert orchestrator.worktrees_by_id[worktree.worktree_id].state is WorktreeState.CLOSED
    assert (entrypoint / "worker-output.txt").read_text(encoding="utf-8") == "done\n"


async def test_merge_loop_requeues_and_rolls_back_on_pre_merge_validation_failure(
    tmp_path: Path,
) -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable is not available")
    if shutil.which("sh") is None:
        pytest.skip("sh executable is required")

    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    _initialize_git_repo(entrypoint)
    original_head = _run_git(entrypoint, "rev-parse", "HEAD")
    task = Task(id="task-1", title="Task", summary="Task", description="Do the task.")
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=root,
            entrypoint=entrypoint,
            pre_merge_validation_commands=(
                AsyncOrchestratorValidationCommandConfig(
                    argv=(
                        "sh",
                        "-c",
                        "printf pre-merge-out; printf pre-merge-err >&2; exit 7",
                    ),
                ),
            ),
        ),
        task_manager=task_manager_with_tasks(task),
    )
    worktree = await orchestrator.create_fresh_worktree_for_test(
        name="pre-merge-invalid",
        task=task,
    )
    (worktree.path / "worker-output.txt").write_text("done\n", encoding="utf-8")
    _commit_worktree(worktree.path)
    await orchestrator.transition_worktree_for_test(worktree.worktree_id, WorktreeState.MERGE)

    await orchestrator.process_merge_queue_once_for_test()

    assert _run_git(entrypoint, "rev-parse", "HEAD") == original_head
    assert _run_git(entrypoint, "status", "--porcelain") == ""
    assert not (entrypoint / "worker-output.txt").exists()
    assert orchestrator.merge_queue == ()
    assert orchestrator.worker_queue == (worktree.worktree_id,)
    updated = orchestrator.worktrees_by_id[worktree.worktree_id]
    assert updated.state is WorktreeState.PENDING
    assert updated.discussion[-1].role is AsyncOrchestratorAgentRole.ORCHESTRATOR
    assert "Pre-merge validation failed" in updated.discussion[-1].message
    assert "pre-merge-out" in updated.discussion[-1].message
    assert "pre-merge-err" in updated.discussion[-1].message
    discussion_log = worktree.path / ".tend" / "discussion.md"
    assert "Pre-merge validation failed" in discussion_log.read_text(encoding="utf-8")


async def test_merge_loop_uses_configured_target_branch(tmp_path: Path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable is not available")

    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    _initialize_git_repo(entrypoint)
    _run_git(entrypoint, "branch", "release", "main")
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=root,
            entrypoint=entrypoint,
            merge_target_branch="release",
        ),
    )
    worktree = await orchestrator.create_fresh_worktree_for_test(name="release-target")
    (worktree.path / "release-output.txt").write_text("done\n", encoding="utf-8")
    _commit_worktree(worktree.path)
    await orchestrator.transition_worktree_for_test(worktree.worktree_id, WorktreeState.MERGE)

    await orchestrator.process_merge_queue_once_for_test()

    assert orchestrator.worktrees_by_id[worktree.worktree_id].state is WorktreeState.CLOSED
    assert _run_git(entrypoint, "branch", "--show-current") == "release"
    assert (entrypoint / "release-output.txt").read_text(encoding="utf-8") == "done\n"
    _run_git(entrypoint, "checkout", "main")
    assert not (entrypoint / "release-output.txt").exists()


async def test_merge_loop_requeues_without_checkout_when_entrypoint_is_dirty(
    tmp_path: Path,
) -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable is not available")

    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    _initialize_git_repo(entrypoint)
    _run_git(entrypoint, "branch", "release", "main")
    task = Task(id="task-1", title="Task", summary="Task", description="Do the task.")
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=root,
            entrypoint=entrypoint,
            merge_target_branch="release",
        ),
        task_manager=task_manager_with_tasks(task),
    )
    worktree = await orchestrator.create_fresh_worktree_for_test(
        name="dirty-entrypoint",
        task=task,
    )
    (worktree.path / "worker-output.txt").write_text("done\n", encoding="utf-8")
    (entrypoint / "local-notes.txt").write_text("do not overwrite\n", encoding="utf-8")
    await orchestrator.transition_worktree_for_test(worktree.worktree_id, WorktreeState.MERGE)

    await orchestrator.process_merge_queue_once_for_test()

    assert _run_git(entrypoint, "branch", "--show-current") == "main"
    assert "?? local-notes.txt" in _run_git(entrypoint, "status", "--porcelain")
    assert not (entrypoint / "worker-output.txt").exists()
    assert orchestrator.merge_queue == ()
    assert orchestrator.worker_queue == (worktree.worktree_id,)
    updated = orchestrator.worktrees_by_id[worktree.worktree_id]
    assert updated.state is WorktreeState.PENDING
    assert updated.discussion[-1].role is AsyncOrchestratorAgentRole.ORCHESTRATOR
    assert "Entrypoint repository is dirty" in updated.discussion[-1].message
    assert "local-notes.txt" in updated.discussion[-1].message
    discussion_log = worktree.path / ".tend" / "discussion.md"
    assert "Entrypoint repository is dirty" in discussion_log.read_text(encoding="utf-8")


async def test_merge_loop_requeues_when_entrypoint_status_check_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "orch"
    entrypoint = tmp_path / "entrypoint"
    worktree_path = tmp_path / "worktree"
    entrypoint.mkdir()
    worktree_path.mkdir()
    merge_called = False
    task = Task(id="task-1", title="Task", summary="Task", description="Do the task.")
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(root=root, entrypoint=entrypoint),
        task_manager=task_manager_with_tasks(task),
        worktrees=(
            AsyncOrchestratorWorktree(
                worktree_id="worktree_000001",
                task_id=task.id,
                path=worktree_path,
                head="abc123",
                state=WorktreeState.MERGE,
            ),
        ),
    )

    def fake_status(repo: Path) -> str:
        del repo
        raise subprocess.CalledProcessError(
            returncode=128,
            cmd=("git", "status", "--porcelain"),
            stderr="fatal: not a git repository",
        )

    def fake_merge(
        *,
        entrypoint: Path,
        worktree: Path,
        commit_message: str,
        target_branch: str,
    ) -> None:
        nonlocal merge_called
        del entrypoint, worktree, commit_message, target_branch
        merge_called = True

    monkeypatch.setattr(orchestrator_module, "_git_status_porcelain", fake_status)
    monkeypatch.setattr(orchestrator_module, "_merge_worktree_into_target_branch", fake_merge)

    await orchestrator.merge_worktree_id_for_test("worktree_000001")

    assert not merge_called
    worktree = orchestrator.worktrees_by_id["worktree_000001"]
    assert worktree.state is WorktreeState.PENDING
    assert orchestrator.worker_queue == ("worktree_000001",)
    message = worktree.discussion[-1].message
    assert "Entrypoint repository status check failed" in message
    assert "fatal: not a git repository" in message
    assert "merge race" not in message
    assert "git merge main" not in message
    assert "git rebase main" not in message


async def test_merge_path_serializes_success_with_runtime_merge_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class LockCheckingOrchestrator(WorktreeTestingOrchestrator):
        closed_transition_held_lock = False

        async def _transition_worktree(
            self,
            worktree_id: str,
            state: WorktreeState,
            *,
            expected: WorktreeState | None = None,
        ) -> None:
            if state is WorktreeState.CLOSED:
                self.closed_transition_held_lock = self.runtime.merge_lock.locked()
            await super()._transition_worktree(worktree_id, state, expected=expected)

    root = tmp_path / "orch"
    entrypoint = tmp_path / "entrypoint"
    worktree_path = tmp_path / "worktree"
    entrypoint.mkdir()
    worktree_path.mkdir()
    merge_called_with_lock: bool | None = None
    # Exercises the legacy in-entrypoint merge helper (mocked below). The default
    # staging merge path is covered in test_merge_staging_worktree.py.
    orchestrator = LockCheckingOrchestrator(
        AsyncOrchestratorConfig(
            root=root, entrypoint=entrypoint, merge_validation_worktree=False
        ),
        worktrees=(
            AsyncOrchestratorWorktree(
                worktree_id="worktree_000001",
                path=worktree_path,
                head="abc123",
                state=WorktreeState.MERGE,
            ),
        ),
    )

    def fake_merge(
        *,
        entrypoint: Path,
        worktree: Path,
        commit_message: str,
        target_branch: str,
    ) -> orchestrator_module._MergeWorktreeResult:  # pyright: ignore[reportPrivateUsage]
        nonlocal merge_called_with_lock
        del entrypoint, worktree, commit_message, target_branch
        merge_called_with_lock = orchestrator.runtime.merge_lock.locked()
        return orchestrator_module._MergeWorktreeResult(  # pyright: ignore[reportPrivateUsage]
            original_head="abc123",
        )

    def fake_status(repo: Path) -> str:
        del repo
        return ""

    def fake_task_check(
        *,
        entrypoint: Path,
        original_head: str,
    ) -> None:
        del entrypoint, original_head
        return None

    monkeypatch.setattr(orchestrator_module, "_git_status_porcelain", fake_status)
    monkeypatch.setattr(orchestrator_module, "_merge_worktree_into_target_branch", fake_merge)
    monkeypatch.setattr(orchestrator_module, "_check_post_merge_task_tree", fake_task_check)

    await orchestrator.runtime.merge_lock.acquire()
    merge_task = asyncio.create_task(
        orchestrator.merge_worktree_id_for_test("worktree_000001")
    )
    try:
        await asyncio.sleep(0)
        assert not merge_task.done()
        assert merge_called_with_lock is None
        assert (
            orchestrator.worktrees_by_id["worktree_000001"].state
            is WorktreeState.MERGE
        )
    finally:
        if orchestrator.runtime.merge_lock.locked():
            orchestrator.runtime.merge_lock.release()

    await merge_task

    assert merge_called_with_lock is True
    assert orchestrator.closed_transition_held_lock
    assert orchestrator.worktrees_by_id["worktree_000001"].state is WorktreeState.CLOSED


async def test_merge_path_holds_runtime_merge_lock_for_failure_feedback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class LockCheckingOrchestrator(WorktreeTestingOrchestrator):
        feedback_transition_held_lock = False

        async def _record_orchestrator_message_and_transition(
            self,
            worktree_id: str,
            *,
            message: str,
            state: WorktreeState,
            expected: WorktreeState = WorktreeState.MERGE,
        ) -> None:
            self.feedback_transition_held_lock = self.runtime.merge_lock.locked()
            await super()._record_orchestrator_message_and_transition(
                worktree_id,
                message=message,
                state=state,
                expected=expected,
            )

    root = tmp_path / "orch"
    entrypoint = tmp_path / "entrypoint"
    worktree_path = tmp_path / "worktree"
    entrypoint.mkdir()
    worktree_path.mkdir()
    merge_called_with_lock: bool | None = None
    # Exercises the legacy in-entrypoint merge helper (mocked below). The default
    # staging merge path is covered in test_merge_staging_worktree.py.
    orchestrator = LockCheckingOrchestrator(
        AsyncOrchestratorConfig(
            root=root, entrypoint=entrypoint, merge_validation_worktree=False
        ),
        worktrees=(
            AsyncOrchestratorWorktree(
                worktree_id="worktree_000001",
                path=worktree_path,
                head="abc123",
                state=WorktreeState.MERGE,
            ),
        ),
    )

    def fake_merge(
        *,
        entrypoint: Path,
        worktree: Path,
        commit_message: str,
        target_branch: str,
    ) -> None:
        nonlocal merge_called_with_lock
        del entrypoint, worktree, commit_message, target_branch
        merge_called_with_lock = orchestrator.runtime.merge_lock.locked()
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=("git", "merge"),
            stderr="conflict details",
        )

    def fake_status(repo: Path) -> str:
        del repo
        return ""

    monkeypatch.setattr(orchestrator_module, "_git_status_porcelain", fake_status)
    monkeypatch.setattr(orchestrator_module, "_merge_worktree_into_target_branch", fake_merge)

    await orchestrator.merge_worktree_id_for_test("worktree_000001")

    assert merge_called_with_lock is True
    assert orchestrator.feedback_transition_held_lock
    worktree = orchestrator.worktrees_by_id["worktree_000001"]
    assert worktree.state is WorktreeState.PENDING
    assert worktree.discussion[-1].role is AsyncOrchestratorAgentRole.ORCHESTRATOR
    assert "conflict details" in worktree.discussion[-1].message


async def test_ready_task_gets_fresh_worktree_after_merge_closes_old_worktree(
    tmp_path: Path,
) -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable is not available")

    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    ready = Task(id="task-1", title="Ready", summary="Ready", description="Ready to run.")
    _initialize_git_repo(entrypoint)
    _write_and_commit_tasks(entrypoint, ready)
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(root=root, entrypoint=entrypoint),
        task_manager=task_manager_with_tasks(ready),
    )
    worktree = await orchestrator.create_fresh_worktree_for_test(task=ready)
    (worktree.path / "worker-output.txt").write_text("done\n", encoding="utf-8")
    _commit_worktree(worktree.path)
    await orchestrator.transition_worktree_for_test(worktree.worktree_id, WorktreeState.MERGE)

    await orchestrator.process_merge_queue_once_for_test()
    await orchestrator.ensure_ready_task_worktrees_once_for_test()

    assert orchestrator.merge_queue == ()
    assert orchestrator.worker_queue == ("worktree_000002",)
    assert worktree_ids_for_task(orchestrator.store, ready) == (
        "worktree_000001",
        "worktree_000002",
    )
    assert orchestrator.worktrees_by_id[worktree.worktree_id].state is WorktreeState.CLOSED
    assert orchestrator.worktrees_by_id["worktree_000002"].state is WorktreeState.PENDING
    assert (entrypoint / "worker-output.txt").read_text(encoding="utf-8") == "done\n"


def test_merge_failure_discussion_message_recovers_with_named_target_branch() -> None:
    """The discussion message must give the worker enough to recover without
    redoing the task: the configured target branch name, the concrete
    ``git merge <target>`` command, the "do not redo" reassurance, and the
    conflicting file from git's output. Exercising a non-default target branch
    confirms the template is parameterised (not hard-coded to ``main``)."""

    exc = subprocess.CalledProcessError(
        returncode=1,
        cmd=["git", "merge", "--no-edit", "-m", "commit", "worker-head"],
        output="",
        stderr=(
            "Auto-merging conflict.txt\n"
            "CONFLICT (content): Merge conflict in conflict.txt\n"
            "Automatic merge failed; fix conflicts and then commit the result.\n"
        ),
    )
    message = orchestrator_module._merge_failure_discussion_message(  # pyright: ignore[reportPrivateUsage]
        exc, target_branch="develop"
    )
    assert "`develop`" in message
    assert "git merge develop" in message
    assert "git rebase develop" in message
    assert "preserved" in message.lower()
    assert "do not redo" in message.lower()
    assert "conflict.txt" in message  # surfaced from the included git stderr
    # Don't let "main" leak in when a different target branch is configured.
    assert "git merge main" not in message
    assert "`main`" not in message


async def test_merge_loop_requeues_worktree_when_merge_fails(tmp_path: Path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable is not available")

    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    _initialize_git_repo(entrypoint)
    (entrypoint / "conflict.txt").write_text("base\n", encoding="utf-8")
    _run_git(entrypoint, "add", "conflict.txt")
    _run_git(entrypoint, "commit", "-m", "add conflict base")
    task = Task(id="task-1", title="Task", summary="Task", description="Do the task.")
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(root=root, entrypoint=entrypoint),
        task_manager=task_manager_with_tasks(task),
    )
    worktree = await orchestrator.create_fresh_worktree_for_test(
        name="conflicting",
        task=task,
    )
    (entrypoint / "conflict.txt").write_text("main\n", encoding="utf-8")
    _run_git(entrypoint, "add", "conflict.txt")
    _run_git(entrypoint, "commit", "-m", "main change")
    (worktree.path / "conflict.txt").write_text("worker\n", encoding="utf-8")
    _commit_worktree(worktree.path)
    await orchestrator.transition_worktree_for_test(worktree.worktree_id, WorktreeState.MERGE)

    await orchestrator.process_merge_queue_once_for_test()

    assert orchestrator.merge_queue == ()
    assert orchestrator.worker_queue == (worktree.worktree_id,)
    updated = orchestrator.worktrees_by_id[worktree.worktree_id]
    assert updated.state is WorktreeState.PENDING
    assert updated.discussion[-1].role is AsyncOrchestratorAgentRole.ORCHESTRATOR
    message = updated.discussion[-1].message
    # The recovery message must (a) name the target branch concretely,
    # (b) reassure the worker its prior work is preserved (don't restart
    # from scratch), (c) give the concrete `git merge <target>` recovery
    # command, and (d) surface the conflicting file.
    assert "`main`" in message
    assert "preserved" in message.lower() and "do not redo" in message.lower()
    assert "git merge main" in message
    assert "conflict.txt" in message
    discussion_log = worktree.path / ".tend" / "discussion.md"
    assert "## 1. Orchestrator" in discussion_log.read_text(encoding="utf-8")
    assert "conflict.txt" in discussion_log.read_text(encoding="utf-8")
    assert _run_git(entrypoint, "status", "--porcelain") == ""


async def test_run_registers_control_store_and_applies_noop_command(
    tmp_path: Path,
) -> None:
    discovery_started = asyncio.Event()

    class BlockingDiscoveryOrchestrator(AsyncOrchestrator):
        async def _enqueue_ready_tasks_forever(self) -> None:
            discovery_started.set()
            await asyncio.Event().wait()

    orchestrator = BlockingDiscoveryOrchestrator(
        AsyncOrchestratorConfig(root=tmp_path / "orch", entrypoint=tmp_path / "entrypoint"),
    )
    run_task = asyncio.create_task(orchestrator.run())

    try:
        await asyncio.wait_for(discovery_started.wait(), timeout=1.0)
        run = orchestrator.control_store.latest_run()
        assert run is not None
        assert run.status == "running"
        assert run.worker_limit == orchestrator.runtime.worker_agent_limit
        assert run.reviewer_limit == orchestrator.runtime.reviewer_agent_limit

        first_heartbeat = run.heartbeat_at
        refreshed_run = run
        for _ in range(50):
            await asyncio.sleep(0.05)
            latest = orchestrator.control_store.latest_run()
            assert latest is not None
            if latest.heartbeat_at != first_heartbeat:
                refreshed_run = latest
                break
        else:  # pragma: no cover - assertion message for rare scheduler stalls
            pytest.fail("control heartbeat was not refreshed")
        assert refreshed_run.status == "running"

        pending = orchestrator.control_store.enqueue_command(
            "noop",
            params={"source": "test"},
        )
        completed = pending
        for _ in range(50):
            await asyncio.sleep(0.05)
            latest_command = orchestrator.control_store.get_command(pending.id)
            assert latest_command is not None
            if latest_command.status == "succeeded":
                completed = latest_command
                break
        else:  # pragma: no cover - assertion message for rare scheduler stalls
            pytest.fail("noop control command was not completed")

        assert completed.run_id == refreshed_run.run_id
        assert completed.result == {"applied": True}
        latest_run = orchestrator.control_store.latest_run()
        assert latest_run is not None
        assert latest_run.applied_command_id == pending.id
    finally:
        run_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run_task

    final_run = orchestrator.control_store.latest_run()
    assert final_run is not None
    assert final_run.status == "failed"
    assert final_run.status_reason == "cancelled"


async def test_control_service_retries_transient_store_io_errors(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    discovery_started = asyncio.Event()
    heartbeat_failed = asyncio.Event()
    heartbeat_retried = asyncio.Event()
    event_loop = asyncio.get_running_loop()

    class FlakyHeartbeatControlStore(SQLiteAsyncOrchestratorStore):
        fail_next_heartbeat: bool

        def __init__(self, root: Path) -> None:
            super().__init__(root)
            self.fail_next_heartbeat = True

        def record_run_heartbeat(
            self,
            *,
            run_id: str,
            status: ControlRunStatus,
            status_reason: str | None = None,
            worker_limit: int | None = None,
            reviewer_limit: int | None = None,
            paused: bool = False,
            drain_requested: bool = False,
            active_agents: Sequence[ControlActiveAgentSnapshot] = (),
        ) -> ControlRunRecord:
            if self.fail_next_heartbeat:
                self.fail_next_heartbeat = False
                event_loop.call_soon_threadsafe(heartbeat_failed.set)
                raise AsyncOrchestratorControlStoreIOError("temporary control-store outage")
            event_loop.call_soon_threadsafe(heartbeat_retried.set)
            return super().record_run_heartbeat(
                run_id=run_id,
                status=status,
                status_reason=status_reason,
                worker_limit=worker_limit,
                reviewer_limit=reviewer_limit,
                paused=paused,
                drain_requested=drain_requested,
                active_agents=active_agents,
            )

    class BlockingDiscoveryOrchestrator(AsyncOrchestrator):
        async def _enqueue_ready_tasks_forever(self) -> None:
            discovery_started.set()
            await asyncio.Event().wait()

    caplog.set_level(logging.WARNING, logger=orchestrator_module.__name__)
    orchestrator = BlockingDiscoveryOrchestrator(
        AsyncOrchestratorConfig(root=tmp_path / "orch", entrypoint=tmp_path / "entrypoint"),
    )
    orchestrator.control_store = FlakyHeartbeatControlStore(orchestrator.config.root)
    run_task = asyncio.create_task(orchestrator.run())

    try:
        await asyncio.wait_for(discovery_started.wait(), timeout=1.0)
        await asyncio.wait_for(heartbeat_failed.wait(), timeout=1.0)
        await asyncio.wait_for(heartbeat_retried.wait(), timeout=1.0)
        assert not run_task.done()
        assert "temporary control-store outage" in caplog.text
    finally:
        run_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run_task


async def test_control_service_preserves_cancellation_when_store_call_fails(
    tmp_path: Path,
) -> None:
    discovery_started = asyncio.Event()
    heartbeat_started = asyncio.Event()
    release_heartbeat = threading.Event()
    event_loop = asyncio.get_running_loop()

    class BlockingFailingHeartbeatControlStore(SQLiteAsyncOrchestratorStore):
        def record_run_heartbeat(
            self,
            *,
            run_id: str,
            status: ControlRunStatus,
            status_reason: str | None = None,
            worker_limit: int | None = None,
            reviewer_limit: int | None = None,
            paused: bool = False,
            drain_requested: bool = False,
            active_agents: Sequence[ControlActiveAgentSnapshot] = (),
        ) -> ControlRunRecord:
            event_loop.call_soon_threadsafe(heartbeat_started.set)
            release_heartbeat.wait(timeout=1.0)
            raise AsyncOrchestratorControlStoreIOError("heartbeat failed during cancellation")

    class BlockingDiscoveryOrchestrator(AsyncOrchestrator):
        async def _enqueue_ready_tasks_forever(self) -> None:
            discovery_started.set()
            await asyncio.Event().wait()

    orchestrator = BlockingDiscoveryOrchestrator(
        AsyncOrchestratorConfig(root=tmp_path / "orch", entrypoint=tmp_path / "entrypoint"),
    )
    orchestrator.control_store = BlockingFailingHeartbeatControlStore(orchestrator.config.root)
    run_task = asyncio.create_task(orchestrator.run())

    try:
        await asyncio.wait_for(discovery_started.wait(), timeout=1.0)
        await asyncio.wait_for(heartbeat_started.wait(), timeout=1.0)
        run_task.cancel()
        await asyncio.sleep(0)
        release_heartbeat.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(run_task, timeout=1.0)
    finally:
        release_heartbeat.set()
        if not run_task.done():
            run_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await run_task


async def test_control_heartbeat_records_active_agent_snapshot(tmp_path: Path) -> None:
    task = Task(
        id="task-a",
        title="Task A",
        summary="Task A summary",
        description="Task A description",
    )
    worktree = AsyncOrchestratorWorktree(
        worktree_id="worktree_000001",
        path=tmp_path / "worktree",
        head="abc123",
        task_id=task.id,
        state=WorktreeState.WORKER_RUNNING,
    )

    class SummaryOnlyStore(SQLiteAsyncOrchestratorStore):
        __slots__ = ("summary_calls",)

        summary_calls: int

        def __init__(self, root: Path) -> None:
            super().__init__(root)
            self.summary_calls = 0

        def get_worktree(self, worktree_id: str) -> AsyncOrchestratorWorktree | None:
            raise AssertionError(f"heartbeat should not N+1 get {worktree_id}")

        def worktree_control_summaries(
            self,
            worktree_ids: Sequence[str],
        ) -> dict[str, tuple[str | None, str]]:
            self.summary_calls += 1
            return super().worktree_control_summaries(worktree_ids)

    store = SummaryOnlyStore(tmp_path / "orch")
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(root=tmp_path / "orch", entrypoint=tmp_path / "entrypoint"),
        task_manager=TaskManager(tasks=[task]),
        worktrees=(worktree,),
        store=store,
    )
    run_id = "run_test"
    release = asyncio.Event()

    async def running_agent() -> None:
        await release.wait()

    agent_task = asyncio.create_task(running_agent())
    orchestrator.runtime.worker_agent_tasks[worktree.worktree_id] = agent_task
    orchestrator.set_control_run_id_for_test(run_id)
    orchestrator.control_store.register_run(run_id=run_id, pid=123)

    try:
        await orchestrator.record_control_heartbeat_once_for_test()
    finally:
        release.set()
        await agent_task

    agents = orchestrator.control_store.list_active_agents(run_id=run_id)
    assert len(agents) == 1
    assert agents[0].role == "worker"
    assert agents[0].worktree_id == worktree.worktree_id
    assert agents[0].task_id == task.id
    assert agents[0].worktree_state == "worker_running"
    assert store.summary_calls == 1


async def test_control_commands_mutate_runtime_state(tmp_path: Path) -> None:
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(root=tmp_path / "orch", entrypoint=tmp_path / "entrypoint"),
    )
    run_id = "run_test"
    orchestrator.set_control_run_id_for_test(run_id)
    orchestrator.control_store.register_run(run_id=run_id, pid=123)

    async def apply(
        command: ControlCommandName,
        params: dict[str, JsonValue] | None = None,
    ) -> ControlCommandRecord:
        pending = orchestrator.control_store.enqueue_command(
            command,
            params={} if params is None else params,
            run_id=run_id,
        )
        claimed = orchestrator.control_store.claim_pending_command(run_id=run_id)
        assert claimed is not None
        await orchestrator.apply_control_command_for_test(claimed)
        completed = orchestrator.control_store.get_command(pending.id)
        assert completed is not None
        return completed

    paused = await apply("pause")
    assert paused.status == "succeeded"
    assert paused.result == {"paused": True}
    assert orchestrator.runtime.paused is True

    resumed = await apply("resume")
    assert resumed.status == "succeeded"
    assert resumed.result == {"paused": False}
    assert orchestrator.runtime.paused is False

    limits = await apply("limits", {"workers": 0, "reviewers": 2})
    assert limits.status == "succeeded"
    assert limits.result == {"worker_limit": 0, "reviewer_limit": 2}
    assert orchestrator.runtime.worker_agent_limit == 0
    assert orchestrator.runtime.reviewer_agent_limit == 2

    budget = await apply("budget", {"max_cost": "25.50"})
    assert budget.status == "succeeded"
    assert budget.result == {
        "max_cost": "25.50",
        "currency": "USD",
        "budget_stop_recorded": False,
    }
    assert orchestrator.config.budget.max_cost == Decimal("25.50")

    restored_limits = await apply("limits", {"workers": 1})
    assert restored_limits.status == "succeeded"
    assert restored_limits.result == {"worker_limit": 1, "reviewer_limit": 2}
    assert orchestrator.runtime.worker_agent_limit == 1

    drain = await apply("drain")
    assert drain.status == "succeeded"
    assert drain.result == {"drain_requested": True}
    assert orchestrator.runtime.draining is True

    agent_cancelled = asyncio.Event()

    async def agent_task() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            agent_cancelled.set()

    task = asyncio.create_task(agent_task())
    await asyncio.sleep(0)
    orchestrator.runtime.worker_agent_tasks["wt_test"] = task
    stopped = await apply("stop", {"now": True})
    assert stopped.status == "succeeded"
    assert stopped.result == {
        "stopping": True,
        "now": True,
        "cancelled_agent_tasks": 1,
    }
    assert orchestrator.runtime.stopping is True
    assert orchestrator.runtime.worker_agent_tasks == {}
    assert agent_cancelled.is_set()

    run = orchestrator.control_store.get_run(run_id)
    assert run is not None
    assert run.status == "stopping"
    assert run.paused is False
    assert run.drain_requested is True


async def test_stop_now_propagates_value_error_agent_failure(tmp_path: Path) -> None:
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(root=tmp_path / "orch", entrypoint=tmp_path / "entrypoint"),
    )
    run_id = "run_test"
    orchestrator.set_control_run_id_for_test(run_id)
    orchestrator.control_store.register_run(run_id=run_id, pid=123)

    async def failed_agent() -> None:
        raise ValueError("agent validation failed internally")

    task = asyncio.create_task(failed_agent())
    await asyncio.sleep(0)
    orchestrator.runtime.worker_agent_tasks["wt_test"] = task
    pending = orchestrator.control_store.enqueue_command(
        "stop",
        params={"now": True},
        run_id=run_id,
    )
    claimed = orchestrator.control_store.claim_pending_command(run_id=run_id)
    assert claimed is not None

    with pytest.raises(ValueError, match="agent validation failed internally"):
        await orchestrator.apply_control_command_for_test(claimed)

    still_claimed = orchestrator.control_store.get_command(pending.id)
    assert still_claimed is not None
    assert still_claimed.status == "claimed"
    assert orchestrator.runtime.worker_agent_tasks == {}


async def test_stop_now_terminates_uncancellable_validation_subprocess(
    tmp_path: Path,
) -> None:
    if shutil.which("git") is None or shutil.which("sh") is None:
        pytest.skip("git and sh executables are required")

    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    started = tmp_path / "validation-started"
    terminated = tmp_path / "validation-terminated"
    release = tmp_path / "validation-release"
    pidfile = tmp_path / "validation.pid"
    _initialize_git_repo(entrypoint)
    _write_and_commit_tasks(
        entrypoint,
        Task(id="task-1", title="Task", summary="Task", description="Do it."),
    )
    worker_output = json.dumps(
        {"schema_version": 1, "status": "completed", "summary": "done"}
    )
    validation_script = (
        "echo \"$$\" > \"$1\"; touch \"$2\"; "
        "trap 'touch \"$3\"; exit 0' TERM; "
        "while [ ! -f \"$4\" ]; do sleep 0.1; done"
    )
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=root,
            entrypoint=entrypoint,
            worker_agent_command=AsyncOrchestratorAgentCommandConfig(
                argv=("sh", "-c", 'printf "%s\\n" "$1"', "worker", worker_output),
            ),
            validation_commands=(
                AsyncOrchestratorValidationCommandConfig(
                    argv=(
                        "sh",
                        "-c",
                        validation_script,
                        "validation",
                        str(pidfile),
                        str(started),
                        str(terminated),
                        str(release),
                    ),
                ),
            ),
        ),
    )
    run_id = "run_test"
    orchestrator.set_control_run_id_for_test(run_id)
    orchestrator.control_store.register_run(run_id=run_id, pid=123)

    await orchestrator.sync_task_manager_once_for_test()
    await orchestrator.ensure_ready_task_worktrees_once_for_test()
    await orchestrator.spawn_worker_agents_once_for_test()
    await _wait_for_path(started)

    async def release_validation_later() -> None:
        await asyncio.sleep(1.5)
        release.write_text("release\n", encoding="utf-8")

    pending: ControlCommandRecord | None = None
    elapsed = 0.0
    watchdog = asyncio.create_task(release_validation_later())
    try:
        pending = orchestrator.control_store.enqueue_command(
            "stop",
            params={"now": True},
            run_id=run_id,
        )
        claimed = orchestrator.control_store.claim_pending_command(run_id=run_id)
        assert claimed is not None

        started_at = asyncio.get_running_loop().time()
        await orchestrator.apply_control_command_for_test(claimed)
        elapsed = asyncio.get_running_loop().time() - started_at
    finally:
        watchdog.cancel()
        try:
            await watchdog
        except asyncio.CancelledError:
            pass
        if pidfile.exists() and not terminated.exists():
            try:
                os.kill(int(pidfile.read_text(encoding="utf-8")), signal.SIGKILL)
            except ProcessLookupError:
                pass

    assert pending is not None
    completed = orchestrator.control_store.get_command(pending.id)
    assert completed is not None
    assert completed.status == "succeeded"
    assert elapsed < 1.0
    assert terminated.exists()
    assert not release.exists()
    assert orchestrator.runtime.worker_agent_tasks == {}


async def test_stop_now_kills_validation_child_after_leader_exits(
    tmp_path: Path,
) -> None:
    if os.name == "nt" or shutil.which("git") is None or shutil.which("sh") is None:
        pytest.skip("git and POSIX sh executables are required")

    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    started = tmp_path / "validation-started"
    leader_pidfile = tmp_path / "validation-leader.pid"
    child_pidfile = tmp_path / "validation-child.pid"
    _initialize_git_repo(entrypoint)
    _write_and_commit_tasks(
        entrypoint,
        Task(id="task-1", title="Task", summary="Task", description="Do it."),
    )
    worker_output = json.dumps(
        {"schema_version": 1, "status": "completed", "summary": "done"}
    )
    validation_script = (
        "echo \"$$\" > \"$1\"; "
        "trap 'exit 143' TERM; "
        "( trap '' TERM; echo \"$$\" > \"$2\"; touch \"$3\"; "
        "while :; do sleep 1; done ) & "
        "wait"
    )
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=root,
            entrypoint=entrypoint,
            worker_agent_command=AsyncOrchestratorAgentCommandConfig(
                argv=("sh", "-c", 'printf "%s\\n" "$1"', "worker", worker_output),
            ),
            validation_commands=(
                AsyncOrchestratorValidationCommandConfig(
                    argv=(
                        "sh",
                        "-c",
                        validation_script,
                        "validation",
                        str(leader_pidfile),
                        str(child_pidfile),
                        str(started),
                    ),
                ),
            ),
        ),
    )
    run_id = "run_test"
    orchestrator.set_control_run_id_for_test(run_id)
    orchestrator.control_store.register_run(run_id=run_id, pid=123)

    await orchestrator.sync_task_manager_once_for_test()
    await orchestrator.ensure_ready_task_worktrees_once_for_test()
    await orchestrator.spawn_worker_agents_once_for_test()
    await _wait_for_path(started)
    child_pid = int(child_pidfile.read_text(encoding="utf-8"))

    try:
        pending = orchestrator.control_store.enqueue_command(
            "stop",
            params={"now": True},
            run_id=run_id,
        )
        claimed = orchestrator.control_store.claim_pending_command(run_id=run_id)
        assert claimed is not None

        started_at = asyncio.get_running_loop().time()
        await asyncio.wait_for(
            orchestrator.apply_control_command_for_test(claimed),
            timeout=3.0,
        )
        elapsed = asyncio.get_running_loop().time() - started_at

        completed = orchestrator.control_store.get_command(pending.id)
        assert completed is not None
        assert completed.status == "succeeded"
        assert elapsed < 2.5
        await _wait_for_pid_to_stop(child_pid)
        assert orchestrator.runtime.worker_agent_tasks == {}
    finally:
        if leader_pidfile.exists():
            try:
                os.killpg(
                    int(leader_pidfile.read_text(encoding="utf-8")),
                    signal.SIGKILL,
                )
            except ProcessLookupError:
                pass
        try:
            os.kill(child_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


async def test_invalid_control_command_params_fail_command(tmp_path: Path) -> None:
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(root=tmp_path / "orch", entrypoint=tmp_path / "entrypoint"),
    )
    run_id = "run_test"
    orchestrator.set_control_run_id_for_test(run_id)
    orchestrator.control_store.register_run(run_id=run_id, pid=123)
    pending = orchestrator.control_store.enqueue_command(
        "limits",
        params={"workers": -1},
        run_id=run_id,
    )
    claimed = orchestrator.control_store.claim_pending_command(run_id=run_id)
    assert claimed is not None

    await orchestrator.apply_control_command_for_test(claimed)

    completed = orchestrator.control_store.get_command(pending.id)
    assert completed is not None
    assert completed.status == "failed"
    assert completed.error == "workers must be a non-negative integer"
    assert orchestrator.runtime.worker_agent_limit == 20


@pytest.mark.parametrize(
    ("limit_params", "expected_error"),
    (
        (
            {"workers": 0},
            "cannot request drain while worker launch limit is 0; "
            "raise the worker limit or use stop semantics",
        ),
        (
            {"reviewers": 0},
            "cannot request drain while reviewer launch limit is 0; "
            "raise the reviewer limit or use stop semantics",
        ),
    ),
)
async def test_drain_command_rejects_zero_agent_limits(
    tmp_path: Path,
    limit_params: dict[str, JsonValue],
    expected_error: str,
) -> None:
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(root=tmp_path / "orch", entrypoint=tmp_path / "entrypoint"),
    )
    run_id = "run_test"
    orchestrator.set_control_run_id_for_test(run_id)
    orchestrator.control_store.register_run(run_id=run_id, pid=123)

    if "workers" in limit_params:
        orchestrator.runtime.set_worker_agent_limit(0)
    if "reviewers" in limit_params:
        orchestrator.runtime.set_reviewer_agent_limit(0)
    pending = orchestrator.control_store.enqueue_command("drain", run_id=run_id)
    claimed = orchestrator.control_store.claim_pending_command(run_id=run_id)
    assert claimed is not None

    await orchestrator.apply_control_command_for_test(claimed)

    completed = orchestrator.control_store.get_command(pending.id)
    assert completed is not None
    assert completed.status == "failed"
    assert completed.error == expected_error
    assert orchestrator.runtime.draining is False


@pytest.mark.parametrize(
    ("limit_params", "expected_error"),
    (
        (
            {"workers": 0},
            "cannot set worker launch limit to 0 while drain is active; "
            "raise the worker limit or use stop semantics",
        ),
        (
            {"reviewers": 0},
            "cannot set reviewer launch limit to 0 while drain is active; "
            "raise the reviewer limit or use stop semantics",
        ),
    ),
)
async def test_limits_command_rejects_zero_agent_limits_during_drain(
    tmp_path: Path,
    limit_params: dict[str, JsonValue],
    expected_error: str,
) -> None:
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(root=tmp_path / "orch", entrypoint=tmp_path / "entrypoint"),
    )
    run_id = "run_test"
    orchestrator.set_control_run_id_for_test(run_id)
    orchestrator.control_store.register_run(run_id=run_id, pid=123)
    orchestrator.request_drain_for_test()
    pending = orchestrator.control_store.enqueue_command(
        "limits",
        params=limit_params,
        run_id=run_id,
    )
    claimed = orchestrator.control_store.claim_pending_command(run_id=run_id)
    assert claimed is not None

    await orchestrator.apply_control_command_for_test(claimed)

    completed = orchestrator.control_store.get_command(pending.id)
    assert completed is not None
    assert completed.status == "failed"
    assert completed.error == expected_error
    assert orchestrator.runtime.worker_agent_limit == 20
    assert orchestrator.runtime.reviewer_agent_limit == 20


async def test_run_cancels_claimed_control_command_on_shutdown(tmp_path: Path) -> None:
    discovery_started = asyncio.Event()
    command_claimed = asyncio.Event()

    class BlockingControlOrchestrator(AsyncOrchestrator):
        async def _enqueue_ready_tasks_forever(self) -> None:
            discovery_started.set()
            await asyncio.Event().wait()

        async def _apply_control_command(self, command: ControlCommandRecord) -> None:
            command_claimed.set()
            await asyncio.Event().wait()

    orchestrator = BlockingControlOrchestrator(
        AsyncOrchestratorConfig(root=tmp_path / "orch", entrypoint=tmp_path / "entrypoint"),
    )
    run_task = asyncio.create_task(orchestrator.run())

    await asyncio.wait_for(discovery_started.wait(), timeout=1.0)
    pending = orchestrator.control_store.enqueue_command("noop")
    await asyncio.wait_for(command_claimed.wait(), timeout=1.0)
    run_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run_task

    cancelled = orchestrator.control_store.get_command(pending.id)
    assert cancelled is not None
    assert cancelled.status == "cancelled"
    assert cancelled.error == "run was cancelled before command completed"


async def test_run_starts_background_runners_until_cancelled(tmp_path: Path) -> None:
    discovery_started = asyncio.Event()
    task_queue_started = asyncio.Event()
    workers_started = asyncio.Event()
    reviewers_started = asyncio.Event()
    merge_started = asyncio.Event()

    class RecordingOrchestrator(AsyncOrchestrator):
        async def _enqueue_ready_tasks_forever(self) -> None:
            discovery_started.set()
            await asyncio.Event().wait()

        async def _process_task_queue_forever(self) -> None:
            task_queue_started.set()
            await asyncio.Event().wait()

        async def _spawn_worker_agents_forever(self) -> None:
            workers_started.set()
            await asyncio.Event().wait()

        async def _spawn_reviewer_agents_forever(self) -> None:
            reviewers_started.set()
            await asyncio.Event().wait()

        async def _process_merge_queue_forever(self) -> None:
            merge_started.set()
            await asyncio.Event().wait()

    orchestrator = RecordingOrchestrator(
        AsyncOrchestratorConfig(root=tmp_path / "orch", entrypoint=tmp_path / "entrypoint"),
    )
    run_task = asyncio.create_task(orchestrator.run())

    await asyncio.wait_for(discovery_started.wait(), timeout=1.0)
    await asyncio.wait_for(task_queue_started.wait(), timeout=1.0)
    await asyncio.wait_for(workers_started.wait(), timeout=1.0)
    await asyncio.wait_for(reviewers_started.wait(), timeout=1.0)
    await asyncio.wait_for(merge_started.wait(), timeout=1.0)
    run_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run_task


async def test_run_stops_when_all_tasks_are_complete(tmp_path: Path) -> None:
    complete = Task(
        id="task-1",
        title="Done", summary="Done",
        description="Already complete.",
        status=TaskStatus.COMPLETE,
    )
    entrypoint = tmp_path / "entrypoint"
    # The committed tasks/ must agree: completion is confirmed against the
    # authoritative on-disk view, not only the advisory in-memory state.
    write_task(entrypoint / "tasks" / "task-1.yaml", complete)

    class CompleteTasksOrchestrator(WorktreeTestingOrchestrator):
        async def _sync_task_manager_once(self) -> None:
            return None

    orchestrator = CompleteTasksOrchestrator(
        AsyncOrchestratorConfig(root=tmp_path / "orch", entrypoint=entrypoint),
        task_manager=task_manager_with_tasks(complete),
    )

    result = await asyncio.wait_for(orchestrator.run(), timeout=1.0)

    assert result.root == tmp_path / "orch"
    assert result.entrypoint == tmp_path / "entrypoint"
    assert result.stop is not None
    assert result.stop.reason == "all_tasks_complete"


async def test_run_result_records_operator_drain_stop(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger=orchestrator_module.__name__)

    class DrainAtStartOrchestrator(AsyncOrchestrator):
        async def _sync_task_manager_once(self) -> None:
            self.runtime.request_drain()

    orchestrator = DrainAtStartOrchestrator(
        AsyncOrchestratorConfig(root=tmp_path / "orch", entrypoint=tmp_path / "entrypoint"),
    )

    result = await asyncio.wait_for(orchestrator.run(), timeout=1.0)

    assert result.stop is not None
    assert result.stop.reason == "operator_drain"
    assert result.budget_stop is None
    assert "async orchestrator run settled: reason=operator_drain" in caplog.messages
    assert "all async orchestrator tasks are complete" not in caplog.messages


async def test_terminal_control_stop_precedes_all_tasks_complete(tmp_path: Path) -> None:
    complete = Task(
        id="task-1",
        title="Done",
        summary="Done",
        description="Already complete.",
        status=TaskStatus.COMPLETE,
    )
    entrypoint = tmp_path / "entrypoint"
    write_task(entrypoint / "tasks" / "task-1.yaml", complete)

    class DrainCompleteTasksOrchestrator(WorktreeTestingOrchestrator):
        async def _sync_task_manager_once(self) -> None:
            self.runtime.request_drain()

    orchestrator = DrainCompleteTasksOrchestrator(
        AsyncOrchestratorConfig(root=tmp_path / "orch", entrypoint=entrypoint),
        task_manager=task_manager_with_tasks(complete),
    )

    result = await asyncio.wait_for(orchestrator.run(), timeout=1.0)

    assert result.stop is not None
    assert result.stop.reason == "operator_drain"


async def test_operator_drain_run_drains_worker_review_and_merge_queues(
    tmp_path: Path,
) -> None:
    worker_task = Task(
        id="task-worker",
        title="Worker",
        summary="Worker",
        description="Run worker.",
    )
    review_task = Task(
        id="task-review",
        title="Review",
        summary="Review",
        description="Run reviewer.",
    )
    pending_worktree = AsyncOrchestratorWorktree(
        worktree_id="worktree_000001",
        task_id=worker_task.id,
        path=tmp_path / "pending-worktree",
        head="abc123",
        state=WorktreeState.PENDING,
    )
    review_worktree = AsyncOrchestratorWorktree(
        worktree_id="worktree_000002",
        task_id=review_task.id,
        path=tmp_path / "review-worktree",
        head="def456",
        state=WorktreeState.REVIEW,
    )

    class DrainQueuedWorkOrchestrator(WorktreeTestingOrchestrator):
        async def _sync_task_manager_once(self) -> None:
            self.runtime.request_drain()

        async def _run_worker_agent_for_worktree_id(self, worktree_id: str) -> None:
            await self._transition_worktree(worktree_id, WorktreeState.REVIEW)

        async def _run_reviewer_agent_for_worktree_id(self, worktree_id: str) -> None:
            await self._transition_worktree(worktree_id, WorktreeState.MERGE)

        async def _merge_worktree_id(self, worktree_id: str) -> None:
            await self._transition_worktree(worktree_id, WorktreeState.CLOSED)

    orchestrator = DrainQueuedWorkOrchestrator(
        AsyncOrchestratorConfig(
            root=tmp_path / "orch",
            entrypoint=tmp_path / "entrypoint",
            worker_agent_command=AsyncOrchestratorAgentCommandConfig(argv=("noop",)),
            reviewer_agent_command=AsyncOrchestratorAgentCommandConfig(argv=("noop",)),
            cleanup_closed_worktrees=False,
        ),
        task_manager=task_manager_with_tasks(worker_task, review_task),
        worktrees=(pending_worktree, review_worktree),
    )

    result = await asyncio.wait_for(orchestrator.run(), timeout=3.0)

    assert result.stop is not None
    assert result.stop.reason == "operator_drain"
    assert orchestrator.worker_queue == ()
    assert orchestrator.review_queue == ()
    assert orchestrator.merge_queue == ()
    assert (
        orchestrator.worktrees_by_id[pending_worktree.worktree_id].state
        is WorktreeState.CLOSED
    )
    assert (
        orchestrator.worktrees_by_id[review_worktree.worktree_id].state
        is WorktreeState.CLOSED
    )


async def test_create_fresh_worktree_rejects_existing_target(tmp_path: Path) -> None:
    root = tmp_path / "orch"
    entrypoint = tmp_path / "entrypoint"
    target = root / "worktrees" / "entrypoint-copy"
    target.mkdir(parents=True)

    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(root=root, entrypoint=entrypoint),
    )

    with pytest.raises(FileExistsError, match="worktree path already exists"):
        await orchestrator.create_fresh_worktree_for_test(name="entrypoint-copy")


def task_manager_with_tasks(*tasks: Task) -> TaskManager:
    return TaskManager(tasks=list(tasks))


class _StubUsageOrchestrator(WorktreeTestingOrchestrator):
    """Testing orchestrator whose aggregate usage cost is fixed for budget tests.

    Stubs the strict per-session aggregator the budget guard goes through so a
    test can choose the accumulated cost (and currency) directly. A stub cost in
    a different currency than the budget reproduces the per-session currency
    mismatch path and raises ``AsyncCostBudgetCurrencyMismatchError`` — same
    fail-closed semantics as the production aggregator.
    """

    _stub_cost: Cost | None = None

    def set_stub_cost(self, cost: Cost | None) -> None:
        self._stub_cost = cost

    def _accumulated_cost_strict(  # type: ignore[override]
        self,
        worktrees: object,
        currency: str,
    ) -> Cost | None:
        if self._stub_cost is None:
            return None
        if self._stub_cost.currency != currency:
            raise AsyncCostBudgetCurrencyMismatchError(
                "stub cost currency mismatch: "
                f"stub is in {self._stub_cost.currency} but budget is configured in {currency}"
            )
        return self._stub_cost.model_copy(deep=True)


def _budget_orchestrator(tmp_path: Path, *, max_cost: str, currency: str = "USD") -> (
    _StubUsageOrchestrator
):
    return _StubUsageOrchestrator(
        AsyncOrchestratorConfig(
            root=tmp_path / "orch",
            entrypoint=tmp_path / "entrypoint",
            budget=AsyncOrchestratorBudgetConfig(max_cost=Decimal(max_cost), currency=currency),
        ),
    )


async def test_budget_command_succeeds_before_fresh_ceiling_can_stop_run(
    tmp_path: Path,
) -> None:
    first_cost_started = asyncio.Event()
    release_first_cost = threading.Event()
    completion_requested = asyncio.Event()
    first_cost_call = True
    first_cost_lock = threading.Lock()
    event_loop = asyncio.get_running_loop()

    class BudgetRaceOrchestrator(_StubUsageOrchestrator):
        def _accumulated_cost_strict(  # type: ignore[override]
            self,
            worktrees: object,
            currency: str,
        ) -> Cost | None:
            nonlocal first_cost_call
            wait_for_release = False
            with first_cost_lock:
                if first_cost_call:
                    first_cost_call = False
                    wait_for_release = True
            if wait_for_release:
                event_loop.call_soon_threadsafe(first_cost_started.set)
                release_first_cost.wait(timeout=2.0)
            return Cost(amount=Decimal("10"), currency=currency)

        async def _enqueue_ready_tasks_forever(self) -> None:
            while True:
                if self._budget_stop is None:
                    await self._record_budget_stop_if_exceeded()
                terminal_stop = self._admission_policy().terminal_stop
                if terminal_stop is not None and await self._in_flight_work_settled():
                    self._complete_run(terminal_stop)
                await asyncio.sleep(0.01)

        def _complete_run(self, stop: AsyncOrchestratorRunStop) -> None:
            completion_requested.set()
            super()._complete_run(stop)

    orchestrator = BudgetRaceOrchestrator(
        AsyncOrchestratorConfig(root=tmp_path / "orch", entrypoint=tmp_path / "entrypoint"),
    )
    pending: ControlCommandRecord | None = None
    result = None
    run_task = asyncio.create_task(orchestrator.run())
    try:
        for _ in range(50):
            if orchestrator.control_store.latest_run() is not None:
                break
            await asyncio.sleep(0.05)
        else:  # pragma: no cover - assertion message for rare scheduler stalls
            pytest.fail("control run was not registered")

        pending = orchestrator.control_store.enqueue_command(
            "budget",
            params={"max_cost": "1"},
        )
        await asyncio.wait_for(first_cost_started.wait(), timeout=2.0)
        await asyncio.sleep(0.1)
        assert not completion_requested.is_set()

        release_first_cost.set()
        result = await asyncio.wait_for(run_task, timeout=2.0)
    finally:
        release_first_cost.set()
        if not run_task.done():
            run_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await run_task

    assert pending is not None
    assert result is not None
    completed = orchestrator.control_store.get_command(pending.id)
    assert completed is not None
    assert completed.status == "succeeded"
    assert completed.result == {
        "max_cost": "1",
        "currency": "USD",
        "budget_stop_recorded": True,
    }
    assert result.stop == AsyncOrchestratorRunStop(reason="max_cost_exceeded")
    final_run = orchestrator.control_store.latest_run()
    assert final_run is not None
    assert final_run.status == "stopped"
    assert final_run.status_reason == "max_cost_exceeded"


async def test_budget_stop_records_when_cost_meets_ceiling(tmp_path: Path) -> None:
    orchestrator = _budget_orchestrator(tmp_path, max_cost="10")
    # Inclusive ceiling: exactly hitting max_cost stops the run.
    orchestrator.set_stub_cost(Cost(amount=Decimal("10"), currency="USD"))

    exceeded = await orchestrator.record_budget_stop_if_exceeded_for_test()

    assert exceeded is True
    budget_stop = orchestrator.budget_stop_for_test
    assert budget_stop is not None
    assert budget_stop.reason == "max_cost_exceeded"
    assert budget_stop.accumulated_cost == "10"
    # First-breach amount is also recorded immutably (mirrors sync #70).
    assert budget_stop.breach_accumulated_cost == "10"
    assert budget_stop.max_cost == "10"
    assert budget_stop.currency == "USD"


async def test_budget_stop_ignores_stale_limit_after_live_budget_update(
    tmp_path: Path,
) -> None:
    class BudgetUpdatingOrchestrator(_StubUsageOrchestrator):
        def _accumulated_cost_strict(  # type: ignore[override]
            self,
            worktrees: object,
            currency: str,
        ) -> Cost | None:
            self.config.budget = self.config.budget.model_copy(
                update={"max_cost": Decimal("20")}
            )
            return Cost(amount=Decimal("15"), currency=currency)

    orchestrator = BudgetUpdatingOrchestrator(
        AsyncOrchestratorConfig(
            root=tmp_path / "orch",
            entrypoint=tmp_path / "entrypoint",
            budget=AsyncOrchestratorBudgetConfig(max_cost=Decimal("10")),
        )
    )

    exceeded = await orchestrator.record_budget_stop_if_exceeded_for_test()

    assert exceeded is False
    assert orchestrator.budget_stop_for_test is None
    assert orchestrator.config.budget.max_cost == Decimal("20")


async def test_budget_stop_not_recorded_below_ceiling(tmp_path: Path) -> None:
    orchestrator = _budget_orchestrator(tmp_path, max_cost="10")
    orchestrator.set_stub_cost(Cost(amount=Decimal("9.99"), currency="USD"))

    exceeded = await orchestrator.record_budget_stop_if_exceeded_for_test()

    assert exceeded is False
    assert orchestrator.budget_stop_for_test is None


async def test_budget_stop_disabled_without_max_cost(tmp_path: Path) -> None:
    orchestrator = _StubUsageOrchestrator(
        AsyncOrchestratorConfig(root=tmp_path / "orch", entrypoint=tmp_path / "entrypoint"),
    )
    orchestrator.set_stub_cost(Cost(amount=Decimal("1000"), currency="USD"))

    assert await orchestrator.record_budget_stop_if_exceeded_for_test() is False
    assert orchestrator.budget_stop_for_test is None


async def test_budget_stop_fails_closed_on_currency_mismatch(tmp_path: Path) -> None:
    # An accumulated cost in a different currency than ``budget.currency`` cannot
    # be compared against the ceiling; the strict aggregator raises and the guard
    # fails closed (matches the shared #70 behavior, replacing the
    # earlier "log + fail open" semantics).
    orchestrator = _budget_orchestrator(tmp_path, max_cost="10", currency="USD")
    orchestrator.set_stub_cost(Cost(amount=Decimal("100"), currency="EUR"))

    with pytest.raises(AsyncCostBudgetCurrencyMismatchError, match="EUR"):
        await orchestrator.record_budget_stop_if_exceeded_for_test()

    # No partial budget-stop record is left behind on the failure path.
    assert orchestrator.budget_stop_for_test is None


async def test_in_flight_work_settled_true_when_pipeline_idle(tmp_path: Path) -> None:
    orchestrator = _budget_orchestrator(tmp_path, max_cost="10")

    # No queued items, no running agents, no worktrees -> settled.
    assert await orchestrator.in_flight_work_settled_for_test() is True


async def test_in_flight_work_not_settled_with_running_reviewer_task(
    tmp_path: Path,
) -> None:
    orchestrator = _budget_orchestrator(tmp_path, max_cost="10")
    release = asyncio.Event()

    async def _pending() -> None:
        await release.wait()

    task = asyncio.create_task(_pending())
    orchestrator.runtime.reviewer_agent_tasks["worktree_000001"] = task
    try:
        assert await orchestrator.in_flight_work_settled_for_test() is False
    finally:
        release.set()
        await task


# ---------------------------------------------------------------------------
# Regression tests for freezing
# cost-incurring stages on a budget breach, settling when the gated spawner
# can no longer prune done agent tasks, refreshing ``accumulated_cost`` while
# keeping ``breach_accumulated_cost`` pinned, and failing closed on a
# per-session currency mismatch.
# ---------------------------------------------------------------------------


def _budget_stop_at(amount: str) -> AsyncOrchestratorBudgetStop:
    """Build a fully-formed budget-stop record (max_cost=``amount``) for tests."""

    return AsyncOrchestratorBudgetStop(
        breach_accumulated_cost=amount,
        accumulated_cost=amount,
        max_cost=amount,
        currency="USD",
    )


async def test_pause_preserves_queued_ready_task_without_creating_worktree(
    tmp_path: Path,
) -> None:
    task = Task(id="task-1", title="Task", summary="Task", description="Do it.")
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(root=tmp_path / "orch", entrypoint=tmp_path / "entrypoint"),
        task_manager=task_manager_with_tasks(task),
    )
    await orchestrator.enqueue_ready_tasks_once_for_test()
    assert orchestrator.task_queue == (task.id,)

    orchestrator.pause_scheduling_for_test()
    await orchestrator.process_task_queue_once_for_test()

    assert orchestrator.task_queue == (task.id,)
    assert orchestrator.worktree_ids == ()


async def test_budget_freeze_preserves_queued_agent_work(tmp_path: Path) -> None:
    task = Task(id="task-1", title="Task", summary="Task", description="Do it.")
    pending_worktree = AsyncOrchestratorWorktree(
        worktree_id="worktree_000001",
        task_id=task.id,
        path=tmp_path / "pending-worktree",
        head="abc123",
        state=WorktreeState.PENDING,
    )
    review_worktree = AsyncOrchestratorWorktree(
        worktree_id="worktree_000002",
        task_id=task.id,
        path=tmp_path / "review-worktree",
        head="def456",
        state=WorktreeState.REVIEW,
    )
    orchestrator = _StubUsageOrchestrator(
        AsyncOrchestratorConfig(
            root=tmp_path / "orch",
            entrypoint=tmp_path / "entrypoint",
            worker_agent_command=AsyncOrchestratorAgentCommandConfig(argv=("noop",)),
            reviewer_agent_command=AsyncOrchestratorAgentCommandConfig(argv=("noop",)),
            budget=AsyncOrchestratorBudgetConfig(max_cost=Decimal("10")),
        ),
        task_manager=task_manager_with_tasks(task),
        worktrees=(pending_worktree, review_worktree),
    )
    orchestrator.force_budget_stop_for_test(_budget_stop_at("10"))

    await orchestrator.spawn_worker_agents_once_for_test()
    await orchestrator.spawn_reviewer_agents_once_for_test()

    assert orchestrator.worker_queue == (pending_worktree.worktree_id,)
    assert orchestrator.review_queue == (review_worktree.worktree_id,)
    assert orchestrator.runtime.worker_agent_tasks == {}
    assert orchestrator.runtime.reviewer_agent_tasks == {}


async def test_operator_drain_waits_for_queued_worktree_work(
    tmp_path: Path,
) -> None:
    active_task = Task(id="task-1", title="Task", summary="Task", description="Do it.")
    ready_task = Task(
        id="task-2", title="Ready", summary="Ready", description="Do it too."
    )
    pending_worktree = AsyncOrchestratorWorktree(
        worktree_id="worktree_000001",
        task_id=active_task.id,
        path=tmp_path / "pending-worktree",
        head="abc123",
        state=WorktreeState.PENDING,
    )
    review_worktree = AsyncOrchestratorWorktree(
        worktree_id="worktree_000002",
        task_id=active_task.id,
        path=tmp_path / "review-worktree",
        head="def456",
        state=WorktreeState.REVIEW,
    )
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(root=tmp_path / "orch", entrypoint=tmp_path / "entrypoint"),
        task_manager=task_manager_with_tasks(active_task, ready_task),
        worktrees=(pending_worktree, review_worktree),
    )
    await orchestrator.enqueue_ready_tasks_once_for_test()
    orchestrator.request_drain_for_test()

    assert orchestrator.task_queue == (ready_task.id,)
    assert orchestrator.worker_queue == (pending_worktree.worktree_id,)
    assert orchestrator.review_queue == (review_worktree.worktree_id,)
    assert await orchestrator.in_flight_work_settled_for_test() is False

    # Even if a queue item was already claimed, durable worktree state prevents
    # operator drain from exiting until the worktree reaches CLOSED.
    assert orchestrator.runtime.worker_queue.claim(pending_worktree.worktree_id) is True
    assert orchestrator.runtime.review_queue.claim(review_worktree.worktree_id) is True
    assert await orchestrator.in_flight_work_settled_for_test() is False


async def test_operator_drain_ignores_visible_ready_task_items(tmp_path: Path) -> None:
    task = Task(id="task-1", title="Task", summary="Task", description="Do it.")
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(root=tmp_path / "orch", entrypoint=tmp_path / "entrypoint"),
        task_manager=task_manager_with_tasks(task),
    )
    await orchestrator.enqueue_ready_tasks_once_for_test()
    orchestrator.request_drain_for_test()

    assert orchestrator.task_queue == (task.id,)
    assert await orchestrator.in_flight_work_settled_for_test() is True


async def test_budget_freeze_skips_creating_worktrees_for_ready_tasks(
    tmp_path: Path,
) -> None:
    # Finding 1 (Codex): a queued ready task must not produce a fresh worktree
    # after the budget breach, otherwise the ready-task queue keeps creating new
    # cost-incurring entries.
    task = Task(id="task-1", title="Task", summary="Task", description="Do it.")
    orchestrator = _StubUsageOrchestrator(
        AsyncOrchestratorConfig(
            root=tmp_path / "orch",
            entrypoint=tmp_path / "entrypoint",
            budget=AsyncOrchestratorBudgetConfig(max_cost=Decimal("10")),
        ),
        task_manager=task_manager_with_tasks(task),
    )
    orchestrator.force_budget_stop_for_test(_budget_stop_at("10"))

    result = await orchestrator.ensure_worktree_for_ready_task_id_for_test(task.id)

    assert result is None
    assert orchestrator.worktree_ids == ()


async def test_budget_freeze_skips_spawning_worker_agent(tmp_path: Path) -> None:
    # Finding 1 (Codex): a queued worker worktree (e.g. an existing PENDING
    # worktree at the moment of breach, or one rerouted by a validation failure
    # after the breach) must not start a new worker agent.
    task = Task(id="task-1", title="Task", summary="Task", description="Do it.")
    worktree = AsyncOrchestratorWorktree(
        worktree_id="worktree_000001",
        task_id=task.id,
        path=tmp_path / "worktree",
        head="abc123",
        state=WorktreeState.PENDING,
    )
    orchestrator = _StubUsageOrchestrator(
        AsyncOrchestratorConfig(
            root=tmp_path / "orch",
            entrypoint=tmp_path / "entrypoint",
            worker_agent_command=AsyncOrchestratorAgentCommandConfig(argv=("noop",)),
            budget=AsyncOrchestratorBudgetConfig(max_cost=Decimal("10")),
        ),
        task_manager=task_manager_with_tasks(task),
        worktrees=(worktree,),
    )
    orchestrator.force_budget_stop_for_test(_budget_stop_at("10"))

    # Simulate the spawner having claimed the queue item and dispatching to the
    # role-specific spawn helper. With the freeze in place the helper must not
    # create an asyncio.Task; the manual ``task_done()`` keeps the runtime queue
    # balanced with the upstream claim.
    orchestrator.runtime.worker_queue.enqueue(worktree.worktree_id)
    assert orchestrator.runtime.worker_queue.claim(worktree.worktree_id) is True

    orchestrator.spawn_worker_agent_task_for_test(worktree.worktree_id)

    assert orchestrator.runtime.worker_agent_tasks == {}


async def test_budget_freeze_skips_spawning_reviewer_agent(tmp_path: Path) -> None:
    # Finding 1 (Codex): same as the worker case, for reviewer agents.
    task = Task(id="task-1", title="Task", summary="Task", description="Do it.")
    worktree = AsyncOrchestratorWorktree(
        worktree_id="worktree_000001",
        task_id=task.id,
        path=tmp_path / "worktree",
        head="abc123",
        state=WorktreeState.REVIEW,
    )
    orchestrator = _StubUsageOrchestrator(
        AsyncOrchestratorConfig(
            root=tmp_path / "orch",
            entrypoint=tmp_path / "entrypoint",
            reviewer_agent_command=AsyncOrchestratorAgentCommandConfig(argv=("noop",)),
            budget=AsyncOrchestratorBudgetConfig(max_cost=Decimal("10")),
        ),
        task_manager=task_manager_with_tasks(task),
        worktrees=(worktree,),
    )
    orchestrator.force_budget_stop_for_test(_budget_stop_at("10"))

    orchestrator.runtime.review_queue.enqueue(worktree.worktree_id)
    assert orchestrator.runtime.review_queue.claim(worktree.worktree_id) is True

    orchestrator.spawn_reviewer_agent_task_for_test(worktree.worktree_id)

    assert orchestrator.runtime.reviewer_agent_tasks == {}


async def test_in_flight_work_settled_ignores_queued_agents_when_limit_is_zero(
    tmp_path: Path,
) -> None:
    task = Task(id="task-1", title="Task", summary="Task", description="Do it.")
    pending_worktree = AsyncOrchestratorWorktree(
        worktree_id="worktree_000001",
        task_id=task.id,
        path=tmp_path / "pending-worktree",
        head="abc123",
        state=WorktreeState.PENDING,
    )
    review_worktree = AsyncOrchestratorWorktree(
        worktree_id="worktree_000002",
        task_id=task.id,
        path=tmp_path / "review-worktree",
        head="def456",
        state=WorktreeState.REVIEW,
    )
    orchestrator = _StubUsageOrchestrator(
        AsyncOrchestratorConfig(
            root=tmp_path / "orch",
            entrypoint=tmp_path / "entrypoint",
            budget=AsyncOrchestratorBudgetConfig(max_cost=Decimal("10")),
        ),
        task_manager=task_manager_with_tasks(task),
        worktrees=(pending_worktree, review_worktree),
    )
    orchestrator.runtime.set_worker_agent_limit(0)
    orchestrator.runtime.set_reviewer_agent_limit(0)
    orchestrator.force_budget_stop_for_test(_budget_stop_at("10"))

    assert orchestrator.worker_queue == (pending_worktree.worktree_id,)
    assert orchestrator.review_queue == (review_worktree.worktree_id,)
    assert await orchestrator.in_flight_work_settled_for_test() is True


async def test_in_flight_work_settled_prunes_completed_agent_tasks(
    tmp_path: Path,
) -> None:
    # Finding 2 (Codex): while frozen on a budget breach the spawner no longer
    # runs its top-of-loop prune. A completed-but-undrained asyncio.Task in
    # ``worker_agent_tasks`` must not pin the settle check forever, or the run
    # never terminates.
    orchestrator = _budget_orchestrator(tmp_path, max_cost="10")

    async def _already_done() -> None:
        return None

    done_task: asyncio.Task[None] = asyncio.create_task(_already_done())
    await done_task
    assert done_task.done()
    orchestrator.runtime.worker_agent_tasks["worktree_done"] = done_task

    assert await orchestrator.in_flight_work_settled_for_test() is True
    # The stale entry was pruned by the settle check itself.
    assert orchestrator.runtime.worker_agent_tasks == {}


async def test_budget_stop_refresh_updates_accumulated_cost_pinned_breach(
    tmp_path: Path,
) -> None:
    # Finding 3 (Codex): once recorded, ``accumulated_cost`` refreshes to the
    # latest aggregate so the returned record matches "final cost" semantics,
    # while ``breach_accumulated_cost`` stays pinned to the first breach.
    orchestrator = _budget_orchestrator(tmp_path, max_cost="10")
    orchestrator.set_stub_cost(Cost(amount=Decimal("10"), currency="USD"))
    assert await orchestrator.record_budget_stop_if_exceeded_for_test() is True
    initial = orchestrator.budget_stop_for_test
    assert initial is not None
    assert initial.breach_accumulated_cost == "10"
    assert initial.accumulated_cost == "10"

    # An in-flight worker that was already paying when the ceiling was crossed
    # finishes and bumps the aggregate; the refresh path must pick that up.
    orchestrator.set_stub_cost(Cost(amount=Decimal("11.50"), currency="USD"))
    await orchestrator.refresh_budget_stop_accumulated_cost_for_test()

    refreshed = orchestrator.budget_stop_for_test
    assert refreshed is not None
    assert refreshed.breach_accumulated_cost == "10"
    assert refreshed.accumulated_cost == "11.50"
    assert refreshed.max_cost == "10"


async def test_budget_stop_refresh_fails_closed_on_post_breach_currency_mismatch(
    tmp_path: Path,
) -> None:
    # Finding 4 (Codex), refresh path: a post-breach session in another currency
    # must also fail closed rather than silently leaving ``accumulated_cost``
    # stale. The same strict aggregator the breach check uses gates the refresh.
    orchestrator = _budget_orchestrator(tmp_path, max_cost="10", currency="USD")
    orchestrator.set_stub_cost(Cost(amount=Decimal("10"), currency="USD"))
    assert await orchestrator.record_budget_stop_if_exceeded_for_test() is True

    orchestrator.set_stub_cost(Cost(amount=Decimal("12"), currency="EUR"))

    with pytest.raises(AsyncCostBudgetCurrencyMismatchError, match="EUR"):
        await orchestrator.refresh_budget_stop_accumulated_cost_for_test()


# ---------------------------------------------------------------------------
# Regression tests for a REVIEW-pin hang and an ExceptionGroup that hid the
# fail-closed currency error from the CLI.
# ---------------------------------------------------------------------------


async def test_in_flight_work_settled_under_freeze_does_not_pin_on_drained_review(
    tmp_path: Path,
) -> None:
    # Reproduces Codex's exact post-freeze state: a worktree is durably in
    # REVIEW but the gated reviewer spawner already popped and discarded the
    # ``review_queue`` item, so the queue is empty and no reviewer asyncio.Task
    # is running. The previous settle check pinned on durable REVIEW state and
    # never let the budget-stopped run terminate (resume re-enqueues it). This
    # test asserts the run can settle on its own.
    task = Task(id="task-1", title="Task", summary="Task", description="Do it.")
    review_worktree = AsyncOrchestratorWorktree(
        worktree_id="worktree_000001",
        task_id=task.id,
        path=tmp_path / "worktree",
        head="abc123",
        state=WorktreeState.REVIEW,
    )
    orchestrator = _StubUsageOrchestrator(
        AsyncOrchestratorConfig(
            root=tmp_path / "orch",
            entrypoint=tmp_path / "entrypoint",
            budget=AsyncOrchestratorBudgetConfig(max_cost=Decimal("10")),
        ),
        task_manager=task_manager_with_tasks(task),
        worktrees=(review_worktree,),
    )
    orchestrator.force_budget_stop_for_test(_budget_stop_at("10"))
    # The runtime auto-enqueued the REVIEW worktree on construction; simulate
    # the gated spawner having claimed (and so discarded) that queue item.
    assert orchestrator.runtime.review_queue.claim(review_worktree.worktree_id) is True
    assert orchestrator.runtime.review_queue.has_claimed_items is False
    assert orchestrator.runtime.reviewer_agent_tasks == {}

    assert await orchestrator.in_flight_work_settled_for_test() is True


async def test_in_flight_work_settled_still_pins_on_active_merge(
    tmp_path: Path,
) -> None:
    # The MERGE durable state must keep pinning: the merge processor is kept
    # alive under freeze and during ``_merge_worktree_id`` the queue item has
    # already been claimed (out of ``_queued``), so durable MERGE is the only
    # signal that a merge is still in progress.
    task = Task(id="task-1", title="Task", summary="Task", description="Do it.")
    merge_worktree = AsyncOrchestratorWorktree(
        worktree_id="worktree_000001",
        task_id=task.id,
        path=tmp_path / "worktree",
        head="abc123",
        state=WorktreeState.MERGE,
    )
    orchestrator = _StubUsageOrchestrator(
        AsyncOrchestratorConfig(
            root=tmp_path / "orch",
            entrypoint=tmp_path / "entrypoint",
            budget=AsyncOrchestratorBudgetConfig(max_cost=Decimal("10")),
        ),
        task_manager=task_manager_with_tasks(task),
        worktrees=(merge_worktree,),
    )
    orchestrator.force_budget_stop_for_test(_budget_stop_at("10"))
    # Simulate the merge processor having claimed the merge_queue item and being
    # mid-merge — the durable state is still MERGE.
    assert orchestrator.runtime.merge_queue.claim(merge_worktree.worktree_id) is True
    assert orchestrator.runtime.merge_queue.has_claimed_items is False

    assert await orchestrator.in_flight_work_settled_for_test() is False


async def test_run_unwraps_framework_error_from_task_group(tmp_path: Path) -> None:
    # The strict currency check raises ``AsyncCostBudgetCurrencyMismatchError``
    # from ``_enqueue_ready_tasks_forever`` inside the TaskGroup. Without
    # explicit unwrapping in ``run()``, the TaskGroup wraps it in an
    # ``ExceptionGroup`` and the CLI's ``except FrameworkError`` handler never
    # catches it. This test pins the bare-exception propagation so the CLI
    # maps the failure to ``INTERNAL_SOFTWARE`` (exit 70) as documented.
    task = Task(id="task-1", title="Task", summary="Task", description="Do it.")

    class _CurrencyMismatchOrchestrator(WorktreeTestingOrchestrator):
        async def _sync_task_manager_once(self) -> None:
            return None

        def _accumulated_cost_strict(
            self,
            worktrees: object,  # noqa: ARG002 — unused in stub
            currency: str,
        ) -> Cost | None:
            raise AsyncCostBudgetCurrencyMismatchError(
                f"stub session in EUR but budget configured in {currency}"
            )

    orchestrator = _CurrencyMismatchOrchestrator(
        AsyncOrchestratorConfig(
            root=tmp_path / "orch",
            entrypoint=tmp_path / "entrypoint",
            budget=AsyncOrchestratorBudgetConfig(max_cost=Decimal("1")),
        ),
        task_manager=task_manager_with_tasks(task),
    )

    with pytest.raises(AsyncCostBudgetCurrencyMismatchError, match="EUR"):
        await asyncio.wait_for(orchestrator.run(), timeout=5.0)


async def _wait_for_path(path: Path) -> None:
    for _ in range(50):
        if path.exists():
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"timed out waiting for path: {path}")


async def _wait_for_pid_to_stop(pid: int) -> None:
    for _ in range(50):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"timed out waiting for process to stop: {pid}")


async def test_transition_to_closed_removes_worktree_by_default(
    tmp_path: Path,
) -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable is not available")

    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    _initialize_git_repo(entrypoint)
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(root=root, entrypoint=entrypoint),
    )
    assert orchestrator.config.cleanup_closed_worktrees is True
    worktree = await orchestrator.create_fresh_worktree_for_test(name="entrypoint-copy")
    # Commit something inside the worktree and publish it to the target branch
    # so the cleanup safety checks know it carries no local-only committed work.
    (worktree.path / "work.txt").write_text("done\n", encoding="utf-8")
    _run_git(worktree.path, "add", "work.txt")
    _run_git(worktree.path, "commit", "-m", "worktree work")
    committed = _run_git(worktree.path, "rev-parse", "--verify", "HEAD")
    _run_git(entrypoint, "merge", "--ff-only", committed)
    assert worktree.path.is_dir()

    await orchestrator.transition_worktree_for_test(
        worktree.worktree_id, WorktreeState.CLOSED
    )

    # Working tree is gone, and the linked-worktree admin entry was pruned.
    assert not worktree.path.exists()
    listed = _run_git(entrypoint, "worktree", "list", "--porcelain")
    assert str(worktree.path) not in listed
    # The committed object remains reachable from the target branch — no work lost.
    subprocess.run(
        ["git", "cat-file", "-e", committed],
        cwd=entrypoint,
        check=True,
    )
    # State bookkeeping is unchanged: the worktree is still tracked as CLOSED.
    assert (
        orchestrator.worktrees_by_id[worktree.worktree_id].state
        is WorktreeState.CLOSED
    )


async def test_transition_to_closed_skips_cleanup_when_worktree_is_dirty(
    tmp_path: Path,
) -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable is not available")

    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    _initialize_git_repo(entrypoint)
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(root=root, entrypoint=entrypoint),
    )
    worktree = await orchestrator.create_fresh_worktree_for_test(name="dirty-closed")
    (worktree.path / "uncommitted.txt").write_text("preserve me\n", encoding="utf-8")

    await orchestrator.transition_worktree_for_test(
        worktree.worktree_id, WorktreeState.CLOSED
    )

    # Cleanup is default-on, but a dirty CLOSED worktree is preserved rather than
    # silently discarding local-only data. This keeps the low-level #92 merge
    # contract safe even if a test or future path bypasses the #105 dirty gate.
    assert (
        (worktree.path / "uncommitted.txt").read_text(encoding="utf-8")
        == "preserve me\n"
    )
    listed = _run_git(entrypoint, "worktree", "list", "--porcelain")
    assert str(worktree.path) in listed
    assert (
        orchestrator.worktrees_by_id[worktree.worktree_id].state
        is WorktreeState.CLOSED
    )


async def test_transition_to_closed_skips_cleanup_when_commits_are_unmerged(
    tmp_path: Path,
) -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable is not available")

    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    _initialize_git_repo(entrypoint)
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(root=root, entrypoint=entrypoint),
    )
    worktree = await orchestrator.create_fresh_worktree_for_test(name="unmerged-closed")
    (worktree.path / "committed.txt").write_text("preserve commit\n", encoding="utf-8")
    _commit_worktree(worktree.path, "unmerged work")

    await orchestrator.transition_worktree_for_test(
        worktree.worktree_id, WorktreeState.CLOSED
    )

    # A clean worktree can still contain committed local-only work if a caller
    # closes it without first publishing it. Preserve that too.
    assert (
        (worktree.path / "committed.txt").read_text(encoding="utf-8")
        == "preserve commit\n"
    )
    listed = _run_git(entrypoint, "worktree", "list", "--porcelain")
    assert str(worktree.path) in listed
    assert (
        orchestrator.worktrees_by_id[worktree.worktree_id].state
        is WorktreeState.CLOSED
    )


async def test_transition_to_closed_keeps_worktree_when_cleanup_disabled(
    tmp_path: Path,
) -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable is not available")

    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    _initialize_git_repo(entrypoint)
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(
            root=root,
            entrypoint=entrypoint,
            cleanup_closed_worktrees=False,
        ),
    )
    worktree = await orchestrator.create_fresh_worktree_for_test(name="entrypoint-copy")

    await orchestrator.transition_worktree_for_test(
        worktree.worktree_id, WorktreeState.CLOSED
    )

    # Opting out preserves the closed worktree for inspection.
    assert worktree.path.is_dir()


async def test_transition_to_closed_prunes_admin_entry_when_worktree_path_missing(
    tmp_path: Path,
) -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable is not available")

    entrypoint = tmp_path / "entrypoint"
    root = tmp_path / "orch"
    _initialize_git_repo(entrypoint)
    orchestrator = WorktreeTestingOrchestrator(
        AsyncOrchestratorConfig(root=root, entrypoint=entrypoint),
    )
    worktree = await orchestrator.create_fresh_worktree_for_test(name="entrypoint-copy")
    assert worktree.path.is_dir()

    shutil.rmtree(worktree.path)
    listed_before = _run_git(entrypoint, "worktree", "list", "--porcelain")
    assert str(worktree.path) in listed_before

    await orchestrator.transition_worktree_for_test(
        worktree.worktree_id, WorktreeState.CLOSED
    )

    listed_after = _run_git(entrypoint, "worktree", "list", "--porcelain")
    assert str(worktree.path) not in listed_after
    assert (
        orchestrator.worktrees_by_id[worktree.worktree_id].state
        is WorktreeState.CLOSED
    )


def _write_and_commit_tasks(repo: Path, *tasks: Task) -> None:
    for task in tasks:
        write_task(repo / "tasks" / f"{task.id}.yaml", task)
    _run_git(repo, "add", "tasks")
    _run_git(repo, "commit", "-m", "write test tasks")


def _initialize_git_repo(repo: Path) -> None:
    repo.mkdir()
    _run_git(repo, "init")
    _run_git(repo, "config", "user.email", "test@example.com")
    _run_git(repo, "config", "user.name", "Test User")
    _run_git(repo, "checkout", "-b", "main")
    (repo / "README.md").write_text("initial\n", encoding="utf-8")
    _run_git(repo, "add", "README.md")
    _run_git(repo, "commit", "-m", "initial")


def _run_git(cwd: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _commit_worktree(worktree_path: Path, message: str = "worker output") -> None:
    """Commit a worktree's changes the way a worker now must.

    The orchestrator no longer auto-commits a worker's dirty tree at merge time
    (workers own their commits), so tests that expect a contribution to merge
    must stage and commit it themselves. ``.tend/`` orchestrator metadata is
    already in the worktree's ``info/exclude``, so ``add -A`` will not stage it.
    """

    _run_git(worktree_path, "add", "-A")
    _run_git(worktree_path, "commit", "-m", message)
