from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from tend.orchestrator.control_store import SQLiteAsyncOrchestratorStore
from tend.orchestrator.state import (
    AsyncOrchestratorAgentRole,
    AsyncOrchestratorWorktree,
    WorktreeState,
)
from tend.orchestrator.task_manager import TaskManager
from tend.orchestrator.tasks import Task


def seed_store_state(
    root_or_store: Path | SQLiteAsyncOrchestratorStore,
    *,
    task_manager: TaskManager | None = None,
    worktrees: Iterable[AsyncOrchestratorWorktree] = (),
) -> SQLiteAsyncOrchestratorStore:
    """Replace test durable state with the supplied task/worktree snapshot."""

    store = (
        root_or_store
        if isinstance(root_or_store, SQLiteAsyncOrchestratorStore)
        else SQLiteAsyncOrchestratorStore(root_or_store)
    )
    store.clear_state()
    store.replace_task_snapshot(TaskManager() if task_manager is None else task_manager)
    for worktree in worktrees:
        store.allocate_worktree(
            task_id=worktree.task_id,
            path=worktree.path,
            head=worktree.head,
            worktree_id=worktree.worktree_id,
        )
        if worktree.state is not WorktreeState.PENDING:
            changed = store.set_worktree_state(
                worktree.worktree_id,
                expected=WorktreeState.PENDING,
                new=worktree.state,
            )
            if not changed:
                raise AssertionError(f"failed to seed worktree state: {worktree.worktree_id}")
        for discussion in worktree.discussion:
            store.append_discussion(
                worktree.worktree_id,
                role=discussion.role,
                message=discussion.message,
            )
        for verdict in worktree.review_verdicts:
            store.append_review_verdict(worktree.worktree_id, verdict)
        if worktree.worker_session_started:
            store.mark_agent_session_started(
                worktree.worktree_id,
                AsyncOrchestratorAgentRole.WORKER,
            )
        if worktree.reviewer_session_started:
            store.mark_agent_session_started(
                worktree.worktree_id,
                AsyncOrchestratorAgentRole.REVIEWER,
            )
        if worktree.worker_session_usage is not None:
            store.set_agent_session_usage(
                worktree.worktree_id,
                AsyncOrchestratorAgentRole.WORKER,
                worktree.worker_session_usage,
            )
        if worktree.reviewer_session_usage is not None:
            store.set_agent_session_usage(
                worktree.worktree_id,
                AsyncOrchestratorAgentRole.REVIEWER,
                worktree.reviewer_session_usage,
            )
    return store


def worktree_ids_for_task(
    store: SQLiteAsyncOrchestratorStore,
    task: Task | str,
) -> tuple[str, ...]:
    task_id = task.id if isinstance(task, Task) else task
    return tuple(worktree.worktree_id for worktree in store.worktrees_for_task(task_id))
