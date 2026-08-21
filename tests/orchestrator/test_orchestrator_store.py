from __future__ import annotations

import asyncio
from decimal import Decimal
from pathlib import Path

import pytest

from tend._common.agent_outputs import ReviewVerdictOutput
from tend.agent.persistence.events import (
    ModelRequestStartedEvent,
    ModelRequestStartedPayload,
    ModelResponseCompletedEvent,
    ModelResponseCompletedPayload,
)
from tend.agent.session import Session
from tend.llm.usage import Cost, TokenUsage, Usage
from tend.orchestrator.control_store import SQLiteAsyncOrchestratorStore
from tend.orchestrator.state import (
    AsyncOrchestratorAgentRole,
    AsyncOrchestratorDiscussionMessage,
    AsyncOrchestratorWorktree,
    WorktreeState,
)
from tend.orchestrator.task_manager import TaskManager
from tend.orchestrator.tasks import Task
from tend.orchestrator.usage import agent_session_dir, aggregate_agent_session_usage


def test_orchestrator_store_allocates_worktrees_and_sequences(tmp_path: Path) -> None:
    store = SQLiteAsyncOrchestratorStore(tmp_path)

    assert store.state_exists() is False
    store.initialize_state()
    assert store.state_exists() is True
    assert store.next_worktree_sequence() == 1

    first_id = store.allocate_worktree(
        task_id="task-1",
        path=tmp_path / "worktree-1",
        head="abc123",
    )
    second_id = store.allocate_worktree(
        task_id="task-1",
        path=tmp_path / "worktree-2",
        head="def456",
    )

    assert first_id == "worktree_000001"
    assert second_id == "worktree_000002"
    assert store.next_worktree_sequence() == 3
    assert [worktree.worktree_id for worktree in store.list_worktrees()] == [
        first_id,
        second_id,
    ]
    first = store.get_worktree(first_id)
    assert first is not None
    assert first.state is WorktreeState.PENDING
    assert first.task_id == "task-1"
    assert first.path == tmp_path / "worktree-1"
    assert first.head == "abc123"


def test_orchestrator_store_allocates_exact_worktree_id_and_advances_sequence(
    tmp_path: Path,
) -> None:
    store = SQLiteAsyncOrchestratorStore(tmp_path)

    allocated = store.allocate_worktree(
        task_id="task-1",
        path=tmp_path / "worktree-3",
        head="abc123",
        worktree_id="worktree_000003",
    )

    assert allocated == "worktree_000003"
    assert store.next_worktree_sequence() == 4
    next_allocated = store.allocate_worktree(
        task_id="task-1",
        path=tmp_path / "worktree-4",
        head="def456",
    )
    assert next_allocated == "worktree_000004"


def test_orchestrator_store_cas_transitions_worktree_state(tmp_path: Path) -> None:
    store = SQLiteAsyncOrchestratorStore(tmp_path)
    worktree_id = store.allocate_worktree(
        task_id="task-1",
        path=tmp_path / "worktree",
        head="abc123",
    )

    assert store.set_worktree_state(
        worktree_id,
        expected=WorktreeState.PENDING,
        new=WorktreeState.WORKER_RUNNING,
    )
    assert not store.set_worktree_state(
        worktree_id,
        expected=WorktreeState.PENDING,
        new=WorktreeState.REVIEW,
    )
    worktree = store.get_worktree(worktree_id)
    assert worktree is not None
    assert worktree.state is WorktreeState.WORKER_RUNNING


async def test_orchestrator_store_cas_allows_one_concurrent_winner(
    tmp_path: Path,
) -> None:
    store = SQLiteAsyncOrchestratorStore(tmp_path)
    worktree_id = store.allocate_worktree(
        task_id="task-1",
        path=tmp_path / "worktree",
        head="abc123",
    )

    first, second = await asyncio.gather(
        asyncio.to_thread(
            store.set_worktree_state,
            worktree_id,
            expected=WorktreeState.PENDING,
            new=WorktreeState.WORKER_RUNNING,
        ),
        asyncio.to_thread(
            store.set_worktree_state,
            worktree_id,
            expected=WorktreeState.PENDING,
            new=WorktreeState.REVIEW,
        ),
    )

    assert {first, second} == {False, True}
    worktree = store.get_worktree(worktree_id)
    assert worktree is not None
    assert worktree.state in {WorktreeState.WORKER_RUNNING, WorktreeState.REVIEW}


def test_orchestrator_store_resets_running_worktrees(tmp_path: Path) -> None:
    store = SQLiteAsyncOrchestratorStore(tmp_path)
    first_id = store.allocate_worktree(
        task_id="task-1",
        path=tmp_path / "worktree-1",
        head="abc123",
    )
    second_id = store.allocate_worktree(
        task_id="task-2",
        path=tmp_path / "worktree-2",
        head="def456",
    )
    review_id = store.allocate_worktree(
        task_id="task-3",
        path=tmp_path / "worktree-3",
        head="789abc",
    )
    for worktree_id in (first_id, second_id):
        assert store.set_worktree_state(
            worktree_id,
            expected=WorktreeState.PENDING,
            new=WorktreeState.WORKER_RUNNING,
        )
    assert store.set_worktree_state(
        review_id,
        expected=WorktreeState.PENDING,
        new=WorktreeState.REVIEW,
    )

    assert store.reset_running_worktrees() == 2

    states = {worktree.worktree_id: worktree.state for worktree in store.list_worktrees()}
    assert states == {
        first_id: WorktreeState.PENDING,
        second_id: WorktreeState.PENDING,
        review_id: WorktreeState.REVIEW,
    }


def test_orchestrator_store_rolls_back_failed_transition_without_torn_write(
    tmp_path: Path,
) -> None:
    store = SQLiteAsyncOrchestratorStore(tmp_path)
    worktree_id = store.allocate_worktree(
        task_id="task-1",
        path=tmp_path / "worktree",
        head="abc123",
    )

    with pytest.raises(ValueError, match="discussion messages must not be blank"):
        store.record_worktree_transition(
            worktree_id,
            expected=WorktreeState.PENDING,
            new=WorktreeState.REVIEW,
            discussion_messages=(
                (AsyncOrchestratorAgentRole.WORKER, "   "),
            ),
        )

    worktree = store.get_worktree(worktree_id)
    assert worktree is not None
    assert worktree.state is WorktreeState.PENDING
    assert worktree.discussion == ()


def test_orchestrator_store_records_session_started_and_usage(
    tmp_path: Path,
) -> None:
    store = SQLiteAsyncOrchestratorStore(tmp_path)
    worktree_id = store.allocate_worktree(
        task_id="task-1",
        path=tmp_path / "worktree",
        head="abc123",
    )
    worker_usage = Usage(
        tokens=TokenUsage(input_tokens=10, output_tokens=2),
        cost=Cost(amount=Decimal("0.0100"), currency="USD", pricing_source="test"),
        model_requests=1,
    )
    reviewer_usage = Usage(
        tokens=TokenUsage(input_tokens=4, reasoning_tokens=1),
        model_requests=1,
    )

    store.mark_agent_session_started(worktree_id, AsyncOrchestratorAgentRole.WORKER)
    store.mark_agent_session_started(worktree_id, AsyncOrchestratorAgentRole.REVIEWER)
    store.set_agent_session_usage(
        worktree_id,
        AsyncOrchestratorAgentRole.WORKER,
        worker_usage,
    )
    store.set_agent_session_usage(
        worktree_id,
        AsyncOrchestratorAgentRole.REVIEWER,
        reviewer_usage,
    )

    worktree = store.get_worktree(worktree_id)
    assert worktree is not None
    assert worktree.worker_session_started is True
    assert worktree.reviewer_session_started is True
    assert worktree.worker_session_usage == worker_usage
    assert worktree.reviewer_session_usage == reviewer_usage


def test_orchestrator_store_guarded_usage_write_preserves_fresher_snapshot(
    tmp_path: Path,
) -> None:
    store = SQLiteAsyncOrchestratorStore(tmp_path)
    worktree_id = store.allocate_worktree(
        task_id="task-1",
        path=tmp_path / "worktree",
        head="abc123",
    )
    stale_usage = Usage(
        tokens=TokenUsage(input_tokens=1),
        cost=Cost(amount=Decimal("1.00"), currency="USD", pricing_source="stale"),
    )
    fresh_usage = Usage(
        tokens=TokenUsage(input_tokens=2),
        cost=Cost(amount=Decimal("2.00"), currency="USD", pricing_source="fresh"),
    )

    store.mark_agent_session_started(worktree_id, AsyncOrchestratorAgentRole.WORKER)
    store.set_agent_session_usage(worktree_id, AsyncOrchestratorAgentRole.WORKER, fresh_usage)

    assert not store.set_agent_session_usage_if_missing_and_inactive(
        worktree_id,
        AsyncOrchestratorAgentRole.WORKER,
        stale_usage,
        expected_state=WorktreeState.PENDING,
    )
    worktree = store.get_worktree(worktree_id)
    assert worktree is not None
    assert worktree.worker_session_usage == fresh_usage

    empty_id = store.allocate_worktree(
        task_id="task-1",
        path=tmp_path / "empty-worktree",
        head="def456",
    )
    assert store.set_worktree_state(
        empty_id,
        expected=WorktreeState.PENDING,
        new=WorktreeState.WORKER_RUNNING,
    )
    assert not store.set_agent_session_usage_if_missing_and_inactive(
        empty_id,
        AsyncOrchestratorAgentRole.WORKER,
        stale_usage,
        expected_state=WorktreeState.WORKER_RUNNING,
    )


def test_orchestrator_store_appends_discussion_and_verdicts_in_order(
    tmp_path: Path,
) -> None:
    store = SQLiteAsyncOrchestratorStore(tmp_path)
    worktree_id = store.allocate_worktree(
        task_id="task-1",
        path=tmp_path / "worktree",
        head="abc123",
    )
    first_verdict = _review_verdict("request_changes", notes="needs a fix")
    second_verdict = _review_verdict("approve", notes="looks good")

    store.append_discussion(
        worktree_id,
        role=AsyncOrchestratorAgentRole.WORKER,
        message="Implemented the feature.",
    )
    store.append_discussion(
        worktree_id,
        role=AsyncOrchestratorAgentRole.ORCHESTRATOR,
        message="Queued review.",
    )
    store.append_review_verdict(worktree_id, first_verdict)
    store.append_review_verdict(worktree_id, second_verdict)

    worktree = store.get_worktree(worktree_id)
    assert worktree is not None
    assert worktree.discussion == (
        AsyncOrchestratorDiscussionMessage(
            role=AsyncOrchestratorAgentRole.WORKER,
            message="Implemented the feature.",
        ),
        AsyncOrchestratorDiscussionMessage(
            role=AsyncOrchestratorAgentRole.ORCHESTRATOR,
            message="Queued review.",
        ),
    )
    assert worktree.review_verdicts == (first_verdict, second_verdict)


def test_orchestrator_store_replaces_task_snapshot_and_detaches_orphans(
    tmp_path: Path,
) -> None:
    store = SQLiteAsyncOrchestratorStore(tmp_path)
    kept_task = _task("task-1")
    orphaned_task = _task("task-2")
    store.replace_task_snapshot(TaskManager(tasks=[kept_task, orphaned_task]))
    kept_id = store.allocate_worktree(
        task_id=kept_task.id,
        path=tmp_path / "kept",
        head="abc123",
    )
    orphaned_id = store.allocate_worktree(
        task_id=orphaned_task.id,
        path=tmp_path / "orphaned",
        head="def456",
    )

    store.replace_task_snapshot(TaskManager(tasks=[kept_task]))

    assert store.load_task_snapshot() == TaskManager(tasks=[kept_task])
    kept = store.get_worktree(kept_id)
    orphaned = store.get_worktree(orphaned_id)
    assert kept is not None
    assert orphaned is not None
    assert kept.task_id == kept_task.id
    assert orphaned.task_id is None
    assert [worktree.worktree_id for worktree in store.worktrees_for_task(kept_task.id)] == [
        kept_id
    ]
    assert store.worktrees_for_task(orphaned_task.id) == ()


def test_orchestrator_store_reconstructs_async_worktree_round_trip(
    tmp_path: Path,
) -> None:
    store = SQLiteAsyncOrchestratorStore(tmp_path)
    task = _task("task-1")
    store.replace_task_snapshot(TaskManager(tasks=[task]))
    worktree_id = store.allocate_worktree(
        task_id=task.id,
        path=tmp_path / "worktree",
        head="abc123",
    )
    usage = Usage(tokens=TokenUsage(input_tokens=5), model_requests=1)
    verdict = _review_verdict("approve", notes="ready")
    assert store.set_worktree_state(
        worktree_id,
        expected=WorktreeState.PENDING,
        new=WorktreeState.REVIEW,
    )
    store.mark_agent_session_started(worktree_id, AsyncOrchestratorAgentRole.WORKER)
    store.set_agent_session_usage(worktree_id, AsyncOrchestratorAgentRole.WORKER, usage)
    store.append_discussion(
        worktree_id,
        role=AsyncOrchestratorAgentRole.WORKER,
        message="Ready for review.",
    )
    store.append_review_verdict(worktree_id, verdict)
    expected = AsyncOrchestratorWorktree(
        worktree_id=worktree_id,
        task_id=task.id,
        path=tmp_path / "worktree",
        head="abc123",
        state=WorktreeState.REVIEW,
        discussion=(
            AsyncOrchestratorDiscussionMessage(
                role=AsyncOrchestratorAgentRole.WORKER,
                message="Ready for review.",
            ),
        ),
        review_verdicts=(verdict,),
        worker_session_started=True,
        worker_session_usage=usage,
    )

    assert store.get_worktree(worktree_id) == expected
    assert store.list_worktrees() == (expected,)
    assert store.worktrees_for_task(task.id) == (expected,)
    assert store.non_closed_worktrees_for_task(task.id) == (expected,)

    assert store.set_worktree_state(
        worktree_id,
        expected=WorktreeState.REVIEW,
        new=WorktreeState.CLOSED,
    )
    assert store.non_closed_worktrees_for_task(task.id) == ()


def test_orchestrator_store_aggregate_usage_matches_session_aggregation_and_activity(
    tmp_path: Path,
) -> None:
    root = tmp_path / "orch"
    store = SQLiteAsyncOrchestratorStore(root)
    terminal_id = store.allocate_worktree(
        task_id="task-1",
        path=tmp_path / "terminal-worktree",
        head="abc123",
    )
    active_id = store.allocate_worktree(
        task_id="task-2",
        path=tmp_path / "active-worktree",
        head="def456",
    )
    terminal_snapshot = Usage(
        tokens=TokenUsage(input_tokens=7),
        model_requests=1,
    )
    active_stale_snapshot = Usage(
        tokens=TokenUsage(input_tokens=3),
        model_requests=1,
    )
    store.mark_agent_session_started(terminal_id, AsyncOrchestratorAgentRole.WORKER)
    store.set_agent_session_usage(
        terminal_id,
        AsyncOrchestratorAgentRole.WORKER,
        terminal_snapshot,
    )
    assert store.set_worktree_state(
        terminal_id,
        expected=WorktreeState.PENDING,
        new=WorktreeState.CLOSED,
    )
    store.mark_agent_session_started(active_id, AsyncOrchestratorAgentRole.WORKER)
    store.set_agent_session_usage(
        active_id,
        AsyncOrchestratorAgentRole.WORKER,
        active_stale_snapshot,
    )
    assert store.set_worktree_state(
        active_id,
        expected=WorktreeState.PENDING,
        new=WorktreeState.WORKER_RUNNING,
    )
    _write_session_usage(
        root,
        worktree_id=terminal_id,
        role=AsyncOrchestratorAgentRole.WORKER,
        usage=Usage(tokens=TokenUsage(input_tokens=100)),
    )
    _write_session_usage(
        root,
        worktree_id=active_id,
        role=AsyncOrchestratorAgentRole.WORKER,
        usage=Usage(tokens=TokenUsage(input_tokens=11, output_tokens=2)),
    )

    worktrees = store.list_worktrees()
    usage = store.aggregate_usage(root)

    assert usage == aggregate_agent_session_usage(root, worktrees)
    assert usage == Usage(
        tokens=TokenUsage(input_tokens=18, output_tokens=2),
        model_requests=2,
    )


def test_orchestrator_store_aggregate_usage_omits_mixed_currency_costs(
    tmp_path: Path,
) -> None:
    root = tmp_path / "orch"
    store = SQLiteAsyncOrchestratorStore(root)
    worktree_id = store.allocate_worktree(
        task_id="task-1",
        path=tmp_path / "worktree",
        head="abc123",
    )
    store.mark_agent_session_started(worktree_id, AsyncOrchestratorAgentRole.WORKER)
    store.mark_agent_session_started(worktree_id, AsyncOrchestratorAgentRole.REVIEWER)
    _write_session_usage(
        root,
        worktree_id=worktree_id,
        role=AsyncOrchestratorAgentRole.WORKER,
        usage=Usage(
            tokens=TokenUsage(input_tokens=1),
            cost=Cost(amount=Decimal("1.00"), currency="USD"),
        ),
    )
    _write_session_usage(
        root,
        worktree_id=worktree_id,
        role=AsyncOrchestratorAgentRole.REVIEWER,
        usage=Usage(
            tokens=TokenUsage(output_tokens=2),
            cost=Cost(amount=Decimal("1.00"), currency="EUR"),
        ),
    )

    worktrees = store.list_worktrees()
    usage = store.aggregate_usage(root)

    assert usage == aggregate_agent_session_usage(root, worktrees)
    assert usage.tokens.input_tokens == 1
    assert usage.tokens.output_tokens == 2
    assert usage.model_requests == 2
    assert usage.cost is None


def _task(task_id: str) -> Task:
    return Task(
        id=task_id,
        title=f"Task {task_id}",
        summary=f"Task {task_id}",
        description=f"Complete {task_id}.",
    )


def _review_verdict(
    verdict: str,
    *,
    notes: str,
) -> ReviewVerdictOutput:
    if verdict == "approve":
        return ReviewVerdictOutput(
            schema_version=1,
            verdict="approve",
            notes=notes,
        )
    return ReviewVerdictOutput(
        schema_version=1,
        verdict="request_changes",
        notes=notes,
        feedback_text="Please fix the failing case.",
    )


def _write_session_usage(
    root: Path,
    *,
    worktree_id: str,
    role: AsyncOrchestratorAgentRole,
    usage: Usage,
) -> None:
    with Session.create(
        agent_session_dir(root, worktree_id, role),
        session_id=f"sess_{worktree_id}_{role.value}",
        sync_writes=False,
    ) as session:
        turn_id = f"turn_{role.value}"
        request_id = f"model_req_{role.value}"
        session.append_event(
            ModelRequestStartedEvent(
                session_id=session.session_id,
                turn_id=turn_id,
                parent_event_id=session.last_event_id,
                sequence=session.next_sequence,
                payload=ModelRequestStartedPayload(request_id=request_id),
            )
        )
        session.append_event(
            ModelResponseCompletedEvent(
                session_id=session.session_id,
                turn_id=turn_id,
                parent_event_id=session.last_event_id,
                sequence=session.next_sequence,
                payload=ModelResponseCompletedPayload(
                    request_id=request_id,
                    response_id=f"model_resp_{role.value}",
                    usage=usage,
                ),
            )
        )
