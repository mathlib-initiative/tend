from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from tests.orchestrator.store_helpers import seed_store_state

from tend.agent.persistence.events import (
    ModelRequestStartedEvent,
    ModelRequestStartedPayload,
    ModelResponseCompletedEvent,
    ModelResponseCompletedPayload,
)
from tend.agent.session import Session
from tend.llm.usage import Cost, TokenUsage, Usage
from tend.orchestrator.config import AsyncOrchestratorConfig
from tend.orchestrator.control_store import SQLiteAsyncOrchestratorStore
from tend.orchestrator.orchestrator import AsyncOrchestrator
from tend.orchestrator.state import (
    AsyncOrchestratorAgentRole,
    AsyncOrchestratorWorktree,
    WorktreeState,
)
from tend.orchestrator.task_io import write_task
from tend.orchestrator.task_manager import TaskManager
from tend.orchestrator.tasks import Task, TaskStatus
from tend.orchestrator.usage import (
    agent_session_dir,
    aggregate_agent_session_usage,
    format_usage_summary,
)


async def test_run_result_aggregates_usage_from_managed_tend_agent_sessions(
    tmp_path: Path,
) -> None:
    root = tmp_path / "orch"
    entrypoint = tmp_path / "entrypoint"
    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()
    task = Task(
        id="task-1",
        title="Complete task", summary="Complete task",
        description="Already complete.",
        status=TaskStatus.COMPLETE,
    )
    write_task(entrypoint / "tasks" / "task-1.yaml", task)
    worker_usage = Usage(
        tokens=TokenUsage(input_tokens=100, output_tokens=25, cache_read_tokens=10),
        cost=Cost(amount=Decimal("0.0123"), currency="USD", pricing_source="test"),
    )
    reviewer_usage = Usage(
        tokens=TokenUsage(input_tokens=40, output_tokens=5, reasoning_tokens=3),
        cost=Cost(amount=Decimal("0.004"), currency="USD", pricing_source="test"),
    )
    _write_session_usage(
        root,
        worktree_id="worktree_000001",
        role=AsyncOrchestratorAgentRole.WORKER,
        usage=worker_usage,
    )
    _write_session_usage(
        root,
        worktree_id="worktree_000001",
        role=AsyncOrchestratorAgentRole.REVIEWER,
        usage=reviewer_usage,
    )
    store = seed_store_state(
        SQLiteAsyncOrchestratorStore(root),
        task_manager=TaskManager(tasks=[task]),
        worktrees=(
            AsyncOrchestratorWorktree(
                worktree_id="worktree_000001",
                task_id=task.id,
                path=worktree_path,
                head="abc123",
                state=WorktreeState.CLOSED,
                worker_session_started=True,
                reviewer_session_started=True,
            ),
        ),
    )
    orchestrator = AsyncOrchestrator(
        AsyncOrchestratorConfig(root=root, entrypoint=entrypoint),
        store=store,
    )

    result = await orchestrator.run()

    assert result.usage == Usage(
        tokens=TokenUsage(
            input_tokens=140,
            output_tokens=30,
            reasoning_tokens=3,
            cache_read_tokens=10,
        ),
        cost=Cost(amount=Decimal("0.0163"), currency="USD", pricing_source="test"),
        model_requests=2,
    )
    assert orchestrator.usage == result.usage
    assert store.aggregate_usage(root) == result.usage
    assert "input=140" in format_usage_summary(result.usage)
    assert "cost=0.0163 USD" in format_usage_summary(result.usage)


def test_aggregate_agent_session_usage_ignores_empty_or_foreign_sessions(
    tmp_path: Path,
) -> None:
    root = tmp_path / "orch"
    worktree = AsyncOrchestratorWorktree(
        worktree_id="worktree_000001",
        path=tmp_path / "worktree",
        head="abc123",
    )
    agent_session_dir(
        root,
        worktree.worktree_id,
        AsyncOrchestratorAgentRole.WORKER,
    ).mkdir(parents=True)

    assert aggregate_agent_session_usage(root, [worktree]) == Usage()


def test_aggregate_agent_session_usage_omits_mixed_currency_costs(tmp_path: Path) -> None:
    root = tmp_path / "orch"
    worktree = AsyncOrchestratorWorktree(
        worktree_id="worktree_000001",
        path=tmp_path / "worktree",
        head="abc123",
        worker_session_started=True,
        reviewer_session_started=True,
    )
    _write_session_usage(
        root,
        worktree_id=worktree.worktree_id,
        role=AsyncOrchestratorAgentRole.WORKER,
        usage=Usage(
            tokens=TokenUsage(input_tokens=1),
            cost=Cost(amount=Decimal("1.00"), currency="USD"),
        ),
    )
    _write_session_usage(
        root,
        worktree_id=worktree.worktree_id,
        role=AsyncOrchestratorAgentRole.REVIEWER,
        usage=Usage(
            tokens=TokenUsage(output_tokens=2),
            cost=Cost(amount=Decimal("1.00"), currency="EUR"),
        ),
    )

    usage = aggregate_agent_session_usage(root, [worktree])

    assert usage.tokens.input_tokens == 1
    assert usage.tokens.output_tokens == 2
    assert usage.model_requests == 2
    assert usage.cost is None


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
