from __future__ import annotations

import os
import sqlite3
from collections.abc import Callable
from pathlib import Path

import pytest

from tend._common.agent_outputs import ReviewVerdictOutput
from tend.llm.usage import Usage
from tend.orchestrator.control_store import (
    ASYNC_ORCHESTRATOR_DB_FILENAME,
    ASYNC_ORCHESTRATOR_SCHEMA_VERSION,
    AsyncOrchestratorControlCommandError,
    AsyncOrchestratorControlSchemaError,
    ControlActiveAgentSnapshot,
    SQLiteAsyncOrchestratorStore,
)
from tend.orchestrator.state import AsyncOrchestratorAgentRole, WorktreeState
from tend.orchestrator.task_manager import TaskManager


def test_control_store_initializes_schema(tmp_path: Path) -> None:
    store = SQLiteAsyncOrchestratorStore(tmp_path)

    store.initialize()

    assert store.path == tmp_path / ASYNC_ORCHESTRATOR_DB_FILENAME
    assert store.path.exists()
    with sqlite3.connect(store.path) as conn:
        row = conn.execute("SELECT version FROM schema WHERE id = 1").fetchone()
        tables = {
            str(table_row[0])
            for table_row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
    assert row == (ASYNC_ORCHESTRATOR_SCHEMA_VERSION,)
    assert {
        "schema",
        "runs",
        "control_commands",
        "active_agents",
        "async_meta",
        "worktrees",
        "worktree_discussion",
        "worktree_review_verdicts",
    } <= tables
    assert store.latest_run() is None
    assert store.list_commands() == ()


def test_control_store_registers_and_updates_run(tmp_path: Path) -> None:
    store = SQLiteAsyncOrchestratorStore(tmp_path)

    registered = store.register_run(
        run_id="run_test",
        pid=os.getpid(),
        worker_limit=3,
        reviewer_limit=2,
        paused=False,
        drain_requested=False,
    )

    assert registered.run_id == "run_test"
    assert registered.pid == os.getpid()
    assert registered.status == "running"
    assert registered.worker_limit == 3
    assert registered.reviewer_limit == 2
    assert registered.paused is False
    assert registered.drain_requested is False
    assert registered.started_at == registered.heartbeat_at

    heartbeat = store.record_run_heartbeat(
        run_id="run_test",
        status="draining",
        status_reason="operator_drain",
        worker_limit=1,
        reviewer_limit=0,
        paused=True,
        drain_requested=True,
    )

    assert heartbeat.status == "draining"
    assert heartbeat.status_reason == "operator_drain"
    assert heartbeat.worker_limit == 1
    assert heartbeat.reviewer_limit == 0
    assert heartbeat.paused is True
    assert heartbeat.drain_requested is True
    assert heartbeat.heartbeat_at >= registered.heartbeat_at

    finished = store.record_run_finished(
        run_id="run_test",
        status="stopped",
        status_reason="operator_drain",
        worker_limit=1,
        reviewer_limit=0,
        paused=True,
        drain_requested=True,
    )

    assert finished.status == "stopped"
    assert finished.status_reason == "operator_drain"
    assert store.latest_run() == finished

    late_heartbeat = store.record_run_heartbeat(
        run_id="run_test",
        status="running",
        worker_limit=3,
        reviewer_limit=2,
        paused=False,
        drain_requested=False,
    )

    assert late_heartbeat.status == "stopped"
    assert late_heartbeat.status_reason == "operator_drain"
    assert store.latest_run() == finished


def test_control_store_records_and_clears_active_agent_snapshots(
    tmp_path: Path,
) -> None:
    store = SQLiteAsyncOrchestratorStore(tmp_path)
    store.register_run(run_id="run_test", pid=os.getpid())

    store.record_run_heartbeat(
        run_id="run_test",
        status="running",
        active_agents=(
            ControlActiveAgentSnapshot(
                role="worker",
                worktree_id="worktree_000001",
                task_id="task-a",
                worktree_state="worker_running",
            ),
            ControlActiveAgentSnapshot(
                role="reviewer",
                worktree_id="worktree_000002",
                task_id="task-b",
                worktree_state="review",
            ),
        ),
    )

    agents = store.list_active_agents(run_id="run_test")
    assert [(agent.role, agent.worktree_id) for agent in agents] == [
        ("worker", "worktree_000001"),
        ("reviewer", "worktree_000002"),
    ]
    assert agents[0].task_id == "task-a"
    assert agents[0].worktree_state == "worker_running"

    store.record_run_heartbeat(run_id="run_test", status="running")

    assert store.list_active_agents(run_id="run_test") == ()

    store.record_run_heartbeat(
        run_id="run_test",
        status="running",
        active_agents=(
            ControlActiveAgentSnapshot(
                role="reviewer",
                worktree_id="worktree_000002",
                task_id="task-b",
                worktree_state="review",
            ),
        ),
    )

    assert [agent.role for agent in store.list_active_agents(run_id="run_test")] == [
        "reviewer"
    ]

    store.record_run_finished(run_id="run_test", status="completed")

    assert store.list_active_agents(run_id="run_test") == ()

    store.record_run_heartbeat(
        run_id="run_test",
        status="running",
        active_agents=(
            ControlActiveAgentSnapshot(role="worker", worktree_id="worktree_000003"),
        ),
    )

    assert store.list_active_agents(run_id="run_test") == ()


def test_control_store_rejects_future_schema_without_current_ddl(
    tmp_path: Path,
) -> None:
    store = SQLiteAsyncOrchestratorStore(tmp_path)
    store.path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(store.path) as conn:
        conn.execute(
            "CREATE TABLE schema (id INTEGER PRIMARY KEY, version INTEGER NOT NULL)"
        )
        conn.execute(
            "INSERT INTO schema (id, version) VALUES (1, ?)",
            (ASYNC_ORCHESTRATOR_SCHEMA_VERSION + 1,),
        )
        conn.execute("CREATE TABLE future_only (id INTEGER PRIMARY KEY)")
        initial_schema = conn.execute(
            """
            SELECT type, name, sql FROM sqlite_master
            WHERE name NOT LIKE 'sqlite_%'
            ORDER BY type, name
            """
        ).fetchall()

    with pytest.raises(AsyncOrchestratorControlSchemaError, match="unsupported"):
        store.initialize()

    with sqlite3.connect(store.path) as conn:
        schema = conn.execute(
            """
            SELECT type, name, sql FROM sqlite_master
            WHERE name NOT LIKE 'sqlite_%'
            ORDER BY type, name
            """
        ).fetchall()
    assert schema == initial_schema


def test_control_store_public_operations_reject_unsupported_schema(
    tmp_path: Path,
) -> None:
    store = SQLiteAsyncOrchestratorStore(tmp_path)
    store.register_run(run_id="run_test", pid=os.getpid())
    pending = store.enqueue_command("noop", run_id="run_test")
    claimed = store.claim_pending_command(run_id="run_test")
    assert claimed is not None
    worktree_id = store.allocate_worktree(
        task_id="task-1",
        path=tmp_path / "worktree",
        head="abc123",
    )
    verdict = ReviewVerdictOutput(
        schema_version=1,
        verdict="approve",
        notes="looks good",
    )
    with sqlite3.connect(store.path) as conn:
        conn.execute(
            "UPDATE schema SET version = ? WHERE id = 1",
            (ASYNC_ORCHESTRATOR_SCHEMA_VERSION + 1,),
        )

    operations: tuple[tuple[str, Callable[[], object]], ...] = (
        (
            "register_run",
            lambda: store.register_run(run_id="run_late", pid=os.getpid()),
        ),
        (
            "record_run_heartbeat",
            lambda: store.record_run_heartbeat(run_id="run_test", status="running"),
        ),
        (
            "record_run_finished",
            lambda: store.record_run_finished(run_id="run_test", status="completed"),
        ),
        ("get_run", lambda: store.get_run("run_test")),
        ("latest_run", store.latest_run),
        ("enqueue_command", lambda: store.enqueue_command("noop")),
        (
            "enqueue_command_for_latest_active_run",
            lambda: store.enqueue_command_for_latest_active_run("pause"),
        ),
        ("claim_pending_command", lambda: store.claim_pending_command(run_id="run_test")),
        (
            "record_command_succeeded",
            lambda: store.record_command_succeeded(claimed.id),
        ),
        (
            "record_command_failed",
            lambda: store.record_command_failed(claimed.id, error="boom"),
        ),
        ("get_command", lambda: store.get_command(pending.id)),
        (
            "cancel_incomplete_commands_for_run",
            lambda: store.cancel_incomplete_commands_for_run(
                run_id="run_test",
                error="cancelled",
            ),
        ),
        ("list_commands", store.list_commands),
        ("list_active_agents", store.list_active_agents),
        ("state_exists", store.state_exists),
        (
            "allocate_worktree",
            lambda: store.allocate_worktree(
                task_id="task-1",
                path=tmp_path / "late-worktree",
                head="def456",
            ),
        ),
        (
            "set_worktree_state",
            lambda: store.set_worktree_state(
                worktree_id,
                expected=WorktreeState.PENDING,
                new=WorktreeState.REVIEW,
            ),
        ),
        ("reset_running_worktrees", store.reset_running_worktrees),
        (
            "mark_agent_session_started",
            lambda: store.mark_agent_session_started(
                worktree_id,
                AsyncOrchestratorAgentRole.WORKER,
            ),
        ),
        (
            "set_agent_session_usage",
            lambda: store.set_agent_session_usage(
                worktree_id,
                AsyncOrchestratorAgentRole.WORKER,
                Usage(),
            ),
        ),
        (
            "append_discussion",
            lambda: store.append_discussion(
                worktree_id,
                role=AsyncOrchestratorAgentRole.WORKER,
                message="hello",
            ),
        ),
        (
            "append_review_verdict",
            lambda: store.append_review_verdict(worktree_id, verdict),
        ),
        ("replace_task_snapshot", lambda: store.replace_task_snapshot(TaskManager())),
        ("get_worktree", lambda: store.get_worktree(worktree_id)),
        ("list_worktrees", store.list_worktrees),
        ("worktrees_for_task", lambda: store.worktrees_for_task("task-1")),
        (
            "non_closed_worktrees_for_task",
            lambda: store.non_closed_worktrees_for_task("task-1"),
        ),
        ("next_worktree_sequence", store.next_worktree_sequence),
        ("load_task_snapshot", store.load_task_snapshot),
        ("aggregate_usage", lambda: store.aggregate_usage(tmp_path)),
    )
    for name, operation in operations:
        try:
            operation()
        except AsyncOrchestratorControlSchemaError as exc:
            assert "unsupported" in str(exc), name
        else:  # pragma: no cover - assertion path
            pytest.fail(f"{name} did not reject the unsupported schema")


def test_control_store_list_active_agents_skips_initialize_when_schema_is_current(
    tmp_path: Path,
) -> None:
    class NoInitializeControlStore(SQLiteAsyncOrchestratorStore):
        def initialize(self) -> None:
            raise AssertionError("list_active_agents should not initialize current schema")

    store = SQLiteAsyncOrchestratorStore(tmp_path)
    store.register_run(run_id="run_test", pid=os.getpid())
    store.record_run_heartbeat(
        run_id="run_test",
        status="running",
        active_agents=(
            ControlActiveAgentSnapshot(role="worker", worktree_id="worktree_000001"),
        ),
    )
    read_store = NoInitializeControlStore(tmp_path)

    agents = read_store.list_active_agents(run_id="run_test")

    assert [(agent.role, agent.worktree_id) for agent in agents] == [
        ("worker", "worktree_000001")
    ]


def test_control_store_orders_runs_by_insertion(tmp_path: Path) -> None:
    store = SQLiteAsyncOrchestratorStore(tmp_path)
    first = store.register_run(run_id="run_first", pid=os.getpid())
    second = store.register_run(run_id="run_second", pid=os.getpid())
    with sqlite3.connect(store.path) as conn:
        conn.execute("UPDATE runs SET started_at = '2026-01-01T00:00:00Z'")

    latest = store.latest_run()
    assert first.run_id == "run_first"
    assert latest is not None
    assert latest.run_id == second.run_id


def test_control_store_enqueues_command_for_latest_active_run(tmp_path: Path) -> None:
    store = SQLiteAsyncOrchestratorStore(tmp_path)
    store.register_run(run_id="run_first", pid=os.getpid())

    queued = store.enqueue_command_for_latest_active_run("pause")

    assert queued is not None
    run, command = queued
    assert run.run_id == "run_first"
    assert command.run_id == "run_first"
    assert command.command == "pause"
    assert command.status == "pending"

    store.register_run(run_id="run_second", pid=os.getpid())
    store.record_run_finished(run_id="run_second", status="completed")

    assert store.enqueue_command_for_latest_active_run("pause") is None


def test_control_store_claims_commands_by_insertion_order(tmp_path: Path) -> None:
    store = SQLiteAsyncOrchestratorStore(tmp_path)
    store.register_run(run_id="run_test", pid=os.getpid())
    with sqlite3.connect(store.path) as conn:
        conn.execute(
            """
            INSERT INTO control_commands (
              id, command, params_json, status, created_at
            ) VALUES ('cmd_z', 'noop', '{}', 'pending', '2026-01-01T00:00:00Z')
            """
        )
        conn.execute(
            """
            INSERT INTO control_commands (
              id, command, params_json, status, created_at
            ) VALUES ('cmd_a', 'noop', '{}', 'pending', '2026-01-01T00:00:00Z')
            """
        )

    claimed = store.claim_pending_command(run_id="run_test")
    recent_ids = tuple(command.id for command in store.list_commands(limit=2))

    assert claimed is not None
    assert claimed.id == "cmd_z"
    assert recent_ids == ("cmd_a", "cmd_z")


def test_control_store_claims_and_completes_commands(tmp_path: Path) -> None:
    store = SQLiteAsyncOrchestratorStore(tmp_path)
    store.register_run(run_id="run_test", pid=os.getpid())
    pending = store.enqueue_command("noop", params={"source": "test"})

    assert pending.run_id is None
    assert pending.status == "pending"
    assert pending.params == {"source": "test"}

    claimed = store.claim_pending_command(run_id="run_test")

    assert claimed is not None
    assert claimed.id == pending.id
    assert claimed.run_id == "run_test"
    assert claimed.status == "claimed"
    assert claimed.claimed_at is not None

    completed = store.record_command_succeeded(
        claimed.id,
        result={"applied": True},
    )

    assert completed.status == "succeeded"
    assert completed.completed_at is not None
    assert completed.result == {"applied": True}
    assert completed.error is None
    assert store.get_command(claimed.id) == completed
    assert store.claim_pending_command(run_id="run_test") is None
    latest_run = store.latest_run()
    assert latest_run is not None
    assert latest_run.applied_command_id == claimed.id


def test_control_store_rejects_non_finite_json_when_enqueuing(
    tmp_path: Path,
) -> None:
    store = SQLiteAsyncOrchestratorStore(tmp_path)

    with pytest.raises(AsyncOrchestratorControlCommandError, match="finite"):
        store.enqueue_command("noop", params={"bad": float("nan")})


def test_control_store_fails_malformed_commands_before_claiming(
    tmp_path: Path,
) -> None:
    store = SQLiteAsyncOrchestratorStore(tmp_path)
    store.register_run(run_id="run_test", pid=os.getpid())
    with sqlite3.connect(store.path) as conn:
        conn.execute(
            """
            INSERT INTO control_commands (
              id, command, params_json, status, created_at
            ) VALUES ('cmd_bad', 'noop', 'not-json', 'pending', '2000-01-01T00:00:00Z')
            """
        )
        conn.execute(
            """
            INSERT INTO control_commands (
              id, command, params_json, status, created_at
            ) VALUES ('cmd_nan', 'noop', '{"bad": NaN}', 'pending', '2000-01-01T00:00:01Z')
            """
        )
        conn.execute(
            """
            INSERT INTO control_commands (
              id, command, params_json, status, created_at, result_json
            ) VALUES (
              'cmd_overflow', 'noop', '{"nested": {"bad": 1e9999}}',
              'pending', '2000-01-01T00:00:02Z', '{"stale": [1e9999]}'
            )
            """
        )
    valid = store.enqueue_command("noop", params={"source": "test"})

    claimed = store.claim_pending_command(run_id="run_test")

    assert claimed is not None
    assert claimed.id == valid.id
    bad = store.get_command("cmd_bad")
    assert bad is not None
    assert bad.status == "failed"
    assert bad.run_id == "run_test"
    assert bad.claimed_at is None
    assert bad.completed_at is not None
    assert bad.params == {}
    assert bad.result == {}
    assert bad.error is not None
    assert "invalid JSON" in bad.error
    nan_command = store.get_command("cmd_nan")
    assert nan_command is not None
    assert nan_command.status == "failed"
    assert nan_command.error is not None
    assert "non-finite" in nan_command.error
    overflow_command = store.get_command("cmd_overflow")
    assert overflow_command is not None
    assert overflow_command.status == "failed"
    assert overflow_command.run_id == "run_test"
    assert overflow_command.claimed_at is None
    assert overflow_command.result == {}
    assert overflow_command.error is not None
    assert "non-finite" in overflow_command.error


def test_control_store_rejects_overflow_result_json_when_reading(tmp_path: Path) -> None:
    store = SQLiteAsyncOrchestratorStore(tmp_path)
    store.register_run(run_id="run_test", pid=os.getpid())
    with sqlite3.connect(store.path) as conn:
        conn.execute(
            """
            INSERT INTO control_commands (
              id, run_id, command, params_json, status, created_at,
              completed_at, result_json
            ) VALUES (
              'cmd_result_overflow', 'run_test', 'noop', '{}', 'succeeded',
              '2000-01-01T00:00:00Z', '2000-01-01T00:00:01Z',
              '{"nested": [1e9999]}'
            )
            """
        )

    with pytest.raises(AsyncOrchestratorControlCommandError, match="non-finite"):
        store.get_command("cmd_result_overflow")


def test_control_store_claim_clears_preexisting_result_fields(tmp_path: Path) -> None:
    store = SQLiteAsyncOrchestratorStore(tmp_path)
    store.register_run(run_id="run_test", pid=os.getpid())
    with sqlite3.connect(store.path) as conn:
        conn.execute(
            """
            INSERT INTO control_commands (
              id, command, params_json, status, created_at, completed_at,
              result_json, error
            ) VALUES (
              'cmd_dirty', 'noop', '{}', 'pending', '2000-01-01T00:00:00Z',
              '2000-01-01T00:00:01Z', 'not-json', 'old error'
            )
            """
        )

    claimed = store.claim_pending_command(run_id="run_test")

    assert claimed is not None
    assert claimed.id == "cmd_dirty"
    assert claimed.status == "claimed"
    assert claimed.completed_at is None
    assert claimed.result is None
    assert claimed.error is None


def test_control_store_cancels_incomplete_run_commands(tmp_path: Path) -> None:
    store = SQLiteAsyncOrchestratorStore(tmp_path)
    store.register_run(run_id="run_test", pid=os.getpid())
    claimed_source = store.enqueue_command("noop", run_id="run_test")
    claimed = store.claim_pending_command(run_id="run_test")
    assert claimed is not None
    assert claimed.id == claimed_source.id
    targeted = store.enqueue_command("noop", run_id="run_test")
    untargeted = store.enqueue_command("noop")

    cancelled_count = store.cancel_incomplete_commands_for_run(
        run_id="run_test",
        error="run cancelled",
    )

    assert cancelled_count == 2
    claimed_after = store.get_command(claimed.id)
    assert claimed_after is not None
    assert claimed_after.status == "cancelled"
    assert claimed_after.result == {}
    assert claimed_after.error == "run cancelled"
    targeted_after = store.get_command(targeted.id)
    assert targeted_after is not None
    assert targeted_after.status == "cancelled"
    untargeted_after = store.get_command(untargeted.id)
    assert untargeted_after is not None
    assert untargeted_after.status == "pending"


def test_control_store_records_failed_commands(tmp_path: Path) -> None:
    store = SQLiteAsyncOrchestratorStore(tmp_path)
    store.register_run(run_id="run_test", pid=os.getpid())
    pending = store.enqueue_command("pause")
    claimed = store.claim_pending_command(run_id="run_test")

    assert claimed is not None
    assert claimed.id == pending.id

    failed = store.record_command_failed(claimed.id, error="not implemented")

    assert failed.status == "failed"
    assert failed.result == {}
    assert failed.error == "not implemented"
    latest_run = store.latest_run()
    assert latest_run is not None
    assert latest_run.applied_command_id is None
