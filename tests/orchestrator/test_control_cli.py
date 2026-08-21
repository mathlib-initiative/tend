from __future__ import annotations

import asyncio
import os
import sqlite3
import threading
import time
from datetime import UTC, datetime, timedelta
from io import StringIO
from pathlib import Path

from tend.orchestrator.config import AsyncOrchestratorConfig
from tend.orchestrator.control_cli import (
    AsyncOrchestratorControlCliExitCode,
    run_control_cli,
)
from tend.orchestrator.control_store import (
    ControlActiveAgentSnapshot,
    SQLiteAsyncOrchestratorStore,
)
from tend.orchestrator.orchestrator import AsyncOrchestrator
from tend.orchestrator.state import WorktreeState
from tend.orchestrator.task_manager import TaskManager
from tend.orchestrator.tasks import Task


def test_control_cli_status_prints_control_store_summary(tmp_path: Path) -> None:
    root = tmp_path / "orch"
    store = SQLiteAsyncOrchestratorStore(root)
    store.register_run(
        run_id="run_test",
        pid=123,
        status="running",
        worker_limit=4,
        reviewer_limit=2,
        paused=True,
    )
    store.record_run_heartbeat(
        run_id="run_test",
        status="running",
        worker_limit=4,
        reviewer_limit=2,
        paused=True,
        active_agents=(
            ControlActiveAgentSnapshot(
                role="worker",
                worktree_id="worktree_000001",
                task_id="task-a",
                worktree_state="worker_running",
            ),
        ),
    )
    store.enqueue_command("noop", run_id="run_test")
    task = Task(id="task-a", title="Task A", summary="Task A", description="Do it.")
    store.replace_task_snapshot(TaskManager(tasks=[task]))
    worktree_id = store.allocate_worktree(
        task_id=task.id,
        path=root / "worktrees" / "worktree_000001",
        head="abc123",
    )
    assert store.set_worktree_state(
        worktree_id,
        expected=WorktreeState.PENDING,
        new=WorktreeState.REVIEW,
    )
    stdout = StringIO()

    exit_code = run_control_cli(
        ["status", "--root", str(root), "--commands", "1"],
        stdout=stdout,
    )

    output = stdout.getvalue()
    assert exit_code == int(AsyncOrchestratorControlCliExitCode.SUCCESS)
    assert "async orchestrator control status" in output
    assert f"control: loaded ({store.path})" in output
    assert "run: run_test" in output
    assert "run_status: running" in output
    assert "limits: workers=4, reviewers=2" in output
    assert "flags: paused=true, drain_requested=false" in output
    assert "active_agents: workers=1, reviewers=0, total=1" in output
    assert "worker worktree_000001 task=task-a worktree_state=worker_running" in output
    assert "recent_commands: 1" in output
    assert "state: loaded" in output
    assert "tasks: total=1, open=1, complete=0" in output
    assert "worktrees: total=1, pending=0, worker_running=0, review=1, merge=0, closed=0" in output
    assert "usage: loaded" in output


def test_control_cli_enqueues_command_for_latest_active_run(tmp_path: Path) -> None:
    root = tmp_path / "orch"
    store = SQLiteAsyncOrchestratorStore(root)
    store.register_run(run_id="run_test", pid=os.getpid())
    stdout = StringIO()

    exit_code = run_control_cli(["pause", "--root", str(root)], stdout=stdout)

    commands = store.list_commands(limit=1)
    assert exit_code == int(AsyncOrchestratorControlCliExitCode.SUCCESS)
    assert len(commands) == 1
    assert commands[0].command == "pause"
    assert commands[0].run_id == "run_test"
    assert commands[0].status == "pending"
    assert "queued pause command" in stdout.getvalue()


def test_control_cli_rejects_stale_latest_run_when_guard_requested(
    tmp_path: Path,
) -> None:
    root = tmp_path / "orch"
    store = SQLiteAsyncOrchestratorStore(root)
    store.register_run(run_id="run_test", pid=os.getpid())
    stale_heartbeat = (datetime.now(UTC) - timedelta(seconds=120)).isoformat()
    with sqlite3.connect(store.path) as conn:
        conn.execute(
            "UPDATE runs SET heartbeat_at = ? WHERE run_id = ?",
            (stale_heartbeat, "run_test"),
        )
    stderr = StringIO()

    exit_code = run_control_cli(
        [
            "pause",
            "--root",
            str(root),
            "--max-heartbeat-age",
            "1",
        ],
        stderr=stderr,
    )

    assert exit_code == int(AsyncOrchestratorControlCliExitCode.CONFIGURATION_OR_USAGE)
    assert "heartbeat is stale" in stderr.getvalue()
    assert store.list_commands() == ()


def test_control_cli_rejects_commands_when_latest_run_is_terminal(
    tmp_path: Path,
) -> None:
    root = tmp_path / "orch"
    store = SQLiteAsyncOrchestratorStore(root)
    store.register_run(run_id="run_test", pid=os.getpid())
    store.record_run_finished(run_id="run_test", status="completed")
    stderr = StringIO()

    exit_code = run_control_cli(["pause", "--root", str(root)], stderr=stderr)

    assert exit_code == int(AsyncOrchestratorControlCliExitCode.CONFIGURATION_OR_USAGE)
    assert "no active async orchestrator run" in stderr.getvalue()
    assert store.list_commands() == ()


def test_control_cli_rejects_non_finite_wait_values(tmp_path: Path) -> None:
    root = tmp_path / "orch"
    SQLiteAsyncOrchestratorStore(root).register_run(run_id="run_test", pid=os.getpid())

    for args in (
        ["pause", "--root", str(root), "--wait", "--wait-timeout", "nan"],
        ["pause", "--root", str(root), "--wait", "--poll-interval", "inf"],
    ):
        stderr = StringIO()
        exit_code = run_control_cli(args, stderr=stderr)
        assert exit_code == int(AsyncOrchestratorControlCliExitCode.CONFIGURATION_OR_USAGE)
        assert "value must be finite" in stderr.getvalue()


def test_control_cli_drain_wait_fails_when_run_fails(tmp_path: Path) -> None:
    root = tmp_path / "orch"
    run_id = "run_test"
    store = SQLiteAsyncOrchestratorStore(root)
    store.register_run(run_id=run_id, pid=os.getpid())
    stdout = StringIO()
    stderr = StringIO()

    thread_errors: list[BaseException] = []

    def apply_and_fail_run() -> None:
        try:
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                command = store.claim_pending_command(run_id=run_id)
                if command is not None:
                    store.record_command_succeeded(
                        command.id,
                        result={"drain_requested": True},
                    )
                    store.record_run_finished(
                        run_id=run_id,
                        status="failed",
                        status_reason="boom",
                    )
                    return
                time.sleep(0.01)
            raise AssertionError("command was not enqueued")
        except BaseException as exc:  # pragma: no cover - re-raised in main thread
            thread_errors.append(exc)

    worker = threading.Thread(target=apply_and_fail_run)
    worker.start()
    try:
        exit_code = run_control_cli(
            [
                "drain",
                "--root",
                str(root),
                "--wait",
                "--wait-timeout",
                "2",
                "--poll-interval",
                "0.01",
            ],
            stdout=stdout,
            stderr=stderr,
        )
    finally:
        worker.join(timeout=2.0)

    assert thread_errors == []
    assert exit_code == int(AsyncOrchestratorControlCliExitCode.CONFIGURATION_OR_USAGE)
    assert "run finished: failed (boom)" in stdout.getvalue()
    assert "error[run_failed]" in stderr.getvalue()
    assert "boom" in stderr.getvalue()


def test_control_cli_wait_timeout_is_not_extended_by_poll_interval(
    tmp_path: Path,
) -> None:
    root = tmp_path / "orch"
    SQLiteAsyncOrchestratorStore(root).register_run(run_id="run_test", pid=os.getpid())
    stderr = StringIO()

    started = time.monotonic()
    exit_code = run_control_cli(
        [
            "pause",
            "--root",
            str(root),
            "--wait",
            "--wait-timeout",
            "0.01",
            "--poll-interval",
            "10",
        ],
        stderr=stderr,
    )
    elapsed = time.monotonic() - started

    assert exit_code == int(AsyncOrchestratorControlCliExitCode.CONFIGURATION_OR_USAGE)
    assert "timed out waiting for control command" in stderr.getvalue()
    assert elapsed < 1.0


def test_control_cli_rejects_limits_without_values(tmp_path: Path) -> None:
    root = tmp_path / "orch"
    SQLiteAsyncOrchestratorStore(root).register_run(run_id="run_test", pid=os.getpid())
    stderr = StringIO()

    exit_code = run_control_cli(["limits", "--root", str(root)], stderr=stderr)

    assert exit_code == int(AsyncOrchestratorControlCliExitCode.CONFIGURATION_OR_USAGE)
    assert "limits requires at least one" in stderr.getvalue()


async def test_control_cli_waits_for_live_pause_command(tmp_path: Path) -> None:
    discovery_started = asyncio.Event()
    root = tmp_path / "orch"

    class BlockingDiscoveryOrchestrator(AsyncOrchestrator):
        async def _enqueue_ready_tasks_forever(self) -> None:
            discovery_started.set()
            await asyncio.Event().wait()

    orchestrator = BlockingDiscoveryOrchestrator(
        AsyncOrchestratorConfig(root=root, entrypoint=tmp_path / "entrypoint"),
    )
    run_task = asyncio.create_task(orchestrator.run())

    try:
        await asyncio.wait_for(discovery_started.wait(), timeout=1.0)
        stdout = StringIO()
        stderr = StringIO()
        exit_code = await asyncio.to_thread(
            run_control_cli,
            [
                "pause",
                "--root",
                str(root),
                "--wait",
                "--wait-timeout",
                "3",
                "--poll-interval",
                "0.05",
            ],
            stdout=stdout,
            stderr=stderr,
        )

        assert exit_code == int(AsyncOrchestratorControlCliExitCode.SUCCESS)
        assert stderr.getvalue() == ""
        assert "command succeeded" in stdout.getvalue()
        assert orchestrator.runtime.paused is True
    finally:
        run_task.cancel()
        try:
            await run_task
        except asyncio.CancelledError:
            pass
