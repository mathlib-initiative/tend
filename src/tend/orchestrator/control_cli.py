"""Command-line controls for a live async orchestrator run."""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from enum import IntEnum
from pathlib import Path
from typing import NoReturn, TextIO, cast

from pydantic import JsonValue

from tend._common.errors import FrameworkError
from tend.orchestrator.control_store import (
    ControlActiveAgentRecord,
    ControlCommandName,
    ControlCommandRecord,
    ControlRunRecord,
    SQLiteAsyncOrchestratorStore,
)
from tend.orchestrator.state import AsyncOrchestratorWorktree, WorktreeState
from tend.orchestrator.task_manager import TaskManager
from tend.orchestrator.tasks import TaskStatus
from tend.orchestrator.usage import format_usage_summary

_TERMINAL_RUN_STATUSES = frozenset({"stopped", "completed", "failed"})
_TERMINAL_COMMAND_STATUSES = frozenset({"succeeded", "failed", "cancelled"})
_DEFAULT_WAIT_POLL_INTERVAL_SECONDS = 0.25


class AsyncOrchestratorControlCliExitCode(IntEnum):
    """Process exit codes returned by ``tend-control``."""

    SUCCESS = 0
    ERROR = 1
    CONFIGURATION_OR_USAGE = 2


class AsyncOrchestratorControlCliError(Exception):
    """Expected CLI error with a stable error code for stderr output."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class _ControlArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise AsyncOrchestratorControlCliError("cli_usage_error", message)


@dataclass(frozen=True, slots=True)
class _WaitDeadline:
    """Shared timeout budget for command and run waits."""

    started_at: float
    timeout_seconds: float | None

    @classmethod
    def from_timeout(cls, timeout_seconds: float | None) -> _WaitDeadline:
        return cls(started_at=time.monotonic(), timeout_seconds=timeout_seconds)

    def remaining_seconds(self) -> float | None:
        timeout = self.timeout_seconds
        if timeout is None:
            return None
        return max(0.0, timeout - (time.monotonic() - self.started_at))

    def timed_out(self) -> bool:
        remaining = self.remaining_seconds()
        return remaining is not None and remaining <= 0.0


def main(argv: Sequence[str] | None = None) -> int:
    """Console-script entrypoint for ``tend-control``."""

    return run_control_cli(argv)


def run_control_cli(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    prog: str = "tend-control",
) -> int:
    """Parse CLI args, run one control command, and return an exit code."""

    args = tuple(sys.argv[1:] if argv is None else argv)
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr
    try:
        namespace = _build_parser(prog).parse_args(list(args))
        command = cast(str, namespace.command)
        if command == "status":
            _handle_status(namespace, stdout=out)
            return int(AsyncOrchestratorControlCliExitCode.SUCCESS)
        _handle_enqueue_command(namespace, stdout=out)
        return int(AsyncOrchestratorControlCliExitCode.SUCCESS)
    except AsyncOrchestratorControlCliError as exc:
        _write_error(exc.code, str(exc), err)
        return int(AsyncOrchestratorControlCliExitCode.CONFIGURATION_OR_USAGE)
    except FrameworkError as exc:
        _write_error("control_error", str(exc), err)
        return int(AsyncOrchestratorControlCliExitCode.ERROR)
    except OSError as exc:
        _write_error("filesystem_error", str(exc), err)
        return int(AsyncOrchestratorControlCliExitCode.ERROR)


def _handle_status(namespace: argparse.Namespace, *, stdout: TextIO) -> None:
    root = _required_root(namespace)
    store = SQLiteAsyncOrchestratorStore(root)

    stdout.write("async orchestrator control status\n")
    stdout.write(f"root: {root}\n")

    if store.path.exists():
        run = store.latest_run()
        stdout.write(f"control: loaded ({store.path})\n")
        if run is None:
            stdout.write("run: none\n")
        else:
            stdout.write(f"run: {run.run_id}\n")
            stdout.write(
                f"run_status: {run.status}{_reason_suffix(run.status_reason)}\n"
            )
            stdout.write(f"pid: {run.pid}\n")
            stdout.write(f"started_at: {run.started_at}\n")
            stdout.write(f"heartbeat_at: {run.heartbeat_at}\n")
            heartbeat_age = _heartbeat_age_seconds(run)
            if heartbeat_age is not None:
                stdout.write(f"heartbeat_age_seconds: {heartbeat_age:.1f}\n")
            stdout.write(
                "limits: "
                f"workers={_optional_int(run.worker_limit)}, "
                f"reviewers={_optional_int(run.reviewer_limit)}\n"
            )
            stdout.write(
                "flags: "
                f"paused={_bool_word(run.paused)}, "
                f"drain_requested={_bool_word(run.drain_requested)}\n"
            )
            stdout.write(
                "last_applied_command: "
                f"{run.applied_command_id if run.applied_command_id is not None else 'none'}\n"
            )
            _write_active_agents(store.list_active_agents(run_id=run.run_id), stdout=stdout)
        _write_recent_commands(store, limit=cast(int, namespace.commands), stdout=stdout)
    else:
        stdout.write(f"control: missing ({store.path})\n")
        stdout.write("run: unavailable\n")

    if store.state_exists():
        task_manager = store.load_task_snapshot()
        worktrees = store.list_worktrees()
        stdout.write(f"state: loaded ({store.path})\n")
        stdout.write(f"tasks: {_task_counts_summary(task_manager)}\n")
        stdout.write(f"worktrees: {_worktree_counts_summary(worktrees)}\n")
        stdout.write(f"inferred queues: {_queue_counts_summary(worktrees)}\n")
        usage = store.aggregate_usage(root)
        stdout.write(f"usage: loaded ({store.path})\n")
        stdout.write(f"aggregate {format_usage_summary(usage)}\n")
    else:
        stdout.write(f"state: missing ({store.path})\n")
        stdout.write("tasks: unavailable\n")
        stdout.write("worktrees: unavailable\n")
        stdout.write("inferred queues: unavailable\n")
        stdout.write(f"usage: missing ({store.path})\n")


def _write_active_agents(
    active_agents: Sequence[ControlActiveAgentRecord],
    *,
    stdout: TextIO,
) -> None:
    worker_count = sum(1 for agent in active_agents if agent.role == "worker")
    reviewer_count = sum(1 for agent in active_agents if agent.role == "reviewer")
    stdout.write(
        "active_agents: "
        f"workers={worker_count}, reviewers={reviewer_count}, "
        f"total={len(active_agents)}\n"
    )
    for agent in active_agents:
        stdout.write(f"  {agent.role} {agent.worktree_id}")
        if agent.task_id is not None:
            stdout.write(f" task={agent.task_id}")
        if agent.worktree_state is not None:
            stdout.write(f" worktree_state={agent.worktree_state}")
        stdout.write(f" recorded_at={agent.recorded_at}\n")


def _write_recent_commands(
    store: SQLiteAsyncOrchestratorStore,
    *,
    limit: int,
    stdout: TextIO,
) -> None:
    commands = store.list_commands(limit=limit)
    stdout.write(f"recent_commands: {len(commands)}\n")
    for command in commands:
        stdout.write(
            "  "
            f"{command.id} {command.command} {command.status}"
            f" run={command.run_id if command.run_id is not None else 'unbound'}"
            f" created_at={command.created_at}"
        )
        if command.error is not None:
            stdout.write(f" error={command.error}")
        stdout.write("\n")


def _handle_enqueue_command(namespace: argparse.Namespace, *, stdout: TextIO) -> None:
    root = _required_root(namespace)
    store = SQLiteAsyncOrchestratorStore(root)
    command_name = cast(ControlCommandName, namespace.command)
    params = _params_for_command(namespace)
    _reject_stale_latest_run_if_requested(
        store,
        cast(float | None, namespace.max_heartbeat_age),
    )
    queued = store.enqueue_command_for_latest_active_run(command_name, params=params)
    if queued is None:
        raise AsyncOrchestratorControlCliError(
            "no_active_run",
            "no active async orchestrator run is registered in the control database",
        )
    run, command = queued
    stdout.write(f"queued {command.command} command: {command.id} (run {run.run_id})\n")

    if not cast(bool, namespace.wait):
        return

    deadline = _WaitDeadline.from_timeout(cast(float | None, namespace.wait_timeout))
    poll_interval = cast(float, namespace.poll_interval)
    completed = _wait_for_command(store, command.id, deadline, poll_interval)
    stdout.write(f"command {completed.status}: {completed.id}\n")
    if completed.error is not None:
        stdout.write(f"command_error: {completed.error}\n")
    if completed.status != "succeeded":
        raise AsyncOrchestratorControlCliError(
            "command_failed",
            f"control command {completed.id} {completed.status}",
        )

    if completed.command in {"drain", "stop"}:
        finished = _wait_for_run_terminal(store, run.run_id, deadline, poll_interval)
        stdout.write(
            f"run finished: {finished.status}{_reason_suffix(finished.status_reason)}\n"
        )
        if finished.status == "failed":
            raise AsyncOrchestratorControlCliError(
                "run_failed",
                "async orchestrator run failed while waiting for "
                f"{completed.command}: {finished.status_reason or 'unknown reason'}",
            )


def _reject_stale_latest_run_if_requested(
    store: SQLiteAsyncOrchestratorStore,
    max_heartbeat_age: float | None,
) -> None:
    if max_heartbeat_age is None:
        return
    run = store.latest_run()
    if run is None or run.status in _TERMINAL_RUN_STATUSES:
        return
    heartbeat_age = _heartbeat_age_seconds(run)
    if heartbeat_age is None:
        raise AsyncOrchestratorControlCliError(
            "stale_run",
            "latest active async orchestrator run has an unreadable heartbeat; "
            "refusing to enqueue control command",
        )
    if heartbeat_age > max_heartbeat_age:
        raise AsyncOrchestratorControlCliError(
            "stale_run",
            "latest active async orchestrator run heartbeat is stale "
            f"(age {heartbeat_age:.1f}s > max {max_heartbeat_age:.1f}s); "
            "refusing to enqueue control command",
        )


def _wait_for_command(
    store: SQLiteAsyncOrchestratorStore,
    command_id: str,
    deadline: _WaitDeadline,
    poll_interval: float,
) -> ControlCommandRecord:
    while True:
        command = store.get_command(command_id)
        if command is None:
            raise AsyncOrchestratorControlCliError(
                "command_missing",
                f"control command {command_id} disappeared from the control database",
            )
        if command.status in _TERMINAL_COMMAND_STATUSES:
            return command
        if deadline.timed_out():
            raise AsyncOrchestratorControlCliError(
                "wait_timeout",
                f"timed out waiting for control command {command_id} to complete",
            )
        time.sleep(_bounded_sleep_seconds(deadline, poll_interval))


def _wait_for_run_terminal(
    store: SQLiteAsyncOrchestratorStore,
    run_id: str,
    deadline: _WaitDeadline,
    poll_interval: float,
) -> ControlRunRecord:
    while True:
        run = store.get_run(run_id)
        if run is None:
            raise AsyncOrchestratorControlCliError(
                "run_missing",
                f"async orchestrator run {run_id} disappeared from the control database",
            )
        if run.status in _TERMINAL_RUN_STATUSES:
            return run
        if deadline.timed_out():
            raise AsyncOrchestratorControlCliError(
                "wait_timeout",
                f"timed out waiting for async orchestrator run {run_id} to finish",
            )
        time.sleep(_bounded_sleep_seconds(deadline, poll_interval))


def _params_for_command(namespace: argparse.Namespace) -> Mapping[str, JsonValue]:
    command = cast(str, namespace.command)
    if command in {"pause", "resume", "drain"}:
        return {}
    if command == "stop":
        return {"now": cast(bool, namespace.now)}
    if command == "limits":
        params: dict[str, JsonValue] = {}
        workers = cast(int | None, namespace.workers)
        reviewers = cast(int | None, namespace.reviewers)
        if workers is not None:
            params["workers"] = workers
        if reviewers is not None:
            params["reviewers"] = reviewers
        if not params:
            raise AsyncOrchestratorControlCliError(
                "cli_usage_error",
                "limits requires at least one of --workers or --reviewers",
            )
        return params
    if command == "budget":
        return {"max_cost": format(cast(Decimal, namespace.max_cost), "f")}
    raise AsyncOrchestratorControlCliError("cli_usage_error", f"unknown command: {command}")


def _bounded_sleep_seconds(deadline: _WaitDeadline, poll_interval: float) -> float:
    remaining = deadline.remaining_seconds()
    if remaining is None:
        return poll_interval
    return min(poll_interval, remaining)


def _task_counts_summary(task_manager: TaskManager) -> str:
    counts = {status: 0 for status in TaskStatus}
    for task in task_manager.tasks:
        counts[task.status] += 1
    parts = [f"total={len(task_manager.tasks)}"]
    parts.extend(f"{status.value}={counts[status]}" for status in TaskStatus)
    return ", ".join(parts)


def _worktree_counts_summary(worktrees: Sequence[AsyncOrchestratorWorktree]) -> str:
    counts = {worktree_state: 0 for worktree_state in WorktreeState}
    for worktree in worktrees:
        counts[worktree.state] += 1
    parts = [f"total={len(worktrees)}"]
    parts.extend(
        f"{worktree_state.value}={counts[worktree_state]}"
        for worktree_state in WorktreeState
    )
    return ", ".join(parts)


def _queue_counts_summary(worktrees: Sequence[AsyncOrchestratorWorktree]) -> str:
    worker_count = sum(
        1
        for worktree in worktrees
        if worktree.state is WorktreeState.PENDING and worktree.task_id is not None
    )
    reviewer_count = sum(
        1
        for worktree in worktrees
        if worktree.state is WorktreeState.REVIEW
    )
    merge_count = sum(
        1
        for worktree in worktrees
        if worktree.state is WorktreeState.MERGE
    )
    return f"worker={worker_count}, reviewer={reviewer_count}, merge={merge_count}"


def _build_parser(prog: str) -> argparse.ArgumentParser:
    parser = _ControlArgumentParser(
        prog=prog,
        description="Inspect and control a live async orchestrator run.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    status = subparsers.add_parser(
        "status",
        help="Print control, state, and usage status for an orchestration root.",
    )
    _add_root_arg(status)
    status.add_argument(
        "--commands",
        type=_non_negative_int_arg,
        default=5,
        metavar="N",
        help="Number of recent control commands to print (default: 5).",
    )

    for command, help_text in (
        ("pause", "Pause new cost-incurring work."),
        ("resume", "Resume new cost-incurring work after a pause."),
        ("drain", "Stop creating fresh work; drain queued work with nonzero limits."),
    ):
        subparser = subparsers.add_parser(command, help=help_text)
        _add_root_arg(subparser)
        _add_wait_args(subparser)

    stop = subparsers.add_parser(
        "stop",
        help="Request terminal stop semantics for the live run.",
    )
    _add_root_arg(stop)
    stop.add_argument(
        "--now",
        action="store_true",
        help="Cancel currently running worker/reviewer agents after requesting stop.",
    )
    _add_wait_args(stop)

    limits = subparsers.add_parser(
        "limits",
        help="Change live worker/reviewer launch limits.",
    )
    _add_root_arg(limits)
    limits.add_argument(
        "--workers",
        type=_non_negative_int_arg,
        metavar="N",
        help="Set the live worker-agent launch limit (0 pauses launches unless draining).",
    )
    limits.add_argument(
        "--reviewers",
        type=_non_negative_int_arg,
        metavar="N",
        help="Set the live reviewer-agent launch limit (0 pauses launches unless draining).",
    )
    _add_wait_args(limits)

    budget = subparsers.add_parser(
        "budget",
        help="Change the live max-cost ceiling for the run.",
    )
    _add_root_arg(budget)
    budget.add_argument(
        "--max-cost",
        required=True,
        type=_max_cost_arg,
        metavar="AMOUNT",
        help="Set the live max-cost ceiling in the run's configured budget currency.",
    )
    _add_wait_args(budget)

    return parser


def _add_root_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--root",
        required=True,
        type=Path,
        help="Async orchestrator root directory.",
    )


def _add_wait_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for the command to be applied; drain/stop also wait for run exit.",
    )
    parser.add_argument(
        "--wait-timeout",
        type=_positive_float_arg,
        metavar="SECONDS",
        help="Maximum seconds to wait before failing (default: wait forever).",
    )
    parser.add_argument(
        "--poll-interval",
        type=_positive_float_arg,
        default=_DEFAULT_WAIT_POLL_INTERVAL_SECONDS,
        metavar="SECONDS",
        help=f"Polling interval while waiting (default: {_DEFAULT_WAIT_POLL_INTERVAL_SECONDS}).",
    )
    parser.add_argument(
        "--max-heartbeat-age",
        type=_positive_float_arg,
        metavar="SECONDS",
        help="Refuse to enqueue if the latest active run heartbeat is older than this.",
    )


def _required_root(namespace: argparse.Namespace) -> Path:
    value = getattr(namespace, "root", None)
    if not isinstance(value, Path):
        raise AsyncOrchestratorControlCliError("cli_usage_error", "--root is required")
    return value.expanduser().resolve()


def _non_negative_int_arg(value: str) -> int:
    try:
        parsed = int(value, 10)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _positive_float_arg(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a number, got {value!r}") from exc
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("value must be finite")
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed


def _max_cost_arg(value: str) -> Decimal:
    text = value.strip()
    try:
        parsed = Decimal(text)
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError(f"expected a decimal amount, got {value!r}") from exc
    if not parsed.is_finite():
        raise argparse.ArgumentTypeError("max cost must be finite")
    if parsed <= 0:
        raise argparse.ArgumentTypeError("max cost must be greater than 0")
    return parsed


def _heartbeat_age_seconds(run: ControlRunRecord) -> float | None:
    heartbeat = _parse_timestamp(run.heartbeat_at)
    if heartbeat is None:
        return None
    return max(0.0, (datetime.now(UTC) - heartbeat).total_seconds())


def _parse_timestamp(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _optional_int(value: int | None) -> str:
    return "unknown" if value is None else str(value)


def _bool_word(value: bool) -> str:
    return "true" if value else "false"


def _reason_suffix(reason: str | None) -> str:
    return "" if reason is None else f" ({reason})"


def _write_error(code: str, message: str, stderr: TextIO) -> None:
    stderr.write(f"error[{code}]: {message}\n")


__all__ = (
    "AsyncOrchestratorControlCliError",
    "AsyncOrchestratorControlCliExitCode",
    "main",
    "run_control_cli",
)
