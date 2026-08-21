"""SQLite storage for live async orchestrator control and state."""

from __future__ import annotations

import json
import math
import sqlite3
import uuid
from collections.abc import Iterable, Mapping, Sequence
from contextlib import closing
from os import PathLike
from pathlib import Path
from typing import Literal, cast

from pydantic import JsonValue

from tend._common.agent_outputs import ReviewVerdictOutput
from tend._common.errors import FrameworkError
from tend._common.sqlite import (
    begin_immediate,
    connect_read_only,
    connect_read_write,
    map_sqlite_errors,
)
from tend._common.types import JsonObject, StrictModel, format_sequence_id, utc_timestamp
from tend.llm.usage import Usage
from tend.orchestrator.state import (
    AsyncOrchestratorAgentRole,
    AsyncOrchestratorDiscussionMessage,
    AsyncOrchestratorWorktree,
    WorktreeState,
)
from tend.orchestrator.task_manager import TaskManager
from tend.orchestrator.usage import aggregate_agent_session_usage

ASYNC_ORCHESTRATOR_DB_FILENAME = "orchestrator.sqlite"
ASYNC_ORCHESTRATOR_SCHEMA_VERSION = 1

type PathInput = str | PathLike[str]

ControlRunStatus = Literal[
    "starting",
    "running",
    "draining",
    "stopping",
    "stopped",
    "completed",
    "failed",
]
ControlCommandName = Literal[
    "noop",
    "pause",
    "resume",
    "drain",
    "stop",
    "limits",
    "budget",
]
ControlCommandStatus = Literal[
    "pending",
    "claimed",
    "succeeded",
    "failed",
    "cancelled",
]
ControlActiveAgentRole = Literal["worker", "reviewer"]

_RUN_STATUSES: tuple[str, ...] = (
    "starting",
    "running",
    "draining",
    "stopping",
    "stopped",
    "completed",
    "failed",
)
_COMMAND_NAMES: tuple[str, ...] = (
    "noop",
    "pause",
    "resume",
    "drain",
    "stop",
    "limits",
    "budget",
)
_COMMAND_STATUSES: tuple[str, ...] = (
    "pending",
    "claimed",
    "succeeded",
    "failed",
    "cancelled",
)
_TERMINAL_RUN_STATUSES: tuple[str, ...] = ("stopped", "completed", "failed")
_ACTIVE_AGENT_ROLES: tuple[str, ...] = ("worker", "reviewer")
_WORKTREE_STATES: tuple[str, ...] = (
    "pending",
    "worker_running",
    "review",
    "merge",
    "closed",
)
_DISCUSSION_ROLES: tuple[str, ...] = ("worker", "reviewer", "orchestrator")


def _sql_string_list(values: tuple[str, ...]) -> str:
    return "(" + ", ".join(f"'{value}'" for value in values) + ")"


class AsyncOrchestratorControlStoreError(FrameworkError):
    """Base error for async orchestrator control-store failures."""


class AsyncOrchestratorControlSchemaError(AsyncOrchestratorControlStoreError):
    """Raised when the control database schema is missing or unsupported."""


class AsyncOrchestratorControlStoreIOError(AsyncOrchestratorControlStoreError):
    """Raised when SQLite/filesystem operations fail for the control store."""


class AsyncOrchestratorControlCommandError(AsyncOrchestratorControlStoreError):
    """Raised when a control command cannot be read or updated."""


class ControlRunRecord(StrictModel):
    """Current durable metadata for one orchestrator run."""

    run_id: str
    pid: int
    started_at: str
    heartbeat_at: str
    status: ControlRunStatus
    status_reason: str | None = None
    applied_command_id: str | None = None
    worker_limit: int | None = None
    reviewer_limit: int | None = None
    paused: bool = False
    drain_requested: bool = False


class ControlCommandRecord(StrictModel):
    """Durable operator command exchanged through the control store."""

    id: str
    run_id: str | None = None
    command: ControlCommandName
    params: JsonObject
    status: ControlCommandStatus
    created_at: str
    claimed_at: str | None = None
    completed_at: str | None = None
    result: JsonObject | None = None
    error: str | None = None


class ControlActiveAgentSnapshot(StrictModel):
    """One live agent task sampled by the orchestrator heartbeat."""

    role: ControlActiveAgentRole
    worktree_id: str
    task_id: str | None = None
    worktree_state: str | None = None


class ControlActiveAgentRecord(ControlActiveAgentSnapshot):
    """Durable live-agent row stored for a run."""

    run_id: str
    recorded_at: str


class SQLiteAsyncOrchestratorStore:
    """SQLite-backed control and state store under an orchestration root.

    The store intentionally uses short, per-method connections and transactions.
    It is safe for the running orchestrator and external control clients to share
    without taking the orchestrator's exclusive root lock.
    """

    __slots__ = ("path", "root")

    root: Path
    path: Path

    def __init__(self, root: PathInput, *, path: PathInput | None = None) -> None:
        self.root = _to_path(root, field_name="root")
        self.path = (
            self.root / ASYNC_ORCHESTRATOR_DB_FILENAME
            if path is None
            else _to_path(path, field_name="path")
        )

    def initialize(self) -> None:
        """Create or verify the unified orchestrator schema."""

        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise AsyncOrchestratorControlStoreIOError(
                f"failed to create async orchestrator database directory "
                f"{self.path.parent}: {exc}"
            ) from exc

        with closing(self._connection()) as conn:
            with map_sqlite_errors(
                AsyncOrchestratorControlStoreIOError,
                f"failed to initialize async orchestrator database {self.path}",
            ):
                with conn:
                    version = _read_schema_version(conn)
                    if version is None:
                        if _database_has_user_tables(conn):
                            raise AsyncOrchestratorControlSchemaError(
                                "missing async orchestrator schema version in "
                                f"{self.path}"
                            )
                        conn.executescript(_SCHEMA_SQL)
                        conn.execute(
                            "INSERT INTO schema (id, version) VALUES (1, ?)",
                            (ASYNC_ORCHESTRATOR_SCHEMA_VERSION,),
                        )
                        _ensure_async_meta_row(conn)
                        return
                    _ensure_schema_current(conn)
                    conn.executescript(_SCHEMA_SQL)
                    _ensure_async_meta_row(conn)

    def register_run(
        self,
        *,
        run_id: str,
        pid: int,
        status: ControlRunStatus = "running",
        status_reason: str | None = None,
        worker_limit: int | None = None,
        reviewer_limit: int | None = None,
        paused: bool = False,
        drain_requested: bool = False,
    ) -> ControlRunRecord:
        """Insert and return a run row for a newly started orchestrator process."""

        self.initialize()
        now = utc_timestamp()
        with closing(self._connection()) as conn:
            with map_sqlite_errors(
                AsyncOrchestratorControlStoreIOError,
                f"failed to register async orchestrator run in {self.path}",
            ):
                with conn:
                    _ensure_schema_current(conn)
                    conn.execute(
                        """
                        INSERT INTO runs (
                          run_id, pid, started_at, heartbeat_at, status,
                          status_reason, worker_limit, reviewer_limit, paused,
                          drain_requested
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            run_id,
                            pid,
                            now,
                            now,
                            status,
                            status_reason,
                            worker_limit,
                            reviewer_limit,
                            _bool_to_int(paused),
                            _bool_to_int(drain_requested),
                        ),
                    )
                    row = _select_run(conn, run_id)
        if row is None:
            raise AsyncOrchestratorControlStoreIOError(
                f"registered async orchestrator run {run_id} was not found"
            )
        return _run_from_row(row)

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
        """Refresh a run heartbeat and live scheduling metadata.

        Heartbeats must not revive terminal rows: a delayed heartbeat from the
        control service can race with terminal cleanup, so terminal status is
        monotonic once recorded.
        """

        return self._update_run(
            run_id=run_id,
            status=status,
            status_reason=status_reason,
            worker_limit=worker_limit,
            reviewer_limit=reviewer_limit,
            paused=paused,
            drain_requested=drain_requested,
            active_agents=active_agents,
            preserve_terminal=True,
        )

    def record_run_finished(
        self,
        *,
        run_id: str,
        status: Literal["stopped", "completed", "failed"],
        status_reason: str | None = None,
        worker_limit: int | None = None,
        reviewer_limit: int | None = None,
        paused: bool = False,
        drain_requested: bool = False,
    ) -> ControlRunRecord:
        """Record the terminal state for a run."""

        return self._update_run(
            run_id=run_id,
            status=status,
            status_reason=status_reason,
            worker_limit=worker_limit,
            reviewer_limit=reviewer_limit,
            paused=paused,
            drain_requested=drain_requested,
            active_agents=(),
            preserve_terminal=False,
        )

    def get_run(self, run_id: str) -> ControlRunRecord | None:
        """Return one run row, or ``None`` when it is absent."""

        if not self.path.exists():
            return None
        with closing(self._connection()) as conn:
            with map_sqlite_errors(
                AsyncOrchestratorControlStoreIOError,
                f"failed to read async orchestrator run {run_id} from {self.path}",
            ):
                if not _ensure_schema_current(conn):
                    return None
                row = _select_run(conn, run_id)
        return None if row is None else _run_from_row(row)

    def latest_run(self) -> ControlRunRecord | None:
        """Return the most recently started run, or ``None`` when no run exists."""

        if not self.path.exists():
            return None
        with closing(self._connection()) as conn:
            with map_sqlite_errors(
                AsyncOrchestratorControlStoreIOError,
                f"failed to read latest async orchestrator run from {self.path}",
            ):
                if not _ensure_schema_current(conn):
                    return None
                row = conn.execute(
                    """
                    SELECT * FROM runs
                    ORDER BY rowid DESC
                    LIMIT 1
                    """
                ).fetchone()
        return None if row is None else _run_from_row(row)

    def enqueue_command(
        self,
        command: ControlCommandName,
        *,
        params: Mapping[str, JsonValue] | None = None,
        run_id: str | None = None,
        command_id: str | None = None,
    ) -> ControlCommandRecord:
        """Insert a pending control command and return its durable record."""

        self.initialize()
        command_id = new_control_command_id() if command_id is None else command_id
        params_json = _dump_json_object({} if params is None else dict(params))
        now = utc_timestamp()
        with closing(self._connection()) as conn:
            with map_sqlite_errors(
                AsyncOrchestratorControlStoreIOError,
                f"failed to enqueue async orchestrator control command in {self.path}",
            ):
                with conn:
                    _ensure_schema_current(conn)
                    conn.execute(
                        """
                        INSERT INTO control_commands (
                          id, run_id, command, params_json, status, created_at
                        ) VALUES (?, ?, ?, ?, 'pending', ?)
                        """,
                        (command_id, run_id, command, params_json, now),
                    )
                    row = _select_command(conn, command_id)
        if row is None:
            raise AsyncOrchestratorControlCommandError(
                f"enqueued async orchestrator control command {command_id} was not found"
            )
        return _command_from_row(row)

    def enqueue_command_for_latest_active_run(
        self,
        command: ControlCommandName,
        *,
        params: Mapping[str, JsonValue] | None = None,
        command_id: str | None = None,
    ) -> tuple[ControlRunRecord, ControlCommandRecord] | None:
        """Enqueue a command for the latest run if it is still non-terminal."""

        if not self.path.exists():
            return None
        command_id = new_control_command_id() if command_id is None else command_id
        params_json = _dump_json_object({} if params is None else dict(params))
        now = utc_timestamp()
        with closing(self._connection()) as conn:
            with map_sqlite_errors(
                AsyncOrchestratorControlStoreIOError,
                "failed to enqueue async orchestrator control command for latest "
                f"active run in {self.path}",
            ):
                with begin_immediate(conn):
                    if not _ensure_schema_current(conn):
                        return None
                    run_row = conn.execute(
                        """
                        SELECT * FROM runs
                        ORDER BY rowid DESC
                        LIMIT 1
                        """
                    ).fetchone()
                    if run_row is None or _run_status_is_terminal(run_row):
                        return None
                    run_id = str(run_row["run_id"])
                    conn.execute(
                        """
                        INSERT INTO control_commands (
                          id, run_id, command, params_json, status, created_at
                        ) VALUES (?, ?, ?, ?, 'pending', ?)
                        """,
                        (command_id, run_id, command, params_json, now),
                    )
                    command_row = _select_command(conn, command_id)
        if command_row is None:
            raise AsyncOrchestratorControlCommandError(
                f"enqueued async orchestrator control command {command_id} was not found"
            )
        return _run_from_row(run_row), _command_from_row(command_row)

    def claim_pending_command(self, *, run_id: str) -> ControlCommandRecord | None:
        """Atomically claim the oldest pending command for ``run_id``.

        Commands with ``run_id IS NULL`` target the current active run and are
        bound to ``run_id`` at claim time.
        """

        updated: sqlite3.Row | None = None
        with closing(self._connection()) as conn:
            with map_sqlite_errors(
                AsyncOrchestratorControlStoreIOError,
                f"failed to claim async orchestrator control command from {self.path}",
            ):
                with begin_immediate(conn):
                    if not _ensure_schema_current(conn):
                        return None
                    while True:
                        now = utc_timestamp()
                        row = conn.execute(
                            """
                            SELECT * FROM control_commands
                            WHERE status = 'pending'
                              AND (run_id IS NULL OR run_id = ?)
                            ORDER BY rowid ASC
                            LIMIT 1
                            """,
                            (run_id,),
                        ).fetchone()
                        if row is None:
                            return None
                        command_id = str(row["id"])
                        error = _command_claim_validation_error(row)
                        if error is not None:
                            conn.execute(
                                """
                                UPDATE control_commands
                                SET status = 'failed', run_id = ?, completed_at = ?,
                                    params_json = '{}', result_json = '{}', error = ?
                                WHERE id = ? AND status = 'pending'
                                """,
                                (run_id, now, error, command_id),
                            )
                            continue
                        conn.execute(
                            """
                            UPDATE control_commands
                            SET status = 'claimed', run_id = ?, claimed_at = ?,
                                completed_at = NULL, result_json = NULL, error = NULL
                            WHERE id = ?
                            """,
                            (run_id, now, command_id),
                        )
                        updated = _select_command(conn, command_id)
                        break
        return None if updated is None else _command_from_row(updated)

    def record_command_succeeded(
        self,
        command_id: str,
        *,
        result: Mapping[str, JsonValue] | None = None,
    ) -> ControlCommandRecord:
        """Mark a claimed command as succeeded and update its run cursor."""

        return self._complete_command(
            command_id,
            status="succeeded",
            result={} if result is None else dict(result),
            error=None,
        )

    def record_command_failed(
        self,
        command_id: str,
        *,
        error: str,
        result: Mapping[str, JsonValue] | None = None,
    ) -> ControlCommandRecord:
        """Mark a claimed command as failed and store its error text."""

        return self._complete_command(
            command_id,
            status="failed",
            result={} if result is None else dict(result),
            error=error,
        )

    def get_command(self, command_id: str) -> ControlCommandRecord | None:
        """Return one command row, or ``None`` when it is absent."""

        if not self.path.exists():
            return None
        with closing(self._connection()) as conn:
            with map_sqlite_errors(
                AsyncOrchestratorControlStoreIOError,
                f"failed to read async orchestrator control command {command_id} from {self.path}",
            ):
                if not _ensure_schema_current(conn):
                    return None
                row = _select_command(conn, command_id)
        return None if row is None else _command_from_row(row)

    def cancel_incomplete_commands_for_run(self, *, run_id: str, error: str) -> int:
        """Cancel pending/claimed commands bound to a run during shutdown."""

        if not self.path.exists():
            return 0
        now = utc_timestamp()
        with closing(self._connection()) as conn:
            with map_sqlite_errors(
                AsyncOrchestratorControlStoreIOError,
                "failed to cancel incomplete async orchestrator control commands "
                f"for run {run_id} in {self.path}",
            ):
                with conn:
                    if not _ensure_schema_current(conn):
                        return 0
                    cursor = conn.execute(
                        """
                        UPDATE control_commands
                        SET status = 'cancelled', completed_at = ?,
                            result_json = '{}', error = ?
                        WHERE run_id = ? AND status IN ('pending', 'claimed')
                        """,
                        (now, error, run_id),
                    )
        return cursor.rowcount

    def list_commands(self, *, limit: int = 20) -> tuple[ControlCommandRecord, ...]:
        """Return recent commands ordered from newest to oldest."""

        if limit < 0:
            raise ValueError("command list limit must be non-negative")
        if not self.path.exists() or limit == 0:
            return ()
        with closing(self._connection()) as conn:
            with map_sqlite_errors(
                AsyncOrchestratorControlStoreIOError,
                f"failed to list async orchestrator control commands from {self.path}",
            ):
                if not _ensure_schema_current(conn):
                    return ()
                rows = conn.execute(
                    """
                    SELECT * FROM control_commands
                    ORDER BY rowid DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
        return tuple(_command_from_row(row) for row in rows)

    def list_active_agents(
        self,
        *,
        run_id: str | None = None,
    ) -> tuple[ControlActiveAgentRecord, ...]:
        """Return the last heartbeat's active-agent snapshot."""

        if not self.path.exists():
            return ()
        with closing(self._read_only_connection()) as conn:
            with map_sqlite_errors(
                AsyncOrchestratorControlStoreIOError,
                f"failed to list async orchestrator active agents from {self.path}",
            ):
                if not _ensure_schema_current(conn):
                    return ()
                if run_id is None:
                    latest = conn.execute(
                        """
                        SELECT run_id FROM runs
                        ORDER BY rowid DESC
                        LIMIT 1
                        """
                    ).fetchone()
                    if latest is None:
                        return ()
                    run_id = str(latest["run_id"])
                rows = conn.execute(
                    """
                    SELECT * FROM active_agents
                    WHERE run_id = ?
                    ORDER BY CASE role WHEN 'worker' THEN 0 ELSE 1 END, worktree_id ASC
                    """,
                    (run_id,),
                ).fetchall()
        return tuple(_active_agent_from_row(row) for row in rows)

    def initialize_state(self) -> None:
        """Initialize the durable async-orchestrator state tables."""

        self.initialize()

    def state_exists(self) -> bool:
        """Return whether the unified database has initialized state tables."""

        if not self.path.exists():
            return False
        with closing(self._read_only_connection()) as conn:
            with map_sqlite_errors(
                AsyncOrchestratorControlStoreIOError,
                f"failed to inspect async orchestrator state in {self.path}",
            ):
                if not _ensure_schema_current(conn):
                    return False
                return _select_async_meta(conn) is not None

    def allocate_worktree(
        self,
        *,
        task_id: str | None,
        path: PathInput,
        head: str,
        worktree_id: str | None = None,
    ) -> str:
        """Allocate and insert a pending worktree row, returning its ID.

        When ``worktree_id`` is provided, that exact ID is inserted and the
        metadata sequence is advanced past it. This lets callers choose an ID
        that is also safe for existing on-disk worktree paths before git side
        effects run.
        """

        self.initialize()
        if task_id is not None and not task_id.strip():
            raise ValueError("task ID must not be blank")
        if not head.strip():
            raise ValueError("worktree head must not be blank")
        worktree_path = _to_path(path, field_name="path")
        with closing(self._connection()) as conn:
            with map_sqlite_errors(
                AsyncOrchestratorControlStoreIOError,
                f"failed to allocate async orchestrator worktree in {self.path}",
            ):
                with begin_immediate(conn):
                    _ensure_schema_current(conn)
                    next_sequence = _read_next_worktree_sequence(conn)
                    if worktree_id is None:
                        sequence = next_sequence
                        while True:
                            allocated_worktree_id = format_sequence_id(
                                "worktree",
                                sequence,
                            )
                            existing = conn.execute(
                                "SELECT 1 FROM worktrees WHERE worktree_id = ?",
                                (allocated_worktree_id,),
                            ).fetchone()
                            if existing is None:
                                break
                            sequence += 1
                    else:
                        allocated_worktree_id = worktree_id
                        sequence = _worktree_sequence_from_id(worktree_id)
                        existing = conn.execute(
                            "SELECT 1 FROM worktrees WHERE worktree_id = ?",
                            (allocated_worktree_id,),
                        ).fetchone()
                        if existing is not None:
                            raise ValueError(
                                f"worktree ID already exists: {allocated_worktree_id}"
                            )
                    conn.execute(
                        """
                        INSERT INTO worktrees (
                          worktree_id, created_seq, state, task_id, path, head
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            allocated_worktree_id,
                            sequence,
                            WorktreeState.PENDING.value,
                            task_id,
                            str(worktree_path),
                            head,
                        ),
                    )
                    conn.execute(
                        """
                        UPDATE async_meta
                        SET next_worktree_sequence = ?, updated_at = ?
                        WHERE singleton = 1
                        """,
                        (max(next_sequence, sequence + 1), utc_timestamp()),
                    )
        return allocated_worktree_id

    def set_worktree_state(
        self,
        worktree_id: str,
        *,
        expected: WorktreeState,
        new: WorktreeState,
    ) -> bool:
        """Compare-and-swap a worktree state, returning whether it changed."""

        self.initialize()
        with closing(self._connection()) as conn:
            with map_sqlite_errors(
                AsyncOrchestratorControlStoreIOError,
                f"failed to update async orchestrator worktree {worktree_id} in {self.path}",
            ):
                with begin_immediate(conn):
                    _ensure_schema_current(conn)
                    cursor = conn.execute(
                        """
                        UPDATE worktrees
                        SET state = ?
                        WHERE worktree_id = ? AND state = ?
                        """,
                        (new.value, worktree_id, expected.value),
                    )
        return cursor.rowcount == 1

    def reset_running_worktrees(self) -> int:
        """Return worker-running worktrees to pending and return the count."""

        self.initialize()
        with closing(self._connection()) as conn:
            with map_sqlite_errors(
                AsyncOrchestratorControlStoreIOError,
                f"failed to reset async orchestrator running worktrees in {self.path}",
            ):
                with begin_immediate(conn):
                    _ensure_schema_current(conn)
                    cursor = conn.execute(
                        """
                        UPDATE worktrees
                        SET state = ?
                        WHERE state = ?
                        """,
                        (WorktreeState.PENDING.value, WorktreeState.WORKER_RUNNING.value),
                    )
        return cursor.rowcount

    def clear_state(self) -> None:
        """Clear durable worktree/task state while preserving control history."""

        self.initialize()
        with closing(self._connection()) as conn:
            with map_sqlite_errors(
                AsyncOrchestratorControlStoreIOError,
                f"failed to clear async orchestrator state in {self.path}",
            ):
                with begin_immediate(conn):
                    _ensure_schema_current(conn)
                    conn.execute("DELETE FROM worktree_review_verdicts")
                    conn.execute("DELETE FROM worktree_discussion")
                    conn.execute("DELETE FROM worktrees")
                    conn.execute(
                        """
                        UPDATE async_meta
                        SET next_worktree_sequence = 1,
                            task_manager_json = ?,
                            updated_at = ?
                        WHERE singleton = 1
                        """,
                        (TaskManager().model_dump_json(), utc_timestamp()),
                    )

    def mark_agent_session_started(
        self,
        worktree_id: str,
        role: AsyncOrchestratorAgentRole,
    ) -> None:
        """Mark that ``role`` has started an agent session for ``worktree_id``."""

        self.initialize()
        column = _session_started_column(role)
        with closing(self._connection()) as conn:
            with map_sqlite_errors(
                AsyncOrchestratorControlStoreIOError,
                "failed to mark async orchestrator agent session started "
                f"for {worktree_id} in {self.path}",
            ):
                with begin_immediate(conn):
                    _ensure_schema_current(conn)
                    if column is None:
                        _require_worktree_exists(conn, worktree_id)
                        return
                    cursor = conn.execute(
                        f"UPDATE worktrees SET {column} = 1 WHERE worktree_id = ?",
                        (worktree_id,),
                    )
                    _raise_if_missing_worktree(cursor.rowcount, worktree_id)

    def set_agent_session_usage(
        self,
        worktree_id: str,
        role: AsyncOrchestratorAgentRole,
        usage: Usage,
    ) -> None:
        """Store ``role``'s latest per-worktree usage snapshot."""

        self.initialize()
        column = _session_usage_column(role)
        with closing(self._connection()) as conn:
            with map_sqlite_errors(
                AsyncOrchestratorControlStoreIOError,
                "failed to store async orchestrator agent session usage "
                f"for {worktree_id} in {self.path}",
            ):
                with begin_immediate(conn):
                    _ensure_schema_current(conn)
                    if column is None:
                        _require_worktree_exists(conn, worktree_id)
                        return
                    cursor = conn.execute(
                        f"UPDATE worktrees SET {column} = ? WHERE worktree_id = ?",
                        (usage.model_dump_json(), worktree_id),
                    )
                    _raise_if_missing_worktree(cursor.rowcount, worktree_id)

    def set_agent_session_usage_if_missing_and_inactive(
        self,
        worktree_id: str,
        role: AsyncOrchestratorAgentRole,
        usage: Usage,
        *,
        expected_state: WorktreeState,
    ) -> bool:
        """Store usage only if no snapshot exists and the worktree is inactive."""

        self.initialize()
        column = _session_usage_column(role)
        if column is None:
            return False
        if _role_active_state(role) is expected_state:
            return False
        with closing(self._connection()) as conn:
            with map_sqlite_errors(
                AsyncOrchestratorControlStoreIOError,
                "failed to conditionally store async orchestrator agent session usage "
                f"for {worktree_id} in {self.path}",
            ):
                with begin_immediate(conn):
                    _ensure_schema_current(conn)
                    cursor = conn.execute(
                        f"""
                        UPDATE worktrees
                        SET {column} = ?
                        WHERE worktree_id = ?
                          AND state = ?
                          AND {column} IS NULL
                        """,
                        (usage.model_dump_json(), worktree_id, expected_state.value),
                    )
        return cursor.rowcount == 1

    def append_discussion(
        self,
        worktree_id: str,
        *,
        role: AsyncOrchestratorAgentRole,
        message: str,
    ) -> None:
        """Append one discussion message for ``worktree_id``."""

        self.initialize()
        discussion = AsyncOrchestratorDiscussionMessage(role=role, message=message)
        with closing(self._connection()) as conn:
            with map_sqlite_errors(
                AsyncOrchestratorControlStoreIOError,
                f"failed to append async orchestrator discussion for {worktree_id} in {self.path}",
            ):
                with begin_immediate(conn):
                    _ensure_schema_current(conn)
                    seq = _next_child_sequence(conn, "worktree_discussion", worktree_id)
                    conn.execute(
                        """
                        INSERT INTO worktree_discussion (worktree_id, seq, role, message)
                        VALUES (?, ?, ?, ?)
                        """,
                        (
                            worktree_id,
                            seq,
                            discussion.role.value,
                            discussion.message,
                        ),
                    )

    def append_review_verdict(
        self,
        worktree_id: str,
        verdict: ReviewVerdictOutput,
    ) -> None:
        """Append one structured review verdict for ``worktree_id``."""

        self.initialize()
        with closing(self._connection()) as conn:
            with map_sqlite_errors(
                AsyncOrchestratorControlStoreIOError,
                "failed to append async orchestrator review verdict "
                f"for {worktree_id} in {self.path}",
            ):
                with begin_immediate(conn):
                    _ensure_schema_current(conn)
                    seq = _next_child_sequence(
                        conn,
                        "worktree_review_verdicts",
                        worktree_id,
                    )
                    conn.execute(
                        """
                        INSERT INTO worktree_review_verdicts (
                          worktree_id, seq, verdict_json
                        ) VALUES (?, ?, ?)
                        """,
                        (worktree_id, seq, verdict.model_dump_json()),
                    )

    def record_worktree_transition(
        self,
        worktree_id: str,
        *,
        expected: WorktreeState,
        new: WorktreeState,
        discussion_messages: Iterable[tuple[AsyncOrchestratorAgentRole, str]] = (),
        review_verdict: ReviewVerdictOutput | None = None,
    ) -> AsyncOrchestratorWorktree | None:
        """Atomically append transition artifacts and CAS a worktree state."""

        self.initialize()
        with closing(self._connection()) as conn:
            with map_sqlite_errors(
                AsyncOrchestratorControlStoreIOError,
                f"failed to transition async orchestrator worktree {worktree_id} in {self.path}",
            ):
                with begin_immediate(conn):
                    _ensure_schema_current(conn)
                    cursor = conn.execute(
                        """
                        UPDATE worktrees
                        SET state = ?
                        WHERE worktree_id = ? AND state = ?
                        """,
                        (new.value, worktree_id, expected.value),
                    )
                    if cursor.rowcount != 1:
                        return None
                    for role, message in discussion_messages:
                        discussion = AsyncOrchestratorDiscussionMessage(
                            role=role,
                            message=message,
                        )
                        seq = _next_child_sequence(
                            conn,
                            "worktree_discussion",
                            worktree_id,
                        )
                        conn.execute(
                            """
                            INSERT INTO worktree_discussion (
                              worktree_id, seq, role, message
                            ) VALUES (?, ?, ?, ?)
                            """,
                            (
                                worktree_id,
                                seq,
                                discussion.role.value,
                                discussion.message,
                            ),
                        )
                    if review_verdict is not None:
                        seq = _next_child_sequence(
                            conn,
                            "worktree_review_verdicts",
                            worktree_id,
                        )
                        conn.execute(
                            """
                            INSERT INTO worktree_review_verdicts (
                              worktree_id, seq, verdict_json
                            ) VALUES (?, ?, ?)
                            """,
                            (worktree_id, seq, review_verdict.model_dump_json()),
                        )
                    row = _select_worktree(conn, worktree_id)
                    if row is None:
                        raise AsyncOrchestratorControlStoreIOError(
                            f"transitioned async orchestrator worktree {worktree_id} was not found"
                        )
                    return _worktree_from_row(conn, row)

    def replace_task_snapshot(self, task_manager: TaskManager) -> None:
        """Replace the task-manager JSON snapshot and detach orphan worktrees."""

        self.initialize()
        task_ids = task_manager.task_ids
        with closing(self._connection()) as conn:
            with map_sqlite_errors(
                AsyncOrchestratorControlStoreIOError,
                f"failed to replace async orchestrator task snapshot in {self.path}",
            ):
                with begin_immediate(conn):
                    _ensure_schema_current(conn)
                    conn.execute(
                        """
                        UPDATE async_meta
                        SET task_manager_json = ?, updated_at = ?
                        WHERE singleton = 1
                        """,
                        (task_manager.model_dump_json(), utc_timestamp()),
                    )
                    if task_ids:
                        placeholders = ", ".join("?" for _ in task_ids)
                        conn.execute(
                            """
                            UPDATE worktrees
                            SET task_id = NULL
                            WHERE task_id IS NOT NULL
                              AND task_id NOT IN (PLACEHOLDERS)
                            """.replace("PLACEHOLDERS", placeholders),
                            task_ids,
                        )
                    else:
                        conn.execute(
                            "UPDATE worktrees SET task_id = NULL WHERE task_id IS NOT NULL"
                        )

    def get_worktree(self, worktree_id: str) -> AsyncOrchestratorWorktree | None:
        """Return one reconstructed worktree, or ``None`` when it is absent."""

        if not self.path.exists():
            return None
        with closing(self._read_only_connection()) as conn:
            with map_sqlite_errors(
                AsyncOrchestratorControlStoreIOError,
                f"failed to read async orchestrator worktree {worktree_id} from {self.path}",
            ):
                if not _ensure_schema_current(conn):
                    return None
                row = _select_worktree(conn, worktree_id)
                return None if row is None else _worktree_from_row(conn, row)

    def list_worktrees(self) -> tuple[AsyncOrchestratorWorktree, ...]:
        """Return all reconstructed worktrees in insertion order."""

        if not self.path.exists():
            return ()
        with closing(self._read_only_connection()) as conn:
            with map_sqlite_errors(
                AsyncOrchestratorControlStoreIOError,
                f"failed to list async orchestrator worktrees from {self.path}",
            ):
                if not _ensure_schema_current(conn):
                    return ()
                rows = conn.execute(
                    """
                    SELECT * FROM worktrees
                    ORDER BY created_seq ASC, worktree_id ASC
                    """
                ).fetchall()
                return tuple(_worktree_from_row(conn, row) for row in rows)

    def worktrees_for_task(
        self,
        task_id: str,
    ) -> tuple[AsyncOrchestratorWorktree, ...]:
        """Return reconstructed worktrees attached to ``task_id``."""

        if not self.path.exists():
            return ()
        with closing(self._read_only_connection()) as conn:
            with map_sqlite_errors(
                AsyncOrchestratorControlStoreIOError,
                "failed to list async orchestrator worktrees for task "
                f"{task_id} from {self.path}",
            ):
                if not _ensure_schema_current(conn):
                    return ()
                rows = conn.execute(
                    """
                    SELECT * FROM worktrees
                    WHERE task_id = ?
                    ORDER BY created_seq ASC, worktree_id ASC
                    """,
                    (task_id,),
                ).fetchall()
                return tuple(_worktree_from_row(conn, row) for row in rows)

    def non_closed_worktrees_for_task(
        self,
        task_id: str,
    ) -> tuple[AsyncOrchestratorWorktree, ...]:
        """Return task worktrees whose state is not closed."""

        if not self.path.exists():
            return ()
        with closing(self._read_only_connection()) as conn:
            with map_sqlite_errors(
                AsyncOrchestratorControlStoreIOError,
                "failed to list non-closed async orchestrator worktrees for task "
                f"{task_id} from {self.path}",
            ):
                if not _ensure_schema_current(conn):
                    return ()
                rows = conn.execute(
                    """
                    SELECT * FROM worktrees
                    WHERE task_id = ? AND state != ?
                    ORDER BY created_seq ASC, worktree_id ASC
                    """,
                    (task_id, WorktreeState.CLOSED.value),
                ).fetchall()
                return tuple(_worktree_from_row(conn, row) for row in rows)

    def worktree_control_summaries(
        self,
        worktree_ids: Sequence[str],
    ) -> dict[str, tuple[str | None, str]]:
        """Return lightweight ``task_id``/state summaries keyed by worktree ID."""

        ordered_ids = tuple(dict.fromkeys(worktree_ids))
        if not ordered_ids or not self.path.exists():
            return {}
        with closing(self._read_only_connection()) as conn:
            with map_sqlite_errors(
                AsyncOrchestratorControlStoreIOError,
                f"failed to read async orchestrator worktree summaries from {self.path}",
            ):
                if not _ensure_schema_current(conn):
                    return {}
                placeholders = ", ".join("?" for _ in ordered_ids)
                rows = conn.execute(
                    f"""
                    SELECT worktree_id, task_id, state
                    FROM worktrees
                    WHERE worktree_id IN ({placeholders})
                    """,
                    ordered_ids,
                ).fetchall()
        return {
            str(row["worktree_id"]): (_optional_str(row["task_id"]), str(row["state"]))
            for row in rows
        }

    def next_worktree_sequence(self) -> int:
        """Return the next worktree sequence number from state metadata."""

        if not self.path.exists():
            return 1
        with closing(self._read_only_connection()) as conn:
            with map_sqlite_errors(
                AsyncOrchestratorControlStoreIOError,
                f"failed to read async orchestrator worktree sequence from {self.path}",
            ):
                if not _ensure_schema_current(conn):
                    return 1
                return _read_next_worktree_sequence(conn)

    def load_task_snapshot(self) -> TaskManager:
        """Return the stored task-manager snapshot."""

        if not self.path.exists():
            return TaskManager()
        with closing(self._read_only_connection()) as conn:
            with map_sqlite_errors(
                AsyncOrchestratorControlStoreIOError,
                f"failed to load async orchestrator task snapshot from {self.path}",
            ):
                if not _ensure_schema_current(conn):
                    return TaskManager()
                row = _select_async_meta(conn)
                if row is None:
                    raise AsyncOrchestratorControlSchemaError(
                        f"async orchestrator state metadata is missing in {self.path}"
                    )
                return TaskManager.model_validate_json(str(row["task_manager_json"]))

    def aggregate_usage(self, root: PathInput) -> Usage:
        """Aggregate usage from stored worktree snapshots and live session logs."""

        return aggregate_agent_session_usage(
            _to_path(root, field_name="root"),
            self.list_worktrees(),
        )

    def _update_run(
        self,
        *,
        run_id: str,
        status: ControlRunStatus,
        status_reason: str | None,
        worker_limit: int | None,
        reviewer_limit: int | None,
        paused: bool,
        drain_requested: bool,
        active_agents: Sequence[ControlActiveAgentSnapshot],
        preserve_terminal: bool,
    ) -> ControlRunRecord:
        now = utc_timestamp()
        terminal_guard = ""
        if preserve_terminal:
            terminal_guard = (
                f" AND status NOT IN {_sql_string_list(_TERMINAL_RUN_STATUSES)}"
            )
        with closing(self._connection()) as conn:
            with map_sqlite_errors(
                AsyncOrchestratorControlStoreIOError,
                f"failed to update async orchestrator run {run_id} in {self.path}",
            ):
                with conn:
                    _ensure_schema_current(conn)
                    cursor = conn.execute(
                        f"""
                        UPDATE runs
                        SET heartbeat_at = ?, status = ?, status_reason = ?,
                            worker_limit = ?, reviewer_limit = ?, paused = ?,
                            drain_requested = ?
                        WHERE run_id = ?{terminal_guard}
                        """,
                        (
                            now,
                            status,
                            status_reason,
                            worker_limit,
                            reviewer_limit,
                            _bool_to_int(paused),
                            _bool_to_int(drain_requested),
                            run_id,
                        ),
                    )
                    row = _select_run(conn, run_id)
                    if cursor.rowcount == 1:
                        _replace_active_agents(conn, run_id, active_agents, recorded_at=now)
                    if cursor.rowcount != 1:
                        if row is None:
                            raise AsyncOrchestratorControlCommandError(
                                f"async orchestrator run {run_id} is not registered"
                            )
                        if not preserve_terminal or not _run_status_is_terminal(row):
                            raise AsyncOrchestratorControlCommandError(
                                f"failed to update async orchestrator run {run_id}"
                            )
        if row is None:
            raise AsyncOrchestratorControlStoreIOError(
                f"updated async orchestrator run {run_id} was not found"
            )
        return _run_from_row(row)

    def _complete_command(
        self,
        command_id: str,
        *,
        status: Literal["succeeded", "failed"],
        result: JsonObject,
        error: str | None,
    ) -> ControlCommandRecord:
        now = utc_timestamp()
        result_json = _dump_json_object(result)
        with closing(self._connection()) as conn:
            with map_sqlite_errors(
                AsyncOrchestratorControlStoreIOError,
                "failed to complete async orchestrator control command "
                f"{command_id} in {self.path}",
            ):
                with begin_immediate(conn):
                    _ensure_schema_current(conn)
                    cursor = conn.execute(
                        """
                        UPDATE control_commands
                        SET status = ?, completed_at = ?, result_json = ?, error = ?
                        WHERE id = ? AND status = 'claimed'
                        """,
                        (status, now, result_json, error, command_id),
                    )
                    if cursor.rowcount != 1:
                        raise AsyncOrchestratorControlCommandError(
                            f"async orchestrator control command {command_id} "
                            "does not exist or is not claimed"
                        )
                    row = _select_command(conn, command_id)
                    if status == "succeeded" and row is not None and row["run_id"] is not None:
                        conn.execute(
                            "UPDATE runs SET applied_command_id = ? WHERE run_id = ?",
                            (command_id, row["run_id"]),
                        )
        if row is None:
            raise AsyncOrchestratorControlCommandError(
                f"completed async orchestrator control command {command_id} was not found"
            )
        return _command_from_row(row)

    def _schema_is_current(self) -> bool:
        with closing(self._read_only_connection()) as conn:
            with map_sqlite_errors(
                AsyncOrchestratorControlStoreIOError,
                f"failed to read async orchestrator control schema from {self.path}",
            ):
                return _ensure_schema_current(conn)

    def _connection(self) -> sqlite3.Connection:
        with map_sqlite_errors(
            AsyncOrchestratorControlStoreIOError,
            f"failed to open async orchestrator control database {self.path}",
        ):
            return connect_read_write(self.path)

    def _read_only_connection(self) -> sqlite3.Connection:
        with map_sqlite_errors(
            AsyncOrchestratorControlStoreIOError,
            f"failed to open async orchestrator control database {self.path}",
        ):
            return connect_read_only(self.path)


_SCHEMA_SQL = f"""
CREATE TABLE IF NOT EXISTS schema (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY,
  pid INTEGER NOT NULL,
  started_at TEXT NOT NULL,
  heartbeat_at TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN {_sql_string_list(_RUN_STATUSES)}),
  status_reason TEXT,
  applied_command_id TEXT,
  worker_limit INTEGER,
  reviewer_limit INTEGER,
  paused INTEGER NOT NULL DEFAULT 0 CHECK (paused IN (0, 1)),
  drain_requested INTEGER NOT NULL DEFAULT 0 CHECK (drain_requested IN (0, 1))
);

CREATE TABLE IF NOT EXISTS control_commands (
  id TEXT PRIMARY KEY,
  run_id TEXT REFERENCES runs(run_id),
  command TEXT NOT NULL CHECK (command IN {_sql_string_list(_COMMAND_NAMES)}),
  params_json TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN {_sql_string_list(_COMMAND_STATUSES)}),
  created_at TEXT NOT NULL,
  claimed_at TEXT,
  completed_at TEXT,
  result_json TEXT,
  error TEXT
);

CREATE INDEX IF NOT EXISTS control_commands_pending
  ON control_commands(status, created_at);

CREATE TABLE IF NOT EXISTS active_agents (
  run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
  role TEXT NOT NULL CHECK (role IN {_sql_string_list(_ACTIVE_AGENT_ROLES)}),
  worktree_id TEXT NOT NULL,
  task_id TEXT,
  worktree_state TEXT,
  recorded_at TEXT NOT NULL,
  PRIMARY KEY (run_id, role, worktree_id)
);

CREATE INDEX IF NOT EXISTS active_agents_run_id
  ON active_agents(run_id);

CREATE TABLE IF NOT EXISTS async_meta (
  singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
  next_worktree_sequence INTEGER NOT NULL DEFAULT 1,
  task_manager_json TEXT NOT NULL DEFAULT '{{"tasks":[]}}',
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS worktrees (
  worktree_id TEXT PRIMARY KEY,
  created_seq INTEGER NOT NULL,
  state TEXT NOT NULL CHECK (state IN {_sql_string_list(_WORKTREE_STATES)}),
  task_id TEXT,
  path TEXT NOT NULL,
  head TEXT NOT NULL,
  worker_session_started INTEGER NOT NULL DEFAULT 0 CHECK (worker_session_started IN (0, 1)),
  reviewer_session_started INTEGER NOT NULL DEFAULT 0 CHECK (reviewer_session_started IN (0, 1)),
  worker_session_usage_json TEXT,
  reviewer_session_usage_json TEXT
);

CREATE INDEX IF NOT EXISTS worktrees_by_task
  ON worktrees(task_id);
CREATE INDEX IF NOT EXISTS worktrees_by_state
  ON worktrees(state);

CREATE TABLE IF NOT EXISTS worktree_discussion (
  worktree_id TEXT NOT NULL REFERENCES worktrees(worktree_id) ON DELETE CASCADE,
  seq INTEGER NOT NULL,
  role TEXT NOT NULL CHECK (role IN {_sql_string_list(_DISCUSSION_ROLES)}),
  message TEXT NOT NULL,
  PRIMARY KEY (worktree_id, seq)
);

CREATE TABLE IF NOT EXISTS worktree_review_verdicts (
  worktree_id TEXT NOT NULL REFERENCES worktrees(worktree_id) ON DELETE CASCADE,
  seq INTEGER NOT NULL,
  verdict_json TEXT NOT NULL,
  PRIMARY KEY (worktree_id, seq)
);
"""


def new_control_run_id() -> str:
    """Return a unique durable run identifier."""

    return f"run_{uuid.uuid4().hex}"


def new_control_command_id() -> str:
    """Return a unique durable control-command identifier."""

    return f"cmd_{uuid.uuid4().hex}"


def _to_path(value: PathInput, *, field_name: str) -> Path:
    if isinstance(value, str) and not value:
        raise ValueError(f"{field_name} path must be non-empty")
    path = Path(value)
    if "\x00" in str(path):
        raise ValueError(f"{field_name} path must not contain NUL")
    return path


def _bool_to_int(value: bool) -> int:
    return 1 if value else 0


def _select_run(conn: sqlite3.Connection, run_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()


def _select_command(conn: sqlite3.Connection, command_id: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM control_commands WHERE id = ?", (command_id,)
    ).fetchone()


def _select_worktree(conn: sqlite3.Connection, worktree_id: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM worktrees WHERE worktree_id = ?",
        (worktree_id,),
    ).fetchone()


def _select_async_meta(conn: sqlite3.Connection) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM async_meta WHERE singleton = 1").fetchone()


def _ensure_async_meta_row(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        INSERT OR IGNORE INTO async_meta (
          singleton, next_worktree_sequence, task_manager_json, updated_at
        ) VALUES (1, 1, ?, ?)
        """,
        (TaskManager().model_dump_json(), utc_timestamp()),
    )


def _read_next_worktree_sequence(conn: sqlite3.Connection) -> int:
    row = _select_async_meta(conn)
    if row is None:
        raise AsyncOrchestratorControlSchemaError(
            "async orchestrator state metadata is missing"
        )
    return int(row["next_worktree_sequence"])


def _worktree_sequence_from_id(worktree_id: str) -> int:
    prefix = "worktree_"
    if not worktree_id.startswith(prefix):
        raise ValueError(f"invalid worktree ID: {worktree_id}")
    sequence_text = worktree_id.removeprefix(prefix)
    if not sequence_text.isdigit():
        raise ValueError(f"invalid worktree ID: {worktree_id}")
    sequence = int(sequence_text)
    if sequence < 1:
        raise ValueError(f"invalid worktree ID: {worktree_id}")
    return sequence


def _next_child_sequence(
    conn: sqlite3.Connection,
    table: Literal["worktree_discussion", "worktree_review_verdicts"],
    worktree_id: str,
) -> int:
    _require_worktree_exists(conn, worktree_id)
    row = conn.execute(
        f"SELECT COALESCE(MAX(seq), 0) + 1 AS next_seq FROM {table} WHERE worktree_id = ?",
        (worktree_id,),
    ).fetchone()
    if row is None:
        raise AsyncOrchestratorControlStoreIOError(
            f"failed to allocate child sequence for worktree {worktree_id}"
        )
    return int(row["next_seq"])


def _require_worktree_exists(conn: sqlite3.Connection, worktree_id: str) -> None:
    if _select_worktree(conn, worktree_id) is None:
        raise ValueError(f"unknown worktree ID: {worktree_id}")


def _raise_if_missing_worktree(rowcount: int, worktree_id: str) -> None:
    if rowcount != 1:
        raise ValueError(f"unknown worktree ID: {worktree_id}")


def _session_started_column(role: AsyncOrchestratorAgentRole) -> str | None:
    if role is AsyncOrchestratorAgentRole.WORKER:
        return "worker_session_started"
    if role is AsyncOrchestratorAgentRole.REVIEWER:
        return "reviewer_session_started"
    if role is AsyncOrchestratorAgentRole.ORCHESTRATOR:
        return None
    raise ValueError(f"unknown async orchestrator agent role: {role}")


def _session_usage_column(role: AsyncOrchestratorAgentRole) -> str | None:
    if role is AsyncOrchestratorAgentRole.WORKER:
        return "worker_session_usage_json"
    if role is AsyncOrchestratorAgentRole.REVIEWER:
        return "reviewer_session_usage_json"
    if role is AsyncOrchestratorAgentRole.ORCHESTRATOR:
        return None
    raise ValueError(f"unknown async orchestrator agent role: {role}")


def _role_active_state(role: AsyncOrchestratorAgentRole) -> WorktreeState | None:
    if role is AsyncOrchestratorAgentRole.WORKER:
        return WorktreeState.WORKER_RUNNING
    if role is AsyncOrchestratorAgentRole.REVIEWER:
        return WorktreeState.REVIEW
    if role is AsyncOrchestratorAgentRole.ORCHESTRATOR:
        return None
    raise ValueError(f"unknown async orchestrator agent role: {role}")


def _ensure_schema_current(conn: sqlite3.Connection) -> bool:
    """Return whether the DB is initialized, raising for unsupported schemas."""

    version = _read_schema_version(conn)
    if version == ASYNC_ORCHESTRATOR_SCHEMA_VERSION:
        return True
    if version is None:
        if _database_has_user_tables(conn):
            raise AsyncOrchestratorControlSchemaError(
                "missing async orchestrator schema version"
            )
        return False
    raise AsyncOrchestratorControlSchemaError(
        f"unsupported async orchestrator schema version: {version}; supported "
        f"version is {ASYNC_ORCHESTRATOR_SCHEMA_VERSION}"
    )


def _database_has_user_tables(conn: sqlite3.Connection) -> bool:
    return (
        conn.execute(
            """
            SELECT 1 FROM sqlite_master
            WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
            LIMIT 1
            """
        ).fetchone()
        is not None
    )


def _read_schema_version(conn: sqlite3.Connection) -> int | None:
    schema_table = conn.execute(
        """
        SELECT 1 FROM sqlite_master
        WHERE type = 'table' AND name = 'schema'
        """
    ).fetchone()
    if schema_table is None:
        return None
    row = conn.execute("SELECT version FROM schema WHERE id = 1").fetchone()
    return None if row is None else int(row["version"])


def _replace_active_agents(
    conn: sqlite3.Connection,
    run_id: str,
    active_agents: Sequence[ControlActiveAgentSnapshot],
    *,
    recorded_at: str,
) -> None:
    conn.execute("DELETE FROM active_agents WHERE run_id = ?", (run_id,))
    conn.executemany(
        """
        INSERT INTO active_agents (
          run_id, role, worktree_id, task_id, worktree_state, recorded_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            (
                run_id,
                agent.role,
                agent.worktree_id,
                agent.task_id,
                agent.worktree_state,
                recorded_at,
            )
            for agent in active_agents
        ),
    )


def _run_status_is_terminal(row: sqlite3.Row) -> bool:
    return str(row["status"]) in _TERMINAL_RUN_STATUSES


def _command_claim_validation_error(row: sqlite3.Row) -> str | None:
    try:
        _load_json_object(str(row["params_json"]), field_name="params_json")
    except AsyncOrchestratorControlCommandError as exc:
        return str(exc)
    return None


def _run_from_row(row: sqlite3.Row) -> ControlRunRecord:
    return ControlRunRecord(
        run_id=str(row["run_id"]),
        pid=int(row["pid"]),
        started_at=str(row["started_at"]),
        heartbeat_at=str(row["heartbeat_at"]),
        status=cast(ControlRunStatus, str(row["status"])),
        status_reason=_optional_str(row["status_reason"]),
        applied_command_id=_optional_str(row["applied_command_id"]),
        worker_limit=_optional_int(row["worker_limit"]),
        reviewer_limit=_optional_int(row["reviewer_limit"]),
        paused=bool(row["paused"]),
        drain_requested=bool(row["drain_requested"]),
    )


def _command_from_row(row: sqlite3.Row) -> ControlCommandRecord:
    return ControlCommandRecord(
        id=str(row["id"]),
        run_id=_optional_str(row["run_id"]),
        command=cast(ControlCommandName, str(row["command"])),
        params=_load_json_object(str(row["params_json"]), field_name="params_json"),
        status=cast(ControlCommandStatus, str(row["status"])),
        created_at=str(row["created_at"]),
        claimed_at=_optional_str(row["claimed_at"]),
        completed_at=_optional_str(row["completed_at"]),
        result=(
            None
            if row["result_json"] is None
            else _load_json_object(str(row["result_json"]), field_name="result_json")
        ),
        error=_optional_str(row["error"]),
    )


def _active_agent_from_row(row: sqlite3.Row) -> ControlActiveAgentRecord:
    return ControlActiveAgentRecord(
        run_id=str(row["run_id"]),
        role=cast(ControlActiveAgentRole, str(row["role"])),
        worktree_id=str(row["worktree_id"]),
        task_id=_optional_str(row["task_id"]),
        worktree_state=_optional_str(row["worktree_state"]),
        recorded_at=str(row["recorded_at"]),
    )


def _worktree_from_row(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
) -> AsyncOrchestratorWorktree:
    worktree_id = str(row["worktree_id"])
    discussion_rows = conn.execute(
        """
        SELECT * FROM worktree_discussion
        WHERE worktree_id = ?
        ORDER BY seq ASC
        """,
        (worktree_id,),
    ).fetchall()
    verdict_rows = conn.execute(
        """
        SELECT * FROM worktree_review_verdicts
        WHERE worktree_id = ?
        ORDER BY seq ASC
        """,
        (worktree_id,),
    ).fetchall()
    return AsyncOrchestratorWorktree(
        worktree_id=worktree_id,
        path=Path(str(row["path"])),
        head=str(row["head"]),
        task_id=_optional_str(row["task_id"]),
        state=WorktreeState(str(row["state"])),
        discussion=tuple(_discussion_from_row(child) for child in discussion_rows),
        review_verdicts=tuple(_review_verdict_from_row(child) for child in verdict_rows),
        worker_session_started=bool(row["worker_session_started"]),
        reviewer_session_started=bool(row["reviewer_session_started"]),
        worker_session_usage=_optional_usage(row["worker_session_usage_json"]),
        reviewer_session_usage=_optional_usage(row["reviewer_session_usage_json"]),
    )


def _discussion_from_row(row: sqlite3.Row) -> AsyncOrchestratorDiscussionMessage:
    return AsyncOrchestratorDiscussionMessage(
        role=AsyncOrchestratorAgentRole(str(row["role"])),
        message=str(row["message"]),
    )


def _review_verdict_from_row(row: sqlite3.Row) -> ReviewVerdictOutput:
    return ReviewVerdictOutput.model_validate_json(str(row["verdict_json"]))


def _optional_usage(value: object) -> Usage | None:
    if value is None:
        return None
    return Usage.model_validate_json(str(value))


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value)
    raise TypeError(f"expected integer-compatible SQLite value, got {type(value).__name__}")


def _dump_json_object(value: Mapping[str, JsonValue]) -> str:
    try:
        return json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except ValueError as exc:
        raise AsyncOrchestratorControlCommandError(
            "async orchestrator control command JSON values must be finite"
        ) from exc


def _load_json_object(text: str, *, field_name: str) -> JsonObject:
    try:
        value: object = json.loads(
            text,
            parse_constant=_reject_json_constant,
            parse_float=_parse_finite_json_float,
        )
    except json.JSONDecodeError as exc:
        raise AsyncOrchestratorControlCommandError(
            f"async orchestrator control command {field_name} is invalid JSON"
        ) from exc
    except ValueError as exc:
        raise AsyncOrchestratorControlCommandError(
            f"async orchestrator control command {field_name} has non-finite JSON values"
        ) from exc
    if not isinstance(value, dict):
        raise AsyncOrchestratorControlCommandError(
            f"async orchestrator control command {field_name} must be a JSON object"
        )
    json_object = cast(dict[object, object], value)
    _reject_non_finite_json_values(json_object, field_name=field_name)
    return cast(JsonObject, json_object)


def _parse_finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite JSON value is not allowed: {value}")
    return parsed


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON value is not allowed: {value}")


def _reject_non_finite_json_values(value: object, *, field_name: str) -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise AsyncOrchestratorControlCommandError(
                f"async orchestrator control command {field_name} "
                "has non-finite JSON values"
            )
        return
    if isinstance(value, dict):
        for child in cast(dict[object, object], value).values():
            _reject_non_finite_json_values(child, field_name=field_name)
        return
    if isinstance(value, list):
        for child in cast(list[object], value):
            _reject_non_finite_json_values(child, field_name=field_name)


__all__ = (
    "ASYNC_ORCHESTRATOR_DB_FILENAME",
    "ASYNC_ORCHESTRATOR_SCHEMA_VERSION",
    "AsyncOrchestratorControlCommandError",
    "AsyncOrchestratorControlSchemaError",
    "AsyncOrchestratorControlStoreError",
    "AsyncOrchestratorControlStoreIOError",
    "ControlActiveAgentRecord",
    "ControlActiveAgentRole",
    "ControlActiveAgentSnapshot",
    "ControlCommandName",
    "ControlCommandRecord",
    "ControlCommandStatus",
    "ControlRunRecord",
    "ControlRunStatus",
    "SQLiteAsyncOrchestratorStore",
    "new_control_command_id",
    "new_control_run_id",
)
