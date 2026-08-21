"""Parent async orchestrator type."""

from __future__ import annotations

import asyncio
import errno
import json
import logging
import os
import shutil
import signal
import stat
import subprocess
import tempfile
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from fnmatch import fnmatch
from pathlib import Path, PurePosixPath
from re import Pattern, compile
from types import MappingProxyType
from typing import Literal, TypeVar, cast

from pydantic import BaseModel, Field

from tend._common.agent_outputs import ReviewVerdictOutput, WorkerContributionOutput
from tend._common.config_files import ConfigFileError, read_yaml_config_data
from tend._common.errors import FrameworkError
from tend._common.types import JsonObject, StrictModel, format_sequence_id
from tend.llm.usage import Cost, Usage
from tend.orchestrator.agent_runner import (
    oom_score_adj_preexec,
    run_agent_command,
)
from tend.orchestrator.config import (
    AsyncOrchestratorAgentCommandConfig,
    AsyncOrchestratorConfig,
    AsyncOrchestratorValidationCommandConfig,
    AsyncOrchestratorWorktreeSetupCommandConfig,
)
from tend.orchestrator.control_store import (
    AsyncOrchestratorControlStoreIOError,
    ControlActiveAgentRole,
    ControlActiveAgentSnapshot,
    ControlCommandRecord,
    ControlRunStatus,
    SQLiteAsyncOrchestratorStore,
    new_control_run_id,
)
from tend.orchestrator.discussion import (
    write_discussion_log_file,
    write_review_verdict_artifact,
)
from tend.orchestrator.runtime import AsyncOrchestratorRuntime, prune_done_agent_tasks
from tend.orchestrator.state import (
    AsyncOrchestratorAgentRole,
    AsyncOrchestratorWorktree,
    WorktreeState,
)
from tend.orchestrator.task_io import (
    DEFAULT_TASK_FILE_GLOB,
    TASKS_DIRECTORY_NAME,
    load_entrypoint_task_manager,
    task_directory,
)
from tend.orchestrator.task_manager import TaskManager
from tend.orchestrator.task_validation import (
    TaskValidationFailure as _TaskValidationFailure,
)
from tend.orchestrator.task_validation import (
    validate_task_directory,
)
from tend.orchestrator.tasks import Task, TaskStatus, task_priority_rank
from tend.orchestrator.usage import (
    agent_session_is_active,
    format_usage_summary,
    load_agent_session_usage,
    resolve_agent_session_usage,
)
from tend.workspace.mirror import (
    MirrorExistingPathPolicy,
    MirrorReflinkMode,
    WorkspaceMirrorConfig,
    mirror_workspace,
)

# Discovery poll cadence. The reload is now off the entrypoint guard lock (see
# ``_sync_task_manager_once``), so this only bounds discovery *latency* and the
# CPU/disk cost of re-parsing the whole task tree — 1s is ample for runs whose
# units of work take minutes, and avoids re-parsing hundreds of YAMLs ~4x/sec.
_TASK_DISCOVERY_POLL_INTERVAL_SECONDS = 1.0
# Directory name (under ``<root>``) of the long-lived staging worktree used for
# merge validation when ``merge_validation_worktree`` is enabled. Kept out of
# ``<root>/worktrees`` so the closed-worktree pruning never touches it.
_VALIDATION_WORKTREE_DIRNAME = "staging"
# Durable proof that staging's mirror and setup completed. Keep it outside the
# checkout so git reset/clean, workspace mirroring, and tracked paths cannot
# create or restore readiness before provisioning succeeds.
_VALIDATION_WORKTREE_PROVISIONED_SUFFIX = ".provisioned"
# ``<root>/.build-cache`` holds a consistent snapshot of the staging worktree's
# ``.lake/build`` (Lean build artifacts), refreshed after each successful
# validated merge and copied into new task worktrees when ``seed_worktree_build``
# is enabled. Kept directly under ``<root>`` (a sibling of ``worktrees`` and
# ``staging``) so closed-worktree pruning never touches it. The relative build
# subtree mirrored from a worktree root.
_BUILD_CACHE_DIRNAME = ".build-cache"
# Worker discussions are durable store records. Keep a representative path
# sample instead of expanding an adversarial graph's full attribution set.
_MAX_DISCUSSION_OFFENDING_PATHS = 30
_LAKE_BUILD_RELPATH = ".lake/build"
_WORKTREE_NAME_PATTERN: Pattern[str] = compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_ORCHESTRATOR_GIT_METADATA_PATHSPEC = ".tend"
_ORCHESTRATOR_GIT_METADATA_EXCLUDE_PATHSPEC = f":(exclude){_ORCHESTRATOR_GIT_METADATA_PATHSPEC}"
_LOGGER = logging.getLogger(__name__)
# Control heartbeats are for human/operator liveness, not scheduler latency. Keep
# the interval short enough for tests and CLI status freshness without turning
# SQLite into a busy loop.
_CONTROL_HEARTBEAT_INTERVAL_SECONDS = 0.25
_VALIDATION_TERMINATION_GRACE_SECONDS = 5.0
_VALIDATION_KILL_GRACE_SECONDS = 1.0
# Bound both graceful termination and post-escalation settling of an interrupted
# setup command. Popen can itself wedge before publishing a process to signal; in
# that case cancellation abandons the worker after this bound rather than hanging.
_VALIDATION_PROVISION_CANCEL_SETTLE_SECONDS = 5.0
_VALIDATION_OUTPUT_POLL_INTERVAL_SECONDS = 0.05
# ``os.killpg`` on a validation process group is inherently racy against a build
# tool that keeps forking workers: a worker forked concurrently with signal
# delivery misses the group signal (pending signals are not inherited across
# ``fork``) and, once its parent dies, nothing ever signals it again. Observed in
# the 2026-07 CFT run as ``lean`` processes with PPID 1 surviving for days at
# multi-GB RSS after repeated validation kills (#132/#146). After a kill
# completes we therefore sweep ``/proc`` for processes still owned by the killed
# leader's process group/session and terminate them.
_VALIDATION_ORPHAN_REAP_GRACE_SECONDS = 1.0
# A single SIGKILL snapshot loses the scan-vs-fork race, so escalation uses
# bounded scan→kill passes. Failed/unverifiable outcomes and incomplete scans
# may be transient and therefore cannot prove that a stable-looking or empty
# set has converged. A fresh final scan supplies the survivor/incomplete-scan
# warning. Async inter-pass sleeps yield on the normal path; synchronous
# cancellation cleanup also stops starting passes after a wall-clock deadline.
_VALIDATION_ORPHAN_KILL_PASS_LIMIT = 10
_VALIDATION_ORPHAN_KILL_PASS_INTERVAL_SECONDS = 0.05
_VALIDATION_ORPHAN_SYNC_CLEANUP_DEADLINE_SECONDS = 2.0
# ``/proc/<pid>/stat`` states meaning "already dead": zombies (Z) and dead (X/x)
# processes cannot be signalled into anything and vanish only when their reaper
# parent collects them, so they are never kill candidates.
_VALIDATION_ORPHAN_DEAD_PROCESS_STATES = frozenset({"Z", "X", "x"})
_PROC_ROOT = Path("/proc")
# Signals whose exit is classified as infrastructure cancellation. This is a
# heuristic on signal semantics — an exit status alone cannot prove who sent
# the signal — but these typically indicate exogenous termination: an operator
# kill, a systemd/container shutdown, or the kernel OOM killer (SIGKILL, the
# original issue #132 motivation). A validation killed by one of these says
# nothing about batch validity, so the batched merge retries it in place
# instead of bisecting a healthy batch. Any other signal exit (SIGSEGV,
# SIGABRT, SIGQUIT, ...) is treated as a validator crash and stays an ordinary
# validation failure: a deterministic crash is evidence of a real failure, and
# retrying it in place would convert that failure signal into a pass. SIGQUIT
# is deliberately excluded — it is traditionally a core-dump/fatal signal,
# weak evidence of cancellation.
_VALIDATION_CANCELLATION_SIGNAL_NAMES = ("SIGTERM", "SIGINT", "SIGKILL", "SIGHUP")


def _cancellation_signals_available_in(namespace: object) -> frozenset[int]:
    """Resolve the cancellation-signal allowlist against ``namespace``.

    Filtered by availability rather than naming attributes unconditionally:
    Windows' ``signal`` module exposes no SIGKILL/SIGHUP, and referencing them
    at import time would raise ``AttributeError``. Negative returncodes are
    POSIX-only, so the entries missing on Windows could never match there
    anyway.
    """

    return frozenset(
        int(getattr(namespace, name))
        for name in _VALIDATION_CANCELLATION_SIGNAL_NAMES
        if hasattr(namespace, name)
    )


_VALIDATION_CANCELLATION_SIGNALS = _cancellation_signals_available_in(signal)
_T = TypeVar("_T")


class AsyncCostBudgetCurrencyMismatchError(FrameworkError):
    """Accumulated run cost is denominated in a different currency than the budget.

    The cost ceiling can only be enforced when accumulated cost and the configured
    ``budget.currency`` share the same currency. Comparing across currencies would
    silently disable the guard, so we fail closed and raise instead.
    """


class AsyncOrchestratorBudgetStop(StrictModel):
    """Recorded reason and cost when a run stops on its cost ceiling.

    ``breach_accumulated_cost`` is the total at the moment the ceiling was first
    crossed and never changes after that. ``accumulated_cost`` is refreshed on
    every settle-poll (and once more at run exit) so the returned record reflects
    the run's final cost, which can creep above the breach total as in-flight
    work that was already paid for settles.
    """

    reason: Literal["max_cost_exceeded"] = "max_cost_exceeded"
    breach_accumulated_cost: str
    accumulated_cost: str
    max_cost: str
    currency: str


class AsyncOrchestratorRunStop(StrictModel):
    """Typed terminal reason for an async orchestrator run."""

    reason: Literal[
        "all_tasks_complete",
        "max_cost_exceeded",
        "operator_drain",
        "operator_stop",
    ]
    message: str | None = None


def _empty_str_int_map() -> dict[str, int]:
    return {}


class AsyncOrchestratorRunSummary(StrictModel):
    """Run-level counters captured at the end of an async orchestrator run."""

    tasks_total: int = 0
    tasks_by_status: dict[str, int] = Field(default_factory=_empty_str_int_map)
    worktrees_total: int = 0
    worktrees_by_state: dict[str, int] = Field(default_factory=_empty_str_int_map)

    @classmethod
    def from_snapshots(
        cls,
        task_manager: TaskManager,
        worktrees: Sequence[AsyncOrchestratorWorktree],
    ) -> AsyncOrchestratorRunSummary:
        """Build run-level counters from final store snapshots."""

        tasks_by_status = {status.value: 0 for status in TaskStatus}
        for task in task_manager.tasks:
            tasks_by_status[task.status.value] += 1
        worktrees_by_state = {worktree_state.value: 0 for worktree_state in WorktreeState}
        for worktree in worktrees:
            worktrees_by_state[worktree.state.value] += 1
        return cls(
            tasks_total=len(task_manager.tasks),
            tasks_by_status=tasks_by_status,
            worktrees_total=len(worktrees),
            worktrees_by_state=worktrees_by_state,
        )


class AsyncOrchestratorRunResult(StrictModel):
    """Result returned by an async orchestrator run."""

    root: Path
    entrypoint: Path
    usage: Usage = Field(default_factory=Usage)
    summary: AsyncOrchestratorRunSummary = Field(default_factory=AsyncOrchestratorRunSummary)
    stop: AsyncOrchestratorRunStop | None = None
    budget_stop: AsyncOrchestratorBudgetStop | None = None


class _AsyncOrchestratorComplete(Exception):
    """Raised internally to stop services after all tasks are complete."""


class _WorktreeCreationAdmissionClosed(Exception):
    """Raised internally when admission closes before worktree allocation."""


class _ControlCommandApplicationError(Exception):
    """Expected operator-command validation or state-rejection failure."""


@dataclass(frozen=True, slots=True)
class _AgentSpec:
    """Role-specific contract for running and interpreting an agent."""

    role: AsyncOrchestratorAgentRole
    command: AsyncOrchestratorAgentCommandConfig
    runnable_state: WorktreeState
    failure_state: WorktreeState
    output_type: type[BaseModel]
    running_state: WorktreeState | None = None
    requires_task: bool = False


@dataclass(frozen=True, slots=True)
class _RunAdmissionPolicy:
    """Snapshot of the run-control gates used by schedulers."""

    budget_stopped: bool
    paused: bool
    draining: bool
    stopping: bool

    @property
    def can_start_fresh_work(self) -> bool:
        """Return whether schedulers may create new cost-incurring work."""

        return not (self.budget_stopped or self.paused or self.draining or self.stopping)

    @property
    def can_continue_queued_work(self) -> bool:
        """Return whether existing worker/reviewer work may continue."""

        if self.budget_stopped or self.stopping:
            return False
        return self.draining or not self.paused

    @property
    def can_enqueue_ready_tasks(self) -> bool:
        return self.can_start_fresh_work

    @property
    def can_create_worktree(self) -> bool:
        return self.can_start_fresh_work

    @property
    def can_launch_worker(self) -> bool:
        return self.can_continue_queued_work

    @property
    def can_launch_reviewer(self) -> bool:
        return self.can_continue_queued_work

    @property
    def terminal_stop(self) -> AsyncOrchestratorRunStop | None:
        """Return the terminal run-stop record requested by this policy, if any."""

        if self.budget_stopped:
            return AsyncOrchestratorRunStop(reason="max_cost_exceeded")
        if self.stopping:
            return AsyncOrchestratorRunStop(reason="operator_stop")
        if self.draining:
            return AsyncOrchestratorRunStop(reason="operator_drain")
        return None


@dataclass(frozen=True, slots=True)
class _ValidationCommandFailure:
    """Captured validation command failure details.

    ``timed_out`` marks a command killed for exceeding its configured
    ``timeout_seconds``; ``stdout``/``stderr`` then hold whatever output was
    buffered at kill time, which the batched merge mines for heuristic
    attribution (issue #133).

    ``cancelled`` marks a command that exited on a cancellation signal
    (``_VALIDATION_CANCELLATION_SIGNALS``) outside the timeout path. The
    classification is a heuristic on signal semantics: those signals typically
    indicate exogenous termination (kernel OOM killer, operator kill,
    systemd/container shutdown), though an exit status alone cannot prove who
    sent the signal. The orchestrator's own cancellation never produces a
    classified result — it catches ``asyncio.CancelledError``, terminates the
    POSIX process group (only the direct child on Windows), and re-raises. A
    cancelled validation carries no
    information about the validity of the change under test, so callers may
    retry it rather than book it as an ordinary failure (issue #132). Exits on
    any other signal (SIGSEGV, SIGABRT, SIGQUIT, ...) are validator crashes
    (``crashed``) and stay ordinary failures. Timeouts are *not* cancelled:
    they record ``returncode=None`` with a distinct ``error`` before the kill
    escalation runs.

    ``command_index`` is the position of the failed command in the configured
    sequence, so callers can budget retries per command.
    """

    argv: tuple[str, ...]
    returncode: int | None
    stdout: str = ""
    stderr: str = ""
    error: str = ""
    timed_out: bool = False
    cancelled: bool = False
    command_index: int = 0

    @property
    def signal_number(self) -> int | None:
        """POSIX signal that terminated the command, or ``None`` for plain exits."""

        if self.returncode is not None and self.returncode < 0:
            return -self.returncode
        return None

    @property
    def crashed(self) -> bool:
        """True when the command exited on a non-cancellation (crash) signal."""

        return self.signal_number is not None and not self.cancelled


@dataclass(slots=True)
class _CancellationRetryBudget:
    """Episode-wide budget of cancellation retries for batch validation.

    One instance covers an entire publish/isolation episode — the top-level
    batch and every bisection node under it: each command index gets at most
    one cancellation retry across the whole episode, bounding total retries at
    ``len(commands)`` no matter how many nodes revalidate. A fresh budget per
    node would let a persistent external killer multiply killed validations by
    the O(2N) bisection tree, amplifying the very OOM pressure the
    cancellation retry mitigates.
    """

    retried_command_indices: set[int] = field(default_factory=set[int])

    def try_consume(self, command_index: int) -> bool:
        """Reserve the retry for ``command_index``; ``False`` once spent."""

        if command_index in self.retried_command_indices:
            return False
        self.retried_command_indices.add(command_index)
        return True


@dataclass(frozen=True, slots=True)
class _GroupValidationFailure:
    """A member group that failed staged validation during batched-merge isolation.

    ``members`` is the conflict-free assembled subset the failure applies to
    (conflicting members were already bounced during assembly). ``message`` is
    the worker-facing discussion message of *this group's own* failed
    validation — used verbatim when the group is a single member and therefore
    bounces (verify-before-bounce). ``reported_paths`` carries the failure's
    file-attribution hints for partitioning the group.
    """

    members: list[tuple[AsyncOrchestratorWorktree, str]]
    message: str
    reported_paths: set[str]


@dataclass(frozen=True, slots=True)
class _MergeWorktreeResult:
    """Result of applying a worktree merge in the entrypoint repository.

    ``original_head`` is the entrypoint HEAD captured immediately before the
    merge, used to roll back on a post-merge validation failure. It is ``None``
    when the worktree had no committed changes to merge: workers own their own
    commits, so an empty contribution lands nothing and is returned to the
    worker rather than being merged.
    """

    original_head: str | None


class AsyncOrchestrator:
    """Parent class for the async orchestrator implementation."""

    __slots__ = (
        "_budget_stop",
        "_build_cache_lock",
        "_control_run_id",
        "_run_stop",
        "_validation_worktree_ready",
        "config",
        "runtime",
        "store",
    )

    config: AsyncOrchestratorConfig
    runtime: AsyncOrchestratorRuntime
    store: SQLiteAsyncOrchestratorStore
    _budget_stop: AsyncOrchestratorBudgetStop | None
    _control_run_id: str | None
    _run_stop: AsyncOrchestratorRunStop | None

    def __init__(
        self,
        config: AsyncOrchestratorConfig,
        *,
        store: SQLiteAsyncOrchestratorStore | None = None,
        check_resume_health: bool = False,
    ) -> None:
        self.config = config
        self.store = SQLiteAsyncOrchestratorStore(config.root) if store is None else store
        if check_resume_health:
            reset_count = self.store.reset_running_worktrees()
            if reset_count:
                _LOGGER.info(
                    "reset %d running async worktree(s) for process resume",
                    reset_count,
                )
        runtime_worktrees: Iterable[AsyncOrchestratorWorktree]
        stored_worktrees = self.store.list_worktrees()
        if check_resume_health:
            runtime_worktrees = _healthy_resume_worktrees_for_runtime(stored_worktrees)
        else:
            runtime_worktrees = stored_worktrees
        self.runtime = AsyncOrchestratorRuntime(
            runtime_worktrees,
            worker_agent_limit=config.max_concurrent_worker_agents,
            reviewer_agent_limit=config.max_concurrent_reviewer_agents,
        )
        self._budget_stop = None
        self._control_run_id = None
        self._run_stop = None
        # Lazily set true once the staging validation worktree has been created
        # (or reused on resume). Only used when ``merge_validation_worktree``.
        self._validation_worktree_ready = False
        # Serializes refresh (post-merge snapshot) and reads (seed-on-creation)
        # of the ``<root>/.build-cache`` build snapshot, so a worktree never
        # seeds from a half-written cache. Only used when ``seed_worktree_build``.
        self._build_cache_lock = asyncio.Lock()

    @property
    def _entrypoint_guard_lock(self) -> asyncio.Lock:
        """Lock guarding consistent entrypoint reads and the create-worktree decision.

        With ``merge_validation_worktree`` the entrypoint is mutated only by the
        brief fast-forward publish step (also under ``entrypoint_lock``), so
        ready-task worktree creation only needs ``entrypoint_lock`` and is *not*
        blocked while a staging validation build runs. The legacy in-entrypoint
        merge path mutates the entrypoint for the whole merge (including the
        build), so creation must share ``merge_lock`` with it. Worktree creation
        must branch only from a committed, stable entrypoint.
        """

        return (
            self.runtime.entrypoint_lock
            if self.config.merge_validation_worktree
            else self.runtime.merge_lock
        )

    @property
    def control_store(self) -> SQLiteAsyncOrchestratorStore:
        """Backward-compatible alias for the unified orchestrator store."""

        return self.store

    @control_store.setter
    def control_store(self, value: SQLiteAsyncOrchestratorStore) -> None:
        self.store = value

    @property
    def worktree_ids(self) -> tuple[str, ...]:
        """Return worktree IDs created by this orchestrator instance."""

        return tuple(worktree.worktree_id for worktree in self.store.list_worktrees())

    @property
    def worktrees_by_id(self) -> Mapping[str, AsyncOrchestratorWorktree]:
        """Return created worktrees keyed by orchestrator-local worktree ID."""

        return MappingProxyType(
            {worktree.worktree_id: worktree for worktree in self.store.list_worktrees()}
        )

    @property
    def task_queue(self) -> tuple[str, ...]:
        """Return queued ready task IDs in FIFO order."""

        return self.runtime.task_queue.items

    @property
    def worker_queue(self) -> tuple[str, ...]:
        """Return queued worker worktree IDs in FIFO order."""

        return self.runtime.worker_queue.items

    @property
    def review_queue(self) -> tuple[str, ...]:
        """Return queued review worktree IDs in FIFO order."""

        return self.runtime.review_queue.items

    @property
    def merge_queue(self) -> tuple[str, ...]:
        """Return queued merge worktree IDs in FIFO order."""

        return self.runtime.merge_queue.items

    @property
    def usage(self) -> Usage:
        """Return aggregate usage from managed tend agent sessions."""

        return self.store.aggregate_usage(self.config.root)

    def _admission_policy(self) -> _RunAdmissionPolicy:
        """Return the current scheduler admission policy snapshot."""

        return _RunAdmissionPolicy(
            budget_stopped=self._budget_stop is not None,
            paused=self.runtime.paused,
            draining=self.runtime.draining,
            stopping=self.runtime.stopping,
        )

    def _effective_worker_agent_limit(self) -> int:
        """Return the current worker launch limit after admission gates."""

        if not self._admission_policy().can_launch_worker:
            return 0
        return self.runtime.worker_agent_limit

    def _effective_reviewer_agent_limit(self) -> int:
        """Return the current reviewer launch limit after admission gates."""

        if not self._admission_policy().can_launch_reviewer:
            return 0
        return self.runtime.reviewer_agent_limit

    def _complete_run(self, stop: AsyncOrchestratorRunStop) -> None:
        """Record the terminal stop reason and stop the service TaskGroup."""

        self._run_stop = stop
        raise _AsyncOrchestratorComplete

    async def _to_thread_call(self, call: Callable[[], _T]) -> _T:
        """Run a blocking call without abandoning its worker thread.

        ``asyncio.to_thread`` cannot stop already-running synchronous work when
        the awaiting task is cancelled. Shielding and then awaiting the worker
        task on cancellation ensures the orchestrator does not treat a cancelled
        async task as settled while its thread is still touching SQLite, git, or
        another non-cancellable critical section.
        """

        task = asyncio.create_task(asyncio.to_thread(call))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            # Re-entrant signals must not cancel the wrapper task and abandon
            # the still-running executor operation.
            await _wait_for_task_settle_ignoring_cancellation(
                task,
                timeout_seconds=None,
            )
            raise

    async def _run_cancellable_worktree_setup_command(
        self,
        command: AsyncOrchestratorWorktreeSetupCommandConfig,
        *,
        entrypoint: Path,
        worktree: Path,
    ) -> None:
        """Run setup in a tracked session and terminate it on cancellation.

        POSIX setup commands run in a process-group session whose descendants
        are signalled together. On Windows, only the direct child is terminated;
        descendant containment requires POSIX process groups and remains part of
        the stronger containment work tracked in issue #152.
        """

        runner = _WorktreeSetupCommandRunner(
            command,
            entrypoint=entrypoint,
            worktree=worktree,
        )
        task = asyncio.create_task(_run_in_abandonable_thread(runner.run))
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            terminate_signal = int(signal.SIGTERM)
            escalation_signal = int(getattr(signal, "SIGKILL", terminate_signal))
            runner.signal_process_group(terminate_signal)
            settled = await _wait_for_task_settle_ignoring_cancellation(
                task,
                timeout_seconds=_VALIDATION_PROVISION_CANCEL_SETTLE_SECONDS,
            )
            if not settled:
                _LOGGER.warning(
                    "provisioning setup command did not terminate after signal %s; "
                    "escalating its POSIX process group/direct child with signal %s",
                    terminate_signal,
                    escalation_signal,
                )
            # On POSIX, terminate same-session descendants even if the setup
            # leader exited gracefully during the first window. Windows can
            # signal only the direct child and repeats SIGTERM when SIGKILL is
            # unavailable.
            runner.signal_process_group(escalation_signal)
            if not settled:
                settled = await _wait_for_task_settle_ignoring_cancellation(
                    task,
                    timeout_seconds=_VALIDATION_PROVISION_CANCEL_SETTLE_SECONDS,
                )
                if not settled:
                    _LOGGER.warning(
                        "provisioning setup command is unresponsive and will be "
                        "abandoned after cancellation (follow-up issue #152)"
                    )
                    task.add_done_callback(_consume_abandoned_thread_result)
            raise

    async def _store_call(self, call: Callable[[], _T]) -> _T:
        """Run a short SQLite store call without abandoning its thread."""

        return await self._to_thread_call(call)

    async def _control_store_call(self, call: Callable[[], _T]) -> _T:
        """Run a short control-store call without abandoning its thread."""

        return await self._store_call(call)

    def _live_control_status(self) -> tuple[ControlRunStatus, str | None]:
        """Return the live control-store status represented by runtime flags."""

        if self.runtime.stopping:
            return "stopping", "operator_stop"
        if self.runtime.draining:
            return "draining", "operator_drain"
        if self._budget_stop is not None:
            return "draining", "max_cost_exceeded"
        return "running", None

    async def _control_active_agents_snapshot(
        self,
    ) -> tuple[ControlActiveAgentSnapshot, ...]:
        """Return active worker/reviewer agents for a heartbeat snapshot."""

        active_by_role = (
            ("worker", self.runtime.active_agent_worktree_ids("worker")),
            ("reviewer", self.runtime.active_agent_worktree_ids("reviewer")),
        )
        worktree_ids = tuple(
            dict.fromkeys(
                worktree_id
                for _, role_worktree_ids in active_by_role
                for worktree_id in role_worktree_ids
            )
        )
        summaries = await self._store_call(
            lambda: self.store.worktree_control_summaries(worktree_ids)
        )
        snapshots: list[ControlActiveAgentSnapshot] = []
        for role, role_worktree_ids in active_by_role:
            for worktree_id in role_worktree_ids:
                summary = summaries.get(worktree_id)
                snapshots.append(
                    ControlActiveAgentSnapshot(
                        role=cast(ControlActiveAgentRole, role),
                        worktree_id=worktree_id,
                        task_id=None if summary is None else summary[0],
                        worktree_state=None if summary is None else summary[1],
                    )
                )
        return tuple(snapshots)

    def _terminal_control_status(
        self,
        run_error: BaseException | None,
    ) -> tuple[Literal["stopped", "completed", "failed"], str | None]:
        """Return the terminal control-store status for a finished run."""

        if run_error is not None:
            if isinstance(run_error, asyncio.CancelledError):
                return "failed", "cancelled"
            return "failed", type(run_error).__name__
        if self._run_stop is None:
            return "stopped", "run_ended"
        if self._run_stop.reason == "all_tasks_complete":
            return "completed", self._run_stop.reason
        return "stopped", self._run_stop.reason

    async def _register_control_run(self) -> None:
        """Create the durable control-store row for this process run."""

        run_id = new_control_run_id()
        self._control_run_id = run_id
        status, status_reason = self._live_control_status()
        try:
            await self._control_store_call(
                lambda: self.control_store.register_run(
                    run_id=run_id,
                    pid=os.getpid(),
                    status=status,
                    status_reason=status_reason,
                    worker_limit=self.runtime.worker_agent_limit,
                    reviewer_limit=self.runtime.reviewer_agent_limit,
                    paused=self.runtime.paused,
                    drain_requested=self.runtime.draining,
                )
            )
        except asyncio.CancelledError:
            raise
        except BaseException:
            self._control_run_id = None
            raise

    async def _record_control_heartbeat_once(self) -> None:
        """Persist a heartbeat/status snapshot for the registered run."""

        run_id = self._control_run_id
        if run_id is None:
            return
        status, status_reason = self._live_control_status()
        active_agents = await self._control_active_agents_snapshot()
        await self._control_store_call(
            lambda: self.control_store.record_run_heartbeat(
                run_id=run_id,
                status=status,
                status_reason=status_reason,
                worker_limit=self.runtime.worker_agent_limit,
                reviewer_limit=self.runtime.reviewer_agent_limit,
                paused=self.runtime.paused,
                drain_requested=self.runtime.draining,
                active_agents=active_agents,
            )
        )

    async def _record_control_run_finished(
        self,
        run_error: BaseException | None,
    ) -> None:
        """Persist the terminal control-store status for the registered run."""

        run_id = self._control_run_id
        if run_id is None:
            return
        status, status_reason = self._terminal_control_status(run_error)
        await self._control_store_call(
            lambda: self.control_store.record_run_finished(
                run_id=run_id,
                status=status,
                status_reason=status_reason,
                worker_limit=self.runtime.worker_agent_limit,
                reviewer_limit=self.runtime.reviewer_agent_limit,
                paused=self.runtime.paused,
                drain_requested=self.runtime.draining,
            )
        )

    async def _cancel_incomplete_control_commands(
        self,
        run_error: BaseException | None,
    ) -> None:
        """Cancel durable commands this run claimed but did not finish."""

        run_id = self._control_run_id
        if run_id is None:
            return
        error = "run ended before command completed"
        if isinstance(run_error, asyncio.CancelledError):
            error = "run was cancelled before command completed"
        elif run_error is not None:
            error = f"run failed before command completed: {type(run_error).__name__}"
        await self._control_store_call(
            lambda: self.control_store.cancel_incomplete_commands_for_run(
                run_id=run_id,
                error=error,
            )
        )

    async def _control_service_forever(self) -> None:
        """Publish heartbeats and apply currently supported control commands."""

        while True:
            try:
                await self._apply_pending_control_commands_once()
                await self._record_control_heartbeat_once()
            except AsyncOrchestratorControlStoreIOError as exc:
                _LOGGER.warning(
                    "async orchestrator control-store operation failed; will retry: %s",
                    exc,
                )
            await asyncio.sleep(_CONTROL_HEARTBEAT_INTERVAL_SECONDS)

    async def _apply_pending_control_commands_once(self) -> None:
        """Claim and apply all currently pending commands for this run."""

        run_id = self._control_run_id
        if run_id is None:
            return
        while True:
            command = await self._control_store_call(
                lambda: self.control_store.claim_pending_command(run_id=run_id)
            )
            if command is None:
                return
            await self._apply_control_command(command)

    async def _apply_control_command(self, command: ControlCommandRecord) -> None:
        """Apply one claimed command and persist its completion status."""

        try:
            result = await self._apply_control_command_to_runtime(command)
        except _ControlCommandApplicationError as exc:
            error = str(exc)
            await self._control_store_call(
                lambda: self.control_store.record_command_failed(
                    command.id,
                    error=error,
                )
            )
            return
        await self._control_store_call(
            lambda: self.control_store.record_command_succeeded(
                command.id,
                result=result,
            )
        )
        await self._record_control_heartbeat_once()

    async def _apply_control_command_to_runtime(
        self,
        command: ControlCommandRecord,
    ) -> JsonObject:
        """Mutate live runtime/config controls for one validated command."""

        params = command.params
        if command.command == "noop":
            return {"applied": True}
        if command.command == "pause":
            _require_control_params(command.command, params, allowed=())
            self._ensure_control_command_can_start(command.command)
            self.runtime.set_paused(True)
            return {"paused": True}
        if command.command == "resume":
            _require_control_params(command.command, params, allowed=())
            self._ensure_control_command_can_start(command.command)
            self.runtime.set_paused(False)
            return {"paused": False}
        if command.command == "limits":
            _require_control_params(command.command, params, allowed=("workers", "reviewers"))
            worker_limit = _optional_control_int_param(params, "workers")
            reviewer_limit = _optional_control_int_param(params, "reviewers")
            if worker_limit is None and reviewer_limit is None:
                raise _ControlCommandApplicationError(
                    "limits command requires workers and/or reviewers"
                )
            if self.runtime.draining:
                if worker_limit == 0:
                    raise _ControlCommandApplicationError(
                        "cannot set worker launch limit to 0 while drain is active; "
                        "raise the worker limit or use stop semantics"
                    )
                if reviewer_limit == 0:
                    raise _ControlCommandApplicationError(
                        "cannot set reviewer launch limit to 0 while drain is active; "
                        "raise the reviewer limit or use stop semantics"
                    )
            if worker_limit is not None:
                self.runtime.set_worker_agent_limit(worker_limit)
            if reviewer_limit is not None:
                self.runtime.set_reviewer_agent_limit(reviewer_limit)
            return {
                "worker_limit": self.runtime.worker_agent_limit,
                "reviewer_limit": self.runtime.reviewer_agent_limit,
            }
        if command.command == "drain":
            _require_control_params(command.command, params, allowed=())
            if self.runtime.stopping:
                raise _ControlCommandApplicationError(
                    "cannot request drain after stop is active"
                )
            if self.runtime.worker_agent_limit == 0:
                raise _ControlCommandApplicationError(
                    "cannot request drain while worker launch limit is 0; "
                    "raise the worker limit or use stop semantics"
                )
            if self.runtime.reviewer_agent_limit == 0:
                raise _ControlCommandApplicationError(
                    "cannot request drain while reviewer launch limit is 0; "
                    "raise the reviewer limit or use stop semantics"
                )
            self.runtime.request_drain()
            return {"drain_requested": True}
        if command.command == "stop":
            _require_control_params(command.command, params, allowed=("now",))
            now = _control_bool_param(params, "now", default=False)
            active_agents = self.runtime.active_agent_count()
            self.runtime.request_stop()
            if now:
                await self.runtime.cancel_agent_tasks(raise_failures=True)
            return {
                "stopping": True,
                "now": now,
                "cancelled_agent_tasks": active_agents if now else 0,
            }
        if command.command == "budget":
            _require_control_params(command.command, params, allowed=("max_cost",))
            self._ensure_control_command_can_start(command.command)
            max_cost = _control_decimal_param(params, "max_cost")
            budget_currency = self.config.budget.currency
            budget_stop = await self._budget_stop_for_ceiling(max_cost, budget_currency)
            self.config.budget = self.config.budget.model_copy(
                update={"max_cost": max_cost}
            )
            budget_stop_recorded = self._budget_stop is not None
            if budget_stop is not None and self._budget_stop is None:
                self._record_budget_stop(budget_stop)
                budget_stop_recorded = True
            return {
                "max_cost": format(max_cost, "f"),
                "currency": self.config.budget.currency,
                "budget_stop_recorded": budget_stop_recorded,
            }
        raise _ControlCommandApplicationError(
            f"unsupported control command: {command.command}"
        )

    def _ensure_control_command_can_start(self, command: str) -> None:
        """Reject commands that cannot reopen an already-terminal admission gate."""

        if self.runtime.stopping:
            raise _ControlCommandApplicationError(
                f"cannot apply {command} after stop is active"
            )
        if self.runtime.draining:
            raise _ControlCommandApplicationError(
                f"cannot apply {command} after drain is active"
            )
        if self._budget_stop is not None:
            raise _ControlCommandApplicationError(
                f"cannot apply {command} after budget stop is active"
            )

    async def run(self) -> AsyncOrchestratorRunResult:
        """Run the async orchestrator background services until cancelled.

        Ready-task discovery is the only polling service.  It syncs task YAML
        from disk and enqueues ready tasks.  The other services block on
        runtime queues and wake immediately when work is available.
        """

        _LOGGER.info(
            "starting async orchestrator: root=%s entrypoint=%s",
            self.config.root,
            self.config.entrypoint,
        )
        run_error: BaseException | None = None
        try:
            await self._store_call(self.store.initialize_state)
            await self._register_control_run()
            try:
                async with asyncio.TaskGroup() as task_group:
                    services = (
                        (self._control_service_forever, "async-orchestrator-control"),
                        (
                            self._enqueue_ready_tasks_forever,
                            "async-orchestrator-ready-task-discovery",
                        ),
                        (self._process_task_queue_forever, "async-orchestrator-task-queue"),
                        (self._spawn_worker_agents_forever, "async-orchestrator-worker-agents"),
                        (
                            self._spawn_reviewer_agents_forever,
                            "async-orchestrator-reviewer-agents",
                        ),
                        (self._process_merge_queue_forever, "async-orchestrator-merge-queue"),
                        # One-shot: capture usage snapshots for terminal sessions
                        # resumed from state written before the snapshot field
                        # existed. Runs CONCURRENTLY with the services (not before
                        # them) so a large resume's backfill never blocks startup;
                        # it returns once done and aggregation reads each session
                        # at most once until then.
                        (
                            self._backfill_session_usage_once,
                            "async-orchestrator-usage-backfill",
                        ),
                    )
                    for service, name in services:
                        task_group.create_task(service(), name=name)
            except* _AsyncOrchestratorComplete:
                if self._run_stop is None:
                    _LOGGER.info("async orchestrator run settled")
                elif self._run_stop.reason == "all_tasks_complete":
                    _LOGGER.info("all async orchestrator tasks are complete")
                else:
                    _LOGGER.info(
                        "async orchestrator run settled: reason=%s",
                        self._run_stop.reason,
                    )
            except* FrameworkError as exc_group:
                # Unwrap any framework error raised inside the TaskGroup so the
                # CLI's ``except FrameworkError`` handler can catch it directly
                # (it would otherwise see an ``ExceptionGroup`` and fall through
                # to the unhandled-exception path). Covers both the existing
                # state/usage store errors and budget guard mismatches.
                raise exc_group.exceptions[0] from None
        except BaseException as exc:
            run_error = exc
            raise
        finally:
            _LOGGER.info("stopping async orchestrator")
            await self.runtime.cancel_agent_tasks()
            await self._record_control_run_finished(run_error)
            await self._cancel_incomplete_control_commands(run_error)

        # Refresh once more after agent cancellation so the returned
        # ``accumulated_cost`` reflects any post-breach session that finished
        # writing its usage between the last poll-loop refresh and shutdown.
        # Defensive (the polling loop already refreshes on every settle tick),
        # but mirrors the shared end-of-tick refresh path.
        if self._budget_stop is not None:
            await self._refresh_budget_stop_accumulated_cost()
        usage = await self._store_call(lambda: self.store.aggregate_usage(self.config.root))
        task_manager, worktrees = await self._store_call(
            lambda: (self.store.load_task_snapshot(), self.store.list_worktrees())
        )
        _LOGGER.info("async orchestrator %s", format_usage_summary(usage))
        return AsyncOrchestratorRunResult(
            root=self.config.root,
            entrypoint=self.config.entrypoint,
            usage=usage,
            summary=AsyncOrchestratorRunSummary.from_snapshots(task_manager, worktrees),
            stop=self._run_stop,
            budget_stop=self._budget_stop,
        )

    async def _enqueue_ready_tasks_forever(self) -> None:
        """Poll task files and enqueue newly ready tasks.

        Once a terminal run-control gate closes (configured ``max_cost`` breach
        or operator drain/stop), the loop stops claiming fresh ready tasks,
        refreshes budget cost if applicable, and completes when the relevant
        pipeline work has drained. Non-terminal pause uses the same admission
        policy but does not request exit. The merge processor and discussion
        transitions continue so MERGE worktrees and agent failures still resolve
        cleanly. Budget drain mirrors the sync ``#70`` freeze semantics, while
        operator drain continues already-created worker/reviewer worktrees.
        """

        while True:
            await self._sync_task_manager_once()
            admission = self._admission_policy()
            terminal_stop = admission.terminal_stop
            # If a terminal run-control gate is already active when the final
            # task completes, report the operative control reason rather than
            # overwriting it with all_tasks_complete.
            if terminal_stop is None and await self._all_tasks_complete_once():
                # Discovery's sync is advisory and lock-free, so ``state`` can
                # momentarily hold a torn/stale all-complete view read mid
                # merge-publish (e.g. a merge that completes a task and adds an
                # open follow-up, observed after the completion landed but before
                # the new file did). Completion is terminal, so re-confirm against
                # an authoritative guard-locked reload before stopping. The locked
                # reload is paid only on this rare all-complete edge.
                if await self._all_tasks_complete_locked():
                    self._complete_run(
                        AsyncOrchestratorRunStop(reason="all_tasks_complete")
                    )
            if self._budget_stop is None:
                await self._record_budget_stop_if_exceeded()
            admission = self._admission_policy()
            terminal_stop = admission.terminal_stop
            if terminal_stop is not None:
                if self._budget_stop is not None:
                    await self._refresh_budget_stop_accumulated_cost()
                if await self._in_flight_work_settled():
                    self._complete_run(terminal_stop)
            else:
                await self._enqueue_ready_tasks_once()
            await asyncio.sleep(_TASK_DISCOVERY_POLL_INTERVAL_SECONDS)

    async def _record_budget_stop_if_exceeded(self) -> bool:
        """Return whether the cost ceiling is reached, recording the breach once.

        Uses a strict per-session currency check (``_accumulated_cost_strict``):
        any session whose cost currency differs from ``budget.currency`` raises
        :class:`AsyncCostBudgetCurrencyMismatchError` so the guard fails closed
        instead of silently letting a mismatch disable the ceiling. Inclusive
        ``>=`` semantics match the shared ``#70`` implementation.
        """

        if self._budget_stop is not None:
            return True
        max_cost = self.config.budget.max_cost
        if max_cost is None:
            return False
        budget_currency = self.config.budget.currency
        budget_stop = await self._budget_stop_for_ceiling(max_cost, budget_currency)
        # A live ``budget`` command can update the ceiling while the usage read
        # above is still running in a worker thread.  Re-check the sampled budget
        # before recording an irreversible terminal drain so an older, lower
        # limit cannot win a race with an operator raising the ceiling.
        if (
            self.config.budget.max_cost != max_cost
            or self.config.budget.currency != budget_currency
            or self._budget_stop is not None
        ):
            return self._budget_stop is not None
        if budget_stop is None:
            return False
        self._record_budget_stop(budget_stop)
        return True

    async def _budget_stop_for_ceiling(
        self,
        max_cost: Decimal,
        budget_currency: str,
    ) -> AsyncOrchestratorBudgetStop | None:
        worktrees = await self._store_call(self.store.list_worktrees)
        cost = await asyncio.to_thread(
            self._accumulated_cost_strict,
            worktrees,
            budget_currency,
        )
        if cost is None or cost.amount < max_cost:
            return None
        amount = format(cost.amount, "f")
        return AsyncOrchestratorBudgetStop(
            breach_accumulated_cost=amount,
            accumulated_cost=amount,
            max_cost=format(max_cost, "f"),
            currency=budget_currency,
        )

    def _record_budget_stop(self, budget_stop: AsyncOrchestratorBudgetStop) -> None:
        self._budget_stop = budget_stop
        _LOGGER.warning(
            "async run reached cost ceiling: accumulated %s %s >= max_cost %s %s; "
            "freezing cost-incurring stages and draining merges/transitions",
            budget_stop.breach_accumulated_cost,
            budget_stop.currency,
            budget_stop.max_cost,
            budget_stop.currency,
        )

    async def _refresh_budget_stop_accumulated_cost(self) -> None:
        """Refresh ``accumulated_cost`` to the latest aggregate after a breach.

        ``breach_accumulated_cost`` stays pinned to the first-breach value; only
        ``accumulated_cost`` is updated so the returned record reflects costs
        that accrued from in-flight worker/reviewer sessions that completed
        after the freeze. Uses the same strict per-session aggregator as the
        initial breach check, so a post-breach session in another currency
        still fails closed.
        """

        if self._budget_stop is None:
            return
        worktrees = await self._store_call(self.store.list_worktrees)
        cost = await asyncio.to_thread(
            self._accumulated_cost_strict,
            worktrees,
            self._budget_stop.currency,
        )
        if cost is None:
            return
        amount = format(cost.amount, "f")
        if amount == self._budget_stop.accumulated_cost:
            return
        self._budget_stop = self._budget_stop.model_copy(
            update={"accumulated_cost": amount}
        )

    async def _in_flight_work_settled(self) -> bool:
        """Return whether the run can exit cleanly under a terminal control gate.

        Budget stops and operator stops keep freeze semantics: once no ready-task
        creation is in progress, no merge is active, and no worker/reviewer
        ``asyncio.Task`` is still running, the run may exit even if unlaunched
        worker/reviewer work remains for a future resume.

        Operator drain is stricter. It stops fresh ready-task worktree creation,
        but continues already-created worker/reviewer/merge work until the
        worker, review, and merge queues are settled and no durable
        ``PENDING``/``REVIEW``/``MERGE`` worktree remains. Visible ready-task
        items are still ignored because drain intentionally refuses to create
        fresh worktrees for them; only reserved/in-progress ready-task creation
        can keep drain open.
        """

        operator_drain = (
            self.runtime.draining
            and self._budget_stop is None
            and not self.runtime.stopping
        )
        prune_done_agent_tasks(self.runtime.worker_agent_tasks)
        prune_done_agent_tasks(self.runtime.reviewer_agent_tasks)
        if (
            self.runtime.task_queue.has_reserved_items
            or self.runtime.merge_queue.has_claimed_items
        ):
            return False
        if operator_drain and (
            self.runtime.worker_queue.has_claimed_items
            or self.runtime.review_queue.has_claimed_items
        ):
            return False
        if self.runtime.worker_agent_tasks or self.runtime.reviewer_agent_tasks:
            return False
        worktrees = await self._store_call(self.store.list_worktrees)
        for worktree in worktrees:
            if worktree.state is WorktreeState.MERGE:
                return False
            if operator_drain and self._worktree_requires_operator_drain(worktree):
                return False
        return True

    @staticmethod
    def _worktree_requires_operator_drain(worktree: AsyncOrchestratorWorktree) -> bool:
        """Return whether ``worktree`` represents queued work operator drain must finish."""

        return (
            (worktree.state is WorktreeState.PENDING and worktree.task_id is not None)
            or worktree.state is WorktreeState.REVIEW
            or worktree.state is WorktreeState.MERGE
        )

    def _accumulated_cost_strict(
        self,
        worktrees: Iterable[AsyncOrchestratorWorktree],
        currency: str,
    ) -> Cost | None:
        """Sum priced per-session ``Cost`` while enforcing ``currency`` strictly.

        Returns ``None`` when no priced session has been recorded; otherwise
        returns a ``Cost`` denominated in ``currency``. Any per-session cost
        whose currency differs raises
        :class:`AsyncCostBudgetCurrencyMismatchError` so the budget guard
        cannot be silently disabled by a session charged in another currency.

        Iterates raw ``load_agent_session_usage`` reads (not the lenient
        :func:`aggregate_agent_session_usage`, which drops cost on the first
        mismatch and would silently disable the guard).
        """

        total: Cost | None = None
        for worktree in sorted(worktrees, key=lambda item: item.worktree_id):
            for role in (
                AsyncOrchestratorAgentRole.WORKER,
                AsyncOrchestratorAgentRole.REVIEWER,
            ):
                usage = resolve_agent_session_usage(
                    self.config.root,
                    worktree,
                    role,
                )
                if usage is None or usage.cost is None:
                    continue
                if usage.cost.currency != currency:
                    raise AsyncCostBudgetCurrencyMismatchError(
                        f"async cost budget currency mismatch: worktree "
                        f"{worktree.worktree_id} {role.value} session cost is in "
                        f"{usage.cost.currency} but max_cost is configured in "
                        f"{currency}; refusing to compare across currencies "
                        "because that would silently disable the budget guard"
                    )
                total = (
                    usage.cost.model_copy(deep=True)
                    if total is None
                    else total.add(usage.cost)
                )
        return total

    async def _record_session_usage(
        self,
        worktree_id: str,
        role: AsyncOrchestratorAgentRole,
    ) -> None:
        """Snapshot a stopped agent session's usage onto its worktree.

        Read once here (the session has stopped) and store on the worktree, so
        usage aggregation reads the stored value instead of re-replaying this
        immutable log on every state save.
        """

        usage = await self._to_thread_call(
            lambda: load_agent_session_usage(
                _absolute_path(self.config.root),
                worktree_id,
                role,
            )
        )
        if usage is None:
            return
        try:
            await self._store_call(
                lambda: self.store.set_agent_session_usage(worktree_id, role, usage)
            )
        except ValueError:
            return

    async def _backfill_session_usage_once(self) -> None:
        """One-time: snapshot usage for terminal sessions that lack a stored value.

        Pre-existing worktrees (resumed state written before the usage snapshot
        field existed) have no stored usage; without this, aggregation would
        re-read their immutable logs on every save forever. Read each such
        terminal session's log once and store it; active sessions are skipped
        (they are snapshotted when they stop).

        The reads happen before writes so each immutable log is replayed at most
        once; before storing a snapshot, the worktree is re-read from SQLite so a
        fresher stopped-session snapshot is never clobbered.
        """

        worktrees = await self._store_call(self.store.list_worktrees)
        root = _absolute_path(self.config.root)
        # Read-only pass (no state writes, so no usage re-aggregation triggered).
        # The eligibility checks here are against the snapshot above and only
        # decide whether to bother reading the log — the authoritative decision
        # is re-made under the lock at apply time (see below).
        updates: list[tuple[str, AsyncOrchestratorAgentRole, Usage]] = []
        for worktree in worktrees:
            for role in (
                AsyncOrchestratorAgentRole.WORKER,
                AsyncOrchestratorAgentRole.REVIEWER,
            ):
                if not worktree.agent_session_started(role):
                    continue  # never ran for this role — no log to read
                if worktree.agent_session_usage(role) is not None:
                    continue
                if agent_session_is_active(worktree, role):
                    continue
                usage = await asyncio.to_thread(
                    load_agent_session_usage,
                    root,
                    worktree.worktree_id,
                    role,
                )
                if usage is not None:
                    updates.append((worktree.worktree_id, role, usage))
        if not updates:
            return
        applied = 0
        for worktree_id, role, usage in updates:
            target_worktree_id = worktree_id
            target_role = role
            worktree = await self._store_call(
                lambda target_worktree_id=target_worktree_id: self.store.get_worktree(
                    target_worktree_id
                )
            )
            if worktree is None:
                continue
            if worktree.agent_session_usage(role) is not None:
                continue
            if agent_session_is_active(worktree, role):
                continue
            usage_snapshot = usage
            expected_state = worktree.state
            def store_usage_if_still_missing(
                target_worktree_id: str = target_worktree_id,
                target_role: AsyncOrchestratorAgentRole = target_role,
                usage_snapshot: Usage = usage_snapshot,
                expected_state: WorktreeState = expected_state,
            ) -> bool:
                return self.store.set_agent_session_usage_if_missing_and_inactive(
                    target_worktree_id,
                    target_role,
                    usage_snapshot,
                    expected_state=expected_state,
                )

            stored = await self._store_call(store_usage_if_still_missing)
            if stored:
                applied += 1
        _LOGGER.info("backfilled usage snapshots for %d sessions", applied)

    async def _sync_task_manager_once(self) -> None:
        """Sync the SQLite task snapshot from ``<entrypoint>/tasks/*.yaml`` once.

        This runs in the hot discovery loop (every
        ``_TASK_DISCOVERY_POLL_INTERVAL_SECONDS``). The full-tree YAML parse —
        whose cost scales with the number of task files — is done **without**
        holding ``_entrypoint_guard_lock``; the SQLite snapshot replacement is a
        short transaction.

        Discovery is *advisory*: it merely enqueues **candidate** ready tasks.
        The authoritative worktree-creation path
        (``_ensure_worktree_for_ready_task_id``) re-reads the task tree under the
        guard lock before acting, so creation decides against a consistent,
        committed ``tasks/`` view regardless
        of what discovery sees. Keeping the parse off the guard lock stops a
        growing task tree from monopolising the lock that worktree creation
        needs every tick — which otherwise throttles creation as the run scales.

        A torn read during a merge publish is tolerated: a stale/partial view
        self-corrects on the next poll and is filtered by the locked
        re-validation in creation; a parse failure simply skips this poll.
        """

        try:
            task_manager = await asyncio.to_thread(
                load_entrypoint_task_manager,
                _absolute_path(self.config.entrypoint),
            )
        except Exception as exc:  # noqa: BLE001 - advisory poll; retry next tick
            _LOGGER.debug("async task discovery reload skipped (transient): %s", exc)
            return
        await self._store_call(lambda: self.store.replace_task_snapshot(task_manager))
        _LOGGER.debug("synced async task manager: tasks=%d", len(task_manager.tasks))

    async def _reload_task_manager_locked(self) -> TaskManager:
        """Reload tasks while the caller holds the entrypoint guard lock.

        The entrypoint repository is shared by task discovery, worktree creation,
        and merges. Keeping task reloads behind the guard lock
        (``_entrypoint_guard_lock`` — ``merge_lock`` legacy, ``entrypoint_lock``
        with the staging worktree) means callers make decisions against the same
        serialized view of ``tasks/`` and git ``HEAD`` that the merge publish
        step updates.
        """

        task_manager = await asyncio.to_thread(
            load_entrypoint_task_manager,
            _absolute_path(self.config.entrypoint),
        )
        await self._store_call(lambda: self.store.replace_task_snapshot(task_manager))
        _LOGGER.debug("synced async task manager: tasks=%d", len(task_manager.tasks))
        return task_manager

    async def _all_tasks_complete_once(self) -> bool:
        """Return true when a non-empty task set is entirely complete.

        Reads the advisory SQLite task snapshot discovery maintains — a cheap
        gate. A True result is re-confirmed by ``_all_tasks_complete_locked``
        before the run is allowed to stop (see ``_enqueue_ready_tasks_forever``).
        """

        task_manager = await self._store_call(self.store.load_task_snapshot)
        tasks = tuple(task_manager.tasks)
        return bool(tasks) and all(task.status is TaskStatus.COMPLETE for task in tasks)

    async def _all_tasks_complete_locked(self) -> bool:
        """Authoritative all-complete check: reload ``tasks/`` under the guard lock.

        Unlike ``_all_tasks_complete_once`` (which trusts discovery's advisory,
        lock-free view), this reloads from the committed ``tasks/`` under
        ``_entrypoint_guard_lock`` — the same serialized view of ``tasks/`` and
        git ``HEAD`` that worktree creation and the merge publish use — so it can
        never decide completion from a tree read mid-publish. Used to gate the
        terminal ``_AsyncOrchestratorComplete``.
        """

        async with self._entrypoint_guard_lock:
            task_manager = await self._reload_task_manager_locked()
        tasks = tuple(task_manager.tasks)
        return bool(tasks) and all(task.status is TaskStatus.COMPLETE for task in tasks)

    async def _enqueue_ready_tasks_once(self) -> None:
        """Refresh queued ready-task priorities and enqueue new work when allowed."""

        can_enqueue_new_tasks = self._admission_policy().can_enqueue_ready_tasks
        task_manager, worktrees = await self._store_call(
            lambda: (self.store.load_task_snapshot(), self.store.list_worktrees())
        )
        non_closed_task_ids = {
            worktree.task_id
            for worktree in worktrees
            if worktree.task_id is not None and worktree.state is not WorktreeState.CLOSED
        }
        ready_tasks = task_manager.ready_tasks()
        for task in ready_tasks:
            if task.id in non_closed_task_ids:
                continue
            was_queued = task.id in self.runtime.task_queue
            if not can_enqueue_new_tasks and not was_queued:
                continue
            self.runtime.task_queue.enqueue(
                task.id,
                priority=task_priority_rank(task.priority),
            )
            if not was_queued:
                _LOGGER.info(
                    "queued ready async task: %s priority=%s",
                    task.id,
                    task.priority.value,
                )
        self.runtime.task_queue.reorder(task.id for task in ready_tasks)

    async def _process_task_queue_forever(self) -> None:
        """Process ready-task queue items into fresh task worktrees."""

        while True:
            await self.runtime.process_queue_item(
                self.runtime.task_queue,
                self._ensure_worktree_for_ready_task_id,
                wait=True,
                keep_reserved=True,
                can_process=lambda: self._admission_policy().can_create_worktree,
            )

    async def _process_task_queue_once(self) -> None:
        """Process currently queued ready tasks without waiting for more."""

        while await self.runtime.process_queue_item(
            self.runtime.task_queue,
            self._ensure_worktree_for_ready_task_id,
            wait=False,
            keep_reserved=True,
            can_process=lambda: self._admission_policy().can_create_worktree,
        ):
            pass

    async def _ensure_ready_task_worktrees_once(self) -> None:
        """Compatibility helper: enqueue ready tasks and drain the task queue once."""

        await self._enqueue_ready_tasks_once()
        await self._process_task_queue_once()

    async def _ensure_worktree_for_ready_task_id(
        self,
        task_id: str,
    ) -> AsyncOrchestratorWorktree | None:
        """Create and enqueue a worker worktree if ``task_id`` is still ready.

        When admission is closed this returns ``None`` without creating a
        worktree. The queue processor preserves the reserved queue item if the
        gate closed mid-handler, so pause/resume can reuse the same in-memory
        queue instead of relying on a process restart.
        """

        if not self._admission_policy().can_create_worktree:
            return None
        async with self._entrypoint_guard_lock:
            if not self._admission_policy().can_create_worktree:
                return None
            task_manager = await self._reload_task_manager_locked()
            async with self.runtime.worktree_creation_lock:
                if not self._admission_policy().can_create_worktree:
                    return None
                ready_tasks = task_manager.ready_tasks()
                ready_tasks_by_id = {ready_task.id: ready_task for ready_task in ready_tasks}
                current_task = ready_tasks_by_id.get(task_id)
                if current_task is None:
                    return None
                worktrees = await self._store_call(self.store.list_worktrees)
                non_closed_task_ids = {
                    worktree.task_id
                    for worktree in worktrees
                    if worktree.task_id is not None
                    and worktree.state is not WorktreeState.CLOSED
                }
                current_worktree_candidates = tuple(
                    ready_task
                    for ready_task in ready_tasks
                    if ready_task.id not in non_closed_task_ids
                )
                if (
                    not current_worktree_candidates
                    or current_worktree_candidates[0].id != task_id
                ):
                    return None
                try:
                    return await self._create_fresh_worktree_locked(
                        task=current_task,
                        name=None,
                        attach_task_only_if_ready=True,
                        admission_check=(
                            lambda: self._admission_policy().can_create_worktree
                        ),
                    )
                except _WorktreeCreationAdmissionClosed:
                    return None

    async def _spawn_worker_agents_forever(self) -> None:
        """Continuously spawn worker agents for queued pending worktrees."""

        if self.config.worker_agent_command is None:
            await asyncio.Event().wait()
        await self.runtime.spawn_agent_tasks_forever(
            queue=self.runtime.worker_queue,
            tasks=self.runtime.worker_agent_tasks,
            get_max_concurrent=self._effective_worker_agent_limit,
            spawn_task=self._spawn_worker_agent_task,
        )

    async def _spawn_worker_agents_once(self) -> None:
        """Spawn worker agents from currently queued worktrees without waiting."""

        if self.config.worker_agent_command is None:
            return
        await self.runtime.spawn_agent_tasks_once(
            queue=self.runtime.worker_queue,
            tasks=self.runtime.worker_agent_tasks,
            get_max_concurrent=self._effective_worker_agent_limit,
            spawn_task=self._spawn_worker_agent_task,
        )

    def _spawn_worker_agent_task(self, worktree_id: str) -> None:
        if not self._admission_policy().can_launch_worker:
            # Defensive only: the spawner reads an effective limit of zero while
            # admission is closed, so it normally never claims an item here.
            self.runtime.worker_queue.task_done()
            return
        self.runtime.spawn_agent_task(
            worktree_id,
            queue=self.runtime.worker_queue,
            tasks=self.runtime.worker_agent_tasks,
            name="worker",
            run=self._run_worker_agent_for_worktree_id,
        )

    async def _run_worker_agent_for_worktree_id(self, worktree_id: str) -> None:
        """Tag, run, and route one worker worktree queue item."""

        command = self.config.worker_agent_command
        if command is None:
            return
        await self._run_agent_for_worktree_id(
            worktree_id,
            _AgentSpec(
                role=AsyncOrchestratorAgentRole.WORKER,
                command=command,
                runnable_state=WorktreeState.PENDING,
                running_state=WorktreeState.WORKER_RUNNING,
                failure_state=WorktreeState.PENDING,
                requires_task=True,
                output_type=WorkerContributionOutput,
            ),
        )

    async def _spawn_reviewer_agents_forever(self) -> None:
        """Continuously spawn reviewer agents for queued review worktrees."""

        if self.config.reviewer_agent_command is None:
            await asyncio.Event().wait()
        await self.runtime.spawn_agent_tasks_forever(
            queue=self.runtime.review_queue,
            tasks=self.runtime.reviewer_agent_tasks,
            get_max_concurrent=self._effective_reviewer_agent_limit,
            spawn_task=self._spawn_reviewer_agent_task,
        )

    async def _spawn_reviewer_agents_once(self) -> None:
        """Spawn reviewer agents from currently queued worktrees without waiting."""

        if self.config.reviewer_agent_command is None:
            return
        await self.runtime.spawn_agent_tasks_once(
            queue=self.runtime.review_queue,
            tasks=self.runtime.reviewer_agent_tasks,
            get_max_concurrent=self._effective_reviewer_agent_limit,
            spawn_task=self._spawn_reviewer_agent_task,
        )

    def _spawn_reviewer_agent_task(self, worktree_id: str) -> None:
        if not self._admission_policy().can_launch_reviewer:
            # Defensive only: the spawner reads an effective limit of zero while
            # admission is closed, so it normally never claims an item here.
            self.runtime.review_queue.task_done()
            return
        self.runtime.spawn_agent_task(
            worktree_id,
            queue=self.runtime.review_queue,
            tasks=self.runtime.reviewer_agent_tasks,
            name="reviewer",
            run=self._run_reviewer_agent_for_worktree_id,
        )

    async def _run_reviewer_agent_for_worktree_id(self, worktree_id: str) -> None:
        """Run the reviewer command and apply its structured decision."""

        command = self.config.reviewer_agent_command
        if command is None:
            return
        await self._run_agent_for_worktree_id(
            worktree_id,
            _AgentSpec(
                role=AsyncOrchestratorAgentRole.REVIEWER,
                command=command,
                runnable_state=WorktreeState.REVIEW,
                failure_state=WorktreeState.REVIEW,
                output_type=ReviewVerdictOutput,
            ),
        )

    async def _run_agent_for_worktree_id(
        self,
        worktree_id: str,
        spec: _AgentSpec,
    ) -> None:
        """Run one role-agnostic agent spec for a queued worktree."""

        worktree = await self._store_call(lambda: self.store.get_worktree(worktree_id))
        if worktree is None or worktree.state is not spec.runnable_state:
            return
        if spec.requires_task and worktree.task_id is None:
            return
        resume = worktree.agent_session_started(spec.role)
        claimed_worktree_id = worktree.worktree_id
        running_state = spec.running_state
        if running_state is not None:
            claimed = await self._store_call(
                lambda: self.store.set_worktree_state(
                    claimed_worktree_id,
                    expected=spec.runnable_state,
                    new=running_state,
                )
            )
            if not claimed:
                return
        try:
            await self._store_call(
                lambda: self.store.mark_agent_session_started(
                    claimed_worktree_id,
                    spec.role,
                )
            )
        except ValueError:
            return
        refreshed = await self._store_call(lambda: self.store.get_worktree(worktree_id))
        if refreshed is None:
            return
        worktree = refreshed

        await self._to_thread_call(lambda: write_discussion_log_file(worktree))
        _LOGGER.info(
            "running %s agent for worktree %s",
            spec.role.value,
            worktree.worktree_id,
        )
        exit_code, stdout = await run_agent_command(
            self.config,
            spec.command,
            worktree=worktree,
            role=spec.role,
            resume=resume,
        )
        # The agent has stopped, so its session log is final until a possible
        # re-run (which returns the worktree to this role's active state and is
        # re-snapshotted then). Capture its usage onto the worktree now so usage
        # aggregation reads this stored value instead of re-replaying the log on
        # every state save. Done on all exit paths (success/failure/bad-output).
        await self._record_session_usage(worktree.worktree_id, spec.role)
        if exit_code != 0:
            _LOGGER.warning(
                "%s agent failed for worktree %s: exit_code=%d",
                spec.role.value,
                worktree.worktree_id,
                exit_code,
            )
            await self._transition_worktree(
                worktree.worktree_id,
                spec.failure_state,
                expected=spec.running_state or spec.runnable_state,
            )
            return
        try:
            output = _parse_agent_output(stdout, spec.output_type)
        except ValueError:
            _LOGGER.warning(
                "%s agent returned invalid structured output for worktree %s",
                spec.role.value,
                worktree.worktree_id,
            )
            await self._transition_worktree(
                worktree.worktree_id,
                spec.failure_state,
                expected=spec.running_state or spec.runnable_state,
            )
            return
        next_state = _agent_success_state(output)
        blocked_non_task_feedback: str | None = None
        if (
            isinstance(output, WorkerContributionOutput)
            and output.status == "blocked"
            and not await self._to_thread_call(
                lambda: _worktree_has_unmerged_commits(
                    worktree.path,
                    self.config.merge_target_branch,
                )
            )
        ):
            # Blocked contract: a ``blocked`` worker is expected to commit its
            # task-graph edits (a new prerequisite task + its own ``depends_on``) and
            # record progress in the task body. A ``blocked`` return that committed
            # nothing has nothing to merge and left no recorded progress, so close the
            # worktree: the still-open task re-enqueues a brand-new worktree (a fresh
            # session and a fresh model draw that may read the contract correctly)
            # rather than resuming the same confused session in place. The natural
            # ceiling on repeated respawns is the run's max-cost budget.
            _LOGGER.info(
                "worker returned blocked with no committed changes for worktree %s; "
                "closing for a fresh respawn",
                worktree.worktree_id,
            )
            await self._record_agent_message_and_transition(
                worktree.worktree_id,
                role=spec.role,
                output=output,
                state=WorktreeState.CLOSED,
                expected=spec.running_state or spec.runnable_state,
            )
            return
        # First dirty gate: run before validation so validation commands assess
        # only the worker's committed tree (uncommitted files cannot mask or
        # cause a validation result). On a blocked contribution this requeues
        # before ``blocked_non_task_feedback`` is computed below; that feedback
        # is simply regenerated on the next pass once the tree is clean.
        if (
            isinstance(output, WorkerContributionOutput)
            and next_state is WorktreeState.REVIEW
            and await self._requeue_if_worktree_dirty_before_review(
                worktree,
                role=spec.role,
                output=output,
            )
        ):
            return
        if (
            isinstance(output, WorkerContributionOutput)
            and output.status == "blocked"
            and next_state is WorktreeState.REVIEW
        ):
            blocked_non_task_feedback = await self._to_thread_call(
                lambda: _blocked_non_task_change_feedback(
                    worktree.path,
                    self.config.merge_target_branch,
                )
            )
        if (
            isinstance(output, WorkerContributionOutput)
            and next_state is WorktreeState.REVIEW
            and self.config.validation_commands
        ):
            validation_failure = await _run_validation_commands_async(
                self.config.validation_commands,
                worktree.path,
                self.config.agent_oom_score_adj,
            )
            if validation_failure is not None:
                _LOGGER.warning(
                    "validation failed for async worktree %s: command=%s exit_code=%s",
                    worktree.worktree_id,
                    _format_command(validation_failure.argv),
                    validation_failure.returncode,
                )
                await self._record_discussion_messages_and_transition(
                    worktree.worktree_id,
                    messages=(
                        (spec.role, _agent_discussion_message(output)),
                        *(
                            (
                                (
                                    AsyncOrchestratorAgentRole.ORCHESTRATOR,
                                    blocked_non_task_feedback,
                                ),
                            )
                            if blocked_non_task_feedback is not None
                            else ()
                        ),
                        (
                            AsyncOrchestratorAgentRole.ORCHESTRATOR,
                            _validation_failure_discussion_message(validation_failure),
                        ),
                    ),
                    state=WorktreeState.PENDING,
                    expected=spec.running_state or spec.runnable_state,
                )
                return
        # Second dirty gate: re-check after validation so a command's own side
        # effects (files written outside ``.tend/``) cannot reach the reviewer as
        # uncommitted changes. Only meaningful when validation actually ran.
        if (
            isinstance(output, WorkerContributionOutput)
            and next_state is WorktreeState.REVIEW
            and self.config.validation_commands
            and await self._requeue_if_worktree_dirty_before_review(
                worktree,
                role=spec.role,
                output=output,
            )
        ):
            return
        _LOGGER.info(
            "%s agent completed for worktree %s: next_state=%s",
            spec.role.value,
            worktree.worktree_id,
            next_state.value,
        )
        if blocked_non_task_feedback is not None:
            await self._record_discussion_messages_and_transition(
                worktree.worktree_id,
                messages=(
                    (spec.role, _agent_discussion_message(output)),
                    (
                        AsyncOrchestratorAgentRole.ORCHESTRATOR,
                        blocked_non_task_feedback,
                    ),
                ),
                state=next_state,
                expected=spec.running_state or spec.runnable_state,
            )
            return
        await self._record_agent_message_and_transition(
            worktree.worktree_id,
            role=spec.role,
            output=output,
            state=next_state,
            expected=spec.running_state or spec.runnable_state,
        )

    async def _requeue_if_worktree_dirty_before_review(
        self,
        worktree: AsyncOrchestratorWorktree,
        *,
        role: AsyncOrchestratorAgentRole,
        output: WorkerContributionOutput,
    ) -> bool:
        """Return a review-bound worker to PENDING if its worktree is dirty."""

        dirty_status = await self._to_thread_call(
            lambda: _worktree_dirty_status_excluding_orchestrator_metadata(worktree.path)
        )
        if not dirty_status:
            return False

        # Reviewers and validation commands must assess exactly the tree that
        # can later be merged. Since the merge path now lands only committed
        # worker changes, a dirty worktree would let uncommitted files influence
        # review/build checks and then be silently absent from the published
        # branch. Return it to the worker first so the worker either commits
        # intended changes or removes/reverts accidental ones.
        _LOGGER.info(
            "worker left uncommitted changes before review for worktree %s; "
            "returning it to the worker queue",
            worktree.worktree_id,
        )
        await self._record_discussion_messages_and_transition(
            worktree.worktree_id,
            messages=(
                (role, _agent_discussion_message(output)),
                (
                    AsyncOrchestratorAgentRole.ORCHESTRATOR,
                    _dirty_worktree_before_review_discussion_message(dirty_status),
                ),
            ),
            state=WorktreeState.PENDING,
            expected=WorktreeState.WORKER_RUNNING,
        )
        return True

    async def _process_merge_queue_forever(self) -> None:
        """Continuously merge approved worktrees into the configured target branch."""

        while True:
            await self.runtime.process_queue_item(
                self.runtime.merge_queue,
                self._merge_worktree_id,
                wait=True,
            )

    async def _process_merge_queue_once(self) -> None:
        """Process currently queued merge worktrees without waiting for more."""

        while await self.runtime.process_queue_item(
            self.runtime.merge_queue,
            self._merge_worktree_id,
            wait=False,
        ):
            pass

    async def _merge_worktree_id(self, worktree_id: str) -> None:
        """Try to merge a worktree, closing or requeueing afterward."""

        async with self.runtime.merge_lock:
            worktree = await self._store_call(lambda: self.store.get_worktree(worktree_id))
            if worktree is None or worktree.state is not WorktreeState.MERGE:
                return

            # Batched staging path: the first queued worktree to win ``merge_lock``
            # drains *all* currently-ready worktrees and validates them in one
            # staging build. Sibling queue items, dequeued later, find their
            # worktrees no longer in MERGE and no-op. Only with the staging
            # worktree (the legacy in-entrypoint path keeps one-at-a-time merges).
            if self.config.batched_merge and self.config.merge_validation_worktree:
                await self._merge_batch_via_staging(worktree_id)
                return

            entrypoint = _absolute_path(self.config.entrypoint)
            try:
                entrypoint_status = await asyncio.to_thread(
                    _git_status_porcelain,
                    entrypoint,
                )
            except subprocess.CalledProcessError as exc:
                _LOGGER.warning(
                    "entrypoint status check failed before merging async worktree %s: %s",
                    worktree.worktree_id,
                    _called_process_error_summary(exc),
                )
                await self._record_orchestrator_message_and_transition(
                    worktree.worktree_id,
                    message=_entrypoint_status_failure_discussion_message(exc),
                    state=WorktreeState.PENDING,
                )
                return
            if entrypoint_status:
                _LOGGER.warning(
                    "entrypoint is dirty before merging async worktree %s",
                    worktree.worktree_id,
                )
                await self._record_orchestrator_message_and_transition(
                    worktree.worktree_id,
                    message=_dirty_entrypoint_discussion_message(entrypoint_status),
                    state=WorktreeState.PENDING,
                )
                return

            if self.config.merge_validation_worktree:
                # Staging path: trial-merge + validate in <root>/staging and only
                # fast-forward the pristine entrypoint to a validated commit. The
                # entrypoint is never reverted. (Still under ``merge_lock`` so the
                # single staging worktree is used by one merge at a time.)
                await self._merge_worktree_via_staging(
                    worktree=worktree, entrypoint=entrypoint
                )
                return

            _LOGGER.info(
                "merging async worktree: %s target_branch=%s",
                worktree.worktree_id,
                self.config.merge_target_branch,
            )
            try:
                merge_result = await asyncio.to_thread(
                    _merge_worktree_into_target_branch,
                    entrypoint=entrypoint,
                    worktree=worktree.path,
                    commit_message=f"async orchestrator worktree {worktree.worktree_id}",
                    target_branch=self.config.merge_target_branch,
                )
            except subprocess.CalledProcessError as exc:
                _LOGGER.warning(
                    "merge failed for async worktree %s: %s",
                    worktree.worktree_id,
                    _called_process_error_summary(exc),
                )
                await self._record_orchestrator_message_and_transition(
                    worktree.worktree_id,
                    message=_merge_failure_discussion_message(
                        exc, target_branch=self.config.merge_target_branch
                    ),
                    state=WorktreeState.PENDING,
                )
                return
            if merge_result.original_head is None:
                if await self._close_worktree_if_merge_already_landed(
                    worktree,
                    target_branch=self.config.merge_target_branch,
                ):
                    return
                # The worker committed nothing, so there is nothing to merge.
                # Return it to the worker queue with an explanatory message; the
                # worktree (and any uncommitted work in it) is left untouched.
                _LOGGER.info(
                    "async worktree %s had nothing committed to merge; "
                    "returning it to the worker queue",
                    worktree.worktree_id,
                )
                await self._record_orchestrator_message_and_transition(
                    worktree.worktree_id,
                    message=_nothing_committed_discussion_message(
                        self.config.merge_target_branch
                    ),
                    state=WorktreeState.PENDING,
                )
                return
            original_head = merge_result.original_head
            # Validate the post-merge task tree before the configured
            # pre_merge_validation_commands run.
            # A merge that introduces a malformed task YAML or a dependency
            # cycle is rejected exactly like a post-merge build failure
            # (merge --abort + reset --hard original_head) and the worktree is
            # returned to PENDING with a discussion message naming the failure.
            # G6 keeps a single bad worker output from tearing down the next
            # scheduling tick; this gate keeps a bad merged contribution from
            # being accepted in the first place.
            task_validation_failure = await asyncio.to_thread(
                _check_post_merge_task_tree,
                entrypoint=entrypoint,
                original_head=original_head,
            )
            if task_validation_failure is not None:
                _LOGGER.warning(
                    "post-merge task validation failed for async worktree %s: %s",
                    worktree.worktree_id,
                    task_validation_failure.summary,
                )
                rollback_failure: subprocess.CalledProcessError | None = None
                try:
                    await asyncio.to_thread(
                        _rollback_entrypoint_to_head,
                        entrypoint,
                        original_head,
                    )
                except subprocess.CalledProcessError as exc:
                    rollback_failure = exc
                    _LOGGER.warning(
                        "rollback failed after task validation failure for "
                        "async worktree %s: %s",
                        worktree.worktree_id,
                        _called_process_error_summary(exc),
                    )
                await self._record_orchestrator_message_and_transition(
                    worktree.worktree_id,
                    message=_task_validation_failure_discussion_message(
                        task_validation_failure,
                        rollback_failure=rollback_failure,
                        original_head=original_head,
                    ),
                    state=WorktreeState.PENDING,
                )
                return
            # The expensive build gate runs only after the build-free task-tree
            # gate above passed; a merge whose diff stays under ``tasks/`` may
            # skip it when the operator opted in.
            if self.config.pre_merge_validation_commands and not (
                await self._skip_build_for_task_only_merge(entrypoint, original_head)
            ):
                validation_failure = await _run_validation_commands_async(
                    self.config.pre_merge_validation_commands,
                    entrypoint,
                    self.config.agent_oom_score_adj,
                )
                if validation_failure is not None:
                    _LOGGER.warning(
                        "pre-merge validation failed for async worktree %s: "
                        "command=%s exit_code=%s",
                        worktree.worktree_id,
                        _format_command(validation_failure.argv),
                        validation_failure.returncode,
                    )
                    rollback_failure: subprocess.CalledProcessError | None = None
                    try:
                        await asyncio.to_thread(
                            _rollback_entrypoint_to_head,
                            entrypoint,
                            original_head,
                        )
                    except subprocess.CalledProcessError as exc:
                        rollback_failure = exc
                        _LOGGER.warning(
                            "rollback failed after pre-merge validation failure for "
                            "async worktree %s: %s",
                            worktree.worktree_id,
                            _called_process_error_summary(exc),
                        )
                    await self._record_orchestrator_message_and_transition(
                        worktree.worktree_id,
                        message=_pre_merge_validation_failure_discussion_message(
                            validation_failure,
                            rollback_failure=rollback_failure,
                        ),
                        state=WorktreeState.PENDING,
                    )
                    return
            _LOGGER.info("merged async worktree: %s", worktree.worktree_id)
            await self._transition_worktree(
                worktree.worktree_id,
                WorktreeState.CLOSED,
                expected=WorktreeState.MERGE,
            )

    async def _close_worktree_if_merge_already_landed(
        self,
        worktree: AsyncOrchestratorWorktree,
        *,
        target_branch: str,
    ) -> bool:
        """Close a MERGE worktree whose commits are already on the target branch."""

        current_head = await asyncio.to_thread(_current_head, worktree.path)
        if current_head == worktree.head:
            return False
        has_unmerged_commits = await asyncio.to_thread(
            _worktree_has_unmerged_commits,
            worktree.path,
            target_branch,
        )
        if has_unmerged_commits:
            return False
        _LOGGER.info(
            "async worktree %s commits are already contained in %s; closing",
            worktree.worktree_id,
            target_branch,
        )
        await self._transition_worktree(
            worktree.worktree_id,
            WorktreeState.CLOSED,
            expected=WorktreeState.MERGE,
        )
        return True

    async def _merge_worktree_via_staging(
        self,
        *,
        worktree: AsyncOrchestratorWorktree,
        entrypoint: Path,
    ) -> None:
        """Trial-merge + validate in the staging worktree; publish on success.

        The pristine entrypoint is only ever advanced (``git merge --ff-only``)
        to a commit that already passed task-tree and ``pre_merge_validation``
        checks in the staging worktree — it is never reset/reverted. The slow
        validation build runs in staging *without* ``entrypoint_lock`` held, so
        ready-task worktree creation is not starved while it runs (only the brief
        fast-forward publish takes ``entrypoint_lock``). Called with
        ``merge_lock`` held, which serializes the single staging worktree across
        merges.
        """

        target_branch = self.config.merge_target_branch
        commit_message = f"async orchestrator worktree {worktree.worktree_id}"

        # Read the committed worker tree and the publish target's tip. A git
        # failure here bounces the worktree back to PENDING (matching the legacy
        # in-entrypoint path) rather than escaping to the merge service and
        # tearing the run down. Workers own their commits, so uncommitted work is
        # preserved in the worktree and never lands in the target branch.
        # ``target_head`` is read from ``target_branch`` (not ``HEAD``) because the
        # publish step fast-forwards ``target_branch``; staging then trial-merges
        # onto exactly the ref the entrypoint will advance, so ``--ff-only`` cannot
        # be defeated by the entrypoint sitting on a different/detached ``HEAD``.
        # ``merge_lock`` keeps ``target_branch`` from moving until we publish.
        # (Staging-worktree creation below is intentionally left unguarded: a
        # failure there is a setup/disk problem where stopping the run is correct.)
        try:
            has_unmerged_commits = await asyncio.to_thread(
                _worktree_has_unmerged_commits,
                worktree.path,
                target_branch,
            )
            if not has_unmerged_commits:
                if await self._close_worktree_if_merge_already_landed(
                    worktree,
                    target_branch=target_branch,
                ):
                    return
                _LOGGER.info(
                    "async worktree %s had nothing committed to merge; "
                    "returning it to the worker queue",
                    worktree.worktree_id,
                )
                await self._record_orchestrator_message_and_transition(
                    worktree.worktree_id,
                    message=_nothing_committed_discussion_message(target_branch),
                    state=WorktreeState.PENDING,
                )
                return
            worktree_head = await asyncio.to_thread(_current_head, worktree.path)
            target_head = await asyncio.to_thread(_branch_head, entrypoint, target_branch)
        except subprocess.CalledProcessError as exc:
            _LOGGER.warning(
                "staging merge preparation failed for async worktree %s: %s",
                worktree.worktree_id,
                _called_process_error_summary(exc),
            )
            await self._record_orchestrator_message_and_transition(
                worktree.worktree_id,
                message=_merge_failure_discussion_message(exc, target_branch=target_branch),
                state=WorktreeState.PENDING,
            )
            return

        staging = await self._ensure_validation_worktree(head=target_head)

        _LOGGER.info(
            "staging merge for async worktree: %s target_branch=%s",
            worktree.worktree_id,
            target_branch,
        )
        try:
            staging_head = await asyncio.to_thread(
                _stage_merge,
                staging,
                target_head=target_head,
                worktree_head=worktree_head,
                commit_message=commit_message,
            )
        except subprocess.CalledProcessError as exc:
            _LOGGER.warning(
                "staging merge failed for async worktree %s: %s",
                worktree.worktree_id,
                _called_process_error_summary(exc),
            )
            await self._record_orchestrator_message_and_transition(
                worktree.worktree_id,
                message=_merge_failure_discussion_message(
                    exc, target_branch=target_branch
                ),
                state=WorktreeState.PENDING,
            )
            return

        # Validate the staged tree (entrypoint untouched). On any failure we only
        # discard the staging trial merge — there is no entrypoint rollback, so
        # the failure messages carry no rollback_failure.
        task_validation_failure = await asyncio.to_thread(
            _check_post_merge_task_tree,
            entrypoint=staging,
            original_head=target_head,
        )
        if task_validation_failure is not None:
            _LOGGER.warning(
                "post-merge task validation failed (staging) for async worktree %s: %s",
                worktree.worktree_id,
                task_validation_failure.summary,
            )
            await asyncio.to_thread(_sync_staging_to_head, staging, target_head)
            await self._record_orchestrator_message_and_transition(
                worktree.worktree_id,
                message=_task_validation_failure_discussion_message(
                    task_validation_failure,
                    rollback_failure=None,
                    original_head=target_head,
                    staged=True,
                ),
                state=WorktreeState.PENDING,
            )
            return

        # The expensive build gate runs only after the build-free task-tree
        # gate above passed; a merge whose diff stays under ``tasks/`` may skip
        # it when the operator opted in.
        if self.config.pre_merge_validation_commands and not (
            await self._skip_build_for_task_only_merge(staging, target_head)
        ):
            validation_failure = await _run_validation_commands_async(
                self.config.pre_merge_validation_commands,
                staging,
                self.config.agent_oom_score_adj,
            )
            if validation_failure is not None:
                _LOGGER.warning(
                    "pre-merge validation failed (staging) for async worktree %s: "
                    "command=%s exit_code=%s",
                    worktree.worktree_id,
                    _format_command(validation_failure.argv),
                    validation_failure.returncode,
                )
                if validation_failure.crashed:
                    # The next single merge validates in this same staging
                    # worktree; do not let it trust artifacts the crashed
                    # validator left behind.
                    await self._purge_staging_after_crash(
                        staging, target_head, validation_failure.signal_number
                    )
                else:
                    await asyncio.to_thread(_sync_staging_to_head, staging, target_head)
                await self._record_orchestrator_message_and_transition(
                    worktree.worktree_id,
                    message=_pre_merge_validation_failure_discussion_message(
                        validation_failure,
                        rollback_failure=None,
                        staged=True,
                    ),
                    state=WorktreeState.PENDING,
                )
                return

        # Publish: advance the entrypoint to the validated commit. Brief window
        # under ``entrypoint_lock`` so worktree creation never mirrors a
        # half-applied fast-forward. ``--ff-only`` because the staging worktree
        # built ``staging_head`` directly on the entrypoint tip read above and
        # ``merge_lock`` keeps that tip from moving until we publish.
        try:
            async with self.runtime.entrypoint_lock:
                await asyncio.to_thread(
                    _publish_validated_head,
                    entrypoint=entrypoint,
                    target_branch=target_branch,
                    validated_head=staging_head,
                )
        except subprocess.CalledProcessError as exc:
            _LOGGER.warning(
                "publishing validated head failed for async worktree %s: %s",
                worktree.worktree_id,
                _called_process_error_summary(exc),
            )
            await asyncio.to_thread(_sync_staging_to_head, staging, target_head)
            await self._record_orchestrator_message_and_transition(
                worktree.worktree_id,
                message=_merge_failure_discussion_message(
                    exc, target_branch=target_branch
                ),
                state=WorktreeState.PENDING,
            )
            return

        _LOGGER.info("merged async worktree via staging: %s", worktree.worktree_id)
        # Staging now holds a built tree matching the just-published ``main``;
        # snapshot it so newly-created worktrees seed an incremental build.
        if self.config.seed_worktree_build:
            await self._refresh_build_cache_from_staging(staging)
        await self._transition_worktree(
            worktree.worktree_id,
            WorktreeState.CLOSED,
            expected=WorktreeState.MERGE,
        )

    async def _merge_batch_via_staging(self, first_worktree_id: str) -> None:
        """Validate queued MERGE worktrees in one staging build; publish or bisect.

        Drains the triggering worktree plus still-visible MERGE queue items in
        FIFO order, up to ``max_merge_batch_size`` when configured, reads each
        worker-owned commit, and hands the set to :meth:`_publish_or_bisect`.
        Called with ``merge_lock`` held (serializes the single staging worktree).
        The legacy in-entrypoint path and ``batched_merge=False`` never reach
        here.
        """

        entrypoint = _absolute_path(self.config.entrypoint)
        target_branch = self.config.merge_target_branch

        ordered_ids = (first_worktree_id, *self.runtime.merge_queue.items)
        seen: set[str] = set()
        members: list[AsyncOrchestratorWorktree] = []
        for worktree_id in ordered_ids:
            if worktree_id in seen:
                continue
            seen.add(worktree_id)
            target_worktree_id = worktree_id
            worktree = await self._store_call(
                lambda target_worktree_id=target_worktree_id: self.store.get_worktree(
                    target_worktree_id
                )
            )
            if worktree is not None and worktree.state is WorktreeState.MERGE:
                members.append(worktree)
        max_batch_size = self.config.max_merge_batch_size
        if max_batch_size is not None:
            members = members[:max_batch_size]
        if not members:
            return

        # The publish step fast-forwards the entrypoint, so it must be clean.
        # Checked once for the batch; on failure the whole batch bounces (as the
        # single path does per worktree).
        try:
            entrypoint_status = await asyncio.to_thread(_git_status_porcelain, entrypoint)
        except subprocess.CalledProcessError as exc:
            await self._bounce_worktrees(
                members, _entrypoint_status_failure_discussion_message(exc)
            )
            return
        if entrypoint_status:
            await self._bounce_worktrees(
                members, _dirty_entrypoint_discussion_message(entrypoint_status)
            )
            return

        try:
            target_head = await asyncio.to_thread(_branch_head, entrypoint, target_branch)
        except subprocess.CalledProcessError as exc:
            await self._bounce_worktrees(
                members, _merge_failure_discussion_message(exc, target_branch=target_branch)
            )
            return

        await self._ensure_validation_worktree(head=target_head)

        # Read each worker-owned committed tree; a prep failure bounces only that
        # worktree. Uncommitted worker files are preserved and never land in the
        # target branch, matching the single-worktree staging path.
        staged: list[tuple[AsyncOrchestratorWorktree, str]] = []
        for worktree in members:
            try:
                has_unmerged_commits = await asyncio.to_thread(
                    _worktree_has_unmerged_commits,
                    worktree.path,
                    target_branch,
                )
                if not has_unmerged_commits:
                    if await self._close_worktree_if_merge_already_landed(
                        worktree,
                        target_branch=target_branch,
                    ):
                        continue
                    await self._record_orchestrator_message_and_transition(
                        worktree.worktree_id,
                        message=_nothing_committed_discussion_message(target_branch),
                        state=WorktreeState.PENDING,
                    )
                    continue
                worktree_head = await asyncio.to_thread(_current_head, worktree.path)
            except subprocess.CalledProcessError as exc:
                await self._bounce_worktrees(
                    [worktree],
                    _merge_failure_discussion_message(exc, target_branch=target_branch),
                )
                continue
            staged.append((worktree, worktree_head))
        if not staged:
            return

        _LOGGER.info(
            "staging batch merge: %d worktree(s) target_branch=%s",
            len(staged),
            target_branch,
        )
        # One cancellation-retry budget for the whole publish/isolation episode.
        await self._publish_or_bisect(staged, target_head, _CancellationRetryBudget())

    async def _publish_or_bisect(
        self,
        members: list[tuple[AsyncOrchestratorWorktree, str]],
        base_head: str,
        retry_budget: _CancellationRetryBudget,
    ) -> str:
        """Validate ``members`` against ``base_head``; publish, bounce, or split.

        The worklist driver for batched-merge isolation. Each queued group is
        assembled and validated (:meth:`_validate_and_publish_group`) against
        the *current* head — publishing advances the head that every later
        group is validated against. A failing multi-member group is partitioned
        (:meth:`_isolation_subgroups`: attributed members probed first, else
        halving) and its subgroups pushed to the *front* of the queue, so the
        processing order is exactly the depth-first order the old recursive
        formulation used, with an O(1) Python stack — a degenerate batch where
        every member fails independently previously recursed once per bounced
        member and hit ``RecursionError`` around 500 members (round-3
        adversarial review). With ``validate_task_graph`` reporting every
        violation at once, that all-bad case now costs O(N log N) subset
        assemblies of build-free gate-1 validations (the halving tree), with no
        builds until a subgroup actually passes the task gate.

        Invariant (verify-before-bounce): the single-member failure below is
        the *sole* bounce site for validation failures, and it is reached only
        from a failed validation of exactly that member's contribution against
        the advancing main, with that failure's own message. Attribution never
        bounces anyone directly — misattribution costs one extra confirming
        validation, never a false bounce — and a survivor merges only after a
        passing build of its exact set (never "by elimination"). Returns the
        head ``main`` was advanced to (``base_head`` if nothing published).
        """

        head = base_head
        pending: deque[list[tuple[AsyncOrchestratorWorktree, str]]] = deque((members,))
        while pending:
            group = pending.popleft()
            if not group:
                continue
            result = await self._validate_and_publish_group(group, head, retry_budget)
            if isinstance(result, str):
                head = result
                continue
            if len(result.members) == 1:
                await self._bounce_worktrees([result.members[0][0]], result.message)
                continue
            subgroups = await self._isolation_subgroups(result.members, head, result.reported_paths)
            for subgroup in reversed(subgroups):
                pending.appendleft(subgroup)
        return head

    async def _validate_and_publish_group(
        self,
        members: list[tuple[AsyncOrchestratorWorktree, str]],
        base_head: str,
        retry_budget: _CancellationRetryBudget,
    ) -> str | _GroupValidationFailure:
        """Assemble ``members`` onto ``base_head``, validate, publish on success.

        Validation has two gates: a build-free task-tree check (strict YAML +
        acyclic ``depends_on`` DAG), then the configured ``lake build``. Returns
        the head ``main`` was advanced to on success (``base_head`` when nothing
        assembled, or when the final fast-forward failed and the group was
        bounced), or a :class:`_GroupValidationFailure` naming the assembled
        members, their failure message, and the attribution hints — splitting
        and bouncing are the caller's (:meth:`_publish_or_bisect`) job.
        Git-conflicting members are dropped and bounced here (only possible for
        a top-level group — a subset of a conflict-free assembly is itself
        conflict-free).
        """

        staging = await self._ensure_validation_worktree(head=base_head)
        target_branch = self.config.merge_target_branch
        entrypoint = _absolute_path(self.config.entrypoint)

        assembled, conflicts, staging_head = await asyncio.to_thread(
            _assemble_batch,
            staging,
            base_head,
            [(worktree.worktree_id, head) for (worktree, head) in members],
        )
        by_id = {worktree.worktree_id: worktree for (worktree, _) in members}
        for worktree_id, exc in conflicts:
            await self._bounce_worktrees(
                [by_id[worktree_id]],
                _merge_failure_discussion_message(exc, target_branch=target_branch),
            )
        ok = [(by_id[worktree_id], head) for (worktree_id, head) in assembled]
        if not ok:
            return base_head

        # Gate 1 — task tree (build-free). Re-checked here (not only in an up-front
        # pre-screen) so the cumulative worklist processing re-validates the task
        # graph against the *advancing* main: a cycle that forms only by
        # combining edits to pre-existing tasks is caught when a later group is
        # validated on top of an earlier group's published merge. A malformed or
        # self-cyclic task file is attributed to its worktree and bounced with no
        # build, so it never reaches the expensive gate below.
        task_failure = await asyncio.to_thread(
            _check_post_merge_task_tree, entrypoint=staging, original_head=base_head
        )
        if task_failure is not None:
            _LOGGER.warning(
                "post-merge task validation failed (staging batch): %s",
                task_failure.summary,
            )
            await asyncio.to_thread(_sync_staging_to_head, staging, base_head)
            return _GroupValidationFailure(
                members=ok,
                message=_task_validation_failure_discussion_message(
                    task_failure,
                    rollback_failure=None,
                    original_head=base_head,
                    staged=True,
                ),
                reported_paths=set(task_failure.offending_paths),
            )

        # Gate 2 — build (expensive). Only runs once the task tree is valid. A
        # batch whose combined diff stays under ``tasks/`` may skip it when the
        # operator opted in, asserting that the validation commands consume
        # nothing under ``tasks/`` — gate 1 above covers only the task graph.
        build_failure: _ValidationCommandFailure | None = None
        if not await self._skip_build_for_task_only_merge(staging, base_head):
            build_failure = await self._validate_build(staging, base_head, retry_budget)
        if build_failure is None:
            try:
                async with self.runtime.entrypoint_lock:
                    await asyncio.to_thread(
                        _publish_validated_head,
                        entrypoint=entrypoint,
                        target_branch=target_branch,
                        validated_head=staging_head,
                    )
            except subprocess.CalledProcessError as exc:
                await asyncio.to_thread(_sync_staging_to_head, staging, base_head)
                await self._bounce_worktrees(
                    [worktree for (worktree, _) in ok],
                    _merge_failure_discussion_message(exc, target_branch=target_branch),
                )
                return base_head
            for worktree, _ in ok:
                _LOGGER.info(
                    "merged async worktree via staging (batched): %s",
                    worktree.worktree_id,
                )
                await self._transition_worktree(
                    worktree.worktree_id,
                    WorktreeState.CLOSED,
                    expected=WorktreeState.MERGE,
                )
            # Staging now matches the published head; refresh the seed cache once.
            if self.config.seed_worktree_build:
                await self._refresh_build_cache_from_staging(staging)
            return staging_head

        # A failing build is attributed by the .lean files Lean reported errors
        # in; a timed-out build reports no error paths, so fall back to the
        # heuristic timeout attribution (issue #133).
        reported_paths = _failed_lean_files(build_failure)
        if not reported_paths and build_failure.timed_out:
            reported_paths = await self._timed_out_reported_paths(build_failure, ok, base_head)
        return _GroupValidationFailure(
            members=ok,
            message=_pre_merge_validation_failure_discussion_message(
                build_failure, rollback_failure=None, staged=True
            ),
            reported_paths=reported_paths,
        )

    async def _skip_build_for_task_only_merge(self, repo: Path, original_head: str) -> bool:
        """Whether this merge may skip the expensive build gate as task-only.

        True only when the operator opted in
        (``skip_build_validation_for_task_only_merges``), a build gate is
        actually configured, and every path the merge changed
        (``original_head..HEAD`` in ``repo``) is under the task directory.
        Callers consult this only after the build-free task-tree gate passed,
        so the task-tree gate has already validated the task files
        themselves. The skip is sound only under the operator's assertion (made
        by enabling the option) that the validation commands consume nothing
        under ``tasks/``. Logs the skip at INFO so merge logs show why no build
        ran.
        """

        if not self.config.skip_build_validation_for_task_only_merges:
            return False
        if not self.config.pre_merge_validation_commands:
            return False
        task_only = await asyncio.to_thread(
            _merge_changed_only_task_paths,
            entrypoint=repo,
            original_head=original_head,
        )
        if task_only:
            _LOGGER.info(
                "skipping pre-merge build validation: merge diff changed only "
                "paths under %s/ and skip_build_validation_for_task_only_merges "
                "is enabled (task-tree gate already passed)",
                TASKS_DIRECTORY_NAME,
            )
        return task_only

    async def _validate_build(
        self,
        staging: Path,
        base_head: str,
        retry_budget: _CancellationRetryBudget,
    ) -> _ValidationCommandFailure | None:
        """Run only the (expensive) pre-merge build in ``staging``.

        Task-tree validity is enforced by gate 1 in :meth:`_publish_or_bisect`
        before this runs, so this only runs the configured ``lake build`` and, on
        failure, resets staging to ``base_head`` and returns the captured failure
        (whose output the caller mines to attribute the break to a specific
        worktree). Returns ``None`` when the build passes or no validation command
        is configured.

        A cancelled validation (killed by a cancellation signal outside the
        timeout path; see ``_VALIDATION_CANCELLATION_SIGNALS``) says nothing
        about batch validity, so it is not booked as a batch failure — that
        would send :meth:`_publish_or_bisect` into an expensive bisection of a
        healthy batch (issue #132). Instead the batch validation is retried in
        place (staging still holds the assembled batch), restarting from the
        first command because later commands may depend on earlier ones.
        ``retry_budget`` is shared across the whole publish/isolation episode
        and grants each command index at most one cancellation retry; once a
        command's retry is spent, a further cancellation falls through, with a
        warning, to the ordinary failure handling so a persistent external
        killer cannot multiply killed validations through the bisection tree.

        A crash-signal failure additionally purges gitignored build output
        from staging and re-provisions the mirror/setup infrastructure
        (:meth:`_purge_staging_after_crash`): the validator's
        incremental-correctness contract cannot be assumed to hold across its
        own crash, so later validations (e.g. bisection halves) must rebuild
        cold rather than trust possibly corrupt artifacts. Cancellation
        retries deliberately do *not* purge — see the interruption-safety
        contract on ``pre_merge_validation_commands``.
        """

        if not self.config.pre_merge_validation_commands:
            return None
        failure: _ValidationCommandFailure | None = None
        while True:
            failure = await _run_validation_commands_async(
                self.config.pre_merge_validation_commands,
                staging,
                self.config.agent_oom_score_adj,
            )
            if failure is None or not failure.cancelled:
                break
            if not retry_budget.try_consume(failure.command_index):
                break
            _LOGGER.info(
                "pre-merge validation cancelled (signal %s) (staging batch): "
                "command=%s; retrying batch validation from the start",
                failure.signal_number,
                _format_command(failure.argv),
            )
        if failure is None:
            return None
        if failure.cancelled:
            _LOGGER.warning(
                "pre-merge validation cancelled (signal %s) with its cancellation "
                "retry budget exhausted (staging batch): command=%s; treating as "
                "a validation failure",
                failure.signal_number,
                _format_command(failure.argv),
            )
        else:
            _LOGGER.warning(
                "pre-merge validation failed (staging batch): command=%s exit_code=%s",
                _format_command(failure.argv),
                failure.returncode,
            )
        if failure.crashed:
            await self._purge_staging_after_crash(staging, base_head, failure.signal_number)
        else:
            await asyncio.to_thread(_sync_staging_to_head, staging, base_head)
        return failure

    async def _isolation_subgroups(
        self,
        members: list[tuple[AsyncOrchestratorWorktree, str]],
        base_head: str,
        reported_paths: set[str],
    ) -> tuple[
        list[tuple[AsyncOrchestratorWorktree, str]],
        list[tuple[AsyncOrchestratorWorktree, str]],
    ]:
        """Partition a failed multi-member group for the isolation worklist.

        ``reported_paths`` (a task file for the task gate — including every
        file on a dependency cycle or complete-depends-on-open path — a Lean
        error path for the build gate, or a heuristically suspect module for a
        timed-out build) is heuristic and can implicate the wrong member, so it
        only selects the partition and its probe order — it never bounces:

        - when it implicates a proper, non-empty subset, that attributed subset
          is probed FIRST (attributed-first preserves FIFO blame when the
          attributed members precede the rest and are actually innocent): if it
          passes on its own it publishes, and the rest is validated against the
          advanced head; if it fails, it is partitioned again at its turn.
          Misattribution therefore costs one extra confirming validation, never
          a false bounce — a solo-validated innocent publishes;
        - when attribution is inconclusive — ``reported_paths`` empty (e.g. a
          build failure whose output names no file) or every member implicated —
          the group is simply halved, and the later half is re-validated (task
          *and* build) against whatever head the first half published.
        """

        culprits = await asyncio.to_thread(
            self._members_touching, members, reported_paths, base_head
        )
        if culprits and len(culprits) < len(members):
            _LOGGER.info(
                "batched merge: failure attributed to %d/%d worktree(s); "
                "validating the attributed member(s) alone before the rest",
                len(culprits),
                len(members),
            )
            culprit_ids = {worktree.worktree_id for (worktree, _) in culprits}
            rest = [m for m in members if m[0].worktree_id not in culprit_ids]
            return culprits, rest
        _LOGGER.info(
            "batched merge: failure not attributable to a single worktree; "
            "halving %d worktree(s) to isolate",
            len(members),
        )
        mid = len(members) // 2
        return members[:mid], members[mid:]

    def _members_touching(
        self,
        members: list[tuple[AsyncOrchestratorWorktree, str]],
        reported_paths: set[str],
        base_head: str,
    ) -> list[tuple[AsyncOrchestratorWorktree, str]]:
        """Members whose own commits changed a file named in ``reported_paths``.

        Attributes a validation failure (a task-file path, or a Lean error path)
        to the worktree(s) most likely responsible so isolation can probe them
        first instead of blind halving — the result is a hint, never a verdict
        (only a failed solo validation bounces). Returns ``[]`` when nothing
        matches (the caller then falls back to halving). Synchronous git diffs —
        call via ``to_thread``.
        """

        if not reported_paths:
            return []
        entrypoint = _absolute_path(self.config.entrypoint)
        out: list[tuple[AsyncOrchestratorWorktree, str]] = []
        for worktree, head in members:
            touched = _worktree_touched_files(entrypoint, base_head, head)
            if any(
                _paths_match(touched_path, reported)
                for touched_path in touched
                for reported in reported_paths
            ):
                out.append((worktree, head))
        return out

    async def _timed_out_reported_paths(
        self,
        failure: _ValidationCommandFailure,
        members: list[tuple[AsyncOrchestratorWorktree, str]],
        base_head: str,
    ) -> set[str]:
        """Heuristic attribution paths for a timed-out batch build (issue #133).

        A timed-out build names no error file, which used to mean worst-case
        bisection — every halving round re-times-out because the slow module is
        still in every subset containing its author. Instead, mine the partial
        lake output buffered at kill time:

        - modules lake explicitly reported in flight (a ``Building`` header with
          no completion line — the killed jobs) are reported directly;
        - otherwise, when lake reported completed modules, report the members'
          touched ``.lean`` files *minus* those completed modules: a member all
          of whose modules finished building is deprioritized, the rest are
          probed first.

        Only applied when the timed-out command's executable is lake
        (:func:`_is_lake_invocation`): ``pre_merge_validation_commands`` are
        arbitrary, and a non-lake validator that happens to print lake-shaped
        ``Built <Module>`` lines must not activate completed-module subtraction
        and probe the wrong member first. Either way the result is only a probe
        order: ``Built`` does not prove innocence (a module can compile fine
        while its change hangs a downstream module), so the caller validates the
        implicated members alone before anyone is bounced — misattribution costs
        one confirming build, not a false bounce. No-signal (or non-lake) output
        returns an empty set and the caller falls back to halving as before.
        """

        if not _is_lake_invocation(failure.argv):
            return set()
        in_flight, completed = _lake_module_progress(failure)
        if in_flight:
            return in_flight
        if not completed:
            return set()

        def touched_lean_files() -> set[str]:
            entrypoint = _absolute_path(self.config.entrypoint)
            touched: set[str] = set()
            for _, head in members:
                touched |= _worktree_touched_files(entrypoint, base_head, head)
            return {path for path in touched if path.endswith(".lean")}

        touched = await asyncio.to_thread(touched_lean_files)
        return {
            path
            for path in touched
            if not any(_paths_match(path, done) for done in completed)
        }

    async def _bounce_worktrees(
        self, worktrees: list[AsyncOrchestratorWorktree], message: str
    ) -> None:
        """Return each worktree to PENDING with ``message`` (re-batched next round)."""

        for worktree in worktrees:
            await self._record_orchestrator_message_and_transition(
                worktree.worktree_id, message=message, state=WorktreeState.PENDING
            )

    async def _ensure_validation_worktree(self, *, head: str) -> Path:
        """Create or reuse the long-lived ``<root>/staging`` validation worktree.

        Created lazily on first merge and reused for the rest of the run (and on
        resume, when the directory already exists). It is mirrored and set up
        exactly like a task worktree so its ``.lake`` build cache stays warm
        across merges, keeping each validation build incremental. ``head`` is
        the merge-target branch tip that staging will trial-merge onto; using it
        here keeps staging setup correct even when the entrypoint checkout's
        ``HEAD`` is detached or points at another branch. It is *not* registered
        as an orchestrator worktree in durable state.

        ``<root>/staging`` is orchestrator-private. ``run_cli`` holds the root's
        durable exclusive advisory lock for the entire ``orchestrator.run()``;
        external registration at this path despite that invariant is unsupported.
        Even so, no existing object is reset or cleaned until ``lstat`` proves it
        is a real directory and Git metadata proves it is this entrypoint's
        registered staging worktree. Anything else is rename-quarantined without
        following it, then replaced with a fresh worktree.
        """

        entrypoint = _absolute_path(self.config.entrypoint)
        staging = _absolute_path(self.config.root) / _VALIDATION_WORKTREE_DIRNAME
        sentinel = _validation_worktree_provisioning_sentinel(staging)
        async with self.runtime.worktree_creation_lock:
            existing = await self._to_thread_call(
                lambda: _prepare_validation_worktree_path(entrypoint, staging)
            )
            if (
                self._validation_worktree_ready
                and existing
                and _provisioning_sentinel_is_regular(sentinel)
            ):
                return staging
            self._validation_worktree_ready = False
            if existing and _provisioning_sentinel_is_regular(sentinel):
                # Resume / reuse only when durable provisioning state agrees:
                # realign staging while preserving its warm ignored cache.
                await self._to_thread_call(lambda: _sync_staging_to_head(staging, head))
                self._validation_worktree_ready = True
            else:
                if existing:
                    _LOGGER.warning(
                        "re-provisioning async staging worktree without provisioning "
                        "sentinel: %s",
                        staging,
                    )
                else:
                    _LOGGER.info("creating async staging validation worktree: %s", staging)
                await self._provision_validation_worktree(
                    staging=staging,
                    head=head,
                    existing=existing,
                )
        return staging

    async def _record_agent_message_and_transition(
        self,
        worktree_id: str,
        *,
        role: AsyncOrchestratorAgentRole,
        output: BaseModel,
        state: WorktreeState,
        expected: WorktreeState,
    ) -> None:
        """Append an agent message, transition the worktree, and refresh the log file.

        A reviewer ``review_verdict`` is also persisted in full (structured) so the
        per-comment breakdown that the discussion message flattens away is retained.
        """

        await self._record_discussion_messages_and_transition(
            worktree_id,
            messages=((role, _agent_discussion_message(output)),),
            state=state,
            expected=expected,
            review_verdict=output if isinstance(output, ReviewVerdictOutput) else None,
        )

    async def _record_orchestrator_message_and_transition(
        self,
        worktree_id: str,
        *,
        message: str,
        state: WorktreeState,
        expected: WorktreeState = WorktreeState.MERGE,
    ) -> None:
        """Append an orchestrator message, transition, and refresh the log file."""

        await self._record_discussion_messages_and_transition(
            worktree_id,
            messages=((AsyncOrchestratorAgentRole.ORCHESTRATOR, message),),
            state=state,
            expected=expected,
        )

    async def _record_discussion_messages_and_transition(
        self,
        worktree_id: str,
        *,
        messages: Sequence[tuple[AsyncOrchestratorAgentRole, str]],
        state: WorktreeState,
        expected: WorktreeState,
        review_verdict: ReviewVerdictOutput | None = None,
    ) -> None:
        """Append discussion messages, transition, and refresh the log file atomically."""

        # Git output is decoded with surrogateescape so raw paths remain
        # byte-exact for in-memory matching and attribution. Discussion text is
        # the persistence boundary: escape only lone surrogates before Pydantic,
        # SQLite, and UTF-8 log writes see it. Ordinary backslashes (including an
        # already-rendered ``\\udcff``) remain unchanged.
        persistable_messages = tuple(
            (role, _utf8_persistable_text(message)) for role, message in messages
        )
        worktree = await self._store_call(
            lambda: self.store.record_worktree_transition(
                worktree_id,
                expected=expected,
                new=state,
                discussion_messages=persistable_messages,
                review_verdict=review_verdict,
            )
        )
        if worktree is None:
            return
        self.runtime.discard_worktree_id(worktree_id)
        self.runtime.enqueue_worktree_for_state(worktree)
        await self._to_thread_call(lambda: write_discussion_log_file(worktree))
        if review_verdict is not None:
            await self._to_thread_call(
                lambda: write_review_verdict_artifact(
                    worktree,
                    review_verdict,
                    index=len(worktree.review_verdicts),
                )
            )
        await self._cleanup_closed_worktree_if_configured(worktree)

    async def _transition_worktree(
        self,
        worktree_id: str,
        state: WorktreeState,
        *,
        expected: WorktreeState | None = None,
    ) -> None:
        """Move an existing worktree to ``state`` and enqueue follow-up work."""

        worktree = await self._store_call(lambda: self.store.get_worktree(worktree_id))
        if worktree is None:
            return
        expected_state = worktree.state if expected is None else expected
        changed = await self._store_call(
            lambda: self.store.set_worktree_state(
                worktree_id,
                expected=expected_state,
                new=state,
            )
        )
        if not changed:
            return
        updated = await self._store_call(lambda: self.store.get_worktree(worktree_id))
        if updated is None:
            return
        self.runtime.discard_worktree_id(worktree_id)
        self.runtime.enqueue_worktree_for_state(updated)
        await self._cleanup_closed_worktree_if_configured(updated)

    async def _cleanup_closed_worktree_if_configured(
        self,
        worktree: AsyncOrchestratorWorktree,
    ) -> None:
        """Remove a CLOSED worktree when configured, after preserving unsafe cases."""

        if (
            worktree.state is not WorktreeState.CLOSED
            or not self.config.cleanup_closed_worktrees
        ):
            return
        await self._remove_closed_worktree(worktree)

    async def _remove_closed_worktree(self, worktree: AsyncOrchestratorWorktree) -> None:
        """Reclaim disk by removing a safe CLOSED worktree's tree (best-effort).

        A CLOSED worktree is normally either already published to the merge target
        or intentionally abandoned because it produced no commits. Before deleting
        the working tree, still verify that there are no non-``.tend`` uncommitted
        changes and no commits absent from the merge target; if either check finds
        local-only work, skip cleanup so the data remains available for inspection.
        Held under ``worktree_creation_lock`` so it serialises with ``git worktree
        add`` / ``prune``, runs off the event loop, and swallows every error so
        cleanup can never disrupt the run.
        """

        entrypoint = _absolute_path(self.config.entrypoint)
        async with self.runtime.worktree_creation_lock:
            await self._to_thread_call(
                lambda: _remove_worktree_tree(
                    entrypoint,
                    worktree.path,
                    worktree_id=worktree.worktree_id,
                    target_branch=self.config.merge_target_branch,
                )
            )

    async def _create_fresh_worktree(
        self,
        *,
        task: Task | str | None = None,
        name: str | None = None,
    ) -> AsyncOrchestratorWorktree:
        """Create a fresh detached git worktree for the entrypoint repository."""

        async with self._entrypoint_guard_lock:
            async with self.runtime.worktree_creation_lock:
                return await self._create_fresh_worktree_locked(task=task, name=name)

    async def _provision_worktree_tree(
        self,
        *,
        worktree_path: Path,
        head: str,
        label: str,
        seed_build: bool = False,
    ) -> None:
        """Add a detached worktree at ``head`` and apply the shared mirror + setup.

        Shared by ready-task worktree creation (``_create_fresh_worktree_locked``)
        and the staging validation worktree (``_ensure_validation_worktree``) so
        their symlink / mirror / setup-command behavior cannot drift: both get the
        same ``workspace_mirror`` config (including ``symlink_paths``) and the same
        ``worktree_setup_command``. ``label`` identifies the worktree in log and
        cleanup messages. On any failure the partially created worktree is cleaned
        up and the error re-raised.
        """

        entrypoint = _absolute_path(self.config.entrypoint)
        existed_before = await self._to_thread_call(
            lambda: _worktree_exists_or_is_registered(entrypoint, worktree_path)
        )
        succeeded = False
        # Record Git success inside the non-cancellable worker, not after its
        # await: cancellation can arrive after `git worktree add` returns but
        # before this coroutine resumes.
        add_succeeded = threading.Event()
        reclaimed_stale_registration = threading.Event()

        def _add_worktree() -> None:
            try:
                _add_detached_worktree(
                    entrypoint,
                    path=worktree_path,
                    head=head,
                )
            except subprocess.CalledProcessError:
                # The whole-run root lock makes a missing registration at an
                # orchestrator-owned path ours. Reclaim an interrupted prior add
                # and retry once; never remove a checkout that appeared meanwhile.
                if (
                    not existed_before
                    or worktree_path.exists()
                    or worktree_path.is_symlink()
                    or not _worktree_is_registered(entrypoint, worktree_path)
                ):
                    raise
                _reclaim_missing_worktree_registration(entrypoint, worktree_path)
                reclaimed_stale_registration.set()
                try:
                    _add_detached_worktree(
                        entrypoint,
                        path=worktree_path,
                        head=head,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        "failed to add async worktree after reclaiming stale Git "
                        f"registration: {worktree_path}"
                    ) from exc
            add_succeeded.set()

        try:
            await self._to_thread_call(_add_worktree)
            await self._to_thread_call(
                lambda: _exclude_orchestrator_metadata_from_git_status(worktree_path)
            )
            await self._apply_worktree_mirror_and_setup(
                worktree_path=worktree_path,
                label=label,
                seed_build=seed_build,
            )
            succeeded = True
        finally:
            if not succeeded:
                # Re-check observable ownership in the cleanup worker. This one
                # blocking call completes even under re-entrant cancellation.
                await self._to_thread_call(
                    lambda: _cleanup_worktree_created_by_provisioning_attempt(
                        entrypoint,
                        worktree_path,
                        worktree_id=label,
                        existed_before=(
                            existed_before and not reclaimed_stale_registration.is_set()
                        ),
                        add_succeeded=add_succeeded.is_set(),
                    )
                )

    async def _apply_worktree_mirror_and_setup(
        self,
        *,
        worktree_path: Path,
        label: str,
        seed_build: bool = False,
    ) -> None:
        """Apply the shared mirror, optional build seed, and setup command.

        The provisioning tail shared by fresh worktree creation
        (:meth:`_provision_worktree_tree`) and the post-crash staging purge
        (:meth:`_purge_staging_after_crash`), so re-provisioned staging cannot
        drift from a freshly created worktree.
        """

        entrypoint = _absolute_path(self.config.entrypoint)
        if self.config.workspace_mirror.enabled:
            _LOGGER.info("mirroring entrypoint workspace into async worktree: %s", label)
            mirror_config = self.config.workspace_mirror.to_workspace_mirror_config()
            await self._to_thread_call(
                lambda: mirror_workspace(
                    entrypoint,
                    worktree_path,
                    config=mirror_config,
                )
            )
        if seed_build and self.config.seed_worktree_build:
            await self._seed_worktree_build_from_cache(worktree_path, label=label)
        setup_command = self.config.worktree_setup_command
        if setup_command is not None:
            _LOGGER.info("running async worktree setup command: %s", label)
            await self._run_cancellable_worktree_setup_command(
                setup_command,
                entrypoint=entrypoint,
                worktree=worktree_path,
            )

    async def _provision_validation_worktree(
        self,
        *,
        staging: Path,
        head: str,
        existing: bool,
    ) -> None:
        """Fully provision staging and durably mark it reusable.

        Every non-success exit, including cancellation, leaves no provisioning
        sentinel, clears the in-memory ready flag, and removes a registered
        staging worktree. An unexpected object is instead rename-quarantined.
        On POSIX, setup cancellation terminates its process group; on Windows it
        terminates only the direct child. If process startup is unresponsive, the
        daemon worker is abandoned after bounded settling before cleanup starts.
        """

        entrypoint = _absolute_path(self.config.entrypoint)
        if existing:
            existing = await self._to_thread_call(
                lambda: _prepare_validation_worktree_path(entrypoint, staging)
            )
        sentinel = _validation_worktree_provisioning_sentinel(staging)
        succeeded = False
        created_here = False
        self._validation_worktree_ready = False
        try:
            sentinel.unlink(missing_ok=True)
            if existing:
                await self._to_thread_call(
                    lambda: _sync_staging_to_head_purging_ignored(staging, head)
                )
                await self._apply_worktree_mirror_and_setup(
                    worktree_path=staging,
                    label=_VALIDATION_WORKTREE_DIRNAME,
                )
            else:
                # Same provisioning (mirror + symlink_paths + setup) as a task
                # worktree, so staging's build environment matches exactly.
                await self._provision_worktree_tree(
                    worktree_path=staging,
                    head=head,
                    label=_VALIDATION_WORKTREE_DIRNAME,
                )
                created_here = True
            await self._to_thread_call(lambda: _write_provisioning_sentinel(sentinel))
            succeeded = True
            self._validation_worktree_ready = True
        finally:
            if not succeeded:
                self._validation_worktree_ready = False
                try:
                    sentinel.unlink(missing_ok=True)
                finally:
                    if existing or created_here:
                        await self._to_thread_call(
                            lambda: _cleanup_failed_worktree_creation(
                                entrypoint,
                                staging,
                                worktree_id=_VALIDATION_WORKTREE_DIRNAME,
                            )
                        )

    async def _purge_staging_after_crash(
        self,
        staging: Path,
        head: str,
        signal_number: int | None,
    ) -> None:
        """Purge crash-tainted ignored state from staging, then re-provision it.

        A crashed validator's leftover artifacts cannot be trusted, but
        ``git clean -ffdx`` also removes staging's provisioned infrastructure.
        Clear the external durable provisioning sentinel first, re-apply the
        mirror and setup command, then restore it only after full success,
        so post-purge staging is indistinguishable from a freshly provisioned
        worktree. Any failure or cancellation clears readiness and attempts to
        remove staging; if removal fails, the absent sentinel forces a full
        re-provision on the next ensure, including after process restart.
        """

        sentinel = _validation_worktree_provisioning_sentinel(staging)
        # This must precede the creation lock and all slow/logging work. If the
        # process is interrupted while waiting, resume cannot warm-sync tainted
        # crash artifacts and must instead perform a full cold reprovision.
        sentinel.unlink(missing_ok=True)
        self._validation_worktree_ready = False
        _LOGGER.warning(
            "purging gitignored build state from staging after validator "
            "crash (signal %s): the next validation there rebuilds cold",
            signal_number,
        )
        async with self.runtime.worktree_creation_lock:
            await self._provision_validation_worktree(
                staging=staging,
                head=head,
                existing=True,
            )

    def _build_cache_dir(self) -> Path:
        """Absolute path of the ``<root>/.build-cache`` Lean build snapshot."""

        return _absolute_path(self.config.root) / _BUILD_CACHE_DIRNAME

    async def _refresh_build_cache_from_staging(self, staging: Path) -> None:
        """Snapshot the staging worktree's ``.lake/build`` into ``<root>/.build-cache``.

        Called after a successful validated merge, when the staging worktree's
        build is consistent and matches the just-published ``main``. Best-effort:
        a snapshot failure is logged and swallowed — it must never break the merge
        path (a stale/absent cache only means the next worktree builds from
        scratch). Serialized against seed reads by ``_build_cache_lock``.
        """

        source = staging / _LAKE_BUILD_RELPATH
        if not source.is_dir():
            return
        cache = self._build_cache_dir()
        async with self._build_cache_lock:
            try:
                await asyncio.to_thread(_replace_dir_copy, source, cache / "build")
            except Exception as exc:  # noqa: BLE001 - best-effort cache refresh
                _LOGGER.warning("failed to refresh worktree build cache from staging: %s", exc)

    async def _seed_worktree_build_from_cache(self, worktree_path: Path, *, label: str) -> None:
        """Copy the ``<root>/.build-cache`` snapshot into a new worktree's ``.lake/build``.

        Makes the worker's first ``lake build`` incremental (recompile only its
        edits + dependents) instead of from scratch. Best-effort: if the cache is
        absent (no successful merge yet) or the copy fails, the worktree simply
        builds from scratch. Serialized against the refresh by ``_build_cache_lock``.
        """

        cached_build = self._build_cache_dir() / "build"
        if not cached_build.is_dir():
            return
        destination = worktree_path / _LAKE_BUILD_RELPATH
        async with self._build_cache_lock:
            try:
                await asyncio.to_thread(_seed_build_dir_copy, cached_build, destination)
                _LOGGER.info("seeded worktree build cache into async worktree: %s", label)
            except Exception as exc:  # noqa: BLE001 - best-effort build seeding
                _LOGGER.warning("failed to seed build cache into worktree %s: %s", label, exc)

    async def _create_fresh_worktree_locked(
        self,
        *,
        task: Task | str | None = None,
        name: str | None = None,
        attach_task_only_if_ready: bool = False,
        admission_check: Callable[[], bool] | None = None,
    ) -> AsyncOrchestratorWorktree:
        """Create a fresh worktree while the worktree creation lock is held."""

        task_id = _task_id(task)
        root = _absolute_path(self.config.root)
        entrypoint = _absolute_path(self.config.entrypoint)
        worktrees_dir = root / "worktrees"

        if admission_check is not None and not admission_check():
            raise _WorktreeCreationAdmissionClosed
        worktrees_dir.mkdir(parents=True, exist_ok=True)
        task_manager = await self._store_call(self.store.load_task_snapshot)
        if task_id is not None and task_id not in task_manager.task_ids:
            raise ValueError(f"worktree references unknown task id: {task_id}")

        sequence = await self._store_call(self.store.next_worktree_sequence)
        if name is not None:
            validate_worktree_name(name)
            worktree_path = worktrees_dir / name
            if worktree_path.exists():
                raise FileExistsError(f"worktree path already exists: {worktree_path}")
            while True:
                worktree_id = format_sequence_id("worktree", sequence)
                target_worktree_id = worktree_id
                existing = await self._store_call(
                    lambda target_worktree_id=target_worktree_id: self.store.get_worktree(
                        target_worktree_id
                    )
                )
                if existing is None:
                    break
                sequence += 1
            worktree_name = name
        else:
            while True:
                worktree_id = format_sequence_id("worktree", sequence)
                worktree_name = worktree_id
                worktree_path = worktrees_dir / worktree_name
                target_worktree_id = worktree_id
                existing = await self._store_call(
                    lambda target_worktree_id=target_worktree_id: self.store.get_worktree(
                        target_worktree_id
                    )
                )
                if existing is None and not worktree_path.exists():
                    break
                sequence += 1

        validate_worktree_name(worktree_name)
        head = await asyncio.to_thread(_current_head, entrypoint)
        _LOGGER.info("creating async worktree: id=%s path=%s", worktree_id, worktree_path)
        await self._provision_worktree_tree(
            worktree_path=worktree_path, head=head, label=worktree_id, seed_build=True
        )

        current_task_id = await self._current_attachable_task_id(
            task_id,
            require_ready=attach_task_only_if_ready,
        )
        try:
            allocated_id = await self._store_call(
                lambda: self.store.allocate_worktree(
                    task_id=current_task_id,
                    path=worktree_path,
                    head=head,
                    worktree_id=worktree_id,
                )
            )
        except Exception:
            await asyncio.to_thread(
                _cleanup_failed_worktree_creation,
                entrypoint,
                worktree_path,
                worktree_id=worktree_id,
            )
            raise
        worktree = await self._store_call(lambda: self.store.get_worktree(allocated_id))
        if worktree is None:
            raise RuntimeError(f"allocated async worktree disappeared: {allocated_id}")
        self.runtime.enqueue_worktree_for_state(worktree)
        _LOGGER.info("created async worktree: id=%s task_id=%s", allocated_id, current_task_id)
        return worktree

    async def _current_attachable_task_id(
        self,
        task_id: str | None,
        *,
        require_ready: bool,
    ) -> str | None:
        if task_id is None:
            return None
        task_manager = await self._store_call(self.store.load_task_snapshot)
        if require_ready:
            ready_task_ids = {task.id for task in task_manager.ready_tasks()}
            return task_id if task_id in ready_task_ids else None
        return task_id if task_id in task_manager.task_ids else None


def _healthy_resume_worktrees_for_runtime(
    worktrees: Iterable[AsyncOrchestratorWorktree],
) -> tuple[AsyncOrchestratorWorktree, ...]:
    """Return worktrees that are safe to enqueue when rebuilding runtime queues."""

    healthy_worktrees: list[AsyncOrchestratorWorktree] = []
    for worktree in worktrees:
        if worktree.state is WorktreeState.CLOSED:
            healthy_worktrees.append(worktree)
            continue
        health_issue = _resume_worktree_health_issue(worktree.path)
        if health_issue is None:
            healthy_worktrees.append(worktree)
            continue
        _LOGGER.warning(
            "async resume health warning: worktree %s (%s) at %s is not resumable: %s; "
            "leaving persisted state unchanged and not queuing this worktree for this run",
            worktree.worktree_id,
            worktree.state.value,
            worktree.path,
            health_issue,
        )
    return tuple(healthy_worktrees)


def _resume_worktree_health_issue(path: Path) -> str | None:
    """Return a reason if ``path`` is missing or does not look like a git worktree."""

    try:
        if not path.exists():
            return "path does not exist"
        if not path.is_dir():
            return "path is not a directory"
        if not (path / ".git").exists():
            return "path is missing .git metadata"
    except OSError as exc:
        return f"could not inspect path: {exc}"

    try:
        inside = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--is-inside-work-tree"],
            check=False,
            capture_output=True,
            text=True,
            errors="replace",
        )
    except OSError as exc:
        return f"could not run git health check: {exc}"
    if inside.returncode != 0:
        output = _process_output_text(inside.stderr) or _process_output_text(inside.stdout)
        if output:
            return f"git worktree check failed: {_trim_text(output, max_length=500)}"
        return f"git worktree check exited with code {inside.returncode}"
    if inside.stdout.strip() != "true":
        return "git does not report this path inside a worktree"

    try:
        top_level = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
            check=False,
            capture_output=True,
            text=True,
            errors="replace",
        )
    except OSError as exc:
        return f"could not run git top-level check: {exc}"
    if top_level.returncode != 0:
        output = _process_output_text(top_level.stderr) or _process_output_text(
            top_level.stdout
        )
        if output:
            return f"git top-level check failed: {_trim_text(output, max_length=500)}"
        return f"git top-level check exited with code {top_level.returncode}"
    top_level_path = _process_output_text(top_level.stdout)
    if not top_level_path:
        return "git top-level check returned an empty path"
    try:
        if Path(top_level_path).resolve() != path.resolve():
            return f"git top-level is {top_level_path}"
    except OSError as exc:
        return f"could not resolve git top-level: {exc}"
    return None


def _parse_agent_output[OutputT: BaseModel](
    stdout: bytes | str, output_type: type[OutputT]
) -> OutputT:
    """Parse an agent's structured output from its stdout.

    Accepts either the bare output object (the schema's fields directly) or an
    ``tend-agent`` ``final_result`` envelope. With an ``output: {tool_name: final_result}``
    agent config, ``tend-agent`` writes the validated tool payload to stdout — as the bare
    object (default) or wrapped in a ``TurnResult`` (under ``--json``). Either way the
    model's free-text reasoning never reaches stdout, so chatty models cannot corrupt
    the contract. Raises ``ValueError`` (caught by the caller) when neither form
    validates.
    """

    text = _stdout_text(stdout)
    try:
        return output_type.model_validate_json(text)
    except ValueError:
        pass
    return output_type.model_validate(_final_result_payload(text))


def _final_result_payload(text: str) -> object:
    """Extract the ``final_result.output`` payload from an tend-agent TurnResult."""

    try:
        data: object = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"agent output is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("agent output JSON is not an object")
    final_result: object = cast("Mapping[str, object]", data).get("final_result")
    if not isinstance(final_result, dict):
        raise ValueError("agent output has no final_result envelope")
    payload: object = cast("Mapping[str, object]", final_result).get("output")
    if not isinstance(payload, Mapping):
        raise ValueError("agent final_result output is missing or not an object")
    return cast("Mapping[str, object]", payload)


def _agent_success_state(output: BaseModel) -> WorktreeState:
    if isinstance(output, WorkerContributionOutput):
        # The shared ``worker_contribution`` contract reports a self-assessed status.
        # ``completed`` / ``needs_review`` / ``blocked`` all send the worktree on to
        # review and (on approval) merge. ``blocked`` means "I could not finish, but I
        # committed task-graph edits (e.g. a prerequisite task plus my own
        # ``depends_on``) and recorded my progress in the task body — merge that and
        # leave my task ``open`` so I get rescheduled once unblocked". A ``blocked``
        # contribution that committed *nothing* is instead closed for a fresh respawn,
        # handled in ``_run_agent_for_worktree_id`` (which has the worktree path needed
        # to check for commits).
        return WorktreeState.REVIEW
    if isinstance(output, ReviewVerdictOutput):
        # The shared ``review_verdict`` contract has no ``deny`` (matching the sync
        # reviewer): ``approve`` merges, ``request_changes`` returns to the worker.
        if output.verdict == "approve":
            return WorktreeState.MERGE
        return WorktreeState.PENDING
    raise TypeError(f"unsupported async orchestrator agent output: {type(output).__name__}")


def _agent_discussion_message(output: BaseModel) -> str:
    """Return the discussion-log message for an agent's structured output.

    Workers report the sync ``worker_contribution`` contract, whose ``summary``
    (plus any ``notes``) becomes the discussion-log message; reviewers report the sync
    ``review_verdict`` contract, whose ``notes`` (plus ``feedback_text`` on
    ``request_changes``) become the discussion-log message.
    """

    if isinstance(output, WorkerContributionOutput):
        if output.notes:
            return f"{output.summary}\n\n{output.notes}"
        return output.summary
    if isinstance(output, ReviewVerdictOutput):
        if output.verdict == "request_changes" and output.feedback_text:
            return f"{output.notes}\n\n{output.feedback_text}"
        return output.notes
    raise TypeError(f"unsupported async orchestrator agent output: {type(output).__name__}")


def _stdout_text(stdout: bytes | str) -> str:
    if isinstance(stdout, bytes):
        return stdout.decode(errors="replace")
    return stdout


def validate_worktree_name(name: str) -> None:
    """Validate a worktree directory name is a single safe path segment."""

    if _WORKTREE_NAME_PATTERN.fullmatch(name) is None or "\\" in name:
        raise ValueError(
            "worktree name must be a single path segment containing only "
            "letters, numbers, dots, underscores, and hyphens"
        )


def _absolute_path(path: Path) -> Path:
    return path.expanduser().resolve()


def _current_head(repo: Path) -> str:
    return _git_stdout(repo, "rev-parse", "--verify", "HEAD")


def _worktree_has_unmerged_commits(worktree: Path, target_branch: str) -> bool:
    """Return whether ``worktree`` has commits not yet on ``target_branch``.

    Distinguishes a ``blocked`` contribution that committed task-graph edits or
    progress (worth merging) from one that committed nothing (close and respawn).
    Counts commits reachable from the worktree ``HEAD`` but not from the merge
    target, so it stays correct even when the target advanced after this worktree
    forked.
    """

    count = _git_stdout(worktree, "rev-list", "--count", f"{target_branch}..HEAD")
    return count.strip() not in ("", "0")


def _blocked_non_task_change_feedback(worktree: Path, target_branch: str) -> str | None:
    paths = _changed_non_task_paths(worktree, target_branch)
    if not paths:
        return None
    diff = _git_stdout(
        worktree,
        "diff",
        "--find-renames",
        f"{target_branch}...HEAD",
        "--",
        *paths,
        _ORCHESTRATOR_GIT_METADATA_EXCLUDE_PATHSPEC,
    )
    path_list = "\n".join(f"- {path}" for path in paths)
    diff_block = _trim_text(diff or "(no path-specific diff produced)", max_length=12000)
    return (
        "Blocked contribution touched non-task files.\n\n"
        "The blocked worker contract allows committed task-graph edits under "
        f"`{TASKS_DIRECTORY_NAME}/` only. The reviewer must request changes unless "
        "these non-task edits are intentionally allowed by a newer contract.\n\n"
        f"Non-task changed paths:\n{path_list}\n\n"
        f"```diff\n{diff_block}\n```"
    )


def _changed_non_task_paths(worktree: Path, target_branch: str) -> tuple[str, ...]:
    output = _git_stdout(
        worktree,
        "diff",
        "--name-status",
        "--find-renames",
        f"{target_branch}...HEAD",
        "--",
        ".",
        _ORCHESTRATOR_GIT_METADATA_EXCLUDE_PATHSPEC,
    )
    paths: set[str] = set()
    for line in output.splitlines():
        fields = line.split("\t")
        if len(fields) < 2:
            continue
        status = fields[0]
        changed_paths = fields[1:] if status[:1] in {"R", "C"} else fields[-1:]
        for path in changed_paths:
            if not _is_task_path(path):
                paths.add(path)
    return tuple(sorted(paths))


def _is_task_path(path: str) -> bool:
    return path == TASKS_DIRECTORY_NAME or path.startswith(f"{TASKS_DIRECTORY_NAME}/")


def _branch_head(repo: Path, branch: str) -> str:
    """Return the commit ``branch`` points at — the fast-forward publish target.

    Read instead of ``HEAD`` so the staging trial merge is built on exactly the
    ref the publish step advances, keeping ``git merge --ff-only`` correct even
    if the entrypoint's ``HEAD`` is ever detached or on another branch.
    """

    return _git_stdout(repo, "rev-parse", "--verify", branch)



def _add_detached_worktree(repo: Path, *, path: Path, head: str) -> None:
    _run_git(repo, "worktree", "add", "--detach", path, head)


def _cp_archive_into(source_dir: Path, destination_dir: Path) -> None:
    """Copy the *contents* of ``source_dir`` into ``destination_dir``.

    Uses the same portable file-copy machinery as workspace mirroring: try a
    reflink clone first where supported, then fall back to ``shutil.copy2``.
    Symlinks are preserved and existing destination files are overwritten.
    Raises ``WorkspaceMirrorError`` on failure (callers treat build-cache copies
    as best-effort).
    """

    mirror_workspace(
        source_dir,
        destination_dir,
        config=WorkspaceMirrorConfig(
            reflink_mode=MirrorReflinkMode.AUTO,
            existing_path_policy=MirrorExistingPathPolicy.OVERWRITE,
        ),
    )


def _seed_build_dir_copy(source_dir: Path, destination_dir: Path) -> None:
    """Replace a worktree's local ``.lake/build`` with ``source_dir``'s contents.

    Seeding must own a safe local build directory: overlaying into an existing
    build leaves stale artifacts, while writing through a symlinked ``.lake`` (or
    symlinked ``.lake/build``) can mutate the entrypoint/shared cache. Reject
    those unsafe shapes; callers treat this as best-effort and leave the worktree
    unseeded.
    """

    _ensure_safe_seed_build_destination(destination_dir)
    _replace_dir_copy(source_dir, destination_dir)


def _ensure_safe_seed_build_destination(destination_dir: Path) -> None:
    lake_dir = destination_dir.parent
    if lake_dir.is_symlink():
        raise RuntimeError(f"refusing to seed build cache through symlinked .lake: {lake_dir}")
    if lake_dir.exists() and not lake_dir.is_dir():
        raise RuntimeError(
            f"refusing to seed build cache because .lake is not a directory: {lake_dir}"
        )
    if destination_dir.is_symlink():
        raise RuntimeError(
            f"refusing to seed build cache over symlinked .lake/build: {destination_dir}"
        )
    if destination_dir.exists() and not destination_dir.is_dir():
        raise RuntimeError(
            "refusing to seed build cache because .lake/build is not a directory: "
            f"{destination_dir}"
        )


def _replace_dir_copy(source_dir: Path, destination_dir: Path) -> None:
    """Atomically replace ``destination_dir`` with a fresh copy of ``source_dir``.

    Copies into a sibling ``<name>.incoming`` then swaps it into place via
    ``os.replace`` (rename), so a reader never observes a half-written snapshot.
    If the final swap fails after the old snapshot was moved aside, the old
    snapshot is restored — the destination is never left missing.
    """

    parent = destination_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    incoming = parent / f"{destination_dir.name}.incoming"
    previous = parent / f"{destination_dir.name}.old"
    for stale in (incoming, previous):
        if stale.exists():
            shutil.rmtree(stale)
    _cp_archive_into(source_dir, incoming)
    moved_old = False
    if destination_dir.exists():
        os.replace(destination_dir, previous)
        moved_old = True
    try:
        os.replace(incoming, destination_dir)
    except OSError:
        if moved_old:  # restore the previous snapshot rather than leave nothing
            os.replace(previous, destination_dir)
        raise
    if previous.exists():
        shutil.rmtree(previous, ignore_errors=True)


def _exclude_orchestrator_metadata_from_git_status(worktree: Path) -> None:
    exclude_path = Path(_git_stdout(worktree, "rev-parse", "--git-path", "info/exclude"))
    if not exclude_path.is_absolute():
        exclude_path = worktree / exclude_path
    exclude_path.parent.mkdir(parents=True, exist_ok=True)
    existing = exclude_path.read_text(encoding="utf-8") if exclude_path.exists() else ""
    pattern = f"{_ORCHESTRATOR_GIT_METADATA_PATHSPEC}/"
    if pattern in existing.splitlines():
        return
    separator = "" if not existing or existing.endswith("\n") else "\n"
    exclude_path.write_text(f"{existing}{separator}{pattern}\n", encoding="utf-8")


async def _run_in_abandonable_thread(call: Callable[[], None]) -> None:
    """Run setup on a daemon thread that cannot block event-loop shutdown."""

    loop = asyncio.get_running_loop()
    completion: asyncio.Future[None] = loop.create_future()

    def _worker() -> None:
        failure: BaseException | None = None
        try:
            call()
        except BaseException as exc:  # propagate the synchronous runner result
            failure = exc

        def _publish_completion() -> None:
            if completion.done():
                return
            if failure is None:
                completion.set_result(None)
            else:
                completion.set_exception(failure)

        try:
            loop.call_soon_threadsafe(_publish_completion)
        except RuntimeError:
            # The loop may already be closed after this daemon was abandoned.
            pass

    threading.Thread(
        target=_worker,
        name="tend-worktree-setup",
        daemon=True,
    ).start()
    await completion


async def _wait_for_task_settle_ignoring_cancellation(
    task: asyncio.Task[object],
    *,
    timeout_seconds: float | None,
) -> bool:
    """Wait for cleanup work despite repeated cancellation; report timeout."""

    loop = asyncio.get_running_loop()
    deadline = None if timeout_seconds is None else loop.time() + timeout_seconds
    while True:
        if task.done():
            try:
                task.result()
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                _LOGGER.debug(
                    "blocking operation failed while cancellation settled: %s",
                    exc,
                )
            return True
        remaining = None if deadline is None else deadline - loop.time()
        if remaining is not None and remaining <= 0:
            return False
        try:
            if remaining is None:
                await asyncio.shield(task)
            else:
                await asyncio.wait_for(asyncio.shield(task), timeout=remaining)
        except asyncio.CancelledError:
            # A second signal must not interrupt subprocess cleanup. The caller
            # decides whether timeout permits abandoning the worker.
            continue
        except TimeoutError:
            return False
        except Exception as exc:
            _LOGGER.debug(
                "blocking operation failed while cancellation settled: %s",
                exc,
            )
            return True
        return True


class _WorktreeSetupCommandRunner:
    """Synchronous tracked setup Popen used from one executor worker thread.

    ``start_new_session`` provides descendant process-group containment on POSIX.
    Windows ignores it here, so cancellation can terminate only the direct child;
    stronger cross-platform containment is tracked in issue #152.
    """

    __slots__ = (
        "_argv",
        "_cancel_signal",
        "_lock",
        "_process",
        "_worktree",
    )

    def __init__(
        self,
        command: AsyncOrchestratorWorktreeSetupCommandConfig,
        *,
        entrypoint: Path,
        worktree: Path,
    ) -> None:
        self._argv = command.argv_for_paths(entrypoint=entrypoint, worktree=worktree)
        self._worktree = worktree
        self._lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None
        self._cancel_signal: int | None = None

    def run(self) -> None:
        process = subprocess.Popen(
            self._argv,
            cwd=self._worktree,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        with self._lock:
            self._process = process
            cancel_signal = self._cancel_signal
        if cancel_signal is not None:
            _signal_setup_process_group_or_process(process, cancel_signal)
        stdout, stderr = process.communicate()
        if process.returncode:
            raise subprocess.CalledProcessError(
                process.returncode,
                self._argv,
                output=stdout,
                stderr=stderr,
            )

    def signal_process_group(self, signum: int) -> None:
        """Record cancellation; signal the POSIX group or Windows direct child."""

        with self._lock:
            self._cancel_signal = signum
            process = self._process
        if process is not None:
            _signal_setup_process_group_or_process(process, signum)


def _signal_setup_process_group_or_process(
    process: subprocess.Popen[str],
    signum: int,
) -> None:
    """Signal a POSIX setup group, or only the direct child on Windows."""

    if os.name != "nt":
        try:
            os.killpg(process.pid, signum)
            return
        except ProcessLookupError:
            return
        except OSError:
            pass
    if process.poll() is not None:
        return
    try:
        process.send_signal(signum)
    except ProcessLookupError:
        return


def _consume_abandoned_thread_result(task: asyncio.Task[object]) -> None:
    """Retrieve the result if an abandoned setup worker eventually settles."""

    try:
        task.result()
    except asyncio.CancelledError:
        pass
    except Exception as exc:
        _LOGGER.debug("abandoned provisioning setup eventually failed: %s", exc)


def _validation_worktree_provisioning_sentinel(staging: Path) -> Path:
    """Return the orchestrator-owned sentinel adjacent to the staging checkout."""

    return staging.with_name(f"{staging.name}{_VALIDATION_WORKTREE_PROVISIONED_SUFFIX}")


def _provisioning_sentinel_is_regular(sentinel: Path) -> bool:
    """Recognize only a regular sentinel itself, never a followed symlink target."""

    try:
        return stat.S_ISREG(sentinel.lstat().st_mode)
    except OSError:
        return False


def _write_provisioning_sentinel(sentinel: Path) -> None:
    """Atomically publish readiness without following an existing symlink."""

    sentinel.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=sentinel.parent,
        prefix=f".{sentinel.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as stream:
            stream.write("provisioned\n")
        os.replace(temporary, sentinel)
    finally:
        temporary.unlink(missing_ok=True)


def _worktree_is_registered(entrypoint: Path, worktree: Path) -> bool:
    """Return whether Git still records ``worktree`` as a linked checkout."""

    target = _absolute_path(worktree)
    completed = _run_git(entrypoint, "worktree", "list", "--porcelain", "-z")
    return any(
        record_field.startswith("worktree ")
        and _absolute_path(Path(record_field.removeprefix("worktree "))) == target
        for record_field in completed.stdout.split("\0")
    )


def _worktree_is_registered_at_exact_path(entrypoint: Path, worktree: Path) -> bool:
    """Check Git's registration without resolving a symlink at ``worktree``."""

    target = Path(os.path.abspath(worktree.expanduser()))
    completed = _run_git(entrypoint, "worktree", "list", "--porcelain", "-z")
    return any(
        record_field.startswith("worktree ")
        and Path(
            os.path.abspath(Path(record_field.removeprefix("worktree ")).expanduser())
        )
        == target
        for record_field in completed.stdout.split("\0")
    )


def _validation_worktree_has_expected_identity(
    entrypoint: Path,
    staging: Path,
) -> bool:
    """Verify staging itself is this entrypoint's real linked-worktree directory."""

    try:
        if not stat.S_ISDIR(staging.lstat().st_mode):
            return False
        gitfile = staging / ".git"
        if not stat.S_ISREG(gitfile.lstat().st_mode):
            return False
        if not _worktree_is_registered_at_exact_path(entrypoint, staging):
            return False
        gitdir_line = gitfile.read_text(encoding="utf-8").strip()
        if not gitdir_line.startswith("gitdir: "):
            return False
        gitdir = Path(gitdir_line.removeprefix("gitdir: "))
        if not gitdir.is_absolute():
            gitdir = gitfile.parent / gitdir
        gitdir = gitdir.resolve()
        common_dir_text = _run_git(entrypoint, "rev-parse", "--git-common-dir").stdout.strip()
        common_dir = Path(common_dir_text)
        if not common_dir.is_absolute():
            common_dir = entrypoint / common_dir
        if gitdir.parent != (common_dir.resolve() / "worktrees"):
            return False
        backlink = Path((gitdir / "gitdir").read_text(encoding="utf-8").strip())
        if not backlink.is_absolute():
            backlink = gitdir / backlink
        return backlink.resolve() == gitfile.resolve()
    except (OSError, UnicodeError, subprocess.SubprocessError):
        return False


def _path_exists_without_following(path: Path) -> bool:
    """Return whether the directory entry exists, including a broken symlink."""

    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _quarantine_unexpected_worktree_path(path: Path, *, reason: str) -> Path:
    """Rename an unexpected path entry itself to a unique inspection path."""

    counter = 0
    while True:
        suffix = str(os.getpid()) if counter == 0 else f"{os.getpid()}-{counter}"
        quarantine = path.with_name(f"{path.name}.invalid-{suffix}")
        if not _path_exists_without_following(quarantine):
            break
        counter += 1
    path.rename(quarantine)
    _LOGGER.warning(
        "quarantined unexpected async worktree path without following it: "
        "path=%s quarantine=%s reason=%s",
        path,
        quarantine,
        reason,
    )
    return quarantine


def _prepare_validation_worktree_path(entrypoint: Path, staging: Path) -> bool:
    """Return whether staging is safe to reuse, quarantining any other object."""

    if not _path_exists_without_following(staging):
        return False
    if _validation_worktree_has_expected_identity(entrypoint, staging):
        return True
    _quarantine_unexpected_worktree_path(
        staging,
        reason="not a real registered staging worktree of the entrypoint",
    )
    # If an interrupted add left a registration at this now-missing path, make
    # it immediately pruneable. A locked registration is reclaimed by the fresh
    # provisioning path's explicit unlock/remove retry.
    _run_git(entrypoint, "worktree", "prune", "--expire", "now", check=False)
    return False


def _worktree_exists_or_is_registered(entrypoint: Path, worktree: Path) -> bool:
    """Observe either a checkout path or its Git worktree registration."""

    return (
        worktree.exists()
        or worktree.is_symlink()
        or _worktree_is_registered(entrypoint, worktree)
    )


def _reclaim_missing_worktree_registration(entrypoint: Path, worktree: Path) -> None:
    """Clear an orchestrator-owned stale registration for a missing checkout."""

    if worktree.exists() or worktree.is_symlink():
        raise RuntimeError(f"refusing to reclaim extant async worktree: {worktree}")
    _LOGGER.warning("reclaiming stale Git worktree registration: %s", worktree)
    # A missing registration may still be locked. Unlock/remove are best-effort;
    # expire-now prune handles fresh stale entries rather than waiting months for
    # Git's default worktree-prune expiry.
    _run_git(entrypoint, "worktree", "unlock", worktree, check=False)
    _run_git(entrypoint, "worktree", "remove", "--force", worktree, check=False)
    _run_git(entrypoint, "worktree", "prune", "--expire", "now", check=False)
    if _worktree_is_registered(entrypoint, worktree):
        raise RuntimeError(
            f"failed to reclaim stale Git worktree registration: {worktree}"
        )


def _cleanup_worktree_created_by_provisioning_attempt(
    entrypoint: Path,
    worktree: Path,
    *,
    worktree_id: str,
    existed_before: bool,
    add_succeeded: bool,
) -> None:
    """Remove an observably owned worktree after unsuccessful provisioning."""

    if (
        existed_before
        or not add_succeeded
        or not _worktree_exists_or_is_registered(entrypoint, worktree)
    ):
        return
    _cleanup_failed_worktree_creation(
        entrypoint,
        worktree,
        worktree_id=worktree_id,
    )


def _cleanup_failed_worktree_creation(
    entrypoint: Path,
    worktree: Path,
    *,
    worktree_id: str,
) -> None:
    _LOGGER.warning(
        "cleaning up async worktree after creation failure: id=%s path=%s",
        worktree_id,
        worktree,
    )
    try:
        # Never ask Git to operate through an unexpected object. This also
        # recovers a directory created before `git worktree add` registered it.
        if _path_exists_without_following(
            worktree
        ) and not _worktree_is_registered_at_exact_path(entrypoint, worktree):
            _quarantine_unexpected_worktree_path(
                worktree,
                reason=f"unregistered failed worktree creation ({worktree_id})",
            )
            _run_git(entrypoint, "worktree", "prune", "--expire", "now", check=False)
            return

        # A locked worktree rejects even a forced removal. Unlock best-effort
        # first so cleanup normally removes both the tree and admin entry.
        _run_git(entrypoint, "worktree", "unlock", worktree, check=False)
        remove_completed = _run_git(
            entrypoint,
            "worktree",
            "remove",
            "--force",
            worktree,
            check=False,
        )
        if remove_completed.returncode != 0:
            if _path_exists_without_following(
                worktree
            ) and not _worktree_is_registered_at_exact_path(entrypoint, worktree):
                _quarantine_unexpected_worktree_path(
                    worktree,
                    reason=f"git remove left an unregistered worktree ({worktree_id})",
                )
            else:
                _LOGGER.warning(
                    "failed to remove async worktree after creation failure: "
                    "id=%s path=%s: %s",
                    worktree_id,
                    worktree,
                    _completed_process_error_summary(remove_completed),
                )
        prune_completed = _run_git(
            entrypoint,
            "worktree",
            "prune",
            "--expire",
            "now",
            check=False,
        )
        if prune_completed.returncode != 0:
            _LOGGER.warning(
                "failed to prune git worktrees after creation failure: id=%s path=%s: %s",
                worktree_id,
                worktree,
                _completed_process_error_summary(prune_completed),
            )
    except Exception as exc:  # pragma: no cover - best-effort cleanup only.
        _LOGGER.warning(
            "failed to clean up async worktree after creation failure: id=%s path=%s: %s",
            worktree_id,
            worktree,
            exc,
        )


def _remove_worktree_tree(
    entrypoint: Path,
    worktree: Path,
    *,
    worktree_id: str,
    target_branch: str,
) -> None:
    """Remove a closed linked worktree's working tree + git admin entry.

    Best-effort: ``git worktree remove --force`` drops both the working tree and
    the ``.git/worktrees/<id>`` administrative entry in one step. If git declines
    (e.g. the directory was already partially removed) we fall back to a direct
    tree delete plus ``git worktree prune`` so the disk is still reclaimed and the
    stale admin entry cleared. Every failure is swallowed — disk reclamation must
    never disrupt the run. Local-only work is preserved by skipping cleanup when
    the worktree is dirty outside ``.tend/`` or still has commits absent from the
    merge target.
    """

    if not worktree.exists():
        _run_git(entrypoint, "worktree", "prune", check=False)
        return
    try:
        dirty_status = _worktree_dirty_status_excluding_orchestrator_metadata(worktree)
        if dirty_status:
            _LOGGER.warning(
                "closed async worktree still has uncommitted changes; skipping cleanup: "
                "id=%s path=%s status=%s",
                worktree_id,
                worktree,
                _trim_text(dirty_status, max_length=1000),
            )
            return
        if _worktree_has_unmerged_commits(worktree, target_branch):
            _LOGGER.warning(
                "closed async worktree still has commits not on %s; skipping cleanup: "
                "id=%s path=%s",
                target_branch,
                worktree_id,
                worktree,
            )
            return
        _LOGGER.info(
            "removing closed async worktree to reclaim disk: id=%s path=%s",
            worktree_id,
            worktree,
        )
        remove_completed = _run_git(
            entrypoint,
            "worktree",
            "remove",
            "--force",
            worktree,
            check=False,
        )
        if remove_completed.returncode != 0:
            _LOGGER.warning(
                "git worktree remove failed for closed async worktree, "
                "falling back to direct delete: id=%s path=%s: %s",
                worktree_id,
                worktree,
                _completed_process_error_summary(remove_completed),
            )
            shutil.rmtree(worktree, ignore_errors=True)
            _run_git(entrypoint, "worktree", "prune", check=False)
    except Exception as exc:  # pragma: no cover - best-effort cleanup only.
        _LOGGER.warning(
            "failed to remove closed async worktree: id=%s path=%s: %s",
            worktree_id,
            worktree,
            exc,
        )


async def _run_validation_commands_async(
    commands: Sequence[AsyncOrchestratorValidationCommandConfig],
    cwd: Path,
    oom_score_adj: int | None = None,
) -> _ValidationCommandFailure | None:
    for command_index, command in enumerate(commands):
        _LOGGER.info("running async validation command: %s", _format_command(command.argv))
        try:
            process = await asyncio.create_subprocess_exec(
                *command.argv,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
                preexec_fn=oom_score_adj_preexec(oom_score_adj),
            )
        except OSError as exc:
            return _ValidationCommandFailure(
                argv=command.argv,
                returncode=None,
                error=str(exc),
                command_index=command_index,
            )
        communicate_task: asyncio.Task[tuple[bytes, bytes]] = asyncio.create_task(
            process.communicate()
        )
        try:
            if command.timeout_seconds is None:
                stdout_bytes, stderr_bytes = await _communicate_watching_leader_exit(
                    process,
                    communicate_task,
                    cwd,
                )
            else:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    _communicate_watching_leader_exit(process, communicate_task, cwd),
                    timeout=command.timeout_seconds,
                )
        except TimeoutError:
            _LOGGER.warning(
                "async validation command timed out after %ss: %s",
                command.timeout_seconds,
                _format_command(command.argv),
            )
            stdout_bytes, stderr_bytes = await _terminate_validation_process(
                process,
                communicate_task,
                cwd,
            )
            return _ValidationCommandFailure(
                argv=command.argv,
                returncode=None,
                stdout=_decode_subprocess_stream(stdout_bytes),
                stderr=_decode_subprocess_stream(stderr_bytes),
                error=f"validation command timed out after {command.timeout_seconds}s",
                timed_out=True,
                command_index=command_index,
            )
        except asyncio.CancelledError:
            _LOGGER.info(
                "terminating async validation command after cancellation: %s",
                _format_command(command.argv),
            )
            await _terminate_validation_process(process, communicate_task, cwd)
            raise
        returncode = process.returncode
        if returncode != 0:
            # A negative returncode outside the timeout path means the leader
            # exited on a signal. Which signal is only a heuristic for *why*:
            # the allowlisted signals typically indicate exogenous termination
            # (kernel OOM killer, operator kill, service-manager shutdown),
            # but an exit status alone cannot prove who sent the signal. (The
            # orchestrator's own cancellation never reaches here: it catches
            # CancelledError above and re-raises.)
            signal_number = -returncode if returncode is not None and returncode < 0 else None
            if signal_number is not None:
                # Descendants that redirected their stdio can survive the
                # leader (pipe EOF let communicate() return), and a caller's
                # retry would then run concurrently with them in the same
                # worktree. Kill the whole POSIX group before returning (only
                # the direct child on Windows) — the leader is already dead, so
                # there is no grace-period concern.
                # Idempotent with the kill in _communicate_watching_leader_exit,
                # which fires when descendants held the pipes open instead.
                _kill_validation_process_group_or_process(process)
                # Compose signal-exit cleanup with the orphan sweep used by the
                # timeout/cancellation path: descendants with redirected stdio
                # can make communicate() finish before the leader-exit watcher
                # observes the signal, and the group kill has the same fork race.
                await _reap_orphaned_validation_processes(cwd, process.pid)
            error = ""
            if signal_number in _VALIDATION_CANCELLATION_SIGNALS:
                # A cancellation signal says nothing about the change under
                # validation; classify as cancelled so callers can retry
                # instead of booking a batch failure (issue #132).
                cancelled = True
                _LOGGER.warning(
                    "async validation command cancelled (signal %d): %s",
                    signal_number,
                    _format_command(command.argv),
                )
            else:
                cancelled = False
                if signal_number is not None:
                    # Any other signal is a validator crash: a deterministic
                    # crash is evidence of a real failure, so it is never
                    # retried — a retry would convert that failure signal
                    # into a pass.
                    error = f"validation command crashed (signal {signal_number})"
                    _LOGGER.warning(
                        "async validation command crashed (signal %d): %s",
                        signal_number,
                        _format_command(command.argv),
                    )
            return _ValidationCommandFailure(
                argv=command.argv,
                returncode=returncode,
                stdout=_decode_subprocess_stream(stdout_bytes),
                stderr=_decode_subprocess_stream(stderr_bytes),
                error=error,
                cancelled=cancelled,
                command_index=command_index,
            )
    return None


async def _communicate_watching_leader_exit(
    process: asyncio.subprocess.Process,
    communicate_task: asyncio.Task[tuple[bytes, bytes]],
    repo: Path,
) -> tuple[bytes, bytes]:
    """Await captured output without trusting pipe EOF after a signal exit.

    ``communicate()`` returns only when the stdout/stderr pipes hit EOF, but a
    leader killed by a signal can leave descendants holding the inherited pipe
    ends, hanging the wait forever (with no timeout configured, until the
    orchestrator itself is cancelled). Poll the leader's exit concurrently —
    ``process.returncode`` is set on waitpid regardless of pipe state, whereas
    ``Process.wait()`` is only woken once every pipe disconnects and would
    hang exactly like ``communicate()`` here. On a signal exit, SIGKILL the
    surviving POSIX group at once and drain the output with a bounded wait. On
    Windows only the direct child can be killed, so descendants may retain the
    pipes; cross-platform containment remains tracked in issue #152. A leader
    that exits normally (returncode >= 0) keeps the plain ``communicate()``
    semantics — its children may legitimately still be writing output.
    """

    while True:
        if communicate_task.done():
            return await communicate_task
        returncode = process.returncode
        if returncode is not None and returncode < 0:
            _LOGGER.warning(
                "validation command leader exited on signal %d while descendants "
                "hold its output pipes; killing its POSIX process group/direct child",
                -returncode,
            )
            return await _kill_validation_process_group_and_drain_output(
                process,
                communicate_task,
                repo,
            )
        try:
            return await asyncio.wait_for(
                asyncio.shield(communicate_task),
                timeout=_VALIDATION_OUTPUT_POLL_INTERVAL_SECONDS,
            )
        except TimeoutError:
            continue


async def _terminate_validation_process(
    process: asyncio.subprocess.Process,
    communicate_task: asyncio.Task[tuple[bytes, bytes]],
    repo: Path,
) -> tuple[bytes, bytes]:
    """Terminate a POSIX validation group, or only its direct child on Windows."""

    try:
        _signal_validation_process_group_or_process(process, signal.SIGTERM)
        communicated = await _wait_for_validation_output(
            process,
            communicate_task,
            timeout_seconds=_VALIDATION_TERMINATION_GRACE_SECONDS,
            stop_when_leader_exits=True,
        )
        if communicated is not None:
            return communicated

        _kill_validation_process_group_or_process(process)
        communicated = await _wait_for_validation_output(
            process,
            communicate_task,
            timeout_seconds=_VALIDATION_KILL_GRACE_SECONDS,
            stop_when_leader_exits=False,
        )
        if communicated is not None:
            return communicated

        communicate_task.cancel()
        try:
            return await asyncio.wait_for(communicate_task, timeout=0.1)
        except asyncio.CancelledError:
            return b"", b""
        except TimeoutError:
            return b"", b""
    finally:
        # The group kill above races against build tools that fork workers
        # continuously (``lake`` spawning ``lean``); sweep for survivors so no
        # orphan outlives the kill (#132/#146). Runs on every exit, including
        # the common early return where the leader dies within the SIGTERM
        # grace and the group SIGKILL is never sent. The leader pid is the
        # sweep's sole kill criterion: the validation was started with
        # ``start_new_session=True``, so descendants initially inherit both
        # ``pgid`` and ``sid`` from the leader. A descendant may change its
        # process group while retaining its session; either identifier still
        # matching the leader proves ownership.
        await _reap_orphaned_validation_processes(repo, process.pid)


async def _kill_validation_process_group_and_drain_output(
    process: asyncio.subprocess.Process,
    communicate_task: asyncio.Task[tuple[bytes, bytes]],
    repo: Path,
) -> tuple[bytes, bytes]:
    """Kill/drain a POSIX validation group; Windows kills only the direct child."""

    try:
        _kill_validation_process_group_or_process(process)
        communicated = await _wait_for_validation_output(
            process,
            communicate_task,
            timeout_seconds=_VALIDATION_KILL_GRACE_SECONDS,
            stop_when_leader_exits=False,
        )
        if communicated is not None:
            return communicated

        communicate_task.cancel()
        try:
            return await asyncio.wait_for(communicate_task, timeout=0.1)
        except asyncio.CancelledError:
            return b"", b""
        except TimeoutError:
            return b"", b""
    finally:
        await _reap_orphaned_validation_processes(repo, process.pid)


async def _wait_for_validation_output(
    process: asyncio.subprocess.Process,
    communicate_task: asyncio.Task[tuple[bytes, bytes]],
    *,
    timeout_seconds: float,
    stop_when_leader_exits: bool,
) -> tuple[bytes, bytes] | None:
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    while True:
        if communicate_task.done():
            return await communicate_task
        if stop_when_leader_exits and process.returncode is not None:
            return None
        remaining_seconds = deadline - asyncio.get_running_loop().time()
        if remaining_seconds <= 0:
            return None
        poll_seconds = remaining_seconds
        if stop_when_leader_exits:
            poll_seconds = min(
                remaining_seconds,
                _VALIDATION_OUTPUT_POLL_INTERVAL_SECONDS,
            )
        try:
            return await asyncio.wait_for(
                asyncio.shield(communicate_task),
                timeout=poll_seconds,
            )
        except TimeoutError:
            continue


def _signal_validation_process_group_or_process(
    process: asyncio.subprocess.Process,
    signum: int,
) -> None:
    """Signal a POSIX validation group, or only the direct child on Windows."""

    if os.name != "nt":
        try:
            os.killpg(process.pid, signum)
            return
        except ProcessLookupError:
            pass
        except OSError:
            pass
    if process.returncode is not None:
        return
    try:
        process.send_signal(signum)
    except ProcessLookupError:
        return


def _kill_validation_process_group_or_process(
    process: asyncio.subprocess.Process,
) -> None:
    """SIGKILL a POSIX validation process group (Windows: direct child only).

    POSIX termination guarantees stop at the leader's process group: a
    descendant that detaches into a new session (``setsid``/daemonize) escapes
    the group kill and can outlive the validation. On Windows no group kill is
    available here, so ordinary descendants can also survive. The operator
    contract therefore requires POSIX process groups for descendant containment;
    stronger containment is tracked in issue #152.
    """

    if os.name != "nt":
        try:
            os.killpg(process.pid, signal.SIGKILL)
            return
        except ProcessLookupError:
            pass
        except OSError:
            pass
    if process.returncode is not None:
        return
    try:
        process.kill()
    except ProcessLookupError:
        return


@dataclass(frozen=True)
class _OrphanedValidationProcess:
    """A process that outlived its killed validation's group/session kill.

    ``start_time`` is the kernel starttime (clock ticks since boot, field 22 of
    ``/proc/<pid>/stat``) captured at scan time; re-checking it immediately
    before signalling detects pid reuse — a recycled pid virtually never shares
    the dead process's starttime, and would additionally have to sit in the
    killed leader's process group/session to pass re-verification.
    ``cmdline_head`` is log enrichment only, never a filter.
    """

    pid: int
    start_time: int
    cmdline_head: str


@dataclass(frozen=True)
class _ProcessStatFields:
    """The ``/proc/<pid>/stat`` fields the orphan reaper gates on."""

    comm: str
    state: str
    pgid: int
    sid: int
    start_time: int


@dataclass(frozen=True)
class _UnverifiableStat:
    """A stat read that failed without confirming the process is gone.

    EACCES/EIO/garbled content does not prove exit, so callers must neither
    signal the pid (identity unknown) nor report it as reaped (it may live on).
    """

    detail: str


def _read_process_stat_fields(
    stat_path: Path,
) -> _ProcessStatFields | _UnverifiableStat | None:
    """Parse comm, state, pgid, sid, and starttime from ``/proc/<pid>/stat``.

    The second field (``comm``) is an arbitrary byte string that may contain
    spaces, parentheses, and even newlines, so the fixed space-separated tail
    is located after the *last* ``)`` in the file. The tail then starts at
    field 3 (state); pgid, sid, and starttime are fields 5, 6, and 22.

    Returns ``None`` only when the read *confirms* the process is gone (the
    proc entry vanished: ENOENT/ESRCH). Any other failure — permission or I/O
    errors, a non-directory entry, malformed content — yields an
    ``_UnverifiableStat``: the process may still exist but cannot be
    identified, which is not the same thing as being gone.
    """

    try:
        raw = stat_path.read_bytes()
    except OSError as exc:
        if exc.errno in (errno.ENOENT, errno.ESRCH):
            return None
        return _UnverifiableStat(detail=f"errno={exc.errno}")
    _, _, rest = raw.partition(b"(")
    comm_bytes, _, tail = rest.rpartition(b")")
    fields = tail.split()
    if len(fields) < 20:
        return _UnverifiableStat(detail="malformed stat")
    try:
        return _ProcessStatFields(
            comm=comm_bytes.decode("utf-8", errors="replace"),
            state=fields[0].decode("utf-8", errors="replace"),
            pgid=int(fields[2]),
            sid=int(fields[3]),
            start_time=int(fields[19]),
        )
    except ValueError:
        return _UnverifiableStat(detail="malformed stat")


def _describe_orphan_process(proc_entry: Path, stat_fields: _ProcessStatFields) -> str:
    """Describe a reap candidate for the WARNING log line.

    Enrichment only — never a filter. Falls back to the stat ``comm`` when the
    cmdline is unreadable or empty.
    """

    try:
        cmdline_bytes = (proc_entry / "cmdline").read_bytes()
    except OSError:
        cmdline_bytes = b""
    argv = [
        argument.decode("utf-8", errors="replace")
        for argument in cmdline_bytes.split(b"\0")
        if argument
    ]
    if not argv:
        return f"[{stat_fields.comm}]"
    return _trim_text(" ".join(argv), max_length=200)


def _find_orphaned_validation_processes(
    leader_pid: int,
    proc_root: Path = _PROC_ROOT,
) -> tuple[tuple[_OrphanedValidationProcess, ...], bool]:
    """Scan ``proc_root`` for live processes owned by the killed ``leader_pid``.

    Ownership — process group id *or* session id equal to the leader's pid —
    is the **sole kill criterion**. Validation commands are launched with
    ``start_new_session=True``, so the leader's pid is both its session id and
    its process-group id and every descendant inherits them; a session cannot
    be joined from outside (``setsid`` always creates a fresh session keyed by
    the caller's own pid, and ``setpgid`` can only move a process between
    groups of its own session), so anything carrying the dead leader's
    pgid/sid is abandoned validation work — exactly the set the original
    ``killpg`` was entitled to kill. Executable names and argv are *not*
    consulted: a ``lake build`` run with cwd inside the repo carries no repo
    path in argv, ``lean Foo.lean`` uses a relative path, and helper wrappers
    can have any name — no spelling of name or arguments can be relied on to
    identify (or is needed to identify) group members. They are captured for
    log enrichment only.

    Residual risks, both accepted: a descendant that deliberately calls
    ``setsid`` escapes the sweep (nothing in the lean toolchain does, and the
    fork race this sweep exists for always leaves survivors inside the
    leader's group); and the dead leader's pid being recycled as a *new*
    session/group leader inside the sub-second reap window would require the
    kernel's sequential pid allocation to wrap the entire pid space within
    that window — not a realistic channel.

    Zombie/dead processes (state Z/X) are skipped: they cannot be signalled
    into anything and their pid slot persists until their reaper parent
    collects them. A numeric process entry whose stat cannot be read or parsed
    never becomes a signal candidate because ownership cannot be established.
    It is nevertheless possibly live and possibly owned by the killed leader,
    so it makes the scan incomplete and is never treated as proof of absence.
    This branch sees arbitrary host processes, hence no per-pid logging.

    Returns ``(candidates, scan_complete)``. ``scan_complete`` is false when
    ``proc_root`` could not be enumerated or any possibly-live numeric entry
    was unverifiable; callers must rescan rather than treating an empty,
    incomplete result as convergence.
    """

    try:
        entries = sorted(entry.name for entry in proc_root.iterdir())
    except OSError:
        return (), False
    own_pid = os.getpid()
    orphans: list[_OrphanedValidationProcess] = []
    scan_complete = True
    for entry in entries:
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid == own_pid:
            continue
        stat_fields = _read_process_stat_fields(proc_root / entry / "stat")
        if stat_fields is None:
            continue  # Gone mid-scan: confirmed absent.
        if isinstance(stat_fields, _UnverifiableStat):
            # It may still be an owned process. Do not signal without verified
            # identity, but do not let this scan prove convergence either.
            scan_complete = False
            continue
        if stat_fields.pgid != leader_pid and stat_fields.sid != leader_pid:
            continue  # Not owned by the killed validation: never touched.
        if stat_fields.state in _VALIDATION_ORPHAN_DEAD_PROCESS_STATES:
            continue  # Zombie/dead: nothing to signal; its parent will collect it.
        orphans.append(
            _OrphanedValidationProcess(
                pid=pid,
                start_time=stat_fields.start_time,
                cmdline_head=_describe_orphan_process(proc_root / entry, stat_fields),
            )
        )
    return tuple(orphans), scan_complete


_OrphanSignalOutcome = Literal["delivered", "gone", "failed", "escaped", "unverifiable"]


def _reverify_orphan_identity(
    orphan: _OrphanedValidationProcess,
    leader_pid: int,
    proc_root: Path,
) -> Literal["verified", "gone", "escaped", "unverifiable"]:
    """Re-check, immediately before signalling, that the pid is still the orphan.

    ``verified`` requires the same starttime as at scan time (a recycled pid
    virtually never shares the dead process's starttime) *and* continued
    membership in the killed leader's group/session (a process that called
    ``setsid`` after the scan has escaped the sweep by the accepted narrowing
    and must not be signalled). A vanished entry or changed starttime confirms
    the scanned process is ``gone``; a stat that exists but cannot be read or
    parsed is ``unverifiable`` — logged at WARNING and never signalled, since
    neither identity nor exit is established.
    """

    stat_fields = _read_process_stat_fields(proc_root / str(orphan.pid) / "stat")
    if stat_fields is None:
        return "gone"
    if isinstance(stat_fields, _UnverifiableStat):
        _LOGGER.warning(
            "cannot verify orphaned validation process before signalling; "
            "leaving it alone: pid=%d detail=%s cmd=%s",
            orphan.pid,
            stat_fields.detail,
            orphan.cmdline_head,
        )
        return "unverifiable"
    if stat_fields.start_time != orphan.start_time:
        return "gone"  # Exited and pid recycled: the scanned process is gone.
    if stat_fields.pgid != leader_pid and stat_fields.sid != leader_pid:
        return "escaped"
    return "verified"


def _log_orphan_signal_failure(
    orphan: _OrphanedValidationProcess, signum: int, exc: OSError
) -> None:
    _LOGGER.warning(
        "failed to signal orphaned validation process: pid=%d sig=%d errno=%s cmd=%s",
        orphan.pid,
        signum,
        exc.errno,
        orphan.cmdline_head,
    )


def _verify_and_signal_orphan(
    orphan: _OrphanedValidationProcess,
    leader_pid: int,
    signum: int,
    proc_root: Path,
) -> _OrphanSignalOutcome:
    """Re-verify the scanned orphan's identity, then signal it.

    Preferred path pins identity with a per-signal pidfd: ``os.pidfd_open``
    (``ProcessLookupError`` → gone) pins the pid so it cannot be recycled
    while we hold the fd, the stat re-verification then confirms the fd
    belongs to the scanned process (had the pid been recycled *before* the
    open, the stat re-read would describe the recycled process and its
    starttime mismatch stops the signal), and ``signal.pidfd_send_signal``
    delivers through the fd — race-free end to end. The fd is opened and
    closed within this single call; nothing survives the reaper's grace sleep.

    Fallback when pidfds are unavailable (non-Linux Python builds, ENOSYS,
    fd exhaustion): the same stat re-verification followed by plain
    ``os.kill``. Residual window: the verified target can exit and its pid be
    recycled to *any* unrelated process between the stat read and the kill,
    and that unrelated process can be mis-signalled without another ownership
    check. This microsecond-scale race is accepted only where the primary
    pidfd path is unavailable.

    Outcomes: ``delivered`` — the signal reached the verified process;
    ``gone`` — confirmed already gone (ESRCH on open/send, entry vanished, or
    starttime changed); ``failed`` — the send failed (``EPERM``, ...), logged
    at WARNING with errno; ``escaped`` — alive but no longer in the leader's
    group/session, deliberately not signalled; ``unverifiable`` — stat
    unreadable/garbled without confirming exit (EACCES/EIO/malformed), logged
    at WARNING, not signalled. Only delivered and gone count as reaped.
    """

    pidfd_open = getattr(os, "pidfd_open", None)
    pidfd_send_signal = getattr(signal, "pidfd_send_signal", None)
    if pidfd_open is not None and pidfd_send_signal is not None:
        try:
            pidfd: int = pidfd_open(orphan.pid)
        except ProcessLookupError:
            return "gone"
        except OSError:
            pass  # pidfd unsupported/exhausted here: fall back to stat+kill.
        else:
            try:
                verdict = _reverify_orphan_identity(orphan, leader_pid, proc_root)
                if verdict != "verified":
                    return verdict
                try:
                    pidfd_send_signal(pidfd, signum)
                except ProcessLookupError:
                    return "gone"  # Exited (and was reaped) since the open.
                except OSError as exc:
                    _log_orphan_signal_failure(orphan, signum, exc)
                    return "failed"
                return "delivered"
            finally:
                # Cleanup failure must not abort the sweep or mask the timeout/
                # cancellation whose finally block invoked the reaper.
                with suppress(OSError):
                    os.close(pidfd)
    verdict = _reverify_orphan_identity(orphan, leader_pid, proc_root)
    if verdict != "verified":
        return verdict
    try:
        os.kill(orphan.pid, signum)
    except ProcessLookupError:
        return "gone"  # Vanished between re-verify and kill.
    except OSError as exc:
        _log_orphan_signal_failure(orphan, signum, exc)
        return "failed"
    return "delivered"


def _sigkill_validation_orphans_once(
    repo: Path,
    leader_pid: int,
    proc_root: Path,
) -> tuple[tuple[_OrphanedValidationProcess, ...], bool, tuple[int, ...]]:
    """Synchronously run the shared scan-and-SIGKILL pass body."""

    survivors, scan_complete = _find_orphaned_validation_processes(
        leader_pid, proc_root
    )
    reaped: list[int] = []
    for survivor in survivors:
        outcome = _verify_and_signal_orphan(
            survivor, leader_pid, signal.SIGKILL, proc_root
        )
        if outcome == "delivered":
            _LOGGER.warning(
                "reaping orphaned validation process that survived process-group kill: "
                "pid=%d repo=%s sig=SIGKILL cmd=%s",
                survivor.pid,
                repo,
                survivor.cmdline_head,
            )
        if outcome in ("delivered", "gone"):
            reaped.append(survivor.pid)
    return survivors, scan_complete, tuple(reaped)


def _warn_about_surviving_validation_orphans(
    repo: Path,
    leader_pid: int,
    proc_root: Path,
    *,
    stopped_reason: str,
) -> None:
    """Freshly scan and warn that bounded cleanup did not converge."""

    unresolved, scan_complete = _find_orphaned_validation_processes(
        leader_pid, proc_root
    )
    survivor_details = "; ".join(
        f"pid={survivor.pid} cmd={survivor.cmdline_head}" for survivor in unresolved
    )
    if not scan_complete:
        _LOGGER.warning(
            "orphaned validation cleanup bounded by %s; two-scan convergence not "
            "established; cleanup could not be fully verified after SIGKILL passes: "
            "repo=%s known_survivors=%s",
            stopped_reason,
            repo,
            survivor_details or "none discovered",
        )
    elif unresolved:
        _LOGGER.warning(
            "orphaned validation cleanup bounded by %s; two-scan convergence not "
            "established; orphaned validation processes were not reaped after "
            "SIGKILL passes: repo=%s survivors=%s",
            stopped_reason,
            repo,
            survivor_details,
        )
    else:
        _LOGGER.warning(
            "orphaned validation cleanup bounded by %s; two-scan convergence not "
            "established; final scan found no survivors: repo=%s",
            stopped_reason,
            repo,
        )


async def _sigkill_surviving_validation_orphans(
    repo: Path,
    leader_pid: int,
    proc_root: Path,
) -> tuple[int, ...]:
    """Repeatedly scan for remaining group/session members and SIGKILL them.

    A single snapshot always loses the scan-vs-fork race against a surviving
    TERM-ignoring member that keeps forking: children born after the snapshot
    are missed. Convergence therefore requires two consecutive empty, complete
    scans. This closes the discovery-time fork/exit race: if durable child B is
    absent from scan i because listed parent A forks B and exits during A's stat
    read, B was forked before scan i finished reading stats and thus before scan
    i+1's fresh, later ``/proc`` enumeration, which necessarily includes B.
    A nonempty or incomplete scan resets the consecutive-empty count.

    Every nonempty pass is followed by another pass until the cap, regardless
    of whether the candidate set looks stable. Failed or unverifiable outcomes
    may be transient, so they cannot prove convergence. Fresh scans remain
    safe: a process appearing only in a later scan can only have been forked by
    a surviving member of the killed validation's group/session.

    Work is bounded by ``_VALIDATION_ORPHAN_KILL_PASS_LIMIT``. Per-pass
    synchronous cost scales with the number of host processes scanned and
    owned survivors signalled, so the pass cap is not a hard wall-clock bound.
    At the cap, a fresh final scan reports that convergence was not established,
    including its current survivor/verification state. Async inter-pass sleeps
    let killed processes leave ``/proc`` and yield to the event loop. Returns
    pids for which a SIGKILL was delivered
    or that were confirmed already gone.
    """

    reaped: dict[int, None] = {}
    consecutive_empty_complete_scans = 0
    convergence_established = False
    for pass_index in range(_VALIDATION_ORPHAN_KILL_PASS_LIMIT):
        survivors, scan_complete, pass_reaped = _sigkill_validation_orphans_once(
            repo, leader_pid, proc_root
        )
        if not survivors and scan_complete:
            consecutive_empty_complete_scans += 1
            if consecutive_empty_complete_scans >= 2:
                convergence_established = True
                break
        else:
            consecutive_empty_complete_scans = 0
        for pid in pass_reaped:
            reaped[pid] = None
        if pass_index + 1 < _VALIDATION_ORPHAN_KILL_PASS_LIMIT:
            await asyncio.sleep(_VALIDATION_ORPHAN_KILL_PASS_INTERVAL_SECONDS)
    if not convergence_established:
        _warn_about_surviving_validation_orphans(
            repo,
            leader_pid,
            proc_root,
            stopped_reason="pass cap",
        )
    return tuple(reaped)


def _sigkill_surviving_validation_orphans_sync(
    repo: Path,
    leader_pid: int,
    proc_root: Path,
) -> tuple[int, ...]:
    """Synchronous SIGKILL passes for cancellation teardown.

    Work is bounded by the pass cap and by a wall-clock deadline checked before
    starting another pass. As in the async loop, only two consecutive empty,
    complete scans establish convergence. A single synchronous pass cannot be
    interrupted and its cost scales with the number of host processes scanned
    and owned survivors signalled, so it may itself carry cleanup beyond the
    deadline. Once the deadline has passed, the last completed pass supplies a
    possibly stale warning instead of starting another expensive ``/proc`` scan.
    """

    reaped: dict[int, None] = {}
    deadline = time.monotonic() + _VALIDATION_ORPHAN_SYNC_CLEANUP_DEADLINE_SECONDS
    stopped_reason: str | None = None
    consecutive_empty_complete_scans = 0
    convergence_established = False
    last_survivors: tuple[_OrphanedValidationProcess, ...] = ()
    last_scan_complete = False
    for pass_index in range(_VALIDATION_ORPHAN_KILL_PASS_LIMIT):
        if pass_index > 0 and time.monotonic() >= deadline:
            stopped_reason = "synchronous cleanup deadline"
            break
        survivors, scan_complete, pass_reaped = _sigkill_validation_orphans_once(
            repo, leader_pid, proc_root
        )
        last_survivors = survivors
        last_scan_complete = scan_complete
        if not survivors and scan_complete:
            consecutive_empty_complete_scans += 1
            if consecutive_empty_complete_scans >= 2:
                convergence_established = True
                break
        else:
            consecutive_empty_complete_scans = 0
        for pid in pass_reaped:
            reaped[pid] = None
        if pass_index + 1 < _VALIDATION_ORPHAN_KILL_PASS_LIMIT:
            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds <= 0:
                stopped_reason = "synchronous cleanup deadline"
                break
            time.sleep(
                min(
                    _VALIDATION_ORPHAN_KILL_PASS_INTERVAL_SECONDS,
                    remaining_seconds,
                )
            )
    if convergence_established:
        return tuple(reaped)
    if stopped_reason is None and time.monotonic() >= deadline:
        stopped_reason = "synchronous cleanup deadline"
    if stopped_reason is not None:
        survivor_details = "; ".join(
            f"pid={survivor.pid} cmd={survivor.cmdline_head}"
            for survivor in last_survivors
        )
        _LOGGER.warning(
            "orphaned validation cleanup bounded by %s; two-scan convergence not "
            "established; stopped without a fresh post-deadline /proc scan: "
            "repo=%s known_survivors_from_last_completed_pass=%s "
            "last_scan_complete=%s survivor_list_may_be_stale=true",
            stopped_reason,
            repo,
            survivor_details or "none discovered",
            last_scan_complete,
        )
    else:
        _warn_about_surviving_validation_orphans(
            repo,
            leader_pid,
            proc_root,
            stopped_reason="pass cap",
        )
    return tuple(reaped)


async def _reap_orphaned_validation_processes(
    repo: Path,
    leader_pid: int,
    *,
    proc_root: Path = _PROC_ROOT,
    grace_seconds: float = _VALIDATION_ORPHAN_REAP_GRACE_SECONDS,
) -> tuple[int, ...]:
    """Kill every process that survived the ``leader_pid`` group kill.

    The group ``SIGTERM``/``SIGKILL`` can miss workers forked concurrently with
    signal delivery (see the note on ``_VALIDATION_ORPHAN_REAP_GRACE_SECONDS``);
    such orphans reparent to init and burn CPU/memory indefinitely. Ownership —
    pgid or sid equal to the dead leader's pid — is the sole kill criterion
    (see ``_find_orphaned_validation_processes``; ``repo`` is used only to
    contextualize log lines). Survivors get ``SIGTERM``, a short grace, then
    bounded scan→``SIGKILL`` passes (see
    ``_sigkill_surviving_validation_orphans``). Each candidate's identity is
    re-verified immediately before every signal and pinned when pidfds are
    available. The stat+kill fallback retains the documented pid-recycling
    race when pidfds are unavailable.

    Returns the pids for which a signal was actually delivered or that were
    confirmed already gone by kill time; failed kills (e.g. ``EPERM``) and
    unverifiable candidates are logged at WARNING and excluded. The clean fast
    path requires a second fresh empty, complete scan for the same discovery-
    time race argument documented by ``_sigkill_surviving_validation_orphans``.
    An incomplete initial scan proceeds to grace and bounded rescans, and an
    unavailable ``/proc`` is ultimately reported as unverifiable. Cancellation
    during the grace or an inter-pass sleep runs the same bounded multi-pass
    ``SIGKILL`` convergence loop synchronously before propagating.
    """

    survivors, scan_complete = _find_orphaned_validation_processes(
        leader_pid, proc_root
    )
    if not survivors and scan_complete:
        survivors, scan_complete = _find_orphaned_validation_processes(
            leader_pid, proc_root
        )
        if not survivors and scan_complete:
            return ()
    reaped: dict[int, None] = {}
    for survivor in survivors:
        outcome = _verify_and_signal_orphan(survivor, leader_pid, signal.SIGTERM, proc_root)
        if outcome == "delivered":
            _LOGGER.warning(
                "reaping orphaned validation process that survived process-group kill: "
                "pid=%d repo=%s sig=SIGTERM cmd=%s",
                survivor.pid,
                repo,
                survivor.cmdline_head,
            )
        if outcome in ("delivered", "gone"):
            reaped[survivor.pid] = None
    try:
        await asyncio.sleep(grace_seconds)
        for pid in await _sigkill_surviving_validation_orphans(repo, leader_pid, proc_root):
            reaped[pid] = None
    except asyncio.CancelledError:
        # Cancellation can land in the grace or an inter-pass sleep. Teardown
        # may block briefly so the same bounded multi-pass algorithm closes the
        # scan-vs-fork race before cancellation propagates.
        _sigkill_surviving_validation_orphans_sync(repo, leader_pid, proc_root)
        raise
    return tuple(reaped)



def _decode_subprocess_stream(stream: str | bytes | None) -> str:
    """Normalize a captured subprocess stream (which may be bytes on timeout)."""

    if stream is None:
        return ""
    if isinstance(stream, bytes):
        return stream.decode(errors="replace")
    return stream


def _merge_worktree_into_target_branch(
    *,
    entrypoint: Path,
    worktree: Path,
    commit_message: str,
    target_branch: str,
) -> _MergeWorktreeResult:
    # Workers own their commits (worker prompt v4+: "commit your work yourself;
    # anything left uncommitted at session end is discarded"). Merge only what
    # the worker actually committed — never ``git add -A`` the dirty worktree.
    # Uncommitted work is left untouched in the worktree (a resumed session can
    # still commit it) and never reaches the target branch. A contribution that
    # committed nothing has nothing to land, so signal that to the caller.
    if not _worktree_has_unmerged_commits(worktree, target_branch):
        return _MergeWorktreeResult(original_head=None)
    worktree_head = _current_head(worktree)
    _run_git(entrypoint, "checkout", target_branch)
    original_head = _current_head(entrypoint)
    try:
        _run_git(entrypoint, "merge", "--no-edit", "-m", commit_message, worktree_head)
    except subprocess.CalledProcessError:
        _run_git(entrypoint, "merge", "--abort", check=False)
        _run_git(entrypoint, "reset", "--hard", original_head, check=False)
        raise
    return _MergeWorktreeResult(original_head=original_head)


def _rollback_entrypoint_to_head(entrypoint: Path, head: str) -> None:
    _run_git(entrypoint, "merge", "--abort", check=False)
    _run_git(entrypoint, "reset", "--hard", head)


def _sync_staging_to_head(staging: Path, head: str) -> None:
    """Realign the detached staging worktree to ``head``, discarding a trial merge.

    ``.lake`` (and other gitignored build output) is untracked, so ``reset
    --hard`` plus ``clean -fd`` leaves the warm build cache in place while
    removing non-ignored validation scratch files that could block a later trial
    merge at the same path.
    """

    _run_git(staging, "merge", "--abort", check=False)
    _run_git(staging, "reset", "--hard", head)
    _run_git(staging, "clean", "-fd")


def _sync_staging_to_head_purging_ignored(staging: Path, head: str) -> None:
    """Like :func:`_sync_staging_to_head`, but also purge gitignored output.

    Used after a validator crash: the warm-cache pipeline trusts validators to
    be interruption-safe and incrementally correct, but that contract cannot be
    assumed to hold across the validator's own crash, and artifacts it left
    behind could make a post-crash revalidation (e.g. a bisection half) pass
    when a clean validation would still crash. ``clean -ffdx`` removes ignored
    files and nested repositories too, so the next validation in staging
    rebuilds cold. It thereby also removes staging's provisioned infrastructure;
    callers must clear the external readiness sentinel and re-provision — use
    :meth:`AsyncOrchestrator._purge_staging_after_crash` rather than calling this
    directly.
    """

    _run_git(staging, "merge", "--abort", check=False)
    _run_git(staging, "reset", "--hard", head)
    _run_git(staging, "clean", "-ffdx")


def _stage_merge(
    staging: Path,
    *,
    target_head: str,
    worktree_head: str,
    commit_message: str,
) -> str:
    """Trial-merge ``worktree_head`` onto ``target_head`` inside the staging worktree.

    Returns the resulting staging ``HEAD`` (a descendant of ``target_head``, so
    the entrypoint can later fast-forward to it). On a merge conflict the staging
    worktree is aborted/reset and the error re-raised; the entrypoint is never
    touched.
    """

    _sync_staging_to_head(staging, target_head)
    try:
        _run_git(staging, "merge", "--no-edit", "-m", commit_message, worktree_head)
    except subprocess.CalledProcessError:
        _run_git(staging, "merge", "--abort", check=False)
        _run_git(staging, "reset", "--hard", target_head, check=False)
        raise
    return _current_head(staging)


def _assemble_batch(
    staging: Path,
    base_head: str,
    members: list[tuple[str, str]],
) -> tuple[list[tuple[str, str]], list[tuple[str, subprocess.CalledProcessError]], str]:
    """Sequentially trial-merge ``members`` onto ``base_head`` inside ``staging``.

    ``members`` is ``[(worktree_id, head), …]``. Returns
    ``(assembled, conflicts, staging_head)`` where ``assembled`` are the members
    that merged cleanly (in order), ``conflicts`` are ``(worktree_id, error)`` for
    members that conflicted with the partial assembly (aborted and skipped, so the
    rest still proceed), and ``staging_head`` is the resulting tip (a descendant of
    ``base_head``). The entrypoint is never touched.
    """

    _sync_staging_to_head(staging, base_head)
    assembled: list[tuple[str, str]] = []
    conflicts: list[tuple[str, subprocess.CalledProcessError]] = []
    for worktree_id, head in members:
        try:
            _run_git(
                staging,
                "merge",
                "--no-edit",
                "-m",
                f"async orchestrator worktree {worktree_id}",
                head,
            )
        except subprocess.CalledProcessError as exc:
            # Abort just this member's merge; the partial assembly (prior members)
            # stays intact and the rest are still attempted.
            _run_git(staging, "merge", "--abort", check=False)
            conflicts.append((worktree_id, exc))
            continue
        assembled.append((worktree_id, head))
    return assembled, conflicts, _current_head(staging)


# A Lean diagnostic names its location as ``<path>.lean:<line>:<col>``. The order
# relative to the severity word is tool-dependent: ``lake build`` prints severity
# first (``error: RiemannSurface/Foo.lean:12:0: unsolved goals``) while a direct
# ``lean`` invocation prints it last (``…/Foo.lean:12:0: error: …``). Match the
# location token alone and decide error-vs-warning per line, so both orderings
# are handled and the (often numerous) warning lines are excluded.
_LEAN_LOCATION_RE: Pattern[str] = compile(r"([A-Za-z0-9_./-]+\.lean):\d+:\d+")


def _failed_lean_files(failure: _ValidationCommandFailure) -> set[str]:
    """The ``.lean`` files Lean reported *errors* in, from a build failure.

    Scans the full (untrimmed) build output line by line and collects the file
    from any line that carries an error location and the word ``error`` but not
    ``warning`` — order-agnostic across lake's ``error: <path>…`` and lean's
    ``<path>…: error`` forms, and skipping warning lines. Used to attribute a
    batched build failure to the worktree(s) that touched those files. Empty when
    no error names a file (timeouts, linker/build-level errors, or an error only
    in a downstream file the batch did not touch) — the caller then falls back to
    halving.
    """

    files: set[str] = set()
    joined = "\n".join(part for part in (failure.stdout, failure.stderr) if part)
    for line in joined.splitlines():
        lowered = line.lower()
        if "error" not in lowered or "warning" in lowered:
            continue
        for match in _LEAN_LOCATION_RE.finditer(line):
            files.add(match.group(1).lstrip("./"))
    return files


# A lake per-module job line: an optional status mark (``✔``/``✖``/``⚠``/``ℹ``),
# an optional ``[i/n]`` job counter, a job verb, and the module name. Observed
# forms (Lean 4.31 lake, plus the older counter-first job-start style):
#   ``✔ [2/5] Built Probe.Fast (208ms)``
#   ``⚠ [8477/8618] Replayed RiemannSurface.PartII.Foo``
#   ``✖ [3/5] Building Probe.Slow (186ms)``   (failed/killed job header)
#   ``[123/456] Building Mathlib.Data.List.Basic``
_LAKE_MODULE_JOB_RE: Pattern[str] = compile(
    r"^\s*(?:[^\w\s\[]{1,3}\s+)?(?:\[\d+/\d+\]\s+)?(Building|Built|Replayed)\s+"
    r"([A-Za-z_«][\w«»'.]*)"
)


def _lake_module_progress(failure: _ValidationCommandFailure) -> tuple[set[str], set[str]]:
    """Parse lake per-module progress from build output into source-file sets.

    Returns ``(in_flight, completed)`` as repo-relative ``.lean`` paths mapped
    from module names (``CFT.Cup.X`` -> ``CFT/Cup/X.lean``). ``completed`` holds
    modules lake reported done (``Built``/``Replayed``); ``in_flight`` holds
    modules named by a ``Building`` header with no completion line — on current
    lake that header is only printed for a job that failed (e.g. was killed at a
    timeout), on older/verbose lake it is printed at job start. Both sets are
    best-effort: output truncated at kill time may name no in-flight module at
    all. Used by :meth:`AsyncOrchestrator._timed_out_reported_paths` to
    attribute a timed-out batch build (issue #133).
    """

    building: set[str] = set()
    completed: set[str] = set()
    joined = "\n".join(part for part in (failure.stdout, failure.stderr) if part)
    for line in joined.splitlines():
        match = _LAKE_MODULE_JOB_RE.match(line)
        if match is None:
            continue
        verb, module = match.groups()
        (building if verb == "Building" else completed).add(_module_source_path(module))
    return building - completed, completed


def _module_source_path(module: str) -> str:
    """Repo-relative ``.lean`` source path for a dotted Lean module name."""

    return module.replace(".", "/") + ".lean"


# Prefix executables that wrap another command rather than being the command:
# ``taskset -c 0-31 lake build`` is still a lake invocation.
_LAKE_WRAPPER_BASENAMES: frozenset[str] = frozenset(
    {"taskset", "nice", "ionice", "env", "timeout", "stdbuf", "chrt", "numactl", "setsid"}
)


def _is_lake_invocation(argv: tuple[str, ...]) -> bool:
    """Whether a validation command's *executable* is lake, behind known wrappers.

    The timed-out-build attribution heuristics parse lake-specific progress
    lines, so they must only run for commands whose executable is lake. The
    rule: ``argv[0]``'s basename must be ``lake`` (direct calls, absolute paths
    like ``/usr/bin/lake``), or a known wrapper executable — in which case the
    remaining tokens are scanned for a ``lake`` basename, skipping option-like
    tokens (leading ``-`` or containing ``=``, which also covers ``env``
    assignments) and tolerating wrapper option values (``taskset -c 0-31``,
    ``timeout 30s``) whose arity we cannot know. A mere *argument* named lake
    (``validator --config lake``) does not match, and a shell-wrapped call
    (``sh -c "lake build"``) is not recognized — the miss only means falling
    back to bisection, the safe direction. The residual false positive (a
    non-lake command run under a known wrapper with a later bare ``lake``
    token) is accepted; it also only affects probe order, never who bounces.
    """

    if not argv:
        return False
    first = PurePosixPath(argv[0]).name
    if first == "lake":
        return True
    if first not in _LAKE_WRAPPER_BASENAMES:
        return False
    for token in argv[1:]:
        if token.startswith("-") or "=" in token:
            continue
        if PurePosixPath(token).name == "lake":
            return True
    return False


def _worktree_touched_files(repo: Path, base_head: str, head: str) -> set[str]:
    """Repo-relative files a worktree's own commits changed (diff from fork point).

    Diffs ``merge-base(base_head, head)..head`` so only the worktree's *own*
    contribution counts, not files that merely differ because ``main`` moved on.
    """

    try:
        merge_base = _run_git(repo, "merge-base", base_head, head).stdout.strip() or base_head
    except subprocess.CalledProcessError:
        merge_base = base_head
    try:
        # Disable rename detection so a rename is represented as deletion(old)
        # plus addition(new). Validation can report the pre-merge path of a task
        # whose id disappeared; retaining that old path is required to attribute
        # a rename-plus-id-change to the responsible member.
        out = _run_git(
            repo,
            "diff",
            "--name-only",
            "--no-renames",
            "-z",
            merge_base,
            head,
        ).stdout
    except subprocess.CalledProcessError:
        return set()
    return {path for path in out.split("\0") if path}


def _paths_match(a: str, b: str) -> bool:
    """Whether two paths refer to the same file, tolerant of prefix differences.

    Validation output may name a file by an absolute or staging-rooted path while
    ``git diff`` yields a repo-relative one; treat a path that is a trailing
    path-segment suffix of the other as a match.
    """

    a = a.lstrip("./")
    b = b.lstrip("./")
    return a == b or a.endswith("/" + b) or b.endswith("/" + a)


def _publish_validated_head(
    *,
    entrypoint: Path,
    target_branch: str,
    validated_head: str,
) -> None:
    """Fast-forward ``target_branch`` in the entrypoint to an already-validated commit.

    ``--ff-only`` because ``validated_head`` was built directly on the
    entrypoint's current tip in the staging worktree; the pristine entrypoint is
    only ever advanced, never reset/reverted.
    """

    _run_git(entrypoint, "checkout", target_branch)
    _run_git(entrypoint, "merge", "--ff-only", validated_head)


def _check_post_merge_task_tree(
    *,
    entrypoint: Path,
    original_head: str,
) -> _TaskValidationFailure | None:
    """Validate the post-merge ``tasks/`` tree; return a failure or ``None``.

    When the merge touched the task directory, the post-merge tree must parse
    strictly and form an acyclic
    DAG: a malformed YAML file or a dependency cycle returns a failure so the
    caller can roll the merge back like a post-merge build failure. When the
    merge did not touch ``tasks/`` the gate is skipped entirely so an unrelated
    contribution is never blamed for a pre-existing problem under ``tasks/``.

    The gate iterates files individually rather than calling
    ``load_tasks_strict`` so that the offending file path can be surfaced in
    the failure dataclass (and therefore the discussion message that the
    worker session sees on PENDING).

    An offending task id with no declaring file in the post-merge tree (a
    dependency orphaned by *deleting* the file that declared it) is resolved
    against the pre-merge tree at ``original_head``, so the failure names the
    deleted path and the batched merge can attribute it to the member whose
    diff deleted it.
    """

    if not _merge_touched_task_directory(
        entrypoint=entrypoint,
        original_head=original_head,
    ):
        return None
    return validate_task_directory(
        task_directory(entrypoint),
        resolve_missing_task_id=lambda task_id: _task_paths_declaring_id_at_head(
            entrypoint, original_head, task_id
        ),
    )


def _task_paths_declaring_id_at_head(repo: Path, head: str, task_id: str) -> tuple[str, ...]:
    """Repo-relative task file(s) that declared ``task_id`` in the tree at ``head``.

    Backs the unknown-dependency attribution of :func:`_check_post_merge_task_tree`:
    the missing id has no declaring file in the post-merge tree, so blame the
    file that declared it pre-merge — a member whose diff *deleted* that path is
    then attributed by ``_members_touching`` (deletions appear in
    ``git diff --name-only``). Parsing is deliberately lenient (any YAML mapping
    with a matching ``id``): the pre-merge tree already passed the strict gate,
    and a lookup miss only means falling back to bisection. Synchronous git
    calls — call via ``to_thread``.
    """

    try:
        listing = _run_git(
            repo,
            "ls-tree",
            "--name-only",
            "-z",
            head,
            "--",
            f"{TASKS_DIRECTORY_NAME}/",
        )
    except subprocess.CalledProcessError:
        return ()
    out: list[str] = []
    # NUL-delimited output disables Git's C-quoting and preserves unusual path
    # bytes via _run_git's surrogateescape decoding. Do not strip paths: leading
    # or trailing whitespace can be part of a valid filename.
    for path in listing.stdout.split("\0"):
        if not path or not fnmatch(PurePosixPath(path).name, DEFAULT_TASK_FILE_GLOB):
            continue
        try:
            text = _run_git(repo, "show", f"{head}:{path}").stdout
        except subprocess.CalledProcessError:
            continue
        try:
            data = read_yaml_config_data(text, path=path, kind="async orchestrator task")
        except ConfigFileError:
            continue
        if isinstance(data, dict) and data.get("id") == task_id:
            out.append(path)
    return tuple(out)


def _merge_diff_paths(*, entrypoint: Path, original_head: str) -> tuple[str, ...]:
    """Return changed paths losslessly from Git's NUL-delimited byte output."""

    diff_output = _run_git_bytes(
        entrypoint,
        "diff",
        "--name-only",
        "-z",
        "--no-renames",
        f"{original_head}..HEAD",
    ).stdout
    # Git's -z format is raw bytes separated and terminated by NUL. fsdecode
    # uses surrogateescape on POSIX, preserving every legal non-NUL filename
    # byte without allowing strict UTF-8 decoding to crash the orchestrator.
    return tuple(os.fsdecode(path) for path in diff_output.split(b"\0") if path)


def _merge_touched_task_directory(
    *,
    entrypoint: Path,
    original_head: str,
) -> bool:
    """Return ``True`` when the merge changed anything under ``tasks/``.

    Touch detection is a deliberate superset: any change anywhere under the
    task directory arms the gate, which then validates exactly the file set
    the scheduler reads. Validating on a change the scheduler would not scan
    is harmless; missing a relevant change is not. Raw paths use the same
    lossless parser as the task-only detector so Git quoting cannot make the
    two gates disagree.
    """

    try:
        paths = _merge_diff_paths(
            entrypoint=entrypoint,
            original_head=original_head,
        )
        return any(_is_task_path(path) for path in paths)
    except subprocess.CalledProcessError as exc:
        # A diff infrastructure failure is treated as "could not prove the
        # merge did not touch tasks/", so the gate is armed: better to
        # validate spuriously than to skip a relevant change.
        _LOGGER.warning(
            "post-merge task-touch diff failed in %s: %s",
            entrypoint,
            _called_process_error_summary(exc),
        )
        return True
    except Exception as exc:
        # Parsing/classification must fail toward running task validation too.
        _LOGGER.warning(
            "post-merge task-touch diff could not be parsed in %s: %s",
            entrypoint,
            exc,
        )
        return True


def _merge_changed_only_task_paths(
    *,
    entrypoint: Path,
    original_head: str,
) -> bool:
    """Return ``True`` when the merge changed at least one path, all under ``tasks/``.

    Powers the optional task-only build skip
    (``skip_build_validation_for_task_only_merges``). Enabling the option is an
    **operator assertion** that nothing in the configured validation commands
    consumes files under the task directory (e.g. Lean ``include_str
    "tasks/..."`` or custom Lake facets reading task files) — tend does not
    verify that assertion; when it does not hold, do not enable the option.
    Counterpart of :func:`_merge_touched_task_directory` with the opposite
    conservative default: a diff infrastructure failure — or an **empty** diff,
    which proves nothing about the merge — returns ``False`` ("could not prove
    the merge is task-only"), so the build gate still runs. Paths are read
    NUL-delimited and never trimmed, so legal filenames with leading/trailing
    whitespace classify exactly as git reports them. Better to build spuriously
    than to skip validation of a non-task change.
    """

    try:
        paths = _merge_diff_paths(
            entrypoint=entrypoint,
            original_head=original_head,
        )
        if not paths:
            return False
        return all(_is_task_path(path) for path in paths)
    except subprocess.CalledProcessError as exc:
        _LOGGER.warning(
            "task-only merge diff failed in %s: %s",
            entrypoint,
            _called_process_error_summary(exc),
        )
        return False
    except Exception as exc:
        # Any unexpected decoding, parsing, or classification failure means we
        # could not prove the merge task-only, so conservatively run the build.
        _LOGGER.warning(
            "task-only merge diff could not be classified in %s: %s",
            entrypoint,
            exc,
        )
        return False


def _git_status_porcelain(repo: Path) -> str:
    return _git_stdout(repo, "status", "--porcelain")


def _worktree_dirty_status_excluding_orchestrator_metadata(worktree: Path) -> str:
    """Return dirty worktree status excluding async orchestrator metadata."""

    return _git_stdout(
        worktree,
        "status",
        "--porcelain",
        "--",
        ".",
        _ORCHESTRATOR_GIT_METADATA_EXCLUDE_PATHSPEC,
    )


def _validation_failure_discussion_message(failure: _ValidationCommandFailure) -> str:
    lines = [
        "Validation failed; this worktree has been returned to the worker queue.",
        "Please fix the validation failure, then summarize the fix for review.",
        "",
        f"Command: `{_format_command(failure.argv)}`",
    ]
    _append_validation_failure_details(lines, failure)
    return "\n".join(lines)


def _pre_merge_validation_failure_discussion_message(
    failure: _ValidationCommandFailure,
    *,
    rollback_failure: subprocess.CalledProcessError | None,
    staged: bool = False,
) -> str:
    lines = [
        "Pre-merge validation failed; this worktree has been returned to the worker queue.",
        "The merge was treated as failed. Please fix the worktree so the merged result "
        "passes validation, then summarize the fix for review.",
    ]
    if staged:
        lines.append(
            "The staged trial merge was discarded; the entrypoint repository was left untouched."
        )
    elif rollback_failure is None:
        lines.append("The entrypoint repository was reset to its pre-merge HEAD.")
    else:
        lines.append(
            "Rollback to the pre-merge HEAD failed; inspect and repair the entrypoint "
            "repository before retrying."
        )
    lines.extend(("", f"Command: `{_format_command(failure.argv)}`"))
    _append_validation_failure_details(lines, failure)
    if rollback_failure is not None:
        lines.extend(
            (
                "",
                "Rollback failure:",
                f"Command: `{_format_command(rollback_failure.cmd)}`",
                f"Exit code: {rollback_failure.returncode}",
            )
        )
        stdout = _process_output_text(rollback_failure.stdout)
        stderr = _process_output_text(rollback_failure.stderr)
        if stdout:
            lines.extend(("", "Rollback stdout:", "```", _trim_text(stdout), "```"))
        if stderr:
            lines.extend(("", "Rollback stderr:", "```", _trim_text(stderr), "```"))
    return "\n".join(lines)


def _append_validation_failure_details(
    lines: list[str],
    failure: _ValidationCommandFailure,
) -> None:
    if failure.returncode is not None:
        lines.append(f"Exit code: {failure.returncode}")
    if failure.error:
        lines.extend(("", "Error:", "```", _trim_text(failure.error), "```"))
    stdout = _process_output_text(failure.stdout)
    stderr = _process_output_text(failure.stderr)
    if stdout:
        lines.extend(("", "Stdout:", "```", _trim_text(stdout), "```"))
    if stderr:
        lines.extend(("", "Stderr:", "```", _trim_text(stderr), "```"))


def _task_validation_failure_discussion_message(
    failure: _TaskValidationFailure,
    *,
    rollback_failure: subprocess.CalledProcessError | None,
    original_head: str,
    staged: bool = False,
) -> str:
    lines = [
        "Post-merge task validation failed; this worktree has been returned to the worker queue.",
        "The merge was treated as failed. Please fix the task files (valid YAML and an "
        "acyclic `depends_on` graph) on the worktree branch, then summarize the fix for review.",
    ]
    if staged:
        lines.append(
            "The staged trial merge was discarded; the entrypoint repository was left untouched."
        )
    elif rollback_failure is None:
        lines.append(
            f"The entrypoint repository was reset to its pre-merge HEAD `{original_head}`."
        )
    else:
        lines.append(
            "Rollback to the pre-merge HEAD `" + original_head + "` failed; inspect and repair "
            "the entrypoint repository before retrying."
        )
    if failure.offending_paths:
        # ``offending_paths`` remains complete for in-memory merge attribution,
        # but only a bounded, de-duplicated sample enters the durable worker
        # discussion message. Stop once one overflow item proves omission;
        # never build a full-size temporary dictionary or tuple.
        displayed_paths: list[str] = []
        displayed_seen: set[str] = set()
        has_more = False
        for index, path in enumerate(failure.offending_paths):
            if index == _MAX_DISCUSSION_OFFENDING_PATHS:
                has_more = True
                break
            if path not in displayed_seen:
                displayed_seen.add(path)
                displayed_paths.append(path)
        lines.extend(("", "Offending file(s):"))
        lines.extend(f"- `{path}`" for path in displayed_paths)
        if has_more:
            # Internal attribution producers already de-duplicate paths, so the
            # tuple length gives the exact normal-case omission count without
            # enumerating the untrusted remainder. For a manually-constructed
            # failure containing later duplicates this remains a safe upper
            # bound for human-readable output.
            omitted = len(failure.offending_paths) - len(displayed_paths)
            lines.append(f"- ... and {omitted} more")
    lines.extend(("", "Task validation error:", "```", failure.detail, "```"))
    lines.extend(
        (
            "",
            "A task file's allowed top-level keys are exactly: `schema_version`, "
            "`id`, `title`, `status` (`open` or `complete`), `priority` "
            "(`default`, `high`, or `max`), `depends_on`, `summary`, "
            "`description`, `notes`. `description` is the spec (what the task must "
            "accomplish); record any progress, findings, or blocker hand-off for "
            "the next worker in `notes` (a free-form string).",
        )
    )
    if rollback_failure is not None:
        lines.extend(
            (
                "",
                "Rollback failure:",
                f"Command: `{_format_command(rollback_failure.cmd)}`",
                f"Exit code: {rollback_failure.returncode}",
            )
        )
        stdout = _process_output_text(rollback_failure.stdout)
        stderr = _process_output_text(rollback_failure.stderr)
        if stdout:
            lines.extend(("", "Rollback stdout:", "```", _trim_text(stdout), "```"))
        if stderr:
            lines.extend(("", "Rollback stderr:", "```", _trim_text(stderr), "```"))
    return "\n".join(lines)


def _dirty_entrypoint_discussion_message(status: str) -> str:
    lines = [
        "Entrypoint repository is dirty; merge was not attempted.",
        "This worktree has been returned to the worker queue. Clean, commit, or stash "
        "the entrypoint changes, then retry.",
        "",
        "Entrypoint `git status --porcelain` output:",
        "```",
        _trim_text(status),
        "```",
    ]
    return "\n".join(lines)


def _dirty_worktree_before_review_discussion_message(status: str) -> str:
    lines = [
        "Uncommitted worktree changes detected; this worktree has been returned "
        "to the worker queue before review.",
        "Workers own their commits, and reviewers/validation only assess the "
        "committed tree that can later be merged. Commit the files you intend "
        "to land (and only those), or remove/revert unintended changes, then "
        "finish again.",
        "For a `blocked` result, commit only task-graph/progress edits under "
        f"`{TASKS_DIRECTORY_NAME}/`; do not commit work-in-progress code.",
        "",
        "Worktree `git status --porcelain` output (excluding `.tend/` "
        "orchestrator metadata):",
        "```",
        _trim_text(status),
        "```",
    ]
    return "\n".join(lines)


def _nothing_committed_discussion_message(target_branch: str) -> str:
    lines = [
        "Nothing to merge: this worktree has no commits beyond "
        f"`{target_branch}`, so there is nothing to land.",
        "",
        "Workers own their commits — uncommitted work is never merged. Any "
        "uncommitted changes are left untouched in your worktree, so you can "
        "still commit them. Commit the files you intend to land (and only "
        "those), then finish again. If there is genuinely nothing to commit, "
        "this task will simply be retried.",
    ]
    return "\n".join(lines)


def _entrypoint_status_failure_discussion_message(
    exc: subprocess.CalledProcessError,
) -> str:
    lines = [
        "Entrypoint repository status check failed; merge was not attempted.",
        "This worktree has been returned to the worker queue. Inspect and repair "
        "the entrypoint repository, then retry.",
        "",
        f"Command: `{_format_command(exc.cmd)}`",
        f"Exit code: {exc.returncode}",
    ]
    stdout = _process_output_text(exc.stdout)
    stderr = _process_output_text(exc.stderr)
    if stdout:
        lines.extend(("", "Stdout:", "```", _trim_text(stdout), "```"))
    if stderr:
        lines.extend(("", "Stderr:", "```", _trim_text(stderr), "```"))
    return "\n".join(lines)


def _merge_failure_discussion_message(
    exc: subprocess.CalledProcessError,
    *,
    target_branch: str,
) -> str:
    lines = [
        f"Merge into `{target_branch}` failed — typically a merge race: another "
        f"worker's contribution merged into `{target_branch}` while this worktree "
        f"was being worked on, and the new commits conflict with this branch.",
        "",
        f"**Your committed work and session context are preserved — do not redo "
        f"the task.** Merge `{target_branch}` into this branch yourself "
        f"(`git merge {target_branch}` or `git rebase {target_branch}`), resolve "
        f"any conflict markers (the conflicting files are named in the git "
        f"output below), commit, and then call `final_result` to signal the "
        f"contribution is ready again.",
        "",
        f"Failed command: `{_format_command(exc.cmd)}`",
        f"Exit code: {exc.returncode}",
    ]
    stdout = _process_output_text(exc.stdout)
    stderr = _process_output_text(exc.stderr)
    if stdout:
        lines.extend(("", "Stdout:", "```", _trim_text(stdout), "```"))
    if stderr:
        lines.extend(("", "Stderr:", "```", _trim_text(stderr), "```"))
    return "\n".join(lines)


def _called_process_error_summary(exc: subprocess.CalledProcessError) -> str:
    output = _process_output_text(exc.stderr) or _process_output_text(exc.stdout)
    if output:
        return _trim_text(output, max_length=500).replace("\n", " | ")
    return f"command exited with code {exc.returncode}: {_format_command(exc.cmd)}"


def _completed_process_error_summary(completed: subprocess.CompletedProcess[str]) -> str:
    output = _process_output_text(completed.stderr) or _process_output_text(
        completed.stdout
    )
    if output:
        return _trim_text(output, max_length=500).replace("\n", " | ")
    return (
        f"command exited with code {completed.returncode}: "
        f"{_format_command(completed.args)}"
    )


def _format_command(command: object) -> str:
    if isinstance(command, Sequence) and not isinstance(command, (str, bytes, bytearray)):
        return " ".join(str(part) for part in cast(Sequence[object], command))
    return str(command)


def _process_output_text(output: object) -> str:
    if isinstance(output, bytes):
        return output.decode(errors="replace").strip()
    if isinstance(output, str):
        return output.strip()
    return ""


def _trim_text(text: str, *, max_length: int = 4000) -> str:
    if len(text) <= max_length:
        return text
    return f"{text[:max_length]}\n... <truncated>"


def _utf8_persistable_text(text: str) -> str:
    """Escape lone surrogates while preserving ordinary Unicode text."""

    return text.encode("utf-8", "backslashreplace").decode("utf-8")


def _git_stdout(repo: Path, *args: str | Path) -> str:
    return _run_git(repo, *args).stdout.strip()


def _run_git(
    repo: Path,
    *args: str | Path,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *(str(arg) for arg in args)],
        cwd=repo,
        check=check,
        capture_output=True,
        text=True,
        # Git's NUL-delimited filename output contains raw path bytes. Preserve
        # every non-UTF-8 byte as a surrogate so splitting and later subprocess
        # arguments round-trip through the filesystem encoding without crashes.
        errors="surrogateescape",
    )


def _run_git_bytes(
    repo: Path,
    *args: str | Path,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    """Run Git without text decoding for commands that return raw paths."""

    return subprocess.run(
        ["git", *(str(arg) for arg in args)],
        cwd=repo,
        check=check,
        capture_output=True,
    )


def _require_control_params(
    command: str,
    params: Mapping[str, object],
    *,
    allowed: tuple[str, ...],
) -> None:
    """Reject unexpected command params before mutating runtime state."""

    allowed_keys = set(allowed)
    extra_keys = sorted(set(params) - allowed_keys)
    if extra_keys:
        joined = ", ".join(extra_keys)
        raise _ControlCommandApplicationError(
            f"{command} command has unsupported parameter(s): {joined}"
        )


def _optional_control_int_param(
    params: Mapping[str, object],
    name: str,
) -> int | None:
    if name not in params:
        return None
    value = params[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise _ControlCommandApplicationError(f"{name} must be a non-negative integer")
    if value < 0:
        raise _ControlCommandApplicationError(f"{name} must be a non-negative integer")
    return value


def _control_bool_param(
    params: Mapping[str, object],
    name: str,
    *,
    default: bool,
) -> bool:
    if name not in params:
        return default
    value = params[name]
    if not isinstance(value, bool):
        raise _ControlCommandApplicationError(f"{name} must be a boolean")
    return value


def _control_decimal_param(params: Mapping[str, object], name: str) -> Decimal:
    value = params.get(name)
    if not isinstance(value, str):
        raise _ControlCommandApplicationError(f"{name} must be a decimal string")
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise _ControlCommandApplicationError(f"{name} must be a decimal string") from exc
    if not parsed.is_finite():
        raise _ControlCommandApplicationError(f"{name} must be finite")
    if parsed <= 0:
        raise _ControlCommandApplicationError(f"{name} must be greater than 0")
    return parsed


def _task_id(task: Task | str | None) -> str | None:
    if task is None:
        return None
    if isinstance(task, Task):
        return task.id
    if not task.strip():
        raise ValueError("task ID must not be blank")
    return task
