"""Runtime-only coordination state for the async orchestrator."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Coroutine, Iterable
from typing import Literal

from tend.orchestrator.queues import RuntimeQueue
from tend.orchestrator.state import AsyncOrchestratorWorktree, WorktreeState

_LOGGER = logging.getLogger(__name__)


class AsyncOrchestratorRuntime:
    """Queues, locks, running tasks, and live controls outside durable state."""

    __slots__ = (
        "_draining",
        "_paused",
        "_reviewer_agent_limit",
        "_stopping",
        "_worker_agent_limit",
        "entrypoint_lock",
        "merge_lock",
        "merge_queue",
        "review_queue",
        "reviewer_agent_tasks",
        "task_queue",
        "worker_agent_tasks",
        "worker_queue",
        "worktree_creation_lock",
    )

    worktree_creation_lock: asyncio.Lock
    # Serializes entrypoint repository access by ready-task discovery and merges.
    merge_lock: asyncio.Lock
    # Guards mutations of (and consistent reads from) the pristine entrypoint
    # working tree + HEAD: the merge's publish step and ready-task worktree
    # creation. Distinct from ``merge_lock`` so that the staging-worktree
    # validation build (which never touches the entrypoint) does not block
    # worktree creation while it runs. Only meaningful when the staging
    # validation worktree is enabled; otherwise the legacy in-entrypoint merge
    # path keeps using ``merge_lock`` for both.
    entrypoint_lock: asyncio.Lock
    task_queue: RuntimeQueue
    worker_queue: RuntimeQueue
    review_queue: RuntimeQueue
    merge_queue: RuntimeQueue
    worker_agent_tasks: dict[str, asyncio.Task[None]]
    reviewer_agent_tasks: dict[str, asyncio.Task[None]]
    _worker_agent_limit: int
    _reviewer_agent_limit: int

    def __init__(
        self,
        worktrees: Iterable[AsyncOrchestratorWorktree] = (),
        *,
        worker_agent_limit: int = 20,
        reviewer_agent_limit: int = 20,
    ) -> None:
        self.worktree_creation_lock = asyncio.Lock()
        self.merge_lock = asyncio.Lock()
        self.entrypoint_lock = asyncio.Lock()
        self.task_queue = RuntimeQueue()
        self.worker_queue = RuntimeQueue()
        self.review_queue = RuntimeQueue()
        self.merge_queue = RuntimeQueue()
        self.worker_agent_tasks = {}
        self.reviewer_agent_tasks = {}
        self._worker_agent_limit = _validate_agent_limit(worker_agent_limit)
        self._reviewer_agent_limit = _validate_agent_limit(reviewer_agent_limit)
        self._paused = False
        self._draining = False
        self._stopping = False
        for worktree in worktrees:
            self.enqueue_worktree_for_state(worktree)

    @property
    def paused(self) -> bool:
        """Return whether cost-incurring schedulers are paused."""

        return self._paused

    @property
    def draining(self) -> bool:
        """Return whether the run is gracefully draining toward terminal exit."""

        return self._draining

    @property
    def stopping(self) -> bool:
        """Return whether an immediate stop has been requested."""

        return self._stopping

    @property
    def worker_agent_limit(self) -> int:
        """Return the current effective worker-agent launch limit."""

        return self._worker_agent_limit

    @property
    def reviewer_agent_limit(self) -> int:
        """Return the current effective reviewer-agent launch limit."""

        return self._reviewer_agent_limit

    def set_worker_agent_limit(self, limit: int) -> None:
        """Set the current effective worker-agent launch limit and wake schedulers."""

        limit = _validate_agent_limit(limit)
        if limit == self._worker_agent_limit:
            return
        self._worker_agent_limit = limit
        self.worker_queue.notify_changed()

    def set_reviewer_agent_limit(self, limit: int) -> None:
        """Set the current effective reviewer-agent launch limit and wake schedulers."""

        limit = _validate_agent_limit(limit)
        if limit == self._reviewer_agent_limit:
            return
        self._reviewer_agent_limit = limit
        self.review_queue.notify_changed()

    def set_paused(self, paused: bool) -> None:
        """Pause or resume non-terminal cost-incurring scheduling."""

        if paused == self._paused:
            return
        self._paused = paused
        self.notify_scheduler_change()

    def request_drain(self) -> None:
        """Request a graceful terminal drain of in-flight work."""

        if self._draining:
            return
        self._draining = True
        self.notify_scheduler_change()

    def request_stop(self) -> None:
        """Request a terminal stop.

        The first external stop command will add active cancellation semantics;
        for now this flag shares the same admission closure and wakeup path as
        graceful drain so tests can exercise the policy without a control DB.
        """

        if self._stopping:
            return
        self._stopping = True
        self.notify_scheduler_change()

    def notify_scheduler_change(self) -> None:
        """Wake schedulers blocked on queues, limits, or run-control state."""

        self.task_queue.notify_changed()
        self.worker_queue.notify_changed()
        self.review_queue.notify_changed()
        self.merge_queue.notify_changed()

    def active_agent_worktree_ids(
        self,
        role: Literal["worker", "reviewer"] | None = None,
    ) -> tuple[str, ...]:
        """Return worktree IDs for currently running agent tasks."""

        if role == "worker":
            return _active_agent_worktree_ids(self.worker_agent_tasks)
        if role == "reviewer":
            return _active_agent_worktree_ids(self.reviewer_agent_tasks)
        return (
            *_active_agent_worktree_ids(self.worker_agent_tasks),
            *_active_agent_worktree_ids(self.reviewer_agent_tasks),
        )

    def active_agent_count(
        self,
        role: Literal["worker", "reviewer"] | None = None,
    ) -> int:
        """Return the number of currently running agent tasks."""

        return len(self.active_agent_worktree_ids(role))

    async def process_queue_item(
        self,
        queue: RuntimeQueue,
        handler: Callable[[str], Awaitable[object]],
        *,
        wait: bool,
        keep_reserved: bool = False,
        can_process: Callable[[], bool] | None = None,
    ) -> bool:
        """Process one queue item; return false only when no item can be processed.

        ``can_process`` is checked before a visible item is reserved/claimed, so
        pause/drain/budget admission gates do not consume queue items while they
        are closed. When a reserved handler notices that admission closed during
        its own awaits and returns ``None``, the item is released back to the
        front of the FIFO instead of being discarded.
        """

        def allowed() -> bool:
            return True if can_process is None else can_process()

        while True:
            if not allowed():
                if not wait:
                    return False
                version = queue.change_version
                await queue.wait_for_change_since(version)
                continue
            try:
                item = queue.get_nowait()
            except asyncio.QueueEmpty:
                if not wait:
                    return False
                version = queue.change_version
                await queue.wait_for_change_since(version)
                continue
            if not allowed():
                if not wait:
                    return False
                continue
            break

        handler_result: object = None
        handler_completed = False
        claimed = False
        try:
            claimed = queue.reserve(item) if keep_reserved else queue.claim(item)
            if claimed:
                handler_result = await handler(item)
                handler_completed = True
            return True
        finally:
            if keep_reserved:
                if claimed and handler_completed and handler_result is None and not allowed():
                    queue.release(item)
                else:
                    queue.discard(item)
            queue.task_done()

    async def spawn_agent_tasks_forever(
        self,
        *,
        queue: RuntimeQueue,
        tasks: dict[str, asyncio.Task[None]],
        get_max_concurrent: Callable[[], int],
        spawn_task: Callable[[str], None],
    ) -> None:
        """Continuously spawn agent tasks from a role queue."""

        while True:
            _prune_done_tasks(tasks)
            while len(tasks) < get_max_concurrent():
                try:
                    worktree_id = queue.claim_nowait()
                except asyncio.QueueEmpty:
                    break
                spawn_task(worktree_id)
                _prune_done_tasks(tasks)
            version = queue.change_version
            await queue.wait_for_change_since(version)

    async def spawn_agent_tasks_once(
        self,
        *,
        queue: RuntimeQueue,
        tasks: dict[str, asyncio.Task[None]],
        get_max_concurrent: Callable[[], int],
        spawn_task: Callable[[str], None],
    ) -> None:
        """Spawn currently queued agent tasks without waiting for more."""

        _prune_done_tasks(tasks)
        while len(tasks) < get_max_concurrent():
            try:
                worktree_id = queue.claim_nowait()
            except asyncio.QueueEmpty:
                return
            spawn_task(worktree_id)
            _prune_done_tasks(tasks)

    def spawn_agent_task(
        self,
        worktree_id: str,
        *,
        queue: RuntimeQueue,
        tasks: dict[str, asyncio.Task[None]],
        name: str,
        run: Callable[[str], Coroutine[object, object, None]],
    ) -> None:
        """Create one agent task and wire queue completion to task completion."""

        task: asyncio.Task[None] = asyncio.create_task(
            run(worktree_id),
            name=f"async-orchestrator-{name}-{worktree_id}",
        )
        task.add_done_callback(lambda _task: _agent_task_done(queue))
        tasks[worktree_id] = task
        _LOGGER.debug("spawned async %s agent task for worktree %s", name, worktree_id)

    async def cancel_agent_tasks(self, *, raise_failures: bool = False) -> None:
        """Cancel any worker/reviewer agent tasks owned by this runtime.

        Final process cleanup is best-effort and swallows failures, preserving
        the historical shutdown behavior. Operator ``stop --now`` uses
        ``raise_failures=True`` so a worker/reviewer task that already failed (or
        fails while unwinding cancellation) still tears down the run instead of
        being reported as a successful operator stop.
        """

        tasks = [*self.worker_agent_tasks.values(), *self.reviewer_agent_tasks.values()]
        if not tasks:
            return
        _LOGGER.info("cancelling %d async agent task(s)", len(tasks))
        for task in tasks:
            task.cancel()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        self.worker_agent_tasks.clear()
        self.reviewer_agent_tasks.clear()
        self.worker_queue.notify_changed()
        self.review_queue.notify_changed()
        if raise_failures:
            for result in results:
                if isinstance(result, asyncio.CancelledError):
                    continue
                if isinstance(result, BaseException):
                    raise result

    def enqueue_worktree_for_state(self, worktree: AsyncOrchestratorWorktree) -> None:
        """Route a worktree ID to the runtime queue matching its current state."""

        if worktree.state is WorktreeState.PENDING and worktree.task_id is not None:
            self.worker_queue.enqueue(worktree.worktree_id)
        elif worktree.state is WorktreeState.REVIEW:
            self.review_queue.enqueue(worktree.worktree_id)
        elif worktree.state is WorktreeState.MERGE:
            self.merge_queue.enqueue(worktree.worktree_id)

    def discard_worktree_id(self, worktree_id: str) -> None:
        """Remove a worktree ID from all runtime queues."""

        self.worker_queue.discard(worktree_id)
        self.review_queue.discard(worktree_id)
        self.merge_queue.discard(worktree_id)


def prune_done_agent_tasks(tasks: dict[str, asyncio.Task[None]]) -> None:
    """Remove completed agent tasks from ``tasks`` and re-raise their failures.

    Exposed for use outside the spawner loop (e.g. the budget-stop settle check)
    so a queue-blocked spawner can't leave a ``done()`` entry pinned in the dict
    forever. Cancellations are swallowed; any other exception is re-raised so the
    enclosing TaskGroup still tears the run down on agent failure.
    """

    for worktree_id, task in tuple(tasks.items()):
        if not task.done():
            continue
        del tasks[worktree_id]
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception:
            _LOGGER.exception("async agent task failed for worktree %s", worktree_id)
            raise


# Backwards-compatible private alias for the in-module spawn loops.
_prune_done_tasks = prune_done_agent_tasks


def _active_agent_worktree_ids(tasks: dict[str, asyncio.Task[None]]) -> tuple[str, ...]:
    return tuple(worktree_id for worktree_id, task in tasks.items() if not task.done())


def _agent_task_done(queue: RuntimeQueue) -> None:
    queue.task_done()
    queue.notify_changed()


def _validate_agent_limit(limit: int) -> int:
    if limit < 0:
        raise ValueError("agent concurrency limit must be non-negative")
    return limit
