from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import suppress

from tend.orchestrator.queues import RuntimeQueue
from tend.orchestrator.runtime import AsyncOrchestratorRuntime


def test_runtime_queue_orders_items_by_priority_then_fifo() -> None:
    queue = RuntimeQueue()
    queue.enqueue("default-1", priority=2)
    queue.enqueue("high-1", priority=1)
    queue.enqueue("default-2", priority=2)
    queue.enqueue("max-1", priority=0)
    queue.enqueue("high-2", priority=1)

    assert queue.items == (
        "max-1",
        "high-1",
        "high-2",
        "default-1",
        "default-2",
    )

    queue.enqueue("default-2", priority=0)
    assert queue.items == ("max-1", "default-2", "high-1", "high-2", "default-1")


def test_runtime_queue_release_restores_reserved_item_to_front_without_duplicates() -> None:
    queue = RuntimeQueue()
    queue.enqueue("first")
    queue.enqueue("reserved")
    queue.enqueue("third")

    assert queue.reserve("reserved") is True
    assert queue.items == ("first", "third")

    queue.enqueue("reserved")
    assert queue.items == ("first", "third")

    assert queue.release("reserved") is True
    assert queue.items == ("reserved", "first", "third")

    assert queue.release("reserved") is False
    queue.enqueue("reserved")
    assert queue.items == ("reserved", "first", "third")


async def test_process_queue_item_releases_reserved_item_when_admission_closes() -> None:
    runtime = AsyncOrchestratorRuntime()
    queue = RuntimeQueue()
    can_process = True
    queue.enqueue("ready-task")
    queue.enqueue("next-task")

    async def handler(item: str) -> object:
        nonlocal can_process
        assert item == "ready-task"
        assert queue.items == ("next-task",)
        can_process = False
        return None

    processed = await runtime.process_queue_item(
        queue,
        handler,
        wait=False,
        keep_reserved=True,
        can_process=lambda: can_process,
    )

    assert processed is True
    assert queue.items == ("ready-task", "next-task")
    assert queue.has_claimed_items is True


async def test_process_queue_item_discards_reserved_item_when_handler_consumes_it() -> None:
    runtime = AsyncOrchestratorRuntime()
    queue = RuntimeQueue()
    can_process = True
    queue.enqueue("ready-task")
    queue.enqueue("next-task")

    async def handler(item: str) -> object:
        nonlocal can_process
        assert item == "ready-task"
        can_process = False
        return object()

    processed = await runtime.process_queue_item(
        queue,
        handler,
        wait=False,
        keep_reserved=True,
        can_process=lambda: can_process,
    )

    assert processed is True
    assert queue.items == ("next-task",)
    assert "ready-task" not in queue


async def test_process_queue_item_discards_reserved_item_when_admission_stays_open() -> None:
    runtime = AsyncOrchestratorRuntime()
    queue = RuntimeQueue()
    queue.enqueue("ready-task")
    queue.enqueue("next-task")

    async def handler(item: str) -> object:
        assert item == "ready-task"
        return None

    processed = await runtime.process_queue_item(
        queue,
        handler,
        wait=False,
        keep_reserved=True,
        can_process=lambda: True,
    )

    assert processed is True
    assert queue.items == ("next-task",)
    assert "ready-task" not in queue


async def test_spawn_agents_once_honors_zero_limit_without_claiming_queue() -> None:
    runtime = AsyncOrchestratorRuntime(worker_agent_limit=0)
    runtime.worker_queue.enqueue("worktree-1")

    await runtime.spawn_agent_tasks_once(
        queue=runtime.worker_queue,
        tasks=runtime.worker_agent_tasks,
        get_max_concurrent=lambda: runtime.worker_agent_limit,
        spawn_task=lambda _worktree_id: None,
    )

    assert runtime.worker_agent_tasks == {}
    assert runtime.worker_queue.items == ("worktree-1",)


async def test_spawn_agents_once_uses_updated_limit() -> None:
    runtime = AsyncOrchestratorRuntime(worker_agent_limit=0)
    release = asyncio.Event()
    runtime.worker_queue.enqueue("worktree-1")
    runtime.worker_queue.enqueue("worktree-2")

    async def run_agent(_worktree_id: str) -> None:
        await release.wait()

    def spawn_agent(worktree_id: str) -> None:
        runtime.spawn_agent_task(
            worktree_id,
            queue=runtime.worker_queue,
            tasks=runtime.worker_agent_tasks,
            name="test-worker",
            run=run_agent,
        )

    try:
        runtime.set_worker_agent_limit(1)
        await runtime.spawn_agent_tasks_once(
            queue=runtime.worker_queue,
            tasks=runtime.worker_agent_tasks,
            get_max_concurrent=lambda: runtime.worker_agent_limit,
            spawn_task=spawn_agent,
        )

        assert set(runtime.worker_agent_tasks) == {"worktree-1"}
        assert runtime.worker_queue.items == ("worktree-2",)

        runtime.set_worker_agent_limit(2)
        await runtime.spawn_agent_tasks_once(
            queue=runtime.worker_queue,
            tasks=runtime.worker_agent_tasks,
            get_max_concurrent=lambda: runtime.worker_agent_limit,
            spawn_task=spawn_agent,
        )

        assert set(runtime.worker_agent_tasks) == {"worktree-1", "worktree-2"}
        assert runtime.worker_queue.items == ()
    finally:
        release.set()
        await _drain_agent_tasks(runtime.worker_agent_tasks)


async def test_lowering_agent_limit_suppresses_new_launches_without_cancelling_active() -> None:
    runtime = AsyncOrchestratorRuntime(worker_agent_limit=2)
    release = asyncio.Event()
    runtime.worker_queue.enqueue("worktree-1")
    runtime.worker_queue.enqueue("worktree-2")

    async def run_agent(_worktree_id: str) -> None:
        await release.wait()

    def spawn_agent(worktree_id: str) -> None:
        runtime.spawn_agent_task(
            worktree_id,
            queue=runtime.worker_queue,
            tasks=runtime.worker_agent_tasks,
            name="test-worker",
            run=run_agent,
        )

    try:
        await runtime.spawn_agent_tasks_once(
            queue=runtime.worker_queue,
            tasks=runtime.worker_agent_tasks,
            get_max_concurrent=lambda: runtime.worker_agent_limit,
            spawn_task=spawn_agent,
        )
        runtime.worker_queue.enqueue("worktree-3")

        runtime.set_worker_agent_limit(1)
        await runtime.spawn_agent_tasks_once(
            queue=runtime.worker_queue,
            tasks=runtime.worker_agent_tasks,
            get_max_concurrent=lambda: runtime.worker_agent_limit,
            spawn_task=spawn_agent,
        )

        assert set(runtime.worker_agent_tasks) == {"worktree-1", "worktree-2"}
        assert all(not task.done() for task in runtime.worker_agent_tasks.values())
        assert runtime.worker_queue.items == ("worktree-3",)
    finally:
        release.set()
        await _drain_agent_tasks(runtime.worker_agent_tasks)


async def test_agent_completion_wakes_forever_scheduler_for_queued_work() -> None:
    runtime = AsyncOrchestratorRuntime(worker_agent_limit=1)
    release_first = asyncio.Event()
    release_second = asyncio.Event()
    runtime.worker_queue.enqueue("worktree-1")
    runtime.worker_queue.enqueue("worktree-2")

    async def run_agent(worktree_id: str) -> None:
        if worktree_id == "worktree-1":
            await release_first.wait()
            return
        await release_second.wait()

    def spawn_agent(worktree_id: str) -> None:
        runtime.spawn_agent_task(
            worktree_id,
            queue=runtime.worker_queue,
            tasks=runtime.worker_agent_tasks,
            name="test-worker",
            run=run_agent,
        )

    scheduler = asyncio.create_task(
        runtime.spawn_agent_tasks_forever(
            queue=runtime.worker_queue,
            tasks=runtime.worker_agent_tasks,
            get_max_concurrent=lambda: runtime.worker_agent_limit,
            spawn_task=spawn_agent,
        )
    )
    try:
        await _wait_until(lambda: set(runtime.worker_agent_tasks) == {"worktree-1"})
        assert runtime.worker_queue.items == ("worktree-2",)

        release_first.set()

        await _wait_until(lambda: set(runtime.worker_agent_tasks) == {"worktree-2"})
        assert runtime.worker_queue.items == ()
    finally:
        scheduler.cancel()
        with suppress(asyncio.CancelledError):
            await scheduler
        release_first.set()
        release_second.set()
        await _drain_agent_tasks(runtime.worker_agent_tasks)


async def test_raising_agent_limit_wakes_forever_scheduler_without_task_completion() -> None:
    runtime = AsyncOrchestratorRuntime(worker_agent_limit=1)
    release = asyncio.Event()
    runtime.worker_queue.enqueue("worktree-1")
    runtime.worker_queue.enqueue("worktree-2")

    async def run_agent(_worktree_id: str) -> None:
        await release.wait()

    def spawn_agent(worktree_id: str) -> None:
        runtime.spawn_agent_task(
            worktree_id,
            queue=runtime.worker_queue,
            tasks=runtime.worker_agent_tasks,
            name="test-worker",
            run=run_agent,
        )

    scheduler = asyncio.create_task(
        runtime.spawn_agent_tasks_forever(
            queue=runtime.worker_queue,
            tasks=runtime.worker_agent_tasks,
            get_max_concurrent=lambda: runtime.worker_agent_limit,
            spawn_task=spawn_agent,
        )
    )
    try:
        await _wait_until(lambda: set(runtime.worker_agent_tasks) == {"worktree-1"})
        assert runtime.worker_queue.items == ("worktree-2",)
        assert all(not task.done() for task in runtime.worker_agent_tasks.values())

        runtime.set_worker_agent_limit(2)

        await _wait_until(
            lambda: set(runtime.worker_agent_tasks) == {"worktree-1", "worktree-2"}
        )
        assert runtime.worker_queue.items == ()
    finally:
        scheduler.cancel()
        with suppress(asyncio.CancelledError):
            await scheduler
        release.set()
        await _drain_agent_tasks(runtime.worker_agent_tasks)


async def test_zero_limit_forever_scheduler_preserves_queue_until_resumed() -> None:
    runtime = AsyncOrchestratorRuntime(worker_agent_limit=0)
    release = asyncio.Event()
    runtime.worker_queue.enqueue("worktree-1")

    async def run_agent(_worktree_id: str) -> None:
        await release.wait()

    def spawn_agent(worktree_id: str) -> None:
        runtime.spawn_agent_task(
            worktree_id,
            queue=runtime.worker_queue,
            tasks=runtime.worker_agent_tasks,
            name="test-worker",
            run=run_agent,
        )

    scheduler = asyncio.create_task(
        runtime.spawn_agent_tasks_forever(
            queue=runtime.worker_queue,
            tasks=runtime.worker_agent_tasks,
            get_max_concurrent=lambda: runtime.worker_agent_limit,
            spawn_task=spawn_agent,
        )
    )
    try:
        await asyncio.sleep(0)
        assert runtime.worker_agent_tasks == {}
        assert runtime.worker_queue.items == ("worktree-1",)

        runtime.set_worker_agent_limit(1)

        await _wait_until(lambda: set(runtime.worker_agent_tasks) == {"worktree-1"})
        assert runtime.worker_queue.items == ()
    finally:
        scheduler.cancel()
        with suppress(asyncio.CancelledError):
            await scheduler
        release.set()
        await _drain_agent_tasks(runtime.worker_agent_tasks)


async def test_active_agent_helpers_ignore_completed_tasks() -> None:
    runtime = AsyncOrchestratorRuntime()
    release = asyncio.Event()

    async def running_agent() -> None:
        await release.wait()

    async def completed_agent() -> None:
        return

    running = asyncio.create_task(running_agent())
    completed = asyncio.create_task(completed_agent())
    await asyncio.sleep(0)
    runtime.worker_agent_tasks["worktree_000001"] = running
    runtime.reviewer_agent_tasks["worktree_000002"] = completed

    try:
        assert runtime.active_agent_worktree_ids("worker") == ("worktree_000001",)
        assert runtime.active_agent_worktree_ids("reviewer") == ()
        assert runtime.active_agent_worktree_ids() == ("worktree_000001",)
        assert runtime.active_agent_count() == 1
    finally:
        release.set()
        await running


async def test_cancel_agent_tasks_can_reraise_task_failures() -> None:
    runtime = AsyncOrchestratorRuntime()

    async def failed_agent() -> None:
        raise RuntimeError("agent failed")

    task = asyncio.create_task(failed_agent())
    await asyncio.sleep(0)
    runtime.worker_agent_tasks["worktree-1"] = task

    try:
        with suppress(RuntimeError):
            # First verify the best-effort final-cleanup path still swallows.
            await runtime.cancel_agent_tasks()
        runtime.worker_agent_tasks["worktree-1"] = task
        try:
            await runtime.cancel_agent_tasks(raise_failures=True)
        except RuntimeError as exc:
            assert str(exc) == "agent failed"
        else:  # pragma: no cover - assertion message for unexpected behavior
            raise AssertionError("expected failed task to be reraised")
    finally:
        runtime.worker_agent_tasks.clear()


async def _drain_agent_tasks(tasks: dict[str, asyncio.Task[None]]) -> None:
    if not tasks:
        return
    await asyncio.gather(*tasks.values())


async def _wait_until(
    predicate: Callable[[], bool],
    *,
    timeout_seconds: float = 1.0,
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError("condition was not met before timeout")
        await asyncio.sleep(0.01)
