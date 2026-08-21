"""Runtime queue helpers for the async orchestrator."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable


class RuntimeQueue:
    """A de-duplicating, inspectable priority/FIFO queue for one runtime consumer."""

    __slots__ = ("_change_event", "_change_version", "_items", "_priorities", "_queued")

    _change_event: asyncio.Event
    _change_version: int
    _items: list[str]
    _priorities: dict[str, int]
    _queued: set[str]

    def __init__(self) -> None:
        self._queued = set()
        self._items = []
        self._priorities = {}
        self._change_event = asyncio.Event()
        self._change_version = 0

    def __contains__(self, item: str) -> bool:
        return item in self._queued

    def __bool__(self) -> bool:
        return bool(self._queued)

    @property
    def change_version(self) -> int:
        """Return a monotonic version that changes whenever queue waiters should wake."""

        return self._change_version

    @property
    def has_claimed_items(self) -> bool:
        """Return whether any item is queued or reserved/in-progress.

        Unlike ``items`` (visible FIFO only) this also counts items that have been
        reserved/hidden while being processed, so it reflects whether the queue is
        fully idle.
        """

        return bool(self._queued)

    @property
    def has_reserved_items(self) -> bool:
        """Return whether any item is reserved/hidden but not visible in FIFO order."""

        return any(item not in self._items for item in self._queued)

    @property
    def items(self) -> tuple[str, ...]:
        """Return queued items in visible FIFO order."""

        return tuple(self._items)

    def notify_changed(self) -> None:
        """Wake waiters that re-check queue, limit, or active-task state."""

        self._change_version += 1
        self._change_event.set()

    async def wait_for_change_since(self, version: int) -> None:
        """Block until ``change_version`` differs from ``version``."""

        while self._change_version == version:
            self._change_event.clear()
            if self._change_version != version:
                return
            await self._change_event.wait()

    def enqueue(self, item: str, *, priority: int = 0) -> None:
        """Enqueue ``item`` unless already queued, ordering by priority.

        Lower integer priority values are picked first. Items with the same
        priority keep FIFO order, preserving the historical queue behavior when
        all callers use the default priority.
        """

        if item in self._queued:
            if self._priorities.get(item, 0) == priority:
                return
            self._priorities[item] = priority
            self._sort_items_by_priority()
            self.notify_changed()
            return
        self._queued.add(item)
        self._priorities[item] = priority
        self._items.append(item)
        self._sort_items_by_priority()
        self.notify_changed()

    def hide(self, item: str) -> None:
        """Remove ``item`` from visible items while keeping it claimed."""

        if _remove_item(self._items, item):
            self.notify_changed()

    def reserve(self, item: str) -> bool:
        """Hide and return whether ``item`` is queued, keeping it de-duplicated."""

        if item not in self._queued:
            return False
        self.hide(item)
        return True

    def release(self, item: str) -> bool:
        """Make a reserved item visible again, ahead of same-priority FIFO peers."""

        if item not in self._queued or item in self._items:
            return False
        self._items.insert(0, item)
        self._sort_items_by_priority()
        self.notify_changed()
        return True

    def claim(self, item: str) -> bool:
        """Discard and return whether ``item`` was queued when consumed."""

        if item not in self._queued:
            return False
        self.discard(item)
        return True

    def claim_nowait(self) -> str:
        """Claim and return the next visible item without waiting."""

        item = self.get_nowait()
        self.claim(item)
        return item

    def discard(self, item: str) -> None:
        """Remove ``item`` from both de-duplication and visible FIFO state."""

        was_queued = item in self._queued
        self._queued.discard(item)
        self._priorities.pop(item, None)
        removed = _remove_item(self._items, item)
        if was_queued or removed:
            self.notify_changed()

    def reorder(self, ordered_items: Iterable[str]) -> None:
        """Reorder visible queued items according to ``ordered_items``.

        Items absent from ``ordered_items`` stay visible after ordered items and
        retain their relative order. Reserved/hidden items remain claimed but are
        not made visible.
        """

        order = {item: index for index, item in enumerate(ordered_items)}
        if not order or len(self._items) < 2:
            return
        before = tuple(self._items)
        fallback_order = len(order)
        self._items.sort(key=lambda item: order.get(item, fallback_order))
        if tuple(self._items) != before:
            self.notify_changed()

    async def get(self) -> str:
        """Wait for the next visible queue item without claiming it."""

        while not self._items:
            await self.wait_for_change_since(self._change_version)
        return self._items[0]

    def get_nowait(self) -> str:
        """Return the next visible queue item without waiting or claiming it."""

        if not self._items:
            raise asyncio.QueueEmpty
        return self._items[0]

    def task_done(self) -> None:
        """Mark one queue item as handled.

        RuntimeQueue does not expose ``join()``; completion is tracked by
        visible/reserved membership instead. This no-op preserves the existing
        call sites that balance queue handling after an item is claimed.
        """

        pass

    def _sort_items_by_priority(self) -> None:
        self._items.sort(key=lambda item: self._priorities.get(item, 0))


def _remove_item(items: list[str], item: str) -> bool:
    try:
        items.remove(item)
    except ValueError:
        return False
    return True
