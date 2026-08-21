"""Cooperative cancellation state for running turns and tools."""

from __future__ import annotations


class CancellationState:
    """Small cooperative cancellation flag shared by a turn and its tools.

    This object does not cancel asyncio tasks by itself. Callers may request
    cancellation through :meth:`cancel`; the turn loop checks the flag at safe
    boundaries and tool handlers can inspect it through ``ToolContext``.
    """

    __slots__ = ("_is_cancelled", "_reason")

    _is_cancelled: bool
    _reason: str | None

    def __init__(self, *, is_cancelled: bool = False, reason: str | None = None) -> None:
        if reason is not None and not reason:
            raise ValueError("cancellation reason must be non-empty when provided")
        self._is_cancelled = is_cancelled
        self._reason = reason

    @property
    def is_cancelled(self) -> bool:
        """Return whether cooperative cancellation has been requested."""

        return self._is_cancelled

    @property
    def reason(self) -> str | None:
        """Return the optional cancellation reason."""

        return self._reason

    def cancel(self, reason: str | None = None) -> None:
        """Request cooperative cancellation.

        A later non-empty reason replaces an absent reason, while repeated calls
        remain idempotent for callers that only need a flag.
        """

        if reason is not None and not reason:
            raise ValueError("cancellation reason must be non-empty when provided")
        self._is_cancelled = True
        if reason is not None or self._reason is None:
            self._reason = reason


CancellationToken = CancellationState


__all__ = ("CancellationState", "CancellationToken")
