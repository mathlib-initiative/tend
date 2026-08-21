"""Runtime context passed to tool handlers."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path

from tend._common.types import JsonObject
from tend.agent.cancellation import CancellationState
from tend.agent.config import RuntimeConfig
from tend.agent.tools.backends import FilesystemBackend, ProcessBackend

type ToolEventCallback = Callable[[str, JsonObject], Awaitable[None] | None]

ToolCancellationState = CancellationState


class ToolContext:
    """Runtime context passed to async tool handlers.

    The context carries execution metadata and callbacks only. It deliberately
    does not implement filesystem, command, network, or path security policy;
    the process/orchestration sandbox boundary owns sandbox policy.
    """

    __slots__ = (
        "_cancellation",
        "_event_callback",
        "cwd",
        "filesystem_backend",
        "process_backend",
        "runtime_config",
        "session_id",
        "turn_id",
    )

    cwd: Path
    filesystem_backend: FilesystemBackend | None
    process_backend: ProcessBackend | None
    runtime_config: RuntimeConfig
    session_id: str | None
    turn_id: str | None
    _event_callback: ToolEventCallback | None
    _cancellation: CancellationState | None

    def __init__(
        self,
        *,
        cwd: str | Path = ".",
        session_id: str | None = None,
        turn_id: str | None = None,
        runtime_config: RuntimeConfig | None = None,
        event_callback: ToolEventCallback | None = None,
        cancellation: CancellationState | None = None,
        filesystem_backend: FilesystemBackend | None = None,
        process_backend: ProcessBackend | None = None,
    ) -> None:
        self.cwd = Path(cwd)
        self.session_id = session_id
        self.turn_id = turn_id
        self.filesystem_backend = filesystem_backend
        self.process_backend = process_backend
        if runtime_config is None:
            self.runtime_config = RuntimeConfig(cwd=str(self.cwd))
        else:
            self.runtime_config = runtime_config
        self._event_callback = event_callback
        self._cancellation = cancellation

    @property
    def is_cancelled(self) -> bool:
        """Return whether cooperative cancellation has been requested."""

        if self._cancellation is None:
            return False
        return self._cancellation.is_cancelled

    @property
    def cancellation(self) -> CancellationState | None:
        """Return the cancellation state object, if one was supplied."""

        return self._cancellation

    async def emit_event(self, event_type: str, payload: JsonObject | None = None) -> None:
        """Emit a small tool event/log payload through the optional callback."""

        if self._event_callback is None:
            return
        result = self._event_callback(event_type, payload or {})
        if isinstance(result, Awaitable):
            await result


__all__ = ("ToolCancellationState", "ToolContext", "ToolEventCallback")
