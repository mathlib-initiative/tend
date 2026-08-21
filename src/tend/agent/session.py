"""Public session runtime boundary."""

from __future__ import annotations

from pathlib import Path
from types import TracebackType
from typing import Self

from tend._common.errors import PersistenceError
from tend._common.types import advance_id_counter, new_id
from tend.agent.persistence.artifacts import ArtifactStore
from tend.agent.persistence.events import (
    SessionEvent,
    SessionResumedEvent,
    SessionResumedPayload,
    SessionStartedEvent,
    SessionStartedPayload,
    next_event_sequence,
)
from tend.agent.persistence.lock import SessionLock
from tend.agent.persistence.state import SessionState, session_state_from_events
from tend.agent.persistence.store import EventStore, SnapshotStore


class Session:
    """Writable or read-only handle for a persisted session directory.

    Creating or opening a writable session records the corresponding lifecycle
    event immediately. The event log remains the canonical source of truth;
    ``state.json`` is maintained as a small cursor snapshot/cache.
    """

    __slots__ = (
        "session_id",
        "directory",
        "event_store",
        "snapshot_store",
        "_lock_handle",
        "_closed",
        "_state",
        "_event_count",
        "_last_event_id",
        "_last_sequence",
        "_next_sequence",
    )

    session_id: str
    directory: Path
    event_store: EventStore
    snapshot_store: SnapshotStore
    _lock_handle: SessionLock | None
    _closed: bool
    _state: SessionState
    _event_count: int
    _last_event_id: str | None
    _last_sequence: int | None
    _next_sequence: int

    def __init__(
        self,
        *,
        session_id: str,
        directory: str | Path,
        event_store: EventStore,
        snapshot_store: SnapshotStore,
        lock_handle: SessionLock | None,
        state: SessionState,
        next_sequence: int,
    ) -> None:
        self.session_id = session_id
        self.directory = Path(directory)
        self.event_store = event_store
        self.snapshot_store = snapshot_store
        self._lock_handle = lock_handle
        self._closed = False
        self._state = state
        self._event_count = state.event_count
        self._last_event_id = state.last_event_id
        self._last_sequence = state.last_sequence
        self._next_sequence = next_sequence

    @classmethod
    def create(
        cls,
        directory: str | Path,
        *,
        session_id: str | None = None,
        cwd: str | Path | None = None,
        sync_writes: bool = True,
    ) -> Self:
        """Create a new writable session and append ``SessionStarted``.

        Existing non-empty event logs are not overwritten; use :meth:`open` to
        resume them.
        """

        session_dir = Path(directory)
        lock_handle = SessionLock.acquire(session_dir, sync_writes=sync_writes)
        try:
            event_store = EventStore(session_dir, sync_writes=sync_writes)
            snapshot_store = SnapshotStore(session_dir, sync_writes=sync_writes)
            existing_events = event_store.read_all()
            if existing_events:
                raise PersistenceError(
                    f"session already exists at {session_dir}; use Session.open to resume it"
                )
            ArtifactStore(session_dir, sync_writes=sync_writes).ensure_layout()

            resolved_session_id = session_id or new_id("sess")
            started = SessionStartedEvent(
                session_id=resolved_session_id,
                sequence=0,
                payload=SessionStartedPayload(cwd=str(cwd) if cwd is not None else None),
            )
            event_store.append(started)
            state = session_state_from_events([started], session_id=resolved_session_id)
            snapshot_store.write(state)
            return cls(
                session_id=resolved_session_id,
                directory=session_dir,
                event_store=event_store,
                snapshot_store=snapshot_store,
                lock_handle=lock_handle,
                state=state,
                next_sequence=1,
            )
        except Exception:
            lock_handle.release()
            raise

    @classmethod
    def open(
        cls,
        directory: str | Path,
        *,
        writable: bool = True,
        sync_writes: bool = True,
    ) -> Self:
        """Open an existing session, appending ``SessionResumed`` when writable."""

        session_dir = Path(directory)
        if not session_dir.exists():
            raise PersistenceError(f"session directory does not exist: {session_dir}")

        lock_handle = (
            SessionLock.acquire(session_dir, sync_writes=sync_writes) if writable else None
        )
        try:
            event_store = EventStore(session_dir, sync_writes=sync_writes)
            snapshot_store = SnapshotStore(session_dir, sync_writes=sync_writes)
            events = event_store.read_all()
            if not events:
                raise PersistenceError(
                    f"session event log is empty at {event_store.path}; use Session.create first"
                )

            state = session_state_from_events(events)
            snapshot = snapshot_store.read()
            if snapshot is not None and snapshot.session_id != state.session_id:
                raise PersistenceError(
                    "state snapshot session_id does not match canonical event log "
                    f"for {session_dir}"
                )

            # Advance the global ID counter past the highest event ID seen so far.
            # Without this, a resumed process would start allocating IDs from 1,
            # colliding with IDs already in the event log.
            _advance_counter_past_events(events)

            if writable:
                resumed = SessionResumedEvent(
                    parent_event_id=state.last_event_id,
                    sequence=next_event_sequence(events),
                    session_id=state.session_id,
                    payload=SessionResumedPayload(
                        resumed_from_event_id=state.last_event_id,
                        state_event_count=state.event_count,
                    ),
                )
                event_store.append(resumed)
                events.append(resumed)
                state = session_state_from_events(events, session_id=state.session_id)
                snapshot_store.write(state)

            return cls(
                session_id=state.session_id,
                directory=session_dir,
                event_store=event_store,
                snapshot_store=snapshot_store,
                lock_handle=lock_handle,
                state=state,
                next_sequence=next_event_sequence(events),
            )
        except Exception:
            if lock_handle is not None:
                lock_handle.release()
            raise

    @classmethod
    def resume(
        cls,
        directory: str | Path,
        *,
        sync_writes: bool = True,
    ) -> Self:
        """Alias for opening an existing session for writable resume."""

        return cls.open(directory, writable=True, sync_writes=sync_writes)

    @property
    def lock_handle(self) -> SessionLock | None:
        """The held writable lock, or ``None`` for read-only sessions."""

        return self._lock_handle

    @property
    def writable(self) -> bool:
        """Whether this session handle can append events and write snapshots."""

        return self._lock_handle is not None

    @property
    def closed(self) -> bool:
        """Whether this handle has been closed."""

        return self._closed

    @property
    def event_count(self) -> int:
        """Number of canonical events reflected by this handle's state cursor."""

        return self._event_count

    @property
    def last_event_id(self) -> str | None:
        """Last persisted event ID reflected by this handle's state cursor."""

        return self._last_event_id

    @property
    def next_sequence(self) -> int:
        """Suggested next linear event sequence number."""

        return self._next_sequence

    @property
    def state(self) -> SessionState:
        """Return the current replayed state snapshot cache."""

        return self._state.model_copy(deep=True)

    def append_event(self, event: SessionEvent) -> None:
        """Append one event and atomically refresh ``state.json``.

        The caller supplies the event envelope explicitly. This method only
        verifies that it belongs to the session and persists the visible side
        effect; it does not replay completed operations or make resume decisions.
        """

        self._ensure_open()
        if self._lock_handle is None:
            raise PersistenceError("cannot append events through a read-only session")
        if event.session_id != self.session_id:
            raise ValueError("event session_id must match the session handle")

        self.event_store.append(event)
        self._refresh_state_from_events()
        self.snapshot_store.write(self._state)

    def close(self) -> None:
        """Release any held writable lock. Calling more than once is a no-op."""

        if self._closed:
            return
        self._closed = True
        if self._lock_handle is not None:
            self._lock_handle.release()
            self._lock_handle = None

    def __enter__(self) -> Self:
        self._ensure_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    async def __aenter__(self) -> Self:
        self._ensure_open()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _refresh_state_from_events(self) -> None:
        events = self.event_store.read_all()
        self._state = session_state_from_events(events, session_id=self.session_id)
        self._event_count = self._state.event_count
        self._last_event_id = self._state.last_event_id
        self._last_sequence = self._state.last_sequence
        self._next_sequence = next_event_sequence(events)

    def _ensure_open(self) -> None:
        if self._closed:
            raise PersistenceError("session handle is closed")


def create_session(
    directory: str | Path,
    *,
    session_id: str | None = None,
    cwd: str | Path | None = None,
    sync_writes: bool = True,
) -> Session:
    """Create a writable session via :meth:`Session.create`."""

    return Session.create(directory, session_id=session_id, cwd=cwd, sync_writes=sync_writes)


def open_session(
    directory: str | Path,
    *,
    writable: bool = True,
    sync_writes: bool = True,
) -> Session:
    """Open a session via :meth:`Session.open`."""

    return Session.open(directory, writable=writable, sync_writes=sync_writes)


def _advance_counter_past_events(events: list[SessionEvent]) -> None:
    """Advance the global ID counter past the highest event_id in the log.

    All IDs (event, model_req, turn, etc.) share one counter. Event IDs are
    always allocated after the IDs they reference in their payloads, so the
    maximum event_id is the global high-water mark.
    """
    max_seq = 0
    for event in events:
        try:
            seq = int(event.event_id.rsplit("_", 1)[1])
            if seq > max_seq:
                max_seq = seq
        except (ValueError, IndexError):
            pass
    if max_seq > 0:
        advance_id_counter(max_seq + 1)


__all__ = ("Session", "create_session", "open_session")
