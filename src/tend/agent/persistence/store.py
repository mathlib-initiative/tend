"""JSONL event store and atomic state snapshot store."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path

from pydantic import ValidationError

from tend._common.errors import PersistenceError, UnsupportedSchemaVersionError
from tend.agent.persistence.events import SessionEvent, dump_event_json, parse_event_json
from tend.agent.persistence.state import SessionState, dump_state_json, parse_state_json

EVENT_LOG_FILENAME = "events.jsonl"
STATE_SNAPSHOT_FILENAME = "state.json"


class EventStore:
    """Append-only JSONL store for canonical session events."""

    __slots__ = ("directory", "path", "sync_writes")

    directory: Path
    path: Path
    sync_writes: bool

    def __init__(self, session_dir: str | Path, *, sync_writes: bool = True) -> None:
        self.directory = Path(session_dir)
        self.path = self.directory / EVENT_LOG_FILENAME
        self.sync_writes = sync_writes

    def append(self, event: SessionEvent) -> None:
        """Append one compact JSON event followed by a single LF."""

        self.directory.mkdir(parents=True, exist_ok=True)
        line = dump_event_json(event)
        try:
            with self.path.open("a", encoding="utf-8", newline="\n") as file:
                file.write(line)
                file.write("\n")
                if self.sync_writes:
                    file.flush()
                    os.fsync(file.fileno())
            if self.sync_writes:
                _fsync_directory(self.directory)
        except OSError as exc:
            raise PersistenceError(f"failed to append event log {self.path}: {exc}") from exc

    def append_many(self, events: Iterable[SessionEvent]) -> None:
        """Append multiple events in order as one local file operation."""

        event_batch = tuple(events)
        if not event_batch:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        try:
            with self.path.open("a", encoding="utf-8", newline="\n") as file:
                for event in event_batch:
                    file.write(dump_event_json(event))
                    file.write("\n")
                if self.sync_writes:
                    file.flush()
                    os.fsync(file.fileno())
            if self.sync_writes:
                _fsync_directory(self.directory)
        except OSError as exc:
            raise PersistenceError(f"failed to append event log {self.path}: {exc}") from exc

    def read_all(self) -> list[SessionEvent]:
        """Read and validate every event from ``events.jsonl`` in file order."""

        if not self.path.exists():
            return []

        events: list[SessionEvent] = []
        try:
            with self.path.open("r", encoding="utf-8") as file:
                for line_number, line in enumerate(file, start=1):
                    text = line.rstrip("\n")
                    if not text:
                        raise PersistenceError(
                            f"invalid event log line {line_number} in {self.path}: empty line"
                        )
                    try:
                        events.append(parse_event_json(text))
                    except json.JSONDecodeError as exc:
                        raise PersistenceError(
                            f"invalid event log line {line_number} in {self.path}: invalid JSON"
                        ) from exc
                    except UnsupportedSchemaVersionError as exc:
                        raise UnsupportedSchemaVersionError(
                            f"unsupported event schema version at line {line_number} "
                            f"in {self.path}: {exc}"
                        ) from exc
                    except ValidationError as exc:
                        raise PersistenceError(
                            f"invalid event log line {line_number} in {self.path}: {exc}"
                        ) from exc
        except UnicodeDecodeError as exc:
            raise PersistenceError(f"event log {self.path} is not valid UTF-8") from exc
        except OSError as exc:
            raise PersistenceError(f"failed to read event log {self.path}: {exc}") from exc
        return events


class SnapshotStore:
    """Atomic ``state.json`` snapshot/cache store."""

    __slots__ = ("directory", "path", "sync_writes")

    directory: Path
    path: Path
    sync_writes: bool

    def __init__(self, session_dir: str | Path, *, sync_writes: bool = True) -> None:
        self.directory = Path(session_dir)
        self.path = self.directory / STATE_SNAPSHOT_FILENAME
        self.sync_writes = sync_writes

    def read(self) -> SessionState | None:
        """Read and validate ``state.json``, returning ``None`` when absent."""

        if not self.path.exists():
            return None
        try:
            return parse_state_json(self.path.read_text(encoding="utf-8"))
        except UnsupportedSchemaVersionError:
            raise
        except json.JSONDecodeError as exc:
            raise PersistenceError(f"state snapshot {self.path} is invalid JSON") from exc
        except ValidationError as exc:
            raise PersistenceError(f"state snapshot {self.path} is invalid: {exc}") from exc
        except UnicodeDecodeError as exc:
            raise PersistenceError(f"state snapshot {self.path} is not valid UTF-8") from exc
        except OSError as exc:
            raise PersistenceError(f"failed to read state snapshot {self.path}: {exc}") from exc

    def write(self, state: SessionState) -> None:
        """Atomically write ``state.json`` via a same-directory temp file."""

        self.directory.mkdir(parents=True, exist_ok=True)
        temp_fd = -1
        temp_path: Path | None = None
        try:
            temp_fd, temp_name = tempfile.mkstemp(
                prefix=f".{STATE_SNAPSHOT_FILENAME}.",
                suffix=".tmp",
                dir=self.directory,
            )
            temp_path = Path(temp_name)
            with os.fdopen(temp_fd, "w", encoding="utf-8", newline="\n") as file:
                temp_fd = -1
                file.write(dump_state_json(state))
                file.write("\n")
                if self.sync_writes:
                    file.flush()
                    os.fsync(file.fileno())
            os.replace(temp_path, self.path)
            temp_path = None
            if self.sync_writes:
                _fsync_directory(self.directory)
        except OSError as exc:
            raise PersistenceError(f"failed to write state snapshot {self.path}: {exc}") from exc
        finally:
            if temp_fd >= 0:
                os.close(temp_fd)
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    pass


def _fsync_directory(directory: Path) -> None:
    """Best-effort directory fsync for local crash-recovery durability."""

    if os.name == "nt":
        return
    try:
        directory_fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


__all__ = (
    "EVENT_LOG_FILENAME",
    "STATE_SNAPSHOT_FILENAME",
    "EventStore",
    "SnapshotStore",
)
