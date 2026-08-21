"""Local exclusive lock for writable session directories."""

from __future__ import annotations

import errno
import fcntl
import os
from pathlib import Path
from types import TracebackType
from typing import Self

from tend._common.errors import PersistenceError
from tend._common.types import utc_timestamp

SESSION_LOCK_FILENAME = "session.lock"


class SessionLockError(PersistenceError):
    """A writable session lock could not be acquired."""


class SessionLock:
    """Exclusive non-blocking advisory lock for one local session directory.

    The lock is intentionally simple: it uses ``flock`` on ``session.lock`` and
    keeps the file descriptor open while the lock is held. Stale-lock recovery is
    deliberately not implemented in v1.
    """

    __slots__ = ("path", "_fd", "_released")

    path: Path
    _fd: int
    _released: bool

    def __init__(self, path: Path, fd: int) -> None:
        self.path = path
        self._fd = fd
        self._released = False

    @classmethod
    def acquire(cls, session_dir: str | Path, *, sync_writes: bool = True) -> Self:
        """Acquire the session's exclusive writable lock or fail clearly."""

        directory = Path(session_dir)
        path = directory / SESSION_LOCK_FILENAME
        fd = -1
        try:
            directory.mkdir(parents=True, exist_ok=True)
            fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise SessionLockError(f"session is already locked: {path}") from exc
            except OSError as exc:
                if exc.errno in (errno.EACCES, errno.EAGAIN):
                    raise SessionLockError(f"session is already locked: {path}") from exc
                raise

            metadata = f"pid={os.getpid()}\nacquired_at={utc_timestamp()}\n"
            os.ftruncate(fd, 0)
            os.write(fd, metadata.encode("utf-8"))
            if sync_writes:
                os.fsync(fd)
            return cls(path, fd)
        except SessionLockError:
            if fd >= 0:
                os.close(fd)
            raise
        except OSError as exc:
            if fd >= 0:
                os.close(fd)
            raise PersistenceError(f"failed to acquire session lock {path}: {exc}") from exc

    @property
    def released(self) -> bool:
        """Whether this handle has already released its lock."""

        return self._released

    def release(self) -> None:
        """Release the lock. Calling more than once is a no-op."""

        if self._released:
            return
        fd = self._fd
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
        except OSError as exc:
            raise PersistenceError(f"failed to release session lock {self.path}: {exc}") from exc
        self._fd = -1
        self._released = True

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:
            pass


__all__ = ("SESSION_LOCK_FILENAME", "SessionLock", "SessionLockError")
