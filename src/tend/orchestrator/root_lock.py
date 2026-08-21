"""Local exclusive lock for async orchestration roots."""

from __future__ import annotations

import errno
import fcntl
import os
from pathlib import Path
from types import TracebackType
from typing import Self

from tend._common.errors import FrameworkError
from tend._common.types import utc_timestamp

ASYNC_ORCHESTRATOR_ROOT_LOCK_FILENAME = ".tend.lock"


class AsyncOrchestratorRootLockError(FrameworkError):
    """An async orchestration root lock could not be acquired or released."""


class AsyncOrchestratorRootLock:
    """Exclusive non-blocking advisory lock for one async orchestration root.

    The lock uses ``flock`` on ``.tend.lock`` and keeps the file descriptor
    open while the lock is held. Stale-lock recovery is intentionally omitted:
    the operating system releases the advisory lock when the owning process
    exits.
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
    def acquire(
        cls,
        root: str | os.PathLike[str],
        *,
        owner: str,
        sync_writes: bool = True,
    ) -> Self:
        """Acquire the root's exclusive advisory lock or fail clearly."""

        directory = Path(root)
        path = directory / ASYNC_ORCHESTRATOR_ROOT_LOCK_FILENAME
        fd = -1
        try:
            fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise AsyncOrchestratorRootLockError(
                    f"async orchestration root is already locked: {path}"
                ) from exc
            except OSError as exc:
                if exc.errno in (errno.EACCES, errno.EAGAIN):
                    raise AsyncOrchestratorRootLockError(
                        f"async orchestration root is already locked: {path}"
                    ) from exc
                raise

            metadata = (
                f"pid={os.getpid()}\n"
                f"owner={owner}\n"
                f"acquired_at={utc_timestamp()}\n"
            )
            os.ftruncate(fd, 0)
            os.write(fd, metadata.encode("utf-8"))
            if sync_writes:
                os.fsync(fd)
            return cls(path, fd)
        except AsyncOrchestratorRootLockError:
            if fd >= 0:
                os.close(fd)
            raise
        except OSError as exc:
            if fd >= 0:
                os.close(fd)
            raise AsyncOrchestratorRootLockError(
                f"failed to acquire async orchestration root lock {path}: {exc}"
            ) from exc

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
            raise AsyncOrchestratorRootLockError(
                f"failed to release async orchestration root lock {self.path}: {exc}"
            ) from exc
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


__all__ = (
    "ASYNC_ORCHESTRATOR_ROOT_LOCK_FILENAME",
    "AsyncOrchestratorRootLock",
    "AsyncOrchestratorRootLockError",
)
