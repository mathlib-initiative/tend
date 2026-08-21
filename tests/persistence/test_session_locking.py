from __future__ import annotations

from pathlib import Path

import pytest

from tend.agent.persistence.lock import SESSION_LOCK_FILENAME, SessionLock, SessionLockError


def test_session_lock_exclusive_failure(tmp_path: Path) -> None:
    with SessionLock.acquire(tmp_path, sync_writes=False) as first:
        assert first.path == tmp_path / SESSION_LOCK_FILENAME
        assert not first.released
        assert first.path.exists()

        with pytest.raises(SessionLockError, match="already locked"):
            SessionLock.acquire(tmp_path, sync_writes=False)

    assert first.released


def test_session_lock_release_allows_reacquire(tmp_path: Path) -> None:
    lock = SessionLock.acquire(tmp_path, sync_writes=False)
    lock.release()

    with SessionLock.acquire(tmp_path, sync_writes=False) as reacquired:
        assert not reacquired.released


def test_session_lock_release_is_idempotent(tmp_path: Path) -> None:
    lock = SessionLock.acquire(tmp_path, sync_writes=False)

    lock.release()
    lock.release()

    assert lock.released
