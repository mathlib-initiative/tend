"""Shared SQLite helpers for tend stores."""

from __future__ import annotations

import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from os import PathLike
from pathlib import Path
from typing import Literal

from tend._common.errors import FrameworkError

DEFAULT_BUSY_TIMEOUT_MS = 5000
_MAX_USER_VERSION = 2_147_483_647

type SQLitePath = str | PathLike[str]
type SQLiteSynchronousMode = Literal["FULL", "NORMAL"]


def connect_read_write(
    path: SQLitePath,
    *,
    busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
    sync_writes: bool | None = None,
) -> sqlite3.Connection:
    """Open a read-write SQLite connection with the project store defaults.

    The connection uses ``sqlite3.Row`` rows, WAL journaling, the requested busy
    timeout, and foreign-key enforcement. ``sync_writes=None`` preserves
    SQLite's current/default ``synchronous`` setting; passing a bool explicitly
    applies the shared durability mapping.
    """

    _validate_busy_timeout_ms(busy_timeout_ms)
    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(Path(path), timeout=busy_timeout_ms / 1000)
        conn.row_factory = sqlite3.Row
        conn.execute(f"PRAGMA busy_timeout = {busy_timeout_ms}")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        if sync_writes is not None:
            conn.execute(f"PRAGMA synchronous = {sqlite_synchronous_mode(sync_writes)}")
    except sqlite3.Error:
        if conn is not None:
            conn.close()
        raise
    return conn


def connect_read_only(
    path: SQLitePath,
    *,
    busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
) -> sqlite3.Connection:
    """Open a read-only SQLite connection using a ``mode=ro`` file URI."""

    _validate_busy_timeout_ms(busy_timeout_ms)
    conn: sqlite3.Connection | None = None
    try:
        uri = f"{Path(path).resolve().as_uri()}?mode=ro"
        conn = sqlite3.connect(uri, timeout=busy_timeout_ms / 1000, uri=True)
        conn.row_factory = sqlite3.Row
        conn.execute(f"PRAGMA busy_timeout = {busy_timeout_ms}")
    except sqlite3.Error:
        if conn is not None:
            conn.close()
        raise
    return conn


def sqlite_synchronous_mode(sync_writes: bool) -> SQLiteSynchronousMode:
    """Map a sync-write policy to a SQLite ``PRAGMA synchronous`` mode."""

    return "FULL" if sync_writes else "NORMAL"


@contextmanager
def begin_immediate(conn: sqlite3.Connection) -> Generator[sqlite3.Connection, None, None]:
    """Run a ``BEGIN IMMEDIATE`` transaction, committing or rolling back.

    The transaction commits when the block exits normally. Any exception from
    the block or from commit rolls the transaction back before being re-raised.
    """

    try:
        conn.execute("BEGIN IMMEDIATE")
        yield conn
        conn.commit()
    except BaseException:
        if conn.in_transaction:
            conn.rollback()
        raise


def read_user_version(conn: sqlite3.Connection) -> int:
    """Return the database ``PRAGMA user_version`` value."""

    row = conn.execute("PRAGMA user_version").fetchone()
    if row is None:
        raise sqlite3.DatabaseError("PRAGMA user_version did not return a row")
    return int(row[0])


def write_user_version(conn: sqlite3.Connection, version: int) -> None:
    """Set the database ``PRAGMA user_version`` value."""

    if version.__class__ is not int:
        raise TypeError("SQLite user_version must be an integer")
    if version < 0 or version > _MAX_USER_VERSION:
        raise ValueError(
            f"SQLite user_version must be between 0 and {_MAX_USER_VERSION}"
        )
    conn.execute(f"PRAGMA user_version = {version}")


@contextmanager
def map_sqlite_errors[FrameworkErrorT: FrameworkError](
    error_type: type[FrameworkErrorT],
    message: str,
) -> Generator[None, None, None]:
    """Map ``sqlite3.Error`` raised in the block to a framework error type."""

    try:
        yield
    except sqlite3.Error as exc:
        raise error_type(f"{message}: {exc}") from exc


def _validate_busy_timeout_ms(value: int) -> None:
    if value < 0:
        raise ValueError("SQLite busy_timeout must be non-negative")


__all__ = (
    "DEFAULT_BUSY_TIMEOUT_MS",
    "SQLiteSynchronousMode",
    "begin_immediate",
    "connect_read_only",
    "connect_read_write",
    "map_sqlite_errors",
    "read_user_version",
    "sqlite_synchronous_mode",
    "write_user_version",
)
