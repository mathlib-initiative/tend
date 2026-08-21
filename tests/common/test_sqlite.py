from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import cast

import pytest

from tend._common.errors import FrameworkError
from tend._common.sqlite import (
    begin_immediate,
    connect_read_only,
    connect_read_write,
    map_sqlite_errors,
    read_user_version,
    sqlite_synchronous_mode,
    write_user_version,
)


class MappedSqliteError(FrameworkError):
    """Test error used to verify SQLite error mapping."""


def test_read_write_connection_applies_default_pragmas(tmp_path: Path) -> None:
    path = tmp_path / "store.sqlite"

    with closing(connect_read_write(path)) as conn:
        journal_mode = str(_pragma_value(conn, "journal_mode")).lower()
        busy_timeout = int(_pragma_value(conn, "busy_timeout"))
        foreign_keys = int(_pragma_value(conn, "foreign_keys"))

    assert journal_mode == "wal"
    assert busy_timeout == 5000
    assert foreign_keys == 1


def test_read_write_connection_allows_configurable_busy_timeout(
    tmp_path: Path,
) -> None:
    path = tmp_path / "store.sqlite"

    with closing(connect_read_write(path, busy_timeout_ms=1234)) as conn:
        busy_timeout = int(_pragma_value(conn, "busy_timeout"))

    assert busy_timeout == 1234


def test_read_only_connection_uses_uri_and_rejects_writes(tmp_path: Path) -> None:
    path = tmp_path / "store.sqlite"
    with closing(connect_read_write(path)) as conn:
        conn.execute("CREATE TABLE items (value TEXT NOT NULL)")
        conn.execute("INSERT INTO items (value) VALUES ('committed')")
        conn.commit()

    with closing(connect_read_only(path, busy_timeout_ms=4321)) as conn:
        assert int(_pragma_value(conn, "busy_timeout")) == 4321
        row = conn.execute("SELECT value FROM items").fetchone()
        assert row is not None
        assert row["value"] == "committed"
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            conn.execute("INSERT INTO items (value) VALUES ('blocked')")


def test_begin_immediate_commits_on_success_and_rolls_back_on_exception(
    tmp_path: Path,
) -> None:
    path = tmp_path / "store.sqlite"
    with closing(connect_read_write(path)) as conn:
        conn.execute("CREATE TABLE items (value TEXT NOT NULL)")
        conn.commit()

        with begin_immediate(conn):
            conn.execute("INSERT INTO items (value) VALUES ('committed')")

        assert _item_values(path) == ["committed"]

        with pytest.raises(RuntimeError, match="boom"):
            with begin_immediate(conn):
                conn.execute("INSERT INTO items (value) VALUES ('rolled back')")
                raise RuntimeError("boom")

        assert conn.in_transaction is False
        assert _item_values(path) == ["committed"]


def test_begin_immediate_rolls_back_open_transaction_when_begin_fails(
    tmp_path: Path,
) -> None:
    path = tmp_path / "store.sqlite"
    with closing(connect_read_write(path)) as conn:
        conn.execute("CREATE TABLE items (value TEXT NOT NULL)")
        conn.execute("INSERT INTO items (value) VALUES ('committed')")
        conn.commit()

        conn.execute("BEGIN")
        conn.execute("INSERT INTO items (value) VALUES ('uncommitted')")
        assert conn.in_transaction is True
        with pytest.raises(sqlite3.OperationalError, match="transaction"):
            with begin_immediate(conn):
                raise AssertionError("unreachable")

        assert conn.in_transaction is False
        assert _item_values(path) == ["committed"]


def test_user_version_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "store.sqlite"

    with closing(connect_read_write(path)) as conn:
        assert read_user_version(conn) == 0
        write_user_version(conn, 42)
        conn.commit()

    with closing(connect_read_only(path)) as conn:
        assert read_user_version(conn) == 42


def test_write_user_version_rejects_invalid_values(tmp_path: Path) -> None:
    path = tmp_path / "store.sqlite"
    invalid_type_values: tuple[object, ...] = (True, 1.2)

    with closing(connect_read_write(path)) as conn:
        for value in invalid_type_values:
            with pytest.raises(TypeError, match="integer"):
                write_user_version(conn, cast(int, value))
        for value in (-1, 2_147_483_648):
            with pytest.raises(ValueError, match="between"):
                write_user_version(conn, value)
        assert read_user_version(conn) == 0


def test_synchronous_mapping_and_connection_parameter(tmp_path: Path) -> None:
    full_path = tmp_path / "full.sqlite"
    normal_path = tmp_path / "normal.sqlite"

    assert sqlite_synchronous_mode(True) == "FULL"
    assert sqlite_synchronous_mode(False) == "NORMAL"
    with closing(connect_read_write(full_path, sync_writes=True)) as conn:
        full = int(_pragma_value(conn, "synchronous"))
    with closing(connect_read_write(normal_path, sync_writes=False)) as conn:
        normal = int(_pragma_value(conn, "synchronous"))

    assert full == 2
    assert normal == 1


def test_map_sqlite_errors_wraps_sqlite_errors() -> None:
    conn = sqlite3.connect(":memory:")
    try:
        with pytest.raises(MappedSqliteError, match="while testing: no such table") as exc_info:
            with map_sqlite_errors(MappedSqliteError, "while testing"):
                conn.execute("SELECT * FROM missing_table").fetchone()
    finally:
        conn.close()

    assert isinstance(exc_info.value.__cause__, sqlite3.Error)


def test_map_sqlite_errors_preserves_non_sqlite_errors() -> None:
    original = RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom") as exc_info:
        with map_sqlite_errors(MappedSqliteError, "while testing"):
            raise original

    assert exc_info.value is original


def _pragma_value(conn: sqlite3.Connection, name: str) -> str | int:
    row = conn.execute(f"PRAGMA {name}").fetchone()
    assert row is not None
    return cast(str | int, row[0])


def _item_values(path: Path) -> list[str]:
    with closing(connect_read_only(path)) as conn:
        rows = conn.execute("SELECT value FROM items ORDER BY rowid ASC").fetchall()
    return [str(row["value"]) for row in rows]
