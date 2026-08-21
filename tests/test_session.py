from __future__ import annotations

from pathlib import Path

import pytest

from tend._common.errors import PersistenceError, UnsupportedSchemaVersionError
from tend._common.types import StopReason
from tend.agent.persistence.events import (
    EventType,
    SessionResumedEvent,
    SessionStartedEvent,
    TurnCompletedEvent,
    TurnCompletedPayload,
)
from tend.agent.persistence.lock import SessionLockError
from tend.agent.persistence.state import SessionState
from tend.agent.persistence.store import EVENT_LOG_FILENAME, STATE_SNAPSHOT_FILENAME
from tend.agent.session import Session, create_session, open_session


def test_create_session_directory_layout_and_start_event(tmp_path: Path) -> None:
    with Session.create(tmp_path, session_id="sess_1", cwd="/work", sync_writes=False) as session:
        assert session.session_id == "sess_1"
        assert session.directory == tmp_path
        assert session.writable is True
        assert session.event_store.path == tmp_path / EVENT_LOG_FILENAME
        assert session.snapshot_store.path == tmp_path / STATE_SNAPSHOT_FILENAME
        assert session.lock_handle is not None
        assert (tmp_path / "session.lock").exists()

        events = session.event_store.read_all()
        assert len(events) == 1
        assert isinstance(events[0], SessionStartedEvent)
        assert events[0].event_type is EventType.SESSION_STARTED
        assert events[0].session_id == "sess_1"
        assert events[0].sequence == 0
        assert events[0].payload.cwd == "/work"
        assert session.snapshot_store.read() == SessionState(
            session_id="sess_1",
            event_count=1,
            last_event_id=events[0].event_id,
            last_sequence=0,
        )

    assert session.closed is True


def test_create_session_helper_wraps_classmethod(tmp_path: Path) -> None:
    session = create_session(tmp_path, session_id="sess_helper", sync_writes=False)
    try:
        assert isinstance(session, Session)
        assert session.session_id == "sess_helper"
    finally:
        session.close()


def test_open_session_appends_resume_event_and_updates_state(tmp_path: Path) -> None:
    created = Session.create(tmp_path, session_id="sess_1", sync_writes=False)
    first_event_id = created.last_event_id
    created.close()

    with Session.open(tmp_path, sync_writes=False) as resumed:
        assert resumed.session_id == "sess_1"
        assert resumed.event_count == 2
        assert resumed.next_sequence == 2

        events = resumed.event_store.read_all()
        assert len(events) == 2
        assert isinstance(events[1], SessionResumedEvent)
        assert events[1].parent_event_id == first_event_id
        assert events[1].sequence == 1
        assert events[1].payload.resumed_from_event_id == first_event_id
        assert events[1].payload.state_event_count == 1
        assert resumed.snapshot_store.read() == SessionState(
            session_id="sess_1",
            event_count=2,
            last_event_id=events[1].event_id,
            last_sequence=1,
        )


def test_open_session_helper_can_open_read_only_without_resume_event(tmp_path: Path) -> None:
    created = Session.create(tmp_path, session_id="sess_1", sync_writes=False)
    created.close()

    read_only = open_session(tmp_path, writable=False, sync_writes=False)
    try:
        assert read_only.writable is False
        assert read_only.lock_handle is None
        assert read_only.event_count == 1
        assert len(read_only.event_store.read_all()) == 1
    finally:
        read_only.close()


def test_writable_session_cannot_be_opened_concurrently(tmp_path: Path) -> None:
    session = Session.create(tmp_path, session_id="sess_1", sync_writes=False)
    try:
        with pytest.raises(SessionLockError, match="already locked"):
            Session.open(tmp_path, sync_writes=False)
    finally:
        session.close()

    resumed = Session.open(tmp_path, sync_writes=False)
    resumed.close()


def test_create_existing_session_fails_without_overwriting(tmp_path: Path) -> None:
    session = Session.create(tmp_path, session_id="sess_1", sync_writes=False)
    session.close()

    with pytest.raises(PersistenceError, match="use Session.open"):
        Session.create(tmp_path, session_id="sess_2", sync_writes=False)

    read_only = Session.open(tmp_path, writable=False, sync_writes=False)
    try:
        events = read_only.event_store.read_all()
        assert len(events) == 1
        assert events[0].session_id == "sess_1"
    finally:
        read_only.close()


def test_append_event_refreshes_state_snapshot(tmp_path: Path) -> None:
    with Session.create(tmp_path, session_id="sess_1", sync_writes=False) as session:
        event = TurnCompletedEvent(
            session_id="sess_1",
            turn_id="turn_1",
            parent_event_id=session.last_event_id,
            sequence=session.next_sequence,
            payload=TurnCompletedPayload(
                stop_reason=StopReason.FINAL_RESPONSE,
                final_response="done",
            ),
        )

        session.append_event(event)

        assert session.event_count == 2
        assert session.last_event_id == event.event_id
        assert session.next_sequence == 2
        assert session.snapshot_store.read() == SessionState(
            session_id="sess_1",
            event_count=2,
            last_event_id=event.event_id,
            last_sequence=1,
        )


def test_append_event_rejects_read_only_or_wrong_session(tmp_path: Path) -> None:
    created = Session.create(tmp_path, session_id="sess_1", sync_writes=False)
    created.close()
    read_only = Session.open(tmp_path, writable=False, sync_writes=False)
    try:
        event = TurnCompletedEvent(
            session_id="sess_1",
            payload=TurnCompletedPayload(stop_reason=StopReason.FINAL_RESPONSE),
        )
        with pytest.raises(PersistenceError, match="read-only"):
            read_only.append_event(event)
    finally:
        read_only.close()

    writable = Session.open(tmp_path, sync_writes=False)
    try:
        wrong = TurnCompletedEvent(
            session_id="sess_other",
            payload=TurnCompletedPayload(stop_reason=StopReason.FINAL_RESPONSE),
        )
        with pytest.raises(ValueError, match="session_id"):
            writable.append_event(wrong)
    finally:
        writable.close()


def test_closed_session_rejects_append(tmp_path: Path) -> None:
    session = Session.create(tmp_path, session_id="sess_1", sync_writes=False)
    session.close()

    event = TurnCompletedEvent(
        session_id="sess_1",
        payload=TurnCompletedPayload(stop_reason=StopReason.FINAL_RESPONSE),
    )
    with pytest.raises(PersistenceError, match="closed"):
        session.append_event(event)


def test_open_session_fails_on_unsupported_state_schema_version(tmp_path: Path) -> None:
    session = Session.create(tmp_path, session_id="sess_1", sync_writes=False)
    session.close()
    (tmp_path / STATE_SNAPSHOT_FILENAME).write_text(
        '{"schema_version":999,"session_id":"sess_1"}\n',
        encoding="utf-8",
    )

    with pytest.raises(UnsupportedSchemaVersionError, match="unsupported state schema version"):
        Session.open(tmp_path, sync_writes=False)

    (tmp_path / STATE_SNAPSHOT_FILENAME).write_text(
        '{"schema_version":1,"session_id":"sess_1","event_count":1}\n',
        encoding="utf-8",
    )
    resumed = Session.open(tmp_path, sync_writes=False)
    resumed.close()


def test_open_missing_or_empty_session_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(PersistenceError, match="does not exist"):
        Session.open(tmp_path / "missing", sync_writes=False)

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(PersistenceError, match="event log is empty"):
        Session.open(empty, sync_writes=False)
