from __future__ import annotations

from pathlib import Path

import pytest

from tend._common.errors import PersistenceError, UnsupportedSchemaVersionError
from tend._common.types import StopReason
from tend.agent.persistence.events import (
    SessionStartedEvent,
    SessionStartedPayload,
    TurnCompletedEvent,
    TurnCompletedPayload,
    dump_event_json,
)
from tend.agent.persistence.state import (
    SessionState,
    dump_state_json,
    parse_state,
    parse_state_json,
    session_state_from_events,
)
from tend.agent.persistence.store import EventStore, SnapshotStore

_TIMESTAMP = "2026-05-04T12:00:00Z"


def _started_event(sequence: int = 0) -> SessionStartedEvent:
    return SessionStartedEvent(
        event_id="evt_0000000000000001",
        session_id="sess_1",
        sequence=sequence,
        timestamp=_TIMESTAMP,
        payload=SessionStartedPayload(cwd="/work"),
    )


def _completed_event(sequence: int = 1) -> TurnCompletedEvent:
    return TurnCompletedEvent(
        event_id="evt_0000000000000002",
        parent_event_id="evt_0000000000000001",
        session_id="sess_1",
        turn_id="turn_1",
        sequence=sequence,
        timestamp=_TIMESTAMP,
        payload=TurnCompletedPayload(
            stop_reason=StopReason.FINAL_RESPONSE,
            final_response="done",
            model_request_count=1,
        ),
    )


def test_event_store_appends_compact_jsonl_and_reads_round_trip(tmp_path: Path) -> None:
    store = EventStore(tmp_path, sync_writes=False)
    first = _started_event()
    second = _completed_event()

    store.append(first)
    store.append(second)

    assert store.path.read_text(encoding="utf-8") == (
        f"{dump_event_json(first)}\n{dump_event_json(second)}\n"
    )
    assert store.read_all() == [first, second]


def test_event_store_append_many_preserves_order(tmp_path: Path) -> None:
    store = EventStore(tmp_path, sync_writes=False)
    events = [_started_event(), _completed_event()]

    store.append_many(events)

    assert store.read_all() == events


def test_event_store_invalid_jsonl_fails_with_line_number(tmp_path: Path) -> None:
    store = EventStore(tmp_path, sync_writes=False)
    store.path.write_text(f"{dump_event_json(_started_event())}\nnot-json\n", encoding="utf-8")

    with pytest.raises(PersistenceError, match="invalid event log line 2"):
        store.read_all()


def test_event_store_invalid_event_fails_with_line_number(tmp_path: Path) -> None:
    store = EventStore(tmp_path, sync_writes=False)
    store.path.write_text('{"schema_version":1,"event_type":"TurnCompleted"}\n', encoding="utf-8")

    with pytest.raises(PersistenceError, match="invalid event log line 1"):
        store.read_all()


def test_event_store_unsupported_schema_version_fails_clearly(tmp_path: Path) -> None:
    store = EventStore(tmp_path, sync_writes=False)
    store.path.write_text(
        '{"schema_version":999,"event_type":"SessionStarted","session_id":"sess_1"}\n',
        encoding="utf-8",
    )

    with pytest.raises(UnsupportedSchemaVersionError, match="line 1"):
        store.read_all()


def test_snapshot_store_atomic_write_read_and_replace(tmp_path: Path) -> None:
    snapshot_store = SnapshotStore(tmp_path, sync_writes=False)
    initial = SessionState(session_id="sess_1", event_count=1, last_event_id="evt_1")
    replacement = SessionState(
        session_id="sess_1",
        event_count=2,
        last_event_id="evt_2",
        last_sequence=1,
    )

    snapshot_store.write(initial)
    snapshot_store.write(replacement)

    assert snapshot_store.read() == replacement
    assert snapshot_store.path.read_text(encoding="utf-8") == f"{dump_state_json(replacement)}\n"
    assert not list(tmp_path.glob(".state.json.*.tmp"))


def test_state_json_can_be_absent_while_events_still_load(tmp_path: Path) -> None:
    event_store = EventStore(tmp_path, sync_writes=False)
    snapshot_store = SnapshotStore(tmp_path, sync_writes=False)
    event = _started_event()

    event_store.append(event)

    assert snapshot_store.read() is None
    assert event_store.read_all() == [event]


def test_session_state_cursor_can_be_built_from_events() -> None:
    state = session_state_from_events([_started_event(), _completed_event()])

    assert state == SessionState(
        session_id="sess_1",
        event_count=2,
        last_event_id="evt_0000000000000002",
        last_sequence=1,
    )


def test_session_state_from_empty_events_requires_session_id() -> None:
    with pytest.raises(ValueError, match="session_id is required"):
        session_state_from_events([])

    assert session_state_from_events([], session_id="sess_empty") == SessionState(
        session_id="sess_empty"
    )


def test_session_state_from_events_rejects_mixed_session_ids() -> None:
    other = SessionStartedEvent(
        event_id="evt_other",
        session_id="sess_2",
        sequence=1,
        timestamp=_TIMESTAMP,
    )

    with pytest.raises(ValueError, match="share one session_id"):
        session_state_from_events([_started_event(), other])


def test_state_parse_round_trips_and_rejects_unsupported_schema_versions() -> None:
    state = SessionState(session_id="sess_1", event_count=1, last_event_id="evt_1")
    raw = state.model_dump(mode="json")

    assert parse_state(raw) == state
    assert parse_state_json(dump_state_json(state)) == state

    raw["schema_version"] = 999
    with pytest.raises(UnsupportedSchemaVersionError, match="unsupported state schema version"):
        parse_state(raw)

    raw["schema_version"] = True
    with pytest.raises(UnsupportedSchemaVersionError, match="unsupported state schema version"):
        parse_state(raw)


def test_snapshot_store_invalid_state_fails_clearly(tmp_path: Path) -> None:
    snapshot_store = SnapshotStore(tmp_path, sync_writes=False)
    tmp_path.mkdir(exist_ok=True)
    snapshot_store.path.write_text('{"schema_version":1,"session_id":""}\n', encoding="utf-8")

    with pytest.raises(PersistenceError, match="state snapshot"):
        snapshot_store.read()


def test_snapshot_store_unsupported_state_schema_version_fails_clearly(tmp_path: Path) -> None:
    snapshot_store = SnapshotStore(tmp_path, sync_writes=False)
    tmp_path.mkdir(exist_ok=True)
    snapshot_store.path.write_text(
        '{"schema_version":999,"session_id":"sess_1"}\n',
        encoding="utf-8",
    )

    with pytest.raises(UnsupportedSchemaVersionError, match="unsupported state schema version"):
        snapshot_store.read()
