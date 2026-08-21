from __future__ import annotations

from pathlib import Path

import pytest

from tend._common.errors import PersistenceError
from tend.agent.config import ArtifactConfig, LoggingConfig, RuntimeConfig
from tend.agent.observability import DetailedPayloadRecorder
from tend.agent.persistence.artifacts import (
    ARTIFACT_SUBDIRECTORIES,
    ArtifactKind,
    ArtifactNameStrategy,
    ArtifactStore,
    payload_size_bytes,
    should_inline_size,
)
from tend.agent.persistence.events import ModelRequestStartedEvent, ModelRequestStartedPayload
from tend.agent.session import Session
from tend.llm.models import ModelRequest
from tend.llm.redaction import Redactor
from tend.llm.secrets import REDACTED_VALUE


def test_artifact_store_writes_reads_reference_and_creates_layout(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path, sync_writes=False)

    ref = store.write_text(
        ArtifactKind.TOOL_OUTPUT,
        "full tool output",
        event_id="evt_tool_1",
    )

    assert ref.kind == ArtifactKind.TOOL_OUTPUT.value
    assert ref.path == "tool_outputs/evt_tool_1.txt"
    assert ref.size_bytes == len(b"full tool output")
    assert ref.sha256 is not None
    assert store.read_text(ref) == "full tool output"
    for subdirectory in ARTIFACT_SUBDIRECTORIES.values():
        assert (tmp_path / "artifacts" / subdirectory).is_dir()


def test_artifact_store_can_name_by_content_hash(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path, sync_writes=False)

    first = store.write_json(
        ArtifactKind.MODEL_REQUEST,
        {"request": "same"},
        name_strategy=ArtifactNameStrategy.CONTENT_HASH,
    )
    second = store.write_json(
        ArtifactKind.MODEL_REQUEST,
        {"request": "same"},
        name_strategy=ArtifactNameStrategy.CONTENT_HASH,
    )

    assert first == second
    assert first.sha256 is not None
    assert first.path == f"model_requests/{first.sha256}.json"
    assert store.read_json(first) == {"request": "same"}


def test_artifact_reference_paths_must_stay_under_artifact_root(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path, sync_writes=False)
    ref = store.write_text(ArtifactKind.TOOL_OUTPUT, "safe", event_id="evt_safe")
    unsafe = ref.model_copy(update={"path": "../outside.txt"})

    with pytest.raises(PersistenceError, match="must not contain"):
        store.read_text(unsafe)


def test_inline_threshold_records_inline_or_artifact(tmp_path: Path) -> None:
    runtime = RuntimeConfig(
        logging=LoggingConfig(detailed=True, include_model_payloads=True),
        artifacts=ArtifactConfig(inline_threshold_bytes=256),
    )
    recorder = DetailedPayloadRecorder(tmp_path, runtime_config=runtime, sync_writes=False)

    small = recorder.capture_model_request(
        ModelRequest(request_id="req_small", model_name="scripted"),
        event_id="evt_small",
    )
    large = recorder.capture_model_request(
        ModelRequest(
            request_id="req_large",
            model_name="scripted",
            request_metadata={"x": "y" * 100},
        ),
        event_id="evt_large",
    )

    assert small.inlined is True
    assert small.artifact is None
    assert small.inline_payload is not None
    assert large.inlined is False
    assert large.artifact is not None
    assert large.artifact.path == "model_requests/evt_large.json"
    assert should_inline_size(payload_size_bytes({"a": "b"}), inline_threshold_bytes=256)


def test_detailed_logging_disabled_still_allows_minimal_events(tmp_path: Path) -> None:
    runtime = RuntimeConfig(logging=LoggingConfig(detailed=False))
    recorder = DetailedPayloadRecorder(tmp_path, runtime_config=runtime, sync_writes=False)
    request = ModelRequest(request_id="req_1", model_name="scripted")

    record = recorder.capture_model_request(request, event_id="evt_request_1")
    event = ModelRequestStartedEvent(
        event_id="evt_request_1",
        session_id="sess_1",
        turn_id="turn_1",
        sequence=1,
        payload=ModelRequestStartedPayload(
            request_id=request.request_id,
            request=None,
            request_artifact=record.artifact,
        ),
    )

    assert record.detailed_logging_enabled is False
    assert record.inlined is False
    assert record.inline_payload is None
    assert record.artifact is None
    assert recorder.capture_tool_output(object(), event_id="evt_non_json").omitted_reason == (
        "detailed_logging_disabled"
    )
    assert event.payload.request_id == "req_1"
    assert event.payload.request is None
    assert event.payload.request_artifact is None

    with Session.create(tmp_path / "session", session_id="sess_1", sync_writes=False) as session:
        minimal = event.model_copy(update={"sequence": session.next_sequence})
        session.append_event(minimal)
        assert session.state.incomplete_model_requests["req_1"].request is None


def test_redaction_is_applied_before_writing_detailed_artifacts(tmp_path: Path) -> None:
    runtime = RuntimeConfig(
        logging=LoggingConfig(detailed=True, include_tool_outputs=True),
        artifacts=ArtifactConfig(inline_threshold_bytes=0),
    )
    recorder = DetailedPayloadRecorder(
        tmp_path,
        runtime_config=runtime,
        redactor=Redactor(
            secret_values=["fake-secret-value"],
            secret_source_names=["FAKE_API_TOKEN"],
        ),
        sync_writes=False,
    )

    record = recorder.capture_tool_output(
        "FAKE_API_TOKEN=fake-secret-value\nvisible output",
        event_id="evt_tool_secret",
    )

    assert record.artifact is not None
    stored = recorder.artifact_store.read_text(record.artifact)
    assert "fake-secret-value" not in stored
    assert f"FAKE_API_TOKEN={REDACTED_VALUE}" in stored
    assert "visible output" in stored
