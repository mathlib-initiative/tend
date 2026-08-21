"""Canonical persisted session event schemas.

The event log is the source-of-truth boundary for resumable sessions. These
schemas intentionally use provider-neutral request/response/tool/result models
and artifact references instead of provider-native payloads or secret-bearing
headers.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from enum import StrEnum
from sys import maxsize
from typing import Annotated, Final, Literal, cast

from pydantic import Discriminator, Field, TypeAdapter, model_validator

from tend._common.errors import ErrorInfo, UnsupportedSchemaVersionError
from tend._common.types import JsonObject, StopReason, StrictModel, new_event_id, utc_timestamp
from tend.llm.artifacts import ArtifactRef
from tend.llm.models.requests import ModelRequest, ModelResponse
from tend.llm.models.tools import ToolCall, ToolResult
from tend.llm.usage import Usage

CURRENT_EVENT_SCHEMA_VERSION: Final[int] = 1

_NonNegativeInt = Annotated[int, Field(ge=0)]
_PositiveInt = Annotated[int, Field(ge=1)]
_NonNegativeSeconds = Annotated[float, Field(ge=0)]


def _empty_artifact_refs() -> list[ArtifactRef]:
    return []


class EventType(StrEnum):
    """Canonical v1 event vocabulary written to ``events.jsonl``."""

    SESSION_STARTED = "SessionStarted"
    SESSION_RESUMED = "SessionResumed"
    TURN_STARTED = "TurnStarted"
    MODEL_REQUEST_STARTED = "ModelRequestStarted"
    MODEL_RESPONSE_COMPLETED = "ModelResponseCompleted"
    MODEL_REQUEST_FAILED = "ModelRequestFailed"
    RETRY_SCHEDULED = "RetryScheduled"
    TOOL_CALL_STARTED = "ToolCallStarted"
    TOOL_CALL_COMPLETED = "ToolCallCompleted"
    COMPACTION_STARTED = "CompactionStarted"
    COMPACTION_COMPLETED = "CompactionCompleted"
    TURN_INTERRUPTED = "TurnInterrupted"
    TURN_COMPLETED = "TurnCompleted"


class EventPayload(StrictModel):
    """Common payload fields for provider-neutral events.

    ``artifacts`` is the deliberate escape hatch for large payloads. Secret
    values and request headers are not part of the event payload vocabulary.
    """

    artifacts: list[ArtifactRef] = Field(default_factory=_empty_artifact_refs)
    metadata: JsonObject = Field(default_factory=dict)


class SessionStartedPayload(EventPayload):
    """Payload for a newly created writable session."""

    cwd: str | None = Field(default=None, min_length=1)
    agent_config_artifact: ArtifactRef | None = None
    runtime_config_artifact: ArtifactRef | None = None


class SessionResumedPayload(EventPayload):
    """Payload for reopening an existing session for writable resume."""

    resumed_from_event_id: str | None = Field(default=None, min_length=1)
    state_event_count: _NonNegativeInt | None = None


class TurnStartedPayload(EventPayload):
    """Payload for the start of one user turn."""

    prompt: str | None = Field(default=None, min_length=1)
    prompt_artifact: ArtifactRef | None = None
    input_message_id: str | None = Field(default=None, min_length=1)


class ModelRequestStartedPayload(EventPayload):
    """Payload recorded immediately before invoking a model adapter."""

    request_id: str = Field(min_length=1)
    attempt: _PositiveInt = 1
    request: ModelRequest | None = None
    request_artifact: ArtifactRef | None = None

    @model_validator(mode="after")
    def _validate_request_id(self) -> ModelRequestStartedPayload:
        if self.request is not None and self.request.request_id != self.request_id:
            raise ValueError("request_id must match request.request_id")
        return self


class ModelResponseCompletedPayload(EventPayload):
    """Payload recorded after a model adapter returns a response."""

    request_id: str = Field(min_length=1)
    response_id: str = Field(min_length=1)
    response: ModelResponse | None = None
    response_artifact: ArtifactRef | None = None
    usage: Usage = Field(default_factory=Usage)

    @model_validator(mode="after")
    def _validate_response_ids(self) -> ModelResponseCompletedPayload:
        if self.response is None:
            return self
        if self.response.response_id != self.response_id:
            raise ValueError("response_id must match response.response_id")
        if self.response.request_id is not None and self.response.request_id != self.request_id:
            raise ValueError("request_id must match response.request_id when present")
        return self


class ModelRequestFailedPayload(EventPayload):
    """Payload recorded when a model request attempt fails."""

    request_id: str = Field(min_length=1)
    attempt: _PositiveInt = 1
    error: ErrorInfo
    retryable: bool = False
    usage: Usage = Field(default_factory=Usage)


class RetryScheduledPayload(EventPayload):
    """Payload for provider-neutral retry scheduling decisions."""

    request_id: str = Field(min_length=1)
    attempt: _PositiveInt
    next_attempt: _PositiveInt
    delay_seconds: _NonNegativeSeconds
    retry_after_seconds: _NonNegativeSeconds | None = None
    error: ErrorInfo | None = None

    @model_validator(mode="after")
    def _validate_next_attempt(self) -> RetryScheduledPayload:
        if self.next_attempt <= self.attempt:
            raise ValueError("next_attempt must be greater than attempt")
        return self


class ToolCallStartedPayload(EventPayload):
    """Payload recorded before invoking a tool handler."""

    tool_call: ToolCall


class ToolCallCompletedPayload(EventPayload):
    """Payload recorded after a tool handler returns or fails."""

    result: ToolResult


class CompactionStartedPayload(EventPayload):
    """Payload recorded before generic or future provider-native compaction."""

    compaction_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    planned_message_ids: list[str] = Field(default_factory=list)
    input_artifact: ArtifactRef | None = None


class CompactionCompletedPayload(EventPayload):
    """Payload recorded after compaction creates an active-context summary."""

    compaction_id: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    summary_message_id: str | None = Field(default=None, min_length=1)
    covered_message_ids: list[str] = Field(default_factory=list)
    summary_artifact: ArtifactRef | None = None
    usage: Usage = Field(default_factory=Usage)


class TurnInterruptedPayload(EventPayload):
    """Payload recorded when cancellation or interruption stops a turn."""

    stop_reason: Literal[StopReason.INTERRUPTED] = StopReason.INTERRUPTED
    message: str = Field(min_length=1)
    incomplete_event_id: str | None = Field(default=None, min_length=1)
    error: ErrorInfo | None = None
    usage: Usage = Field(default_factory=Usage)


class TurnCompletedPayload(EventPayload):
    """Payload recorded once a turn reaches a final or structured stop."""

    stop_reason: StopReason
    final_response: str | None = None
    usage: Usage = Field(default_factory=Usage)
    model_request_count: _NonNegativeInt = 0
    tool_call_count: _NonNegativeInt = 0


class EventBase(StrictModel):
    """Common JSONL-friendly event envelope fields."""

    schema_version: Literal[1] = CURRENT_EVENT_SCHEMA_VERSION
    event_id: str = Field(default_factory=new_event_id, min_length=1)
    parent_event_id: str | None = Field(default=None, min_length=1)
    sequence: _NonNegativeInt | None = None
    session_id: str = Field(min_length=1)
    turn_id: str | None = Field(default=None, min_length=1)
    timestamp: str = Field(default_factory=utc_timestamp, min_length=1)


class SessionStartedEvent(EventBase):
    """Event emitted when a session is first created."""

    event_type: Literal[EventType.SESSION_STARTED] = EventType.SESSION_STARTED
    payload: SessionStartedPayload = Field(default_factory=SessionStartedPayload)


class SessionResumedEvent(EventBase):
    """Event emitted when an existing session is resumed."""

    event_type: Literal[EventType.SESSION_RESUMED] = EventType.SESSION_RESUMED
    payload: SessionResumedPayload = Field(default_factory=SessionResumedPayload)


class TurnStartedEvent(EventBase):
    """Event emitted when one user turn starts."""

    event_type: Literal[EventType.TURN_STARTED] = EventType.TURN_STARTED
    payload: TurnStartedPayload = Field(default_factory=TurnStartedPayload)


class ModelRequestStartedEvent(EventBase):
    """Event emitted before a model request starts."""

    event_type: Literal[EventType.MODEL_REQUEST_STARTED] = EventType.MODEL_REQUEST_STARTED
    payload: ModelRequestStartedPayload


class ModelResponseCompletedEvent(EventBase):
    """Event emitted after a model response is received and normalized."""

    event_type: Literal[EventType.MODEL_RESPONSE_COMPLETED] = EventType.MODEL_RESPONSE_COMPLETED
    payload: ModelResponseCompletedPayload


class ModelRequestFailedEvent(EventBase):
    """Event emitted after a model request attempt fails."""

    event_type: Literal[EventType.MODEL_REQUEST_FAILED] = EventType.MODEL_REQUEST_FAILED
    payload: ModelRequestFailedPayload


class RetryScheduledEvent(EventBase):
    """Event emitted when retry policy schedules another attempt."""

    event_type: Literal[EventType.RETRY_SCHEDULED] = EventType.RETRY_SCHEDULED
    payload: RetryScheduledPayload


class ToolCallStartedEvent(EventBase):
    """Event emitted before a tool call starts."""

    event_type: Literal[EventType.TOOL_CALL_STARTED] = EventType.TOOL_CALL_STARTED
    payload: ToolCallStartedPayload


class ToolCallCompletedEvent(EventBase):
    """Event emitted after a tool call completes with success or failure."""

    event_type: Literal[EventType.TOOL_CALL_COMPLETED] = EventType.TOOL_CALL_COMPLETED
    payload: ToolCallCompletedPayload


class CompactionStartedEvent(EventBase):
    """Event emitted before a compaction request starts."""

    event_type: Literal[EventType.COMPACTION_STARTED] = EventType.COMPACTION_STARTED
    payload: CompactionStartedPayload


class CompactionCompletedEvent(EventBase):
    """Event emitted after a compaction summary is produced."""

    event_type: Literal[EventType.COMPACTION_COMPLETED] = EventType.COMPACTION_COMPLETED
    payload: CompactionCompletedPayload


class TurnInterruptedEvent(EventBase):
    """Event emitted when a turn is interrupted before normal completion."""

    event_type: Literal[EventType.TURN_INTERRUPTED] = EventType.TURN_INTERRUPTED
    payload: TurnInterruptedPayload


class TurnCompletedEvent(EventBase):
    """Event emitted when a turn ends for any structured stop reason."""

    event_type: Literal[EventType.TURN_COMPLETED] = EventType.TURN_COMPLETED
    payload: TurnCompletedPayload


_EVENT_DISCRIMINATOR: Discriminator = Discriminator("event_type")

type SessionEvent = Annotated[
    SessionStartedEvent
    | SessionResumedEvent
    | TurnStartedEvent
    | ModelRequestStartedEvent
    | ModelResponseCompletedEvent
    | ModelRequestFailedEvent
    | RetryScheduledEvent
    | ToolCallStartedEvent
    | ToolCallCompletedEvent
    | CompactionStartedEvent
    | CompactionCompletedEvent
    | TurnInterruptedEvent
    | TurnCompletedEvent,
    _EVENT_DISCRIMINATOR,
]

_SESSION_EVENT_ADAPTER: TypeAdapter[SessionEvent] = TypeAdapter(SessionEvent)


def parse_event(value: object) -> SessionEvent:
    """Validate a Python or JSON-compatible value as a supported event.

    Strict Pydantic models expect enum instances for in-memory Python objects,
    while JSONL stores enum values as strings. Mapping inputs are therefore
    treated as JSON-compatible payloads and validated through Pydantic's JSON
    path so event-log round trips deserialize cleanly.
    """

    _ensure_supported_schema_version(value)
    if isinstance(value, Mapping):
        data = json.dumps(value, separators=(",", ":"))
        return _SESSION_EVENT_ADAPTER.validate_json(data)
    return _SESSION_EVENT_ADAPTER.validate_python(value)


def parse_event_json(data: str | bytes | bytearray) -> SessionEvent:
    """Validate a JSON object string as a supported persisted event."""

    raw: object = json.loads(data)
    _ensure_supported_schema_version(raw)
    return _SESSION_EVENT_ADAPTER.validate_json(data)


def dump_event_json(event: SessionEvent) -> str:
    """Serialize one event as compact JSON suitable for a JSONL line."""

    return _SESSION_EVENT_ADAPTER.dump_json(event, by_alias=False).decode("utf-8")


def next_event_sequence(events: Iterable[EventBase]) -> int:
    """Return the next monotonic sequence number after existing events."""

    highest = -1
    for event in events:
        if event.sequence is not None and event.sequence > highest:
            highest = event.sequence
    return highest + 1


def event_order_key(event: EventBase) -> tuple[int, str]:
    """Return a deterministic ordering key using sequence then event ID."""

    sequence = event.sequence if event.sequence is not None else maxsize
    return (sequence, event.event_id)


def _ensure_supported_schema_version(value: object) -> None:
    if isinstance(value, EventBase):
        return
    if not isinstance(value, Mapping):
        return
    if "schema_version" not in value:
        return
    schema_version = cast(object, value["schema_version"])
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != CURRENT_EVENT_SCHEMA_VERSION
    ):
        raise UnsupportedSchemaVersionError(
            "unsupported event schema version "
            f"{schema_version!r}; supported version is {CURRENT_EVENT_SCHEMA_VERSION}"
        )


__all__ = (
    "CURRENT_EVENT_SCHEMA_VERSION",
    "CompactionCompletedEvent",
    "CompactionCompletedPayload",
    "CompactionStartedEvent",
    "CompactionStartedPayload",
    "EventBase",
    "EventPayload",
    "EventType",
    "ModelRequestFailedEvent",
    "ModelRequestFailedPayload",
    "ModelRequestStartedEvent",
    "ModelRequestStartedPayload",
    "ModelResponseCompletedEvent",
    "ModelResponseCompletedPayload",
    "RetryScheduledEvent",
    "RetryScheduledPayload",
    "SessionEvent",
    "SessionResumedEvent",
    "SessionResumedPayload",
    "SessionStartedEvent",
    "SessionStartedPayload",
    "ToolCallCompletedEvent",
    "ToolCallCompletedPayload",
    "ToolCallStartedEvent",
    "ToolCallStartedPayload",
    "TurnCompletedEvent",
    "TurnCompletedPayload",
    "TurnInterruptedEvent",
    "TurnInterruptedPayload",
    "TurnStartedEvent",
    "TurnStartedPayload",
    "dump_event_json",
    "event_order_key",
    "next_event_sequence",
    "parse_event",
    "parse_event_json",
)
