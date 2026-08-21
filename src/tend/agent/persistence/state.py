"""Persisted session state snapshot schemas.

``events.jsonl`` remains canonical. ``state.json`` is a cache rebuilt by
replaying the event log; it records enough completed/incomplete operation
identity to make resume decisions without rerunning completed effects.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Annotated, Final, Literal, cast

from pydantic import Field, TypeAdapter

from tend._common.errors import ErrorInfo, UnsupportedSchemaVersionError
from tend._common.types import JsonObject, StrictModel
from tend.agent.persistence.events import EventBase
from tend.llm.context_estimation import ContextEstimate
from tend.llm.models.requests import ModelRequest
from tend.llm.models.tools import ToolCall, ToolResult
from tend.llm.usage import Usage

CURRENT_STATE_SCHEMA_VERSION: Final[int] = 1

_NonNegativeInt = Annotated[int, Field(ge=0)]
_PositiveInt = Annotated[int, Field(ge=1)]
_NonNegativeOrder = Annotated[int, Field(ge=0)]


def _empty_json_object() -> JsonObject:
    return {}


class CompletedModelRequest(StrictModel):
    """A model request that has a persisted normalized response."""

    request_id: str = Field(min_length=1)
    started_event_id: str = Field(min_length=1)
    completed_event_id: str = Field(min_length=1)
    turn_id: str | None = Field(default=None, min_length=1)
    attempt: _PositiveInt = 1
    response_id: str = Field(min_length=1)
    provider_response_id: str | None = Field(default=None, min_length=1)
    provider_name: str | None = Field(default=None, min_length=1)
    failed_event_ids: list[str] = Field(default_factory=list)


class IncompleteModelRequest(StrictModel):
    """A model request attempt that was started but has no terminal event."""

    request_id: str = Field(min_length=1)
    started_event_id: str = Field(min_length=1)
    turn_id: str | None = Field(default=None, min_length=1)
    attempt: _PositiveInt = 1
    request: ModelRequest | None = None
    failed_event_ids: list[str] = Field(default_factory=list)
    last_failed_event_id: str | None = Field(default=None, min_length=1)
    last_error: ErrorInfo | None = None
    interrupted_event_id: str | None = Field(default=None, min_length=1)


class CompletedToolCall(StrictModel):
    """A tool call with a persisted final ``ToolResult``."""

    tool_call_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    started_event_id: str = Field(min_length=1)
    completed_event_id: str = Field(min_length=1)
    turn_id: str | None = Field(default=None, min_length=1)
    order: _NonNegativeOrder = 0
    result: ToolResult


class InterruptedToolCall(StrictModel):
    """A started tool call that is marked interrupted during replay.

    Replay never reruns tools. The synthetic result is model-visible state for a
    later turn-loop phase; it is not appended to the event log by replay.
    """

    tool_call_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    started_event_id: str = Field(min_length=1)
    interrupted_event_id: str | None = Field(default=None, min_length=1)
    turn_id: str | None = Field(default=None, min_length=1)
    order: _NonNegativeOrder = 0
    tool_call: ToolCall
    result: ToolResult


class CompletedCompaction(StrictModel):
    """A completed active-context compaction summary from replayed events."""

    compaction_id: str = Field(min_length=1)
    started_event_id: str = Field(min_length=1)
    completed_event_id: str = Field(min_length=1)
    turn_id: str | None = Field(default=None, min_length=1)
    reason: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    summary_message_id: str | None = Field(default=None, min_length=1)
    covered_message_ids: list[str] = Field(default_factory=list)
    planned_message_ids: list[str] = Field(default_factory=list)
    preserved_message_ids: list[str] = Field(default_factory=list)
    compact_start_index: _NonNegativeInt | None = None
    compact_end_index: _NonNegativeInt | None = None
    split_turn_prefix: bool = False
    usage: Usage = Field(default_factory=Usage)
    config: JsonObject = Field(default_factory=_empty_json_object)
    plan: JsonObject = Field(default_factory=_empty_json_object)
    metadata: JsonObject = Field(default_factory=_empty_json_object)


class SessionState(StrictModel):
    """Atomic ``state.json`` snapshot/cache rebuilt from canonical events."""

    schema_version: Literal[1] = CURRENT_STATE_SCHEMA_VERSION
    session_id: str = Field(min_length=1)
    event_count: _NonNegativeInt = 0
    last_event_id: str | None = Field(default=None, min_length=1)
    last_sequence: _NonNegativeInt | None = None
    metadata: JsonObject = Field(default_factory=_empty_json_object)
    usage: Usage = Field(default_factory=Usage)
    turn_usage: dict[str, Usage] = Field(default_factory=dict)
    model_request_usage: dict[str, Usage] = Field(default_factory=dict)
    compaction_usage: dict[str, Usage] = Field(default_factory=dict)
    completed_compactions: dict[str, CompletedCompaction] = Field(default_factory=dict)
    model_request_context_estimates: dict[str, ContextEstimate] = Field(default_factory=dict)
    latest_context_estimate: ContextEstimate | None = None

    completed_model_requests: dict[str, CompletedModelRequest] = Field(default_factory=dict)
    incomplete_model_requests: dict[str, IncompleteModelRequest] = Field(default_factory=dict)
    model_request_start_event_ids: dict[str, str] = Field(default_factory=dict)
    model_response_completed_event_ids: dict[str, str] = Field(default_factory=dict)
    provider_response_ids: dict[str, str] = Field(default_factory=dict)

    completed_tool_calls: dict[str, CompletedToolCall] = Field(default_factory=dict)
    interrupted_tool_calls: dict[str, InterruptedToolCall] = Field(default_factory=dict)
    tool_call_start_event_ids: dict[str, str] = Field(default_factory=dict)
    tool_call_completed_event_ids: dict[str, str] = Field(default_factory=dict)


_SESSION_STATE_ADAPTER: Final[TypeAdapter[SessionState]] = TypeAdapter(SessionState)


def session_state_from_events(
    events: Iterable[EventBase],
    *,
    session_id: str | None = None,
) -> SessionState:
    """Replay events into a deterministic ``SessionState`` cache.

    ``events.jsonl`` is canonical; callers may rebuild this value even when an
    existing ``state.json`` snapshot is stale. The local import avoids a module
    cycle because ``replay`` depends on the state models defined above.
    """

    from tend.agent.persistence.replay import replay_events

    return replay_events(events, session_id=session_id)


def parse_state(value: object) -> SessionState:
    """Validate a Python or JSON-compatible value as a supported state snapshot."""

    if isinstance(value, SessionState):
        return value
    _ensure_supported_state_schema_version(value)
    if isinstance(value, Mapping):
        data = json.dumps(value, separators=(",", ":"))
        return _SESSION_STATE_ADAPTER.validate_json(data)
    return _SESSION_STATE_ADAPTER.validate_python(value)


def parse_state_json(data: str | bytes | bytearray) -> SessionState:
    """Validate a JSON object string as a supported persisted state snapshot."""

    raw: object = json.loads(data)
    _ensure_supported_state_schema_version(raw)
    return _SESSION_STATE_ADAPTER.validate_json(data)


def dump_state_json(state: SessionState) -> str:
    """Serialize one state snapshot as compact JSON suitable for ``state.json``."""

    return state.model_dump_json()


def _ensure_supported_state_schema_version(value: object) -> None:
    if isinstance(value, SessionState):
        return
    if not isinstance(value, Mapping):
        return
    if "schema_version" not in value:
        return
    schema_version = cast(object, value["schema_version"])
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != CURRENT_STATE_SCHEMA_VERSION
    ):
        raise UnsupportedSchemaVersionError(
            "unsupported state schema version "
            f"{schema_version!r}; supported version is {CURRENT_STATE_SCHEMA_VERSION}"
        )


__all__ = (
    "CURRENT_STATE_SCHEMA_VERSION",
    "CompletedCompaction",
    "CompletedModelRequest",
    "CompletedToolCall",
    "IncompleteModelRequest",
    "InterruptedToolCall",
    "SessionState",
    "dump_state_json",
    "parse_state",
    "parse_state_json",
    "session_state_from_events",
)
