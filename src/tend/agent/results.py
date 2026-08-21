"""Public turn result schemas."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, JsonValue, model_validator

from tend._common.errors import ErrorInfo
from tend._common.types import JsonObject, StopReason, StrictModel
from tend.agent.persistence.state import SessionState
from tend.llm.context_estimation import ContextEstimate
from tend.llm.models.tools import ToolCall, ToolResult
from tend.llm.usage import Usage

_NonNegativeInt = Annotated[int, Field(ge=0)]


def _empty_json_object() -> JsonObject:
    return {}


def _empty_tool_calls() -> list[ToolCall]:
    return []


def _empty_tool_results() -> list[ToolResult]:
    return []


class StopResult(StrictModel):
    """Structured non-final or diagnostic turn stop details."""

    reason: StopReason
    message: str | None = Field(default=None, min_length=1)
    error: ErrorInfo | None = None
    details: JsonObject = Field(default_factory=_empty_json_object)


class FinalResultOutput(StrictModel):
    """Validated structured output submitted through the ``final_result`` tool."""

    tool_name: Literal["final_result"] = "final_result"
    tool_call_id: str = Field(min_length=1)
    output: JsonValue | None = None
    arguments: JsonObject


class TurnResult(StrictModel):
    """Result returned by one complete ``Agent.run_turn`` invocation."""

    turn_id: str = Field(min_length=1)
    final_response: str | None = None
    final_result: FinalResultOutput | None = None
    stop_reason: StopReason
    stop: StopResult | None = None
    usage: Usage = Field(default_factory=Usage)
    context_estimate: ContextEstimate | None = None
    tool_calls: list[ToolCall] = Field(default_factory=_empty_tool_calls)
    tool_results: list[ToolResult] = Field(default_factory=_empty_tool_results)
    session_id: str | None = Field(default=None, min_length=1)
    session_state: SessionState | None = None
    model_request_count: _NonNegativeInt = 0
    tool_call_count: _NonNegativeInt = 0

    @property
    def final_text(self) -> str | None:
        """Alias for ``final_response`` used by model-layer naming."""

        return self.final_response

    @model_validator(mode="after")
    def _validate_stop_consistency(self) -> TurnResult:
        if self.stop is not None and self.stop.reason is not self.stop_reason:
            raise ValueError("stop.reason must match stop_reason")
        if self.final_response is not None and self.final_result is not None:
            raise ValueError("final_response and final_result are mutually exclusive")
        if self.final_response is not None and self.stop_reason is not StopReason.FINAL_RESPONSE:
            raise ValueError("final_response requires final_response stop reason")
        if self.final_result is not None and self.stop_reason is not StopReason.FINAL_RESULT:
            raise ValueError("final_result requires final_result stop reason")
        if self.stop_reason is StopReason.FINAL_RESPONSE and self.final_response is None:
            raise ValueError("final_response stop reason requires final_response")
        if self.stop_reason is StopReason.FINAL_RESULT and self.final_result is None:
            raise ValueError("final_result stop reason requires final_result")
        if self.stop_reason not in {StopReason.FINAL_RESPONSE, StopReason.FINAL_RESULT}:
            if self.final_response is not None or self.final_result is not None:
                raise ValueError("non-final stops must not include final output")
        return self


__all__ = ("FinalResultOutput", "StopResult", "TurnResult")
