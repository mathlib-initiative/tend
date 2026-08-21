"""Sequential provider-neutral tool-call executor."""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping
from typing import Any, cast

from pydantic import BaseModel, JsonValue, TypeAdapter, ValidationError

from tend._common.types import JsonObject, utc_timestamp
from tend.agent.tools.base import Tool
from tend.agent.tools.context import ToolContext
from tend.llm.models.tools import ToolCall, ToolError, ToolResult
from tend.llm.truncation import TruncationInfo

type EnabledTools = Iterable[Tool[Any]] | Mapping[str, Tool[Any]]

TOOL_CALL_STARTED_EVENT = "ToolCallStarted"
TOOL_CALL_COMPLETED_EVENT = "ToolCallCompleted"

_JSON_VALUE_ADAPTER: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)
_JSON_OBJECT_ADAPTER: TypeAdapter[JsonObject] = TypeAdapter(JsonObject)


async def execute_tool_calls(
    tool_calls: Iterable[ToolCall],
    tools: EnabledTools,
    context: ToolContext,
) -> tuple[ToolResult, ...]:
    """Execute provider-neutral tool calls sequentially in provider order.

    Calls are ordered by ``ToolCall.order`` with caller-provided input order as a
    stable tie-breaker. Validation failures, unknown tools, handler-returned
    errors, and raised handler exceptions all become model-visible
    ``ToolResult(success=False)`` values; they do not stop later calls from
    running.
    """

    enabled_tools = _enabled_tool_map(tools)
    ordered_calls = sorted(enumerate(tuple(tool_calls)), key=lambda item: (item[1].order, item[0]))
    results: list[ToolResult] = []

    for _input_index, tool_call in ordered_calls:
        if context.is_cancelled:
            break
        result = await _execute_one(tool_call, enabled_tools, context)
        results.append(result)

    return tuple(results)


async def _execute_one(
    tool_call: ToolCall,
    enabled_tools: Mapping[str, Tool[Any]],
    context: ToolContext,
) -> ToolResult:
    started_at = utc_timestamp()
    started_perf = time.perf_counter()
    await _emit_started(context, tool_call, started_at=started_at)

    tool = enabled_tools.get(tool_call.tool_name)
    if tool is None:
        result = _failure_result(
            tool_call,
            error=ToolError(
                error_type="unknown_tool",
                message=f"Unknown tool requested: {tool_call.tool_name}",
                details=_json_object(
                    {
                        "tool_name": tool_call.tool_name,
                        "enabled_tools": sorted(enabled_tools),
                    }
                ),
            ),
            output=f"[Tool error: unknown tool requested: {tool_call.tool_name}]",
            started_at=started_at,
            started_perf=started_perf,
        )
        await _emit_completed(context, result)
        return result

    try:
        validated_arguments = tool.validate_arguments(tool_call.arguments)
    except ValidationError as exc:
        result = _failure_result(
            tool_call,
            error=ToolError(
                error_type="validation_error",
                message=f"Arguments for tool '{tool_call.tool_name}' failed validation.",
                details=_json_object(
                    {
                        "tool_name": tool_call.tool_name,
                        "arguments": tool_call.arguments,
                        "validation_errors": _validation_error_details(exc),
                    }
                ),
            ),
            output=f"[Tool argument validation error: {exc.errors(include_url=False)[0]['msg']}]",
            started_at=started_at,
            started_perf=started_perf,
        )
        await _emit_completed(context, result)
        return result
    except Exception as exc:
        result = _failure_result(
            tool_call,
            error=ToolError(
                error_type="validation_error",
                message=f"Argument preparation/validation failed: {type(exc).__name__}: {exc}",
                details=_json_object(
                    {
                        "tool_name": tool_call.tool_name,
                        "arguments": tool_call.arguments,
                        "exception_type": type(exc).__name__,
                        "exception_message": str(exc),
                    }
                ),
            ),
            output=f"[Tool argument validation error: {type(exc).__name__}: {exc}]",
            started_at=started_at,
            started_perf=started_perf,
        )
        await _emit_completed(context, result)
        return result

    try:
        handler_output = await tool.run(context, validated_arguments)
    except Exception as exc:
        result = _failure_result(
            tool_call,
            error=ToolError(
                error_type="handler_exception",
                message=f"{type(exc).__name__}: {exc}",
                details=_json_object(
                    {
                        "tool_name": tool_call.tool_name,
                        "exception_type": type(exc).__name__,
                        "exception_message": str(exc),
                    }
                ),
            ),
            output=f"[Tool handler exception: {type(exc).__name__}: {exc}]",
            started_at=started_at,
            started_perf=started_perf,
        )
    else:
        result = _result_from_handler_output(
            tool_call,
            handler_output,
            started_at=started_at,
            started_perf=started_perf,
        )

    await _emit_completed(context, result)
    return result


async def _emit_started(context: ToolContext, tool_call: ToolCall, *, started_at: str) -> None:
    await context.emit_event(
        TOOL_CALL_STARTED_EVENT,
        {
            "tool_call_id": tool_call.call_id,
            "tool_name": tool_call.tool_name,
            "order": tool_call.order,
            "started_at": started_at,
            "provider_item_id": tool_call.provider_item_id,
            "provider_call_id": tool_call.provider_call_id,
            "provider_tool_use_id": tool_call.provider_tool_use_id,
            "tool_call": tool_call.model_dump(mode="json"),
        },
    )


async def _emit_completed(context: ToolContext, result: ToolResult) -> None:
    await context.emit_event(
        TOOL_CALL_COMPLETED_EVENT,
        {
            "tool_call_id": result.tool_call_id,
            "tool_name": result.tool_name,
            "order": result.order,
            "success": result.success,
            "timed_out": result.timed_out,
            "truncated": result.truncated,
            "started_at": result.started_at,
            "ended_at": result.ended_at,
            "duration_ms": result.duration_ms,
            "error": None if result.error is None else result.error.model_dump(mode="json"),
            "result": result.model_dump(mode="json"),
        },
    )


def _enabled_tool_map(tools: EnabledTools) -> dict[str, Tool[Any]]:
    if isinstance(tools, Mapping):
        values: Iterable[Tool[Any]] = cast(Mapping[str, Tool[Any]], tools).values()
    else:
        values = tools

    enabled: dict[str, Tool[Any]] = {}
    for tool in values:
        if tool.name in enabled:
            raise ValueError(f"duplicate enabled tool name: {tool.name}")
        enabled[tool.name] = tool
    return enabled


def _result_from_handler_output(
    tool_call: ToolCall,
    handler_output: object,
    *,
    started_at: str,
    started_perf: float,
) -> ToolResult:
    if isinstance(handler_output, ToolResult):
        return _tool_result(
            tool_call,
            success=handler_output.success,
            output=handler_output.output,
            error=handler_output.error,
            started_at=started_at,
            started_perf=started_perf,
            timed_out=handler_output.timed_out,
            truncated=handler_output.truncated,
            truncation=handler_output.truncation,
        )

    if isinstance(handler_output, ToolError):
        return _failure_result(
            tool_call,
            error=handler_output,
            output=f"[Tool error: {handler_output.message}]",
            started_at=started_at,
            started_perf=started_perf,
        )

    if isinstance(handler_output, BaseModel):
        return _result_from_model(
            tool_call,
            handler_output,
            started_at=started_at,
            started_perf=started_perf,
        )

    return _tool_result(
        tool_call,
        success=True,
        output=_json_value(handler_output),
        error=None,
        started_at=started_at,
        started_perf=started_perf,
    )


def _result_from_model(
    tool_call: ToolCall,
    model: BaseModel,
    *,
    started_at: str,
    started_perf: float,
) -> ToolResult:
    dumped: dict[str, object] = model.model_dump(mode="python")
    output = (
        _json_value(dumped["output"])
        if "output" in dumped
        else _json_value(model.model_dump(mode="json"))
    )
    success = _bool_field(dumped, "success", default=True)
    error = _parse_tool_error(dumped.get("error"))
    timed_out = _bool_field(dumped, "timed_out", default=False)
    truncation = _parse_truncation(dumped.get("truncation"))
    truncated_without_metadata = (
        _bool_field(dumped, "truncated", default=False) and truncation is None
    )
    if truncated_without_metadata:
        return _failure_result(
            tool_call,
            error=ToolError(
                error_type="tool_result_error",
                message=(
                    f"Tool '{tool_call.tool_name}' returned truncated=true without "
                    "truncation metadata."
                ),
                details={"tool_name": tool_call.tool_name, "output": output},
            ),
            output=f"[Tool result error: missing truncation metadata for {tool_call.tool_name}]",
            started_at=started_at,
            started_perf=started_perf,
        )
    truncated = truncation.truncated if truncation is not None else False

    if error is not None:
        success = False
    if timed_out and error is None:
        success = False
        error = ToolError(
            error_type="timeout",
            message=f"Tool '{tool_call.tool_name}' timed out.",
            details={"tool_name": tool_call.tool_name},
        )
    if not success and error is None:
        error = ToolError(
            error_type="tool_error",
            message=f"Tool '{tool_call.tool_name}' returned an unsuccessful result.",
            details={"tool_name": tool_call.tool_name, "output": output},
        )

    return _tool_result(
        tool_call,
        success=success,
        output=output,
        error=error,
        started_at=started_at,
        started_perf=started_perf,
        timed_out=timed_out,
        truncated=truncated,
        truncation=truncation,
    )


def _tool_result(
    tool_call: ToolCall,
    *,
    success: bool,
    output: JsonValue | None,
    error: ToolError | None,
    started_at: str,
    started_perf: float,
    timed_out: bool = False,
    truncated: bool = False,
    truncation: TruncationInfo | None = None,
) -> ToolResult:
    ended_at = utc_timestamp()
    return ToolResult(
        tool_call_id=tool_call.call_id,
        tool_name=tool_call.tool_name,
        arguments=tool_call.arguments,
        success=success,
        output=output,
        error=error,
        started_at=started_at,
        ended_at=ended_at,
        duration_ms=_duration_ms_since(started_perf),
        timed_out=timed_out,
        truncated=truncated,
        truncation=truncation,
        order=tool_call.order,
        provider_item_id=tool_call.provider_item_id,
        provider_call_id=tool_call.provider_call_id,
        provider_tool_use_id=tool_call.provider_tool_use_id,
        provider_metadata=tool_call.provider_metadata,
    )


def _failure_result(
    tool_call: ToolCall,
    *,
    error: ToolError,
    output: str,
    started_at: str,
    started_perf: float,
    timed_out: bool = False,
) -> ToolResult:
    return _tool_result(
        tool_call,
        success=False,
        output=output,
        error=error,
        started_at=started_at,
        started_perf=started_perf,
        timed_out=timed_out,
    )


def _duration_ms_since(started_perf: float) -> float:
    return max((time.perf_counter() - started_perf) * 1000.0, 0.0)


def _validation_error_details(error: ValidationError) -> JsonValue:
    return _JSON_VALUE_ADAPTER.validate_json(
        error.json(include_url=False, include_context=False)
    )


def _bool_field(values: Mapping[str, object], field_name: str, *, default: bool) -> bool:
    value = values.get(field_name)
    if isinstance(value, bool):
        return value
    return default


def _parse_tool_error(value: object) -> ToolError | None:
    if value is None:
        return None
    if isinstance(value, ToolError):
        return value
    try:
        return ToolError.model_validate(value)
    except ValidationError:
        return ToolError(
            error_type="tool_error",
            message="Tool returned an invalid structured error payload.",
            details={"returned_error": _json_value(value)},
        )


def _parse_truncation(value: object) -> TruncationInfo | None:
    if value is None:
        return None
    if isinstance(value, TruncationInfo):
        return value
    try:
        return TruncationInfo.model_validate(value)
    except ValidationError:
        return None


def _json_value(value: object) -> JsonValue:
    if isinstance(value, BaseModel):
        return _json_value(value.model_dump(mode="json"))
    try:
        return _JSON_VALUE_ADAPTER.validate_python(value)
    except ValidationError:
        return str(value)


def _json_object(value: Mapping[str, object]) -> JsonObject:
    return _JSON_OBJECT_ADAPTER.validate_python(dict(value))


__all__ = (
    "EnabledTools",
    "TOOL_CALL_COMPLETED_EVENT",
    "TOOL_CALL_STARTED_EVENT",
    "execute_tool_calls",
)
