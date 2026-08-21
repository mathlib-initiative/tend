from __future__ import annotations

import asyncio

import pytest
from pydantic import Field, ValidationError

from tend._common.types import JsonObject, StrictModel
from tend.agent.tools import Tool, ToolCancellationState, ToolContext, ToolDefinition


class EchoArguments(StrictModel):
    message: str = Field(min_length=1)
    repeat: int = Field(default=1, ge=1)


async def test_tool_definition_from_arguments_model_exports_strict_schema() -> None:
    async def handler(_context: ToolContext, arguments: EchoArguments) -> dict[str, str]:
        return {"echo": arguments.message}

    tool = Tool.from_arguments_model(
        name="echo",
        description="Echo a message.",
        arguments_model=EchoArguments,
        handler=handler,
        default_timeout_seconds=3.0,
        default_output_limit_bytes=100,
    )

    assert tool.definition.name == "echo"
    assert tool.definition.arguments_schema["type"] == "object"
    assert tool.definition.arguments_schema["additionalProperties"] is False
    assert tool.definition.arguments_schema["required"] == ["message"]
    assert tool.definition.default_timeout_seconds == 3.0
    assert tool.definition.default_output_limit_bytes == 100


async def test_tool_validates_arguments_and_runs_handler_without_wrapping_result() -> None:
    seen: list[tuple[str, str, int]] = []

    async def handler(context: ToolContext, arguments: EchoArguments) -> dict[str, object]:
        seen.append((str(context.cwd), arguments.message, arguments.repeat))
        return {"echo": arguments.message * arguments.repeat}

    tool = Tool.from_arguments_model(
        name="echo",
        description="Echo a message.",
        arguments_model=EchoArguments,
        handler=handler,
    )
    context = ToolContext(cwd="/tmp/work", session_id="session_1", turn_id="turn_1")

    arguments = tool.validate_arguments({"message": "ha", "repeat": 2})
    output = await tool.run(context, arguments)

    assert arguments == EchoArguments(message="ha", repeat=2)
    assert output == {"echo": "haha"}
    assert seen == [("/tmp/work", "ha", 2)]


async def test_tool_argument_preparer_runs_before_strict_validation() -> None:
    async def handler(_context: ToolContext, arguments: EchoArguments) -> str:
        return arguments.message

    def preparer(arguments: JsonObject) -> JsonObject:
        arguments["message"] = arguments.pop("text")
        return arguments

    tool = Tool.from_arguments_model(
        name="echo",
        description="Echo a message.",
        arguments_model=EchoArguments,
        handler=handler,
        argument_preparer=preparer,
    )

    arguments = tool.validate_arguments({"text": "prepared"})
    output = await tool.run(ToolContext(), arguments)

    assert arguments == EchoArguments(message="prepared", repeat=1)
    assert output == "prepared"


async def test_tool_strict_validation_failure_propagates_for_executor_to_convert_later() -> None:
    called = False

    async def handler(_context: ToolContext, _arguments: EchoArguments) -> str:
        nonlocal called
        called = True
        return "not called"

    tool = Tool.from_arguments_model(
        name="echo",
        description="Echo a message.",
        arguments_model=EchoArguments,
        handler=handler,
    )

    with pytest.raises(ValidationError):
        tool.validate_arguments({"message": "ok", "unexpected": True})

    assert called is False


async def test_tool_handler_exception_propagates_for_executor_to_convert_later() -> None:
    async def handler(_context: ToolContext, _arguments: EchoArguments) -> str:
        raise RuntimeError("boom")

    tool = Tool.from_arguments_model(
        name="echo",
        description="Echo a message.",
        arguments_model=EchoArguments,
        handler=handler,
    )
    arguments = tool.validate_arguments({"message": "ok"})

    with pytest.raises(RuntimeError, match="boom"):
        await tool.run(ToolContext(), arguments)


def test_tool_definition_rejects_non_strict_argument_schemas() -> None:
    with pytest.raises(ValidationError, match="JSON object schema"):
        ToolDefinition(name="bad", description="Bad.", arguments_schema={"type": "string"})

    with pytest.raises(ValidationError, match="additionalProperties"):
        ToolDefinition(
            name="bad",
            description="Bad.",
            arguments_schema={"type": "object", "additionalProperties": True},
        )

    with pytest.raises(ValidationError, match="greater than 0"):
        ToolDefinition(name="bad", description="Bad.", default_timeout_seconds=0)


def test_tool_context_event_callback_and_cancellation_state() -> None:
    seen: list[tuple[str, JsonObject]] = []

    def callback(event_type: str, payload: JsonObject) -> None:
        seen.append((event_type, payload))

    cancellation = ToolCancellationState()
    context = ToolContext(
        cwd="/tmp/work",
        session_id="session_1",
        turn_id="turn_1",
        event_callback=callback,
        cancellation=cancellation,
    )

    assert context.is_cancelled is False
    cancellation.cancel()
    assert context.is_cancelled is True

    async def run_emit() -> None:
        await context.emit_event("tool_note", {"value": 1})

    asyncio.run(run_emit())

    assert seen == [("tool_note", {"value": 1})]
