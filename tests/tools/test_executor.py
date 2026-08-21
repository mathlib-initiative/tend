from __future__ import annotations

from pydantic import Field

from tend._common.types import JsonObject, StrictModel
from tend.agent.tools import Tool, ToolContext, execute_tool_calls
from tend.llm.models.tools import ToolCall, ToolError
from tend.llm.truncation import TruncationInfo, TruncationPolicy


class EchoArguments(StrictModel):
    message: str = Field(min_length=1)


async def test_successful_sequence_of_two_calls_uses_provider_order() -> None:
    seen_messages: list[str] = []
    seen_events: list[tuple[str, JsonObject]] = []

    async def handler(_context: ToolContext, arguments: EchoArguments) -> JsonObject:
        seen_messages.append(arguments.message)
        return {"echo": arguments.message}

    def event_callback(event_type: str, payload: JsonObject) -> None:
        seen_events.append((event_type, payload))

    tool = Tool.from_arguments_model(
        name="echo",
        description="Echo a message.",
        arguments_model=EchoArguments,
        handler=handler,
    )
    calls = [
        ToolCall(
            call_id="call_second",
            tool_name="echo",
            arguments={"message": "second"},
            order=2,
            provider_item_id="item_second",
            provider_call_id="provider_call_second",
        ),
        ToolCall(
            call_id="call_first",
            tool_name="echo",
            arguments={"message": "first"},
            order=1,
            provider_item_id="item_first",
            provider_call_id="provider_call_first",
        ),
    ]

    results = await execute_tool_calls(calls, [tool], ToolContext(event_callback=event_callback))

    assert seen_messages == ["first", "second"]
    assert [result.tool_call_id for result in results] == ["call_first", "call_second"]
    assert [result.order for result in results] == [1, 2]
    assert results[0].success is True
    assert results[0].output == {"echo": "first"}
    assert results[0].provider_item_id == "item_first"
    assert results[0].provider_call_id == "provider_call_first"
    assert results[0].started_at is not None
    assert results[0].ended_at is not None
    assert results[0].duration_ms is not None
    assert results[0].duration_ms >= 0
    assert [event_type for event_type, _payload in seen_events] == [
        "ToolCallStarted",
        "ToolCallCompleted",
        "ToolCallStarted",
        "ToolCallCompleted",
    ]
    assert seen_events[0][1]["tool_call_id"] == "call_first"
    assert seen_events[1][1]["success"] is True


async def test_validation_failure_returns_failed_tool_result_without_handler_call() -> None:
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
    call = ToolCall(call_id="call_bad_args", tool_name="echo", arguments={}, order=0)

    results = await execute_tool_calls([call], [tool], ToolContext())

    assert called is False
    assert len(results) == 1
    result = results[0]
    assert result.success is False
    assert result.tool_call_id == "call_bad_args"
    assert result.error is not None
    assert result.error.error_type == "validation_error"
    assert result.arguments == {}
    assert result.timed_out is False
    assert result.truncated is False


async def test_handler_exception_returns_failed_result_and_later_calls_continue() -> None:
    completed: list[str] = []

    async def raising_handler(_context: ToolContext, _arguments: EchoArguments) -> str:
        raise RuntimeError("boom")

    async def ok_handler(_context: ToolContext, arguments: EchoArguments) -> str:
        completed.append(arguments.message)
        return "ok"

    bad_tool = Tool.from_arguments_model(
        name="bad",
        description="Raise.",
        arguments_model=EchoArguments,
        handler=raising_handler,
    )
    ok_tool = Tool.from_arguments_model(
        name="ok",
        description="Succeed.",
        arguments_model=EchoArguments,
        handler=ok_handler,
    )

    results = await execute_tool_calls(
        [
            ToolCall(call_id="call_bad", tool_name="bad", arguments={"message": "bad"}, order=0),
            ToolCall(call_id="call_ok", tool_name="ok", arguments={"message": "after"}, order=1),
        ],
        [bad_tool, ok_tool],
        ToolContext(),
    )

    assert [result.tool_call_id for result in results] == ["call_bad", "call_ok"]
    assert results[0].success is False
    assert results[0].error is not None
    assert results[0].error.error_type == "handler_exception"
    assert "boom" in results[0].error.message
    assert results[1].success is True
    assert results[1].output == "ok"
    assert completed == ["after"]


async def test_unknown_tool_returns_failed_tool_result() -> None:
    call = ToolCall(
        call_id="call_missing",
        tool_name="missing",
        arguments={"value": 1},
        order=0,
        provider_tool_use_id="anthropic_tool_use_id",
    )

    results = await execute_tool_calls([call], [], ToolContext())

    assert len(results) == 1
    result = results[0]
    assert result.success is False
    assert result.tool_call_id == "call_missing"
    assert result.tool_name == "missing"
    assert result.provider_tool_use_id == "anthropic_tool_use_id"
    assert result.error is not None
    assert result.error.error_type == "unknown_tool"
    assert result.error.details["enabled_tools"] == []


async def test_handler_returned_error_becomes_failed_tool_result() -> None:
    class ReturnedError(StrictModel):
        success: bool
        output: str
        error: ToolError

    async def handler(_context: ToolContext, _arguments: EchoArguments) -> ReturnedError:
        return ReturnedError(
            success=False,
            output="[tool reported an error]",
            error=ToolError(
                error_type="reported_error",
                message="the tool reported failure",
                details={"source": "handler"},
            ),
        )

    tool = Tool.from_arguments_model(
        name="reporting",
        description="Report a structured error.",
        arguments_model=EchoArguments,
        handler=handler,
    )

    results = await execute_tool_calls(
        [ToolCall(call_id="call_reported", tool_name="reporting", arguments={"message": "x"})],
        [tool],
        ToolContext(),
    )

    assert len(results) == 1
    result = results[0]
    assert result.success is False
    assert result.output == "[tool reported an error]"
    assert result.error is not None
    assert result.error.error_type == "reported_error"
    assert result.error.details == {"source": "handler"}


async def test_handler_returned_truncation_metadata_is_preserved() -> None:
    truncation = TruncationInfo(
        truncated=True,
        policy=TruncationPolicy.HEAD,
        original_size_bytes=10,
        original_line_count=1,
        returned_size_bytes=4,
        returned_line_count=1,
        omitted_size_bytes=6,
        omitted_line_count=0,
    )

    class ReturnedTruncated(StrictModel):
        success: bool
        output: str
        truncated: bool
        truncation: TruncationInfo

    async def handler(_context: ToolContext, _arguments: EchoArguments) -> ReturnedTruncated:
        return ReturnedTruncated(
            success=True,
            output="abcd",
            truncated=True,
            truncation=truncation,
        )

    tool = Tool.from_arguments_model(
        name="truncated",
        description="Return truncation metadata.",
        arguments_model=EchoArguments,
        handler=handler,
    )

    results = await execute_tool_calls(
        [ToolCall(call_id="call_truncated", tool_name="truncated", arguments={"message": "x"})],
        [tool],
        ToolContext(),
    )

    assert len(results) == 1
    result = results[0]
    assert result.success is True
    assert result.output == "abcd"
    assert result.truncated is True
    assert result.truncation == truncation


async def test_equal_order_preserves_input_order() -> None:
    async def handler(_context: ToolContext, arguments: EchoArguments) -> str:
        return arguments.message

    tool = Tool.from_arguments_model(
        name="echo",
        description="Echo a message.",
        arguments_model=EchoArguments,
        handler=handler,
    )

    results = await execute_tool_calls(
        [
            ToolCall(call_id="call_a", tool_name="echo", arguments={"message": "a"}, order=1),
            ToolCall(call_id="call_b", tool_name="echo", arguments={"message": "b"}, order=1),
        ],
        [tool],
        ToolContext(),
    )

    assert [result.tool_call_id for result in results] == ["call_a", "call_b"]
    assert [result.output for result in results] == ["a", "b"]
