from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import pytest
from pydantic import BaseModel, Field, ValidationError

from tend import Agent, FinalResultOutput
from tend._common.types import JsonObject, StopReason, StrictModel
from tend.agent.config import AgentConfig, RuntimeConfig, RuntimeLimitsConfig
from tend.agent.context import assistant_tool_calls
from tend.agent.outputs import ReviewVerdictOutput
from tend.agent.persistence.events import (
    EventType,
    ToolCallCompletedEvent,
    ToolCallStartedEvent,
    TurnCompletedEvent,
)
from tend.agent.session import Session
from tend.agent.tools import Tool, ToolContext
from tend.llm.models import (
    AssistantMessage,
    ModelResponse,
    TextContent,
    ToolCall,
    ToolResultMessage,
)
from tend.llm.testing import ScriptedModel
from tend.llm.truncation import TruncationInfo, TruncationPolicy
from tend.llm.usage import TokenUsage, Usage


class EchoArguments(StrictModel):
    message: str = Field(min_length=1)


class FinalPayload(StrictModel):
    message: str = Field(min_length=1)
    count: int = Field(default=1, ge=1)


class LooseChild(BaseModel):
    value: str


class LoosePayload(BaseModel):
    message: str
    child: LooseChild


class TruncatedToolOutput(StrictModel):
    success: bool
    output: str
    truncated: bool
    truncation: TruncationInfo


def _final_response(text: str, *, response_id: str = "model_resp_final") -> ModelResponse:
    return ModelResponse(
        response_id=response_id,
        assistant_message=AssistantMessage(content=[TextContent(text=text)]),
        stop_reason=StopReason.FINAL_RESPONSE,
    )


def _echo_tool(seen: list[str] | None = None) -> Tool[EchoArguments]:
    async def handler(_context: ToolContext, arguments: EchoArguments) -> dict[str, str]:
        if seen is not None:
            seen.append(arguments.message)
        return {"echo": arguments.message}

    return Tool.from_arguments_model(
        name="echo",
        description="Echo a message.",
        arguments_model=EchoArguments,
        handler=handler,
    )


def test_final_result_tool_name_is_reserved_for_agent_output() -> None:
    async def handler(_context: ToolContext, arguments: EchoArguments) -> str:
        return arguments.message

    tool = Tool.from_arguments_model(
        name="final_result",
        description="Reserved name.",
        arguments_model=EchoArguments,
        handler=handler,
    )

    with pytest.raises(ValueError, match="reserved"):
        Agent("System prompt.", model=ScriptedModel(), tools=[tool])


async def test_one_tool_call_followed_by_final_response_persists_lifecycle(
    tmp_path: Path,
) -> None:
    tool_call = ToolCall(
        call_id="call_echo",
        tool_name="echo",
        arguments={"message": "hello"},
        order=0,
        provider_item_id="item_echo",
        provider_call_id="provider_call_echo",
    )
    model = ScriptedModel(
        [
            ModelResponse(
                response_id="model_resp_tools",
                tool_calls=[tool_call],
                usage=Usage(tokens=TokenUsage(input_tokens=5, output_tokens=2)),
            ),
            _final_response("done", response_id="model_resp_done"),
        ]
    )
    agent = Agent("System prompt.", model=model, tools=[_echo_tool()], model_name="scripted")

    with Session.create(tmp_path, session_id="sess_tools", sync_writes=False) as session:
        result = await agent.run_turn("Use the tool", session=session)
        events = session.event_store.read_all()

    assert result.final_response == "done"
    assert result.stop_reason is StopReason.FINAL_RESPONSE
    assert result.stop is None
    assert [call.call_id for call in result.tool_calls] == ["call_echo"]
    assert len(result.tool_results) == 1
    assert result.tool_results[0].success is True
    assert result.tool_results[0].output == {"echo": "hello"}
    assert result.tool_results[0].provider_call_id == "provider_call_echo"
    assert result.model_request_count == 2
    assert result.tool_call_count == 1
    assert result.usage.model_requests == 2
    assert result.usage.tool_calls == 1
    assert result.usage.tokens.input_tokens == 5
    assert result.session_state is not None
    assert set(result.session_state.completed_tool_calls) == {"call_echo"}

    requests = model.requests
    assert len(requests) == 2
    follow_up = requests[1]
    assert [message.role.value for message in follow_up.messages] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    assistant = follow_up.messages[2]
    assert isinstance(assistant, AssistantMessage)
    assert assistant_tool_calls(assistant) == (tool_call,)
    tool_message = follow_up.messages[3]
    assert isinstance(tool_message, ToolResultMessage)
    assert tool_message.result.output == {"echo": "hello"}
    assert tool_message.content == [TextContent(text='{"echo":"hello"}')]
    assert len(follow_up.tools) == 1
    assert follow_up.tools[0]["name"] == "echo"

    assert [event.event_type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.TURN_STARTED,
        EventType.MODEL_REQUEST_STARTED,
        EventType.MODEL_RESPONSE_COMPLETED,
        EventType.TOOL_CALL_STARTED,
        EventType.TOOL_CALL_COMPLETED,
        EventType.MODEL_REQUEST_STARTED,
        EventType.MODEL_RESPONSE_COMPLETED,
        EventType.TURN_COMPLETED,
    ]
    assert [event.sequence for event in events] == list(range(len(events)))
    assert isinstance(events[4], ToolCallStartedEvent)
    assert events[4].payload.tool_call == tool_call
    assert isinstance(events[5], ToolCallCompletedEvent)
    assert events[5].payload.result.tool_call_id == "call_echo"
    assert isinstance(events[-1], TurnCompletedEvent)
    assert events[-1].payload.model_request_count == 2
    assert events[-1].payload.tool_call_count == 1


async def test_truncated_tool_result_persists_and_reaches_follow_up_request(
    tmp_path: Path,
) -> None:
    tool_call = ToolCall(
        call_id="call_truncated",
        tool_name="truncated_echo",
        arguments={"message": "large output"},
        order=0,
    )

    async def handler(_context: ToolContext, _arguments: EchoArguments) -> TruncatedToolOutput:
        return TruncatedToolOutput(
            success=True,
            output="line 1\n[Output truncated]",
            truncated=True,
            truncation=TruncationInfo(
                truncated=True,
                policy=TruncationPolicy.HEAD,
                original_size_bytes=100,
                original_line_count=10,
                returned_size_bytes=24,
                returned_line_count=2,
                omitted_size_bytes=76,
                omitted_line_count=8,
            ),
        )

    tool = Tool.from_arguments_model(
        name="truncated_echo",
        description="Return a truncated response.",
        arguments_model=EchoArguments,
        handler=handler,
    )
    model = ScriptedModel(
        [
            ModelResponse(response_id="model_resp_truncated_tool", tool_calls=[tool_call]),
            _final_response("done", response_id="model_resp_done"),
        ]
    )
    agent = Agent("System prompt.", model=model, tools=[tool], model_name="scripted")

    with Session.create(tmp_path, session_id="sess_truncated", sync_writes=False) as session:
        result = await agent.run_turn("Use the tool", session=session)
        events = session.event_store.read_all()
        session_state = session.state

    assert result.final_response == "done"
    assert len(result.tool_results) == 1
    tool_result = result.tool_results[0]
    assert tool_result.truncated is True
    assert tool_result.truncation is not None
    assert tool_result.truncation.policy is TruncationPolicy.HEAD

    completed_events = [event for event in events if isinstance(event, ToolCallCompletedEvent)]
    assert len(completed_events) == 1
    completed_result = completed_events[0].payload.result
    assert completed_result.truncated is True
    assert completed_result.truncation is not None
    assert completed_result.truncation.policy is TruncationPolicy.HEAD

    follow_up_messages = model.requests[1].messages
    tool_messages = [
        message for message in follow_up_messages if isinstance(message, ToolResultMessage)
    ]
    assert len(tool_messages) == 1
    assert tool_messages[0].result == tool_result
    assert tool_messages[0].tool_call_id == "call_truncated"
    assert set(session_state.completed_tool_calls) == {"call_truncated"}


async def test_two_tool_calls_preserve_provider_order_through_turn_loop() -> None:
    seen_messages: list[str] = []
    call_second = ToolCall(
        call_id="call_second",
        tool_name="echo",
        arguments={"message": "second"},
        order=2,
    )
    call_first = ToolCall(
        call_id="call_first",
        tool_name="echo",
        arguments={"message": "first"},
        order=1,
    )
    model = ScriptedModel(
        [
            ModelResponse(response_id="model_resp_tools", tool_calls=[call_second, call_first]),
            _final_response("ordered"),
        ]
    )
    agent = Agent("System prompt.", model=model, tools=[_echo_tool(seen_messages)])

    result = await agent.run_turn("Call both tools")

    assert seen_messages == ["first", "second"]
    assert [result.tool_call_id for result in result.tool_results] == [
        "call_first",
        "call_second",
    ]
    assert [result.order for result in result.tool_results] == [1, 2]
    follow_up = model.requests[1]
    assert [
        message.tool_call_id
        for message in follow_up.messages
        if isinstance(message, ToolResultMessage)
    ] == ["call_first", "call_second"]


async def test_tool_validation_failure_is_sent_back_to_model() -> None:
    handler_called = False

    async def handler(_context: ToolContext, _arguments: EchoArguments) -> str:
        nonlocal handler_called
        handler_called = True
        return "not called"

    tool = Tool.from_arguments_model(
        name="echo",
        description="Echo a message.",
        arguments_model=EchoArguments,
        handler=handler,
    )
    model = ScriptedModel(
        [
            ModelResponse(
                response_id="model_resp_bad_args",
                tool_calls=[ToolCall(call_id="call_bad_args", tool_name="echo", arguments={})],
            ),
            _final_response("recovered"),
        ]
    )
    agent = Agent("System prompt.", model=model, tools=[tool])

    result = await agent.run_turn("Call with invalid args")

    assert handler_called is False
    assert result.final_response == "recovered"
    assert len(result.tool_results) == 1
    tool_result = result.tool_results[0]
    assert tool_result.success is False
    assert tool_result.error is not None
    assert tool_result.error.error_type == "validation_error"

    follow_up_result = model.requests[1].messages[-1]
    assert isinstance(follow_up_result, ToolResultMessage)
    assert follow_up_result.result == tool_result
    assert follow_up_result.content[0] == TextContent(
        text="Tool error (validation_error): Arguments for tool 'echo' failed validation."
    )


async def test_tool_handler_exception_is_sent_back_to_model() -> None:
    async def handler(_context: ToolContext, _arguments: EchoArguments) -> str:
        raise RuntimeError("boom")

    tool = Tool.from_arguments_model(
        name="echo",
        description="Echo a message.",
        arguments_model=EchoArguments,
        handler=handler,
    )
    model = ScriptedModel(
        [
            ModelResponse(
                response_id="model_resp_exception",
                tool_calls=[
                    ToolCall(
                        call_id="call_exception",
                        tool_name="echo",
                        arguments={"message": "raise"},
                    )
                ],
            ),
            _final_response("handled"),
        ]
    )
    agent = Agent("System prompt.", model=model, tools=[tool])

    result = await agent.run_turn("Call a failing tool")

    assert result.final_response == "handled"
    assert len(result.tool_results) == 1
    tool_result = result.tool_results[0]
    assert tool_result.success is False
    assert tool_result.error is not None
    assert tool_result.error.error_type == "handler_exception"
    assert "boom" in tool_result.error.message

    follow_up_result = model.requests[1].messages[-1]
    assert isinstance(follow_up_result, ToolResultMessage)
    assert follow_up_result.result == tool_result
    assert follow_up_result.content[0] == TextContent(
        text="Tool error (handler_exception): RuntimeError: boom"
    )


async def test_ordinary_tool_followed_by_completed_empty_response_is_final() -> None:
    tool_call = ToolCall(
        call_id="call_echo",
        tool_name="echo",
        arguments={"message": "hello"},
    )
    model = ScriptedModel(
        [
            ModelResponse(response_id="model_resp_tools", tool_calls=[tool_call]),
            ModelResponse(response_id="model_resp_empty"),
        ]
    )
    agent = Agent("System prompt.", model=model, tools=[_echo_tool()])

    result = await agent.run_turn("Use the tool once")

    assert result.final_response == ""
    assert result.stop_reason is StopReason.FINAL_RESPONSE
    assert result.stop is None
    assert result.model_request_count == 2
    assert result.tool_call_count == 1


async def test_agent_from_config_resolves_output_schema_name() -> None:
    model = ScriptedModel([_final_response("done")])
    config = AgentConfig.model_validate_json(
        json.dumps(
            {
                "system_prompt": "System prompt.",
                "model": {
                    "provider": "openai",
                    "api": "openai_responses",
                    "model_name": "gpt-5",
                },
                "output": {"schema_name": "review_verdict"},
            }
        )
    )
    agent = Agent.from_config(config, model=model)

    await agent.run_turn("Capture request tools")

    request = model.requests[0]
    assert [tool["name"] for tool in request.tools] == ["final_result"]
    assert request.tools[0]["arguments_schema"] == ReviewVerdictOutput.model_json_schema()


async def test_output_type_exposes_agent_scoped_final_result_schema() -> None:
    model = ScriptedModel([_final_response("done")])
    agent = Agent("System prompt.", model=model, output_type=FinalPayload)

    result = await agent.run_turn("Finish with text")

    assert result.final_response == "done"
    request = model.requests[0]
    assert [tool["name"] for tool in request.tools] == ["final_result"]
    final_result_schema = request.tools[0]["arguments_schema"]
    assert final_result_schema == FinalPayload.model_json_schema()
    assert agent.tools[0].definition.metadata == {
        "tend_tool_kind": "output",
        "terminates_turn": True,
    }


async def test_output_type_accepts_base_model_and_normalizes_schema_recursively() -> None:
    model = ScriptedModel([_final_response("done")])
    agent = Agent("System prompt.", model=model, output_type=LoosePayload)

    result = await agent.run_turn("Finish with text")

    assert result.final_response == "done"
    schema = cast(JsonObject, model.requests[0].tools[0]["arguments_schema"])
    assert schema["type"] == "object"
    assert schema["required"] == ["message", "child"]
    assert schema["additionalProperties"] is False
    defs = cast(JsonObject, schema["$defs"])
    child_schema = cast(JsonObject, defs["LooseChild"])
    assert child_schema["type"] == "object"
    assert child_schema["additionalProperties"] is False


async def test_final_result_base_model_validation_forbids_nested_extra_fields() -> None:
    model = ScriptedModel(
        [
            ModelResponse(
                response_id="model_resp_nested_extra",
                tool_calls=[
                    ToolCall(
                        call_id="call_bad_nested_extra",
                        tool_name="final_result",
                        arguments={
                            "message": "ok",
                            "extra_root": "rejected",
                            "child": {"value": "nested", "extra_child": "rejected"},
                        },
                    )
                ],
            ),
            _final_response("recovered"),
        ]
    )
    agent = Agent("System prompt.", model=model, output_type=LoosePayload)

    result = await agent.run_turn("Return extra fields")

    assert result.final_response == "recovered"
    assert len(result.tool_results) == 1
    tool_result = result.tool_results[0]
    assert tool_result.success is False
    assert tool_result.error is not None
    assert tool_result.error.error_type == "validation_error"
    validation_errors = tool_result.error.details["validation_errors"]
    assert isinstance(validation_errors, list)
    locations: set[tuple[object, ...]] = set()
    for error in validation_errors:
        if not isinstance(error, Mapping):
            continue
        location = error.get("loc")
        if isinstance(location, Sequence) and not isinstance(location, str | bytes):
            locations.add(tuple(location))
    assert ("extra_root",) in locations
    assert ("child", "extra_child") in locations


async def test_final_result_output_requires_final_result_tool_name() -> None:
    with pytest.raises(ValidationError):
        FinalResultOutput.model_validate(
            {
                "tool_name": "not_final_result",
                "tool_call_id": "call_final",
                "output": {"message": "ok"},
                "arguments": {"message": "ok"},
            }
        )


async def test_final_result_call_terminates_without_follow_up_request() -> None:
    final_call = ToolCall(
        call_id="call_final",
        tool_name="final_result",
        arguments={"message": "ok", "count": 2},
    )
    model = ScriptedModel(
        [
            ModelResponse(response_id="model_resp_final_tool", tool_calls=[final_call]),
            _final_response("should not be consumed"),
        ]
    )
    agent = Agent("System prompt.", model=model, output_type=FinalPayload)

    result = await agent.run_turn("Return a structured result")

    assert result.final_response is None
    assert result.final_result is not None
    assert result.final_result.tool_call_id == "call_final"
    assert result.final_result.output == {"message": "ok", "count": 2}
    assert result.final_result.arguments == {"message": "ok", "count": 2}
    assert result.stop_reason is StopReason.FINAL_RESULT
    assert result.stop is None
    assert result.model_request_count == 1
    assert result.tool_call_count == 1
    assert len(result.tool_results) == 1
    assert result.tool_results[0].success is True
    assert model.remaining_steps == 1


async def test_first_valid_final_result_wins_before_later_output_calls_and_limits() -> None:
    first_final_call = ToolCall(
        call_id="call_final_first",
        tool_name="final_result",
        arguments={"message": "first"},
        order=0,
    )
    second_final_call = ToolCall(
        call_id="call_final_second",
        tool_name="final_result",
        arguments={"message": "second"},
        order=1,
    )
    model = ScriptedModel(
        [
            ModelResponse(
                response_id="model_resp_multiple_final_tools",
                tool_calls=[first_final_call, second_final_call],
            )
        ]
    )
    agent = Agent("System prompt.", model=model, output_type=FinalPayload)
    config = RuntimeConfig(limits=RuntimeLimitsConfig(max_tool_calls=1))

    result = await agent.run_turn("Return a structured result", config=config)

    assert result.stop_reason is StopReason.FINAL_RESULT
    assert result.final_result is not None
    assert result.final_result.tool_call_id == "call_final_first"
    assert result.final_result.output == {"message": "first", "count": 1}
    assert result.tool_call_count == 1
    assert [tool_result.tool_call_id for tool_result in result.tool_results] == [
        "call_final_first"
    ]


async def test_final_result_call_skips_ordinary_tool_calls_in_same_response() -> None:
    seen_messages: list[str] = []
    final_call = ToolCall(
        call_id="call_final",
        tool_name="final_result",
        arguments={"message": "done"},
        order=1,
    )
    ordinary_call = ToolCall(
        call_id="call_echo",
        tool_name="echo",
        arguments={"message": "should not run"},
        order=0,
    )
    model = ScriptedModel(
        [ModelResponse(response_id="model_resp_mixed", tool_calls=[ordinary_call, final_call])]
    )
    agent = Agent(
        "System prompt.",
        model=model,
        tools=[_echo_tool(seen_messages)],
        output_type=FinalPayload,
    )

    result = await agent.run_turn("Return final result")

    assert result.stop_reason is StopReason.FINAL_RESULT
    assert result.final_result is not None
    assert result.final_result.output == {"message": "done", "count": 1}
    assert seen_messages == []
    assert [call.call_id for call in result.tool_calls] == ["call_echo", "call_final"]
    assert [tool_result.tool_call_id for tool_result in result.tool_results] == ["call_final"]


async def test_invalid_final_result_arguments_are_sent_back_to_model() -> None:
    model = ScriptedModel(
        [
            ModelResponse(
                response_id="model_resp_bad_final_result",
                tool_calls=[
                    ToolCall(call_id="call_bad_final", tool_name="final_result", arguments={})
                ],
            ),
            _final_response("recovered"),
        ]
    )
    agent = Agent("System prompt.", model=model, output_type=FinalPayload)

    result = await agent.run_turn("Return bad then recover")

    assert result.final_response == "recovered"
    assert result.stop_reason is StopReason.FINAL_RESPONSE
    assert len(result.tool_results) == 1
    tool_result = result.tool_results[0]
    assert tool_result.tool_name == "final_result"
    assert tool_result.success is False
    assert tool_result.error is not None
    assert tool_result.error.error_type == "validation_error"
    follow_up_result = model.requests[1].messages[-1]
    assert isinstance(follow_up_result, ToolResultMessage)
    assert follow_up_result.result == tool_result


async def test_invalid_final_result_and_ordinary_tool_results_follow_provider_order() -> None:
    ordinary_call = ToolCall(
        call_id="call_echo",
        tool_name="echo",
        arguments={"message": "first"},
        order=0,
    )
    final_call = ToolCall(
        call_id="call_bad_final",
        tool_name="final_result",
        arguments={},
        order=1,
    )
    model = ScriptedModel(
        [
            ModelResponse(
                response_id="model_resp_bad_final_result_mixed",
                tool_calls=[ordinary_call, final_call],
            ),
            _final_response("recovered"),
        ]
    )
    agent = Agent("System prompt.", model=model, tools=[_echo_tool()], output_type=FinalPayload)

    result = await agent.run_turn("Return bad final result with an ordinary tool")

    assert result.final_response == "recovered"
    assert [tool_result.tool_call_id for tool_result in result.tool_results] == [
        "call_echo",
        "call_bad_final",
    ]
    follow_up_results = [
        message for message in model.requests[1].messages if isinstance(message, ToolResultMessage)
    ]
    assert [message.tool_call_id for message in follow_up_results] == [
        "call_echo",
        "call_bad_final",
    ]


async def test_final_result_without_output_type_is_unknown_tool_path() -> None:
    model = ScriptedModel(
        [
            ModelResponse(
                response_id="model_resp_unknown_final_result",
                tool_calls=[
                    ToolCall(
                        call_id="call_unknown_final",
                        tool_name="final_result",
                        arguments={"message": "ok"},
                    )
                ],
            ),
            _final_response("recovered"),
        ]
    )
    agent = Agent("System prompt.", model=model)

    result = await agent.run_turn("Try final_result without output type")

    assert result.final_response == "recovered"
    assert model.requests[0].tools == []
    assert len(result.tool_results) == 1
    tool_result = result.tool_results[0]
    assert tool_result.success is False
    assert tool_result.error is not None
    assert tool_result.error.error_type == "unknown_tool"


async def test_max_iteration_placeholder_stops_after_tool_result_without_next_request() -> None:
    tool_call = ToolCall(
        call_id="call_echo",
        tool_name="echo",
        arguments={"message": "hello"},
    )
    model = ScriptedModel(
        [
            ModelResponse(response_id="model_resp_tools", tool_calls=[tool_call]),
            _final_response("should not be consumed"),
        ]
    )
    agent = Agent("System prompt.", model=model, tools=[_echo_tool()])
    config = RuntimeConfig(limits=RuntimeLimitsConfig(max_iterations=1))

    result = await agent.run_turn("Use the tool once", config=config)

    assert result.final_response is None
    assert result.stop_reason is StopReason.MAX_ITERATIONS
    assert result.stop is not None
    assert result.stop.reason is StopReason.MAX_ITERATIONS
    assert result.model_request_count == 1
    assert result.tool_call_count == 1
    assert len(result.tool_results) == 1
    assert model.remaining_steps == 1
