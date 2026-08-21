from __future__ import annotations

import pytest

from tend.agent.context import (
    ASSISTANT_PROVIDER_METADATA_KEY,
    ActiveCompactionSummary,
    assistant_message_from_response,
    assistant_message_from_tool_calls,
    assistant_tool_calls,
    build_active_context,
    validate_context_messages,
)
from tend.agent.persistence.state import InterruptedToolCall, SessionState
from tend.llm.models import (
    AssistantMessage,
    CompactionSummaryContent,
    MessageRole,
    ModelResponse,
    ProviderMetadata,
    TextContent,
    ToolCall,
    ToolError,
    ToolResult,
    ToolResultMessage,
)


def test_initial_context_contains_system_prompt_and_new_user_prompt() -> None:
    context = build_active_context(
        system_prompt="You are concise.",
        new_user_prompt="Hello.",
        input_message_id="msg_user_1",
    )

    assert context.input_message_id == "msg_user_1"
    assert [message.role for message in context.messages] == [
        MessageRole.SYSTEM,
        MessageRole.USER,
    ]
    assert context.messages[0].content == [TextContent(text="You are concise.")]
    assert context.messages[1].content == [TextContent(text="Hello.")]
    assert context.tool_pairing.tool_call_ids == []


def test_context_preserves_assistant_tool_call_and_linked_tool_result() -> None:
    tool_call = ToolCall(
        call_id="call_local_1",
        tool_name="read_file",
        arguments={"path": "README.md"},
        order=0,
        provider_item_id="fc_item_1",
        provider_call_id="call_provider_1",
        provider_tool_use_id="toolu_1",
        provider_metadata={"native_status": "completed"},
    )
    assistant = assistant_message_from_response(
        ModelResponse(
            response_id="model_resp_1",
            assistant_message=AssistantMessage(
                message_id="msg_assistant_1",
                content=[TextContent(text="I will read that file.")],
            ),
            tool_calls=[tool_call],
            provider_metadata=ProviderMetadata(
                provider_name="openai_responses",
                response_id="resp_provider_1",
                item_ids=["fc_item_1"],
            ),
        )
    )
    result = ToolResult(
        tool_call_id="call_local_1",
        tool_name="read_file",
        arguments={"path": "README.md"},
        success=True,
        output="contents",
        order=0,
        provider_item_id="fc_item_1",
        provider_call_id="call_provider_1",
        provider_tool_use_id="toolu_1",
    )
    tool_result_message = ToolResultMessage.from_result(result, message_id="msg_tool_1")

    context = build_active_context(
        system_prompt="System instructions.",
        tail_messages=[assistant, tool_result_message],
        new_user_prompt="Continue.",
    )

    assistant_in_context = context.messages[1]
    assert isinstance(assistant_in_context, AssistantMessage)
    preserved_call = assistant_tool_calls(assistant_in_context)[0]
    assert preserved_call == tool_call
    assert preserved_call.provider_item_id == "fc_item_1"
    assert preserved_call.provider_call_id == "call_provider_1"
    assert preserved_call.provider_tool_use_id == "toolu_1"
    assert assistant_in_context.provider_metadata[ASSISTANT_PROVIDER_METADATA_KEY] == {
        "provider_name": "openai_responses",
        "model_name": None,
        "response_id": "resp_provider_1",
        "previous_response_id": None,
        "native_stop_reason": None,
        "item_ids": ["fc_item_1"],
        "items": [],
        "continuation_strategy": "stateless_replay",
        "provider_side_continuation_available": None,
        "stateless_continuation_required": False,
        "redacted_raw_details": {},
        "artifact_reference_ids": [],
    }

    result_in_context = context.messages[2]
    assert isinstance(result_in_context, ToolResultMessage)
    assert result_in_context.result.provider_call_id == "call_provider_1"
    assert result_in_context.result.provider_tool_use_id == "toolu_1"
    assert context.tool_pairing.tool_call_ids == ["call_local_1"]
    assert context.tool_pairing.tool_result_ids == ["call_local_1"]
    assert context.unresolved_tool_call_ids == ()


def test_context_includes_compaction_summary_before_uncompacted_tail() -> None:
    tail = AssistantMessage(
        message_id="msg_recent_assistant",
        content=[TextContent(text="Recent answer kept verbatim.")],
    )

    context = build_active_context(
        system_prompt="System instructions.",
        compaction_summaries=[
            ActiveCompactionSummary(
                message_id="msg_summary_1",
                summary="Goal, progress, and next steps summarized.",
                covered_message_ids=["msg_old_1", "msg_old_2"],
            )
        ],
        tail_messages=[tail],
        new_user_prompt="Next task.",
    )

    assert [message.role for message in context.messages] == [
        MessageRole.SYSTEM,
        MessageRole.ASSISTANT,
        MessageRole.ASSISTANT,
        MessageRole.USER,
    ]
    summary_message = context.messages[1]
    assert isinstance(summary_message, AssistantMessage)
    assert summary_message.message_id == "msg_summary_1"
    assert summary_message.content == [
        CompactionSummaryContent(
            summary="Goal, progress, and next steps summarized.",
            covered_message_ids=["msg_old_1", "msg_old_2"],
        )
    ]
    assert context.messages[2] == tail


def test_unresolved_assistant_tool_calls_are_preserved_and_reported() -> None:
    tool_call = ToolCall(
        call_id="call_pending",
        tool_name="bash",
        arguments={"command": "printf ok"},
        provider_call_id="call_provider_pending",
    )
    assistant = assistant_message_from_tool_calls(
        [tool_call],
        text="I need to run a command.",
        message_id="msg_assistant_pending",
    )

    context = build_active_context(
        system_prompt="System instructions.",
        tail_messages=[assistant],
        new_user_prompt="Continue after pending work.",
    )

    assistant_in_context = context.messages[1]
    assert isinstance(assistant_in_context, AssistantMessage)
    assert assistant_tool_calls(assistant_in_context) == (tool_call,)
    assert context.unresolved_tool_call_ids == ("call_pending",)

    with pytest.raises(ValueError, match="without tool results"):
        validate_context_messages(context.messages, allow_unresolved_tool_calls=False)


def test_interrupted_tool_calls_from_session_state_become_linked_tool_results() -> None:
    tool_call = ToolCall(
        call_id="call_interrupted",
        tool_name="grep",
        arguments={"pattern": "needle", "path": "."},
        order=2,
        provider_tool_use_id="toolu_interrupted",
    )
    interrupted_result = ToolResult(
        tool_call_id="call_interrupted",
        tool_name="grep",
        arguments=tool_call.arguments,
        success=False,
        output="[Tool interrupted before completion: grep]",
        error=ToolError(error_type="interrupted", message="interrupted before completion"),
        order=2,
        provider_tool_use_id="toolu_interrupted",
    )
    state = SessionState(
        session_id="sess_1",
        interrupted_tool_calls={
            "call_interrupted": InterruptedToolCall(
                tool_call_id="call_interrupted",
                tool_name="grep",
                started_event_id="evt_tool_started",
                turn_id="turn_1",
                order=2,
                tool_call=tool_call,
                result=interrupted_result,
            )
        },
    )
    assistant = assistant_message_from_tool_calls(
        [tool_call],
        message_id="msg_assistant_interrupted",
    )

    context = build_active_context(
        system_prompt="System instructions.",
        session_state=state,
        tail_messages=[assistant],
        new_user_prompt="Recover.",
    )

    assert [type(message) for message in context.messages] == [
        type(context.messages[0]),
        AssistantMessage,
        ToolResultMessage,
        type(context.messages[3]),
    ]
    inserted_result = context.messages[2]
    assert isinstance(inserted_result, ToolResultMessage)
    assert inserted_result.tool_call_id == "call_interrupted"
    assert inserted_result.result.success is False
    assert inserted_result.result.provider_tool_use_id == "toolu_interrupted"
    assert context.unresolved_tool_call_ids == ()
