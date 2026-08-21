import pytest
from pydantic import TypeAdapter, ValidationError

from tend._common.types import StopReason
from tend.llm.models import (
    AssistantMessage,
    ContinuationStrategy,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ProviderCompletionStatus,
    ProviderItemKind,
    ProviderItemMetadata,
    ProviderMetadata,
    ReasoningContinuationMetadata,
    ReasoningDisplayPolicy,
    ReasoningEffort,
    ReasoningMetadata,
    ReasoningSettings,
    ReasoningSummary,
    ReasoningSummaryPreference,
    TextContent,
    ToolCall,
    ToolError,
    ToolResult,
    ToolResultMessage,
    normalize_tool_arguments,
)
from tend.llm.usage import TokenUsage, Usage

type ModelMessageValue = AssistantMessage | ToolResultMessage


def test_tool_call_arguments_normalize_from_openai_string_and_anthropic_object() -> None:
    openai_call = ToolCall.from_provider_arguments(
        call_id="call_local",
        tool_name="read_file",
        arguments='{"path":"README.md","offset":1}',
        order=0,
        provider_item_id="fc_item_1",
        provider_call_id="call_provider_1",
    )
    anthropic_call = ToolCall.from_provider_arguments(
        call_id="call_local",
        tool_name="read_file",
        arguments={"path": "README.md", "offset": 1},
        order=0,
        provider_tool_use_id="toolu_1",
    )

    assert openai_call.arguments == anthropic_call.arguments == {
        "path": "README.md",
        "offset": 1,
    }
    assert openai_call.provider_item_id == "fc_item_1"
    assert openai_call.provider_call_id == "call_provider_1"
    assert anthropic_call.provider_tool_use_id == "toolu_1"


def test_tool_argument_normalization_rejects_non_objects_and_invalid_json() -> None:
    with pytest.raises(ValueError, match="JSON string is invalid"):
        normalize_tool_arguments("{")

    with pytest.raises(ValueError, match="must be a JSON object"):
        normalize_tool_arguments("[]")


def test_tool_result_success_error_linkage_validation() -> None:
    success = ToolResult(
        tool_call_id="call_1",
        tool_name="ls",
        arguments={"path": "."},
        success=True,
        output="ok",
        order=3,
    )
    message = ToolResultMessage.from_result(success, message_id="msg_tool")

    assert message.tool_call_id == "call_1"
    assert message.tool_name == "ls"
    assert message.content == [TextContent(text="ok")]
    assert message.result == success

    with pytest.raises(ValidationError):
        ToolResult(
            tool_call_id="call_2",
            tool_name="ls",
            success=True,
            error=ToolError(error_type="unexpected", message="should not be present"),
        )

    with pytest.raises(ValidationError):
        ToolResult(tool_call_id="call_3", tool_name="ls", success=False)

    failure = ToolResult(
        tool_call_id="call_4",
        tool_name="grep",
        success=False,
        error=ToolError(error_type="validation", message="pattern is required"),
    )
    failure_message = ToolResultMessage.from_result(failure)

    assert failure_message.content == [
        TextContent(text="Tool error (validation): pattern is required")
    ]

    with pytest.raises(ValidationError):
        ToolResultMessage(
            tool_call_id="different",
            tool_name="grep",
            result=failure,
        )


def test_tool_result_message_round_trips_in_model_request_messages() -> None:
    result = ToolResult(
        tool_call_id="call_1",
        tool_name="read_file",
        arguments={"path": "README.md"},
        success=True,
        output={"content": "hello"},
    )
    tool_message = ToolResultMessage.from_result(result, message_id="msg_tool_1")
    request = ModelRequest(
        request_id="model_req_1",
        messages=[
            AssistantMessage(
                message_id="msg_assistant_1",
                content=[TextContent(text="I will inspect the file.")],
            ),
            tool_message,
        ],
    )

    adapter: TypeAdapter[ModelMessageValue] = TypeAdapter(ModelMessage)
    restored = ModelRequest.model_validate_json(request.model_dump_json())

    assert restored == request
    assert adapter.validate_python(tool_message.model_dump(mode="json")) == tool_message


@pytest.mark.parametrize(
    ("effort", "serialized"),
    [(ReasoningEffort.XHIGH, "xhigh"), (ReasoningEffort.MAX, "max")],
)
def test_reasoning_effort_accepts_extended_levels(
    effort: ReasoningEffort, serialized: str
) -> None:
    settings = ReasoningSettings(effort=effort)

    assert settings.model_dump(mode="json")["effort"] == serialized


def test_reasoning_metadata_serializes_without_becoming_final_text() -> None:
    response = ModelResponse(
        response_id="model_resp_1",
        assistant_message=AssistantMessage(
            message_id="msg_assistant_1",
            content=[TextContent(text="done")],
        ),
        reasoning=ReasoningMetadata(
            requested=ReasoningSettings(
                effort=ReasoningEffort.LOW,
                summary=ReasoningSummaryPreference.AUTO,
                display_policy=ReasoningDisplayPolicy.SUMMARY_ONLY,
            ),
            summaries=[ReasoningSummary(text="Checked constraints and chose a small answer.")],
            reasoning_tokens=7,
            provider_private_continuation=[
                ReasoningContinuationMetadata(
                    provider_name="anthropic",
                    kind="thinking",
                    order=0,
                    signature="sig_test",
                    encrypted_content="encrypted_test",
                )
            ],
        ),
    )

    dumped = response.model_dump(mode="json")

    assert response.final_text == "done"
    assert dumped["assistant_message"]["content"] == [{"kind": "text", "text": "done"}]
    assert dumped["reasoning"]["provider_private_continuation"] == [
        {
            "provider_name": "anthropic",
            "kind": "thinking",
            "order": 0,
            "provider_item_id": None,
            "provider_block_id": None,
            "encrypted_content": "encrypted_test",
            "signature": "sig_test",
            "redacted_details": {},
        }
    ]


def test_model_response_contains_tool_calls_stop_usage_incomplete_and_provider_ids() -> None:
    tool_call = ToolCall.from_provider_arguments(
        call_id="call_1",
        tool_name="bash",
        arguments={"command": "printf ok"},
        order=0,
        provider_item_id="fc_1",
        provider_call_id="call_provider_1",
    )
    response = ModelResponse(
        response_id="model_resp_1",
        request_id="model_req_1",
        assistant_message=AssistantMessage(
            message_id="msg_assistant_1",
            content=[TextContent(text="I need to run a command.")],
        ),
        tool_calls=[tool_call],
        stop_reason=StopReason.PROVIDER_STOP_REASON,
        provider_completion_status=ProviderCompletionStatus.INCOMPLETE,
        incomplete_details={"reason": "max_output_tokens"},
        usage=Usage(tokens=TokenUsage(input_tokens=3, output_tokens=5, reasoning_tokens=2)),
        provider_metadata=ProviderMetadata(
            provider_name="openai",
            model_name="gpt-5",
            response_id="resp_1",
            native_stop_reason="incomplete",
            item_ids=["rs_1", "fc_1"],
            items=[
                ProviderItemMetadata(
                    kind=ProviderItemKind.REASONING,
                    order=0,
                    provider_item_id="rs_1",
                ),
                ProviderItemMetadata(
                    kind=ProviderItemKind.FUNCTION_CALL,
                    order=1,
                    provider_item_id="fc_1",
                    provider_call_id="call_provider_1",
                    tool_name="bash",
                    status="completed",
                ),
            ],
            continuation_strategy=ContinuationStrategy.STATELESS_REPLAY,
            stateless_continuation_required=True,
        ),
    )

    assert response.final_text == "I need to run a command."
    assert response.tool_calls == [tool_call]
    assert response.incomplete_details == {"reason": "max_output_tokens"}
    assert response.usage.tokens.reasoning_tokens == 2
    assert response.provider_metadata is not None
    assert response.provider_metadata.response_id == "resp_1"
    assert response.provider_metadata.items[1].provider_call_id == "call_provider_1"


def test_model_response_rejects_duplicate_tool_call_ids() -> None:
    first = ToolCall(call_id="call_1", tool_name="ls")
    second = ToolCall(call_id="call_1", tool_name="grep")

    with pytest.raises(ValidationError):
        ModelResponse(tool_calls=[first, second])
