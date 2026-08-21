from __future__ import annotations

from pathlib import Path

from pydantic import Field

from tend import Agent
from tend._common.types import StopReason, StrictModel
from tend.agent.config import CompactionConfig, RuntimeConfig, UsageConfig
from tend.agent.context import (
    ActiveCompactionSummary,
    assistant_tool_calls,
    build_active_context,
)
from tend.agent.persistence.events import CompactionCompletedEvent, ModelRequestStartedEvent
from tend.agent.session import Session
from tend.agent.tools import Tool, ToolContext
from tend.llm.context_estimation import TokenEstimatorConfig
from tend.llm.models import (
    AssistantMessage,
    CompactionSummaryContent,
    ModelResponse,
    TextContent,
    ToolCall,
    ToolResultMessage,
)
from tend.llm.testing import ScriptedModel
from tend.llm.usage import TokenUsage, Usage

ESTIMATOR = TokenEstimatorConfig(
    chars_per_token=1000.0,
    tokens_per_message=1,
    tokens_per_content_part=0,
    tokens_per_tool_call=0,
    tokens_per_tool_result=0,
    tokens_per_tool_schema=0,
    tokens_per_reasoning_settings=0,
)


class EchoArguments(StrictModel):
    message: str = Field(min_length=1)


def _echo_tool() -> Tool[EchoArguments]:
    async def handler(_context: ToolContext, arguments: EchoArguments) -> str:
        return arguments.message

    return Tool.from_arguments_model(
        name="echo",
        description="Echo a message for compaction resume tests.",
        arguments_model=EchoArguments,
        handler=handler,
    )


def _tool_call() -> ToolCall:
    return ToolCall(
        call_id="call_echo",
        tool_name="echo",
        arguments={"message": "hello"},
        order=0,
    )


def _tool_response() -> ModelResponse:
    return ModelResponse(response_id="model_resp_tool", tool_calls=[_tool_call()])


def _text_response(text: str, *, response_id: str, usage: Usage | None = None) -> ModelResponse:
    return ModelResponse(
        response_id=response_id,
        assistant_message=AssistantMessage(content=[TextContent(text=text)]),
        stop_reason=StopReason.FINAL_RESPONSE,
        usage=usage or Usage(),
    )


def _compacting_config() -> RuntimeConfig:
    return RuntimeConfig(
        compaction=CompactionConfig(
            threshold_messages=2,
            reserve_tokens=0,
            keep_recent_tokens=1,
            target_tokens=1,
        ),
        usage=UsageConfig(token_estimator=ESTIMATOR),
    )


async def test_compaction_state_survives_reload_and_reconstructs_active_context(
    tmp_path: Path,
) -> None:
    session_dir = tmp_path / "session"
    compaction_usage = Usage(tokens=TokenUsage(input_tokens=7, output_tokens=3))
    model = ScriptedModel(
        [
            _tool_response(),
            _text_response(
                "## Goal\nContinue with the echoed result preserved.",
                response_id="model_resp_summary",
                usage=compaction_usage,
            ),
            _text_response("done", response_id="model_resp_done"),
        ]
    )
    agent = Agent("System prompt.", model=model, tools=[_echo_tool()], model_name="scripted")

    with Session.create(
        session_dir,
        session_id="sess_compaction_resume",
        sync_writes=False,
    ) as session:
        result = await agent.run_turn(
            "Use the tool and keep going.",
            session=session,
            config=_compacting_config(),
        )

    assert result.final_response == "done"

    with Session.open(session_dir, writable=False, sync_writes=False) as resumed:
        state = resumed.state
        events = resumed.event_store.read_all()

    assert len(state.completed_compactions) == 1
    completed = next(iter(state.completed_compactions.values()))
    assert completed.summary.startswith("## Goal")
    assert completed.usage.tokens == compaction_usage.tokens

    compaction_events = [event for event in events if isinstance(event, CompactionCompletedEvent)]
    assert len(compaction_events) == 1
    assert compaction_events[0].payload.compaction_id == completed.compaction_id
    assert compaction_events[0].payload.covered_message_ids == completed.covered_message_ids

    request_events = [event for event in events if isinstance(event, ModelRequestStartedEvent)]
    final_request = request_events[-1].payload.request
    assert final_request is not None
    persisted_summary = final_request.messages[1]
    assert isinstance(persisted_summary, AssistantMessage)
    assert isinstance(persisted_summary.content[0], CompactionSummaryContent)
    assert persisted_summary.content[0].summary == completed.summary

    tail_messages = final_request.messages[2:]
    context = build_active_context(
        system_prompt="System prompt.",
        compaction_summaries=[
            ActiveCompactionSummary(
                message_id=completed.summary_message_id,
                summary=completed.summary,
                covered_message_ids=completed.covered_message_ids,
            )
        ],
        tail_messages=tail_messages,
        new_user_prompt="Resume with the next task.",
    )

    assert [message.role.value for message in context.messages] == [
        "system",
        "assistant",
        "assistant",
        "tool",
        "user",
    ]
    reconstructed_summary = context.messages[1]
    assert isinstance(reconstructed_summary, AssistantMessage)
    summary_part = reconstructed_summary.content[0]
    assert isinstance(summary_part, CompactionSummaryContent)
    assert summary_part.summary == completed.summary
    assert summary_part.covered_message_ids == completed.covered_message_ids

    tail_assistant = context.messages[2]
    assert isinstance(tail_assistant, AssistantMessage)
    assert [call.call_id for call in assistant_tool_calls(tail_assistant)] == ["call_echo"]
    tail_tool_result = context.messages[3]
    assert isinstance(tail_tool_result, ToolResultMessage)
    assert tail_tool_result.tool_call_id == "call_echo"
    assert context.tool_pairing.tool_call_ids == ["call_echo"]
    assert context.tool_pairing.tool_result_ids == ["call_echo"]
    assert context.unresolved_tool_call_ids == ()
