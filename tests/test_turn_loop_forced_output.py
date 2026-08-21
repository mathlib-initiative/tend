"""Forced final_result on a terminal turn the model tried to end with prose.

When an agent has a required ``final_result`` output tool and the model finishes with a
plain text response instead of calling it, the turn loop re-asks once with the tool forced
(``request_metadata["force_tool_name"]``) and reasoning dropped — up to a small cap, after
which it falls back to the plain final response.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import Field

from tend import Agent
from tend._common.types import JsonObject, StopReason, StrictModel
from tend.agent.config import CompactionConfig, RuntimeConfig, UsageConfig
from tend.agent.tools.base import Tool
from tend.agent.tools.context import ToolContext
from tend.agent.turn_loop import run_turn
from tend.llm.context_estimation import (
    CONTEXT_ESTIMATE_METADATA_KEY,
    TokenEstimatorConfig,
    estimate_message_tokens,
)
from tend.llm.models import (
    AssistantMessage,
    ModelProfile,
    ModelResponse,
    ProviderApi,
    TextContent,
    ToolCall,
    ToolCapabilities,
    ToolResultMessage,
)
from tend.llm.models.reasoning import ReasoningSettings
from tend.llm.testing import ScriptedModel
from tend.llm.usage import TokenUsage, Usage


class FinalPayload(StrictModel):
    message: str = Field(min_length=1)
    count: int = Field(default=1, ge=1)


class EchoArguments(StrictModel):
    message: str = Field(min_length=1)


def _forcing_profile(*, supports_forced: bool = True) -> ModelProfile:
    return ModelProfile(
        provider_name="scripted",
        model_name="scripted",
        api=ProviderApi.OPENAI_RESPONSES,
        tools=ToolCapabilities(
            supports_tool_calling=True,
            supports_serial_tool_calls=True,
            supports_forced_tool_choice=supports_forced,
        ),
    )


def _prose(text: str, *, usage: Usage | None = None) -> ModelResponse:
    return ModelResponse(
        response_id=f"prose_{text}",
        assistant_message=AssistantMessage(content=[TextContent(text=text)]),
        stop_reason=StopReason.FINAL_RESPONSE,
        usage=usage or Usage(),
    )


def _final_result_call(arguments: JsonObject) -> ModelResponse:
    return ModelResponse(
        response_id="final_result_call",
        tool_calls=[ToolCall(call_id="call_1", tool_name="final_result", arguments=arguments)],
    )


def _output_tools() -> list[Tool[Any]]:
    # Build the agent-scoped final_result output tool via the public Agent API; the throwaway
    # model is never invoked (only its built tools are read).
    return list(Agent("System prompt.", model=ScriptedModel(), output_type=FinalPayload).tools)


def _echo_tool(*, output: str) -> Tool[EchoArguments]:
    async def handler(_context: ToolContext, _arguments: EchoArguments) -> str:
        return output

    return Tool.from_arguments_model(
        name="echo",
        description="Echo a message.",
        arguments_model=EchoArguments,
        handler=handler,
    )


def _request_estimated_tokens(request_index: int, model: ScriptedModel) -> int:
    metadata = model.requests[request_index].request_metadata[CONTEXT_ESTIMATE_METADATA_KEY]
    assert isinstance(metadata, Mapping)
    estimated_tokens = metadata["estimated_tokens"]
    assert isinstance(estimated_tokens, int)
    return estimated_tokens


async def test_forced_reask_recovers_structured_output_and_drops_reasoning() -> None:
    model = ScriptedModel(
        [_prose("Here is my analysis."), _final_result_call({"message": "done", "count": 2})],
        profile=_forcing_profile(),
    )

    result = await run_turn(
        system_prompt="System prompt.",
        model=model,
        prompt="Do the task.",
        tools=_output_tools(),
        reasoning=ReasoningSettings(),
    )

    # The prose turn was converted into a forced final_result, so the turn ends structured.
    assert result.stop_reason is StopReason.FINAL_RESULT
    assert result.final_result is not None
    assert result.final_result.output == {"message": "done", "count": 2}

    assert len(model.requests) == 2
    # First request: normal — reasoning passed through, tool not forced.
    assert "force_tool_name" not in model.requests[0].request_metadata
    assert model.requests[0].reasoning is not None
    # Second (re-ask): final_result forced, reasoning dropped (forced choice is incompatible
    # with extended thinking on some models, where it would otherwise be silently dropped).
    # disable_reasoning is the authoritative signal so adapter default reasoning can't
    # re-enable thinking and silently drop the force.
    assert model.requests[1].request_metadata.get("force_tool_name") == "final_result"
    assert model.requests[1].reasoning is None
    assert model.requests[1].request_metadata.get("disable_reasoning") is True


async def test_forced_reask_refreshes_post_anchor_delta_without_stale_tool_result() -> None:
    estimator = TokenEstimatorConfig(
        chars_per_token=10.0,
        tokens_per_message=0,
        tokens_per_content_part=0,
        tokens_per_tool_call=0,
        tokens_per_tool_result=0,
        tokens_per_tool_schema=0,
        tokens_per_reasoning_settings=0,
    )
    threshold_tokens = 250
    model = ScriptedModel(
        [
            ModelResponse(
                response_id="tool_response",
                tool_calls=[
                    ToolCall(
                        call_id="call_echo",
                        tool_name="echo",
                        arguments={"message": "hello"},
                    )
                ],
                usage=Usage(tokens=TokenUsage(input_tokens=20, output_tokens=5)),
            ),
            _prose(
                "Finished in prose.",
                usage=Usage(tokens=TokenUsage(input_tokens=200, output_tokens=10)),
            ),
            _final_result_call({"message": "done", "count": 2}),
        ],
        profile=_forcing_profile(),
    )
    config = RuntimeConfig(
        compaction=CompactionConfig(
            threshold_tokens=threshold_tokens,
            threshold_messages=100,
            reserve_tokens=0,
            keep_recent_tokens=1,
            target_tokens=1,
        ),
        usage=UsageConfig(token_estimator=estimator),
    )

    result = await run_turn(
        system_prompt="System prompt.",
        model=model,
        prompt="Do the task.",
        tools=[_echo_tool(output="x" * 1000), *_output_tools()],
        config=config,
    )

    assert result.stop_reason is StopReason.FINAL_RESULT
    assert len(model.requests) == 3
    assert all(
        request.request_metadata.get("purpose") != "generic_compaction"
        for request in model.requests
    )

    forced_request = model.requests[2]
    nudge_message = forced_request.messages[-1]
    correct_estimate = 210 + estimate_message_tokens(nudge_message, estimator)
    assert _request_estimated_tokens(2, model) == correct_estimate

    stale_tool_result = next(
        message
        for message in model.requests[1].messages
        if isinstance(message, ToolResultMessage)
    )
    stale_estimate = 210 + estimate_message_tokens(stale_tool_result, estimator)
    assert correct_estimate < threshold_tokens < stale_estimate


async def test_forced_reask_respects_cap_and_falls_back_to_final_response() -> None:
    model = ScriptedModel(
        [_prose("first"), _prose("second"), _prose("third")],
        profile=_forcing_profile(),
    )

    result = await run_turn(
        system_prompt="System prompt.",
        model=model,
        prompt="Do the task.",
        tools=_output_tools(),
    )

    # The model keeps narrating; after the re-ask cap (2) the loop accepts the prose.
    assert result.stop_reason is StopReason.FINAL_RESPONSE
    assert result.final_response == "third"
    assert len(model.requests) == 3  # 1 initial + 2 forced re-asks
    assert "force_tool_name" not in model.requests[0].request_metadata
    assert model.requests[1].request_metadata.get("force_tool_name") == "final_result"
    assert model.requests[2].request_metadata.get("force_tool_name") == "final_result"


async def test_no_forced_reask_when_model_lacks_forced_tool_choice() -> None:
    model = ScriptedModel(
        [_prose("just prose")],
        profile=_forcing_profile(supports_forced=False),
    )

    result = await run_turn(
        system_prompt="System prompt.",
        model=model,
        prompt="Do the task.",
        tools=_output_tools(),
    )

    # No forced-tool-choice support → original behavior: accept the prose immediately.
    assert result.stop_reason is StopReason.FINAL_RESPONSE
    assert result.final_response == "just prose"
    assert len(model.requests) == 1
    assert "force_tool_name" not in model.requests[0].request_metadata
