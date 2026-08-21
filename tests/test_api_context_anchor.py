from __future__ import annotations

from collections.abc import Mapping

from pydantic import Field

from tend import Agent
from tend._common.types import StopReason, StrictModel
from tend.agent.config import RuntimeConfig
from tend.agent.tools import Tool, ToolContext
from tend.llm.config import RetryConfig
from tend.llm.context_estimation import CONTEXT_ESTIMATE_METADATA_KEY
from tend.llm.models import AssistantMessage, TextContent
from tend.llm.models.requests import ModelRequest, ModelResponse
from tend.llm.models.tools import ToolCall
from tend.llm.providers.errors import ProviderRequestError
from tend.llm.retries import RetryErrorCategory
from tend.llm.testing import ScriptedModel
from tend.llm.usage import TokenUsage, Usage


class EchoArguments(StrictModel):
    message: str = Field(min_length=1)


def _echo_tool() -> Tool[EchoArguments]:
    async def handler(_context: ToolContext, arguments: EchoArguments) -> dict[str, str]:
        return {"echo": arguments.message}

    return Tool.from_arguments_model(
        name="echo",
        description="Echo a message.",
        arguments_model=EchoArguments,
        handler=handler,
    )


def _tool_call() -> ToolCall:
    return ToolCall(call_id="call_echo", tool_name="echo", arguments={"message": "hello"})


def _final_response(text: str = "done") -> ModelResponse:
    return ModelResponse(
        response_id=f"model_resp_{text}",
        assistant_message=AssistantMessage(content=[TextContent(text=text)]),
        stop_reason=StopReason.FINAL_RESPONSE,
    )


def _estimator(request: ModelRequest) -> str:
    metadata = request.request_metadata[CONTEXT_ESTIMATE_METADATA_KEY]
    assert isinstance(metadata, Mapping)
    estimator = metadata["estimator"]
    assert isinstance(estimator, str)
    return estimator


def _estimated_tokens(request: ModelRequest) -> int:
    metadata = request.request_metadata[CONTEXT_ESTIMATE_METADATA_KEY]
    assert isinstance(metadata, Mapping)
    tokens = metadata["estimated_tokens"]
    assert isinstance(tokens, int)
    return tokens


async def test_api_usage_anchors_subsequent_context_estimate() -> None:
    model = ScriptedModel(
        [
            ModelResponse(
                response_id="model_resp_tool",
                tool_calls=[_tool_call()],
                usage=Usage(tokens=TokenUsage(input_tokens=100, output_tokens=20)),
            ),
            _final_response(),
        ]
    )
    agent = Agent("System prompt.", model=model, tools=[_echo_tool()])

    await agent.run_turn("Use a tool")

    requests = model.requests
    assert len(requests) == 2
    # No API anchor exists before the first response, so the cold-start request
    # uses the char-based estimator.
    assert _estimator(requests[0]) == "simple_chars"
    # The follow-up anchors on the previous response totals (input 100 + output
    # 20 = 120) plus the appended tool-result delta.
    assert _estimator(requests[1]) == "api_anchor"
    assert _estimated_tokens(requests[1]) >= 120


async def test_absent_provider_usage_does_not_become_a_zero_anchor() -> None:
    model = ScriptedModel(
        [
            # A response that reports no usage (every token count is 0).
            ModelResponse(response_id="model_resp_tool", tool_calls=[_tool_call()]),
            _final_response(),
        ]
    )
    agent = Agent("System prompt.", model=model, tools=[_echo_tool()])

    await agent.run_turn("Use a tool")

    requests = model.requests
    assert len(requests) == 2
    # The follow-up must fall back to the char estimator rather than anchoring on
    # a zero total, which would collapse the estimate to just the new delta.
    assert _estimator(requests[1]) == "simple_chars"


async def test_retry_preserves_post_anchor_message_delta() -> None:
    model = ScriptedModel(
        [
            ModelResponse(
                response_id="model_resp_tool",
                tool_calls=[_tool_call()],
                usage=Usage(tokens=TokenUsage(input_tokens=100, output_tokens=20)),
            ),
            ProviderRequestError(category=RetryErrorCategory.TIMEOUT, message="temporary"),
            _final_response(),
        ]
    )
    agent = Agent("System prompt.", model=model, tools=[_echo_tool()])
    config = RuntimeConfig(
        retries=RetryConfig(
            max_attempts=2,
            initial_delay_seconds=0.0,
            max_delay_seconds=0.0,
            jitter=False,
        )
    )

    await agent.run_turn("Use a tool", config=config)

    requests = model.requests
    assert len(requests) == 3
    # Both the failed follow-up and its retry anchor on the same response and
    # must carry the same post-anchor tool-result delta. Without preservation the
    # retry would collapse to the bare anchor (120).
    assert _estimator(requests[1]) == "api_anchor"
    assert _estimator(requests[2]) == "api_anchor"
    assert _estimated_tokens(requests[2]) == _estimated_tokens(requests[1])
    assert _estimated_tokens(requests[2]) > 120
