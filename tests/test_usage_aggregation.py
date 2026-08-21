from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from pydantic import Field

from tend import Agent
from tend._common.types import StopReason, StrictModel
from tend.agent.config import RuntimeConfig, UsageConfig
from tend.agent.session import Session
from tend.agent.tools import Tool, ToolContext
from tend.llm.context_estimation import CONTEXT_ESTIMATE_METADATA_KEY, TokenEstimatorConfig
from tend.llm.models import (
    AssistantMessage,
    ContextWindow,
    ModelProfile,
    ProviderApi,
    TextContent,
)
from tend.llm.models.profiles import TokenPricing
from tend.llm.models.requests import ModelResponse
from tend.llm.models.tools import ToolCall
from tend.llm.testing import ScriptedModel
from tend.llm.usage import Cost, TokenUsage, Usage


class EchoArguments(StrictModel):
    message: str = Field(min_length=1)


def _profile(
    *,
    context_window_tokens: int | None = None,
    pricing: TokenPricing | None = None,
) -> ModelProfile:
    return ModelProfile(
        provider_name="scripted_provider",
        model_name="scripted_model",
        api=ProviderApi.OPENAI_RESPONSES,
        context_window=ContextWindow(tokens=context_window_tokens)
        if context_window_tokens is not None
        else None,
        pricing=pricing,
    )


def _echo_tool() -> Tool[EchoArguments]:
    async def handler(_context: ToolContext, arguments: EchoArguments) -> str:
        return arguments.message

    return Tool.from_arguments_model(
        name="echo",
        description="Echo a message.",
        arguments_model=EchoArguments,
        handler=handler,
    )


def _final_response(text: str, *, usage: Usage | None = None) -> ModelResponse:
    return ModelResponse(
        response_id=f"model_resp_{text}",
        assistant_message=AssistantMessage(content=[TextContent(text=text)]),
        stop_reason=StopReason.FINAL_RESPONSE,
        usage=usage or Usage(),
    )


async def test_turn_and_session_usage_aggregate_multiple_model_responses(
    tmp_path: Path,
) -> None:
    first_usage = Usage(
        tokens=TokenUsage(
            input_tokens=5,
            output_tokens=2,
            reasoning_tokens=3,
            cache_read_tokens=7,
            cache_write_tokens=11,
            provider_details={"audio_tokens": 13, "z_tokens": 17},
        )
    )
    second_usage = Usage(
        tokens=TokenUsage(
            input_tokens=19,
            output_tokens=23,
            reasoning_tokens=29,
            cache_read_tokens=31,
            cache_write_tokens=37,
            provider_details={"audio_tokens": 41, "other_tokens": 43},
        )
    )
    model = ScriptedModel(
        [
            ModelResponse(
                response_id="model_resp_tool",
                tool_calls=[
                    ToolCall(
                        call_id="call_echo",
                        tool_name="echo",
                        arguments={"message": "hello"},
                    )
                ],
                usage=first_usage,
            ),
            _final_response("done", usage=second_usage),
        ]
    )
    agent = Agent("System prompt.", model=model, tools=[_echo_tool()])

    with Session.create(tmp_path, session_id="sess_usage", sync_writes=False) as session:
        result = await agent.run_turn("Use a tool", session=session)
        state = session.state

    expected_usage = Usage(
        tokens=TokenUsage(
            input_tokens=24,
            output_tokens=25,
            reasoning_tokens=32,
            cache_read_tokens=38,
            cache_write_tokens=48,
            provider_details={"audio_tokens": 54, "other_tokens": 43, "z_tokens": 17},
        ),
        model_requests=2,
        tool_calls=1,
    )
    assert result.usage == expected_usage
    assert state.usage == expected_usage
    assert state.turn_usage == {result.turn_id: expected_usage}
    assert set(state.model_request_usage) == {request.request_id for request in model.requests}


async def test_missing_provider_usage_does_not_crash_or_fabricate_cost() -> None:
    model = ScriptedModel([_final_response("ok")], profile=_profile())
    agent = Agent("System prompt.", model=model)

    result = await agent.run_turn("No usage reported")

    assert result.final_response == "ok"
    assert result.usage == Usage(model_requests=1)
    assert result.usage.cost is None


async def test_profile_pricing_calculates_cost_only_when_configured() -> None:
    pricing = TokenPricing(
        input_per_million_tokens=Decimal("1.00"),
        output_per_million_tokens=Decimal("2.00"),
        reasoning_per_million_tokens=Decimal("3.00"),
        cache_read_per_million_tokens=Decimal("0.10"),
        cache_write_per_million_tokens=Decimal("0.20"),
        source="test-price-card",
    )
    reported_usage = Usage(
        tokens=TokenUsage(
            input_tokens=1_000_000,
            output_tokens=2_000_000,
            reasoning_tokens=1_000_000,
            cache_read_tokens=10_000_000,
            cache_write_tokens=10_000_000,
        )
    )
    model = ScriptedModel(
        [_final_response("priced", usage=reported_usage)],
        profile=_profile(pricing=pricing),
    )
    agent = Agent("System prompt.", model=model)

    result = await agent.run_turn("Price this")

    assert result.usage.cost == Cost(
        amount=Decimal("11.00"),
        currency="USD",
        pricing_source="test-price-card",
    )


async def test_absent_pricing_leaves_cost_unknown() -> None:
    reported_usage = Usage(tokens=TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000))
    model = ScriptedModel([_final_response("unpriced", usage=reported_usage)], profile=_profile())
    agent = Agent("System prompt.", model=model)

    result = await agent.run_turn("Do not price this")

    assert result.usage.tokens.input_tokens == 1_000_000
    assert result.usage.cost is None


async def test_context_estimate_is_exposed_in_result_request_and_session_state(
    tmp_path: Path,
) -> None:
    model = ScriptedModel(
        [_final_response("estimated")],
        profile=_profile(context_window_tokens=100),
    )
    agent = Agent("System prompt.", model=model)
    config = RuntimeConfig(
        usage=UsageConfig(token_estimator=TokenEstimatorConfig(chars_per_token=10.0))
    )

    with Session.create(tmp_path, session_id="sess_context", sync_writes=False) as session:
        result = await agent.run_turn(
            "Estimate the current context",
            session=session,
            config=config,
        )
        state = session.state

    estimate = result.context_estimate
    assert estimate is not None
    assert estimate.estimated_tokens > 0
    assert estimate.context_window_tokens == 100
    assert estimate.context_usage_ratio is not None
    assert estimate.context_usage_percent is not None
    assert abs(estimate.context_usage_ratio - (estimate.estimated_tokens / 100)) < 1e-12
    assert abs(estimate.context_usage_percent - estimate.estimated_tokens) < 1e-12
    assert estimate.remaining_context_tokens == max(100 - estimate.estimated_tokens, 0)

    request = model.last_request
    assert request is not None
    assert CONTEXT_ESTIMATE_METADATA_KEY in request.request_metadata
    assert state.latest_context_estimate == estimate
    assert state.model_request_context_estimates == {request.request_id: estimate}


async def test_context_estimation_can_be_disabled() -> None:
    model = ScriptedModel(
        [_final_response("no estimate")],
        profile=_profile(context_window_tokens=100),
    )
    agent = Agent("System prompt.", model=model)
    config = RuntimeConfig(usage=UsageConfig(estimate_context_tokens=False))

    result = await agent.run_turn("Skip estimate", config=config)

    assert result.context_estimate is None
    request = model.last_request
    assert request is not None
    assert CONTEXT_ESTIMATE_METADATA_KEY not in request.request_metadata
