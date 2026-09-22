from __future__ import annotations

import pytest
from pydantic import Field

from tend import Agent
from tend._common.types import StrictModel
from tend.agent.compaction import is_safe_compaction_range, plan_compaction
from tend.agent.config import CompactionConfig, RuntimeConfig
from tend.agent.tools import Tool, ToolContext
from tend.llm.context_estimation import (
    RequestTokenEstimate,
    RequestTokenEstimator,
    TokenEstimatorConfig,
    estimate_request_tokens,
)
from tend.llm.history import assistant_message_from_response, assistant_message_from_tool_calls
from tend.llm.models import (
    AssistantMessage,
    CompactionSummaryContent,
    ModelProfile,
    ModelRequest,
    ModelResponse,
    ReasoningContinuationMetadata,
    ReasoningMetadata,
    SystemMessage,
    TextContent,
    ToolCall,
    ToolResult,
    ToolResultMessage,
    UserMessage,
)
from tend.llm.providers import AnthropicMessagesAdapter, OpenAIResponsesAdapter
from tend.llm.usage import TokenUsage, Usage


class CustomEstimator:
    """An adapter capability with no dependency on either built-in provider."""

    def estimate_request_tokens(
        self,
        request: ModelRequest,
        config: TokenEstimatorConfig,
    ) -> RequestTokenEstimate:
        estimate = estimate_request_tokens(request, config=config)
        for index, message in enumerate(request.messages):
            retained = message.provider_metadata.get("custom_retained_tokens", 0)
            assert isinstance(retained, int)
            estimate.message_tokens[index] += retained
        return estimate


@pytest.mark.parametrize("adapter_estimates", [False, True])
def test_underestimated_history_compacts_enough_to_clear_trigger(adapter_estimates: bool) -> None:
    messages = [SystemMessage(content=[TextContent(text="system")])] + [
        AssistantMessage(
            message_id=f"message_{i}",
            content=[TextContent(text="x" * 1700)],
            provider_metadata={"custom_retained_tokens": 500},
        )
        for i in range(161)
    ]
    estimator = CustomEstimator() if adapter_estimates else None
    config = CompactionConfig(
        threshold_tokens=250_000,
        keep_recent_tokens=120_000,
        target_tokens=40_000,
    )
    plan = plan_compaction(
        messages=messages,
        config=config,
        anchor_estimated_tokens=303_496,
        token_estimator=estimator,
    )
    assert plan.should_compact
    assert not plan.char_triggered
    assert plan.token_scale_factor > 1
    assert plan.projected_tokens is not None and plan.projected_tokens <= 225_000
    assert plan.compact_end_index is not None and plan.compact_end_index > 50
    assert plan.recent_token_estimate * plan.token_scale_factor <= config.keep_recent_tokens

    # Simulate the next provider response reporting the projected prompt size,
    # then several ordinary messages. Compaction must not fire again immediately.
    after = [
        messages[0],
        AssistantMessage(
            content=[
                CompactionSummaryContent(
                    summary="Summary.",
                    covered_message_ids=plan.compact_message_ids,
                )
            ]
        ),
        *messages[plan.compact_end_index :],
    ]
    for growth in range(1, 6):
        after.append(AssistantMessage(content=[TextContent(text="next step")]))
        next_plan = plan_compaction(
            messages=after,
            config=config,
            token_estimator=estimator,
            anchor_estimated_tokens=plan.projected_tokens + growth * 500,
        )
        assert not next_plan.should_compact
        assert not next_plan.trigger_reasons


def test_fixed_costs_and_summary_overhead_can_make_compaction_ineffective() -> None:
    class ExpensiveTools(CustomEstimator):
        def estimate_request_tokens(
            self,
            request: ModelRequest,
            config: TokenEstimatorConfig,
        ) -> RequestTokenEstimate:
            estimate = super().estimate_request_tokens(request, config)
            estimate.tool_schema_tokens = 10_000
            return estimate

    plan = plan_compaction(
        messages=[UserMessage(content=[TextContent(text="x" * 1000)]) for _ in range(10)],
        config=CompactionConfig(
            threshold_tokens=10_000, keep_recent_tokens=2000, target_tokens=1000
        ),
        token_estimator=ExpensiveTools(),
    )
    assert not plan.should_compact
    assert plan.skip_reason == "insufficient token reduction"
    assert plan.projected_tokens is not None and plan.projected_tokens > 10_000


def test_zero_estimates_do_not_lose_an_api_anchor() -> None:
    class ZeroEstimator:
        def estimate_request_tokens(
            self,
            request: ModelRequest,
            config: TokenEstimatorConfig,
        ) -> RequestTokenEstimate:
            return RequestTokenEstimate(message_tokens=[0] * len(request.messages))

    plan = plan_compaction(
        messages=[UserMessage(content=[TextContent(text="message")]) for _ in range(4)],
        config=CompactionConfig(threshold_tokens=100, keep_recent_tokens=20, target_tokens=10),
        token_estimator=ZeroEstimator(),
        anchor_estimated_tokens=200,
    )
    assert not plan.should_compact
    assert plan.skip_reason == "insufficient token reduction"
    assert plan.projected_tokens == 210


def test_summary_covered_ids_are_included_in_projection() -> None:
    plan = plan_compaction(
        messages=[
            UserMessage(
                message_id=f"{i}" + "id" * 100,
                content=[TextContent(text="x" * 200)],
            )
            for i in range(4)
        ],
        config=CompactionConfig(threshold_tokens=400, keep_recent_tokens=200, target_tokens=50),
    )
    assert not plan.should_compact
    assert plan.skip_reason == "insufficient token reduction"
    assert plan.projected_tokens is not None and plan.projected_tokens > 400


def test_headroom_search_can_move_past_a_completed_tool_pair() -> None:
    call = ToolCall(call_id="call", tool_name="tool", arguments={"text": "x" * 2000})
    messages = [
        UserMessage(content=[TextContent(text="x" * 2000)]),
        assistant_message_from_tool_calls([call]),
        ToolResultMessage.from_result(
            ToolResult(
                tool_call_id="call",
                tool_name="tool",
                arguments=call.arguments,
                success=True,
                output="x" * 2000,
            )
        ),
        UserMessage(content=[TextContent(text="recent")]),
    ]
    plan = plan_compaction(
        messages=messages,
        config=CompactionConfig(threshold_tokens=2000, keep_recent_tokens=1400, target_tokens=500),
    )
    assert plan.should_compact
    assert plan.compact_end_index == 3
    assert is_safe_compaction_range(messages, 0, plan.compact_end_index)
    assert plan.projected_tokens is not None and plan.projected_tokens <= 1800


def test_forced_overflow_keeps_recovery_even_when_normal_projection_cannot_fit() -> None:
    call = ToolCall(call_id="pending", tool_name="tool", arguments={})
    messages = [
        UserMessage(content=[TextContent(text="old")]),
        assistant_message_from_tool_calls(
            [call], provider_metadata={"custom_retained_tokens": 1000}
        ),
    ]
    config = CompactionConfig(threshold_tokens=100, keep_recent_tokens=50, target_tokens=20)
    normal = plan_compaction(messages=messages, config=config, token_estimator=CustomEstimator())
    assert normal.skip_reason == "insufficient token reduction"
    forced = plan_compaction(
        messages=messages,
        config=config,
        token_estimator=CustomEstimator(),
        force_context_overflow=True,
    )
    assert forced.should_compact
    assert forced.compact_message_ids == [messages[0].message_id]


class EchoArguments(StrictModel):
    text: str = Field(min_length=1)


class MeteredModel:
    """Bills twice the local estimate and keeps working across compactions."""

    def __init__(self, estimator: RequestTokenEstimator) -> None:
        self.estimator = estimator
        self.requests: list[ModelRequest] = []
        self.normal_requests = 0
        self.prompt_tokens: list[int] = []

    @property
    def profile(self) -> ModelProfile | None:
        return None

    def estimate_request_tokens(
        self,
        request: ModelRequest,
        config: TokenEstimatorConfig,
    ) -> RequestTokenEstimate:
        return self.estimator.estimate_request_tokens(request, config)

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request.model_copy(deep=True))
        prompt_tokens = (
            2 * estimate_request_tokens(request, token_estimator=self).parts.total_tokens
        )
        self.prompt_tokens.append(prompt_tokens)
        if request.request_metadata.get("purpose") == "generic_compaction":
            response = ModelResponse(
                assistant_message=AssistantMessage(content=[TextContent(text="Summary.")]),
            )
        else:
            self.normal_requests += 1
            if self.normal_requests == 12:
                response = ModelResponse(
                    assistant_message=AssistantMessage(content=[TextContent(text="done")]),
                )
            else:
                response = ModelResponse(
                    assistant_message=AssistantMessage(
                        provider_metadata={"custom_retained_tokens": 1000},
                    ),
                    tool_calls=[
                        ToolCall(
                            call_id=f"call_{self.normal_requests}",
                            tool_name="echo",
                            arguments={"text": "x" * 1000},
                        )
                    ],
                    reasoning=ReasoningMetadata(
                        reasoning_tokens=1000,
                        provider_private_continuation=[
                            ReasoningContinuationMetadata(
                                provider_name="anthropic",
                                kind="thinking",
                                signature="opaque",
                                redacted_details={"thinking": "retained"},
                            )
                        ],
                    ),
                )
        output_tokens = (
            2
            * estimate_request_tokens(
                ModelRequest(messages=[assistant_message_from_response(response)]),
                token_estimator=self,
            ).parts.message_tokens
        )
        response.usage = Usage(
            tokens=TokenUsage(
                input_tokens=prompt_tokens,
                output_tokens=output_tokens,
            )
        )
        return response


@pytest.mark.parametrize(
    "estimator",
    [
        CustomEstimator(),
        AnthropicMessagesAdapter(model_name="test"),
        OpenAIResponsesAdapter(model_name="test"),
    ],
)
async def test_turn_loop_does_not_recompact_every_request(estimator: RequestTokenEstimator) -> None:
    async def echo(_context: ToolContext, arguments: EchoArguments) -> str:
        return arguments.text

    model = MeteredModel(estimator)
    agent = Agent(
        "System.",
        model=model,
        tools=[
            Tool.from_arguments_model(
                name="echo",
                description="Echo text.",
                arguments_model=EchoArguments,
                handler=echo,
            )
        ],
    )
    result = await agent.run_turn(
        "x" * 20_000,
        config=RuntimeConfig(
            compaction=CompactionConfig(
                threshold_tokens=25_000,
                keep_recent_tokens=6000,
                target_tokens=2000,
            )
        ),
    )
    assert result.final_response == "done"
    compacted = [
        index
        for index, request in enumerate(model.requests)
        if request.request_metadata.get("purpose") == "generic_compaction"
    ]
    assert compacted
    assert len(compacted) < 4
    for index in compacted:
        assert model.prompt_tokens[index + 1] < 22_500
        # At least two ordinary requests follow each compaction without another.
        assert all(
            request.request_metadata.get("purpose") != "generic_compaction"
            for request in model.requests[index + 1 : index + 3]
        )
