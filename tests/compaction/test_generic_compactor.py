from __future__ import annotations

from decimal import Decimal

import pytest

from tend._common.types import StopReason
from tend.agent.compaction import (
    COMPACTION_PROMPT_VERSION,
    SUMMARY_SECTION_HEADINGS,
    CompactionError,
    CompactionPlan,
    GenericSummarizationCompactor,
    build_compaction_request,
    plan_compaction,
)
from tend.agent.config import CompactionConfig
from tend.llm.context_estimation import TokenEstimatorConfig
from tend.llm.models import (
    AssistantMessage,
    CompactionSummaryContent,
    ModelProfile,
    ProviderApi,
    TextContent,
    UserMessage,
)
from tend.llm.models.profiles import TokenPricing
from tend.llm.models.requests import ModelMessage, ModelResponse
from tend.llm.testing import ScriptedModel
from tend.llm.usage import Cost, TokenUsage, Usage

ESTIMATOR = TokenEstimatorConfig(
    chars_per_token=1000.0,
    tokens_per_message=1,
    tokens_per_content_part=0,
    tokens_per_tool_call=0,
    tokens_per_tool_result=0,
    tokens_per_tool_schema=0,
    tokens_per_reasoning_settings=0,
)


def _user(message_id: str, text: str) -> UserMessage:
    return UserMessage(message_id=message_id, content=[TextContent(text=text)])


def _assistant(message_id: str, text: str) -> AssistantMessage:
    return AssistantMessage(message_id=message_id, content=[TextContent(text=text)])


def _messages() -> list[ModelMessage]:
    return [
        _user("msg_old_goal", "Implement resumable sessions."),
        _assistant("msg_old_progress", "Added append-only events and state snapshots."),
        _user("msg_recent", "Continue with compaction."),
    ]


def _config() -> CompactionConfig:
    return CompactionConfig(
        threshold_messages=1,
        reserve_tokens=0,
        keep_recent_tokens=1,
        target_tokens=1,
    )


def _plan(messages: list[ModelMessage]) -> CompactionPlan:
    return plan_compaction(
        messages=messages,
        config=_config(),
        estimator_config=ESTIMATOR,
    )


def _final_response(text: str, *, usage: Usage | None = None) -> ModelResponse:
    return ModelResponse(
        response_id="model_resp_compaction",
        assistant_message=AssistantMessage(content=[TextContent(text=text)]),
        stop_reason=StopReason.FINAL_RESPONSE,
        usage=usage or Usage(),
    )


def test_compaction_request_construction_uses_planned_range_only() -> None:
    messages = _messages()
    plan = _plan(messages)

    request = build_compaction_request(
        messages=messages,
        plan=plan,
        compaction_id="compact_test",
        model_name="summary-model",
    )

    assert request.model_name == "summary-model"
    assert request.tools == []
    assert request.max_output_tokens == 1
    assert request.request_metadata["purpose"] == "generic_compaction"
    assert request.request_metadata["prompt_version"] == COMPACTION_PROMPT_VERSION
    assert request.request_metadata["compaction_id"] == "compact_test"
    assert request.request_metadata["covered_message_ids"] == [
        "msg_old_goal",
        "msg_old_progress",
    ]

    prompt = request.messages[1].content[0]
    assert isinstance(prompt, TextContent)
    assert "Implement resumable sessions." in prompt.text
    assert "Added append-only events" in prompt.text
    assert "Continue with compaction." not in prompt.text
    for heading in SUMMARY_SECTION_HEADINGS:
        assert f"## {heading}" in prompt.text


async def test_non_empty_summary_is_accepted_and_does_not_mutate_history() -> None:
    messages = _messages()
    original_dump = [message.model_dump(mode="json") for message in messages]
    plan = _plan(messages)
    model = ScriptedModel([_final_response("## Goal\nKeep the agent working.")])
    compactor = GenericSummarizationCompactor(model)

    result = await compactor.compact(
        messages=messages,
        plan=plan,
        compaction_id="compact_accept",
    )

    assert result.compaction_id == "compact_accept"
    assert result.summary == "## Goal\nKeep the agent working."
    assert result.covered_message_ids == ["msg_old_goal", "msg_old_progress"]
    assert result.preserved_message_ids == ["msg_recent"]
    assert [message.model_dump(mode="json") for message in messages] == original_dump
    assert len(model.requests) == 1


async def test_empty_summary_fails_with_compaction_error() -> None:
    messages = _messages()
    model = ScriptedModel([_final_response("  \n\t")])

    with pytest.raises(CompactionError, match="non-empty"):
        await GenericSummarizationCompactor(model).compact(
            messages=messages,
            plan=_plan(messages),
        )


async def test_usage_and_profile_cost_metadata_are_preserved() -> None:
    messages = _messages()
    usage = Usage(
        tokens=TokenUsage(
            input_tokens=1_000_000,
            output_tokens=2_000_000,
            reasoning_tokens=3_000_000,
            provider_details={"provider_side": 5},
        )
    )
    pricing = TokenPricing(
        input_per_million_tokens=Decimal("1.00"),
        output_per_million_tokens=Decimal("2.00"),
        reasoning_per_million_tokens=Decimal("3.00"),
        source="test-prices",
    )
    profile = ModelProfile(
        provider_name="scripted",
        model_name="summary-model",
        api=ProviderApi.OPENAI_RESPONSES,
        pricing=pricing,
    )
    model = ScriptedModel([_final_response("Summary text.", usage=usage)], profile=profile)

    result = await GenericSummarizationCompactor(model).compact(
        messages=messages,
        plan=_plan(messages),
    )

    assert result.usage.tokens == usage.tokens
    assert result.usage.model_requests == 1
    assert result.usage.cost == Cost(
        amount=Decimal("14.00"),
        currency="USD",
        pricing_source="test-prices",
    )


async def test_summary_content_becomes_compaction_summary_message() -> None:
    messages = _messages()
    model = ScriptedModel([_final_response("Structured summary.")])

    result = await GenericSummarizationCompactor(model).compact(
        messages=messages,
        plan=_plan(messages),
        compaction_id="compact_message",
    )

    message = result.summary_message
    assert isinstance(message, AssistantMessage)
    assert message.provider_metadata["compaction_id"] == "compact_message"
    assert len(message.content) == 1
    part = message.content[0]
    assert isinstance(part, CompactionSummaryContent)
    assert part.summary == "Structured summary."
    assert part.covered_message_ids == ["msg_old_goal", "msg_old_progress"]
