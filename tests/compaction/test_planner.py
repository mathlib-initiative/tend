from __future__ import annotations

from tend.agent.compaction import (
    CompactionTriggerReason,
    is_safe_compaction_range,
    plan_compaction,
)
from tend.agent.config import CompactionConfig
from tend.agent.context import assistant_message_from_tool_calls
from tend.llm.context_estimation import TokenEstimatorConfig
from tend.llm.models import (
    AssistantMessage,
    ContextWindow,
    ModelProfile,
    ProviderApi,
    SystemMessage,
    TextContent,
    ToolCall,
    ToolResult,
    ToolResultMessage,
    UserMessage,
)

ESTIMATOR = TokenEstimatorConfig(
    chars_per_token=1000.0,
    tokens_per_message=1,
    tokens_per_content_part=0,
    tokens_per_tool_call=0,
    tokens_per_tool_result=0,
    tokens_per_tool_schema=0,
    tokens_per_reasoning_settings=0,
)


def _system(message_id: str = "msg_system") -> SystemMessage:
    return SystemMessage(message_id=message_id, content=[TextContent(text="system")])


def _user(message_id: str, text: str = "user") -> UserMessage:
    return UserMessage(message_id=message_id, content=[TextContent(text=text)])


def _assistant(message_id: str, text: str = "assistant") -> AssistantMessage:
    return AssistantMessage(message_id=message_id, content=[TextContent(text=text)])


def _tool_call(call_id: str = "call_read") -> ToolCall:
    return ToolCall(call_id=call_id, tool_name="read_file", arguments={"path": "README.md"})


def _tool_result(tool_call: ToolCall, message_id: str = "msg_tool") -> ToolResultMessage:
    result = ToolResult(
        tool_call_id=tool_call.call_id,
        tool_name=tool_call.tool_name,
        arguments=tool_call.arguments,
        success=True,
        output="contents",
    )
    return ToolResultMessage.from_result(result, message_id=message_id)


def _config(**updates: object) -> CompactionConfig:
    values: dict[str, object] = {
        "threshold_tokens": None,
        "threshold_messages": 2,
        "reserve_tokens": 0,
        "keep_recent_tokens": 2,
        "target_tokens": 1,
    }
    values.update(updates)
    return CompactionConfig.model_validate(values)


def test_no_compaction_below_threshold() -> None:
    messages = [_system(), _user("msg_user")]

    plan = plan_compaction(
        messages=messages,
        config=_config(threshold_tokens=10_000, threshold_messages=10),
        estimator_config=ESTIMATOR,
    )

    assert plan.should_compact is False
    assert plan.trigger_reasons == []
    assert plan.skip_reason is None
    assert plan.compact_message_ids == []
    assert plan.preserved_message_ids == ["msg_system", "msg_user"]


def test_anchor_above_token_threshold_triggers_when_char_estimate_is_below() -> None:
    messages = [
        _system(),
        _user("msg_old_user"),
        _assistant("msg_old_assistant"),
        _user("msg_recent_user"),
    ]

    plan = plan_compaction(
        messages=messages,
        config=_config(threshold_tokens=100, threshold_messages=100),
        estimator_config=ESTIMATOR,
        anchor_estimated_tokens=101,
    )

    assert plan.estimated_tokens < 100
    assert plan.anchor_estimated_tokens == 101
    assert plan.char_triggered is False
    assert plan.trigger_reasons == [CompactionTriggerReason.THRESHOLD_TOKENS]
    assert plan.should_compact is True


def test_absent_anchor_preserves_existing_planner_behavior() -> None:
    messages = [_system(), _user("msg_user")]
    config = _config(threshold_tokens=10_000, threshold_messages=10)

    baseline = plan_compaction(
        messages=messages,
        config=config,
        estimator_config=ESTIMATOR,
    )
    explicit_none = plan_compaction(
        messages=messages,
        config=config,
        estimator_config=ESTIMATOR,
        anchor_estimated_tokens=None,
    )

    assert explicit_none.model_dump_json() == baseline.model_dump_json()
    assert explicit_none.anchor_estimated_tokens is None


def test_char_estimate_above_token_threshold_triggers_when_anchor_is_below() -> None:
    messages = [
        _system(),
        _user("msg_old_user"),
        _assistant("msg_old_assistant"),
        _user("msg_recent_user"),
    ]

    plan = plan_compaction(
        messages=messages,
        config=_config(threshold_tokens=10, threshold_messages=100),
        estimator_config=ESTIMATOR,
        anchor_estimated_tokens=1,
    )

    assert plan.estimated_tokens > 10
    assert plan.anchor_estimated_tokens == 1
    assert plan.char_triggered is True
    assert plan.trigger_reasons == [CompactionTriggerReason.THRESHOLD_TOKENS]
    assert plan.should_compact is True


def test_anchor_above_context_limit_triggers_when_char_estimate_is_below() -> None:
    profile = ModelProfile(
        provider_name="scripted_provider",
        model_name="scripted_model",
        api=ProviderApi.OPENAI_RESPONSES,
        context_window=ContextWindow(tokens=100),
    )
    messages = [
        _system(),
        _user("msg_old_user"),
        _assistant("msg_old_assistant"),
        _user("msg_recent_user"),
    ]

    plan = plan_compaction(
        messages=messages,
        config=_config(
            threshold_tokens=1000,
            threshold_messages=100,
            reserve_tokens=10,
        ),
        profile=profile,
        estimator_config=ESTIMATOR,
        anchor_estimated_tokens=91,
    )

    assert plan.estimated_tokens < 90
    assert plan.anchor_estimated_tokens == 91
    assert plan.context_limit_tokens == 90
    assert plan.trigger_reasons == [CompactionTriggerReason.CONTEXT_WINDOW]
    assert plan.should_compact is True


def test_zero_anchor_does_not_trigger_below_threshold() -> None:
    plan = plan_compaction(
        messages=[_system(), _user("msg_user")],
        config=_config(threshold_tokens=100, threshold_messages=100),
        estimator_config=ESTIMATOR,
        anchor_estimated_tokens=0,
    )

    assert plan.anchor_estimated_tokens == 0
    assert plan.char_triggered is False
    assert plan.trigger_reasons == []
    assert plan.should_compact is False


def test_huge_anchor_triggers_without_changing_char_estimate_semantics() -> None:
    messages = [
        _system(),
        _user("msg_old_user"),
        _assistant("msg_old_assistant"),
        _user("msg_recent_user"),
    ]

    plan = plan_compaction(
        messages=messages,
        config=_config(threshold_tokens=100, threshold_messages=100),
        estimator_config=ESTIMATOR,
        anchor_estimated_tokens=10**9,
    )

    assert plan.estimated_tokens < 100
    assert plan.anchor_estimated_tokens == 10**9
    assert plan.char_triggered is False
    assert plan.trigger_reasons == [CompactionTriggerReason.THRESHOLD_TOKENS]
    assert plan.should_compact is True


def test_anchor_only_trigger_with_realistic_keep_budget_records_skipped_plan() -> None:
    plan = plan_compaction(
        messages=[_system(), _user("msg_user", text="tiny")],
        config=_config(
            threshold_tokens=50,
            threshold_messages=100,
            keep_recent_tokens=16_000,
            target_tokens=4_000,
        ),
        estimator_config=ESTIMATOR,
        anchor_estimated_tokens=120,
    )

    assert plan.estimated_tokens < 50
    assert plan.anchor_estimated_tokens == 120
    assert plan.char_triggered is False
    assert plan.trigger_reasons == [CompactionTriggerReason.THRESHOLD_TOKENS]
    assert plan.should_compact is False
    assert plan.skip_reason == "no safe compaction range"


def test_threshold_triggered_cut_point_keeps_recent_suffix() -> None:
    messages = [
        _system(),
        _user("msg_old_user"),
        _assistant("msg_old_assistant"),
        _user("msg_recent_user"),
    ]

    plan = plan_compaction(
        messages=messages,
        config=_config(),
        estimator_config=ESTIMATOR,
    )

    assert plan.should_compact is True
    assert plan.trigger_reasons == [CompactionTriggerReason.THRESHOLD_MESSAGES]
    assert plan.compact_start_index == 1
    assert plan.compact_end_index == 3
    assert plan.compact_message_ids == ["msg_old_user", "msg_old_assistant"]
    assert plan.preserved_message_ids == ["msg_system", "msg_recent_user"]


def test_tool_call_and_result_pair_is_not_split_by_recent_budget() -> None:
    tool_call = _tool_call()
    assistant = assistant_message_from_tool_calls(
        [tool_call],
        message_id="msg_assistant_tool",
        text="I will read the file.",
    )
    tool_result = _tool_result(tool_call, message_id="msg_tool_result")
    messages = [
        _system(),
        _user("msg_old_user"),
        assistant,
        tool_result,
        _user("msg_recent_user"),
    ]

    assert is_safe_compaction_range(messages, 1, 3) is False

    plan = plan_compaction(
        messages=messages,
        config=_config(keep_recent_tokens=6),
        estimator_config=ESTIMATOR,
    )

    assert plan.should_compact is True
    assert plan.compact_message_ids == ["msg_old_user"]
    assert "msg_assistant_tool" in plan.preserved_message_ids
    assert "msg_tool_result" in plan.preserved_message_ids


def test_unresolved_tool_call_is_preserved() -> None:
    tool_call = _tool_call("call_pending")
    assistant = assistant_message_from_tool_calls(
        [tool_call],
        message_id="msg_assistant_pending",
        text="I need this tool.",
    )
    messages = [
        _system(),
        _user("msg_old_user"),
        assistant,
        _user("msg_recent_user"),
    ]

    plan = plan_compaction(
        messages=messages,
        config=_config(),
        estimator_config=ESTIMATOR,
    )

    assert plan.should_compact is True
    assert plan.compact_message_ids == ["msg_old_user"]
    assert "msg_assistant_pending" in plan.preserved_message_ids


def test_split_turn_prefix_plan_for_oversized_latest_turn() -> None:
    messages = [
        _system(),
        _user("msg_current_user"),
        _assistant("msg_current_a1"),
        _assistant("msg_current_a2"),
        _assistant("msg_current_a3"),
    ]

    plan = plan_compaction(
        messages=messages,
        config=_config(threshold_messages=1),
        estimator_config=ESTIMATOR,
    )

    assert plan.should_compact is True
    assert plan.split_turn_prefix is True
    assert plan.compact_message_ids == [
        "msg_current_user",
        "msg_current_a1",
        "msg_current_a2",
    ]
    assert plan.preserved_message_ids == ["msg_system", "msg_current_a3"]


def test_reserve_and_keep_recent_budget_uses_context_window() -> None:
    profile = ModelProfile(
        provider_name="scripted_provider",
        model_name="scripted_model",
        api=ProviderApi.OPENAI_RESPONSES,
        context_window=ContextWindow(tokens=10),
    )
    messages = [
        _system(),
        _user("msg_one"),
        _assistant("msg_two"),
        _user("msg_three"),
    ]

    plan = plan_compaction(
        messages=messages,
        config=_config(
            threshold_tokens=1000,
            threshold_messages=100,
            reserve_tokens=3,
            keep_recent_tokens=100,
            target_tokens=2,
        ),
        profile=profile,
        estimator_config=ESTIMATOR,
    )

    assert plan.context_limit_tokens == 7
    assert plan.effective_threshold_tokens == 7
    assert plan.effective_keep_recent_tokens == 5
    assert plan.trigger_reasons == [CompactionTriggerReason.CONTEXT_WINDOW]
    assert plan.should_compact is True
