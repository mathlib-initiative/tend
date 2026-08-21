"""Long-context compaction boundary."""

from tend.agent.compaction.generic import (
    CompactionError,
    GenericCompactionResult,
    GenericSummarizationCompactor,
    apply_compaction_result,
    build_compaction_request,
    compact_messages,
)
from tend.agent.compaction.planner import (
    CompactionPlan,
    CompactionTriggerReason,
    find_safe_compaction_end,
    initial_recent_start,
    is_safe_compaction_range,
    latest_user_message_index,
    leading_instruction_end,
    plan_compaction,
)
from tend.agent.compaction.prompts import (
    COMPACTION_PROMPT_VERSION,
    COMPACTION_SYSTEM_PROMPT,
    SUMMARY_SECTION_HEADINGS,
    compacted_messages_from_plan,
    render_compaction_user_prompt,
    render_message_transcript,
)

__all__ = (
    "COMPACTION_PROMPT_VERSION",
    "COMPACTION_SYSTEM_PROMPT",
    "SUMMARY_SECTION_HEADINGS",
    "CompactionError",
    "CompactionPlan",
    "CompactionTriggerReason",
    "GenericCompactionResult",
    "GenericSummarizationCompactor",
    "apply_compaction_result",
    "build_compaction_request",
    "compact_messages",
    "compacted_messages_from_plan",
    "find_safe_compaction_end",
    "initial_recent_start",
    "is_safe_compaction_range",
    "latest_user_message_index",
    "leading_instruction_end",
    "plan_compaction",
    "render_compaction_user_prompt",
    "render_message_transcript",
)
