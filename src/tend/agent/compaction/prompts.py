"""Structured generic compaction prompts and transcript rendering."""

from __future__ import annotations

import json
from collections.abc import Sequence

from tend.agent.compaction.planner import CompactionPlan
from tend.agent.context import assistant_tool_calls
from tend.llm.models.messages import AssistantMessage, TextContent
from tend.llm.models.requests import ModelMessage
from tend.llm.models.tools import ToolResultMessage, model_visible_tool_result_text

COMPACTION_PROMPT_VERSION = "generic_summarization_v1"

COMPACTION_SYSTEM_PROMPT = """You are a context compaction assistant for a long-horizon
coding agent. Your job is to summarize older conversation history so the agent
can continue working safely without manual intervention. Preserve concrete facts,
decisions, file paths, tool outcomes, constraints, blockers, and next actions.
Do not invent facts. Do not expose hidden chain-of-thought; summarize only
observable context. Return only the requested summary in Markdown."""

SUMMARY_SECTION_HEADINGS: tuple[str, ...] = (
    "Goal",
    "Constraints / Preferences",
    "Completed Work",
    "In-Progress Work",
    "Blockers",
    "Key Decisions",
    "Next Steps",
    "Critical Context",
    "Important Read / Modified Files",
)


def render_compaction_user_prompt(
    *,
    messages: Sequence[ModelMessage],
    plan: CompactionPlan,
) -> str:
    """Render the user prompt for summarizing the planned compacted range."""

    compacted_messages = compacted_messages_from_plan(messages=messages, plan=plan)
    covered_ids = [message.message_id for message in compacted_messages]
    headings = "\n".join(f"## {heading}" for heading in SUMMARY_SECTION_HEADINGS)
    transcript = render_message_transcript(compacted_messages)
    return (
        "Summarize the compacted conversation range below for later active context.\n\n"
        "Compaction metadata:\n"
        f"- Prompt version: {COMPACTION_PROMPT_VERSION}\n"
        f"- Compact start index: {plan.compact_start_index}\n"
        f"- Compact end index: {plan.compact_end_index}\n"
        f"- Target summary tokens: {plan.target_tokens}\n"
        f"- Covered message IDs: {', '.join(covered_ids)}\n"
        f"- Split-turn prefix compaction: {_yes_no(plan.split_turn_prefix)}\n\n"
        "Output requirements:\n"
        "- Use the exact Markdown section headings listed below.\n"
        "- Keep the summary concise but sufficient for autonomous continuation.\n"
        "- Preserve file paths, commands, tool-call results, errors, and important IDs "
        "when useful.\n"
        "- If a section has no known information, write \"None known.\"\n"
        "- Do not include the raw transcript verbatim unless a small excerpt is critical.\n\n"
        "Required summary structure:\n"
        f"{headings}\n\n"
        "Compacted transcript:\n"
        f"{transcript}"
    )


def compacted_messages_from_plan(
    *,
    messages: Sequence[ModelMessage],
    plan: CompactionPlan,
) -> tuple[ModelMessage, ...]:
    """Return the message slice covered by ``plan``, validating staleness."""

    if not plan.should_compact:
        raise ValueError("compaction plan does not request compaction")
    if plan.compact_start_index is None or plan.compact_end_index is None:
        raise ValueError("compaction plan is missing range indices")
    if plan.compact_start_index < 0 or plan.compact_end_index > len(messages):
        raise ValueError("compaction plan range is outside the message list")

    compacted = tuple(messages[plan.compact_start_index : plan.compact_end_index])
    covered_ids = [message.message_id for message in compacted]
    if covered_ids != plan.compact_message_ids:
        raise ValueError("compaction plan message IDs do not match the message list")
    return tuple(message.model_copy(deep=True) for message in compacted)


def render_message_transcript(messages: Sequence[ModelMessage]) -> str:
    """Render provider-neutral messages into a deterministic text transcript."""

    sections: list[str] = []
    for index, message in enumerate(messages):
        sequence = "none" if message.sequence is None else str(message.sequence)
        lines = [
            f"--- Message {index + 1}: id={message.message_id} role={message.role.value} "
            f"sequence={sequence} ---",
        ]
        content = _render_content(message)
        if content:
            lines.append(content)
        else:
            lines.append("[No text content]")

        if isinstance(message, AssistantMessage):
            tool_calls = assistant_tool_calls(message)
            if tool_calls:
                lines.append("Assistant-requested tool calls:")
                for tool_call in tool_calls:
                    arguments = _json_dumps(tool_call.arguments)
                    lines.append(
                        "- "
                        f"order={tool_call.order} call_id={tool_call.call_id} "
                        f"tool={tool_call.tool_name} arguments={arguments}"
                    )
        elif isinstance(message, ToolResultMessage):
            result = message.result
            lines.append(
                "Tool result metadata: "
                f"call_id={message.tool_call_id} tool={message.tool_name} "
                f"success={_yes_no(result.success)} timed_out={_yes_no(result.timed_out)} "
                f"truncated={_yes_no(result.truncated)}"
            )
            if result.error is not None:
                lines.append(
                    "Tool error: "
                    f"type={result.error.error_type} message={result.error.message}"
                )
            lines.append("Tool result visible output:")
            lines.append(model_visible_tool_result_text(result))
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def _render_content(message: ModelMessage) -> str:
    rendered_parts: list[str] = []
    for part in message.content:
        if isinstance(part, TextContent):
            rendered_parts.append(part.text)
        else:
            covered = ", ".join(part.covered_message_ids) or "none"
            rendered_parts.append(
                "[Prior compaction summary covering message IDs: "
                f"{covered}]\n{part.summary}"
            )
    return "\n".join(rendered_parts)


def _json_dumps(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


__all__ = (
    "COMPACTION_PROMPT_VERSION",
    "COMPACTION_SYSTEM_PROMPT",
    "SUMMARY_SECTION_HEADINGS",
    "compacted_messages_from_plan",
    "render_compaction_user_prompt",
    "render_message_transcript",
)
