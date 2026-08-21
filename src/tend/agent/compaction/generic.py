"""Provider-independent summarization compactor."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated

from pydantic import Field, TypeAdapter, model_validator

from tend._common.errors import FrameworkError
from tend._common.types import JsonObject, StrictModel, new_id
from tend.agent.compaction.planner import CompactionPlan
from tend.agent.compaction.prompts import (
    COMPACTION_PROMPT_VERSION,
    COMPACTION_SYSTEM_PROMPT,
    compacted_messages_from_plan,
    render_compaction_user_prompt,
)
from tend.llm.models.base import ModelAdapter
from tend.llm.models.messages import (
    AssistantMessage,
    CompactionSummaryContent,
    SystemMessage,
    TextContent,
    UserMessage,
)
from tend.llm.models.profiles import ModelProfile
from tend.llm.models.reasoning import ReasoningSettings
from tend.llm.models.requests import ModelMessage, ModelRequest, ModelResponse
from tend.llm.usage import Usage, calculate_token_cost, usage_with_model_request_count

_JSON_OBJECT_ADAPTER: TypeAdapter[JsonObject] = TypeAdapter(JsonObject)
_NonNegativeInt = Annotated[int, Field(ge=0)]


def _empty_strings() -> list[str]:
    return []


class CompactionError(FrameworkError):
    """Raised when generic compaction cannot produce a valid summary."""


class GenericCompactionResult(StrictModel):
    """Result of one generic summarization compaction request."""

    compaction_id: str = Field(min_length=1)
    request_id: str = Field(min_length=1)
    response_id: str = Field(min_length=1)
    compact_start_index: _NonNegativeInt
    compact_end_index: _NonNegativeInt
    covered_message_ids: list[str] = Field(default_factory=_empty_strings)
    preserved_message_ids: list[str] = Field(default_factory=_empty_strings)
    split_turn_prefix: bool = False
    summary: str = Field(min_length=1)
    summary_message: AssistantMessage
    usage: Usage = Field(default_factory=Usage)
    plan: CompactionPlan

    @model_validator(mode="after")
    def _validate_summary_message(self) -> GenericCompactionResult:
        if self.compact_end_index <= self.compact_start_index:
            raise ValueError("compaction result range must be non-empty")
        if self.covered_message_ids != self.plan.compact_message_ids:
            raise ValueError("covered_message_ids must match plan compact_message_ids")
        if len(self.summary_message.content) != 1:
            raise ValueError("summary_message must contain exactly one content part")
        part = self.summary_message.content[0]
        if not isinstance(part, CompactionSummaryContent):
            raise ValueError("summary_message content must be a compaction summary")
        if part.summary != self.summary:
            raise ValueError("summary_message summary must match result summary")
        if part.covered_message_ids != self.covered_message_ids:
            raise ValueError("summary_message covered IDs must match result covered IDs")
        return self


def apply_compaction_result(
    messages: Sequence[ModelMessage],
    result: GenericCompactionResult,
) -> list[ModelMessage]:
    """Return active messages with the covered range replaced by the summary.

    The original input sequence is not mutated. The covered message IDs must
    still match the result's planned range so callers cannot accidentally apply
    a summary to a different active context.
    """

    start = result.compact_start_index
    end = result.compact_end_index
    if end > len(messages):
        raise CompactionError("compaction result range exceeds message history length")
    covered_ids = [message.message_id for message in messages[start:end]]
    if covered_ids != result.covered_message_ids:
        raise CompactionError("compaction result covered IDs do not match active history")

    prefix = [message.model_copy(deep=True) for message in messages[:start]]
    suffix = [message.model_copy(deep=True) for message in messages[end:]]
    summary_message = result.summary_message.model_copy(deep=True)
    return [*prefix, summary_message, *suffix]


class GenericSummarizationCompactor:
    """Summarize a planned message range through a provider-neutral model."""

    __slots__ = ("_max_output_tokens", "_model", "_model_name", "_reasoning")

    _max_output_tokens: int | None
    _model: ModelAdapter
    _model_name: str | None
    _reasoning: ReasoningSettings | None

    def __init__(
        self,
        model: ModelAdapter,
        *,
        model_name: str | None = None,
        reasoning: ReasoningSettings | None = None,
        max_output_tokens: int | None = None,
    ) -> None:
        if model_name is not None and not model_name:
            raise ValueError("model_name must be non-empty when provided")
        if max_output_tokens is not None and max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive when provided")
        self._model = model
        self._model_name = model_name
        self._reasoning = reasoning.model_copy(deep=True) if reasoning is not None else None
        self._max_output_tokens = max_output_tokens

    @property
    def model(self) -> ModelAdapter:
        """Return the configured compaction model adapter."""

        return self._model

    async def compact(
        self,
        *,
        messages: Sequence[ModelMessage],
        plan: CompactionPlan,
        compaction_id: str | None = None,
    ) -> GenericCompactionResult:
        """Run one generic compaction request and validate the returned summary."""

        resolved_compaction_id = compaction_id or new_id("compact")
        profile = self._model.profile
        request = build_compaction_request(
            messages=messages,
            plan=plan,
            compaction_id=resolved_compaction_id,
            model_name=self._model_name or _profile_model_name(profile),
            reasoning=self._reasoning,
            max_output_tokens=self._max_output_tokens,
        )
        try:
            response = await self._model.generate(request)
        except Exception as exc:
            raise CompactionError("compaction model request failed") from exc

        summary = _summary_from_response(response)
        usage = _usage_from_response(response, profile=profile)
        summary_message = _summary_message(
            summary=summary,
            covered_message_ids=plan.compact_message_ids,
            compaction_id=resolved_compaction_id,
            request_id=request.request_id,
            response_id=response.response_id,
        )
        return GenericCompactionResult(
            compaction_id=resolved_compaction_id,
            request_id=request.request_id,
            response_id=response.response_id,
            compact_start_index=_required_index(plan.compact_start_index),
            compact_end_index=_required_index(plan.compact_end_index),
            covered_message_ids=list(plan.compact_message_ids),
            preserved_message_ids=list(plan.preserved_message_ids),
            split_turn_prefix=plan.split_turn_prefix,
            summary=summary,
            summary_message=summary_message,
            usage=usage,
            plan=plan.model_copy(deep=True),
        )


async def compact_messages(
    *,
    model: ModelAdapter,
    messages: Sequence[ModelMessage],
    plan: CompactionPlan,
    compaction_id: str | None = None,
    model_name: str | None = None,
    reasoning: ReasoningSettings | None = None,
    max_output_tokens: int | None = None,
) -> GenericCompactionResult:
    """Convenience wrapper for one provider-independent compaction request."""

    compactor = GenericSummarizationCompactor(
        model,
        model_name=model_name,
        reasoning=reasoning,
        max_output_tokens=max_output_tokens,
    )
    return await compactor.compact(
        messages=messages,
        plan=plan,
        compaction_id=compaction_id,
    )


def build_compaction_request(
    *,
    messages: Sequence[ModelMessage],
    plan: CompactionPlan,
    compaction_id: str | None = None,
    model_name: str | None = None,
    reasoning: ReasoningSettings | None = None,
    max_output_tokens: int | None = None,
) -> ModelRequest:
    """Build the provider-neutral model request for a planned compaction."""

    resolved_compaction_id = compaction_id or new_id("compact")
    compacted_messages = compacted_messages_from_plan(messages=messages, plan=plan)
    if not compacted_messages:
        raise CompactionError("compaction plan covers no messages")

    output_limit = max_output_tokens if max_output_tokens is not None else plan.target_tokens
    if output_limit < 1:
        raise CompactionError("compaction output token limit must be positive")

    user_prompt = render_compaction_user_prompt(messages=messages, plan=plan)
    return ModelRequest(
        model_name=model_name,
        messages=[
            SystemMessage(content=[TextContent(text=COMPACTION_SYSTEM_PROMPT)]),
            UserMessage(content=[TextContent(text=user_prompt)]),
        ],
        tools=[],
        reasoning=reasoning.model_copy(deep=True) if reasoning is not None else None,
        max_output_tokens=output_limit,
        request_metadata=_request_metadata(
            compaction_id=resolved_compaction_id,
            plan=plan,
            covered_message_ids=[message.message_id for message in compacted_messages],
        ),
    )


def _summary_from_response(response: ModelResponse) -> str:
    if response.tool_calls:
        raise CompactionError("compaction model returned tool calls instead of a summary")
    summary = response.final_text
    if summary is None:
        raise CompactionError("compaction model response did not include summary text")
    summary = summary.strip()
    if not summary:
        raise CompactionError("compaction summary must be non-empty")
    return summary


def _summary_message(
    *,
    summary: str,
    covered_message_ids: Sequence[str],
    compaction_id: str,
    request_id: str,
    response_id: str,
) -> AssistantMessage:
    metadata = _JSON_OBJECT_ADAPTER.validate_python(
        {
            "compaction_id": compaction_id,
            "compaction_request_id": request_id,
            "compaction_response_id": response_id,
            "prompt_version": COMPACTION_PROMPT_VERSION,
        }
    )
    return AssistantMessage(
        content=[
            CompactionSummaryContent(
                summary=summary,
                covered_message_ids=list(covered_message_ids),
            )
        ],
        provider_metadata=metadata,
    )


def _request_metadata(
    *,
    compaction_id: str,
    plan: CompactionPlan,
    covered_message_ids: Sequence[str],
) -> JsonObject:
    return _JSON_OBJECT_ADAPTER.validate_python(
        {
            "purpose": "generic_compaction",
            "prompt_version": COMPACTION_PROMPT_VERSION,
            "compaction_id": compaction_id,
            "compact_start_index": _required_index(plan.compact_start_index),
            "compact_end_index": _required_index(plan.compact_end_index),
            "covered_message_ids": list(covered_message_ids),
            "preserved_message_ids": list(plan.preserved_message_ids),
            "target_tokens": plan.target_tokens,
            "split_turn_prefix": plan.split_turn_prefix,
            "trigger_reasons": [reason.value for reason in plan.trigger_reasons],
        }
    )


def _usage_from_response(response: ModelResponse, *, profile: ModelProfile | None) -> Usage:
    usage = usage_with_model_request_count(response.usage)
    if usage.cost is not None:
        return usage
    if profile is None:
        return usage
    cost = calculate_token_cost(usage.tokens, profile.pricing)
    if cost is None:
        return usage
    return usage.model_copy(update={"cost": cost}, deep=True)


def _profile_model_name(profile: ModelProfile | None) -> str | None:
    if profile is None:
        return None
    return profile.model_name


def _required_index(index: int | None) -> int:
    if index is None:
        raise CompactionError("compaction plan is missing range indices")
    return index


__all__ = (
    "CompactionError",
    "GenericCompactionResult",
    "GenericSummarizationCompactor",
    "apply_compaction_result",
    "build_compaction_request",
    "compact_messages",
)
