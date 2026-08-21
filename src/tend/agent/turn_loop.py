"""Shared provider-neutral turn loop."""

from __future__ import annotations

import json
from asyncio import CancelledError
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, TypeAdapter

from tend._common.errors import ErrorInfo
from tend._common.types import JsonObject, StopReason, new_id
from tend.agent.cancellation import CancellationState
from tend.agent.compaction import (
    CompactionError,
    CompactionPlan,
    GenericCompactionResult,
    GenericSummarizationCompactor,
    apply_compaction_result,
    plan_compaction,
)
from tend.agent.config import RuntimeConfig
from tend.agent.context import assistant_message_from_response, build_active_context
from tend.agent.limits import MonotonicClock, TurnLimitTracker
from tend.agent.persistence.events import (
    CompactionCompletedEvent,
    CompactionCompletedPayload,
    CompactionStartedEvent,
    CompactionStartedPayload,
    ModelRequestFailedEvent,
    ModelRequestFailedPayload,
    ModelRequestStartedEvent,
    ModelRequestStartedPayload,
    ModelResponseCompletedEvent,
    ModelResponseCompletedPayload,
    ToolCallCompletedEvent,
    ToolCallCompletedPayload,
    ToolCallStartedEvent,
    ToolCallStartedPayload,
    TurnCompletedEvent,
    TurnCompletedPayload,
    TurnInterruptedEvent,
    TurnInterruptedPayload,
    TurnStartedEvent,
    TurnStartedPayload,
)
from tend.agent.results import FinalResultOutput, StopResult, TurnResult
from tend.agent.session import Session
from tend.agent.tools.base import Tool
from tend.agent.tools.context import ToolContext, ToolEventCallback
from tend.agent.tools.executor import (
    TOOL_CALL_COMPLETED_EVENT,
    TOOL_CALL_STARTED_EVENT,
    execute_tool_calls,
)
from tend.llm.context_estimation import (
    CONTEXT_ESTIMATE_METADATA_KEY,
    ContextEstimate,
    context_estimate_to_metadata,
    estimate_context,
    estimate_context_from_api_anchor,
)
from tend.llm.models.base import ModelAdapter
from tend.llm.models.messages import TextContent, UserMessage
from tend.llm.models.profiles import ModelProfile
from tend.llm.models.provider import ProviderCompletionStatus
from tend.llm.models.reasoning import ReasoningSettings
from tend.llm.models.requests import ModelMessage, ModelRequest, ModelResponse
from tend.llm.models.tools import ToolCall, ToolResult, ToolResultMessage
from tend.llm.providers.errors import ProviderRequestError
from tend.llm.retries import decide_retry, wait_for_retry
from tend.llm.usage import (
    Usage,
    calculate_token_cost,
    usage_with_model_request_count,
    usage_with_retry_attempt_count,
)

_JSON_OBJECT_ADAPTER: TypeAdapter[JsonObject] = TypeAdapter(JsonObject)
_FINAL_RESULT_TOOL_NAME = "final_result"
_OUTPUT_TOOL_KIND_METADATA_KEY = "tend_tool_kind"
_OUTPUT_TOOL_KIND = "output"
# When the model tries to end a turn with prose instead of calling the required
# ``final_result`` output tool, force the tool on the next request — up to this many times
# before giving up and accepting the prose as a (non-structured) final response.
_MAX_FORCED_FINAL_RESULT_REASKS = 2
_FORCE_FINAL_RESULT_NUDGE = (
    "Do not write any more prose. Submit your result now by calling the `final_result` "
    "tool exactly once with the required fields."
)
_NATURAL_NATIVE_STOP_REASONS: frozenset[str] = frozenset({"stop", "end_turn", "completed"})
_NON_FINAL_NATIVE_STOP_REASONS: frozenset[str] = frozenset(
    {"tool_use", "pause_turn", "max_tokens", "max_output_tokens", "incomplete", "failed"}
)
_NON_FINAL_RESPONSE_REASONS: frozenset[StopReason] = frozenset(
    {
        StopReason.MAX_MODEL_REQUESTS,
        StopReason.MAX_TOOL_CALLS,
        StopReason.MAX_ITERATIONS,
        StopReason.MAX_WALL_TIME,
        StopReason.MAX_TOKENS,
        StopReason.MAX_COST,
        StopReason.MODEL_ERROR,
        StopReason.INTERRUPTED,
        StopReason.COMPACTION_FAILED,
    }
)
_CONTEXT_OVERFLOW_CODES: frozenset[str] = frozenset(
    {
        "context_length_exceeded",
        "context_overflow",
        "context_too_large",
        "max_context_length_exceeded",
    }
)


@dataclass(frozen=True, slots=True)
class _CompactionOutcome:
    messages: list[ModelMessage]
    usage: Usage
    context_estimate: ContextEstimate | None
    result: GenericCompactionResult


async def run_turn(
    *,
    system_prompt: str,
    model: ModelAdapter,
    prompt: str,
    tools: Iterable[Tool[Any]] = (),
    session: Session | None = None,
    config: RuntimeConfig | None = None,
    model_name: str | None = None,
    reasoning: ReasoningSettings | None = None,
    max_output_tokens: int | None = None,
    cancellation: CancellationState | None = None,
    clock: MonotonicClock | None = None,
) -> TurnResult:
    """Run one provider-neutral turn until final response or structured stop.

    The loop is intentionally provider-agnostic: model adapters return normalized
    ``ModelResponse`` values, built-in/runtime tools execute through the
    sequential executor, and tool results are appended as provider-neutral
    ``ToolResultMessage`` values for follow-up model requests.
    """

    runtime_config = config or RuntimeConfig()
    model_profile = model.profile
    cancellation_state = cancellation or CancellationState()
    limit_tracker = TurnLimitTracker(runtime_config.limits, clock=clock)
    enabled_tools = tuple(tools)
    tool_schemas = tuple(_tool_schemas(enabled_tools))
    output_tool_name = _enabled_output_tool_name(_tool_map(enabled_tools))
    turn_id = new_id("turn", width=3)
    session_state = session.state if session is not None else None
    context = build_active_context(
        system_prompt=system_prompt,
        new_user_prompt=prompt,
        session_state=session_state,
    )
    messages = _copy_model_messages(context.messages)
    tool_event_callback = _tool_event_callback(session, turn_id=turn_id)

    _append_turn_started(
        session,
        turn_id=turn_id,
        prompt=prompt,
        input_message_id=context.input_message_id,
    )

    usage = Usage()
    model_request_count = 0
    tool_call_count = 0
    iteration_count = 0
    all_tool_calls: list[ToolCall] = []
    all_tool_results: list[ToolResult] = []
    latest_context_estimate: ContextEstimate | None = None
    context_overflow_retry_used = False
    skip_pre_request_compaction_once = False
    provider_error_retry_attempt = 1
    provider_error_retry_request_id: str | None = None
    api_context_anchor: int | None = None
    api_anchor_new_messages: list[ModelMessage] = []
    forced_final_result_reasks = 0
    force_final_result_next = False

    while True:
        if api_context_anchor is not None and runtime_config.usage.estimate_context_tokens:
            pending_context_estimate = estimate_context_from_api_anchor(
                anchor_tokens=api_context_anchor,
                new_messages=api_anchor_new_messages,
                profile=model_profile,
                config=runtime_config.usage.token_estimator,
            )
            # Keep ``api_anchor_new_messages`` until the next response refreshes
            # the anchor. A retryable provider error re-enters this loop with the
            # same messages; clearing here would make the retry estimate
            # ``anchor + 0`` and undercount the appended tool-result delta.
        else:
            pending_context_estimate = _estimate_context_for_request(
                messages=messages,
                tool_schemas=tool_schemas,
                runtime_config=runtime_config,
                model_profile=model_profile,
                reasoning=reasoning,
            )
        if pending_context_estimate is not None:
            latest_context_estimate = pending_context_estimate
        if cancellation_state.is_cancelled:
            return _complete_interrupted_turn(
                session,
                turn_id=turn_id,
                message=_cancellation_message(
                    cancellation_state,
                    boundary="before model request",
                ),
                usage=usage,
                model_request_count=model_request_count,
                tool_call_count=tool_call_count,
                tool_calls=all_tool_calls,
                tool_results=all_tool_results,
                context_estimate=latest_context_estimate,
            )

        stop = limit_tracker.check_before_model_request(
            iteration_count=iteration_count,
            model_request_count=model_request_count,
            usage=usage,
        )
        if stop is not None:
            return _complete_stopped_turn(
                session,
                turn_id=turn_id,
                stop=stop,
                final_response=None,
                usage=usage,
                model_request_count=model_request_count,
                tool_call_count=tool_call_count,
                tool_calls=all_tool_calls,
                tool_results=all_tool_results,
                context_estimate=latest_context_estimate,
            )

        if skip_pre_request_compaction_once:
            skip_pre_request_compaction_once = False
        else:
            try:
                compaction = await _maybe_compact_active_context(
                    messages=messages,
                    model=model,
                    model_name=model_name,
                    reasoning=reasoning,
                    tool_schemas=tool_schemas,
                    runtime_config=runtime_config,
                    model_profile=model_profile,
                    session=session,
                    turn_id=turn_id,
                    context_estimate=pending_context_estimate,
                )
            except CancelledError as exc:
                _append_turn_interrupted(
                    session,
                    turn_id=turn_id,
                    message="Turn interrupted while compacting context.",
                    incomplete_event_id=session.last_event_id if session is not None else None,
                    error=_error_info_from_base_exception(exc),
                    usage=usage,
                )
                raise
            except CompactionError as exc:
                return _complete_compaction_failed_turn(
                    session,
                    turn_id=turn_id,
                    error=_error_info_from_compaction_exception(exc),
                    usage=usage,
                    model_request_count=model_request_count,
                    tool_call_count=tool_call_count,
                    tool_calls=all_tool_calls,
                    tool_results=all_tool_results,
                    context_estimate=latest_context_estimate,
                )
            if compaction is not None:
                messages = compaction.messages
                usage = usage.add(compaction.usage)
                pending_context_estimate = compaction.context_estimate
                if pending_context_estimate is not None:
                    latest_context_estimate = pending_context_estimate
                api_context_anchor = None
                api_anchor_new_messages = []
                stop = limit_tracker.check_before_model_request(
                    iteration_count=iteration_count,
                    model_request_count=model_request_count,
                    usage=usage,
                )
                if stop is not None:
                    return _complete_stopped_turn(
                        session,
                        turn_id=turn_id,
                        stop=stop,
                        final_response=None,
                        usage=usage,
                        model_request_count=model_request_count,
                        tool_call_count=tool_call_count,
                        tool_calls=all_tool_calls,
                        tool_results=all_tool_results,
                        context_estimate=latest_context_estimate,
                    )

        request_attempt = provider_error_retry_attempt
        # Annotate explicitly: _build_model_request constructs ModelRequest via **dict[str, Any],
        # which pyright treats as partially unknown under loop widening, leaking Unknown into the
        # loop-carried provider_error_retry_request_id (assigned request.request_id below).
        request: ModelRequest = _build_model_request(
            messages=messages,
            tool_schemas=tool_schemas,
            runtime_config=runtime_config,
            model_name=model_name,
            # On a forced final-result re-ask, drop reasoning: forced tool choice is
            # incompatible with extended thinking on some models (e.g. opus), where the
            # force would otherwise be silently dropped. ``disable_reasoning`` makes this
            # authoritative -- a plain ``reasoning=None`` would otherwise fall back to an
            # adapter's configured default reasoning and re-enable thinking.
            reasoning=None if force_final_result_next else reasoning,
            disable_reasoning=force_final_result_next,
            max_output_tokens=max_output_tokens,
            iteration_count=iteration_count,
            context_estimate=pending_context_estimate,
            request_id=provider_error_retry_request_id,
            force_tool_name=output_tool_name if force_final_result_next else None,
        )
        _append_model_request_started(
            session,
            turn_id=turn_id,
            request=request,
            attempt=request_attempt,
        )
        model_request_count += 1

        try:
            response = await model.generate(request)
        except CancelledError as exc:
            _append_turn_interrupted(
                session,
                turn_id=turn_id,
                message="Turn interrupted while waiting for model response.",
                incomplete_event_id=session.last_event_id if session is not None else None,
                error=_error_info_from_base_exception(exc),
                usage=usage,
            )
            raise
        except Exception as exc:
            failed_request_usage = Usage(model_requests=1)
            if _should_retry_after_context_overflow(
                exc,
                runtime_config=runtime_config,
                retry_used=context_overflow_retry_used,
            ):
                usage = usage.add(failed_request_usage)
                _append_model_request_failed(
                    session,
                    turn_id=turn_id,
                    request_id=request.request_id,
                    attempt=request_attempt,
                    error=_error_info_from_exception(exc),
                    usage=failed_request_usage,
                    retryable=True,
                )
                try:
                    compaction = await _compact_for_context_overflow_retry(
                        messages=messages,
                        model=model,
                        model_name=model_name,
                        reasoning=reasoning,
                        tool_schemas=tool_schemas,
                        runtime_config=runtime_config,
                        model_profile=model_profile,
                        session=session,
                        turn_id=turn_id,
                        context_estimate=latest_context_estimate,
                    )
                except CancelledError as cancel_exc:
                    _append_turn_interrupted(
                        session,
                        turn_id=turn_id,
                        message="Turn interrupted while compacting after context overflow.",
                        incomplete_event_id=session.last_event_id if session is not None else None,
                        error=_error_info_from_base_exception(cancel_exc),
                        usage=usage,
                    )
                    raise
                except CompactionError as compaction_exc:
                    return _complete_compaction_failed_turn(
                        session,
                        turn_id=turn_id,
                        error=_error_info_from_compaction_exception(compaction_exc),
                        usage=usage,
                        model_request_count=model_request_count,
                        tool_call_count=tool_call_count,
                        tool_calls=all_tool_calls,
                        tool_results=all_tool_results,
                        context_estimate=latest_context_estimate,
                    )
                messages = compaction.messages
                usage = usage.add(compaction.usage)
                latest_context_estimate = compaction.context_estimate or latest_context_estimate
                context_overflow_retry_used = True
                skip_pre_request_compaction_once = True
                api_context_anchor = None
                api_anchor_new_messages = []
                continue

            if isinstance(exc, ProviderRequestError):
                decision = decide_retry(
                    runtime_config.retries,
                    category=exc.category,
                    attempt=request_attempt,
                    retry_after=exc.retry_after,
                    request_id=request.request_id,
                    error=_error_info_from_exception(exc),
                )
                if decision.should_retry:
                    # _log.warning(
                    #     "model request retry attempt=%d/%d delay=%.1fs request_id=%s: %s",
                    #     decision.attempt, decision.max_attempts, decision.delay_seconds,
                    #     request.request_id, exc,
                    # )
                    failed_request_usage = usage_with_retry_attempt_count(
                        failed_request_usage
                    )
                    usage = usage.add(failed_request_usage)
                    _append_model_request_failed(
                        session,
                        turn_id=turn_id,
                        request_id=request.request_id,
                        attempt=request_attempt,
                        error=_error_info_from_exception(exc),
                        usage=failed_request_usage,
                        retryable=True,
                    )
                    try:
                        await wait_for_retry(decision)
                    except CancelledError as cancel_exc:
                        _append_turn_interrupted(
                            session,
                            turn_id=turn_id,
                            message="Turn interrupted while waiting to retry model request.",
                            incomplete_event_id=(
                                session.last_event_id if session is not None else None
                            ),
                            error=_error_info_from_base_exception(cancel_exc),
                            usage=usage,
                        )
                        raise
                    provider_error_retry_attempt = decision.next_attempt or (
                        request_attempt + 1
                    )
                    provider_error_retry_request_id = request.request_id
                    continue

            usage = usage.add(failed_request_usage)
            # _log.warning("model request failed request_id=%s: %s", request.request_id, exc)
            return _complete_model_error_turn(
                session,
                turn_id=turn_id,
                request_id=request.request_id,
                attempt=request_attempt,
                error=_error_info_from_exception(exc),
                usage=usage,
                failed_request_usage=failed_request_usage,
                model_request_count=model_request_count,
                tool_call_count=tool_call_count,
                tool_calls=all_tool_calls,
                tool_results=all_tool_results,
                context_estimate=latest_context_estimate,
            )

        response_usage = _response_usage(
            response.usage,
            model_profile=model_profile,
        )
        response = response.model_copy(update={"usage": response_usage}, deep=True)
        usage = usage.add(response_usage)
        iteration_count += 1
        provider_error_retry_attempt = 1
        provider_error_retry_request_id = None
        _t = response_usage.tokens
        # Input side (input + cache reads + cache writes) is disjoint in tend's
        # normalized usage convention, so it sums to the full prompt the model
        # processed; add output for what it wrote. When a provider omits usage
        # every count is 0 -- leave the anchor unset so the next request falls
        # back to the char-based estimator instead of collapsing to just the new
        # tool-result delta.
        _input_side_tokens = _t.input_tokens + _t.cache_read_tokens + _t.cache_write_tokens
        if _input_side_tokens > 0:
            api_context_anchor = _input_side_tokens + _t.output_tokens
        else:
            api_context_anchor = None
        # A successful response supersedes the prior anchor and every delta that
        # was included in this request. Continuation branches below repopulate
        # this list only with messages appended after the fresh response.
        api_anchor_new_messages = []
        # The force signal applied to this request (and was kept across any provider-error
        # retries, which rebuild the request); now that a response arrived, consume it. The
        # natural-completion handler below may re-arm it.
        force_final_result_next = False
        _append_model_response_completed(
            session,
            turn_id=turn_id,
            request_id=request.request_id,
            response=response,
            usage=response_usage,
        )

        if cancellation_state.is_cancelled:
            return _complete_interrupted_turn(
                session,
                turn_id=turn_id,
                message=_cancellation_message(
                    cancellation_state,
                    boundary="after model response",
                ),
                usage=usage,
                model_request_count=model_request_count,
                tool_call_count=tool_call_count,
                tool_calls=all_tool_calls,
                tool_results=all_tool_results,
                context_estimate=latest_context_estimate,
            )

        if response.tool_calls:
            response_tool_calls = _copy_tool_calls(response.tool_calls)
            all_tool_calls.extend(response_tool_calls)
            enabled_tool_map = _tool_map(enabled_tools)
            output_calls, ordinary_calls = _partition_output_tool_calls(
                response_tool_calls,
                enabled_tool_map,
            )
            if cancellation_state.is_cancelled:
                return _complete_interrupted_turn(
                    session,
                    turn_id=turn_id,
                    message=_cancellation_message(
                        cancellation_state,
                        boundary="before tool execution",
                    ),
                    usage=usage,
                    model_request_count=model_request_count,
                    tool_call_count=tool_call_count,
                    tool_calls=all_tool_calls,
                    tool_results=all_tool_results,
                    context_estimate=latest_context_estimate,
                )

            messages.append(assistant_message_from_response(response))
            tool_context = ToolContext(
                cwd=runtime_config.cwd,
                session_id=session.session_id if session is not None else None,
                turn_id=turn_id,
                runtime_config=runtime_config,
                event_callback=tool_event_callback,
                cancellation=cancellation_state,
            )
            batch_tool_results: list[ToolResult] = []

            for output_call in _sort_tool_calls_for_execution(output_calls):
                stop = limit_tracker.check_before_tool_execution(
                    tool_call_count=tool_call_count,
                    requested_tool_count=1,
                    usage=usage,
                )
                if stop is not None:
                    return _complete_stopped_turn(
                        session,
                        turn_id=turn_id,
                        stop=stop,
                        final_response=None,
                        usage=usage,
                        model_request_count=model_request_count,
                        tool_call_count=tool_call_count,
                        tool_calls=all_tool_calls,
                        tool_results=all_tool_results,
                        context_estimate=latest_context_estimate,
                    )

                try:
                    results = await execute_tool_calls(
                        [output_call],
                        enabled_tool_map,
                        tool_context,
                    )
                except CancelledError as exc:
                    _append_turn_interrupted(
                        session,
                        turn_id=turn_id,
                        message="Turn interrupted while executing tools.",
                        incomplete_event_id=session.last_event_id if session is not None else None,
                        error=_error_info_from_base_exception(exc),
                        usage=usage,
                    )
                    raise
                copied_results = _copy_tool_results(results)
                all_tool_results.extend(copied_results)
                batch_tool_results.extend(copied_results)
                tool_call_count += len(copied_results)
                usage = usage.add(Usage(tool_calls=len(copied_results)))
                final_result = _first_successful_final_result(copied_results)
                if final_result is not None:
                    return _complete_final_result_turn(
                        session,
                        turn_id=turn_id,
                        final_result=final_result,
                        usage=usage,
                        model_request_count=model_request_count,
                        tool_call_count=tool_call_count,
                        tool_calls=all_tool_calls,
                        tool_results=all_tool_results,
                        context_estimate=latest_context_estimate,
                    )
                if cancellation_state.is_cancelled:
                    return _complete_interrupted_turn(
                        session,
                        turn_id=turn_id,
                        message=_cancellation_message(
                            cancellation_state,
                            boundary="after tool execution",
                        ),
                        usage=usage,
                        model_request_count=model_request_count,
                        tool_call_count=tool_call_count,
                        tool_calls=all_tool_calls,
                        tool_results=all_tool_results,
                        context_estimate=latest_context_estimate,
                    )

            if ordinary_calls:
                stop = limit_tracker.check_before_tool_execution(
                    tool_call_count=tool_call_count,
                    requested_tool_count=len(ordinary_calls),
                    usage=usage,
                )
                if stop is not None:
                    return _complete_stopped_turn(
                        session,
                        turn_id=turn_id,
                        stop=stop,
                        final_response=None,
                        usage=usage,
                        model_request_count=model_request_count,
                        tool_call_count=tool_call_count,
                        tool_calls=all_tool_calls,
                        tool_results=all_tool_results,
                        context_estimate=latest_context_estimate,
                    )

                try:
                    results = await execute_tool_calls(
                        ordinary_calls,
                        enabled_tool_map,
                        tool_context,
                    )
                except CancelledError as exc:
                    _append_turn_interrupted(
                        session,
                        turn_id=turn_id,
                        message="Turn interrupted while executing tools.",
                        incomplete_event_id=session.last_event_id if session is not None else None,
                        error=_error_info_from_base_exception(exc),
                        usage=usage,
                    )
                    raise
                copied_results = _copy_tool_results(results)
                all_tool_results.extend(copied_results)
                batch_tool_results.extend(copied_results)
                tool_call_count += len(copied_results)
                usage = usage.add(Usage(tool_calls=len(copied_results)))
            ordered_batch_results = _sort_tool_results_for_response(
                batch_tool_results,
                response_tool_calls,
            )
            if ordered_batch_results:
                all_tool_results[-len(ordered_batch_results) :] = ordered_batch_results
            messages.extend(
                ToolResultMessage.from_result(result) for result in ordered_batch_results
            )
            api_anchor_new_messages = [
                ToolResultMessage.from_result(r) for r in ordered_batch_results
            ]
            if cancellation_state.is_cancelled:
                return _complete_interrupted_turn(
                    session,
                    turn_id=turn_id,
                    message=_cancellation_message(
                        cancellation_state,
                        boundary="after tool execution",
                    ),
                    usage=usage,
                    model_request_count=model_request_count,
                    tool_call_count=tool_call_count,
                    tool_calls=all_tool_calls,
                    tool_results=all_tool_results,
                    context_estimate=latest_context_estimate,
                )
            continue

        if _is_natural_completed_response(response):
            if (
                output_tool_name is not None
                and forced_final_result_reasks < _MAX_FORCED_FINAL_RESULT_REASKS
                and model_profile is not None
                and model_profile.tools.supports_forced_tool_choice
            ):
                # The model finished with prose but a required output tool is configured.
                # Keep its text, nudge it, and force ``final_result`` on the next request so
                # the structured result can't be lost to chatter (turn ends as FINAL_RESULT
                # once the forced call lands; otherwise fall through after the re-ask cap).
                forced_final_result_reasks += 1
                force_final_result_next = True
                assistant_message = assistant_message_from_response(response)
                nudge_message = _force_final_result_message()
                messages.append(assistant_message)
                messages.append(nudge_message)
                # The refreshed anchor includes this response's output, so only
                # the nudge was appended after it. Replace any pre-anchor tool
                # result delta rather than double-counting stale messages.
                api_anchor_new_messages = [nudge_message]
                continue
            return _complete_stopped_turn(
                session,
                turn_id=turn_id,
                stop=StopResult(
                    reason=StopReason.FINAL_RESPONSE,
                    message="Final response produced.",
                ),
                final_response=response.final_text or "",
                usage=usage,
                model_request_count=model_request_count,
                tool_call_count=tool_call_count,
                tool_calls=all_tool_calls,
                tool_results=all_tool_results,
                context_estimate=latest_context_estimate,
            )

        return _complete_stopped_turn(
            session,
            turn_id=turn_id,
            stop=_stop_for_non_final_response(response),
            final_response=None,
            usage=usage,
            model_request_count=model_request_count,
            tool_call_count=tool_call_count,
            tool_calls=all_tool_calls,
            tool_results=all_tool_results,
            context_estimate=latest_context_estimate,
        )


def _api_anchor_estimated_tokens(context_estimate: ContextEstimate | None) -> int | None:
    if context_estimate is None or not context_estimate.is_api_anchored:
        return None
    return context_estimate.estimated_tokens


async def _maybe_compact_active_context(
    *,
    messages: Sequence[ModelMessage],
    model: ModelAdapter,
    model_name: str | None,
    reasoning: ReasoningSettings | None,
    tool_schemas: Sequence[JsonObject],
    runtime_config: RuntimeConfig,
    model_profile: ModelProfile | None,
    session: Session | None,
    turn_id: str,
    context_estimate: ContextEstimate | None,
) -> _CompactionOutcome | None:
    plan = plan_compaction(
        messages=messages,
        config=runtime_config.compaction,
        profile=model_profile,
        estimator_config=runtime_config.usage.token_estimator,
        tools=tool_schemas,
        reasoning=reasoning,
        anchor_estimated_tokens=_api_anchor_estimated_tokens(context_estimate),
    )
    if not plan.enabled or not plan.trigger_reasons:
        return None
    if not plan.should_compact:
        if not plan.char_triggered:
            # An anchor-only trigger can reflect provider-side thinking or other
            # overhead absent from the stored messages. With no char-budgeted
            # range to compact, proceeding is safe and can still succeed.
            return None
        raise CompactionError(
            "compaction was required but no safe compaction range was available"
            + (f": {plan.skip_reason}" if plan.skip_reason else "")
        )
    return await _run_generic_compaction(
        messages=messages,
        plan_reason=_compaction_reason(plan.trigger_reasons),
        model=model,
        model_name=model_name,
        reasoning=reasoning,
        tool_schemas=tool_schemas,
        runtime_config=runtime_config,
        model_profile=model_profile,
        session=session,
        turn_id=turn_id,
        context_estimate=context_estimate,
        force_context_overflow=False,
    )


async def _compact_for_context_overflow_retry(
    *,
    messages: Sequence[ModelMessage],
    model: ModelAdapter,
    model_name: str | None,
    reasoning: ReasoningSettings | None,
    tool_schemas: Sequence[JsonObject],
    runtime_config: RuntimeConfig,
    model_profile: ModelProfile | None,
    session: Session | None,
    turn_id: str,
    context_estimate: ContextEstimate | None,
) -> _CompactionOutcome:
    plan = plan_compaction(
        messages=messages,
        config=runtime_config.compaction,
        profile=model_profile,
        estimator_config=runtime_config.usage.token_estimator,
        tools=tool_schemas,
        reasoning=reasoning,
        anchor_estimated_tokens=_api_anchor_estimated_tokens(context_estimate),
        force_context_overflow=True,
    )
    if not plan.enabled:
        raise CompactionError(
            "context-overflow retry requires compaction, but compaction is disabled"
        )
    if not plan.should_compact:
        raise CompactionError(
            "context-overflow retry requires compaction, but no safe range was available"
            + (f": {plan.skip_reason}" if plan.skip_reason else "")
        )
    return await _run_generic_compaction(
        messages=messages,
        plan_reason=_compaction_reason(plan.trigger_reasons),
        model=model,
        model_name=model_name,
        reasoning=reasoning,
        tool_schemas=tool_schemas,
        runtime_config=runtime_config,
        model_profile=model_profile,
        session=session,
        turn_id=turn_id,
        context_estimate=context_estimate,
        force_context_overflow=True,
    )


async def _run_generic_compaction(
    *,
    messages: Sequence[ModelMessage],
    plan_reason: str,
    model: ModelAdapter,
    model_name: str | None,
    reasoning: ReasoningSettings | None,
    tool_schemas: Sequence[JsonObject],
    runtime_config: RuntimeConfig,
    model_profile: ModelProfile | None,
    session: Session | None,
    turn_id: str,
    context_estimate: ContextEstimate | None,
    force_context_overflow: bool,
) -> _CompactionOutcome:
    plan = plan_compaction(
        messages=messages,
        config=runtime_config.compaction,
        profile=model_profile,
        estimator_config=runtime_config.usage.token_estimator,
        tools=tool_schemas,
        reasoning=reasoning,
        anchor_estimated_tokens=_api_anchor_estimated_tokens(context_estimate),
        force_context_overflow=force_context_overflow,
    )
    if not plan.should_compact:
        raise CompactionError(
            "compaction was required but no safe compaction range was available"
            + (f": {plan.skip_reason}" if plan.skip_reason else "")
        )

    compaction_id = new_id("compact")
    _append_compaction_started(
        session,
        turn_id=turn_id,
        compaction_id=compaction_id,
        reason=plan_reason,
        planned_message_ids=plan.compact_message_ids,
        plan=plan,
        runtime_config=runtime_config,
        context_estimate=context_estimate,
    )
    compactor = GenericSummarizationCompactor(
        model,
        model_name=model_name,
        reasoning=reasoning,
    )
    result = await compactor.compact(
        messages=messages,
        plan=plan,
        compaction_id=compaction_id,
    )
    compaction_usage = _compaction_usage(result.usage)
    result = result.model_copy(update={"usage": compaction_usage}, deep=True)
    compacted_messages = apply_compaction_result(messages, result)
    compacted_context_estimate = _estimate_context_for_request(
        messages=compacted_messages,
        tool_schemas=tool_schemas,
        runtime_config=runtime_config,
        model_profile=model_profile,
        reasoning=reasoning,
    )
    _append_compaction_completed(
        session,
        turn_id=turn_id,
        result=result,
        runtime_config=runtime_config,
    )
    return _CompactionOutcome(
        messages=compacted_messages,
        usage=compaction_usage,
        context_estimate=compacted_context_estimate,
        result=result,
    )


def _build_model_request(
    *,
    messages: Sequence[ModelMessage],
    tool_schemas: Sequence[JsonObject],
    runtime_config: RuntimeConfig,
    model_name: str | None,
    reasoning: ReasoningSettings | None,
    max_output_tokens: int | None,
    iteration_count: int,
    context_estimate: ContextEstimate | None,
    request_id: str | None = None,
    force_tool_name: str | None = None,
    disable_reasoning: bool = False,
) -> ModelRequest:
    request_metadata: JsonObject = {
        "runtime_cwd": runtime_config.cwd,
        "turn_iteration": iteration_count,
    }
    if force_tool_name is not None:
        # Consumed by providers (e.g. anthropic_messages._pop_tool_choice_request) to force a
        # specific tool on this one request.
        request_metadata["force_tool_name"] = force_tool_name
    if disable_reasoning:
        # Consumed by providers (_resolve_reasoning) to suppress reasoning/thinking for this
        # one request even when the adapter is configured with default reasoning.
        request_metadata["disable_reasoning"] = True
    if context_estimate is not None:
        request_metadata[CONTEXT_ESTIMATE_METADATA_KEY] = context_estimate_to_metadata(
            context_estimate
        )
    request_kwargs: dict[str, Any] = {
        "model_name": model_name,
        "messages": _copy_model_messages(messages),
        "tools": [_copy_json_object(schema) for schema in tool_schemas],
        "reasoning": reasoning.model_copy(deep=True) if reasoning is not None else None,
        "max_output_tokens": max_output_tokens,
        "request_metadata": request_metadata,
    }
    if request_id is not None:
        request_kwargs["request_id"] = request_id
    return ModelRequest(**request_kwargs)


def _complete_model_error_turn(
    session: Session | None,
    *,
    turn_id: str,
    request_id: str,
    attempt: int,
    error: ErrorInfo,
    usage: Usage,
    failed_request_usage: Usage,
    model_request_count: int,
    tool_call_count: int,
    tool_calls: Iterable[ToolCall],
    tool_results: Iterable[ToolResult],
    context_estimate: ContextEstimate | None = None,
) -> TurnResult:
    _append_model_request_failed(
        session,
        turn_id=turn_id,
        request_id=request_id,
        attempt=attempt,
        error=error,
        usage=failed_request_usage,
    )
    return _complete_stopped_turn(
        session,
        turn_id=turn_id,
        stop=StopResult(
            reason=StopReason.MODEL_ERROR,
            message="Model request failed before a normalized response was returned.",
            error=error,
        ),
        final_response=None,
        usage=usage,
        model_request_count=model_request_count,
        tool_call_count=tool_call_count,
        tool_calls=tool_calls,
        tool_results=tool_results,
        context_estimate=context_estimate,
    )


def _complete_compaction_failed_turn(
    session: Session | None,
    *,
    turn_id: str,
    error: ErrorInfo,
    usage: Usage,
    model_request_count: int,
    tool_call_count: int,
    tool_calls: Iterable[ToolCall],
    tool_results: Iterable[ToolResult],
    context_estimate: ContextEstimate | None = None,
) -> TurnResult:
    return _complete_stopped_turn(
        session,
        turn_id=turn_id,
        stop=StopResult(
            reason=StopReason.COMPACTION_FAILED,
            message="Context compaction failed before a safe model request could be made.",
            error=error,
        ),
        final_response=None,
        usage=usage,
        model_request_count=model_request_count,
        tool_call_count=tool_call_count,
        tool_calls=tool_calls,
        tool_results=tool_results,
        context_estimate=context_estimate,
    )


def _complete_interrupted_turn(
    session: Session | None,
    *,
    turn_id: str,
    message: str,
    usage: Usage,
    model_request_count: int,
    tool_call_count: int,
    tool_calls: Iterable[ToolCall],
    tool_results: Iterable[ToolResult],
    incomplete_event_id: str | None = None,
    error: ErrorInfo | None = None,
    context_estimate: ContextEstimate | None = None,
) -> TurnResult:
    _append_turn_interrupted(
        session,
        turn_id=turn_id,
        message=message,
        incomplete_event_id=incomplete_event_id,
        error=error,
        usage=usage,
    )
    return _complete_stopped_turn(
        session,
        turn_id=turn_id,
        stop=StopResult(
            reason=StopReason.INTERRUPTED,
            message=message,
            error=error,
            details={"incomplete_event_id": incomplete_event_id},
        ),
        final_response=None,
        usage=usage,
        model_request_count=model_request_count,
        tool_call_count=tool_call_count,
        tool_calls=tool_calls,
        tool_results=tool_results,
        context_estimate=context_estimate,
    )


def _complete_stopped_turn(
    session: Session | None,
    *,
    turn_id: str,
    stop: StopResult,
    final_response: str | None,
    usage: Usage,
    model_request_count: int,
    tool_call_count: int,
    tool_calls: Iterable[ToolCall],
    tool_results: Iterable[ToolResult],
    context_estimate: ContextEstimate | None = None,
) -> TurnResult:
    stop_reason = StopReason.FINAL_RESPONSE if final_response is not None else stop.reason
    public_stop = None if final_response is not None else stop
    _append_turn_completed(
        session,
        turn_id=turn_id,
        stop_reason=stop_reason,
        final_response=final_response,
        usage=usage,
        model_request_count=model_request_count,
        tool_call_count=tool_call_count,
    )
    return TurnResult(
        turn_id=turn_id,
        final_response=final_response,
        stop_reason=stop_reason,
        stop=public_stop,
        usage=usage,
        context_estimate=context_estimate.model_copy(deep=True)
        if context_estimate is not None
        else None,
        tool_calls=_copy_tool_calls(tool_calls),
        tool_results=_copy_tool_results(tool_results),
        session_id=session.session_id if session is not None else None,
        session_state=session.state if session is not None else None,
        model_request_count=model_request_count,
        tool_call_count=tool_call_count,
    )


def _complete_final_result_turn(
    session: Session | None,
    *,
    turn_id: str,
    final_result: FinalResultOutput,
    usage: Usage,
    model_request_count: int,
    tool_call_count: int,
    tool_calls: Iterable[ToolCall],
    tool_results: Iterable[ToolResult],
    context_estimate: ContextEstimate | None = None,
) -> TurnResult:
    _append_turn_completed(
        session,
        turn_id=turn_id,
        stop_reason=StopReason.FINAL_RESULT,
        final_response=None,
        usage=usage,
        model_request_count=model_request_count,
        tool_call_count=tool_call_count,
    )
    return TurnResult(
        turn_id=turn_id,
        final_response=None,
        final_result=final_result.model_copy(deep=True),
        stop_reason=StopReason.FINAL_RESULT,
        stop=None,
        usage=usage,
        context_estimate=context_estimate.model_copy(deep=True)
        if context_estimate is not None
        else None,
        tool_calls=_copy_tool_calls(tool_calls),
        tool_results=_copy_tool_results(tool_results),
        session_id=session.session_id if session is not None else None,
        session_state=session.state if session is not None else None,
        model_request_count=model_request_count,
        tool_call_count=tool_call_count,
    )


def _append_turn_started(
    session: Session | None,
    *,
    turn_id: str,
    prompt: str,
    input_message_id: str,
) -> None:
    if session is None:
        return
    event = TurnStartedEvent(
        session_id=session.session_id,
        turn_id=turn_id,
        parent_event_id=session.last_event_id,
        sequence=session.next_sequence,
        payload=TurnStartedPayload(prompt=prompt, input_message_id=input_message_id),
    )
    session.append_event(event)


def _append_model_request_started(
    session: Session | None,
    *,
    turn_id: str,
    request: ModelRequest,
    attempt: int = 1,
) -> None:
    if session is None:
        return
    event = ModelRequestStartedEvent(
        session_id=session.session_id,
        turn_id=turn_id,
        parent_event_id=session.last_event_id,
        sequence=session.next_sequence,
        payload=ModelRequestStartedPayload(
            request_id=request.request_id,
            attempt=attempt,
            request=request,
        ),
    )
    session.append_event(event)


def _append_model_response_completed(
    session: Session | None,
    *,
    turn_id: str,
    request_id: str,
    response: ModelResponse,
    usage: Usage,
) -> None:
    if session is None:
        return
    event = ModelResponseCompletedEvent(
        session_id=session.session_id,
        turn_id=turn_id,
        parent_event_id=session.last_event_id,
        sequence=session.next_sequence,
        payload=ModelResponseCompletedPayload(
            request_id=request_id,
            response_id=response.response_id,
            response=response,
            usage=usage,
        ),
    )
    session.append_event(event)


def _append_model_request_failed(
    session: Session | None,
    *,
    turn_id: str,
    request_id: str,
    attempt: int = 1,
    error: ErrorInfo,
    usage: Usage,
    retryable: bool = False,
) -> None:
    if session is None:
        return
    event = ModelRequestFailedEvent(
        session_id=session.session_id,
        turn_id=turn_id,
        parent_event_id=session.last_event_id,
        sequence=session.next_sequence,
        payload=ModelRequestFailedPayload(
            request_id=request_id,
            attempt=attempt,
            error=error,
            retryable=retryable,
            usage=usage,
        ),
    )
    session.append_event(event)


def _append_compaction_started(
    session: Session | None,
    *,
    turn_id: str,
    compaction_id: str,
    reason: str,
    planned_message_ids: Sequence[str],
    plan: CompactionPlan,
    runtime_config: RuntimeConfig,
    context_estimate: ContextEstimate | None,
) -> None:
    if session is None:
        return
    event = CompactionStartedEvent(
        session_id=session.session_id,
        turn_id=turn_id,
        parent_event_id=session.last_event_id,
        sequence=session.next_sequence,
        payload=CompactionStartedPayload(
            compaction_id=compaction_id,
            reason=reason,
            planned_message_ids=list(planned_message_ids),
            metadata=_compaction_event_metadata(
                plan=plan,
                runtime_config=runtime_config,
                context_estimate=context_estimate,
            ),
        ),
    )
    session.append_event(event)


def _append_compaction_completed(
    session: Session | None,
    *,
    turn_id: str,
    result: GenericCompactionResult,
    runtime_config: RuntimeConfig,
) -> None:
    if session is None:
        return
    event = CompactionCompletedEvent(
        session_id=session.session_id,
        turn_id=turn_id,
        parent_event_id=session.last_event_id,
        sequence=session.next_sequence,
        payload=CompactionCompletedPayload(
            compaction_id=result.compaction_id,
            summary=result.summary,
            summary_message_id=result.summary_message.message_id,
            covered_message_ids=list(result.covered_message_ids),
            usage=result.usage,
            metadata=_compaction_completed_metadata(result, runtime_config=runtime_config),
        ),
    )
    session.append_event(event)


def _append_tool_call_started(
    session: Session | None,
    *,
    turn_id: str,
    tool_call: ToolCall,
) -> None:
    if session is None:
        return
    event = ToolCallStartedEvent(
        session_id=session.session_id,
        turn_id=turn_id,
        parent_event_id=session.last_event_id,
        sequence=session.next_sequence,
        payload=ToolCallStartedPayload(tool_call=tool_call),
    )
    session.append_event(event)


def _append_tool_call_completed(
    session: Session | None,
    *,
    turn_id: str,
    result: ToolResult,
) -> None:
    if session is None:
        return
    event = ToolCallCompletedEvent(
        session_id=session.session_id,
        turn_id=turn_id,
        parent_event_id=session.last_event_id,
        sequence=session.next_sequence,
        payload=ToolCallCompletedPayload(result=result),
    )
    session.append_event(event)


def _append_turn_completed(
    session: Session | None,
    *,
    turn_id: str,
    stop_reason: StopReason,
    final_response: str | None,
    usage: Usage,
    model_request_count: int,
    tool_call_count: int,
) -> None:
    if session is None:
        return
    event = TurnCompletedEvent(
        session_id=session.session_id,
        turn_id=turn_id,
        parent_event_id=session.last_event_id,
        sequence=session.next_sequence,
        payload=TurnCompletedPayload(
            stop_reason=stop_reason,
            final_response=final_response,
            usage=usage,
            model_request_count=model_request_count,
            tool_call_count=tool_call_count,
        ),
    )
    session.append_event(event)


def _append_turn_interrupted(
    session: Session | None,
    *,
    turn_id: str,
    message: str,
    incomplete_event_id: str | None,
    error: ErrorInfo | None,
    usage: Usage,
) -> None:
    if session is None:
        return
    event = TurnInterruptedEvent(
        session_id=session.session_id,
        turn_id=turn_id,
        parent_event_id=session.last_event_id,
        sequence=session.next_sequence,
        payload=TurnInterruptedPayload(
            message=message,
            incomplete_event_id=incomplete_event_id,
            error=error,
            usage=usage,
        ),
    )
    session.append_event(event)


def _validate_json_compatible_model[ModelT: BaseModel](
    model_type: type[ModelT], value: object
) -> ModelT:
    if isinstance(value, model_type):
        return value
    if isinstance(value, Mapping):
        data = json.dumps(value, separators=(",", ":"))
        return model_type.model_validate_json(data)
    return model_type.model_validate(value)


def _tool_event_callback(
    session: Session | None,
    *,
    turn_id: str,
) -> ToolEventCallback | None:
    if session is None:
        return None

    async def callback(event_type: str, payload: JsonObject) -> None:
        if event_type == TOOL_CALL_STARTED_EVENT:
            raw_tool_call = payload.get("tool_call")
            if not isinstance(raw_tool_call, dict):
                raise ValueError("tool start event payload is missing serialized tool_call")
            _append_tool_call_started(
                session,
                turn_id=turn_id,
                tool_call=_validate_json_compatible_model(ToolCall, raw_tool_call),
            )
        elif event_type == TOOL_CALL_COMPLETED_EVENT:
            raw_result = payload.get("result")
            if not isinstance(raw_result, dict):
                raise ValueError("tool completion event payload is missing serialized result")
            _append_tool_call_completed(
                session,
                turn_id=turn_id,
                result=_validate_json_compatible_model(ToolResult, raw_result),
            )

    return callback


def _tool_map(tools: Iterable[Tool[Any]]) -> dict[str, Tool[Any]]:
    enabled: dict[str, Tool[Any]] = {}
    for tool in tools:
        if tool.name in enabled:
            raise ValueError(f"duplicate enabled tool name: {tool.name}")
        enabled[tool.name] = tool
    return enabled


def _partition_output_tool_calls(
    tool_calls: Iterable[ToolCall],
    enabled_tools: Mapping[str, Tool[Any]],
) -> tuple[list[ToolCall], list[ToolCall]]:
    output_calls: list[ToolCall] = []
    ordinary_calls: list[ToolCall] = []
    for call in tool_calls:
        if _is_output_tool_call(call, enabled_tools):
            output_calls.append(call)
        else:
            ordinary_calls.append(call)
    return (output_calls, ordinary_calls)


def _is_output_tool_call(call: ToolCall, enabled_tools: Mapping[str, Tool[Any]]) -> bool:
    tool = enabled_tools.get(call.tool_name)
    if tool is None:
        return False
    return (
        call.tool_name == _FINAL_RESULT_TOOL_NAME
        and tool.definition.metadata.get(_OUTPUT_TOOL_KIND_METADATA_KEY) == _OUTPUT_TOOL_KIND
    )


def _enabled_output_tool_name(enabled_tools: Mapping[str, Tool[Any]]) -> str | None:
    """Return the configured required output tool's name, if one is enabled.

    When an agent is configured with ``output: {tool_name: final_result, required: true}``,
    that tool is present in the enabled set tagged with the output kind. Its presence is what
    lets the turn loop force the tool on a terminal turn the model tried to end with prose.
    """
    tool = enabled_tools.get(_FINAL_RESULT_TOOL_NAME)
    if tool is None:
        return None
    if tool.definition.metadata.get(_OUTPUT_TOOL_KIND_METADATA_KEY) == _OUTPUT_TOOL_KIND:
        return _FINAL_RESULT_TOOL_NAME
    return None


def _force_final_result_message() -> UserMessage:
    return UserMessage(
        message_id=new_id("msg"),
        content=[TextContent(text=_FORCE_FINAL_RESULT_NUDGE)],
    )


def _sort_tool_calls_for_execution(tool_calls: Iterable[ToolCall]) -> list[ToolCall]:
    return [
        call
        for _input_index, call in sorted(
            enumerate(tool_calls), key=lambda item: (item[1].order, item[0])
        )
    ]


def _first_successful_final_result(results: Iterable[ToolResult]) -> FinalResultOutput | None:
    for result in results:
        if result.tool_name == _FINAL_RESULT_TOOL_NAME and result.success:
            return FinalResultOutput(
                tool_name=result.tool_name,
                tool_call_id=result.tool_call_id,
                output=result.output,
                arguments=_copy_json_object(result.arguments),
            )
    return None


def _sort_tool_results_for_response(
    results: Iterable[ToolResult],
    tool_calls: Sequence[ToolCall],
) -> list[ToolResult]:
    call_indexes = {call.call_id: index for index, call in enumerate(tool_calls)}
    return sorted(
        results,
        key=lambda result: (result.order, call_indexes.get(result.tool_call_id, len(call_indexes))),
    )


def _is_natural_completed_response(response: ModelResponse) -> bool:
    if response.provider_completion_status is not ProviderCompletionStatus.COMPLETED:
        return False
    if response.stop_reason in _NON_FINAL_RESPONSE_REASONS:
        return False
    native_stop_reason = _native_stop_reason(response)
    if native_stop_reason in _NON_FINAL_NATIVE_STOP_REASONS:
        return False
    if response.stop_reason is StopReason.FINAL_RESPONSE:
        return True
    if native_stop_reason is None:
        return True
    return native_stop_reason in _NATURAL_NATIVE_STOP_REASONS


def _native_stop_reason(response: ModelResponse) -> str | None:
    if response.provider_metadata is None:
        return None
    return response.provider_metadata.native_stop_reason


def _stop_reason_for_response(response: ModelResponse) -> StopReason:
    if response.stop_reason is not None and response.stop_reason is not StopReason.FINAL_RESPONSE:
        return response.stop_reason
    if response.provider_completion_status is ProviderCompletionStatus.FAILED:
        return StopReason.MODEL_ERROR
    return StopReason.PROVIDER_STOP_REASON


def _stop_for_non_final_response(response: ModelResponse) -> StopResult:
    details: JsonObject = {
        "response_id": response.response_id,
        "provider_completion_status": response.provider_completion_status.value,
        "tool_call_count": len(response.tool_calls),
    }
    if response.incomplete_details:
        details["incomplete_details"] = _copy_json_object(response.incomplete_details)
    if response.provider_metadata is not None:
        provider_metadata = response.provider_metadata.model_dump(mode="json")
        native_stop_reason = provider_metadata.get("native_stop_reason")
        if isinstance(native_stop_reason, str):
            details["native_stop_reason"] = native_stop_reason
    return StopResult(
        reason=_stop_reason_for_response(response),
        message="Model response did not contain final assistant text.",
        details=details,
    )


def _response_usage(
    usage: Usage,
    *,
    model_profile: ModelProfile | None,
) -> Usage:
    tokens = usage.tokens.model_copy(deep=True)

    cost = usage.cost.model_copy(deep=True) if usage.cost is not None else None
    if cost is None and model_profile is not None:
        cost = calculate_token_cost(tokens, model_profile.pricing)

    return usage_with_model_request_count(
        usage.model_copy(update={"tokens": tokens, "cost": cost}, deep=True)
    )


def _compaction_usage(usage: Usage) -> Usage:
    normalized = usage_with_model_request_count(usage)
    tokens = normalized.tokens.model_copy(deep=True)
    cost = normalized.cost.model_copy(deep=True) if normalized.cost is not None else None
    return normalized.model_copy(update={"tokens": tokens, "cost": cost}, deep=True)


def _should_retry_after_context_overflow(
    exc: Exception,
    *,
    runtime_config: RuntimeConfig,
    retry_used: bool,
) -> bool:
    return (
        runtime_config.compaction.enabled
        and runtime_config.compaction.trigger_on_context_overflow
        and not retry_used
        and _is_context_overflow_exception(exc)
    )


def _is_context_overflow_exception(exc: Exception) -> bool:
    for attribute in ("code", "error_type", "category"):
        value = getattr(exc, attribute, None)
        if isinstance(value, str) and value.lower() in _CONTEXT_OVERFLOW_CODES:
            return True
    message = str(exc).lower()
    return "context" in message and any(
        phrase in message
        for phrase in (
            "overflow",
            "too large",
            "too long",
            "maximum context length",
            "context_length_exceeded",
            "max context",
        )
    )


def _compaction_event_metadata(
    *,
    plan: CompactionPlan,
    runtime_config: RuntimeConfig,
    context_estimate: ContextEstimate | None,
) -> JsonObject:
    metadata: dict[str, object] = {
        "plan": plan.model_dump(mode="json"),
        "config": runtime_config.compaction.model_dump(mode="json"),
        "compact_start_index": plan.compact_start_index,
        "compact_end_index": plan.compact_end_index,
        "covered_message_ids": list(plan.compact_message_ids),
        "preserved_message_ids": list(plan.preserved_message_ids),
        "split_turn_prefix": plan.split_turn_prefix,
        "trigger_reasons": [reason.value for reason in plan.trigger_reasons],
    }
    if context_estimate is not None:
        metadata[CONTEXT_ESTIMATE_METADATA_KEY] = context_estimate.model_dump(mode="json")
    return _JSON_OBJECT_ADAPTER.validate_python(metadata)


def _compaction_completed_metadata(
    result: GenericCompactionResult,
    *,
    runtime_config: RuntimeConfig,
) -> JsonObject:
    metadata: dict[str, object] = {
        "request_id": result.request_id,
        "response_id": result.response_id,
        "plan": result.plan.model_dump(mode="json"),
        "config": runtime_config.compaction.model_dump(mode="json"),
        "compact_start_index": result.compact_start_index,
        "compact_end_index": result.compact_end_index,
        "covered_message_ids": list(result.covered_message_ids),
        "preserved_message_ids": list(result.preserved_message_ids),
        "split_turn_prefix": result.split_turn_prefix,
    }
    return _JSON_OBJECT_ADAPTER.validate_python(metadata)


def _compaction_reason(trigger_reasons: Sequence[object]) -> str:
    values: list[str] = []
    for reason in trigger_reasons:
        value = getattr(reason, "value", None)
        values.append(value if isinstance(value, str) else str(reason))
    if not values:
        return "manual"
    return ",".join(values)


def _estimate_context_for_request(
    *,
    messages: Sequence[ModelMessage],
    tool_schemas: Sequence[JsonObject],
    runtime_config: RuntimeConfig,
    model_profile: ModelProfile | None,
    reasoning: ReasoningSettings | None,
) -> ContextEstimate | None:
    if not runtime_config.usage.estimate_context_tokens:
        return None
    return estimate_context(
        messages=messages,
        tools=tool_schemas,
        reasoning=reasoning,
        profile=model_profile,
        config=runtime_config.usage.token_estimator,
    )


def _tool_schemas(tools: Iterable[Tool[Any]]) -> tuple[JsonObject, ...]:
    return tuple(_tool_schema(tool) for tool in tools)


def _tool_schema(tool: Tool[Any]) -> JsonObject:
    definition = tool.definition
    return _copy_json_object(
        {
            "name": definition.name,
            "description": definition.description,
            "arguments_schema": deepcopy(definition.arguments_schema),
        }
    )


def _copy_model_messages(messages: Iterable[ModelMessage]) -> list[ModelMessage]:
    return [message.model_copy(deep=True) for message in messages]


def _copy_tool_calls(tool_calls: Iterable[ToolCall]) -> list[ToolCall]:
    return [tool_call.model_copy(deep=True) for tool_call in tool_calls]


def _copy_tool_results(tool_results: Iterable[ToolResult]) -> list[ToolResult]:
    return [tool_result.model_copy(deep=True) for tool_result in tool_results]


def _copy_json_object(value: JsonObject) -> JsonObject:
    return _JSON_OBJECT_ADAPTER.validate_python(deepcopy(value))


def _cancellation_message(state: CancellationState, *, boundary: str) -> str:
    if state.reason is None:
        return f"Turn interrupted by cooperative cancellation at {boundary}."
    return f"Turn interrupted by cooperative cancellation at {boundary}: {state.reason}"


def _error_info_from_exception(exc: Exception) -> ErrorInfo:
    return ErrorInfo(
        code="model_error",
        message=f"{type(exc).__name__}: {exc}",
        details={
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
        },
    )


def _error_info_from_compaction_exception(exc: CompactionError) -> ErrorInfo:
    details: JsonObject = {
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
    }
    if exc.__cause__ is not None:
        details["cause_type"] = type(exc.__cause__).__name__
        details["cause_message"] = str(exc.__cause__)
    return ErrorInfo(
        code="compaction_failed",
        message=f"{type(exc).__name__}: {exc}",
        details=details,
    )


def _error_info_from_base_exception(exc: BaseException) -> ErrorInfo:
    return ErrorInfo(
        code="interrupted",
        message=f"{type(exc).__name__}: {exc}",
        details={
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
        },
    )


__all__ = ("run_turn",)
