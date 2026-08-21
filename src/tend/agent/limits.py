"""Centralized runtime-limit checks for the shared turn loop."""

from __future__ import annotations

import time
from collections.abc import Callable

from tend._common.types import StopReason
from tend.agent.config import RuntimeLimitsConfig
from tend.agent.results import StopResult
from tend.llm.usage import Usage

type MonotonicClock = Callable[[], float]


class TurnLimitTracker:
    """Evaluate turn limits against an injected monotonic clock and usage."""

    __slots__ = ("_clock", "_limits", "_started_at")

    _clock: MonotonicClock
    _limits: RuntimeLimitsConfig
    _started_at: float

    def __init__(
        self,
        limits: RuntimeLimitsConfig,
        *,
        clock: MonotonicClock | None = None,
    ) -> None:
        self._limits = limits
        self._clock = clock or time.monotonic
        self._started_at = self._clock()

    @property
    def elapsed_seconds(self) -> float:
        """Return non-negative elapsed wall-clock seconds for this turn."""

        return max(self._clock() - self._started_at, 0.0)

    def check_before_model_request(
        self,
        *,
        iteration_count: int,
        model_request_count: int,
        usage: Usage,
    ) -> StopResult | None:
        """Return a stop if another model request would violate a limit."""

        if (
            self._limits.max_iterations is not None
            and iteration_count >= self._limits.max_iterations
        ):
            return StopResult(
                reason=StopReason.MAX_ITERATIONS,
                message=(
                    "Turn stopped before another model request because "
                    "max_iterations was reached."
                ),
                details={
                    "max_iterations": self._limits.max_iterations,
                    "iteration_count": iteration_count,
                },
            )
        if (
            self._limits.max_model_requests is not None
            and model_request_count >= self._limits.max_model_requests
        ):
            return StopResult(
                reason=StopReason.MAX_MODEL_REQUESTS,
                message=(
                    "Turn stopped before another model request because "
                    "max_model_requests was reached."
                ),
                details={
                    "max_model_requests": self._limits.max_model_requests,
                    "model_request_count": model_request_count,
                },
            )
        return self.check_resource_limits(usage, boundary="before_model_request")

    def check_before_tool_execution(
        self,
        *,
        tool_call_count: int,
        requested_tool_count: int,
        usage: Usage,
    ) -> StopResult | None:
        """Return a stop if tool execution would violate a limit."""

        if (
            self._limits.max_tool_calls is not None
            and tool_call_count + requested_tool_count > self._limits.max_tool_calls
        ):
            return StopResult(
                reason=StopReason.MAX_TOOL_CALLS,
                message=(
                    "Turn stopped before tool execution because max_tool_calls "
                    "would be exceeded."
                ),
                details={
                    "max_tool_calls": self._limits.max_tool_calls,
                    "tool_call_count": tool_call_count,
                    "requested_tool_call_count": requested_tool_count,
                },
            )
        return self.check_resource_limits(usage, boundary="before_tool_execution")

    def check_resource_limits(self, usage: Usage, *, boundary: str) -> StopResult | None:
        """Return a stop for wall-time, token, or cost limits at a boundary."""

        wall_time_stop = self._check_wall_time(boundary=boundary)
        if wall_time_stop is not None:
            return wall_time_stop

        token_stop = self._check_tokens(usage, boundary=boundary)
        if token_stop is not None:
            return token_stop

        return self._check_cost(usage, boundary=boundary)

    def _check_wall_time(self, *, boundary: str) -> StopResult | None:
        elapsed = self.elapsed_seconds
        if elapsed < self._limits.max_wall_time_seconds:
            return None
        return StopResult(
            reason=StopReason.MAX_WALL_TIME,
            message="Turn stopped because max_wall_time_seconds was reached.",
            details={
                "boundary": boundary,
                "max_wall_time_seconds": self._limits.max_wall_time_seconds,
                "elapsed_seconds": elapsed,
            },
        )

    def _check_tokens(self, usage: Usage, *, boundary: str) -> StopResult | None:
        max_tokens = self._limits.max_tokens
        if max_tokens is None:
            return None
        total_tokens = total_reported_tokens(usage)
        if total_tokens < max_tokens:
            return None
        return StopResult(
            reason=StopReason.MAX_TOKENS,
            message="Turn stopped because max_tokens was reached.",
            details={
                "boundary": boundary,
                "max_tokens": max_tokens,
                "token_count": total_tokens,
            },
        )

    def _check_cost(self, usage: Usage, *, boundary: str) -> StopResult | None:
        max_cost = self._limits.max_cost
        if max_cost is None or usage.cost is None:
            return None
        if usage.cost.amount < max_cost:
            return None
        return StopResult(
            reason=StopReason.MAX_COST,
            message="Turn stopped because max_cost was reached.",
            details={
                "boundary": boundary,
                "max_cost": str(max_cost),
                "cost_amount": str(usage.cost.amount),
                "currency": usage.cost.currency,
            },
        )


def total_reported_tokens(usage: Usage) -> int:
    """Return a deterministic total across all reported token categories."""

    tokens = usage.tokens
    return (
        tokens.input_tokens
        + tokens.output_tokens
        + tokens.reasoning_tokens
        + tokens.cache_read_tokens
        + tokens.cache_write_tokens
        + sum(tokens.provider_details.values())
    )


__all__ = ("MonotonicClock", "TurnLimitTracker", "total_reported_tokens")
