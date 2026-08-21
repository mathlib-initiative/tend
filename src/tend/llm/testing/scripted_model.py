"""Deterministic scripted model adapter for tests."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable

from tend.llm.models.base import ModelAdapter
from tend.llm.models.profiles import ModelProfile
from tend.llm.models.requests import ModelRequest, ModelResponse

type ScriptedModelStep = ModelResponse | Exception


class ScriptExhaustedError(AssertionError):
    """Raised when a scripted model receives more requests than scripted steps."""


class ScriptedModel(ModelAdapter):
    """Model adapter that returns predefined responses or raises predefined errors.

    The adapter records deep copies of received requests for assertions. Response
    steps are also returned as deep copies so tests and later turn-loop code can
    mutate their local response without changing the original script.
    """

    __slots__ = ("_link_response_request_id", "_profile", "_requests", "_steps")

    _link_response_request_id: bool
    _profile: ModelProfile | None
    _requests: list[ModelRequest]
    _steps: deque[ScriptedModelStep]

    def __init__(
        self,
        steps: Iterable[ScriptedModelStep] = (),
        *,
        profile: ModelProfile | None = None,
        link_response_request_id: bool = True,
    ) -> None:
        self._steps = deque(steps)
        self._profile = profile.model_copy(deep=True) if profile is not None else None
        self._requests = []
        self._link_response_request_id = link_response_request_id

    @property
    def profile(self) -> ModelProfile | None:
        """Return a defensive copy of scripted profile metadata, if configured."""

        if self._profile is None:
            return None
        return self._profile.model_copy(deep=True)

    @property
    def requests(self) -> tuple[ModelRequest, ...]:
        """Return defensive copies of requests received so far."""

        return tuple(request.model_copy(deep=True) for request in self._requests)

    @property
    def last_request(self) -> ModelRequest | None:
        """Return the most recent recorded request, if any."""

        if not self._requests:
            return None
        return self._requests[-1].model_copy(deep=True)

    @property
    def remaining_steps(self) -> int:
        """Return the number of unconsumed scripted response/error steps."""

        return len(self._steps)

    def append_response(self, response: ModelResponse) -> None:
        """Append one response step to the remaining script."""

        self._steps.append(response.model_copy(deep=True))

    def append_exception(self, exception: Exception) -> None:
        """Append one exception step to the remaining script."""

        self._steps.append(exception)

    def clear_requests(self) -> None:
        """Clear recorded requests without modifying remaining scripted steps."""

        self._requests.clear()

    async def generate(self, request: ModelRequest) -> ModelResponse:
        """Record ``request`` and consume the next scripted response/error step."""

        self._requests.append(request.model_copy(deep=True))
        if not self._steps:
            raise ScriptExhaustedError("scripted model has no remaining response steps")

        step = self._steps.popleft()
        if isinstance(step, Exception):
            raise step

        response = step.model_copy(deep=True)
        if self._link_response_request_id and response.request_id is None:
            response.request_id = request.request_id
        return response


__all__ = ("ScriptExhaustedError", "ScriptedModel", "ScriptedModelStep")
