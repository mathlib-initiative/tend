"""Provider-neutral model adapter protocol."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from tend.llm.models.profiles import ModelProfile
from tend.llm.models.requests import ModelRequest, ModelResponse


@runtime_checkable
class ModelAdapter(Protocol):
    """Async provider-neutral model adapter boundary.

    The shared turn loop depends on this protocol rather than concrete provider
    implementations. Provider adapters and deterministic tests all consume and
    return provider-neutral request/response schemas.
    """

    @property
    def profile(self) -> ModelProfile | None:
        """Return known model profile/capability metadata, if available."""
        ...

    async def generate(self, request: ModelRequest) -> ModelResponse:
        """Generate one provider-neutral model response for a request."""
        ...


__all__ = ("ModelAdapter",)
