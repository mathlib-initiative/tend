"""Shared helpers for concrete built-in tools."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field, model_validator

from tend._common.types import StrictModel
from tend.agent.tools.backends import FilesystemBackend
from tend.agent.tools.context import ToolContext
from tend.agent.tools.local_backend import LocalFilesystemBackend
from tend.llm.truncation import TruncationInfo, truncate_head

PositiveInt = Annotated[int, Field(ge=1)]
BoundedCount = Annotated[int, Field(ge=1, le=10_000)]
OutputLimitBytes = Annotated[int, Field(ge=1, le=1_000_000)]
NonNegativeCount = Annotated[int, Field(ge=0)]

DEFAULT_MAX_RESULTS = 200
DEFAULT_MAX_OUTPUT_BYTES = 16_384
DEFAULT_MAX_SEARCH_FILES = 1_000
DEFAULT_MAX_SEARCH_MATCHES = 200
DEFAULT_MAX_SEARCH_OUTPUT_BYTES = 32_768


class TextToolOutput(StrictModel):
    """Common model-visible text plus optional truncation metadata."""

    output: str
    truncated: bool = False
    truncation: TruncationInfo | None = None

    @model_validator(mode="after")
    def _validate_truncation_pair(self) -> TextToolOutput:
        if self.truncated and self.truncation is None:
            raise ValueError("truncated output must include truncation metadata")
        if self.truncation is not None and self.truncation.truncated != self.truncated:
            raise ValueError("truncated flag must match truncation metadata")
        return self


def filesystem_backend(context: ToolContext) -> FilesystemBackend:
    """Return the injected filesystem backend or a local backend for context.cwd."""

    if context.filesystem_backend is not None:
        return context.filesystem_backend
    return LocalFilesystemBackend(cwd=context.cwd)


def head_truncated_text(
    text: str,
    *,
    max_lines: int | None,
    max_bytes: int,
) -> tuple[str, bool, TruncationInfo | None]:
    """Apply head truncation and expose metadata only when truncation occurred."""

    result = truncate_head(text, max_lines=max_lines, max_bytes=max_bytes)
    truncation = result.info if result.info.truncated else None
    return result.text, result.info.truncated, truncation


__all__ = (
    "BoundedCount",
    "DEFAULT_MAX_OUTPUT_BYTES",
    "DEFAULT_MAX_RESULTS",
    "DEFAULT_MAX_SEARCH_FILES",
    "DEFAULT_MAX_SEARCH_MATCHES",
    "DEFAULT_MAX_SEARCH_OUTPUT_BYTES",
    "NonNegativeCount",
    "OutputLimitBytes",
    "PositiveInt",
    "TextToolOutput",
    "filesystem_backend",
    "head_truncated_text",
)
