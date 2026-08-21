"""Concrete ``glob`` built-in tool."""

from __future__ import annotations

from pydantic import Field

from tend._common.types import StrictModel
from tend.agent.tools.base import Tool
from tend.agent.tools.builtin._common import (
    DEFAULT_MAX_OUTPUT_BYTES,
    DEFAULT_MAX_RESULTS,
    BoundedCount,
    NonNegativeCount,
    OutputLimitBytes,
    TextToolOutput,
    filesystem_backend,
    head_truncated_text,
)
from tend.agent.tools.context import ToolContext


class GlobArguments(StrictModel):
    """Arguments for deterministic glob search.

    The pattern is interpreted by the configured filesystem backend. This v1
    tool does not read ignore files or skip high-noise directories automatically;
    use ``root`` and a narrow ``pattern`` for bounded searches.
    """

    pattern: str = Field(min_length=1)
    root: str = Field(default=".", min_length=1)
    max_results: BoundedCount = DEFAULT_MAX_RESULTS
    max_output_bytes: OutputLimitBytes = DEFAULT_MAX_OUTPUT_BYTES


class GlobResult(TextToolOutput):
    """Structured ``glob`` result returned by the built-in handler."""

    pattern: str
    root: str
    total_matches: NonNegativeCount
    returned_matches: NonNegativeCount


async def _run_glob(context: ToolContext, arguments: GlobArguments) -> GlobResult:
    backend = filesystem_backend(context)
    matches = tuple(sorted(await backend.glob(arguments.pattern, root=arguments.root)))
    output = "\n".join(matches) if matches else "[No matches]"
    text, truncated, truncation = head_truncated_text(
        output,
        max_lines=arguments.max_results,
        max_bytes=arguments.max_output_bytes,
    )
    return GlobResult(
        pattern=arguments.pattern,
        root=arguments.root,
        total_matches=len(matches),
        returned_matches=min(len(matches), arguments.max_results),
        output=text,
        truncated=truncated,
        truncation=truncation,
    )


glob_tool: Tool[GlobArguments] = Tool.from_arguments_model(
    name="glob",
    description=(
        "Find paths using a backend glob pattern with deterministic bounded output. Results "
        "are sorted. This v1 tool does not read ignore files or skip noisy directories; narrow "
        "root/pattern or lower max_results/max_output_bytes when needed."
    ),
    arguments_model=GlobArguments,
    handler=_run_glob,
    default_output_limit_bytes=DEFAULT_MAX_OUTPUT_BYTES,
    metadata={"built_in": True},
)


__all__ = ("GlobArguments", "GlobResult", "glob_tool")
