"""Concrete ``ls`` built-in tool."""

from __future__ import annotations

from pydantic import Field

from tend._common.types import StrictModel
from tend.agent.tools.backends import DirectoryEntry
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


class LsArguments(StrictModel):
    """Arguments for listing one directory.

    This v1 tool lists exactly one directory through the configured filesystem
    backend. It does not read ignore files or skip noisy directories; use the
    ``path`` argument and output limits to narrow the listing.
    """

    path: str = Field(default=".", min_length=1)
    max_entries: BoundedCount = DEFAULT_MAX_RESULTS
    max_output_bytes: OutputLimitBytes = DEFAULT_MAX_OUTPUT_BYTES


class LsResult(TextToolOutput):
    """Structured ``ls`` result returned by the built-in handler."""

    path: str
    total_entries: NonNegativeCount
    returned_entries: NonNegativeCount


async def _run_ls(context: ToolContext, arguments: LsArguments) -> LsResult:
    backend = filesystem_backend(context)
    entries = tuple(sorted(await backend.list_dir(arguments.path), key=lambda entry: entry.name))
    output = _format_listing(entries)
    text, truncated, truncation = head_truncated_text(
        output,
        max_lines=arguments.max_entries,
        max_bytes=arguments.max_output_bytes,
    )
    return LsResult(
        path=arguments.path,
        total_entries=len(entries),
        returned_entries=min(len(entries), arguments.max_entries),
        output=text,
        truncated=truncated,
        truncation=truncation,
    )


def _format_listing(entries: tuple[DirectoryEntry, ...]) -> str:
    if not entries:
        return "[No entries]"
    return "\n".join(_format_entry(entry) for entry in entries)


def _format_entry(entry: DirectoryEntry) -> str:
    if entry.is_symlink:
        kind = "symlink"
    elif entry.is_dir:
        kind = "dir"
    elif entry.is_file:
        kind = "file"
    else:
        kind = "other"

    display_name = f"{entry.name}/" if entry.is_dir else entry.name
    size = "-" if entry.is_dir or entry.size_bytes is None else f"{entry.size_bytes} bytes"
    return f"{kind}\t{display_name}\t{size}"


ls_tool: Tool[LsArguments] = Tool.from_arguments_model(
    name="ls",
    description=(
        "List one directory with deterministic bounded output. Results are sorted by name. "
        "This v1 tool does not read ignore files or skip noisy directories; narrow the path "
        "or lower max_entries/max_output_bytes when needed."
    ),
    arguments_model=LsArguments,
    handler=_run_ls,
    default_output_limit_bytes=DEFAULT_MAX_OUTPUT_BYTES,
    metadata={"built_in": True},
)


__all__ = ("LsArguments", "LsResult", "ls_tool")
