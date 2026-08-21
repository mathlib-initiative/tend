"""Concrete ``grep`` built-in tool."""

from __future__ import annotations

import re
from collections.abc import Iterable

from pydantic import Field, field_validator

from tend._common.types import StrictModel
from tend.agent.tools.backends import FilesystemBackend
from tend.agent.tools.base import Tool
from tend.agent.tools.builtin._common import (
    DEFAULT_MAX_SEARCH_FILES,
    DEFAULT_MAX_SEARCH_MATCHES,
    DEFAULT_MAX_SEARCH_OUTPUT_BYTES,
    BoundedCount,
    NonNegativeCount,
    OutputLimitBytes,
    TextToolOutput,
    filesystem_backend,
    head_truncated_text,
)
from tend.agent.tools.context import ToolContext


class GrepArguments(StrictModel):
    """Arguments for regex search over UTF-8 text files.

    ``pattern`` uses Python regular-expression syntax. If ``path`` is a
    directory, ``glob`` selects files below that path and defaults to ``**/*``.
    This v1 tool does not read ignore files or skip noisy directories; narrow the
    path/glob and limits when needed.
    """

    pattern: str = Field(min_length=1)
    path: str = Field(default=".", min_length=1)
    glob: str | None = Field(default=None, min_length=1)
    case_sensitive: bool = True
    max_files: BoundedCount = DEFAULT_MAX_SEARCH_FILES
    max_matches: BoundedCount = DEFAULT_MAX_SEARCH_MATCHES
    max_output_bytes: OutputLimitBytes = DEFAULT_MAX_SEARCH_OUTPUT_BYTES

    @field_validator("pattern")
    @classmethod
    def _validate_regex_pattern(cls, pattern: str) -> str:
        try:
            re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"invalid regular expression: {exc}") from exc
        return pattern


class GrepResult(TextToolOutput):
    """Structured ``grep`` result returned by the built-in handler."""

    pattern: str
    path: str
    glob: str | None = None
    total_candidate_files: NonNegativeCount
    searched_files: NonNegativeCount
    total_matches: NonNegativeCount
    returned_matches: NonNegativeCount
    file_limit_reached: bool = False
    omitted_non_utf8_files: tuple[str, ...] = ()


async def _run_grep(context: ToolContext, arguments: GrepArguments) -> GrepResult:
    backend = filesystem_backend(context)
    candidates = await _candidate_files(backend, arguments)
    searched = candidates[: arguments.max_files]
    flags = 0 if arguments.case_sensitive else re.IGNORECASE
    regex = re.compile(arguments.pattern, flags=flags)

    matches: list[str] = []
    omitted_non_utf8: list[str] = []
    for path in searched:
        try:
            text = await backend.read_text(path, encoding="utf-8")
        except UnicodeDecodeError:
            omitted_non_utf8.append(path)
            continue
        matches.extend(_format_matches(path, text, regex))

    output = "\n".join(matches) if matches else "[No matches]"
    text, truncated, truncation = head_truncated_text(
        output,
        max_lines=arguments.max_matches,
        max_bytes=arguments.max_output_bytes,
    )
    return GrepResult(
        pattern=arguments.pattern,
        path=arguments.path,
        glob=arguments.glob,
        total_candidate_files=len(candidates),
        searched_files=len(searched),
        total_matches=len(matches),
        returned_matches=min(len(matches), arguments.max_matches),
        file_limit_reached=len(candidates) > len(searched),
        omitted_non_utf8_files=tuple(omitted_non_utf8),
        output=text,
        truncated=truncated,
        truncation=truncation,
    )


async def _candidate_files(
    backend: FilesystemBackend,
    arguments: GrepArguments,
) -> tuple[str, ...]:
    stat = await backend.stat(arguments.path)
    if stat.is_file:
        return (arguments.path,)
    if not stat.is_dir:
        return ()

    pattern = arguments.glob or "**/*"
    matches = tuple(sorted(await backend.glob(pattern, root=arguments.path)))
    files: list[str] = []
    for match in matches:
        try:
            match_stat = await backend.stat(match)
        except FileNotFoundError:
            continue
        if match_stat.is_file:
            files.append(match)
    return tuple(sorted(files))


def _format_matches(path: str, text: str, regex: re.Pattern[str]) -> Iterable[str]:
    for line_number, line in enumerate(text.splitlines(), start=1):
        if regex.search(line):
            yield f"{path}:{line_number}:{line}"


grep_tool: Tool[GrepArguments] = Tool.from_arguments_model(
    name="grep",
    description=(
        "Search UTF-8 text files with a regular expression and deterministic bounded output. "
        "Matches are sorted by path and line number. This v1 tool does not read ignore files "
        "or skip noisy directories; narrow path/glob or lower max_files/max_matches/"
        "max_output_bytes when needed."
    ),
    arguments_model=GrepArguments,
    handler=_run_grep,
    default_output_limit_bytes=DEFAULT_MAX_SEARCH_OUTPUT_BYTES,
    metadata={"built_in": True},
)


__all__ = ("GrepArguments", "GrepResult", "grep_tool")
