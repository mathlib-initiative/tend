"""Concrete ``read_file`` built-in tool."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from tend._common.types import StrictModel
from tend.agent.tools.base import Tool
from tend.agent.tools.builtin._common import (
    DEFAULT_MAX_OUTPUT_BYTES,
    BoundedCount,
    NonNegativeCount,
    OutputLimitBytes,
    PositiveInt,
    TextToolOutput,
    filesystem_backend,
    head_truncated_text,
)
from tend.agent.tools.context import ToolContext

ReadFileOmissionReason = Literal[
    "binary",
    "non_utf8",
    "not_found",
    "is_directory",
    "read_error",
    "offset_out_of_range",
]


class ReadFileArguments(StrictModel):
    """Arguments for reading a UTF-8 text file.

    ``offset`` is a 1-based line number. ``limit`` bounds the number of file
    lines selected from that offset. ``max_output_bytes`` is an additional model
    visibility limit applied with head truncation. The tool does not enforce path
    allowlists or file-extension restrictions; sandbox policy belongs to the
    process boundary.
    """

    path: str = Field(min_length=1)
    offset: PositiveInt | None = None
    limit: BoundedCount | None = None
    max_output_bytes: OutputLimitBytes = DEFAULT_MAX_OUTPUT_BYTES


class ReadFileResult(TextToolOutput):
    """Structured ``read_file`` result returned by the built-in handler."""

    path: str
    offset: PositiveInt
    limit: BoundedCount | None = None
    size_bytes: NonNegativeCount | None = None
    total_lines: NonNegativeCount | None = None
    returned_lines: NonNegativeCount = 0
    start_line: NonNegativeCount = 0
    end_line: NonNegativeCount = 0
    has_more: bool = False
    continuation_offset: PositiveInt | None = None
    omitted: bool = False
    omission_reason: ReadFileOmissionReason | None = None

    @model_validator(mode="after")
    def _validate_omission_reason(self) -> ReadFileResult:
        if self.omitted and self.omission_reason is None:
            raise ValueError("omitted read results must include an omission reason")
        if not self.omitted and self.omission_reason is not None:
            raise ValueError("non-omitted read results must not include an omission reason")
        if self.continuation_offset is not None and not self.has_more:
            raise ValueError("continuation_offset requires has_more=true")
        return self


async def _run_read_file(context: ToolContext, arguments: ReadFileArguments) -> ReadFileResult:
    backend = filesystem_backend(context)
    offset = arguments.offset or 1

    try:
        data = await backend.read_bytes(arguments.path)
    except FileNotFoundError:
        return _omitted_result(
            arguments,
            offset=offset,
            reason="not_found",
            output=f"[File read error: file not found: {arguments.path}]",
        )
    except IsADirectoryError:
        return _omitted_result(
            arguments,
            offset=offset,
            reason="is_directory",
            output=f"[File read error: path is a directory: {arguments.path}]",
        )
    except OSError as exc:
        return _omitted_result(
            arguments,
            offset=offset,
            reason="read_error",
            output=f"[File read error: {type(exc).__name__}: {exc}]",
        )

    size_bytes = len(data)
    if b"\x00" in data:
        return _omitted_result(
            arguments,
            offset=offset,
            reason="binary",
            output=f"[Binary file omitted: {arguments.path}]",
            size_bytes=size_bytes,
        )

    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return _omitted_result(
            arguments,
            offset=offset,
            reason="non_utf8",
            output=f"[Non-UTF-8 file omitted: {arguments.path}]",
            size_bytes=size_bytes,
        )

    lines = _normalize_lines(text)
    total_lines = len(lines)
    if total_lines == 0:
        if offset > 1:
            return _range_omitted_result(
                arguments,
                offset=offset,
                total_lines=total_lines,
                size_bytes=size_bytes,
            )
        return ReadFileResult(
            path=arguments.path,
            offset=offset,
            limit=arguments.limit,
            size_bytes=size_bytes,
            total_lines=0,
            returned_lines=0,
            start_line=0,
            end_line=0,
            output="",
        )

    start_index = offset - 1
    if start_index >= total_lines:
        return _range_omitted_result(
            arguments,
            offset=offset,
            total_lines=total_lines,
            size_bytes=size_bytes,
        )

    selected_lines = lines[start_index:]
    if arguments.limit is not None:
        selected_lines = selected_lines[: arguments.limit]

    selected_text = "\n".join(selected_lines)
    output, truncated, truncation = head_truncated_text(
        selected_text,
        max_lines=None,
        max_bytes=arguments.max_output_bytes,
    )
    returned_lines = _returned_file_line_count(
        selected_count=len(selected_lines),
        truncated=truncated,
        omitted_line_count=truncation.omitted_line_count if truncation is not None else None,
    )
    end_line = offset + returned_lines - 1 if returned_lines > 0 else 0
    has_more_lines = start_index + returned_lines < total_lines
    has_more = has_more_lines or truncated
    continuation_offset = offset + returned_lines if has_more_lines and returned_lines > 0 else None

    return ReadFileResult(
        path=arguments.path,
        offset=offset,
        limit=arguments.limit,
        size_bytes=size_bytes,
        total_lines=total_lines,
        returned_lines=returned_lines,
        start_line=offset if returned_lines > 0 else 0,
        end_line=end_line,
        has_more=has_more,
        continuation_offset=continuation_offset,
        output=output,
        truncated=truncated,
        truncation=truncation,
    )


def _normalize_lines(text: str) -> list[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    if normalized == "":
        return []
    lines = normalized.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def _returned_file_line_count(
    *,
    selected_count: int,
    truncated: bool,
    omitted_line_count: int | None,
) -> int:
    if not truncated or omitted_line_count is None:
        return selected_count
    return max(selected_count - omitted_line_count, 0)


def _range_omitted_result(
    arguments: ReadFileArguments,
    *,
    offset: int,
    total_lines: int,
    size_bytes: int,
) -> ReadFileResult:
    plural = "line" if total_lines == 1 else "lines"
    return _omitted_result(
        arguments,
        offset=offset,
        reason="offset_out_of_range",
        output=(
            f"[File read error: offset {offset} is beyond EOF for {arguments.path}; "
            f"file has {total_lines} {plural}.]"
        ),
        size_bytes=size_bytes,
        total_lines=total_lines,
    )


def _omitted_result(
    arguments: ReadFileArguments,
    *,
    offset: int,
    reason: ReadFileOmissionReason,
    output: str,
    size_bytes: int | None = None,
    total_lines: int | None = None,
) -> ReadFileResult:
    return ReadFileResult(
        path=arguments.path,
        offset=offset,
        limit=arguments.limit,
        size_bytes=size_bytes,
        total_lines=total_lines,
        returned_lines=0,
        start_line=0,
        end_line=0,
        output=output,
        omitted=True,
        omission_reason=reason,
    )


read_file_tool: Tool[ReadFileArguments] = Tool.from_arguments_model(
    name="read_file",
    description=(
        "Read UTF-8 text from a file with 1-based line pagination and explicit "
        "omission/truncation metadata. Arguments are path, optional offset, optional "
        "limit, and max_output_bytes. Binary or non-UTF-8 content is omitted rather "
        "than returned as raw bytes. This tool does not enforce path allowlists or "
        "file-extension restrictions; sandbox policy belongs to the process/orchestration "
        "sandbox boundary."
    ),
    arguments_model=ReadFileArguments,
    handler=_run_read_file,
    default_output_limit_bytes=DEFAULT_MAX_OUTPUT_BYTES,
    metadata={"built_in": True},
)


__all__ = ("ReadFileArguments", "ReadFileResult", "read_file_tool")
