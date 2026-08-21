"""Concrete ``copy_lines`` built-in tool.

Copy a contiguous range of lines from one file into another **byte-exact**,
without the model having to reproduce the copied text. This is the safe
primitive for splitting an oversized source file into smaller modules: create
the new module's header with ``write_file``, then ``copy_lines`` each declaration
block into it. The source file is never mutated; remove copied blocks later with
``edit_file`` once the copied module is reviewed.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from tend._common.types import JsonObject, StrictModel
from tend.agent.tools.backends import FilesystemBackend
from tend.agent.tools.base import Tool
from tend.agent.tools.builtin._common import NonNegativeCount, TextToolOutput, filesystem_backend
from tend.agent.tools.context import ToolContext
from tend.agent.tools.local_backend import LocalFilesystemBackend
from tend.llm.models.tools import ToolError

CopyLinesErrorType = Literal[
    "same_file",
    "source_not_found",
    "dest_not_found",
    "is_directory",
    "permission_denied",
    "binary",
    "non_utf8",
    "range_out_of_bounds",
    "dest_line_out_of_bounds",
    "read_error",
    "write_error",
    "encoding_error",
]


class CopyLinesArguments(StrictModel):
    """Arguments for a byte-exact cross-file line copy.

    Lines are 1-based and inclusive. The block ``source_path[start_line:end_line]``
    is inserted into ``dest_path`` immediately after ``dest_after_line``
    (``0`` prepends; ``<line count of dest>`` appends). Source and destination must
    be different files, including common path aliases. The source file is never
    modified. The tool does not enforce path allowlists; sandbox policy belongs to
    the process boundary.
    """

    source_path: str = Field(min_length=1)
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    dest_path: str = Field(min_length=1)
    dest_after_line: NonNegativeCount = 0

    @model_validator(mode="after")
    def _validate_range(self) -> CopyLinesArguments:
        if self.end_line < self.start_line:
            raise ValueError("end_line must be greater than or equal to start_line")
        return self


class CopyLinesResult(TextToolOutput):
    """Structured ``copy_lines`` result returned by the built-in handler."""

    source_path: str
    dest_path: str
    success: bool
    lines_copied: NonNegativeCount = 0
    source_lines_after: NonNegativeCount | None = None
    dest_lines_after: NonNegativeCount | None = None
    error: ToolError | None = None

    @model_validator(mode="after")
    def _validate_success_error_pair(self) -> CopyLinesResult:
        if self.success and self.error is not None:
            raise ValueError("successful copy results must not include an error")
        if not self.success and self.error is None:
            raise ValueError("failed copy results must include an error")
        return self


async def _run_copy_lines(context: ToolContext, arguments: CopyLinesArguments) -> CopyLinesResult:
    backend = filesystem_backend(context)

    if _same_file_path(
        _same_file_cwd(context, backend), arguments.source_path, arguments.dest_path
    ):
        return _error_result(
            arguments,
            error_type="same_file",
            message="copy_lines requires distinct source and destination files",
            details={"source_path": arguments.source_path, "dest_path": arguments.dest_path},
        )

    source_text = await _read_text(backend, arguments, arguments.source_path, which="source")
    if isinstance(source_text, CopyLinesResult):
        return source_text
    dest_text = await _read_text(backend, arguments, arguments.dest_path, which="dest")
    if isinstance(dest_text, CopyLinesResult):
        return dest_text

    source_lines = _split_keepends(source_text)
    dest_lines = _split_keepends(dest_text)

    if arguments.end_line > len(source_lines):
        return _error_result(
            arguments,
            error_type="range_out_of_bounds",
            message=(
                f"lines {arguments.start_line}-{arguments.end_line} exceed "
                f"{arguments.source_path} ({len(source_lines)} lines)"
            ),
            details={
                "source_path": arguments.source_path,
                "start_line": arguments.start_line,
                "end_line": arguments.end_line,
                "source_line_count": len(source_lines),
            },
        )
    if arguments.dest_after_line > len(dest_lines):
        return _error_result(
            arguments,
            error_type="dest_line_out_of_bounds",
            message=(
                f"dest_after_line {arguments.dest_after_line} exceeds "
                f"{arguments.dest_path} ({len(dest_lines)} lines)"
            ),
            details={
                "dest_path": arguments.dest_path,
                "dest_after_line": arguments.dest_after_line,
                "dest_line_count": len(dest_lines),
            },
        )

    block = source_lines[arguments.start_line - 1 : arguments.end_line]
    new_dest = (
        dest_lines[: arguments.dest_after_line] + block + dest_lines[arguments.dest_after_line :]
    )

    write_dest = await _write_text(backend, arguments.dest_path, "".join(new_dest), arguments)
    if write_dest is not None:
        return write_dest

    lines_copied = len(block)
    where = f"after line {arguments.dest_after_line} of {arguments.dest_path}"
    return CopyLinesResult(
        source_path=arguments.source_path,
        dest_path=arguments.dest_path,
        success=True,
        lines_copied=lines_copied,
        source_lines_after=len(source_lines),
        dest_lines_after=len(new_dest),
        output=(
            f"Copied {lines_copied} {_plural(lines_copied, 'line')} "
            f"({arguments.source_path}:{arguments.start_line}-{arguments.end_line}) {where}."
        ),
    )


def _same_file_cwd(context: ToolContext, backend: FilesystemBackend) -> Path:
    if isinstance(backend, LocalFilesystemBackend):
        return backend.cwd
    return context.cwd


def _same_file_path(cwd: Path, source_path: str, dest_path: str) -> bool:
    """Return whether two tool paths refer to the same local path.

    The lexical string comparison catches the exact case; ``samefile`` catches
    links when both files exist; the canonicalized form also catches aliases such
    as ``file`` vs ``./file`` and ``dir/../file`` for the local backend. Custom
    backends still receive the conservative same-string and normalized-path
    protection before any destination write occurs.
    """

    if source_path == dest_path:
        return True

    source = _local_path(cwd, source_path)
    dest = _local_path(cwd, dest_path)
    try:
        if source.samefile(dest):
            return True
    except OSError:
        pass
    return _canonical_path(source) == _canonical_path(dest)


def _local_path(cwd: Path, path: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = cwd / candidate
    return candidate


def _canonical_path(candidate: Path) -> str:
    try:
        resolved = candidate.resolve(strict=False)
    except OSError:
        resolved = candidate.absolute()
    return os.path.normcase(os.path.normpath(str(resolved)))


def _split_keepends(text: str) -> list[str]:
    """Split on ``\n`` boundaries, keeping each line's trailing newline.

    Unlike ``str.splitlines``, only ``\n`` (and the ``\r`` in a ``\r\n`` pair)
    is treated as a line boundary, so other code bytes are never misread as line
    breaks and reassembly via ``"".join`` is exact.
    """

    if text == "":
        return []
    lines = text.split("\n")
    out = [f"{line}\n" for line in lines[:-1]]
    if lines[-1] != "":
        out.append(lines[-1])
    return out


async def _read_text(
    backend: FilesystemBackend,
    arguments: CopyLinesArguments,
    path: str,
    *,
    which: Literal["source", "dest"],
) -> str | CopyLinesResult:
    try:
        data = await backend.read_bytes(path)
    except Exception as exc:  # noqa: BLE001 - classified below
        return _path_error(arguments, path, exc, which=which)
    if b"\x00" in data:
        return _error_result(
            arguments,
            error_type="binary",
            message=f"Binary file cannot be copied as UTF-8 text: {path}",
            details={"path": path, "which": which, "size_bytes": len(data)},
        )
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        return _error_result(
            arguments,
            error_type="non_utf8",
            message=f"File is not valid UTF-8: {path} ({exc})",
            details={"path": path, "which": which, "encoding": "utf-8"},
        )


async def _write_text(
    backend: FilesystemBackend, path: str, text: str, arguments: CopyLinesArguments
) -> CopyLinesResult | None:
    try:
        encoded = text.encode("utf-8")
    except UnicodeEncodeError as exc:
        return _error_result(
            arguments,
            error_type="encoding_error",
            message=f"Content for {path} could not be encoded as UTF-8: {exc}",
            details={"path": path},
        )
    try:
        await backend.write_bytes(path, encoded, create_parents=False)
    except Exception as exc:  # noqa: BLE001 - classified below
        return _error_result(
            arguments,
            error_type=_classify_write_error(exc),
            message=f"{type(exc).__name__}: {exc}",
            details={"path": path, "operation": "write"},
        )
    return None


# The error helpers below build a fully-formed failure CopyLinesResult; the
# caller short-circuits on it (mirrors edit_file's _error_result pattern).
def _path_error(
    arguments: CopyLinesArguments,
    path: str,
    exc: Exception,
    *,
    which: Literal["source", "dest"],
) -> CopyLinesResult:
    error_type: CopyLinesErrorType
    if isinstance(exc, FileNotFoundError):
        error_type = "source_not_found" if which == "source" else "dest_not_found"
        message = f"{which.capitalize()} file not found: {path}"
    elif isinstance(exc, IsADirectoryError):
        error_type = "is_directory"
        message = f"Path is a directory: {path}"
    elif isinstance(exc, PermissionError):
        error_type = "permission_denied"
        message = f"Permission denied: {path}"
    else:
        error_type = "read_error"
        message = f"{type(exc).__name__}: {exc}"
    return _error_result(
        arguments,
        error_type=error_type,
        message=message,
        details={"path": path, "which": which, "operation": "read"},
    )


def _classify_write_error(exc: Exception) -> CopyLinesErrorType:
    if isinstance(exc, IsADirectoryError):
        return "is_directory"
    if isinstance(exc, PermissionError):
        return "permission_denied"
    if isinstance(exc, UnicodeError):
        return "encoding_error"
    return "write_error"


def _error_result(
    arguments: CopyLinesArguments,
    *,
    error_type: CopyLinesErrorType,
    message: str,
    details: JsonObject,
) -> CopyLinesResult:
    return CopyLinesResult(
        source_path=arguments.source_path,
        dest_path=arguments.dest_path,
        success=False,
        lines_copied=0,
        output=f"[copy_lines error: {message}]",
        error=ToolError(error_type=error_type, message=message, details=details),
    )


def _plural(count: int, singular: str) -> str:
    if count == 1:
        return singular
    return f"{singular}s"


copy_lines_tool: Tool[CopyLinesArguments] = Tool.from_arguments_model(
    name="copy_lines",
    description=(
        "Copy a contiguous, 1-based inclusive range of lines from one UTF-8 text "
        "file into another, byte-exact, without reproducing the text. The block "
        "source_path[start_line:end_line] is inserted into dest_path immediately "
        "after dest_after_line (0 prepends; the destination's line count appends). "
        "The source file is never modified; delete copied blocks later with "
        "edit_file after review. Source and destination must differ. Ideal for "
        "splitting an oversized file into smaller modules: write the new module's "
        "header, then copy declaration blocks into it. Does not enforce path "
        "allowlists; sandbox policy belongs to the process boundary."
    ),
    arguments_model=CopyLinesArguments,
    handler=_run_copy_lines,
    metadata={"built_in": True},
)


__all__ = (
    "CopyLinesArguments",
    "CopyLinesResult",
    "copy_lines_tool",
)
