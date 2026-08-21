"""Concrete ``write_file`` built-in tool."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from tend._common.types import JsonObject, StrictModel
from tend.agent.tools.backends import FilesystemBackend
from tend.agent.tools.base import Tool
from tend.agent.tools.builtin._common import NonNegativeCount, TextToolOutput, filesystem_backend
from tend.agent.tools.context import ToolContext
from tend.llm.models.tools import ToolError

WriteFileErrorType = Literal[
    "encoding_error",
    "is_directory",
    "parent_not_found",
    "permission_denied",
    "write_error",
]


class WriteFileArguments(StrictModel):
    """Arguments for writing UTF-8 text to a file.

    ``create_parents`` defaults to true so nested output paths work without a
    separate directory-creation step. The tool does not ask for approval, filter
    paths, or enforce allowlists; sandbox policy belongs to the process boundary.
    """

    path: str = Field(min_length=1)
    content: str
    create_parents: bool = True


class WriteFileResult(TextToolOutput):
    """Structured ``write_file`` result returned by the built-in handler."""

    path: str
    success: bool
    bytes_written: NonNegativeCount = 0
    chars_written: NonNegativeCount = 0
    create_parents: bool = True
    overwritten: bool | None = None
    error: ToolError | None = None

    @model_validator(mode="after")
    def _validate_success_error_pair(self) -> WriteFileResult:
        if self.success and self.error is not None:
            raise ValueError("successful write results must not include an error")
        if not self.success and self.error is None:
            raise ValueError("failed write results must include an error")
        return self


async def _run_write_file(context: ToolContext, arguments: WriteFileArguments) -> WriteFileResult:
    backend = filesystem_backend(context)
    overwritten = await _existing_path_state(backend, arguments.path)

    try:
        encoded = arguments.content.encode("utf-8")
    except UnicodeEncodeError as exc:
        return _error_result(
            arguments,
            error_type="encoding_error",
            message=f"UnicodeEncodeError: {exc}",
            overwritten=overwritten,
        )

    try:
        await backend.write_text(
            arguments.path,
            arguments.content,
            encoding="utf-8",
            create_parents=arguments.create_parents,
        )
    except Exception as exc:  # Backend failures become model-visible write results.
        return _error_result(
            arguments,
            error_type=_classify_write_error(exc),
            message=f"{type(exc).__name__}: {exc}",
            overwritten=overwritten,
        )

    chars_written = len(arguments.content)
    bytes_written = len(encoded)
    return WriteFileResult(
        path=arguments.path,
        success=True,
        bytes_written=bytes_written,
        chars_written=chars_written,
        create_parents=arguments.create_parents,
        overwritten=overwritten,
        output=(
            f"Wrote {bytes_written} {_plural(bytes_written, 'byte')} "
            f"({chars_written} {_plural(chars_written, 'character')}) to {arguments.path}."
        ),
    )


async def _existing_path_state(
    backend: FilesystemBackend,
    path: str,
) -> bool | None:
    try:
        await backend.stat(path)
    except FileNotFoundError:
        return False
    except OSError:
        return None
    return True


def _classify_write_error(exc: Exception) -> WriteFileErrorType:
    if isinstance(exc, IsADirectoryError):
        return "is_directory"
    if isinstance(exc, FileNotFoundError):
        return "parent_not_found"
    if isinstance(exc, PermissionError):
        return "permission_denied"
    if isinstance(exc, UnicodeError):
        return "encoding_error"
    return "write_error"


def _error_result(
    arguments: WriteFileArguments,
    *,
    error_type: WriteFileErrorType,
    message: str,
    overwritten: bool | None,
) -> WriteFileResult:
    error = ToolError(
        error_type=error_type,
        message=message,
        details=_error_details(arguments, exception_message=message),
    )
    return WriteFileResult(
        path=arguments.path,
        success=False,
        bytes_written=0,
        chars_written=0,
        create_parents=arguments.create_parents,
        overwritten=overwritten,
        output=f"[File write error: {message}]",
        error=error,
    )


def _error_details(arguments: WriteFileArguments, *, exception_message: str) -> JsonObject:
    return {
        "path": arguments.path,
        "create_parents": arguments.create_parents,
        "exception_message": exception_message,
    }


def _plural(count: int, singular: str) -> str:
    if count == 1:
        return singular
    return f"{singular}s"


write_file_tool: Tool[WriteFileArguments] = Tool.from_arguments_model(
    name="write_file",
    description=(
        "Write UTF-8 text to a file, creating parent directories by default and "
        "overwriting existing files. Returns path, byte/character counts, overwrite "
        "metadata, and structured model-visible write errors. This tool does not "
        "ask for approval or enforce path allowlists; sandbox policy belongs to the "
        "process/orchestration sandbox boundary."
    ),
    arguments_model=WriteFileArguments,
    handler=_run_write_file,
    metadata={"built_in": True},
)


__all__ = ("WriteFileArguments", "WriteFileResult", "write_file_tool")
