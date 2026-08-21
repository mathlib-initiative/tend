"""Concrete ``bash`` built-in tool."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field, field_validator, model_validator

from tend._common.types import JsonObject, StrictModel
from tend.agent.tools.backends import ProcessBackend, ProcessResult, ToolPath
from tend.agent.tools.base import Tool
from tend.agent.tools.builtin._common import NonNegativeCount, OutputLimitBytes, TextToolOutput
from tend.agent.tools.context import ToolContext
from tend.agent.tools.local_backend import LocalProcessBackend
from tend.llm.models.tools import ToolError
from tend.llm.truncation import TruncationInfo, TruncationPolicy, truncate_tail

PositiveTimeoutSeconds = Annotated[float, Field(gt=0, le=600.0)]
NonNegativeDuration = Annotated[float, Field(ge=0)]
DEFAULT_BASH_TIMEOUT_SECONDS = 60.0
DEFAULT_BASH_OUTPUT_BYTES = 32_768


class BashArguments(StrictModel):
    """Arguments for executing one shell command.

    ``timeout_seconds`` is a reliability bound for this command. The tool passes
    the command directly to the configured process backend without allowlists,
    syntax blocking, path checks, or network restrictions; sandbox policy belongs
    to the process/orchestration sandbox boundary. ``max_output_bytes`` bounds
    each captured stream with tail truncation so long command output cannot
    silently consume model context.
    """

    command: str = Field(min_length=1)
    timeout_seconds: PositiveTimeoutSeconds | None = None
    max_output_bytes: OutputLimitBytes = DEFAULT_BASH_OUTPUT_BYTES

    @field_validator("command")
    @classmethod
    def _validate_nonblank_command(cls, command: str) -> str:
        if not command.strip():
            raise ValueError("command must not be blank")
        return command


class BashResult(TextToolOutput):
    """Structured ``bash`` result returned by the built-in handler."""

    command: str
    success: bool
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    stdout_size_bytes: NonNegativeCount = 0
    stderr_size_bytes: NonNegativeCount = 0
    duration_ms: NonNegativeDuration | None = None
    timed_out: bool = False
    timeout_seconds: PositiveTimeoutSeconds
    error: ToolError | None = None

    @model_validator(mode="after")
    def _validate_success_error_pair(self) -> BashResult:
        if self.success and self.error is not None:
            raise ValueError("successful bash results must not include an error")
        if not self.success and self.error is None:
            raise ValueError("failed bash results must include an error")
        if self.timed_out and self.error is None:
            raise ValueError("timed-out bash results must include an error")
        return self


async def _run_bash(context: ToolContext, arguments: BashArguments) -> BashResult:
    backend, cwd = _process_backend_and_cwd(context)
    timeout_seconds = arguments.timeout_seconds or DEFAULT_BASH_TIMEOUT_SECONDS

    try:
        process_result = await backend.run(
            arguments.command,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
        )
    except Exception as exc:  # Backend failures become model-visible bash results.
        message = f"{type(exc).__name__}: {exc}"
        return _backend_error_result(
            arguments,
            timeout_seconds=timeout_seconds,
            message=message,
            details={
                "command": arguments.command,
                "cwd": str(cwd),
                "timeout_seconds": timeout_seconds,
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
            },
        )

    stdout, stdout_truncated, stdout_truncation = _tail_truncate_stream(
        process_result.stdout,
        max_bytes=arguments.max_output_bytes,
    )
    stderr, stderr_truncated, stderr_truncation = _tail_truncate_stream(
        process_result.stderr,
        max_bytes=arguments.max_output_bytes,
    )
    output = _format_output(process_result, stdout=stdout, stderr=stderr)
    truncated = stdout_truncated or stderr_truncated
    truncation = _aggregate_truncation(
        process_result,
        output=output,
        stdout_truncation=stdout_truncation,
        stderr_truncation=stderr_truncation,
    )

    if process_result.timed_out:
        error = ToolError(
            error_type="timeout",
            message=f"Command timed out after {timeout_seconds:g} seconds.",
            details={
                "command": arguments.command,
                "cwd": process_result.cwd,
                "timeout_seconds": timeout_seconds,
                "exit_code": process_result.exit_code,
            },
        )
        success = False
    else:
        error = None
        success = True

    return BashResult(
        command=arguments.command,
        success=success,
        exit_code=process_result.exit_code,
        stdout=stdout,
        stderr=stderr,
        stdout_truncated=stdout_truncated,
        stderr_truncated=stderr_truncated,
        stdout_size_bytes=_size_bytes(stdout),
        stderr_size_bytes=_size_bytes(stderr),
        duration_ms=process_result.duration_ms,
        timed_out=process_result.timed_out,
        timeout_seconds=timeout_seconds,
        output=output,
        truncated=truncated,
        truncation=truncation,
        error=error,
    )


def _process_backend_and_cwd(context: ToolContext) -> tuple[ProcessBackend, ToolPath]:
    if context.process_backend is not None:
        return context.process_backend, context.cwd
    return LocalProcessBackend(cwd=context.cwd), "."


def _tail_truncate_stream(
    text: str,
    *,
    max_bytes: int,
) -> tuple[str, bool, TruncationInfo | None]:
    result = truncate_tail(text, max_bytes=max_bytes)
    truncation = result.info if result.info.truncated else None
    return result.text, result.info.truncated, truncation


def _aggregate_truncation(
    result: ProcessResult,
    *,
    output: str,
    stdout_truncation: TruncationInfo | None,
    stderr_truncation: TruncationInfo | None,
) -> TruncationInfo | None:
    if stdout_truncation is None and stderr_truncation is None:
        return None

    original_output = _format_output(result, stdout=result.stdout, stderr=result.stderr)
    omitted_size_bytes = _sum_optional_ints(
        stdout_truncation.omitted_size_bytes if stdout_truncation is not None else None,
        stderr_truncation.omitted_size_bytes if stderr_truncation is not None else None,
    )
    omitted_line_count = _sum_optional_ints(
        stdout_truncation.omitted_line_count if stdout_truncation is not None else None,
        stderr_truncation.omitted_line_count if stderr_truncation is not None else None,
    )
    return TruncationInfo(
        truncated=True,
        policy=TruncationPolicy.TAIL,
        original_size_bytes=_size_bytes(original_output),
        original_line_count=_line_count(original_output),
        returned_size_bytes=_size_bytes(output),
        returned_line_count=_line_count(output),
        omitted_size_bytes=omitted_size_bytes,
        omitted_line_count=omitted_line_count,
    )


def _format_output(result: ProcessResult, *, stdout: str, stderr: str) -> str:
    lines = [
        f"Exit code: {_format_exit_code(result.exit_code)}",
        f"Timed out: {_format_bool(result.timed_out)}",
    ]
    if result.duration_ms is not None:
        lines.append(f"Duration: {result.duration_ms:.1f} ms")
    lines.extend(
        (
            "",
            "STDOUT:",
            stdout if stdout else "[empty]",
            "",
            "STDERR:",
            stderr if stderr else "[empty]",
        )
    )
    return "\n".join(lines)


def _backend_error_result(
    arguments: BashArguments,
    *,
    timeout_seconds: float,
    message: str,
    details: JsonObject,
) -> BashResult:
    error = ToolError(error_type="backend_error", message=message, details=details)
    output = f"[Bash tool error: {message}]"
    return BashResult(
        command=arguments.command,
        success=False,
        exit_code=None,
        stdout="",
        stderr="",
        stdout_truncated=False,
        stderr_truncated=False,
        stdout_size_bytes=0,
        stderr_size_bytes=0,
        duration_ms=None,
        timed_out=False,
        timeout_seconds=timeout_seconds,
        output=output,
        error=error,
    )


def _format_exit_code(exit_code: int | None) -> str:
    if exit_code is None:
        return "[unknown]"
    return str(exit_code)


def _format_bool(value: bool) -> str:
    if value:
        return "true"
    return "false"


def _sum_optional_ints(left: int | None, right: int | None) -> int | None:
    if left is None and right is None:
        return None
    return (left or 0) + (right or 0)


def _size_bytes(text: str) -> int:
    return len(text.encode("utf-8"))


def _line_count(text: str) -> int:
    return len(text.splitlines())


bash_tool: Tool[BashArguments] = Tool.from_arguments_model(
    name="bash",
    description=(
        "Execute a shell command in the configured working directory with stdout, stderr, "
        "exit-code, timeout, duration, and tail-truncation metadata. Nonzero exit codes "
        "are reported as command results rather than tool-framework failures. This tool "
        "does not filter commands, block shell syntax, restrict paths, or enforce network "
        "policy; sandbox policy belongs to the process/orchestration sandbox boundary."
    ),
    arguments_model=BashArguments,
    handler=_run_bash,
    default_timeout_seconds=DEFAULT_BASH_TIMEOUT_SECONDS,
    default_output_limit_bytes=DEFAULT_BASH_OUTPUT_BYTES,
    metadata={"built_in": True},
)


__all__ = (
    "BashArguments",
    "BashResult",
    "DEFAULT_BASH_OUTPUT_BYTES",
    "DEFAULT_BASH_TIMEOUT_SECONDS",
    "bash_tool",
)
