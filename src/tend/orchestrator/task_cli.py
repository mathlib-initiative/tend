"""Command-line utilities for orchestrator task directories."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from enum import IntEnum
from pathlib import Path
from typing import NoReturn, TextIO, cast

from tend.orchestrator.task_validation import TaskValidationFailure, validate_task_directory


class TaskCliExitCode(IntEnum):
    """Process exit codes returned by ``tend-task``."""

    SUCCESS = 0
    VALIDATION_FAILED = 1
    CONFIGURATION_OR_USAGE = 2
    INTERNAL_SOFTWARE = 70


class TaskCliError(Exception):
    """Expected task CLI error with a stable error code for stderr output."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class _TaskArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise TaskCliError("cli_usage_error", message)


def main(argv: Sequence[str] | None = None) -> int:
    """Console-script entrypoint for ``tend-task``."""

    return run_task_cli(argv)


def run_task_cli(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    prog: str = "tend-task",
) -> int:
    """Parse CLI args, run one task command, and return an exit code."""

    args = tuple(sys.argv[1:] if argv is None else argv)
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr
    try:
        namespace = _build_parser(prog).parse_args(list(args))
        command = cast(str, namespace.command)
        if command == "verify":
            failure = _handle_verify(namespace, stdout=out)
            if failure is not None:
                _write_validation_failure(failure, stderr=err)
                return int(TaskCliExitCode.VALIDATION_FAILED)
            return int(TaskCliExitCode.SUCCESS)
        raise TaskCliError("cli_usage_error", f"unknown command: {command}")
    except TaskCliError as exc:
        _write_error(exc.code, str(exc), err)
        return int(TaskCliExitCode.CONFIGURATION_OR_USAGE)
    except OSError as exc:
        _write_error("filesystem_error", exc.strerror or str(exc), err)
        return int(TaskCliExitCode.CONFIGURATION_OR_USAGE)


def _handle_verify(
    namespace: argparse.Namespace,
    *,
    stdout: TextIO,
) -> TaskValidationFailure | None:
    task_dir = _required_path_arg(namespace, "task_dir").expanduser().resolve()
    if not task_dir.exists():
        raise TaskCliError("filesystem_error", f"task directory does not exist: {task_dir}")
    if not task_dir.is_dir():
        raise TaskCliError("filesystem_error", f"task path is not a directory: {task_dir}")

    failure = validate_task_directory(task_dir)
    if failure is not None:
        return failure
    stdout.write(f"task set is valid: {task_dir}\n")
    return None


def _build_parser(prog: str) -> argparse.ArgumentParser:
    parser = _TaskArgumentParser(
        prog=prog,
        description="Manage tend orchestrator task directories.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser(
        "verify",
        help="validate a directory of orchestrator task YAML files",
        description=(
            "Validate a task directory with the same strict task-file and "
            "dependency-graph checks used by orchestrated merge validation."
        ),
    )
    verify.add_argument(
        "task_dir",
        metavar="TASK_DIR",
        type=Path,
        help="directory containing task YAML files, usually <entrypoint>/tasks",
    )
    return parser


def _required_path_arg(namespace: argparse.Namespace, name: str) -> Path:
    value = getattr(namespace, name, None)
    if not isinstance(value, Path):
        raise TaskCliError("cli_usage_error", f"missing required path argument: {name}")
    return value


def _write_validation_failure(failure: TaskValidationFailure, *, stderr: TextIO) -> None:
    _write_error("task_validation_error", failure.summary, stderr)
    if failure.offending_paths:
        stderr.write("offending paths:\n")
        for path in failure.offending_paths:
            stderr.write(f"  {path}\n")
    if failure.detail and failure.detail != failure.summary:
        stderr.write("detail:\n")
        stderr.write(f"{failure.detail}\n")


def _write_error(code: str, message: str, stderr: TextIO) -> None:
    stderr.write(f"error[{code}]: {message}\n")


__all__ = ("TaskCliExitCode", "main", "run_task_cli")
