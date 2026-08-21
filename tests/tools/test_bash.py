from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from tend._common.types import JsonObject
from tend.agent.tools import ToolContext, get_builtin_tool
from tend.agent.tools.backends import ProcessResult, ToolPath
from tend.agent.tools.builtin.bash import DEFAULT_BASH_TIMEOUT_SECONDS, BashResult


class ScriptedProcessBackend:
    __slots__ = ("calls", "results")

    calls: list[tuple[str, str, float | None]]
    results: dict[str, ProcessResult]

    def __init__(self, results: dict[str, ProcessResult]) -> None:
        self.results = dict(results)
        self.calls = []

    async def run(
        self,
        command: str,
        *,
        cwd: ToolPath = ".",
        timeout_seconds: float | None = None,
    ) -> ProcessResult:
        self.calls.append((command, str(cwd), timeout_seconds))
        return self.results[command]


class FailingProcessBackend:
    async def run(
        self,
        command: str,
        *,
        cwd: ToolPath = ".",
        timeout_seconds: float | None = None,
    ) -> ProcessResult:
        _ = command, cwd, timeout_seconds
        raise OSError("spawn failed")


async def test_bash_successful_command_with_stdout_uses_process_backend(tmp_path: Path) -> None:
    tool = get_builtin_tool("bash")
    backend = ScriptedProcessBackend(
        {
            "printf ok": ProcessResult(
                command="printf ok",
                cwd=str(tmp_path),
                exit_code=0,
                stdout="ok\n",
                stderr="",
                duration_ms=2.5,
            )
        }
    )

    result = await tool.run(
        ToolContext(cwd=tmp_path, process_backend=backend),
        tool.validate_arguments({"command": "printf ok"}),
    )

    assert isinstance(result, BashResult)
    assert result.success is True
    assert result.error is None
    assert result.exit_code == 0
    assert result.stdout == "ok\n"
    assert result.stderr == ""
    assert result.timed_out is False
    assert result.duration_ms == 2.5
    assert "Exit code: 0" in result.output
    assert "STDOUT:\nok\n" in result.output
    assert backend.calls == [("printf ok", str(tmp_path), DEFAULT_BASH_TIMEOUT_SECONDS)]


async def test_bash_stderr_and_nonzero_exit_are_command_results(tmp_path: Path) -> None:
    tool = get_builtin_tool("bash")
    backend = ScriptedProcessBackend(
        {
            "bad": ProcessResult(
                command="bad",
                cwd=str(tmp_path),
                exit_code=7,
                stdout="partial out",
                stderr="problem\n",
                timed_out=False,
                duration_ms=3.0,
            )
        }
    )

    result = await tool.run(
        ToolContext(cwd=tmp_path, process_backend=backend),
        tool.validate_arguments({"command": "bad", "timeout_seconds": 1.5}),
    )

    assert isinstance(result, BashResult)
    assert result.success is True
    assert result.error is None
    assert result.exit_code == 7
    assert result.stdout == "partial out"
    assert result.stderr == "problem\n"
    assert "Exit code: 7" in result.output
    assert "STDERR:\nproblem\n" in result.output
    assert backend.calls == [("bad", str(tmp_path), 1.5)]


async def test_bash_timeout_becomes_structured_model_visible_error(tmp_path: Path) -> None:
    tool = get_builtin_tool("bash")

    result = await tool.run(
        ToolContext(cwd=tmp_path),
        tool.validate_arguments({"command": "sleep 5", "timeout_seconds": 0.05}),
    )

    assert isinstance(result, BashResult)
    assert result.success is False
    assert result.timed_out is True
    assert result.error is not None
    assert result.error.error_type == "timeout"
    assert "timed out" in result.error.message
    assert "Timed out: true" in result.output
    assert result.exit_code is not None


async def test_bash_uses_tail_truncation_for_long_output(tmp_path: Path) -> None:
    tool = get_builtin_tool("bash")
    backend = ScriptedProcessBackend(
        {
            "long": ProcessResult(
                command="long",
                cwd=str(tmp_path),
                exit_code=0,
                stdout="first line\n" + ("0123456789" * 10),
                stderr="",
                duration_ms=1.0,
            )
        }
    )

    result = await tool.run(
        ToolContext(cwd=tmp_path, process_backend=backend),
        tool.validate_arguments({"command": "long", "max_output_bytes": 12}),
    )

    assert isinstance(result, BashResult)
    assert result.success is True
    assert result.truncated is True
    assert result.stdout_truncated is True
    assert result.stderr_truncated is False
    assert result.truncation is not None
    assert result.truncation.policy == "tail"
    assert result.stdout.startswith("[Output truncated:")
    assert result.stdout.endswith("890123456789")
    assert "[Output truncated:" in result.output


async def test_bash_backend_exception_becomes_structured_error() -> None:
    tool = get_builtin_tool("bash")

    result = await tool.run(
        ToolContext(process_backend=FailingProcessBackend()),
        tool.validate_arguments({"command": "printf ok"}),
    )

    assert isinstance(result, BashResult)
    assert result.success is False
    assert result.error is not None
    assert result.error.error_type == "backend_error"
    assert "spawn failed" in result.error.message
    assert result.error.details["command"] == "printf ok"
    assert "[Bash tool error:" in result.output


async def test_bash_does_not_filter_shell_syntax(tmp_path: Path) -> None:
    tool = get_builtin_tool("bash")

    result = await tool.run(
        ToolContext(cwd=tmp_path),
        tool.validate_arguments({"command": "printf left && printf right", "timeout_seconds": 5.0}),
    )

    assert isinstance(result, BashResult)
    assert result.success is True
    assert result.exit_code == 0
    assert result.stdout == "leftright"


@pytest.mark.parametrize(
    "arguments",
    (
        {},
        {"command": ""},
        {"command": "   "},
        {"command": "printf ok", "unexpected": True},
        {"command": "printf ok", "timeout_seconds": 0},
        {"command": "printf ok", "timeout_seconds": 601.0},
        {"command": "printf ok", "max_output_bytes": 0},
        {"command": 1},
        {"command": "printf ok", "timeout_seconds": "1"},
    ),
)
def test_bash_invalid_arguments_fail_at_tool_validation_layer(arguments: JsonObject) -> None:
    tool = get_builtin_tool("bash")

    with pytest.raises(ValidationError):
        tool.validate_arguments(arguments)
