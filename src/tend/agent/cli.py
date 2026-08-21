"""Direct one-turn console entrypoint for tend-agent."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sys
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import IntEnum
from pathlib import Path
from types import FrameType
from typing import Any, NoReturn, Protocol, TextIO, cast

from pydantic import ValidationError

from tend._common.config_files import ConfigFileError, read_config_model
from tend._common.errors import ConfigurationError, ErrorInfo, FrameworkError
from tend._common.types import JsonObject, StopReason
from tend.agent.agent import Agent
from tend.agent.cancellation import CancellationState
from tend.agent.config import (
    AgentConfig,
    AgentModelConfig,
    CompactionConfigOverrides,
    ModelRequestOverridesPatch,
    ResolvedConfig,
    RuntimeConfig,
    RuntimeConfigOverrides,
    RuntimeLimitsOverrides,
    resolve_config,
)
from tend.agent.results import TurnResult
from tend.agent.session import Session
from tend.llm.models.base import ModelAdapter
from tend.llm.models.profiles import ProviderApi
from tend.llm.providers import AnthropicMessagesAdapter, OpenAIResponsesAdapter

type _SignalHandler = signal.Handlers | int | Callable[[int, FrameType | None], Any] | None


class ExitCode(IntEnum):
    """Documented CLI process exit codes.

    0 means a final response/result was printed/serialized. 1 means the turn reached a
    structured non-final stop. 2 is configuration or command-line usage, 70 is
    an internal/framework software error, and 130 is SIGINT/SIGTERM interruption.
    """

    FINAL_RESPONSE = 0
    NON_FINAL_STOP = 1
    CONFIGURATION_OR_USAGE = 2
    INTERNAL_SOFTWARE = 70
    INTERRUPTED = 130


class ModelAdapterFactory(Protocol):
    """Factory seam used by tests and by the default provider selection path."""

    def __call__(
        self,
        model_config: AgentModelConfig,
        runtime_config: RuntimeConfig,
        environment: Mapping[str, str],
    ) -> ModelAdapter:
        """Return a provider-neutral adapter for the resolved config."""
        ...


@dataclass(frozen=True, slots=True)
class CliRunOptions:
    """Parsed one-turn CLI options."""

    agent_path: Path
    config_path: Path | None = None
    prompt: str | None = None
    json_output: bool = False
    cwd: str | None = None
    session_dir: str | None = None
    session_id: str | None = None
    resume_session: bool = False
    max_iterations: int | None = None
    max_model_requests: int | None = None
    max_tool_calls: int | None = None
    max_wall_time_seconds: float | None = None
    max_tokens: int | None = None
    max_cost: Decimal | None = None
    model_base_url: str | None = None
    model_timeout_seconds: float | None = None
    disable_compaction: bool = False


class CliRunnerError(FrameworkError):
    """Configuration or usage error at the CLI boundary."""

    __slots__ = ("code", "details")

    code: str
    details: JsonObject

    def __init__(self, code: str, message: str, *, details: JsonObject | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}


class _CliArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise CliRunnerError("cli_usage_error", message)


def main(argv: Sequence[str] | None = None) -> int:
    """Console-script entrypoint."""

    return asyncio.run(run_cli(argv))


async def run_cli(
    argv: Sequence[str] | None = None,
    *,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    environment: Mapping[str, str] | None = None,
    model_factory: ModelAdapterFactory | None = None,
    handle_signals: bool = True,
) -> int:
    """Parse CLI args, run one turn, write output, and return an exit code."""

    out = stdout or sys.stdout
    err = stderr or sys.stderr
    cancellation = CancellationState()
    try:
        options = parse_cli_args(sys.argv[1:] if argv is None else argv)
        with _installed_signal_handlers(cancellation, enabled=handle_signals):
            result = await run_cli_turn(
                options,
                stdin=stdin or sys.stdin,
                environment=os.environ if environment is None else environment,
                model_factory=model_factory,
                cancellation=cancellation,
            )
    except CliRunnerError as exc:
        _write_error(_error_info_from_cli_error(exc), err)
        return int(ExitCode.CONFIGURATION_OR_USAGE)
    except ConfigurationError as exc:
        _write_error(_error_info("configuration_error", str(exc)), err)
        return int(ExitCode.CONFIGURATION_OR_USAGE)
    except ValidationError as exc:
        _write_error(
            _error_info(
                "configuration_error",
                f"configuration validation failed: {_validation_error_summary(exc)}",
            ),
            err,
        )
        return int(ExitCode.CONFIGURATION_OR_USAGE)
    except asyncio.CancelledError:
        _write_error(_interrupted_error_info(cancellation), err)
        return int(ExitCode.INTERRUPTED)
    except KeyboardInterrupt:
        cancellation.cancel("received SIGINT")
        _write_error(_interrupted_error_info(cancellation), err)
        return int(ExitCode.INTERRUPTED)
    except FrameworkError as exc:
        _write_error(_error_info("framework_error", str(exc)), err)
        return int(ExitCode.INTERNAL_SOFTWARE)

    _write_turn_result(result, json_output=options.json_output, stdout=out, stderr=err)
    return int(_exit_code_for_turn_result(result))


async def run_cli_turn(
    options: CliRunOptions,
    *,
    stdin: TextIO,
    environment: Mapping[str, str],
    model_factory: ModelAdapterFactory | None = None,
    cancellation: CancellationState | None = None,
) -> TurnResult:
    """Load config, construct runtime objects, and run exactly one agent turn."""

    resolved = load_cli_config(options)
    prompt = _prompt_from_sources(options.prompt, resolved.runtime.prompt, stdin)
    adapter_factory = model_factory or default_model_adapter_factory
    adapter = adapter_factory(
        resolved.agent.model,
        resolved.runtime,
        _allowed_environment(resolved.runtime, environment),
    )
    agent = Agent.from_config(resolved.agent, model=adapter)
    session = _open_or_create_session(options, resolved.runtime)
    cancellation_state = cancellation or CancellationState()
    try:
        return await agent.run_turn(
            prompt,
            session=session,
            config=resolved.runtime,
            cancellation=cancellation_state,
        )
    finally:
        if session is not None:
            session.close()


def parse_cli_args(argv: Sequence[str]) -> CliRunOptions:
    """Parse one-turn CLI arguments."""

    parser = _build_cli_parser()
    namespace = parser.parse_args(list(argv))
    max_cost = _optional_decimal_arg(namespace, "max_cost")
    return CliRunOptions(
        agent_path=Path(_required_str_arg(namespace, "agent")),
        config_path=_optional_path_arg(namespace, "config_path"),
        prompt=_optional_str_arg(namespace, "prompt"),
        json_output=_bool_arg(namespace, "json_output"),
        cwd=_optional_str_arg(namespace, "cwd"),
        session_dir=_optional_str_arg(namespace, "session_dir"),
        session_id=_optional_str_arg(namespace, "session_id"),
        resume_session=_bool_arg(namespace, "resume_session"),
        max_iterations=_optional_int_arg(namespace, "max_iterations"),
        max_model_requests=_optional_int_arg(namespace, "max_model_requests"),
        max_tool_calls=_optional_int_arg(namespace, "max_tool_calls"),
        max_wall_time_seconds=_optional_float_arg(namespace, "max_wall_time_seconds"),
        max_tokens=_optional_int_arg(namespace, "max_tokens"),
        max_cost=max_cost,
        model_base_url=_optional_str_arg(namespace, "model_base_url"),
        model_timeout_seconds=_optional_float_arg(namespace, "model_timeout_seconds"),
        disable_compaction=_bool_arg(namespace, "disable_compaction"),
    )


def load_cli_config(options: CliRunOptions) -> ResolvedConfig:
    """Load agent/runtime config files and apply CLI overrides."""

    agent_config = load_agent_config(options.agent_path)
    cfg = load_runtime_config_overrides(options.config_path) if options.config_path else None
    cli_overrides = runtime_overrides_from_options(options)
    return resolve_config(agent_config, cfg=cfg, cli_overrides=cli_overrides)


def load_agent_config(path: str | Path) -> AgentConfig:
    """Load and validate durable agent YAML/JSON config."""

    config_path = Path(path)
    try:
        return read_config_model(config_path, AgentConfig, kind="agent config")
    except ConfigFileError as exc:
        raise CliRunnerError(
            "configuration_error",
            str(exc),
            details={"path": str(config_path), "kind": "agent config"},
        ) from exc
    except ValidationError as exc:
        raise CliRunnerError(
            "configuration_error",
            f"invalid agent config {config_path}: {_validation_error_summary(exc)}",
            details={"path": str(config_path), "kind": "agent config"},
        ) from exc


def load_runtime_config_overrides(path: str | Path) -> RuntimeConfigOverrides:
    """Load and validate optional runtime YAML/JSON override data."""

    config_path = Path(path)
    try:
        return read_config_model(config_path, RuntimeConfigOverrides, kind="runtime config")
    except ConfigFileError as exc:
        raise CliRunnerError(
            "configuration_error",
            str(exc),
            details={"path": str(config_path), "kind": "runtime config"},
        ) from exc
    except ValidationError as exc:
        raise CliRunnerError(
            "configuration_error",
            f"invalid runtime config {config_path}: {_validation_error_summary(exc)}",
            details={"path": str(config_path), "kind": "runtime config"},
        ) from exc


def runtime_overrides_from_options(options: CliRunOptions) -> RuntimeConfigOverrides:
    """Translate parsed CLI flags into sparse runtime overrides."""

    data: dict[str, object] = {}
    if options.prompt is not None:
        data["prompt"] = options.prompt
    if options.cwd is not None:
        data["cwd"] = options.cwd
    if options.session_dir is not None:
        data["session_dir"] = options.session_dir

    limit_data: dict[str, object] = {}
    _set_if_not_none(limit_data, "max_iterations", options.max_iterations)
    _set_if_not_none(limit_data, "max_model_requests", options.max_model_requests)
    _set_if_not_none(limit_data, "max_tool_calls", options.max_tool_calls)
    _set_if_not_none(limit_data, "max_wall_time_seconds", options.max_wall_time_seconds)
    _set_if_not_none(limit_data, "max_tokens", options.max_tokens)
    _set_if_not_none(limit_data, "max_cost", options.max_cost)
    if limit_data:
        data["limits"] = RuntimeLimitsOverrides.model_validate(limit_data)

    model_data: dict[str, object] = {}
    _set_if_not_none(model_data, "base_url", options.model_base_url)
    _set_if_not_none(model_data, "timeout_seconds", options.model_timeout_seconds)
    if model_data:
        data["model"] = ModelRequestOverridesPatch.model_validate(model_data)

    if options.disable_compaction:
        data["compaction"] = CompactionConfigOverrides(enabled=False)

    return RuntimeConfigOverrides.model_validate(data)


def default_model_adapter_factory(
    model_config: AgentModelConfig,
    runtime_config: RuntimeConfig,
    environment: Mapping[str, str],
) -> ModelAdapter:
    """Construct the concrete provider adapter requested by config."""

    if model_config.api is ProviderApi.OPENAI_RESPONSES:
        return OpenAIResponsesAdapter.from_config(
            model_config,
            runtime_config.to_provider_runtime_config(),
            environment=environment,
        )
    if model_config.api is ProviderApi.ANTHROPIC_MESSAGES:
        return AnthropicMessagesAdapter.from_config(
            model_config,
            runtime_config.to_provider_runtime_config(),
            environment=environment,
        )
    raise ConfigurationError(f"unsupported provider API: {model_config.api.value}")


def _build_cli_parser() -> _CliArgumentParser:
    parser = _CliArgumentParser(
        prog="tend-agent",
        description="Run one tend agent turn.",
        epilog=(
            "Exit codes: 0 final response, 1 structured non-final stop, "
            "2 configuration/usage error, 70 internal software error, "
            "130 interrupted."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--agent", required=True, help="Path to agent.yaml (JSON also accepted).")
    parser.add_argument(
        "--config",
        dest="config_path",
        help="Path to optional cfg.yaml runtime config (JSON also accepted).",
    )
    parser.add_argument("--prompt", help="User prompt for this one turn.")
    parser.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="Emit TurnResult JSON.",
    )
    parser.add_argument("--cwd", help="Working directory used by tools.")
    parser.add_argument("--session-dir", help="Session directory to create or resume.")
    parser.add_argument("--session-id", help="Optional session ID when creating a new session.")
    parser.add_argument(
        "--resume-session",
        action="store_true",
        help="Open --session-dir as an existing writable session.",
    )
    parser.add_argument("--max-iterations", type=int, help="Maximum turn-loop iterations.")
    parser.add_argument("--max-model-requests", type=int, help="Maximum model requests.")
    parser.add_argument("--max-tool-calls", type=int, help="Maximum tool calls.")
    parser.add_argument(
        "--max-wall-time-seconds",
        type=float,
        help="Maximum wall-clock runtime in seconds.",
    )
    parser.add_argument("--max-tokens", type=int, help="Maximum total model tokens.")
    parser.add_argument("--max-cost", help="Maximum monetary cost as a decimal amount.")
    parser.add_argument("--model-base-url", help="Provider base URL override.")
    parser.add_argument(
        "--model-timeout-seconds",
        type=float,
        help="Provider request timeout in seconds.",
    )
    parser.add_argument(
        "--no-compaction",
        dest="disable_compaction",
        action="store_true",
        help="Disable generic compaction for this turn.",
    )
    return parser


def _prompt_from_sources(cli_prompt: str | None, config_prompt: str | None, stdin: TextIO) -> str:
    if cli_prompt is not None:
        return _validate_prompt(cli_prompt, source="--prompt")
    if config_prompt is not None:
        return _validate_prompt(config_prompt, source="runtime config prompt")
    return _validate_prompt(stdin.read(), source="stdin")


def _validate_prompt(prompt: str, *, source: str) -> str:
    if not prompt.strip():
        raise CliRunnerError("cli_usage_error", f"prompt from {source} must be non-empty")
    return prompt


def _allowed_environment(
    runtime_config: RuntimeConfig,
    environment: Mapping[str, str],
) -> dict[str, str]:
    return {
        name: environment[name]
        for name in runtime_config.allowed_environment_names()
        if name in environment
    }


def _open_or_create_session(
    options: CliRunOptions,
    runtime_config: RuntimeConfig,
) -> Session | None:
    if runtime_config.session_dir is None:
        if options.resume_session:
            raise CliRunnerError(
                "cli_usage_error",
                "--resume-session requires --session-dir or runtime session_dir",
            )
        if options.session_id is not None:
            raise CliRunnerError("cli_usage_error", "--session-id requires --session-dir")
        return None

    if options.resume_session:
        return Session.resume(runtime_config.session_dir)
    return Session.create(
        runtime_config.session_dir,
        session_id=options.session_id,
        cwd=runtime_config.cwd,
    )


def _write_turn_result(
    result: TurnResult,
    *,
    json_output: bool,
    stdout: TextIO,
    stderr: TextIO,
) -> None:
    if json_output:
        stdout.write(result.model_dump_json() + "\n")
        return
    if result.final_response is not None:
        if result.final_response:
            stdout.write(result.final_response)
            if not result.final_response.endswith("\n"):
                stdout.write("\n")
        return
    if result.final_result is not None:
        stdout.write(json.dumps(result.final_result.output, sort_keys=True, separators=(",", ":")))
        stdout.write("\n")
        return
    message = "turn stopped"
    if result.stop is not None and result.stop.message is not None:
        message = result.stop.message
    stderr.write(f"{message} ({result.stop_reason.value})\n")


def _exit_code_for_turn_result(result: TurnResult) -> ExitCode:
    if result.stop_reason in {StopReason.FINAL_RESPONSE, StopReason.FINAL_RESULT}:
        return ExitCode.FINAL_RESPONSE
    if result.stop_reason is StopReason.INTERRUPTED:
        return ExitCode.INTERRUPTED
    return ExitCode.NON_FINAL_STOP


@contextmanager
def _installed_signal_handlers(
    cancellation: CancellationState,
    *,
    enabled: bool,
) -> Generator[None]:
    if not enabled:
        yield
        return

    current_task = asyncio.current_task()
    if current_task is None:
        yield
        return

    previous_handlers: dict[signal.Signals, _SignalHandler] = {}

    def handler(signum: int, frame: FrameType | None) -> None:
        del frame
        signal_name = _signal_name(signum)
        cancellation.cancel(f"received {signal_name}")
        current_task.cancel()

    try:
        for signum in _handled_signals():
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, handler)
    except (ValueError, OSError):
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)
        yield
        return

    try:
        yield
    finally:
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)


def _handled_signals() -> tuple[signal.Signals, ...]:
    values = [signal.SIGINT]
    if hasattr(signal, "SIGTERM"):
        values.append(signal.SIGTERM)
    return tuple(values)


def _signal_name(signum: int) -> str:
    try:
        return signal.Signals(signum).name
    except ValueError:
        return f"signal {signum}"


def _interrupted_error_info(cancellation: CancellationState) -> ErrorInfo:
    message = "Turn interrupted."
    if cancellation.reason is not None:
        message = f"Turn interrupted: {cancellation.reason}."
    return _error_info("interrupted", message)


def _write_error(error: ErrorInfo, stderr: TextIO) -> None:
    stderr.write(error.model_dump_json() + "\n")


def _error_info_from_cli_error(error: CliRunnerError) -> ErrorInfo:
    return _error_info(error.code, str(error), details=error.details)


def _error_info(code: str, message: str, *, details: JsonObject | None = None) -> ErrorInfo:
    return ErrorInfo(code=code, message=message or code, details=details or {})


def _validation_error_summary(error: ValidationError) -> str:
    errors = error.errors(include_url=False, include_context=False, include_input=False)
    if not errors:
        return "validation failed"
    first = cast(Mapping[str, object], errors[0])
    loc = first.get("loc")
    loc_text = _validation_location_text(loc)
    raw_message = first.get("msg")
    message = raw_message if isinstance(raw_message, str) else "invalid value"
    return f"{len(errors)} validation error(s); first at {loc_text}: {message}"


def _validation_location_text(location: object) -> str:
    if isinstance(location, list | tuple):
        parts = [str(part) for part in cast(Sequence[object], location)]
        return ".".join(parts) if parts else "<root>"
    if location is None:
        return "<root>"
    return str(location)


def _set_if_not_none(data: dict[str, object], key: str, value: object | None) -> None:
    if value is not None:
        data[key] = value


def _optional_path_arg(namespace: argparse.Namespace, name: str) -> Path | None:
    value = _optional_str_arg(namespace, name)
    if value is None:
        return None
    return Path(value)


def _required_str_arg(namespace: argparse.Namespace, name: str) -> str:
    value = cast(object, getattr(namespace, name))
    if not isinstance(value, str) or not value:
        raise CliRunnerError("cli_usage_error", f"argument {name} is required")
    return value


def _optional_str_arg(namespace: argparse.Namespace, name: str) -> str | None:
    value = cast(object, getattr(namespace, name))
    if value is None:
        return None
    if not isinstance(value, str):
        raise CliRunnerError("cli_usage_error", f"argument {name} must be a string")
    return value


def _optional_int_arg(namespace: argparse.Namespace, name: str) -> int | None:
    value = cast(object, getattr(namespace, name))
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise CliRunnerError("cli_usage_error", f"argument {name} must be an integer")
    return value


def _optional_float_arg(namespace: argparse.Namespace, name: str) -> float | None:
    value = cast(object, getattr(namespace, name))
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CliRunnerError("cli_usage_error", f"argument {name} must be a number")
    return float(value)


def _optional_decimal_arg(namespace: argparse.Namespace, name: str) -> Decimal | None:
    value = _optional_str_arg(namespace, name)
    if value is None:
        return None
    try:
        return Decimal(value)
    except InvalidOperation as exc:
        raise CliRunnerError("cli_usage_error", f"argument {name} must be a decimal") from exc


def _bool_arg(namespace: argparse.Namespace, name: str) -> bool:
    value = cast(object, getattr(namespace, name))
    if not isinstance(value, bool):
        raise CliRunnerError("cli_usage_error", f"argument {name} must be a boolean")
    return value


__all__ = (
    "ExitCode",
    "CliRunOptions",
    "CliRunnerError",
    "ModelAdapterFactory",
    "default_model_adapter_factory",
    "load_agent_config",
    "load_cli_config",
    "load_runtime_config_overrides",
    "main",
    "parse_cli_args",
    "run_cli",
    "run_cli_turn",
    "runtime_overrides_from_options",
)
