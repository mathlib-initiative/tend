"""Command-line entrypoint for the async orchestrator."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import shlex
import shutil
import signal
import subprocess
import sys
from collections.abc import Awaitable, Callable, Generator, Sequence
from contextlib import contextmanager
from decimal import Decimal, InvalidOperation
from enum import IntEnum
from pathlib import Path
from types import FrameType
from typing import Any, Final, NoReturn, Protocol, TextIO, cast

from pydantic import ValidationError

from tend._common.agent_outputs import AgentOutputSchemaName
from tend._common.config_files import (
    ConfigFileError,
    dump_config_model_yaml,
    dump_yaml_data,
    read_config_model,
)
from tend._common.errors import FrameworkError
from tend.orchestrator.code_snapshot import (
    OrchestratorCodeSnapshotError,
    code_dir_for_root,
    code_snapshot_is_present,
    create_code_snapshot,
    require_code_snapshot,
    validate_code_snapshot_location,
)
from tend.orchestrator.config import (
    AsyncOrchestratorAgentCommandConfig,
    AsyncOrchestratorBudgetConfig,
    AsyncOrchestratorConfig,
    AsyncOrchestratorProjectConfig,
    AsyncOrchestratorValidationCommandConfig,
    AsyncOrchestratorWorkspaceMirrorConfig,
    AsyncOrchestratorWorktreeSetupCommandConfig,
)
from tend.orchestrator.control_store import (
    ASYNC_ORCHESTRATOR_DB_FILENAME,
    SQLiteAsyncOrchestratorStore,
)
from tend.orchestrator.detach import spawn_detached
from tend.orchestrator.orchestrator import AsyncOrchestrator, AsyncOrchestratorRunResult
from tend.orchestrator.root_lock import (
    AsyncOrchestratorRootLock,
    AsyncOrchestratorRootLockError,
)
from tend.orchestrator.state import AsyncOrchestratorWorktree, WorktreeState
from tend.orchestrator.task_manager import TaskManager
from tend.orchestrator.tasks import TaskStatus
from tend.orchestrator.usage import format_usage_summary
from tend.prompts import builtin_prompts_root, load_prompt
from tend.workspace.mirror import MirrorReflinkMode

type _SignalHandler = signal.Handlers | int | Callable[[int, FrameType | None], Any] | None


class AsyncOrchestratorRunner(Protocol):
    """Minimal orchestrator interface required by the CLI."""

    def run(self) -> Awaitable[AsyncOrchestratorRunResult]:
        """Run the orchestrator."""
        ...


AsyncOrchestratorFactory = Callable[[AsyncOrchestratorConfig], AsyncOrchestratorRunner]

DEFAULT_CONFIG_FILENAME = "config.yaml"
DEFAULT_LOG_FILENAME = "logs.txt"
DEFAULT_STATE_FILENAME = ASYNC_ORCHESTRATOR_DB_FILENAME
_ASYNC_ORCHESTRATOR_MARKER = ".tend-root"
_RUNTIME_DIRECTORY_NAMES = ("worktrees", "sessions")
_CLI_COMMANDS = frozenset({"init", "run", "clean", "status", "export-state", "validate-config"})
_DETACHED_VALUE_FLAGS: Final = frozenset({"--log-file", "--pid-file"})
_INIT_AGENT_CHOICES = ("pi", "tend")
_DEFAULT_BUILD_GATE_TIMEOUT_SECONDS = 1800.0

# Marker lines that bracket the editable ``UV_PROJECT=...`` assignment inside
# the tend-agent launcher scripts. ``tend run`` rewrites the line between
# these markers to freeze a launched run against ``<root>/code/``; the markers
# let the rewrite find the assignment without parsing the whole script.
_UV_PROJECT_BEGIN_MARKER = "BEGIN tend UV_PROJECT block"
_UV_PROJECT_END_MARKER = "END tend UV_PROJECT block"
_UV_PROJECT_VARIABLE_NAME = "UV_PROJECT"
_LOG_LEVELS = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
}
_LOG_FORMAT = "%(levelname)s:%(name)s:%(message)s"
_LOGGER = logging.getLogger(__name__)

# Default worker prompt registry variant. The worker emits the shared
# ``worker_contribution`` output contract.
_DEFAULT_WORKER_PROMPT_VERSION = "minimal"

# Environment-variable equivalents of the worker prompt ``{task_path}`` /
# ``{worktree_path}`` / ``{contribution_id}`` placeholders. The agent is told
# what these variables mean and resolves them at run time, so no literal
# ``{...}`` placeholder leaks into the generated prompt.
_ASYNC_WORKER_TASK_SUBSTITUTIONS: dict[str, str] = {
    "task_path": "tasks/$TEND_TASK_ID.yaml",
    "worktree_path": "$TEND_WORKTREE_PATH (the current git worktree)",
    "contribution_id": "worktree $TEND_WORKTREE_ID",
}

# Sentinel string used to format-substitute the worker revision prompt at init
# time, leaving ``{feedback_message}`` literally intact in the on-disk template.
# The agent-runner replaces ``{feedback_message}`` per-revision with the
# rendered latest non-worker discussion message before re-launching the worker.
_FEEDBACK_MESSAGE_PASSTHROUGH = "{feedback_message}"


def _worker_prompt_dir(version: str) -> Path:
    return builtin_prompts_root() / "worker" / version


def _render_async_worker_prompt(
    version: str,
    name: str,
    *,
    extra_substitutions: dict[str, str] | None = None,
) -> str:
    """Load ``worker/<version>/<name>.md`` and resolve its placeholders for async.

    The worker prompts carry ``{task_path}`` / ``{worktree_path}`` /
    ``{contribution_id}`` placeholders. The orchestrator has no per-invocation
    template rendering, so we substitute environment-variable equivalents here
    at generation time; no literal
    ``{...}`` survives — except ``{feedback_message}`` in the revision prompt,
    which is intentionally left as a placeholder for the per-revision agent
    runner to substitute.
    """

    template = load_prompt(_worker_prompt_dir(version), name)
    substitutions = dict(_ASYNC_WORKER_TASK_SUBSTITUTIONS)
    if extra_substitutions is not None:
        substitutions.update(extra_substitutions)
    return template.format(**substitutions)


def _render_async_worker_task_prompt(version: str) -> str:
    """Load and resolve the per-invocation ``worker/<version>/task.md`` prompt."""

    return _render_async_worker_prompt(version, "task")


def _render_async_worker_revision_prompt(version: str) -> str:
    """Load ``worker/<version>/revision.md`` with ``{feedback_message}`` left intact.

    The async agent runner substitutes ``{feedback_message}`` per revision with
    the rendered latest non-worker discussion message — covering reviewer
    ``request_changes`` and the four orchestrator-injected feedback paths
    (merge failure, post-merge validation failure, dirty entrypoint, entrypoint
    status-check failure) — before launching the worker (see
    ``agent_runner._materialise_worker_revision_prompt``). All other
    placeholders (``{contribution_id}`` / ``{worktree_path}``) are
    async-resolved here.
    """

    return _render_async_worker_prompt(
        version,
        "revision",
        extra_substitutions={"feedback_message": _FEEDBACK_MESSAGE_PASSTHROUGH},
    )


def _async_worker_system_prompt(version: str) -> str:
    """Load ``worker/<version>/system.md`` as the async worker system prompt."""

    return load_prompt(_worker_prompt_dir(version), "system")


# Default reviewer prompt registry variant. The reviewer emits the shared
# ``review_verdict`` output contract.
_DEFAULT_REVIEWER_PROMPT_VERSION = "minimal"

# Environment-variable equivalents of the reviewer task prompt's
# ``{contribution_id}`` / ``{task_path}`` / ``{worktree_path}`` placeholders.
# The agent is told what these variables mean (see the worker prompt) and
# resolves them at run time, so no literal ``{...}`` placeholder leaks into the
# generated prompt.
_ASYNC_REVIEWER_TASK_SUBSTITUTIONS: dict[str, str] = {
    "contribution_id": "worktree $TEND_WORKTREE_ID",
    "task_path": "tasks/$TEND_TASK_ID.yaml",
    "worktree_path": "$TEND_WORKTREE_PATH (the current git worktree)",
}


def _reviewer_prompt_dir(version: str) -> Path:
    return builtin_prompts_root() / "reviewer" / version


def _render_async_reviewer_prompt(version: str, name: str) -> str:
    """Load ``reviewer/<version>/<name>.md`` and resolve its placeholders for async.

    The reviewer prompts carry ``{contribution_id}`` / ``{task_path}`` /
    ``{worktree_path}`` placeholders. The orchestrator has no per-invocation
    template rendering, so we substitute environment-variable equivalents here
    at generation time; no literal ``{...}`` survives.
    """

    template = load_prompt(_reviewer_prompt_dir(version), name)
    return template.format(**_ASYNC_REVIEWER_TASK_SUBSTITUTIONS)


def _render_async_reviewer_task_prompt(version: str) -> str:
    """Load and resolve the per-invocation ``reviewer/<version>/task.md`` prompt."""

    return _render_async_reviewer_prompt(version, "task")


def _async_reviewer_system_prompt(version: str) -> str:
    """Load ``reviewer/<version>/system.md`` as the async reviewer system prompt."""

    return load_prompt(_reviewer_prompt_dir(version), "system")


_TEND_AGENT_DEFAULT_MODEL = "claude-sonnet-4-5"
_TEND_AGENT_DEFAULT_MAX_OUTPUT_TOKENS = 32_768


class AsyncOrchestratorCliExitCode(IntEnum):
    """Documented async orchestrator CLI exit codes."""

    SUCCESS = 0
    CONFIGURATION_OR_USAGE = 2
    INTERNAL_SOFTWARE = 70
    INTERRUPTED = 130


class AsyncOrchestratorCliError(FrameworkError):
    """Configuration or usage error at the async orchestrator CLI boundary."""

    __slots__ = ("code",)

    code: str

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class _CliArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("allow_abbrev", False)
        super().__init__(*args, **kwargs)

    def error(self, message: str) -> NoReturn:
        raise AsyncOrchestratorCliError("cli_usage_error", message)


def main(argv: Sequence[str] | None = None) -> int:
    """Console-script entrypoint for ``tend`` and ``tend``."""

    return asyncio.run(run_cli(argv))


async def run_cli(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    orchestrator_factory: AsyncOrchestratorFactory | None = None,
    prog: str | None = None,
) -> int:
    """Parse CLI args, run one async-orchestrator command, and return an exit code."""

    argv_provided = argv is not None
    args = tuple(sys.argv[1:] if argv is None else argv)
    program_name = _program_name(prog, argv_provided=argv_provided)
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr
    factory = AsyncOrchestrator if orchestrator_factory is None else orchestrator_factory
    try:
        command, command_args = _split_command(args, prog=program_name)
        if command == "init":
            namespace = _build_init_parser(program_name).parse_args(list(command_args))
            _configure_logging(namespace, stderr=err)
            _handle_init(namespace, stdout=out)
            return int(AsyncOrchestratorCliExitCode.SUCCESS)
        if command == "clean":
            namespace = _build_clean_parser(program_name).parse_args(list(command_args))
            _configure_logging(namespace, stderr=err)
            _handle_clean(namespace, stdout=out, stderr=err)
            return int(AsyncOrchestratorCliExitCode.SUCCESS)
        if command == "status":
            namespace = _build_status_parser(program_name).parse_args(list(command_args))
            _configure_logging(namespace, stderr=err)
            _handle_status(namespace, stdout=out)
            return int(AsyncOrchestratorCliExitCode.SUCCESS)
        if command == "export-state":
            namespace = _build_export_state_parser(program_name).parse_args(list(command_args))
            _configure_logging(namespace, stderr=err)
            _handle_export_state(namespace, stdout=out)
            return int(AsyncOrchestratorCliExitCode.SUCCESS)
        if command == "validate-config":
            namespace = _build_validate_config_parser(program_name).parse_args(list(command_args))
            _configure_logging(namespace, stderr=err)
            _handle_validate_config(namespace, stdout=out)
            return int(AsyncOrchestratorCliExitCode.SUCCESS)
        if command != "run":
            raise AsyncOrchestratorCliError("cli_usage_error", f"unknown command: {command}")
        namespace = _build_run_parser(program_name).parse_args(list(command_args))
        _configure_logging(namespace, stderr=err)
        if _bool_arg(namespace, "dry_run"):
            _handle_run_dry_run(namespace, stdout=out)
            return int(AsyncOrchestratorCliExitCode.SUCCESS)
        if _bool_arg(namespace, "detach"):
            _handle_run_detached(
                namespace,
                command_args=command_args,
                argv_provided=argv_provided,
                stdout=out,
            )
            return int(AsyncOrchestratorCliExitCode.SUCCESS)
        with _acquired_root_lock(namespace, owner="run"):
            file_handler = _add_run_file_logging(namespace)
            try:
                config = _config_from_namespace(namespace)
                resume = _prepare_store_for_run(namespace, config)
                if resume:
                    _reuse_code_for_resume(config)
                else:
                    _freeze_code_for_run(config)
                orchestrator = _create_orchestrator(
                    factory,
                    config,
                    check_resume_health=resume,
                )
                with _installed_signal_handlers():
                    result = await orchestrator.run()
                _write_run_summary(result, stdout=out)
                _write_run_budget_stop(result, stdout=out)
            finally:
                _remove_logging_handler(file_handler)
        return int(AsyncOrchestratorCliExitCode.SUCCESS)
    except asyncio.CancelledError:
        _write_error("interrupted", "Interrupted by signal.", err)
        return int(AsyncOrchestratorCliExitCode.INTERRUPTED)
    except KeyboardInterrupt:
        _write_error("interrupted", "Interrupted by SIGINT.", err)
        return int(AsyncOrchestratorCliExitCode.INTERRUPTED)
    except AsyncOrchestratorCliError as exc:
        _write_error(exc.code, str(exc), err)
        return int(AsyncOrchestratorCliExitCode.CONFIGURATION_OR_USAGE)
    except OrchestratorCodeSnapshotError as exc:
        _write_error("code_snapshot_error", str(exc), err)
        return int(AsyncOrchestratorCliExitCode.CONFIGURATION_OR_USAGE)
    except ValidationError as exc:
        _write_error(
            "configuration_error",
            f"configuration validation failed: {_validation_error_summary(exc)}",
            err,
        )
        return int(AsyncOrchestratorCliExitCode.CONFIGURATION_OR_USAGE)
    except ConfigFileError as exc:
        _write_error("configuration_error", str(exc), err)
        return int(AsyncOrchestratorCliExitCode.CONFIGURATION_OR_USAGE)
    except AsyncOrchestratorRootLockError as exc:
        _write_error("root_lock_error", str(exc), err)
        return int(AsyncOrchestratorCliExitCode.CONFIGURATION_OR_USAGE)
    except OSError as exc:
        _write_error("filesystem_error", exc.strerror or str(exc), err)
        return int(AsyncOrchestratorCliExitCode.CONFIGURATION_OR_USAGE)
    except FrameworkError as exc:
        _write_error("framework_error", str(exc), err)
        return int(AsyncOrchestratorCliExitCode.INTERNAL_SOFTWARE)


def _split_command(args: Sequence[str], *, prog: str) -> tuple[str, tuple[str, ...]]:
    del prog
    if args and args[0] in _CLI_COMMANDS:
        return args[0], tuple(args[1:])
    return "run", tuple(args)


def _program_name(prog: str | None, *, argv_provided: bool) -> str:
    if prog is not None:
        stripped = prog.strip()
        if stripped:
            return stripped
    if argv_provided:
        return "tend"
    script_name = Path(sys.argv[0]).name
    return script_name or "tend"


def _command_prog(prog: str, command: str) -> str:
    return f"{prog} {command}"


def _handle_init(namespace: argparse.Namespace, *, stdout: TextIO) -> None:
    root = _required_path_arg(namespace, "root").expanduser().resolve()
    entrypoint = _required_path_arg(namespace, "entrypoint").expanduser().resolve()
    marker = root / _ASYNC_ORCHESTRATOR_MARKER
    config_path = _config_path(root)
    force = _bool_arg(namespace, "force")
    worktree_setup_command = _init_worktree_setup_command(namespace)
    _ensure_root_outside_entrypoint(root, entrypoint)
    _LOGGER.info("initializing async orchestration root: %s", root)
    if root.exists():
        if not root.is_dir():
            raise AsyncOrchestratorCliError(
                "filesystem_error",
                f"async orchestration root is not a directory: {root}",
            )
        if not marker.is_file() and any(root.iterdir()) and not force:
            raise AsyncOrchestratorCliError(
                "configuration_error",
                f"async orchestration root is not empty; use --force to initialize: {root}",
            )
    if config_path.exists() and not force:
        raise AsyncOrchestratorCliError(
            "configuration_error",
            f"async orchestration config already exists at {config_path}; use --force",
        )
    root.mkdir(parents=True, exist_ok=True)
    for directory_name in _RUNTIME_DIRECTORY_NAMES:
        (root / directory_name).mkdir(parents=True, exist_ok=True)
    marker.write_text("tend async orchestrator root\n", encoding="utf-8")
    worker_command: AsyncOrchestratorAgentCommandConfig | None = None
    reviewer_command: AsyncOrchestratorAgentCommandConfig | None = None
    agent = getattr(namespace, "agent", None)
    worker_prompt_version = cast(
        str,
        getattr(namespace, "worker_prompt_version", _DEFAULT_WORKER_PROMPT_VERSION)
        or _DEFAULT_WORKER_PROMPT_VERSION,
    )
    reviewer_prompt_version = cast(
        str,
        getattr(namespace, "reviewer_prompt_version", _DEFAULT_REVIEWER_PROMPT_VERSION)
        or _DEFAULT_REVIEWER_PROMPT_VERSION,
    )
    tend_project = _optional_path_arg(namespace, "tend_project")
    if agent == "pi":
        if tend_project is not None:
            raise AsyncOrchestratorCliError(
                "cli_usage_error",
                "--tend-project only applies to --agent tend",
            )
        worker_command, reviewer_command = _setup_pi_agent_files(
            root,
            force=force,
            worker_prompt_version=worker_prompt_version,
            reviewer_prompt_version=reviewer_prompt_version,
        )
    elif agent == "tend":
        worker_command, reviewer_command = _setup_tend_agent_files(
            root,
            force=force,
            worker_prompt_version=worker_prompt_version,
            reviewer_prompt_version=reviewer_prompt_version,
            tend_project=tend_project,
        )
    elif tend_project is not None:
        raise AsyncOrchestratorCliError(
            "cli_usage_error",
            "--tend-project requires --agent tend",
        )

    config = AsyncOrchestratorProjectConfig(
        entrypoint=entrypoint,
        worker_agent_command=worker_command,
        reviewer_agent_command=reviewer_command,
        worktree_setup_command=worktree_setup_command,
        workspace_mirror=_init_workspace_mirror(namespace),
        pre_merge_validation_commands=_init_pre_merge_validation_commands(namespace),
        merge_validation_worktree=not _bool_arg(namespace, "no_merge_validation_worktree"),
        seed_worktree_build=_bool_arg(namespace, "seed_worktree_build"),
        batched_merge=not _bool_arg(namespace, "no_batched_merge"),
        max_merge_batch_size=cast("int | None", namespace.max_merge_batch_size),
        skip_build_validation_for_task_only_merges=_bool_arg(
            namespace, "skip_build_validation_for_task_only_merges"
        ),
    )
    config_path.write_text(dump_config_model_yaml(config), encoding="utf-8")
    _LOGGER.info("initialized async orchestration root: %s", root)
    stdout.write(f"initialized async orchestration root at {root}\n")
    stdout.write(f"config: {config_path}\n")


def _ensure_root_outside_entrypoint(root: Path, entrypoint: Path) -> None:
    if root.is_relative_to(entrypoint):
        raise AsyncOrchestratorCliError(
            "configuration_error",
            "async orchestration root must be outside the entrypoint repository; "
            f"choose a root outside the source repository: root={root} entrypoint={entrypoint}",
        )


def _handle_clean(
    namespace: argparse.Namespace,
    *,
    stdout: TextIO,
    stderr: TextIO,
) -> None:
    root = _required_path_arg(namespace, "root").expanduser().resolve()
    dry_run = _bool_arg(namespace, "dry_run")
    skip_git = _bool_arg(namespace, "skip_git")
    entrypoint = _optional_path_arg(namespace, "entrypoint")

    if not root.exists():
        _LOGGER.info("async orchestration root does not exist; nothing to clean: %s", root)
        _write_clean_summary([], dry_run=dry_run, stdout=stdout)
        return
    if not root.is_dir():
        raise AsyncOrchestratorCliError(
            "filesystem_error",
            f"async orchestration root is not a directory: {root}",
        )
    if not (root / _ASYNC_ORCHESTRATOR_MARKER).is_file():
        raise AsyncOrchestratorCliError(
            "configuration_error",
            f"refusing to clean uninitialized async orchestration root: {root}",
        )

    if dry_run:
        _clean_initialized_root(
            root,
            entrypoint=entrypoint,
            skip_git=skip_git,
            dry_run=dry_run,
            stdout=stdout,
            stderr=stderr,
        )
        return

    with AsyncOrchestratorRootLock.acquire(root, owner="clean"):
        _clean_initialized_root(
            root,
            entrypoint=entrypoint,
            skip_git=skip_git,
            dry_run=dry_run,
            stdout=stdout,
            stderr=stderr,
        )


def _clean_initialized_root(
    root: Path,
    *,
    entrypoint: Path | None,
    skip_git: bool,
    dry_run: bool,
    stdout: TextIO,
    stderr: TextIO,
) -> None:
    _LOGGER.info("cleaning async orchestration root: %s", root)
    cleaned: list[str] = []
    warnings: list[str] = []
    if not skip_git:
        if entrypoint is None:
            entrypoint = _entrypoint_from_root_config(root, warnings=warnings)
        if entrypoint is not None:
            _clean_git_worktrees(
                entrypoint.expanduser().resolve(),
                root=root,
                dry_run=dry_run,
                cleaned=cleaned,
                warnings=warnings,
            )

    if dry_run:
        cleaned.append(f"would remove async orchestration root: {root}")
    else:
        shutil.rmtree(root)
        cleaned.append(f"removed async orchestration root: {root}")
        _LOGGER.info("removed async orchestration root: %s", root)

    _write_clean_summary(cleaned, dry_run=dry_run, stdout=stdout)
    for warning in warnings:
        stderr.write(f"warning: {warning}\n")


def _clean_git_worktrees(
    entrypoint: Path,
    *,
    root: Path,
    dry_run: bool,
    cleaned: list[str],
    warnings: list[str],
) -> None:
    worktrees_dir = root / "worktrees"
    if not worktrees_dir.is_dir():
        return
    for child in sorted(worktrees_dir.iterdir()):
        if not child.is_dir():
            continue
        if dry_run:
            cleaned.append(f"would remove git worktree: {child}")
            continue
        completed = _run_git(
            entrypoint,
            "worktree",
            "remove",
            "--force",
            child,
            check=False,
        )
        if completed.returncode == 0:
            cleaned.append(f"removed git worktree: {child}")
            _LOGGER.info("removed git worktree: %s", child)
        else:
            warning = f"could not remove git worktree {child}: {_completed_error(completed)}"
            warnings.append(warning)
            _LOGGER.warning(warning)
    if dry_run:
        cleaned.append(f"would prune git worktrees for: {entrypoint}")
        return
    _run_git(entrypoint, "worktree", "prune", check=False)


def _write_clean_summary(cleaned: Sequence[str], *, dry_run: bool, stdout: TextIO) -> None:
    prefix = "dry run" if dry_run else "clean"
    stdout.write(f"{prefix}: {len(cleaned)} item(s)\n")
    for item in cleaned:
        stdout.write(f"  {item}\n")
    if not cleaned:
        stdout.write("  (nothing to clean)\n")


def _write_run_summary(result: AsyncOrchestratorRunResult, *, stdout: TextIO) -> None:
    """Print a run summary (task/worktree counts + usage) to stdout after a run."""

    summary = result.summary
    tasks = ", ".join(
        [f"total={summary.tasks_total}"]
        + [f"{status}={count}" for status, count in summary.tasks_by_status.items()]
    )
    worktrees = ", ".join(
        [f"total={summary.worktrees_total}"]
        + [f"{state}={count}" for state, count in summary.worktrees_by_state.items()]
    )
    stdout.write("async orchestrator run complete\n")
    stdout.write(f"tasks: {tasks}\n")
    stdout.write(f"worktrees: {worktrees}\n")
    stdout.write(f"{format_usage_summary(result.usage)}\n")


def _write_run_budget_stop(result: AsyncOrchestratorRunResult, *, stdout: TextIO) -> None:
    """Print a one-line budget-stop summary after the run summary, if any."""

    budget_stop = result.budget_stop
    if budget_stop is None:
        return
    breach_suffix = (
        ""
        if budget_stop.breach_accumulated_cost == budget_stop.accumulated_cost
        else f", breach_cost={budget_stop.breach_accumulated_cost} {budget_stop.currency}"
    )
    stdout.write(
        "stopped on cost ceiling: accumulated "
        f"{budget_stop.accumulated_cost} {budget_stop.currency}{breach_suffix} "
        f">= max_cost {budget_stop.max_cost} {budget_stop.currency}\n"
    )


def _handle_status(namespace: argparse.Namespace, *, stdout: TextIO) -> None:
    root = _required_path_arg(namespace, "root").expanduser().resolve()
    store = SQLiteAsyncOrchestratorStore(root)

    stdout.write("async orchestrator status\n")
    stdout.write(f"root: {root}\n")

    if store.state_exists():
        task_manager = store.load_task_snapshot()
        worktrees = store.list_worktrees()
        stdout.write(f"state: loaded ({store.path})\n")
        stdout.write(f"tasks: {_task_counts_summary(task_manager)}\n")
        stdout.write(f"worktrees: {_worktree_counts_summary(worktrees)}\n")
        stdout.write(f"inferred queues: {_queue_counts_summary(worktrees)}\n")
        usage = store.aggregate_usage(root)
        stdout.write(f"usage: loaded ({store.path})\n")
        stdout.write(f"aggregate {format_usage_summary(usage)}\n")
    else:
        stdout.write(f"state: missing ({store.path})\n")
        stdout.write("tasks: unavailable\n")
        stdout.write("worktrees: unavailable\n")
        stdout.write("inferred queues: unavailable\n")
        stdout.write(f"usage: missing ({store.path})\n")


def _handle_export_state(namespace: argparse.Namespace, *, stdout: TextIO) -> None:
    root = _required_path_arg(namespace, "root").expanduser().resolve()
    store = SQLiteAsyncOrchestratorStore(root)
    if not store.state_exists():
        raise AsyncOrchestratorCliError(
            "state_missing",
            f"async orchestrator state database does not exist at {store.path}",
        )

    payload = {
        "worktrees": [
            worktree.model_dump(mode="json") for worktree in store.list_worktrees()
        ],
        "task_snapshot": store.load_task_snapshot().model_dump(mode="json"),
        "usage": store.aggregate_usage(root).model_dump(mode="json"),
    }
    stdout.write(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    stdout.write("\n")


def _task_counts_summary(task_manager: TaskManager) -> str:
    counts = {status: 0 for status in TaskStatus}
    for task in task_manager.tasks:
        counts[task.status] += 1
    parts = [f"total={len(task_manager.tasks)}"]
    parts.extend(f"{status.value}={counts[status]}" for status in TaskStatus)
    return ", ".join(parts)


def _worktree_counts_summary(worktrees: Sequence[AsyncOrchestratorWorktree]) -> str:
    counts = {worktree_state: 0 for worktree_state in WorktreeState}
    for worktree in worktrees:
        counts[worktree.state] += 1
    parts = [f"total={len(worktrees)}"]
    parts.extend(
        f"{worktree_state.value}={counts[worktree_state]}"
        for worktree_state in WorktreeState
    )
    return ", ".join(parts)


def _queue_counts_summary(worktrees: Sequence[AsyncOrchestratorWorktree]) -> str:
    worker_count = sum(
        1
        for worktree in worktrees
        if worktree.state is WorktreeState.PENDING and worktree.task_id is not None
    )
    reviewer_count = sum(
        1
        for worktree in worktrees
        if worktree.state is WorktreeState.REVIEW
    )
    merge_count = sum(
        1
        for worktree in worktrees
        if worktree.state is WorktreeState.MERGE
    )
    return f"worker={worker_count}, reviewer={reviewer_count}, merge={merge_count}"


def _config_from_namespace(namespace: argparse.Namespace) -> AsyncOrchestratorConfig:
    root = _required_path_arg(namespace, "root").expanduser().resolve()
    config_path = _config_path(root)
    project_config = _read_project_config(config_path)
    entrypoint_override = _optional_path_arg(namespace, "entrypoint")
    if entrypoint_override is None:
        entrypoint = _resolve_config_path(project_config.entrypoint, base=config_path.parent)
    else:
        entrypoint = entrypoint_override.expanduser().resolve()
    resolved_config = project_config.to_runtime_config(root=root, entrypoint=entrypoint)
    worker_command = _agent_command_with_resume(
        namespace.worker_agent_command or resolved_config.worker_agent_command,
        namespace.worker_agent_resume_args,
        field_name="worker agent",
    )
    reviewer_command = _agent_command_with_resume(
        namespace.reviewer_agent_command or resolved_config.reviewer_agent_command,
        namespace.reviewer_agent_resume_args,
        field_name="reviewer agent",
    )
    return AsyncOrchestratorConfig(
        root=resolved_config.root,
        entrypoint=resolved_config.entrypoint,
        worker_agent_command=worker_command,
        reviewer_agent_command=reviewer_command,
        worktree_setup_command=(
            namespace.worktree_setup_command or resolved_config.worktree_setup_command
        ),
        workspace_mirror=resolved_config.workspace_mirror,
        validation_commands=resolved_config.validation_commands,
        pre_merge_validation_commands=resolved_config.pre_merge_validation_commands,
        merge_target_branch=resolved_config.merge_target_branch,
        merge_validation_worktree=resolved_config.merge_validation_worktree,
        seed_worktree_build=resolved_config.seed_worktree_build,
        batched_merge=resolved_config.batched_merge,
        max_merge_batch_size=resolved_config.max_merge_batch_size,
        skip_build_validation_for_task_only_merges=(
            resolved_config.skip_build_validation_for_task_only_merges
        ),
        max_concurrent_worker_agents=(
            resolved_config.max_concurrent_worker_agents
            if namespace.max_concurrent_worker_agents is None
            else namespace.max_concurrent_worker_agents
        ),
        max_concurrent_reviewer_agents=(
            resolved_config.max_concurrent_reviewer_agents
            if namespace.max_concurrent_reviewer_agents is None
            else namespace.max_concurrent_reviewer_agents
        ),
        budget=_resolve_budget(namespace, resolved_config.budget),
        agent_oom_score_adj=resolved_config.agent_oom_score_adj,
        cleanup_closed_worktrees=resolved_config.cleanup_closed_worktrees,
    )


def _resolve_budget(
    namespace: argparse.Namespace,
    base: AsyncOrchestratorBudgetConfig,
) -> AsyncOrchestratorBudgetConfig:
    """Return the budget with an optional ``--max-cost`` CLI override applied."""

    max_cost = cast("Decimal | None", getattr(namespace, "max_cost", None))
    if max_cost is None:
        return base
    data = base.model_dump(mode="python")
    data["max_cost"] = max_cost
    return AsyncOrchestratorBudgetConfig.model_validate(data)


def _prepare_store_for_run(
    namespace: argparse.Namespace,
    config: AsyncOrchestratorConfig,
) -> bool:
    """Prepare the unified store and return whether this run is a resume."""

    store = SQLiteAsyncOrchestratorStore(config.root)
    if _bool_arg(namespace, "fresh"):
        if store.state_exists():
            _LOGGER.info("clearing async orchestrator state due to --fresh: %s", store.path)
            store.clear_state()
        return False
    if not store.state_exists():
        return False
    _LOGGER.info("resuming async orchestrator state from: %s", store.path)
    return True


def _handle_run_dry_run(namespace: argparse.Namespace, *, stdout: TextIO) -> None:
    """Resolve and validate the run config and print what would run, without running."""

    config = _config_from_namespace(namespace)
    store = SQLiteAsyncOrchestratorStore(config.root)
    fresh = _bool_arg(namespace, "fresh")
    stdout.write("async orchestrator run dry-run\n")
    _write_config_preview(config, stdout=stdout)
    if fresh:
        stdout.write("state: would start fresh (--fresh)\n")
    elif store.state_exists():
        stdout.write(f"state: would resume from {store.path}\n")
    else:
        stdout.write("state: would start fresh (no saved state)\n")
    stdout.write("dry run: configuration is valid; no worktrees or agents were launched\n")


def _handle_run_detached(
    namespace: argparse.Namespace,
    *,
    command_args: Sequence[str],
    argv_provided: bool,
    stdout: TextIO,
) -> None:
    """Spawn the foreground ``run`` command in a detached child process."""

    root = _required_path_arg(namespace, "root").expanduser().resolve()
    log_file_override = _optional_path_arg(namespace, "log_file")
    pid_file_override = _optional_path_arg(namespace, "pid_file")
    log_file = root / "run.log" if log_file_override is None else log_file_override.expanduser()
    pid_file = root / "run.pid" if pid_file_override is None else pid_file_override.expanduser()
    child_argv = _detached_child_argv(command_args, entry=_orchestrator_entry(argv_provided))
    pid = spawn_detached(child_argv, log_file=log_file, pid_file=pid_file)
    stdout.write(f"launched tend run (detached) for {root}\n")
    stdout.write(f"  pid: {pid} (recorded at {pid_file})\n")
    stdout.write(f"  log: {log_file}\n")


def _orchestrator_entry(argv_provided: bool) -> str:
    """Return the console-script path to use for a detached child invocation."""

    if not argv_provided:
        entry = sys.argv[0]
        if entry:
            return entry
    return shutil.which("tend") or "tend"


def _detached_child_argv(command_args: Sequence[str], *, entry: str) -> list[str]:
    """Return child ``run`` argv with detach-only flags removed."""

    filtered: list[str] = []
    skip_next = False
    for token in command_args:
        if skip_next:
            skip_next = False
            continue
        flag = token.split("=", 1)[0]
        if token == "--detach":
            continue
        if flag in _DETACHED_VALUE_FLAGS:
            skip_next = "=" not in token
            continue
        filtered.append(token)
    return [entry, "run", *filtered]


def _handle_validate_config(namespace: argparse.Namespace, *, stdout: TextIO) -> None:
    """Read and validate ``config.yaml`` and print the resolved configuration."""

    root = _required_path_arg(namespace, "root").expanduser().resolve()
    config_path = _config_path(root)
    project_config = _read_project_config(config_path)
    entrypoint_override = _optional_path_arg(namespace, "entrypoint")
    if entrypoint_override is None:
        entrypoint = _resolve_config_path(project_config.entrypoint, base=config_path.parent)
    else:
        entrypoint = entrypoint_override.expanduser().resolve()
    config = project_config.to_runtime_config(root=root, entrypoint=entrypoint)
    stdout.write(f"async orchestrator config: {config_path}\n")
    _write_config_preview(config, stdout=stdout)
    stdout.write("config is valid\n")


def _write_config_preview(config: AsyncOrchestratorConfig, *, stdout: TextIO) -> None:
    """Print a human-readable preview of a resolved runtime config."""

    stdout.write(f"entrypoint: {config.entrypoint}\n")
    stdout.write(f"merge target branch: {config.merge_target_branch}\n")
    stdout.write(f"worker agent command: {_format_optional_command(config.worker_agent_command)}\n")
    stdout.write(
        f"reviewer agent command: {_format_optional_command(config.reviewer_agent_command)}\n"
    )
    stdout.write(
        "worktree setup command: "
        f"{_format_optional_worktree_setup_command(config.worktree_setup_command)}\n"
    )
    stdout.write(
        "concurrency: "
        f"workers={config.max_concurrent_worker_agents}, "
        f"reviewers={config.max_concurrent_reviewer_agents}\n"
    )
    stdout.write(
        f"validation commands: {_format_validation_commands(config.validation_commands)}\n"
    )
    stdout.write(
        "post-merge build gate: "
        f"{_format_validation_commands(config.pre_merge_validation_commands)}\n"
    )


def _format_optional_command(command: AsyncOrchestratorAgentCommandConfig | None) -> str:
    if command is None:
        return "(unset)"
    return " ".join(command.argv)


def _format_optional_worktree_setup_command(
    command: AsyncOrchestratorWorktreeSetupCommandConfig | None,
) -> str:
    if command is None:
        return "(unset)"
    return " ".join(command.argv)


def _format_validation_commands(
    commands: Sequence[AsyncOrchestratorValidationCommandConfig],
) -> str:
    if not commands:
        return "(none)"
    return "; ".join(_format_validation_command(command) for command in commands)


def _format_validation_command(command: AsyncOrchestratorValidationCommandConfig) -> str:
    text = " ".join(command.argv)
    if command.timeout_seconds is None:
        return text
    return f"{text} (timeout={command.timeout_seconds:g}s)"


def _agent_launcher_scripts(config: AsyncOrchestratorConfig) -> tuple[Path, ...]:
    """Return on-disk launcher script paths for worker/reviewer tend-agent commands.

    Only paths whose first argv element points at an existing file are
    returned; commands that exec an interpreter (or have not been initialized
    via ``tend init --agent tend``) are skipped, since there is no script to
    rewrite.
    """

    scripts: list[Path] = []
    for command in (config.worker_agent_command, config.reviewer_agent_command):
        if command is None or not command.argv:
            continue
        candidate = Path(command.argv[0])
        if candidate.is_file():
            scripts.append(candidate)
    return tuple(scripts)


def _read_uv_project_from_script(script: Path) -> str | None:
    """Return the ``UV_PROJECT`` value embedded in a generated tend-agent script.

    Looks for the marker-bracketed assignment written by ``_tend_agent_script``.
    Returns ``None`` when the script has no ``UV_PROJECT`` block (e.g. pi-agent
    scripts or operator-authored launchers); returns ``""`` when the block is
    present but empty (script will run ``tend-agent`` from ``$PATH``).
    """

    try:
        text = script.read_text(encoding="utf-8")
    except OSError:
        return None
    if _UV_PROJECT_BEGIN_MARKER not in text or _UV_PROJECT_END_MARKER not in text:
        return None
    prefix = f"{_UV_PROJECT_VARIABLE_NAME}="
    for line in text.splitlines():
        stripped = line.lstrip()
        if not stripped.startswith(prefix):
            continue
        value = stripped[len(prefix) :]
        # Strip optional shlex.quote single-quotes / escaped quotes. The init
        # template only ever writes ``UV_PROJECT=<shlex_quoted>``; nothing
        # exotic, so the simple shlex.split round-trip is fine.
        parsed = shlex.split(value)
        if not parsed:
            return ""
        return parsed[0]
    return None


def _rewrite_uv_project_in_script(script: Path, *, uv_project: str) -> None:
    """Rewrite the ``UV_PROJECT=...`` assignment inside a generated tend-agent script.

    Hard-fails if the script does not contain the marker-bracketed block, so a
    misconfigured launcher does not silently keep running against the operator's
    working checkout.
    """

    text = script.read_text(encoding="utf-8")
    if _UV_PROJECT_BEGIN_MARKER not in text or _UV_PROJECT_END_MARKER not in text:
        raise OrchestratorCodeSnapshotError(
            f"tend-agent launcher script {script} has no UV_PROJECT block to rewrite; "
            "re-run ``tend init --agent tend --tend-project <path>`` to regenerate it"
        )
    lines = text.splitlines(keepends=True)
    prefix = f"{_UV_PROJECT_VARIABLE_NAME}="
    quoted = shlex.quote(uv_project)
    new_lines: list[str] = []
    rewritten = False
    for line in lines:
        if not rewritten and line.lstrip().startswith(prefix):
            indent = line[: len(line) - len(line.lstrip())]
            newline = "\n" if line.endswith("\n") else ""
            new_lines.append(f"{indent}{prefix}{quoted}{newline}")
            rewritten = True
        else:
            new_lines.append(line)
    if not rewritten:
        raise OrchestratorCodeSnapshotError(
            f"tend-agent launcher script {script} has a UV_PROJECT block but no "
            f"{prefix}... assignment; refusing to rewrite"
        )
    script.write_text("".join(new_lines), encoding="utf-8")


def _scripts_with_uv_project_block(config: AsyncOrchestratorConfig) -> tuple[Path, ...]:
    """Return generated tend-agent scripts that carry an editable ``UV_PROJECT`` block.

    These are the only launchers that the launch-time freeze can repoint at
    ``<root>/code/``. Returning ``()`` short-circuits the freeze/resume hooks
    (e.g. pi-agent scaffolds or operator-supplied launchers without a block).
    """

    scripts: list[Path] = []
    for script in _agent_launcher_scripts(config):
        if _read_uv_project_from_script(script) is None:
            continue
        scripts.append(script)
    return tuple(scripts)


def _freeze_code_for_run(config: AsyncOrchestratorConfig) -> None:
    """Launch-time: snapshot the operator's tend checkout into ``<root>/code/``.

    Mirrors the shared ``_freeze_code_for_run``: the working
    checkout baked into each generated tend-agent launcher script's
    ``UV_PROJECT`` line is file-copied (honoring an ignore set) into
    ``<root>/code/`` and the scripts are rewritten in-place to point there.
    No-op when no generated script carries an editable ``UV_PROJECT`` block
    (e.g. pi-agent scaffolds or commands that exec a non-script binary).
    """

    scripts = _scripts_with_uv_project_block(config)
    if not scripts:
        return
    sources: list[Path] = []
    for script in scripts:
        value = _read_uv_project_from_script(script)
        if value:
            sources.append(Path(value))
    if not sources:
        # All scripts have empty UV_PROJECT blocks: nothing to freeze.
        return
    source = sources[0].expanduser().resolve()
    for other in sources[1:]:
        resolved = other.expanduser().resolve()
        if resolved != source:
            raise OrchestratorCodeSnapshotError(
                "tend-agent launcher scripts disagree on the tend checkout to "
                f"freeze: {source} vs {resolved}"
            )
    code_dir = code_dir_for_root(config.root)
    validate_code_snapshot_location(source_checkout=source, code_dir=code_dir)
    if code_snapshot_is_present(code_dir):
        # Honor the same escape hatch as sync: a wrapper may have prepared
        # <root>/code/ before us. Reuse rather than refuse.
        code_dir = code_dir.resolve()
    else:
        create_code_snapshot(source_checkout=source, code_dir=code_dir)
    code_dir_text = str(code_dir.resolve())
    for script in scripts:
        _rewrite_uv_project_in_script(script, uv_project=code_dir_text)


def _reuse_code_for_resume(config: AsyncOrchestratorConfig) -> None:
    """Resume-time: reuse ``<root>/code/`` unchanged; hard-fail if missing.

    Pins a resumed run to its launch-time code. When at least one generated
    tend-agent launcher script carries a ``UV_PROJECT`` block we require the
    snapshot and rewrite every block to point at it, even if the operator
    already (correctly) baked ``<root>/code/`` into one of the scripts.
    """

    scripts = _scripts_with_uv_project_block(config)
    if not scripts:
        return
    has_uv_project = any(_read_uv_project_from_script(script) for script in scripts)
    if not has_uv_project:
        # All scripts have empty UV_PROJECT blocks: the operator opted out of
        # ``uv run --project`` at init time, so there is nothing to pin.
        return
    code_dir = require_code_snapshot(code_dir_for_root(config.root))
    code_dir_text = str(code_dir)
    for script in scripts:
        # Skip the rewrite when the script already points at the snapshot
        # (the common case on resume after the launch-time rewrite). Mirrors
        # sync's _repoint_templates_to_code_dir, which guards the same way.
        if _read_uv_project_from_script(script) == code_dir_text:
            continue
        _rewrite_uv_project_in_script(script, uv_project=code_dir_text)


def _create_orchestrator(
    factory: AsyncOrchestratorFactory,
    config: AsyncOrchestratorConfig,
    *,
    check_resume_health: bool,
) -> AsyncOrchestratorRunner:
    if not check_resume_health:
        return factory(config)
    try:
        return cast(
            AsyncOrchestratorRunner,
            cast(Any, factory)(config, check_resume_health=True),
        )
    except TypeError as exc:
        raise AsyncOrchestratorCliError(
            "state_error",
            "saved async orchestrator state exists, but the configured orchestrator "
            "factory does not accept resume health checks",
        ) from exc


def _init_worktree_setup_command(
    namespace: argparse.Namespace,
) -> AsyncOrchestratorWorktreeSetupCommandConfig | None:
    copy_dirs = tuple(cast(Sequence[str], getattr(namespace, "copy_dirs", ()) or ()))
    cow = _bool_arg(namespace, "cow")
    if cow and not copy_dirs:
        raise AsyncOrchestratorCliError(
            "cli_usage_error",
            "--cow requires at least one --copy-dir/--copy_dir value",
        )
    if not copy_dirs:
        return None
    argv = ["cp", "--archive"]
    if cow:
        argv.append("--reflink=always")
    argv.extend(f"{{entrypoint}}/{copy_dir}" for copy_dir in copy_dirs)
    argv.append("{worktree}/")
    return AsyncOrchestratorWorktreeSetupCommandConfig(argv=tuple(argv))


def _init_workspace_mirror(
    namespace: argparse.Namespace,
) -> AsyncOrchestratorWorkspaceMirrorConfig:
    """Build the workspace mirror block written to config.yaml by ``tend init``.

    Defaults to a disabled mirror so existing runs are unaffected.
    ``--mirror-enabled`` turns it on; the per-list flags repeatably append to
    ``symlink_paths`` / ``exclude_names`` / ``exclude_paths``; ``--mirror-reflink``
    sets the reflink policy.
    """

    enabled = _bool_arg(namespace, "mirror_enabled")
    symlink_paths = tuple(cast(Sequence[str], getattr(namespace, "symlink_paths", ()) or ()))
    exclude_names = tuple(
        cast(Sequence[str], getattr(namespace, "mirror_exclude_names", ()) or ())
    )
    exclude_paths = tuple(
        cast(Sequence[str], getattr(namespace, "mirror_exclude_paths", ()) or ())
    )
    reflink_mode = cast(
        MirrorReflinkMode,
        getattr(namespace, "mirror_reflink", MirrorReflinkMode.AUTO) or MirrorReflinkMode.AUTO,
    )
    if not enabled and not (symlink_paths or exclude_names or exclude_paths):
        # Fast path: nothing to write, leave the default disabled config.
        return AsyncOrchestratorWorkspaceMirrorConfig()
    return AsyncOrchestratorWorkspaceMirrorConfig(
        enabled=enabled,
        reflink_mode=reflink_mode,
        symlink_paths=list(symlink_paths),
        exclude_names=list(exclude_names),
        exclude_paths=list(exclude_paths),
    )


def _init_pre_merge_validation_commands(
    namespace: argparse.Namespace,
) -> tuple[AsyncOrchestratorValidationCommandConfig, ...]:
    """Build the post-merge build gate written to config.yaml by ``tend init``.

    A new generic project has no implicit build command. ``--build-command``
    enables the gate and ``--no-build-gate`` explicitly disables it.
    """

    if _bool_arg(namespace, "no_build_gate"):
        return ()
    build_command = cast(
        "tuple[str, ...] | None", getattr(namespace, "build_command", None)
    )
    if build_command is None:
        return ()
    timeout = cast(
        "float | None",
        getattr(namespace, "build_timeout_seconds", None) or _DEFAULT_BUILD_GATE_TIMEOUT_SECONDS,
    )
    return (
        AsyncOrchestratorValidationCommandConfig(
            argv=tuple(build_command), timeout_seconds=timeout
        ),
    )


def _setup_pi_agent_files(
    root: Path,
    *,
    force: bool,
    worker_prompt_version: str = _DEFAULT_WORKER_PROMPT_VERSION,
    reviewer_prompt_version: str = _DEFAULT_REVIEWER_PROMPT_VERSION,
) -> tuple[AsyncOrchestratorAgentCommandConfig, AsyncOrchestratorAgentCommandConfig]:
    worker_system_prompt = root / "prompts" / "worker-system.md"
    worker_prompt = root / "prompts" / "worker.md"
    reviewer_system_prompt = root / "prompts" / "reviewer-system.md"
    reviewer_prompt = root / "prompts" / "reviewer.md"
    worker_script = root / "bin" / "worker-agent.sh"
    reviewer_script = root / "bin" / "reviewer-agent.sh"
    _write_init_file(
        worker_system_prompt,
        f"{_async_worker_system_prompt(worker_prompt_version)}\n",
        force=force,
    )
    _write_init_file(
        worker_prompt,
        f"{_render_async_worker_task_prompt(worker_prompt_version)}\n",
        force=force,
    )
    _write_init_file(
        reviewer_system_prompt,
        f"{_async_reviewer_system_prompt(reviewer_prompt_version)}\n",
        force=force,
    )
    _write_init_file(
        reviewer_prompt,
        f"{_render_async_reviewer_task_prompt(reviewer_prompt_version)}\n",
        force=force,
    )
    _write_init_file(
        worker_script,
        _pi_agent_script(
            "../prompts/worker-system.md",
            "../prompts/worker.md",
            tools="read,bash,edit,write,grep,find,ls",
        ),
        force=force,
        executable=True,
    )
    _write_init_file(
        reviewer_script,
        _pi_agent_script(
            "../prompts/reviewer-system.md",
            "../prompts/reviewer.md",
            tools="read,bash,grep,find,ls",
        ),
        force=force,
        executable=True,
    )
    return (
        AsyncOrchestratorAgentCommandConfig(
            argv=(str(worker_script),),
            resume_argv=("--resume",),
        ),
        AsyncOrchestratorAgentCommandConfig(
            argv=(str(reviewer_script),),
            resume_argv=("--resume",),
        ),
    )


def _setup_tend_agent_files(
    root: Path,
    *,
    force: bool,
    worker_prompt_version: str = _DEFAULT_WORKER_PROMPT_VERSION,
    reviewer_prompt_version: str = _DEFAULT_REVIEWER_PROMPT_VERSION,
    tend_project: Path | None = None,
) -> tuple[AsyncOrchestratorAgentCommandConfig, AsyncOrchestratorAgentCommandConfig]:
    worker_system_prompt = root / "prompts" / "worker-system.md"
    worker_prompt = root / "prompts" / "worker.md"
    worker_revision_prompt = root / "prompts" / "worker-revision.md"
    reviewer_system_prompt = root / "prompts" / "reviewer-system.md"
    reviewer_prompt = root / "prompts" / "reviewer.md"
    worker_agent_config = root / ".tend" / "worker-agent.yaml"
    reviewer_agent_config = root / ".tend" / "reviewer-agent.yaml"
    worker_runtime_config = root / ".tend" / "worker-cfg.yaml"
    reviewer_runtime_config = root / ".tend" / "reviewer-cfg.yaml"
    readme = root / ".tend" / "README.md"
    worker_script = root / "bin" / "worker-agent.sh"
    reviewer_script = root / "bin" / "reviewer-agent.sh"

    # The worker/reviewer system prompts live in editable markdown files referenced
    # by the generated agent configs. The per-invocation prompt files carry the
    # resolved task prompts.
    _write_init_file(
        worker_system_prompt,
        f"{_async_worker_system_prompt(worker_prompt_version)}\n",
        force=force,
    )
    _write_init_file(
        worker_prompt,
        f"{_render_async_worker_task_prompt(worker_prompt_version)}\n",
        force=force,
    )
    # The revision prompt is consulted on ``--resume`` after any non-worker
    # discussion turn (reviewer ``request_changes`` or one of the four
    # orchestrator-injected feedback paths); the agent runner substitutes
    # ``{feedback_message}`` with the rendered feedback before launching.
    _write_init_file(
        worker_revision_prompt,
        f"{_render_async_worker_revision_prompt(worker_prompt_version)}\n",
        force=force,
    )
    _write_init_file(
        reviewer_system_prompt,
        f"{_async_reviewer_system_prompt(reviewer_prompt_version)}\n",
        force=force,
    )
    _write_init_file(
        reviewer_prompt,
        f"{_render_async_reviewer_task_prompt(reviewer_prompt_version)}\n",
        force=force,
    )
    _write_init_file(
        worker_agent_config,
        dump_yaml_data(
            _tend_agent_config(role="worker")
        ),
        force=force,
    )
    _write_init_file(
        reviewer_agent_config,
        dump_yaml_data(
            _tend_agent_config(role="reviewer")
        ),
        force=force,
    )
    _write_init_file(
        worker_runtime_config,
        dump_yaml_data(_tend_agent_runtime_config()),
        force=force,
    )
    _write_init_file(
        reviewer_runtime_config,
        dump_yaml_data(_tend_agent_runtime_config()),
        force=force,
    )
    _write_init_file(readme, _tend_agent_readme(), force=force)
    uv_project_value = (
        None if tend_project is None else str(tend_project.expanduser().resolve())
    )
    _write_init_file(
        worker_script,
        _tend_agent_script(
            "../.tend/worker-agent.yaml",
            "../.tend/worker-cfg.yaml",
            "../prompts/worker.md",
            uv_project=uv_project_value,
            revision_prompt_path="../prompts/worker-revision.md",
        ),
        force=force,
        executable=True,
    )
    _write_init_file(
        reviewer_script,
        _tend_agent_script(
            "../.tend/reviewer-agent.yaml",
            "../.tend/reviewer-cfg.yaml",
            "../prompts/reviewer.md",
            uv_project=uv_project_value,
        ),
        force=force,
        executable=True,
    )
    return (
        AsyncOrchestratorAgentCommandConfig(
            argv=(str(worker_script),),
            resume_argv=("--resume",),
        ),
        AsyncOrchestratorAgentCommandConfig(
            argv=(str(reviewer_script),),
            resume_argv=("--resume",),
        ),
    )


def _tend_agent_config(*, role: str) -> dict[str, object]:
    if role == "worker":
        tools = ["ls", "read_file", "grep", "glob", "write_file", "edit_file", "copy_lines", "bash"]
        # Reuse the shared worker_contribution output schema. The generated config
        # points at an editable markdown system prompt under <root>/prompts/, so
        # operators can tune it between launches without editing YAML or relying on
        # the bundled registry after init.
        schema_name = AgentOutputSchemaName.WORKER_CONTRIBUTION.value
        system_prompt: object = {"path": "../prompts/worker-system.md"}
    elif role == "reviewer":
        tools = ["ls", "read_file", "grep", "glob", "bash"]
        # Reuse the sync reviewer's per-criterion-verdict contract and the shared
        # ``review_verdict`` output schema. The generated config points at an editable
        # markdown system prompt under <root>/prompts/; see worker note above.
        schema_name = AgentOutputSchemaName.REVIEW_VERDICT.value
        system_prompt = {"path": "../prompts/reviewer-system.md"}
    else:
        raise ValueError(f"unknown tend-agent role: {role}")
    return {
        "schema_version": "1",
        "system_prompt": system_prompt,
        "model": {
            "provider": "anthropic",
            "api": "anthropic_messages",
            "model_name": _TEND_AGENT_DEFAULT_MODEL,
            "settings": {
                "reasoning": {"effort": "low"},
                "max_output_tokens": _TEND_AGENT_DEFAULT_MAX_OUTPUT_TOKENS,
            },
        },
        "tools": {"enabled": tools},
        # Capture the result through a schema-validated final_result tool call rather
        # than trusting the model's raw stdout. tend-agent then writes only the validated
        # payload to stdout, so a chatty model's prose can't break the JSON contract.
        "output": {
            "tool_name": "final_result",
            "schema_name": schema_name,
            "required": True,
        },
    }


def _tend_agent_runtime_config() -> dict[str, object]:
    return {
        "limits": {
            "max_wall_time_seconds": 7_200.0,
        },
        "compaction": {
            "enabled": True,
            "reserve_tokens": 16_384,
            "keep_recent_tokens": 20_000,
            "target_tokens": 8_000,
            "trigger_on_context_overflow": True,
        },
        "model": {"timeout_seconds": 600.0},
        "environment": {
            "allowed_env_vars": [
                "ANTHROPIC_API_KEY",
                "ANTHROPIC_BASE_URL",
                "OPENAI_API_KEY",
                "OPENAI_BASE_URL",
            ]
        },
    }


def _tend_agent_readme() -> str:
    return f"""# tend-agent orchestrator config

These files configure `tend-agent` for `tend init --agent tend`.

Default model: `anthropic/{_TEND_AGENT_DEFAULT_MODEL}` via the Anthropic Messages API.
Default response budget: `{_TEND_AGENT_DEFAULT_MAX_OUTPUT_TOKENS}` `max_output_tokens`, sized to
cover a worker's per-task output envelope.

Before running, provide an Anthropic API key:

```bash
export ANTHROPIC_API_KEY="<anthropic-api-key>"
```

Set `TEND_AGENT_BIN` if `tend-agent` is not on `PATH`.

## Editable prompts

Generated agent YAMLs read their system prompts from `../prompts/worker-system.md`
and `../prompts/reviewer-system.md`. The launcher scripts read
`../prompts/worker.md`, `../prompts/worker-revision.md`, and
`../prompts/reviewer.md` for per-turn prompts. Editing these files affects the
next agent launch, including resumed sessions.

## tend checkout pinning (`--tend-project`)

The generated `worker-agent.sh` / `reviewer-agent.sh` scripts carry a
marker-bracketed `UV_PROJECT=...` block. When `tend init --agent tend` is
invoked with `--tend-project <tend-checkout>`, the path is baked into
that block and the script execs via `uv run --project "$UV_PROJECT" -- tend-agent
...` so the agents run against your checkout's resolved environment. At
`tend run`, the orchestrator file-copies that checkout into `<root>/code/`
(honoring `DEFAULT_CODE_IGNORE`) and rewrites every script's `UV_PROJECT` to
point at `<root>/code/`, so long-lived orchestrator runs and their child tend-agent
subprocesses stay pinned to launch-time code while the working checkout keeps
being edited. `tend run` on an existing run reuses the snapshot and
hard-fails if it is gone.

If `--tend-project` is omitted, the block stays empty and the launchers
exec `tend-agent` from `PATH`; the snapshot hook is a no-op.
"""


def _write_init_file(
    path: Path,
    content: str,
    *,
    force: bool,
    executable: bool = False,
) -> None:
    if path.exists() and not force:
        raise AsyncOrchestratorCliError(
            "configuration_error",
            f"async orchestration file already exists at {path}; use --force",
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    if executable:
        path.chmod(0o755)


def _pi_agent_script(system_prompt_path: str, prompt_path: str, *, tools: str) -> str:
    return f'''#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${{BASH_SOURCE[0]}}")" && pwd)"
SYSTEM_PROMPT_FILE="$SCRIPT_DIR/{system_prompt_path}"
PROMPT_FILE="$SCRIPT_DIR/{prompt_path}"
SYSTEM_PROMPT="$(cat "$SYSTEM_PROMPT_FILE")"
PROMPT="$(cat "$PROMPT_FILE")"

RESUME_ARGS=()
if [[ "${{1:-}}" == "--resume" || "${{TEND_AGENT_RESUME:-0}}" == "1" ]]; then
  RESUME_ARGS=(--continue)
fi

PI_COMMAND="${{PI_BIN:-pi}}"
if ! command -v "$PI_COMMAND" >/dev/null 2>&1; then
  echo "pi command not found; install pi or set PI_BIN" >&2
  exit 127
fi

exec "$PI_COMMAND" \
  --print \
  --mode text \
  --session-dir "$TEND_AGENT_SESSION_DIR" \
  "${{RESUME_ARGS[@]}}" \
  --append-system-prompt "$SYSTEM_PROMPT" \
  --tools {tools} \
  "$PROMPT"
'''


def _tend_agent_script(
    agent_path: str,
    config_path: str,
    prompt_path: str,
    *,
    uv_project: str | None = None,
    revision_prompt_path: str | None = None,
) -> str:
    """Render the worker/reviewer shim that launches ``tend-agent``.

    When ``revision_prompt_path`` is provided (worker shim), the shim picks the
    rendered revision prompt at ``$TEND_AGENT_SESSION_DIR/revision-prompt.md``
    if present on a resume invocation — that file is materialised by the
    orchestrator's agent runner with the latest non-worker discussion message
    spliced into ``{feedback_message}`` (reviewer ``request_changes`` or any
    of the four orchestrator-injected feedback paths). When absent (initial
    assignment, worker already replied, or no pending feedback), the shim
    falls back to the initial prompt. The reviewer shim never substitutes;
    it always uses its single prompt.
    """

    uv_project_value = "" if uv_project is None else uv_project
    # ``revision_prompt_path`` being non-None marks this as a worker shim. The
    # template path it names (``prompts/worker-revision.md``) is read by the
    # per-revision agent runner, not by the shim — the runner substitutes
    # ``{feedback_message}`` against the latest non-worker discussion message
    # and writes the result to ``$TEND_AGENT_SESSION_DIR/revision-prompt.md``.
    # The shim's only job is to prefer that substituted prompt over the
    # initial assignment when it's present (resume-after-feedback).
    if revision_prompt_path is None:
        revision_prompt_block = ""
    else:
        # The block runs AFTER resume detection so revision-prompt selection
        # is gated on ``${{#RESUME_ARGS[@]}} -gt 0`` — a stale prompt left over
        # in a session dir cannot affect a non-resume invocation. The runner
        # only writes ``revision-prompt.md`` on actual resumes, but this
        # defensive gate matches the resume semantics exactly and survives
        # session-dir reuse edge cases.
        revision_prompt_block = f'''
# On ``--resume`` with pending feedback (reviewer ``request_changes`` or one of
# the four orchestrator-injected paths: merge failure, post-merge validation
# failure, dirty entrypoint, entrypoint status-check failure), the orchestrator's
# agent runner reads ``{revision_prompt_path}``, substitutes ``{{feedback_message}}``
# with the latest non-worker discussion message, and writes the result to
# ``$TEND_AGENT_SESSION_DIR/revision-prompt.md``. Prefer that substituted
# file when present (and only when we are actually resuming); fall back to the
# initial assignment prompt otherwise (initial run, no pending feedback, or
# older orchestrator without this wiring).
if [[ "${{#RESUME_ARGS[@]}}" -gt 0 && -n "${{TEND_AGENT_SESSION_DIR:-}}" ]]; then
  SESSION_REVISION_PROMPT_FILE="$TEND_AGENT_SESSION_DIR/revision-prompt.md"
  if [[ -s "$SESSION_REVISION_PROMPT_FILE" ]]; then
    PROMPT="$(cat "$SESSION_REVISION_PROMPT_FILE")"
  fi
fi
'''
    return f'''#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${{BASH_SOURCE[0]}}")" && pwd)"
AGENT_FILE="$SCRIPT_DIR/{agent_path}"
CONFIG_FILE="$SCRIPT_DIR/{config_path}"
PROMPT_FILE="$SCRIPT_DIR/{prompt_path}"
PROMPT="$(cat "$PROMPT_FILE")"

# {_UV_PROJECT_BEGIN_MARKER}
# tend checkout used to launch ``tend-agent`` via ``uv run --project``. When
# empty, ``tend-agent`` is run directly from ``$PATH``. ``tend run`` rewrites
# this line in-place to freeze the run against ``<root>/code/``; ``tend
# init --tend-project`` writes the operator's initial checkout here. The
# surrounding marker lines must not be edited.
{_UV_PROJECT_VARIABLE_NAME}={shlex.quote(uv_project_value)}
# {_UV_PROJECT_END_MARKER}

RESUME_ARGS=()
if [[ "${{1:-}}" == "--resume" || "${{TEND_AGENT_RESUME:-0}}" == "1" ]]; then
  if [[ -s "$TEND_AGENT_SESSION_DIR/events.jsonl" ]]; then
    RESUME_ARGS=(--resume-session)
  fi
fi
{revision_prompt_block}

if [[ -n "${_UV_PROJECT_VARIABLE_NAME}" ]]; then
  exec uv run --project "${_UV_PROJECT_VARIABLE_NAME}" -- tend-agent \\
    --agent "$AGENT_FILE" \\
    --config "$CONFIG_FILE" \\
    --cwd "$TEND_WORKTREE_PATH" \\
    --session-dir "$TEND_AGENT_SESSION_DIR" \\
    "${{RESUME_ARGS[@]}}" \\
    --prompt "$PROMPT"
fi

TEND_AGENT_COMMAND="${{TEND_AGENT_BIN:-tend-agent}}"
if ! command -v "$TEND_AGENT_COMMAND" >/dev/null 2>&1; then
  echo "tend-agent command not found; install tend or set TEND_AGENT_BIN" >&2
  exit 127
fi

exec "$TEND_AGENT_COMMAND" \\
  --agent "$AGENT_FILE" \\
  --config "$CONFIG_FILE" \\
  --cwd "$TEND_WORKTREE_PATH" \\
  --session-dir "$TEND_AGENT_SESSION_DIR" \\
  "${{RESUME_ARGS[@]}}" \\
  --prompt "$PROMPT"
'''


def _entrypoint_from_root_config(root: Path, *, warnings: list[str]) -> Path | None:
    config_path = _config_path(root)
    if not config_path.is_file():
        return None
    try:
        project_config = _read_project_config(config_path)
    except ConfigFileError as exc:
        warnings.append(f"could not read async orchestration config {config_path}: {exc}")
        return None
    return _resolve_config_path(project_config.entrypoint, base=config_path.parent)


def _read_project_config(config_path: Path) -> AsyncOrchestratorProjectConfig:
    return read_config_model(
        config_path,
        AsyncOrchestratorProjectConfig,
        kind="async orchestrator config",
    )


def _build_init_parser(prog: str) -> argparse.ArgumentParser:
    parser = _CliArgumentParser(
        prog=_command_prog(prog, "init"),
        description="Initialize an async orchestration root directory.",
    )
    _add_logging_args(parser)
    parser.add_argument(
        "--root",
        default=Path("."),
        type=Path,
        help=(
            "Root directory to initialize outside the entrypoint repository "
            "(default: current directory)."
        ),
    )
    parser.add_argument(
        "--entrypoint",
        default=Path("."),
        type=Path,
        help="Entrypoint git repository to write to config.yaml (default: current directory).",
    )
    parser.add_argument(
        "--agent",
        choices=_INIT_AGENT_CHOICES,
        help="Create default agent command scripts and prompts for this agent runner.",
    )
    parser.add_argument(
        "--worker-prompt-version",
        "--worker_prompt_version",
        dest="worker_prompt_version",
        default=_DEFAULT_WORKER_PROMPT_VERSION,
        metavar="VERSION",
        help=(
            "Worker prompt registry variant under tend/prompts/worker/ to use for "
            f"generated worker agents (default: {_DEFAULT_WORKER_PROMPT_VERSION})."
        ),
    )
    parser.add_argument(
        "--reviewer-prompt-version",
        "--reviewer_prompt_version",
        dest="reviewer_prompt_version",
        default=_DEFAULT_REVIEWER_PROMPT_VERSION,
        metavar="VERSION",
        help=(
            "Reviewer prompt registry variant under tend/prompts/reviewer/ to use for "
            f"generated reviewer agents (default: {_DEFAULT_REVIEWER_PROMPT_VERSION})."
        ),
    )
    parser.add_argument(
        "--copy-dir",
        "--copy_dir",
        dest="copy_dirs",
        action="append",
        type=_copy_dir_arg,
        default=None,
        metavar="DIR",
        help=(
            "Copy a directory from the entrypoint into each new worktree during setup; "
            "may be repeated."
        ),
    )
    parser.add_argument(
        "--cow",
        action="store_true",
        help="Use cp --reflink=always for --copy-dir worktree setup copies.",
    )
    parser.add_argument(
        "--mirror-enabled",
        "--mirror_enabled",
        action="store_true",
        dest="mirror_enabled",
        help=(
            "Enable the workspace mirror that runs after `git worktree add` and "
            "before the worktree-setup command in each new async worktree."
        ),
    )
    parser.add_argument(
        "--symlink-path",
        "--symlink_path",
        dest="symlink_paths",
        action="append",
        type=_mirror_relative_path_arg,
        default=None,
        metavar="PATH",
        help=(
            "Mirror this path as an absolute symlink into the entrypoint instead "
            "of copying it (e.g. .lake/packages/mathlib); may be repeated."
        ),
    )
    parser.add_argument(
        "--mirror-exclude-name",
        "--mirror_exclude_name",
        dest="mirror_exclude_names",
        action="append",
        type=_mirror_segment_arg,
        default=None,
        metavar="NAME",
        help=(
            "Skip any path component with this name anywhere in the mirror tree; "
            "may be repeated."
        ),
    )
    parser.add_argument(
        "--mirror-exclude-path",
        "--mirror_exclude_path",
        dest="mirror_exclude_paths",
        action="append",
        type=_mirror_relative_path_arg,
        default=None,
        metavar="PATH",
        help=(
            "Skip this relative subtree from the source root during mirror; "
            "may be repeated."
        ),
    )
    parser.add_argument(
        "--mirror-reflink",
        "--mirror_reflink",
        dest="mirror_reflink",
        type=_mirror_reflink_arg,
        default=MirrorReflinkMode.AUTO,
        metavar="MODE",
        help=(
            "Reflink policy for mirror file copies (auto/required/never; "
            "default: auto)."
        ),
    )
    parser.add_argument(
        "--build-command",
        type=_argv_arg,
        dest="build_command",
        metavar="COMMAND",
        help=(
            "Shell-like command run as the post-merge build gate after each approved "
            "merge is assembled; with staging validation, the entrypoint is unchanged "
            "on failure (default: disabled)."
        ),
    )
    parser.add_argument(
        "--build-timeout-seconds",
        type=_positive_float_arg,
        dest="build_timeout_seconds",
        metavar="SECONDS",
        help=(
            "Wall-clock timeout for the post-merge build gate command "
            f"(default: {_DEFAULT_BUILD_GATE_TIMEOUT_SECONDS:g})."
        ),
    )
    parser.add_argument(
        "--no-build-gate",
        action="store_true",
        dest="no_build_gate",
        help="Do not write a post-merge build gate to config.yaml.",
    )
    parser.add_argument(
        "--no-merge-validation-worktree",
        action="store_true",
        dest="no_merge_validation_worktree",
        help=(
            "Disable the staging validation worktree (default: enabled). When "
            "disabled, merges validate in the entrypoint and revert on failure, "
            "and ready-task worktree creation shares the merge lock with the "
            "validation build."
        ),
    )
    parser.add_argument(
        "--seed-worktree-build",
        "--seed_worktree_build",
        action="store_true",
        dest="seed_worktree_build",
        help=(
            "Seed each new task worktree's .lake/build from a snapshot of the "
            "staging worktree's build cache (refreshed after every validated "
            "merge), so the worker's first `lake build` is incremental against "
            "current main instead of from scratch. Requires the staging "
            "validation worktree (on by default)."
        ),
    )
    parser.add_argument(
        "--no-batched-merge",
        "--no_batched_merge",
        action="store_true",
        dest="no_batched_merge",
        help=(
            "Disable batched merging (default: enabled). With batching, all ready "
            "worktrees are validated in one staging build per round and published "
            "together (a one-item queue is a batch of one = the same as serial). "
            "Only applies to the staging path."
        ),
    )
    parser.add_argument(
        "--max-merge-batch-size",
        "--max_merge_batch_size",
        dest="max_merge_batch_size",
        type=_positive_int_arg,
        metavar="N",
        help=(
            "Cap the number of MERGE worktrees included in one batched staging "
            "validation. Omit for the default drain-all behavior. Only applies "
            "when batched staging merges are enabled."
        ),
    )
    parser.add_argument(
        "--skip-build-validation-for-task-only-merges",
        "--skip_build_validation_for_task_only_merges",
        action="store_true",
        dest="skip_build_validation_for_task_only_merges",
        help=(
            "Skip the post-merge build gate for merges whose diff changed only "
            "files under tasks/ (default: disabled). Enable only if the "
            "validation commands consume nothing under tasks/ (no include_str, "
            "no custom facets) -- tend does not verify this assertion. The "
            "build-free task-tree gate still runs first and must pass; merges "
            "touching any non-task path build exactly as before."
        ),
    )
    parser.add_argument(
        "--tend-project",
        "--tend_project",
        dest="tend_project",
        type=Path,
        metavar="PATH",
        help=(
            "tend checkout used by generated tend-agent launcher scripts via "
            "``uv run --project``. When set with ``--agent tend``, each tend "
            "run freezes this checkout into ``<root>/code/`` and repoints the "
            "scripts there so the run is pinned to its launch-time code."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow initializing a non-empty directory and overwrite managed files.",
    )
    return parser


def _build_clean_parser(prog: str) -> argparse.ArgumentParser:
    parser = _CliArgumentParser(
        prog=_command_prog(prog, "clean"),
        description="Remove an initialized async orchestration root directory.",
    )
    _add_logging_args(parser)
    parser.add_argument(
        "--root",
        required=True,
        type=Path,
        help="Root directory to clean.",
    )
    parser.add_argument(
        "--entrypoint",
        type=Path,
        help="Entrypoint git repository used to deregister worktrees before removal.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be removed without removing anything.",
    )
    parser.add_argument(
        "--skip-git",
        action="store_true",
        help="Skip git worktree deregistration and only remove the root directory.",
    )
    return parser


def _build_status_parser(prog: str) -> argparse.ArgumentParser:
    parser = _CliArgumentParser(
        prog=_command_prog(prog, "status"),
        description="Print a read-only async orchestrator status summary.",
    )
    _add_logging_args(parser)
    parser.add_argument(
        "--root",
        required=True,
        type=Path,
        help="Root directory containing the async orchestrator SQLite database.",
    )
    return parser


def _build_export_state_parser(prog: str) -> argparse.ArgumentParser:
    parser = _CliArgumentParser(
        prog=_command_prog(prog, "export-state"),
        description="Export the durable async orchestrator state as JSON.",
    )
    _add_logging_args(parser)
    parser.add_argument(
        "--root",
        required=True,
        type=Path,
        help="Root directory containing the async orchestrator SQLite database.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        required=True,
        help="Write a JSON object containing worktrees, task snapshot, and usage.",
    )
    return parser


def _build_run_parser(prog: str) -> argparse.ArgumentParser:
    parser = _CliArgumentParser(
        prog=_command_prog(prog, "run"),
        description="Run the async orchestrator.",
    )
    _add_logging_args(parser)
    parser.add_argument(
        "--root",
        required=True,
        type=Path,
        help=f"Root directory containing {DEFAULT_CONFIG_FILENAME} for this run.",
    )
    parser.add_argument(
        "--entrypoint",
        type=Path,
        help="Override the entrypoint git repository from config.yaml.",
    )
    parser.add_argument(
        "--worker-agent-command",
        type=_agent_command_arg,
        help="Shell-like command string used to launch a worker agent.",
    )
    parser.add_argument(
        "--reviewer-agent-command",
        type=_agent_command_arg,
        help="Shell-like command string used to launch a reviewer agent.",
    )
    parser.add_argument(
        "--worktree-setup-command",
        type=_worktree_setup_command_arg,
        help=(
            "Shell-like command string run after creating each worktree; use "
            "{entrypoint} and {worktree} placeholders for paths."
        ),
    )
    parser.add_argument(
        "--worker-agent-resume-args",
        type=_argv_arg,
        help="Shell-like arguments appended when resuming a worker agent session.",
    )
    parser.add_argument(
        "--reviewer-agent-resume-args",
        type=_argv_arg,
        help="Shell-like arguments appended when resuming a reviewer agent session.",
    )
    parser.add_argument(
        "--max-concurrent-worker-agents",
        type=int,
        help="Override maximum number of worker agents to run concurrently.",
    )
    parser.add_argument(
        "--max-concurrent-reviewer-agents",
        type=int,
        help="Override maximum number of reviewer agents to run concurrently.",
    )
    parser.add_argument(
        "--max-cost",
        type=_max_cost_arg,
        dest="max_cost",
        metavar="AMOUNT",
        help=(
            "Stop claiming new work once accumulated agent cost reaches this "
            "inclusive ceiling (in the configured budget currency); in-flight work "
            "still settles. Overrides budget.max_cost from config.yaml."
        ),
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help=(
            f"Ignore any saved {DEFAULT_STATE_FILENAME} and start with a fresh "
            "orchestrator state."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help=(
            "Resolve and validate the run configuration and print what would run "
            "without creating worktrees or launching agents."
        ),
    )
    parser.add_argument(
        "--detach",
        action="store_true",
        dest="detach",
        help=(
            "Run the orchestrator in a detached background process and return immediately; "
            "stdout/stderr go to a log file and the child PID is written to a pid file."
        ),
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Detached-mode log file (default: <root>/run.log). Ignored without --detach.",
    )
    parser.add_argument(
        "--pid-file",
        type=Path,
        default=None,
        help="Detached-mode pid file (default: <root>/run.pid). Ignored without --detach.",
    )
    return parser


def _build_validate_config_parser(prog: str) -> argparse.ArgumentParser:
    parser = _CliArgumentParser(
        prog=_command_prog(prog, "validate-config"),
        description="Validate an async orchestration config without running.",
    )
    _add_logging_args(parser)
    parser.add_argument(
        "--root",
        required=True,
        type=Path,
        help=f"Root directory containing {DEFAULT_CONFIG_FILENAME} to validate.",
    )
    parser.add_argument(
        "--entrypoint",
        type=Path,
        help="Override the entrypoint git repository from config.yaml during validation.",
    )
    return parser


def _add_logging_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--log-level",
        default=logging.INFO,
        type=_log_level_arg,
        metavar="LEVEL",
        help="Python logging level to emit (default: INFO).",
    )


def _log_level_arg(value: str) -> int:
    level = _LOG_LEVELS.get(value.strip().upper())
    if level is None:
        choices = ", ".join(_LOG_LEVELS)
        raise argparse.ArgumentTypeError(f"log level must be one of: {choices}")
    return level


def _configure_logging(namespace: argparse.Namespace, *, stderr: TextIO) -> None:
    level = getattr(namespace, "log_level", logging.INFO)
    logging.basicConfig(
        level=level,
        stream=stderr,
        format=_LOG_FORMAT,
    )
    logging.getLogger("tend.orchestrator").setLevel(level)


def _acquired_root_lock(namespace: argparse.Namespace, *, owner: str) -> AsyncOrchestratorRootLock:
    root = _required_path_arg(namespace, "root").expanduser().resolve()
    return AsyncOrchestratorRootLock.acquire(root, owner=owner)


def _add_run_file_logging(namespace: argparse.Namespace) -> logging.Handler:
    root = _required_path_arg(namespace, "root").expanduser().resolve()
    log_path = root / DEFAULT_LOG_FILENAME
    level = getattr(namespace, "log_level", logging.INFO)
    handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    logging.getLogger("tend.orchestrator").addHandler(handler)
    _LOGGER.info("writing async orchestrator logs to: %s", log_path)
    return handler


def _remove_logging_handler(handler: logging.Handler) -> None:
    logger = logging.getLogger("tend.orchestrator")
    logger.removeHandler(handler)
    handler.close()


@contextmanager
def _installed_signal_handlers() -> Generator[None]:
    current_task = asyncio.current_task()
    if current_task is None:
        yield
        return

    previous_handlers: dict[signal.Signals, _SignalHandler] = {}

    def handler(signum: int, frame: FrameType | None) -> None:
        del frame
        signal_name = _signal_name(signum)
        _LOGGER.info("received %s; cancelling async orchestrator", signal_name)
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


def _agent_command_arg(value: str) -> AsyncOrchestratorAgentCommandConfig:
    try:
        return AsyncOrchestratorAgentCommandConfig(argv=_argv_arg(value))
    except ValidationError as exc:
        raise argparse.ArgumentTypeError(_validation_error_summary(exc)) from exc


def _worktree_setup_command_arg(value: str) -> AsyncOrchestratorWorktreeSetupCommandConfig:
    try:
        return AsyncOrchestratorWorktreeSetupCommandConfig(argv=_argv_arg(value))
    except ValidationError as exc:
        raise argparse.ArgumentTypeError(_validation_error_summary(exc)) from exc


def _copy_dir_arg(value: str) -> str:
    if not value.strip() or "\x00" in value:
        raise argparse.ArgumentTypeError("copy directory must not be blank or contain NUL")
    path = Path(value)
    if path.is_absolute():
        raise argparse.ArgumentTypeError("copy directory must be relative to the entrypoint")
    if ".." in path.parts:
        raise argparse.ArgumentTypeError("copy directory must not contain '..'")
    return path.as_posix()


def _mirror_relative_path_arg(value: str) -> str:
    if not value.strip() or "\x00" in value:
        raise argparse.ArgumentTypeError(
            "mirror path must not be blank or contain NUL"
        )
    path = Path(value)
    if path.is_absolute():
        raise argparse.ArgumentTypeError("mirror path must be relative to the entrypoint")
    if "." in path.parts or ".." in path.parts:
        raise argparse.ArgumentTypeError("mirror path must not contain '.' or '..'")
    return path.as_posix()


def _mirror_segment_arg(value: str) -> str:
    if not value.strip() or "\x00" in value:
        raise argparse.ArgumentTypeError(
            "mirror exclude name must not be blank or contain NUL"
        )
    if "/" in value or "\\" in value or value in {".", ".."}:
        raise argparse.ArgumentTypeError(
            "mirror exclude name must be a single relative path segment"
        )
    return value


def _mirror_reflink_arg(value: str) -> MirrorReflinkMode:
    try:
        return MirrorReflinkMode(value.strip().lower())
    except ValueError as exc:
        choices = ", ".join(mode.value for mode in MirrorReflinkMode)
        raise argparse.ArgumentTypeError(
            f"mirror reflink mode must be one of: {choices}"
        ) from exc


def _argv_arg(value: str) -> tuple[str, ...]:
    try:
        return tuple(shlex.split(value))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _positive_float_arg(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a number, got {value!r}") from exc
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed


def _positive_int_arg(value: str) -> int:
    try:
        parsed = int(value, 10)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed


def _max_cost_arg(value: str) -> Decimal:
    text = value.strip()
    try:
        parsed = Decimal(text)
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError(f"expected a decimal amount, got {value!r}") from exc
    if not parsed.is_finite():
        raise argparse.ArgumentTypeError("max cost must be finite")
    if parsed <= 0:
        raise argparse.ArgumentTypeError("max cost must be greater than 0")
    return parsed


def _agent_command_with_resume(
    command: AsyncOrchestratorAgentCommandConfig | None,
    resume_argv: tuple[str, ...] | None,
    *,
    field_name: str,
) -> AsyncOrchestratorAgentCommandConfig | None:
    if resume_argv is None:
        return command
    if command is None:
        raise AsyncOrchestratorCliError(
            "cli_usage_error",
            f"{field_name} resume args require {field_name} command",
        )
    return command.model_copy(update={"resume_argv": resume_argv})


def _config_path(root: Path) -> Path:
    return root / DEFAULT_CONFIG_FILENAME


def _resolve_config_path(path: Path, *, base: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (base / expanded).resolve()


def _required_path_arg(namespace: argparse.Namespace, name: str) -> Path:
    value = getattr(namespace, name, None)
    if not isinstance(value, Path):
        raise AsyncOrchestratorCliError("cli_usage_error", f"missing required path: {name}")
    return value


def _optional_path_arg(namespace: argparse.Namespace, name: str) -> Path | None:
    value = getattr(namespace, name, None)
    if value is None or isinstance(value, Path):
        return value
    raise AsyncOrchestratorCliError("cli_usage_error", f"invalid path: {name}")


def _bool_arg(namespace: argparse.Namespace, name: str) -> bool:
    return bool(getattr(namespace, name, False))


def _run_git(
    repo: Path,
    *args: str | Path,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *(str(arg) for arg in args)],
        cwd=repo,
        check=check,
        capture_output=True,
        text=True,
    )


def _completed_error(completed: subprocess.CompletedProcess[str]) -> str:
    stderr = completed.stderr.strip()
    stdout = completed.stdout.strip()
    return stderr or stdout or f"git exited with code {completed.returncode}"


def _validation_error_summary(exc: ValidationError) -> str:
    first_error = exc.errors()[0]
    location = ".".join(str(part) for part in first_error.get("loc", ()))
    message = str(first_error.get("msg", "invalid value"))
    return f"{location}: {message}" if location else message


def _write_error(code: str, message: str, stderr: TextIO) -> None:
    print(f"error[{code}]: {message}", file=stderr)
