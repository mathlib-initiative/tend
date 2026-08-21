"""Configuration models for the async orchestrator."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from os import PathLike, fspath
from pathlib import Path
from string import Formatter
from typing import Annotated, cast

from pydantic import Field, ValidationInfo, field_validator, model_validator

from tend._common.types import StrictModel
from tend.workspace.mirror import (
    MirrorExistingPathPolicy,
    MirrorReflinkMode,
    WorkspaceMirrorConfig,
)

_PositiveInt = Annotated[int, Field(ge=1)]
_PositiveFloat = Annotated[float, Field(gt=0.0)]
# Linux ``oom_score_adj`` is clamped to [-1000, 1000]; we only ever raise it
# (make a process *more* killable), which needs no privilege for same-uid procs.
_OomScoreAdj = Annotated[int, Field(ge=-1000, le=1000)]
# Default applied to spawned agent subprocesses (and, by fork inheritance, their
# ``lake``/``lean`` build descendants) + orchestrator-run build commands. A
# clearly-preferred OOM victim, well above the orchestrator and operator session
# (both at 0), so memory exhaustion reaps a recomputable build, never the
# orchestrator or the operator's terminal. ``None`` disables (no preexec set).
_DEFAULT_AGENT_OOM_SCORE_ADJ = 750
_PositiveDecimal = Annotated[Decimal, Field(gt=Decimal(0))]
_WORKTREE_SETUP_PLACEHOLDERS = frozenset({"entrypoint", "worktree"})
_DEFAULT_MERGE_TARGET_BRANCH = "main"
_FORMATTER = Formatter()


class AsyncOrchestratorWorktreeSetupCommandConfig(StrictModel):
    """Shell-free command run after creating an async orchestrator worktree."""

    argv: tuple[str, ...] = Field(min_length=1)

    @field_validator("argv", mode="before")
    @classmethod
    def _coerce_argv(cls, value: object) -> object:
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return _coerce_argv_sequence(cast(Sequence[object], value), kind="worktree setup")
        return value

    @field_validator("argv")
    @classmethod
    def _validate_argv(cls, argv: tuple[str, ...]) -> tuple[str, ...]:
        _validate_command_argv(argv, kind="worktree setup")
        _validate_worktree_setup_placeholders(argv)
        return argv

    def argv_for_paths(self, *, entrypoint: Path, worktree: Path) -> tuple[str, ...]:
        """Return command arguments with path placeholders expanded."""

        placeholders = {
            "entrypoint": str(entrypoint),
            "worktree": str(worktree),
        }
        return tuple(argument.format(**placeholders) for argument in self.argv)


def _empty_string_list() -> list[str]:
    return []


class AsyncOrchestratorWorkspaceMirrorConfig(StrictModel):
    """Workspace mirror config for newly created orchestrator worktrees.

    This is resolved to the shared
    :class:`tend.workspace.mirror.WorkspaceMirrorConfig` used by the lower-level
    mirror engine.
    """

    enabled: bool = False
    reflink_mode: MirrorReflinkMode = MirrorReflinkMode.AUTO
    existing_path_policy: MirrorExistingPathPolicy = MirrorExistingPathPolicy.SKIP
    exclude_names: list[str] = Field(default_factory=_empty_string_list)
    exclude_paths: list[str] = Field(default_factory=_empty_string_list)
    symlink_paths: list[str] = Field(default_factory=_empty_string_list)

    @model_validator(mode="after")
    def _validate_mirror_options(self) -> AsyncOrchestratorWorkspaceMirrorConfig:
        self.to_workspace_mirror_config()
        return self

    def to_workspace_mirror_config(self) -> WorkspaceMirrorConfig:
        """Return the lower-level workspace mirror configuration."""

        return WorkspaceMirrorConfig(
            reflink_mode=self.reflink_mode,
            existing_path_policy=self.existing_path_policy,
            exclude_names=self.exclude_names,
            exclude_paths=self.exclude_paths,
            symlink_paths=self.symlink_paths,
        )


class AsyncOrchestratorValidationCommandConfig(StrictModel):
    """Shell-free command run as an async orchestrator validation gate.

    ``timeout_seconds`` bounds the wall-clock time the command may run. The merge
    thread runs validation under ``merge_lock``; without a timeout a hung command
    (e.g. a stuck ``lake build``) would stall the whole run indefinitely. ``None``
    (the default) leaves the command unbounded.
    """

    argv: tuple[str, ...] = Field(min_length=1)
    timeout_seconds: _PositiveFloat | None = None

    @field_validator("argv", mode="before")
    @classmethod
    def _coerce_argv(cls, value: object) -> object:
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return _coerce_argv_sequence(cast(Sequence[object], value), kind="validation")
        return value

    @field_validator("argv")
    @classmethod
    def _validate_argv(cls, argv: tuple[str, ...]) -> tuple[str, ...]:
        _validate_command_argv(argv, kind="validation")
        return argv


class AsyncOrchestratorBudgetConfig(StrictModel):
    """Per-run cost budget for an async orchestration run.

    ``max_cost`` is the inclusive ceiling on accumulated agent spend for one run,
    expressed in ``currency``. When set, the run stops claiming new work once the
    accumulated agent-session cost meets or exceeds the ceiling, lets in-flight
    worktrees settle, and records the breach reason and final cost. ``None`` (the
    default) disables the ceiling and preserves the historical unbounded behavior.
    Mirrors the shared ``BudgetConfig`` semantics (inclusive ``>=``).
    """

    max_cost: _PositiveDecimal | None = None
    currency: str = Field(default="USD", min_length=3, max_length=8)

    @field_validator("max_cost", mode="before")
    @classmethod
    def _coerce_max_cost(cls, value: object) -> object:
        return _coerce_decimal(value, field_name="max_cost")


class AsyncOrchestratorAgentCommandConfig(StrictModel):
    """Shell-free command configuration for one async orchestrator agent role."""

    argv: tuple[str, ...] = Field(min_length=1)
    resume_argv: tuple[str, ...] = ()

    @field_validator("argv", "resume_argv", mode="before")
    @classmethod
    def _coerce_argv(cls, value: object) -> object:
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return _coerce_argv_sequence(cast(Sequence[object], value), kind="agent")
        return value

    @field_validator("argv", "resume_argv")
    @classmethod
    def _validate_argv(cls, argv: tuple[str, ...]) -> tuple[str, ...]:
        _validate_command_argv(argv, kind="agent")
        return argv

    def argv_for_resume(self, resume: bool) -> tuple[str, ...]:
        """Return command arguments for a fresh or resumed agent session."""

        return (*self.argv, *self.resume_argv) if resume else self.argv


def _check_seed_worktree_build(
    *, seed_worktree_build: bool, merge_validation_worktree: bool, mirror_enabled: bool
) -> None:
    """Reject ``seed_worktree_build`` without its prerequisites.

    The build-cache snapshot is taken from the staging validation worktree, so
    that worktree must exist (``merge_validation_worktree``). The seeded
    ``.lake/build`` oleans reference the read-only packages (e.g. Mathlib) that
    the workspace mirror provides via ``symlink_paths``; without the mirror the
    worktree would have no packages and rebuild them from scratch, defeating the
    optimization. Both are therefore required.
    """

    if not seed_worktree_build:
        return
    if not merge_validation_worktree:
        raise ValueError(
            "seed_worktree_build requires merge_validation_worktree: the staging "
            "validation worktree is the source of the build-cache snapshot."
        )
    if not mirror_enabled:
        raise ValueError(
            "seed_worktree_build requires workspace_mirror.enabled: seeded "
            ".lake/build artifacts reference the read-only packages the mirror "
            "provides (e.g. Mathlib); without them the worktree would rebuild "
            "them from scratch."
        )


class AsyncOrchestratorProjectConfig(StrictModel):
    """Configuration loaded from ``<root>/config.yaml`` for async orchestration."""

    entrypoint: Path
    worker_agent_command: AsyncOrchestratorAgentCommandConfig | None = None
    reviewer_agent_command: AsyncOrchestratorAgentCommandConfig | None = None
    worktree_setup_command: AsyncOrchestratorWorktreeSetupCommandConfig | None = None
    workspace_mirror: AsyncOrchestratorWorkspaceMirrorConfig = Field(
        default_factory=AsyncOrchestratorWorkspaceMirrorConfig
    )
    # Two validation-command lifecycles, easy to confuse by name. They differ
    # in *when* they run and *what they gate*; despite the ``pre_merge_*`` name,
    # the second one runs **after** the git merge:
    #
    # * ``validation_commands`` — run in the worker's **worktree** after the
    #   worker completes, **before** the reviewer is scheduled. Optional smoke
    #   gate (e.g. quick compile). Off by default in templates. These must be
    #   read-only with respect to the worktree (or confine any writes to
    #   ``.tend/``): the worktree must be clean before review, so a command that
    #   leaves uncommitted files outside ``.tend/`` marks the tree dirty and
    #   requeues the worker every pass, looping until the run's cost budget is
    #   exhausted.
    # * ``pre_merge_validation_commands`` — run after an approved merge has
    #   been assembled, **before** the orchestrator declares the worktree CLOSED
    #   and the merge final. With the default ``merge_validation_worktree=True``
    #   path, merge assembly and validation happen in ``<root>/staging`` and the
    #   entrypoint only fast-forwards after success. With the direct-entrypoint
    #   fallback, failure resets the entrypoint to its pre-merge HEAD. In both
    #   modes the worktree returns to PENDING with the build output as feedback.
    #   Read the misleading ``pre_merge_*`` name as "post-git-merge,
    #   pre-merge-declared-final validation". This is the typical ``lake build``
    #   slot in our templates.
    #
    #   Operator contract: these commands must be interruption-safe and
    #   incrementally correct. The staging pipeline deliberately preserves
    #   gitignored output (e.g. ``.lake``) across batches *and* across the
    #   in-place retry after a signal-cancelled attempt (external kill / OOM
    #   killer), so a command must produce correct results when rerun over
    #   caches left by an interrupted predecessor — as ``lake`` does via
    #   trace-hash rebuilds of partial artifacts. A command that can be fooled
    #   by its own leftovers must clean up its own state. Commands must also
    #   not detach descendants into new sessions (``setsid``/daemonize): the
    #   orchestrator's termination guarantees cover the validation process
    #   group only, and an escaped daemon can outlive the validation and
    #   overlap a retry or the next batch.
    #
    #   Crash-signal exits (SIGSEGV and friends) get one extra guarantee, in
    #   staging-worktree mode only: the orchestrator purges staging's ignored
    #   state and re-provisions its infrastructure (mirror symlinks, setup
    #   command) before anything revalidates there, at the cost of a cold
    #   rebuild. With ``merge_validation_worktree=False`` (validation directly
    #   in the entrypoint) there is deliberately no such purge — the
    #   entrypoint is the user's own repository, and deleting its ignored
    #   files is not the orchestrator's call — so crash leftovers there are
    #   the validation command's own responsibility.
    validation_commands: tuple[AsyncOrchestratorValidationCommandConfig, ...] = ()
    pre_merge_validation_commands: tuple[AsyncOrchestratorValidationCommandConfig, ...] = ()
    merge_target_branch: str = _DEFAULT_MERGE_TARGET_BRANCH
    # When true, the merge pipeline trial-merges and runs validation in a
    # dedicated long-lived *staging* worktree and only fast-forwards the
    # pristine entrypoint to an already-validated commit (never reverting it).
    # This keeps the slow validation build off the entrypoint lock so ready-task
    # worktree creation is not starved while a build runs. See
    # ``AsyncOrchestratorConfig.merge_validation_worktree``.
    merge_validation_worktree: bool = True
    # When true (and ``merge_validation_worktree`` is on), snapshot the staging
    # worktree's ``.lake/build`` after each successful validated merge and seed
    # it into every newly-created task worktree, so the worker's first build is
    # incremental against current ``main`` rather than from scratch. See
    # ``AsyncOrchestratorConfig.seed_worktree_build``.
    seed_worktree_build: bool = False
    # Batch ready worktrees into one staging build per round instead of
    # validating each merge serially. Only effective with
    # ``merge_validation_worktree`` (the legacy in-entrypoint path is unaffected).
    # See ``AsyncOrchestratorConfig.batched_merge``.
    batched_merge: bool = True
    # Optional cap on how many MERGE worktrees one staging batch may include.
    # ``None`` preserves the historical drain-all behavior.
    max_merge_batch_size: _PositiveInt | None = None
    # When true, an approved merge whose diff changed only paths under the task
    # directory (``tasks/``) skips ``pre_merge_validation_commands`` — the
    # build-free post-merge task-tree gate still runs and must pass first. Off
    # by default so existing configs behave unchanged. See
    # ``AsyncOrchestratorConfig.skip_build_validation_for_task_only_merges``.
    skip_build_validation_for_task_only_merges: bool = False
    max_concurrent_worker_agents: _PositiveInt = 20
    max_concurrent_reviewer_agents: _PositiveInt = 20
    budget: AsyncOrchestratorBudgetConfig = Field(default_factory=AsyncOrchestratorBudgetConfig)
    agent_oom_score_adj: _OomScoreAdj | None = _DEFAULT_AGENT_OOM_SCORE_ADJ
    # When true (the default), the orchestrator removes a worktree's working
    # tree as soon as it transitions to CLOSED (i.e. after its commits have been
    # published to the merge target). Worktrees are *linked* git worktrees that
    # share the entrypoint object store, so a safely closed working tree carries
    # no unique committed state — removing it reclaims the per-worktree build
    # output (the dominant disk cost of a long run, hundreds of MB each) while
    # losing no work. As a safety backstop, cleanup skips worktrees that still
    # have non-`.tend` uncommitted changes or commits absent from the merge target.
    # On by default because accumulated worktrees otherwise exhaust the disk on
    # large runs; set it to false to keep closed worktrees on disk for post-hoc
    # inspection.
    cleanup_closed_worktrees: bool = True

    @field_validator("worker_agent_command", "reviewer_agent_command", mode="before")
    @classmethod
    def _coerce_agent_commands(cls, value: object) -> object:
        return _coerce_agent_command_config(value)

    @field_validator("worktree_setup_command", mode="before")
    @classmethod
    def _coerce_worktree_setup_command(cls, value: object) -> object:
        return _coerce_worktree_setup_command_config(value)

    @field_validator("validation_commands", "pre_merge_validation_commands", mode="before")
    @classmethod
    def _coerce_validation_commands(cls, value: object) -> object:
        return _coerce_validation_command_configs(value)

    @field_validator("merge_target_branch")
    @classmethod
    def _validate_merge_target_branch(cls, value: str) -> str:
        return _validate_merge_target_branch(value)

    @field_validator("entrypoint", mode="before")
    @classmethod
    def _coerce_entrypoint(cls, value: object, info: ValidationInfo) -> object:
        return _coerce_path_input(
            value,
            field_name=info.field_name or "entrypoint",
            json_mode=info.mode == "json",
        )

    @field_validator("entrypoint")
    @classmethod
    def _validate_entrypoint(cls, value: Path, info: ValidationInfo) -> Path:
        return _validate_path(value, field_name=info.field_name or "entrypoint")

    @model_validator(mode="after")
    def _validate_seed_worktree_build(self) -> AsyncOrchestratorProjectConfig:
        _check_seed_worktree_build(
            seed_worktree_build=self.seed_worktree_build,
            merge_validation_worktree=self.merge_validation_worktree,
            mirror_enabled=self.workspace_mirror.enabled,
        )
        return self

    def to_runtime_config(
        self,
        *,
        root: Path,
        entrypoint: Path | None = None,
    ) -> AsyncOrchestratorConfig:
        """Build runtime config for ``root``, optionally overriding the entrypoint."""

        return AsyncOrchestratorConfig(
            root=root,
            entrypoint=self.entrypoint if entrypoint is None else entrypoint,
            worker_agent_command=self.worker_agent_command,
            reviewer_agent_command=self.reviewer_agent_command,
            worktree_setup_command=self.worktree_setup_command,
            workspace_mirror=self.workspace_mirror,
            validation_commands=self.validation_commands,
            pre_merge_validation_commands=self.pre_merge_validation_commands,
            merge_target_branch=self.merge_target_branch,
            merge_validation_worktree=self.merge_validation_worktree,
            seed_worktree_build=self.seed_worktree_build,
            batched_merge=self.batched_merge,
            max_merge_batch_size=self.max_merge_batch_size,
            skip_build_validation_for_task_only_merges=(
                self.skip_build_validation_for_task_only_merges
            ),
            max_concurrent_worker_agents=self.max_concurrent_worker_agents,
            max_concurrent_reviewer_agents=self.max_concurrent_reviewer_agents,
            budget=self.budget,
            agent_oom_score_adj=self.agent_oom_score_adj,
            cleanup_closed_worktrees=self.cleanup_closed_worktrees,
        )


class AsyncOrchestratorConfig(StrictModel):
    """Top-level runtime configuration for an async orchestration run."""

    root: Path
    entrypoint: Path
    worker_agent_command: AsyncOrchestratorAgentCommandConfig | None = None
    reviewer_agent_command: AsyncOrchestratorAgentCommandConfig | None = None
    worktree_setup_command: AsyncOrchestratorWorktreeSetupCommandConfig | None = None
    workspace_mirror: AsyncOrchestratorWorkspaceMirrorConfig = Field(
        default_factory=AsyncOrchestratorWorkspaceMirrorConfig
    )
    validation_commands: tuple[AsyncOrchestratorValidationCommandConfig, ...] = ()
    pre_merge_validation_commands: tuple[AsyncOrchestratorValidationCommandConfig, ...] = ()
    merge_target_branch: str = _DEFAULT_MERGE_TARGET_BRANCH
    # See ``AsyncOrchestratorProjectConfig.merge_validation_worktree``. When
    # true the orchestrator maintains a ``<root>/staging`` worktree, runs the
    # trial merge + ``pre_merge_validation_commands`` there, and only
    # fast-forwards the entrypoint to a validated commit. The entrypoint is
    # never reset/reverted, so worktree creation only contends with the brief
    # fast-forward publish (``entrypoint_lock``) rather than the whole build.
    merge_validation_worktree: bool = True
    # Seed each new task worktree's ``.lake/build`` from a snapshot of the
    # staging worktree's build cache (taken after each successful validated
    # merge, so it tracks ``main``). Makes a worker's first ``lake build``
    # incremental instead of from scratch — far less wall-time, and shorter
    # peak-memory windows so fewer builds overlap at peak. Requires
    # ``merge_validation_worktree`` (the staging worktree is the cache source).
    seed_worktree_build: bool = False
    # Batch currently-ready worktrees into a single staging validation build per
    # round instead of validating one merge at a time: one build for K merges. A
    # one-item queue batches as size 1 (= today), so this only engages under a
    # backlog. Effective only with the staging path (``merge_validation_worktree``);
    # the legacy in-entrypoint path ignores it.
    batched_merge: bool = True
    # Optional positive cap on staging batch size. ``None`` means no cap: drain
    # every visible MERGE queue item into the batch, preserving existing behavior.
    max_merge_batch_size: _PositiveInt | None = None
    # When true, skip the expensive ``pre_merge_validation_commands`` gate for a
    # merge whose diff changed only paths under the task directory (``tasks/``).
    # The build-free post-merge task-tree gate (strict YAML parse + acyclic
    # ``depends_on`` DAG) still runs first and must pass. Enabling this is an
    # OPERATOR ASSERTION that the validation commands consume nothing under the
    # task directory (no ``include_str "tasks/..."``, no custom facets reading
    # task files) — tend does not verify that; if the build reads task
    # files, leave this off. Applies to
    # every merge path (batched staging, single staging, legacy in-entrypoint).
    # A merge that touches any non-task path runs the build gate exactly as
    # before. Off by default to preserve current behavior.
    skip_build_validation_for_task_only_merges: bool = False
    max_concurrent_worker_agents: _PositiveInt = 20
    max_concurrent_reviewer_agents: _PositiveInt = 20
    budget: AsyncOrchestratorBudgetConfig = Field(default_factory=AsyncOrchestratorBudgetConfig)
    agent_oom_score_adj: _OomScoreAdj | None = _DEFAULT_AGENT_OOM_SCORE_ADJ
    # See ``AsyncOrchestratorProjectConfig.cleanup_closed_worktrees``.
    cleanup_closed_worktrees: bool = True

    @classmethod
    def from_paths(
        cls,
        *,
        root: str | PathLike[str] | Path,
        entrypoint: str | PathLike[str] | Path,
        worker_agent_command: AsyncOrchestratorAgentCommandConfig | Sequence[str] | None = None,
        reviewer_agent_command: AsyncOrchestratorAgentCommandConfig | Sequence[str] | None = None,
        worktree_setup_command: (
            AsyncOrchestratorWorktreeSetupCommandConfig | Sequence[str] | None
        ) = None,
        workspace_mirror: AsyncOrchestratorWorkspaceMirrorConfig | None = None,
        validation_commands: (
            Sequence[AsyncOrchestratorValidationCommandConfig | Sequence[str]] | None
        ) = None,
        pre_merge_validation_commands: (
            Sequence[AsyncOrchestratorValidationCommandConfig | Sequence[str]] | None
        ) = None,
        merge_target_branch: str = _DEFAULT_MERGE_TARGET_BRANCH,
        merge_validation_worktree: bool = True,
        seed_worktree_build: bool = False,
        batched_merge: bool = True,
        max_merge_batch_size: int | None = None,
        skip_build_validation_for_task_only_merges: bool = False,
        max_concurrent_worker_agents: int = 20,
        max_concurrent_reviewer_agents: int = 20,
        budget: AsyncOrchestratorBudgetConfig | None = None,
        agent_oom_score_adj: int | None = _DEFAULT_AGENT_OOM_SCORE_ADJ,
        cleanup_closed_worktrees: bool = True,
    ) -> AsyncOrchestratorConfig:
        """Build config from orchestration root and entrypoint repository paths."""

        return cls.model_validate(
            {
                "root": root,
                "entrypoint": entrypoint,
                "worker_agent_command": worker_agent_command,
                "reviewer_agent_command": reviewer_agent_command,
                "worktree_setup_command": worktree_setup_command,
                "workspace_mirror": (
                    workspace_mirror
                    if workspace_mirror is not None
                    else AsyncOrchestratorWorkspaceMirrorConfig()
                ),
                "validation_commands": validation_commands,
                "pre_merge_validation_commands": pre_merge_validation_commands,
                "merge_target_branch": merge_target_branch,
                "merge_validation_worktree": merge_validation_worktree,
                "seed_worktree_build": seed_worktree_build,
                "batched_merge": batched_merge,
                "max_merge_batch_size": max_merge_batch_size,
                "skip_build_validation_for_task_only_merges": (
                    skip_build_validation_for_task_only_merges
                ),
                "max_concurrent_worker_agents": max_concurrent_worker_agents,
                "max_concurrent_reviewer_agents": max_concurrent_reviewer_agents,
                "budget": budget if budget is not None else AsyncOrchestratorBudgetConfig(),
                "agent_oom_score_adj": agent_oom_score_adj,
                "cleanup_closed_worktrees": cleanup_closed_worktrees,
            }
        )

    @field_validator("worker_agent_command", "reviewer_agent_command", mode="before")
    @classmethod
    def _coerce_agent_commands(cls, value: object) -> object:
        return _coerce_agent_command_config(value)

    @field_validator("worktree_setup_command", mode="before")
    @classmethod
    def _coerce_worktree_setup_command(cls, value: object) -> object:
        return _coerce_worktree_setup_command_config(value)

    @field_validator("validation_commands", "pre_merge_validation_commands", mode="before")
    @classmethod
    def _coerce_validation_commands(cls, value: object) -> object:
        return _coerce_validation_command_configs(value)

    @field_validator("merge_target_branch")
    @classmethod
    def _validate_merge_target_branch(cls, value: str) -> str:
        return _validate_merge_target_branch(value)

    @field_validator("root", "entrypoint", mode="before")
    @classmethod
    def _coerce_paths(cls, value: object, info: ValidationInfo) -> object:
        return _coerce_path_input(
            value,
            field_name=info.field_name or "path",
            json_mode=info.mode == "json",
        )

    @field_validator("root", "entrypoint")
    @classmethod
    def _validate_paths(cls, value: Path, info: ValidationInfo) -> Path:
        return _validate_path(value, field_name=info.field_name or "path")

    @model_validator(mode="after")
    def _validate_seed_worktree_build(self) -> AsyncOrchestratorConfig:
        _check_seed_worktree_build(
            seed_worktree_build=self.seed_worktree_build,
            merge_validation_worktree=self.merge_validation_worktree,
            mirror_enabled=self.workspace_mirror.enabled,
        )
        return self


def _coerce_agent_command_config(value: object) -> object:
    if value is None or isinstance(value, AsyncOrchestratorAgentCommandConfig):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return AsyncOrchestratorAgentCommandConfig(
            argv=_coerce_argv_sequence(cast(Sequence[object], value), kind="agent")
        )
    return value


def _coerce_worktree_setup_command_config(value: object) -> object:
    if value is None or isinstance(value, AsyncOrchestratorWorktreeSetupCommandConfig):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return AsyncOrchestratorWorktreeSetupCommandConfig(
            argv=_coerce_argv_sequence(cast(Sequence[object], value), kind="worktree setup")
        )
    return value


def _coerce_validation_command_configs(value: object) -> object:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return value
    commands: list[object] = []
    for item in cast(Sequence[object], value):
        if isinstance(item, AsyncOrchestratorValidationCommandConfig):
            commands.append(item)
        elif isinstance(item, Mapping):
            commands.append(cast(object, item))
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            commands.append(
                AsyncOrchestratorValidationCommandConfig(
                    argv=_coerce_argv_sequence(cast(Sequence[object], item), kind="validation")
                )
            )
        else:
            commands.append(item)
    return tuple(commands)


def _coerce_decimal(value: object, *, field_name: str) -> object:
    """Coerce a string or integer to ``Decimal`` for monetary config fields.

    Strings (e.g. ``"50.00"``) and integers are accepted so money can be expressed
    exactly in YAML/JSON config and on the CLI without binary-float rounding.
    ``float`` is rejected because it cannot represent decimal money exactly.
    ``None`` and existing ``Decimal`` values pass through unchanged. Mirrors the
    shared ``config._coerce_decimal``.
    """

    if value is None or isinstance(value, Decimal):
        return value
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a decimal amount, not a boolean")
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError(f"{field_name} must be a non-empty decimal amount")
        try:
            return Decimal(text)
        except InvalidOperation as exc:
            raise ValueError(f"{field_name} must be a valid decimal amount: {value!r}") from exc
    return value


def _validate_command_argv(argv: tuple[str, ...], *, kind: str) -> None:
    for index, argument in enumerate(argv):
        if not argument.strip() or "\x00" in argument:
            raise ValueError(
                f"{kind} command arguments must not be blank or contain NUL: "
                f"argv[{index}]"
            )


def _validate_merge_target_branch(value: str) -> str:
    if not value.strip() or "\x00" in value:
        raise ValueError("merge target branch must not be blank or contain NUL")
    return value


def _validate_worktree_setup_placeholders(argv: tuple[str, ...]) -> None:
    for index, argument in enumerate(argv):
        try:
            parsed = tuple(_FORMATTER.parse(argument))
        except ValueError as exc:
            raise ValueError(
                "worktree setup command placeholders must be valid Python format "
                f"strings: argv[{index}]"
            ) from exc
        for _, field_name, format_spec, conversion in parsed:
            if field_name is None:
                continue
            if (
                field_name not in _WORKTREE_SETUP_PLACEHOLDERS
                or format_spec
                or conversion is not None
            ):
                raise ValueError(
                    "worktree setup command placeholders must be exactly {entrypoint} "
                    f"or {{worktree}}: argv[{index}]"
                )


def _coerce_argv_sequence(value: Sequence[object], *, kind: str) -> tuple[str, ...]:
    argv: list[str] = []
    for argument in value:
        if not isinstance(argument, str):
            raise ValueError(f"{kind} command arguments must be strings")
        argv.append(argument)
    return tuple(argv)


def _coerce_path_input(value: object, *, field_name: str, json_mode: bool) -> object:
    if json_mode:
        if isinstance(value, str):
            _validate_path_text(value, field_name=field_name)
        return value
    return _coerce_path(value, field_name=field_name)


def _coerce_path(value: object, *, field_name: str) -> Path:
    if isinstance(value, Path):
        return value
    if isinstance(value, str):
        _validate_path_text(value, field_name=field_name)
        return Path(value)
    if isinstance(value, PathLike):
        path_text = cast(object, fspath(cast(PathLike[str] | PathLike[bytes], value)))
        if not isinstance(path_text, str):
            raise ValueError(f"{field_name} must be a path-like value")
        _validate_path_text(path_text, field_name=field_name)
        return Path(path_text)
    raise ValueError(f"{field_name} must be a path-like value")


def _validate_path(value: Path, *, field_name: str) -> Path:
    _validate_path_text(str(value), field_name=field_name)
    return value


def _validate_path_text(value: str, *, field_name: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must not be blank")
    if "\x00" in value:
        raise ValueError(f"{field_name} must not contain NUL")
