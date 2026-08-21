from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Mapping, Sequence
from decimal import Decimal
from io import StringIO
from pathlib import Path
from typing import cast

import pytest
from tests.orchestrator.store_helpers import seed_store_state

import tend.orchestrator.cli as cli_module
from tend._common.config_files import read_config_model
from tend.agent.config import AgentConfig, RuntimeConfigOverrides
from tend.llm.usage import Cost, TokenUsage, Usage
from tend.orchestrator.cli import AsyncOrchestratorCliExitCode, run_cli
from tend.orchestrator.config import (
    AsyncOrchestratorConfig,
    AsyncOrchestratorProjectConfig,
)
from tend.orchestrator.control_store import SQLiteAsyncOrchestratorStore
from tend.orchestrator.orchestrator import (
    AsyncOrchestratorBudgetStop,
    AsyncOrchestratorRunResult,
    AsyncOrchestratorRunSummary,
)
from tend.orchestrator.root_lock import AsyncOrchestratorRootLock
from tend.orchestrator.state import AsyncOrchestratorWorktree, WorktreeState
from tend.orchestrator.task_manager import TaskManager
from tend.orchestrator.tasks import Task, TaskStatus
from tend.prompts import builtin_prompts_root, load_prompt


def _seed_store(
    root: Path,
    *,
    task_manager: TaskManager | None = None,
    worktrees: tuple[AsyncOrchestratorWorktree, ...] = (),
) -> SQLiteAsyncOrchestratorStore:
    return seed_store_state(root, task_manager=task_manager, worktrees=worktrees)


async def test_cli_reads_root_config_and_calls_orchestrator_run(tmp_path: Path) -> None:
    seen_configs: list[AsyncOrchestratorConfig] = []
    root = tmp_path / "orch"
    entrypoint = tmp_path / "repo"
    entrypoint.mkdir()

    class RecordingOrchestrator:
        def __init__(self, config: AsyncOrchestratorConfig) -> None:
            self.config = config

        async def run(self) -> AsyncOrchestratorRunResult:
            logging.getLogger("tend.orchestrator.test").info("recorded run log")
            seen_configs.append(self.config)
            return AsyncOrchestratorRunResult(
                root=self.config.root,
                entrypoint=self.config.entrypoint,
            )

    await run_cli(
        # --no-build-gate so this test controls pre_merge_validation_commands itself.
        [
            "init",
            "--root",
            str(root),
            "--entrypoint",
            str(entrypoint),
            "--no-build-gate",
            "--mirror-enabled",
            "--seed-worktree-build",
            "--no-batched-merge",
        ],
        stdout=StringIO(),
    )
    config_path = root / "config.yaml"
    config_text = config_path.read_text(encoding="utf-8")
    config_text = config_text.replace(
        "pre_merge_validation_commands: []",
        "\n".join(
            (
                "pre_merge_validation_commands:",
                "- argv:",
                "  - uv",
                "  - run",
                "  - pytest",
                "  - -m",
                "  - not live",
            )
        ),
    )
    config_text = config_text.replace(
        "validation_commands: []",
        "validation_commands:\n- argv:\n  - uv\n  - run\n  - ruff\n  - check",
    )
    config_text = config_text.replace(
        "merge_target_branch: main",
        "merge_target_branch: release",
    )
    config_path.write_text(config_text, encoding="utf-8")

    exit_code = await run_cli(
        [
            "run",
            "--root",
            str(root),
            "--worker-agent-command",
            "tend-agent --prompt worker",
            "--worker-agent-resume-args=--resume-session",
            "--reviewer-agent-command",
            "tend-agent --prompt reviewer",
            "--reviewer-agent-resume-args=--resume-session --json",
            "--worktree-setup-command",
            "cp --archive {entrypoint}/.lake {worktree}/",
            "--max-concurrent-worker-agents",
            "3",
            "--max-concurrent-reviewer-agents",
            "2",
        ],
        orchestrator_factory=RecordingOrchestrator,
    )

    assert exit_code == int(AsyncOrchestratorCliExitCode.SUCCESS)
    assert len(seen_configs) == 1
    assert seen_configs[0].root == root.resolve()
    assert seen_configs[0].entrypoint == entrypoint.resolve()
    assert seen_configs[0].worker_agent_command is not None
    assert seen_configs[0].worker_agent_command.argv == ("tend-agent", "--prompt", "worker")
    assert seen_configs[0].worker_agent_command.resume_argv == ("--resume-session",)
    assert seen_configs[0].reviewer_agent_command is not None
    assert seen_configs[0].reviewer_agent_command.argv == (
        "tend-agent",
        "--prompt",
        "reviewer",
    )
    assert seen_configs[0].reviewer_agent_command.resume_argv == (
        "--resume-session",
        "--json",
    )
    assert seen_configs[0].worktree_setup_command is not None
    assert seen_configs[0].worktree_setup_command.argv == (
        "cp",
        "--archive",
        "{entrypoint}/.lake",
        "{worktree}/",
    )
    assert [command.argv for command in seen_configs[0].validation_commands] == [
        ("uv", "run", "ruff", "check"),
    ]
    assert [command.argv for command in seen_configs[0].pre_merge_validation_commands] == [
        ("uv", "run", "pytest", "-m", "not live"),
    ]
    assert seen_configs[0].merge_target_branch == "release"
    assert seen_configs[0].workspace_mirror.enabled is True
    assert seen_configs[0].seed_worktree_build is True
    assert seen_configs[0].batched_merge is False
    assert seen_configs[0].max_concurrent_worker_agents == 3
    assert seen_configs[0].max_concurrent_reviewer_agents == 2
    log_text = (root / "logs.txt").read_text(encoding="utf-8")
    assert "writing async orchestrator logs to:" in log_text
    assert "INFO:tend.orchestrator.test:recorded run log" in log_text


async def test_cli_max_cost_flag_overrides_budget(tmp_path: Path) -> None:
    seen_configs: list[AsyncOrchestratorConfig] = []
    root = tmp_path / "orch"
    entrypoint = tmp_path / "repo"
    entrypoint.mkdir()

    class RecordingOrchestrator:
        def __init__(self, config: AsyncOrchestratorConfig) -> None:
            self.config = config

        async def run(self) -> AsyncOrchestratorRunResult:
            seen_configs.append(self.config)
            return AsyncOrchestratorRunResult(
                root=self.config.root,
                entrypoint=self.config.entrypoint,
            )

    await run_cli(
        ["init", "--root", str(root), "--entrypoint", str(entrypoint)],
        stdout=StringIO(),
    )

    exit_code = await run_cli(
        ["run", "--root", str(root), "--max-cost", "12.50"],
        orchestrator_factory=RecordingOrchestrator,
    )

    assert exit_code == int(AsyncOrchestratorCliExitCode.SUCCESS)
    assert len(seen_configs) == 1
    assert seen_configs[0].budget.max_cost == Decimal("12.50")


async def test_cli_rejects_non_finite_max_cost(tmp_path: Path) -> None:
    root = tmp_path / "orch"

    for value in ("NaN", "Infinity", "-Infinity"):
        stderr = StringIO()

        exit_code = await run_cli(
            ["run", "--root", str(root), f"--max-cost={value}"],
            stderr=stderr,
        )

        assert exit_code == int(AsyncOrchestratorCliExitCode.CONFIGURATION_OR_USAGE)
        assert "error[cli_usage_error]" in stderr.getvalue()
        assert "max cost must be finite" in stderr.getvalue()


async def test_cli_run_prints_budget_stop_summary(tmp_path: Path) -> None:
    root = tmp_path / "orch"
    entrypoint = tmp_path / "repo"
    entrypoint.mkdir()
    stdout = StringIO()

    class BudgetStoppedOrchestrator:
        def __init__(self, config: AsyncOrchestratorConfig) -> None:
            self.config = config

        async def run(self) -> AsyncOrchestratorRunResult:
            return AsyncOrchestratorRunResult(
                root=self.config.root,
                entrypoint=self.config.entrypoint,
                budget_stop=AsyncOrchestratorBudgetStop(
                    breach_accumulated_cost="10.00",
                    accumulated_cost="10.00",
                    max_cost="10.00",
                    currency="USD",
                ),
            )

    await run_cli(
        ["init", "--root", str(root), "--entrypoint", str(entrypoint)],
        stdout=StringIO(),
    )

    exit_code = await run_cli(
        ["run", "--root", str(root)],
        stdout=stdout,
        orchestrator_factory=BudgetStoppedOrchestrator,
    )

    assert exit_code == int(AsyncOrchestratorCliExitCode.SUCCESS)
    assert "stopped on cost ceiling" in stdout.getvalue()
    assert "10.00 USD" in stdout.getvalue()
    # When ``accumulated_cost`` equals the breach amount the suffix is elided to
    # avoid noise; the next test covers the divergent case.
    assert "breach_cost=" not in stdout.getvalue()


async def test_cli_run_prints_budget_stop_summary_with_breach_cost_when_diverged(
    tmp_path: Path,
) -> None:
    # When in-flight work settles after the freeze, ``accumulated_cost`` can
    # creep above the first-breach total while ``breach_accumulated_cost`` stays
    # pinned. The CLI surface mirrors sync #70 by emitting a ``breach_cost=``
    # tag so operators can tell the two amounts apart.
    root = tmp_path / "orch"
    entrypoint = tmp_path / "repo"
    entrypoint.mkdir()
    stdout = StringIO()

    class BudgetStoppedOrchestrator:
        def __init__(self, config: AsyncOrchestratorConfig) -> None:
            self.config = config

        async def run(self) -> AsyncOrchestratorRunResult:
            return AsyncOrchestratorRunResult(
                root=self.config.root,
                entrypoint=self.config.entrypoint,
                budget_stop=AsyncOrchestratorBudgetStop(
                    breach_accumulated_cost="10.00",
                    accumulated_cost="11.50",
                    max_cost="10.00",
                    currency="USD",
                ),
            )

    await run_cli(
        ["init", "--root", str(root), "--entrypoint", str(entrypoint)],
        stdout=StringIO(),
    )

    exit_code = await run_cli(
        ["run", "--root", str(root)],
        stdout=stdout,
        orchestrator_factory=BudgetStoppedOrchestrator,
    )

    assert exit_code == int(AsyncOrchestratorCliExitCode.SUCCESS)
    output = stdout.getvalue()
    assert "accumulated 11.50 USD" in output
    assert "breach_cost=10.00 USD" in output
    assert "max_cost 10.00 USD" in output


async def test_cli_auto_resumes_saved_state(tmp_path: Path) -> None:
    seen_worktrees: list[dict[str, AsyncOrchestratorWorktree]] = []
    seen_resume_flags: list[bool] = []
    root = tmp_path / "orch"
    entrypoint = tmp_path / "repo"
    worktree_path = tmp_path / "worktree"
    entrypoint.mkdir()
    worktree_path.mkdir()
    task = Task(id="task-1", title="Task", summary="Task", description="Do it.")
    saved_worktree = AsyncOrchestratorWorktree(
        worktree_id="worktree_000001",
        task_id=task.id,
        path=worktree_path,
        head="abc123",
        state=WorktreeState.WORKER_RUNNING,
        worker_session_started=True,
    )

    class RecordingOrchestrator:
        def __init__(
            self,
            config: AsyncOrchestratorConfig,
            *,
            check_resume_health: bool = False,
        ) -> None:
            self.config = config
            self.store = SQLiteAsyncOrchestratorStore(config.root)
            seen_resume_flags.append(check_resume_health)
            if check_resume_health:
                self.store.reset_running_worktrees()

        async def run(self) -> AsyncOrchestratorRunResult:
            seen_worktrees.append(
                {worktree.worktree_id: worktree for worktree in self.store.list_worktrees()}
            )
            return AsyncOrchestratorRunResult(
                root=self.config.root,
                entrypoint=self.config.entrypoint,
            )

    await run_cli(
        ["init", "--root", str(root), "--entrypoint", str(entrypoint)],
        stdout=StringIO(),
    )
    _seed_store(
        root,
        task_manager=TaskManager(tasks=[task]),
        worktrees=(saved_worktree,),
    )

    exit_code = await run_cli(
        ["run", "--root", str(root)],
        orchestrator_factory=RecordingOrchestrator,
    )

    assert exit_code == int(AsyncOrchestratorCliExitCode.SUCCESS)
    assert seen_resume_flags == [True]
    assert len(seen_worktrees) == 1
    assert seen_worktrees[0]["worktree_000001"].state is WorktreeState.PENDING
    assert seen_worktrees[0]["worktree_000001"].worker_session_started is True


async def test_cli_run_refuses_locked_root(tmp_path: Path) -> None:
    stderr = StringIO()
    root = tmp_path / "orch"
    entrypoint = tmp_path / "repo"
    entrypoint.mkdir()

    class RecordingOrchestrator:
        def __init__(self, config: AsyncOrchestratorConfig) -> None:
            self.config = config

        async def run(self) -> AsyncOrchestratorRunResult:
            raise AssertionError("orchestrator should not run when root is locked")

    await run_cli(
        ["init", "--root", str(root), "--entrypoint", str(entrypoint)],
        stdout=StringIO(),
    )

    with AsyncOrchestratorRootLock.acquire(root, owner="test", sync_writes=False):
        exit_code = await run_cli(
            ["run", "--root", str(root)],
            stderr=stderr,
            orchestrator_factory=RecordingOrchestrator,
        )

    assert exit_code == int(AsyncOrchestratorCliExitCode.CONFIGURATION_OR_USAGE)
    assert "error[root_lock_error]" in stderr.getvalue()
    assert "already locked" in stderr.getvalue()


async def test_cli_run_releases_root_lock_after_success(tmp_path: Path) -> None:
    root = tmp_path / "orch"
    entrypoint = tmp_path / "repo"
    entrypoint.mkdir()

    class RecordingOrchestrator:
        def __init__(self, config: AsyncOrchestratorConfig) -> None:
            self.config = config

        async def run(self) -> AsyncOrchestratorRunResult:
            return AsyncOrchestratorRunResult(
                root=self.config.root,
                entrypoint=self.config.entrypoint,
            )

    await run_cli(
        ["init", "--root", str(root), "--entrypoint", str(entrypoint)],
        stdout=StringIO(),
    )

    exit_code = await run_cli(
        ["run", "--root", str(root)],
        orchestrator_factory=RecordingOrchestrator,
    )

    assert exit_code == int(AsyncOrchestratorCliExitCode.SUCCESS)
    with AsyncOrchestratorRootLock.acquire(root, owner="test", sync_writes=False) as lock:
        assert not lock.released


async def test_cli_fresh_ignores_saved_state(tmp_path: Path) -> None:
    seen_worktrees: list[tuple[AsyncOrchestratorWorktree, ...]] = []
    root = tmp_path / "orch"
    entrypoint = tmp_path / "repo"
    entrypoint.mkdir()
    saved_worktree = AsyncOrchestratorWorktree(
        worktree_id="worktree_000001",
        path=tmp_path / "worktree",
        head="abc123",
    )

    class RecordingOrchestrator:
        def __init__(self, config: AsyncOrchestratorConfig) -> None:
            self.config = config
            self.store = SQLiteAsyncOrchestratorStore(config.root)

        async def run(self) -> AsyncOrchestratorRunResult:
            seen_worktrees.append(self.store.list_worktrees())
            return AsyncOrchestratorRunResult(
                root=self.config.root,
                entrypoint=self.config.entrypoint,
            )

    await run_cli(
        ["init", "--root", str(root), "--entrypoint", str(entrypoint)],
        stdout=StringIO(),
    )
    _seed_store(root, worktrees=(saved_worktree,))

    exit_code = await run_cli(
        ["run", "--root", str(root), "--fresh"],
        orchestrator_factory=RecordingOrchestrator,
    )

    assert exit_code == int(AsyncOrchestratorCliExitCode.SUCCESS)
    assert seen_worktrees == [()]


async def test_cli_status_reports_state_and_usage_without_writes(tmp_path: Path) -> None:
    stdout = StringIO()
    root = tmp_path / "orch"
    open_task = Task(id="task-open", title="Open", summary="Open", description="Do it.")
    complete_task = Task(
        id="task-complete",
        title="Complete", summary="Complete",
        description="Done.",
        status=TaskStatus.COMPLETE,
    )
    worktrees: list[AsyncOrchestratorWorktree] = []
    for index, worktree_state in enumerate(
        (
            WorktreeState.PENDING,
            WorktreeState.WORKER_RUNNING,
            WorktreeState.REVIEW,
            WorktreeState.MERGE,
        ),
        start=1,
    ):
        worktrees.append(
            AsyncOrchestratorWorktree(
                worktree_id=f"worktree_{index:06d}",
                task_id=open_task.id,
                path=tmp_path / f"worktree-{index}",
                head="abc123",
                state=worktree_state,
            )
        )
    usage = Usage(
        tokens=TokenUsage(
            input_tokens=10,
            output_tokens=5,
            reasoning_tokens=2,
            cache_read_tokens=3,
        ),
        cost=Cost(amount=Decimal("1.25"), currency="USD"),
        model_requests=4,
        retry_attempts=1,
        tool_calls=9,
    )

    worktrees[-1] = worktrees[-1].model_copy(
        update={"worker_session_started": True, "worker_session_usage": usage}
    )
    _seed_store(
        root,
        task_manager=TaskManager(tasks=[open_task, complete_task]),
        worktrees=tuple(worktrees),
    )

    exit_code = await run_cli(["status", "--root", str(root)], stdout=stdout)

    assert exit_code == int(AsyncOrchestratorCliExitCode.SUCCESS)
    text = stdout.getvalue()
    assert "async orchestrator status" in text
    assert "state: loaded" in text
    assert "tasks: total=2, open=1, complete=1" in text
    assert (
        "worktrees: total=4, pending=1, worker_running=1, review=1, merge=1, closed=0"
        in text
    )
    assert "inferred queues: worker=1, reviewer=1, merge=1" in text
    assert "usage: loaded" in text
    assert "aggregate usage: input=10, output=5" in text
    assert "cache_read=3" in text
    assert "reasoning=2" in text
    assert "model_requests=4" in text
    assert "cost=1.2500 USD" in text
    assert not (root / "state.json").exists()
    assert not (root / "usage.json").exists()


async def test_cli_export_state_json_dumps_sqlite_state(tmp_path: Path) -> None:
    stdout = StringIO()
    root = tmp_path / "orch"
    task = Task(id="task-open", title="Open", summary="Open", description="Do it.")
    usage = Usage(
        tokens=TokenUsage(input_tokens=3, output_tokens=2),
        cost=Cost(amount=Decimal("0.50"), currency="USD"),
        model_requests=1,
    )
    worktree = AsyncOrchestratorWorktree(
        worktree_id="worktree_000001",
        task_id=task.id,
        path=tmp_path / "worktree",
        head="abc123",
        worker_session_started=True,
        worker_session_usage=usage,
    )
    _seed_store(root, task_manager=TaskManager(tasks=[task]), worktrees=(worktree,))

    exit_code = await run_cli(
        ["export-state", "--root", str(root), "--json"],
        stdout=stdout,
    )

    assert exit_code == int(AsyncOrchestratorCliExitCode.SUCCESS)
    payload = json.loads(stdout.getvalue())
    assert payload["worktrees"] == [worktree.model_dump(mode="json")]
    assert payload["task_snapshot"]["tasks"][0]["id"] == task.id
    assert payload["usage"] == usage.model_dump(mode="json")


async def test_cli_export_state_missing_db_is_error_without_creating_files(
    tmp_path: Path,
) -> None:
    stdout = StringIO()
    stderr = StringIO()
    root = tmp_path / "missing-orch"

    exit_code = await run_cli(
        ["export-state", "--root", str(root), "--json"],
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == int(AsyncOrchestratorCliExitCode.CONFIGURATION_OR_USAGE)
    assert stdout.getvalue() == ""
    assert "error[state_missing]" in stderr.getvalue()
    assert "async orchestrator state database does not exist" in stderr.getvalue()
    assert not root.exists()
    assert not (root / "orchestrator.sqlite").exists()


async def test_cli_status_reports_missing_state_and_usage(tmp_path: Path) -> None:
    stdout = StringIO()
    root = tmp_path / "missing-orch"

    exit_code = await run_cli(["status", "--root", str(root)], stdout=stdout)

    assert exit_code == int(AsyncOrchestratorCliExitCode.SUCCESS)
    text = stdout.getvalue()
    assert f"root: {root.resolve()}" in text
    assert "state: missing" in text
    assert "tasks: unavailable" in text
    assert "worktrees: unavailable" in text
    assert "inferred queues: unavailable" in text
    assert "usage: missing" in text
    assert not root.exists()


async def test_cli_reports_missing_required_paths() -> None:
    stderr = StringIO()

    exit_code = await run_cli([], stderr=stderr)

    assert exit_code == int(AsyncOrchestratorCliExitCode.CONFIGURATION_OR_USAGE)
    assert "error[cli_usage_error]" in stderr.getvalue()


async def test_cli_init_creates_async_orchestration_root(tmp_path: Path) -> None:
    stdout = StringIO()
    root = tmp_path / "orch"

    exit_code = await run_cli(["init", "--root", str(root)], stdout=stdout)

    assert exit_code == int(AsyncOrchestratorCliExitCode.SUCCESS)
    assert (root / ".tend-root").read_text(encoding="utf-8")
    assert (root / "worktrees").is_dir()
    assert (root / "sessions").is_dir()
    project_config = read_config_model(
        root / "config.yaml",
        AsyncOrchestratorProjectConfig,
        kind="async orchestrator config",
    )
    assert project_config.entrypoint == Path.cwd().resolve()
    assert "initialized async orchestration root" in stdout.getvalue()
    assert "config:" in stdout.getvalue()
    # Generic projects opt into their own post-merge validation command.
    assert project_config.pre_merge_validation_commands == ()


async def test_cli_init_build_command_override(tmp_path: Path) -> None:
    root = tmp_path / "orch"

    exit_code = await run_cli(
        [
            "init",
            "--root",
            str(root),
            "--build-command",
            "make check",
            "--build-timeout-seconds",
            "60",
        ],
        stdout=StringIO(),
    )

    assert exit_code == int(AsyncOrchestratorCliExitCode.SUCCESS)
    project_config = read_config_model(
        root / "config.yaml",
        AsyncOrchestratorProjectConfig,
        kind="async orchestrator config",
    )
    assert [command.argv for command in project_config.pre_merge_validation_commands] == [
        ("make", "check"),
    ]
    assert project_config.pre_merge_validation_commands[0].timeout_seconds == 60.0


async def test_cli_init_no_build_gate(tmp_path: Path) -> None:
    root = tmp_path / "orch"

    exit_code = await run_cli(
        ["init", "--root", str(root), "--no-build-gate"],
        stdout=StringIO(),
    )

    assert exit_code == int(AsyncOrchestratorCliExitCode.SUCCESS)
    project_config = read_config_model(
        root / "config.yaml",
        AsyncOrchestratorProjectConfig,
        kind="async orchestrator config",
    )
    assert project_config.pre_merge_validation_commands == ()


async def test_cli_init_workspace_mirror_default_is_disabled(tmp_path: Path) -> None:
    root = tmp_path / "orch"

    exit_code = await run_cli(
        ["init", "--root", str(root), "--no-build-gate"],
        stdout=StringIO(),
    )

    assert exit_code == int(AsyncOrchestratorCliExitCode.SUCCESS)
    project_config = read_config_model(
        root / "config.yaml",
        AsyncOrchestratorProjectConfig,
        kind="async orchestrator config",
    )
    assert project_config.workspace_mirror.enabled is False
    assert project_config.workspace_mirror.symlink_paths == []
    assert project_config.workspace_mirror.exclude_names == []
    assert project_config.workspace_mirror.exclude_paths == []


async def test_cli_init_writes_workspace_mirror_block_from_flags(tmp_path: Path) -> None:
    root = tmp_path / "orch"

    exit_code = await run_cli(
        [
            "init",
            "--root",
            str(root),
            "--no-build-gate",
            "--mirror-enabled",
            "--symlink-path",
            ".lake/packages/mathlib",
            "--symlink-path",
            ".cache",
            "--mirror-exclude-name",
            ".pytest_cache",
            "--mirror-exclude-path",
            "build/intermediate",
            "--mirror-reflink",
            "never",
        ],
        stdout=StringIO(),
    )

    assert exit_code == int(AsyncOrchestratorCliExitCode.SUCCESS)
    project_config = read_config_model(
        root / "config.yaml",
        AsyncOrchestratorProjectConfig,
        kind="async orchestrator config",
    )
    mirror = project_config.workspace_mirror
    assert mirror.enabled is True
    assert mirror.symlink_paths == [".lake/packages/mathlib", ".cache"]
    assert mirror.exclude_names == [".pytest_cache"]
    assert mirror.exclude_paths == ["build/intermediate"]
    assert mirror.reflink_mode.value == "never"


async def test_cli_init_rejects_invalid_mirror_reflink(tmp_path: Path) -> None:
    root = tmp_path / "orch"
    stderr = StringIO()

    exit_code = await run_cli(
        [
            "init",
            "--root",
            str(root),
            "--no-build-gate",
            "--mirror-reflink",
            "bogus",
        ],
        stderr=stderr,
    )

    assert exit_code == int(AsyncOrchestratorCliExitCode.CONFIGURATION_OR_USAGE)
    assert "mirror reflink mode" in stderr.getvalue()


async def test_cli_init_refuses_root_inside_entrypoint(tmp_path: Path) -> None:
    stderr = StringIO()
    entrypoint = tmp_path / "repo"
    root = entrypoint / ".tend"
    entrypoint.mkdir()

    exit_code = await run_cli(
        ["init", "--root", str(root), "--entrypoint", str(entrypoint)],
        stderr=stderr,
    )

    assert exit_code == int(AsyncOrchestratorCliExitCode.CONFIGURATION_OR_USAGE)
    assert not root.exists()
    assert "root must be outside the entrypoint repository" in stderr.getvalue()
    assert "choose a root outside the source repository" in stderr.getvalue()


async def test_cli_init_refuses_root_equal_to_entrypoint(tmp_path: Path) -> None:
    stderr = StringIO()
    entrypoint = tmp_path / "repo"
    entrypoint.mkdir()

    exit_code = await run_cli(
        ["init", "--root", str(entrypoint), "--entrypoint", str(entrypoint)],
        stderr=stderr,
    )

    assert exit_code == int(AsyncOrchestratorCliExitCode.CONFIGURATION_OR_USAGE)
    assert not (entrypoint / ".tend-root").exists()
    assert not (entrypoint / "config.yaml").exists()
    assert "root must be outside the entrypoint repository" in stderr.getvalue()


async def test_cli_init_can_configure_cow_copy_dirs(tmp_path: Path) -> None:
    root = tmp_path / "orch"
    entrypoint = tmp_path / "entrypoint"
    entrypoint.mkdir()

    exit_code = await run_cli(
        [
            "init",
            "--root",
            str(root),
            "--entrypoint",
            str(entrypoint),
            "--cow",
            "--copy_dir",
            ".lake",
        ],
        stdout=StringIO(),
    )

    assert exit_code == int(AsyncOrchestratorCliExitCode.SUCCESS)
    project_config = read_config_model(
        root / "config.yaml",
        AsyncOrchestratorProjectConfig,
        kind="async orchestrator config",
    )
    assert project_config.worktree_setup_command is not None
    assert project_config.worktree_setup_command.argv == (
        "cp",
        "--archive",
        "--reflink=always",
        "{entrypoint}/.lake",
        "{worktree}/",
    )


async def test_cli_init_cow_requires_copy_dir(tmp_path: Path) -> None:
    stderr = StringIO()
    root = tmp_path / "orch"

    exit_code = await run_cli(["init", "--root", str(root), "--cow"], stderr=stderr)

    assert exit_code == int(AsyncOrchestratorCliExitCode.CONFIGURATION_OR_USAGE)
    assert "--cow requires" in stderr.getvalue()
    assert not root.exists()


async def test_cli_init_tend_agent_writes_commands_prompts_scripts_and_configs(
    tmp_path: Path,
) -> None:
    root = tmp_path / "orch"
    entrypoint = tmp_path / "entrypoint"
    entrypoint.mkdir()

    exit_code = await run_cli(
        ["init", "--root", str(root), "--entrypoint", str(entrypoint), "--agent", "tend"],
        stdout=StringIO(),
    )

    assert exit_code == int(AsyncOrchestratorCliExitCode.SUCCESS)
    project_config = read_config_model(
        root / "config.yaml",
        AsyncOrchestratorProjectConfig,
        kind="async orchestrator config",
    )
    worker_script = root / "bin" / "worker-agent.sh"
    reviewer_script = root / "bin" / "reviewer-agent.sh"
    assert project_config.worker_agent_command is not None
    assert project_config.worker_agent_command.argv == (str(worker_script),)
    assert project_config.worker_agent_command.resume_argv == ("--resume",)
    assert project_config.reviewer_agent_command is not None
    assert project_config.reviewer_agent_command.argv == (str(reviewer_script),)
    assert project_config.reviewer_agent_command.resume_argv == ("--resume",)
    assert worker_script.stat().st_mode & 0o111
    assert reviewer_script.stat().st_mode & 0o111
    assert "TEND_AGENT_BIN" in worker_script.read_text(encoding="utf-8")

    worker_agent_path = root / ".tend" / "worker-agent.yaml"
    reviewer_agent_path = root / ".tend" / "reviewer-agent.yaml"
    worker_agent = read_config_model(
        worker_agent_path,
        AgentConfig,
        kind="agent config",
    )
    worker_runtime_path = root / ".tend" / "worker-cfg.yaml"
    reviewer_runtime_path = root / ".tend" / "reviewer-cfg.yaml"
    worker_runtime = read_config_model(
        worker_runtime_path,
        RuntimeConfigOverrides,
        kind="runtime config",
    )
    assert worker_agent.model.provider == "anthropic"
    assert worker_agent.model.model_name == "claude-sonnet-4-5"
    assert worker_agent.model.settings.max_output_tokens == 32_768
    assert worker_agent.output is not None
    assert worker_agent.output.tool_name == "final_result"
    # The async worker now reuses the shared worker_contribution contract.
    assert worker_agent.output.schema_name.value == "worker_contribution"
    # On-disk shape: a file-backed system prompt pointer, not an inlined prompt or
    # bundled-registry pointer. Editing prompts/worker-system.md affects the next
    # agent launch.
    worker_agent_text = worker_agent_path.read_text(encoding="utf-8")
    assert "path: ../prompts/worker-system.md" in worker_agent_text
    assert "registry:" not in worker_agent_text
    worker_system = load_prompt(builtin_prompts_root() / "worker" / "minimal", "system")
    assert (root / "prompts" / "worker-system.md").read_text(
        encoding="utf-8"
    ) == f"{worker_system}\n"
    assert worker_agent.system_prompt == worker_system
    reviewer_agent = read_config_model(
        reviewer_agent_path,
        AgentConfig,
        kind="agent config",
    )
    assert reviewer_agent.output is not None
    # The async reviewer now reuses the shared review_verdict contract.
    assert reviewer_agent.output.schema_name.value == "review_verdict"
    # On-disk shape: a file-backed system prompt pointer, not an inlined prompt or
    # bundled-registry pointer. Editing prompts/reviewer-system.md affects the next
    # agent launch.
    reviewer_agent_text = reviewer_agent_path.read_text(encoding="utf-8")
    assert "path: ../prompts/reviewer-system.md" in reviewer_agent_text
    assert "registry:" not in reviewer_agent_text
    reviewer_system = load_prompt(builtin_prompts_root() / "reviewer" / "minimal", "system")
    assert (root / "prompts" / "reviewer-system.md").read_text(
        encoding="utf-8"
    ) == f"{reviewer_system}\n"
    assert reviewer_agent.system_prompt == reviewer_system
    assert "structured review verdict" in reviewer_agent.system_prompt
    assert worker_agent.model.settings.reasoning is not None
    assert worker_agent.model.settings.reasoning.effort is not None
    assert worker_agent.model.settings.reasoning.effort.value == "low"
    assert worker_runtime.compaction is not None
    assert worker_runtime.compaction.reserve_tokens == 16_384
    assert worker_runtime.compaction.keep_recent_tokens == 20_000
    assert worker_runtime.model is not None
    assert worker_runtime.model.timeout_seconds == 600.0
    # Unbounded RuntimeLimitsConfig values must be omitted from the rendered
    # max_iterations, max_model_requests, and max_tool_calls fields so the
    # YAML so the shared tend-agent default (None = unbounded) is inherited.
    # Same expectation on the reviewer-cfg.yaml.
    for runtime_path in (worker_runtime_path, reviewer_runtime_path):
        runtime_text = runtime_path.read_text(encoding="utf-8")
        assert "max_iterations" not in runtime_text
        assert "max_model_requests" not in runtime_text
        assert "max_tool_calls" not in runtime_text
    assert worker_runtime.limits is not None
    assert worker_runtime.limits.max_iterations is None
    assert worker_runtime.limits.max_model_requests is None
    assert worker_runtime.limits.max_tool_calls is None
    reviewer_runtime = read_config_model(
        reviewer_runtime_path,
        RuntimeConfigOverrides,
        kind="runtime config",
    )
    assert reviewer_runtime.limits is not None
    assert reviewer_runtime.limits.max_iterations is None
    assert reviewer_runtime.limits.max_model_requests is None
    assert reviewer_runtime.limits.max_tool_calls is None
    # The Tend worker per-invocation prompt is the resolved minimal task.md: no {...} leakage.
    worker_prompt_text = (root / "prompts" / "worker.md").read_text(encoding="utf-8")
    assert "final_result" in worker_prompt_text
    assert not re.search(r"\{[a-z_]+\}", worker_prompt_text)
    assert "tasks/$TEND_TASK_ID.yaml" in worker_prompt_text
    # The Tend reviewer per-invocation prompt is the resolved minimal task.md: no {...} leakage.
    reviewer_prompt_text = (root / "prompts" / "reviewer.md").read_text(encoding="utf-8")
    assert "request_changes" in reviewer_prompt_text
    assert "final_result" in reviewer_prompt_text
    assert not re.search(r"\{[a-z_]+\}", reviewer_prompt_text)
    assert "$TEND_WORKTREE_PATH" in reviewer_prompt_text


async def test_cli_init_pi_agent_writes_commands_prompts_and_scripts(tmp_path: Path) -> None:
    root = tmp_path / "orch"
    entrypoint = tmp_path / "entrypoint"
    entrypoint.mkdir()

    exit_code = await run_cli(
        ["init", "--root", str(root), "--entrypoint", str(entrypoint), "--agent", "pi"],
        stdout=StringIO(),
    )

    assert exit_code == int(AsyncOrchestratorCliExitCode.SUCCESS)
    project_config = read_config_model(
        root / "config.yaml",
        AsyncOrchestratorProjectConfig,
        kind="async orchestrator config",
    )
    worker_script = root / "bin" / "worker-agent.sh"
    reviewer_script = root / "bin" / "reviewer-agent.sh"
    assert project_config.entrypoint == entrypoint.resolve()
    assert project_config.worker_agent_command is not None
    assert project_config.worker_agent_command.argv == (str(worker_script),)
    assert project_config.worker_agent_command.resume_argv == ("--resume",)
    assert project_config.reviewer_agent_command is not None
    assert project_config.reviewer_agent_command.argv == (str(reviewer_script),)
    assert project_config.reviewer_agent_command.resume_argv == ("--resume",)
    assert worker_script.stat().st_mode & 0o111
    assert reviewer_script.stat().st_mode & 0o111
    # The pi worker prompt is the minimal task prompt; the system prompt is a separate
    # editable file passed through --append-system-prompt by the generated script.
    worker_script_text = worker_script.read_text(encoding="utf-8")
    assert "--append-system-prompt" in worker_script_text
    worker_system = load_prompt(builtin_prompts_root() / "worker" / "minimal", "system")
    assert (root / "prompts" / "worker-system.md").read_text(
        encoding="utf-8"
    ) == f"{worker_system}\n"
    worker_prompt_text = (root / "prompts" / "worker.md").read_text(encoding="utf-8")
    assert "final_result" in worker_prompt_text
    assert not re.search(r"\{[a-z_]+\}", worker_prompt_text)
    assert "tasks/$TEND_TASK_ID.yaml" in worker_prompt_text
    # The pi reviewer prompt is the minimal task prompt; the system prompt is a separate
    # editable file passed through --append-system-prompt by the generated script.
    reviewer_script_text = reviewer_script.read_text(encoding="utf-8")
    assert "--append-system-prompt" in reviewer_script_text
    reviewer_system = load_prompt(builtin_prompts_root() / "reviewer" / "minimal", "system")
    assert (root / "prompts" / "reviewer-system.md").read_text(
        encoding="utf-8"
    ) == f"{reviewer_system}\n"
    reviewer_prompt_text = (root / "prompts" / "reviewer.md").read_text(encoding="utf-8")
    assert "request_changes" in reviewer_prompt_text
    assert "final_result" in reviewer_prompt_text
    assert not re.search(r"\{[a-z_]+\}", reviewer_prompt_text)
    assert "$TEND_WORKTREE_PATH" in reviewer_prompt_text


async def test_cli_init_reviewer_prompt_version_is_configurable(tmp_path: Path) -> None:
    root = tmp_path / "orch"
    entrypoint = tmp_path / "entrypoint"
    entrypoint.mkdir()

    exit_code = await run_cli(
        [
            "init",
            "--root",
            str(root),
            "--entrypoint",
            str(entrypoint),
            "--agent",
            "tend",
            "--reviewer-prompt-version",
            "minimal",
        ],
        stdout=StringIO(),
    )

    assert exit_code == int(AsyncOrchestratorCliExitCode.SUCCESS)
    reviewer_agent_path = root / ".tend" / "reviewer-agent.yaml"
    reviewer_agent = read_config_model(
        reviewer_agent_path,
        AgentConfig,
        kind="agent config",
    )
    # The version flag selects the system prompt copied to the editable prompt file.
    assert "path: ../prompts/reviewer-system.md" in reviewer_agent_path.read_text(
        encoding="utf-8"
    )
    reviewer_system = load_prompt(builtin_prompts_root() / "reviewer" / "minimal", "system")
    assert (root / "prompts" / "reviewer-system.md").read_text(
        encoding="utf-8"
    ) == f"{reviewer_system}\n"
    assert reviewer_agent.system_prompt.startswith(reviewer_system.split("\n", 1)[0])
    assert reviewer_agent.output is not None
    assert reviewer_agent.output.schema_name.value == "review_verdict"


async def test_cli_init_worker_prompt_version_is_configurable(tmp_path: Path) -> None:
    root = tmp_path / "orch"
    entrypoint = tmp_path / "entrypoint"
    entrypoint.mkdir()

    exit_code = await run_cli(
        [
            "init",
            "--root",
            str(root),
            "--entrypoint",
            str(entrypoint),
            "--agent",
            "tend",
            "--worker-prompt-version",
            "minimal",
        ],
        stdout=StringIO(),
    )

    assert exit_code == int(AsyncOrchestratorCliExitCode.SUCCESS)
    worker_agent_path = root / ".tend" / "worker-agent.yaml"
    worker_agent = read_config_model(
        worker_agent_path,
        AgentConfig,
        kind="agent config",
    )
    # The version flag selects the system prompt copied to the editable prompt file.
    assert "path: ../prompts/worker-system.md" in worker_agent_path.read_text(
        encoding="utf-8"
    )
    worker_system = load_prompt(builtin_prompts_root() / "worker" / "minimal", "system")
    assert (root / "prompts" / "worker-system.md").read_text(
        encoding="utf-8"
    ) == f"{worker_system}\n"
    assert worker_agent.system_prompt.startswith(worker_system.split("\n", 1)[0])
    assert worker_agent.output is not None
    assert worker_agent.output.schema_name.value == "worker_contribution"


async def test_cli_init_refuses_non_empty_uninitialized_root(tmp_path: Path) -> None:
    stderr = StringIO()
    root = tmp_path / "not-empty"
    root.mkdir()
    (root / "keep.txt").write_text("keep\n", encoding="utf-8")

    exit_code = await run_cli(["init", "--root", str(root)], stderr=stderr)

    assert exit_code == int(AsyncOrchestratorCliExitCode.CONFIGURATION_OR_USAGE)
    assert not (root / ".tend-root").exists()
    assert "root is not empty" in stderr.getvalue()


async def test_cli_clean_removes_initialized_async_orchestration_root(tmp_path: Path) -> None:
    stdout = StringIO()
    root = tmp_path / "orch"
    await run_cli(["init", "--root", str(root)], stdout=StringIO())
    (root / "sessions" / "old-session").mkdir()

    exit_code = await run_cli(["clean", "--root", str(root)], stdout=stdout)

    assert exit_code == int(AsyncOrchestratorCliExitCode.SUCCESS)
    assert not root.exists()
    assert "removed async orchestration root" in stdout.getvalue()


async def test_cli_clean_refuses_locked_root(tmp_path: Path) -> None:
    stdout = StringIO()
    stderr = StringIO()
    root = tmp_path / "orch"
    await run_cli(["init", "--root", str(root)], stdout=StringIO())
    (root / "sessions" / "old-session").mkdir()

    with AsyncOrchestratorRootLock.acquire(root, owner="test", sync_writes=False):
        exit_code = await run_cli(["clean", "--root", str(root)], stdout=stdout, stderr=stderr)

    assert exit_code == int(AsyncOrchestratorCliExitCode.CONFIGURATION_OR_USAGE)
    assert root.exists()
    assert (root / "sessions" / "old-session").is_dir()
    assert "error[root_lock_error]" in stderr.getvalue()
    assert "already locked" in stderr.getvalue()
    assert stdout.getvalue() == ""


async def test_cli_clean_refuses_uninitialized_root(tmp_path: Path) -> None:
    stderr = StringIO()
    root = tmp_path / "not-managed"
    root.mkdir()

    exit_code = await run_cli(["clean", "--root", str(root)], stderr=stderr)

    assert exit_code == int(AsyncOrchestratorCliExitCode.CONFIGURATION_OR_USAGE)
    assert root.exists()
    assert "refusing to clean uninitialized async orchestration root" in stderr.getvalue()


async def test_cli_run_prints_run_summary(tmp_path: Path) -> None:
    root = tmp_path / "orch"
    entrypoint = tmp_path / "repo"
    entrypoint.mkdir()
    stdout = StringIO()

    class SummaryOrchestrator:
        def __init__(self, config: AsyncOrchestratorConfig) -> None:
            self.config = config

        async def run(self) -> AsyncOrchestratorRunResult:
            return AsyncOrchestratorRunResult(
                root=self.config.root,
                entrypoint=self.config.entrypoint,
                usage=Usage(
                    tokens=TokenUsage(input_tokens=100, output_tokens=50),
                    cost=Cost(amount=Decimal("1.2500"), currency="USD"),
                ),
                summary=AsyncOrchestratorRunSummary(
                    tasks_total=2,
                    tasks_by_status={"open": 1, "complete": 1},
                    worktrees_total=3,
                    worktrees_by_state={"closed": 2, "pending": 1},
                ),
            )

    await run_cli(
        ["init", "--root", str(root), "--entrypoint", str(entrypoint)],
        stdout=StringIO(),
    )

    exit_code = await run_cli(
        ["run", "--root", str(root)],
        stdout=stdout,
        orchestrator_factory=SummaryOrchestrator,
    )

    assert exit_code == int(AsyncOrchestratorCliExitCode.SUCCESS)
    output = stdout.getvalue()
    assert "async orchestrator run complete" in output
    assert "tasks: total=2" in output
    assert "complete=1" in output
    assert "worktrees: total=3" in output
    assert "closed=2" in output
    assert "cost=1.2500 USD" in output


async def test_cli_run_detach_spawns_child_without_running_in_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "orch"
    stdout = StringIO()
    seen_configs: list[AsyncOrchestratorConfig] = []
    seen_argv: list[str] = []
    seen_log_files: list[Path] = []
    seen_pid_files: list[Path] = []

    def forbidden_factory(config: AsyncOrchestratorConfig) -> cli_module.AsyncOrchestratorRunner:
        seen_configs.append(config)
        raise AssertionError("orchestrator factory should not be invoked for --detach")

    def fake_spawn_detached(
        argv: Sequence[str],
        *,
        log_file: Path,
        pid_file: Path,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> int:
        assert cwd is None
        assert env is None
        seen_argv[:] = list(argv)
        seen_log_files.append(log_file)
        seen_pid_files.append(pid_file)
        pid_file.parent.mkdir(parents=True, exist_ok=True)
        pid_file.write_text("4242\n", encoding="utf-8")
        return 4242

    monkeypatch.setattr(cli_module, "spawn_detached", fake_spawn_detached)

    exit_code = await run_cli(
        ["run", "--root", str(root), "--detach", "--max-cost", "5"],
        stdout=stdout,
        orchestrator_factory=forbidden_factory,
    )

    assert exit_code == int(AsyncOrchestratorCliExitCode.SUCCESS)
    assert seen_configs == []
    assert seen_argv[0].endswith("tend")
    assert seen_argv[1:] == ["run", "--root", str(root), "--max-cost", "5"]
    assert "--detach" not in seen_argv
    assert seen_log_files == [root.resolve() / "run.log"]
    assert seen_pid_files == [root.resolve() / "run.pid"]
    assert (root / "run.pid").read_text(encoding="utf-8") == "4242\n"
    output = stdout.getvalue()
    assert "pid: 4242" in output
    assert str(root.resolve() / "run.log") in output
    assert str(root.resolve() / "run.pid") in output


async def test_cli_run_detach_respects_log_and_pid_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "orch"
    log_file = tmp_path / "custom" / "orchestrator.log"
    pid_file = tmp_path / "custom" / "orchestrator.pid"
    seen_log_files: list[Path] = []
    seen_pid_files: list[Path] = []

    def fake_spawn_detached(
        argv: Sequence[str],
        *,
        log_file: Path,
        pid_file: Path,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> int:
        del argv, cwd, env
        seen_log_files.append(log_file)
        seen_pid_files.append(pid_file)
        pid_file.parent.mkdir(parents=True, exist_ok=True)
        pid_file.write_text("31337\n", encoding="utf-8")
        return 31337

    monkeypatch.setattr(cli_module, "spawn_detached", fake_spawn_detached)

    exit_code = await run_cli(
        [
            "run",
            "--root",
            str(root),
            "--detach",
            "--log-file",
            str(log_file),
            "--pid-file",
            str(pid_file),
        ],
        stdout=StringIO(),
    )

    assert exit_code == int(AsyncOrchestratorCliExitCode.SUCCESS)
    assert seen_log_files == [log_file]
    assert seen_pid_files == [pid_file]
    assert pid_file.read_text(encoding="utf-8") == "31337\n"


async def test_cli_run_detach_dry_run_does_not_spawn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "orch"
    entrypoint = tmp_path / "repo"
    entrypoint.mkdir()
    stdout = StringIO()
    spawned: list[list[str]] = []

    def fake_spawn_detached(
        argv: Sequence[str],
        *,
        log_file: Path,
        pid_file: Path,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> int:
        del log_file, pid_file, cwd, env
        spawned.append(list(argv))
        return 1

    monkeypatch.setattr(cli_module, "spawn_detached", fake_spawn_detached)
    await run_cli(
        ["init", "--root", str(root), "--entrypoint", str(entrypoint)],
        stdout=StringIO(),
    )

    exit_code = await run_cli(
        ["run", "--root", str(root), "--detach", "--dry-run"],
        stdout=stdout,
    )

    assert exit_code == int(AsyncOrchestratorCliExitCode.SUCCESS)
    assert spawned == []
    assert "async orchestrator run dry-run" in stdout.getvalue()


@pytest.mark.parametrize(
    "abbreviated_args",
    (
        ("--det",),
        ("--deta",),
        ("--log-f", "run.log"),
    ),
)
async def test_cli_run_rejects_abbreviated_detach_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    abbreviated_args: tuple[str, ...],
) -> None:
    root = tmp_path / "orch"
    stderr = StringIO()
    spawned: list[list[str]] = []

    def fake_spawn_detached(
        argv: Sequence[str],
        *,
        log_file: Path,
        pid_file: Path,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> int:
        del log_file, pid_file, cwd, env
        spawned.append(list(argv))
        return 1

    monkeypatch.setattr(cli_module, "spawn_detached", fake_spawn_detached)

    exit_code = await run_cli(
        ["run", "--root", str(root), *abbreviated_args],
        stderr=stderr,
    )

    assert exit_code == int(AsyncOrchestratorCliExitCode.CONFIGURATION_OR_USAGE)
    assert spawned == []
    assert "error[cli_usage_error]" in stderr.getvalue()
    assert "unrecognized arguments" in stderr.getvalue()


def test_detached_child_argv_strips_detach_only_flags() -> None:
    detached_child_argv = cast(
        Callable[..., list[str]],
        vars(cli_module)["_detached_child_argv"],
    )
    child_argv = detached_child_argv(
        [
            "--root",
            "orch",
            "--detach",
            "--max-cost",
            "5",
            "--log-file",
            "run.log",
            "--fresh",
            "--pid-file=run.pid",
        ],
        entry="tend",
    )

    assert child_argv == [
        "tend",
        "run",
        "--root",
        "orch",
        "--max-cost",
        "5",
        "--fresh",
    ]


async def test_cli_run_dry_run_validates_and_reports_without_running(tmp_path: Path) -> None:
    root = tmp_path / "orch"
    entrypoint = tmp_path / "repo"
    entrypoint.mkdir()
    stdout = StringIO()
    ran: list[AsyncOrchestratorConfig] = []

    class RecordingOrchestrator:
        def __init__(self, config: AsyncOrchestratorConfig) -> None:
            self.config = config

        async def run(self) -> AsyncOrchestratorRunResult:
            ran.append(self.config)
            return AsyncOrchestratorRunResult(
                root=self.config.root, entrypoint=self.config.entrypoint
            )

    await run_cli(
        ["init", "--root", str(root), "--entrypoint", str(entrypoint)],
        stdout=StringIO(),
    )

    exit_code = await run_cli(
        [
            "run",
            "--root",
            str(root),
            "--dry-run",
            "--worker-agent-command",
            "tend-agent --prompt worker",
            "--worktree-setup-command",
            "cp --archive {entrypoint}/.lake {worktree}/",
        ],
        stdout=stdout,
        orchestrator_factory=RecordingOrchestrator,
    )

    assert exit_code == int(AsyncOrchestratorCliExitCode.SUCCESS)
    # The orchestrator was never constructed/run on a dry run.
    assert ran == []
    output = stdout.getvalue()
    assert "async orchestrator run dry-run" in output
    assert "worker agent command: tend-agent --prompt worker" in output
    assert "worktree setup command: cp --archive {entrypoint}/.lake {worktree}/" in output
    assert "post-merge build gate: (none)" in output
    assert "no worktrees or agents were launched" in output


async def test_cli_validate_config_reports_resolved_config(tmp_path: Path) -> None:
    root = tmp_path / "orch"
    entrypoint = tmp_path / "repo"
    entrypoint.mkdir()
    stdout = StringIO()

    await run_cli(
        ["init", "--root", str(root), "--entrypoint", str(entrypoint)],
        stdout=StringIO(),
    )

    exit_code = await run_cli(["validate-config", "--root", str(root)], stdout=stdout)

    assert exit_code == int(AsyncOrchestratorCliExitCode.SUCCESS)
    output = stdout.getvalue()
    assert "config is valid" in output
    assert f"entrypoint: {entrypoint.resolve()}" in output
    assert "worktree setup command: (unset)" in output
    assert "post-merge build gate: (none)" in output


async def test_cli_validate_config_reports_invalid_config(tmp_path: Path) -> None:
    root = tmp_path / "orch"
    entrypoint = tmp_path / "repo"
    entrypoint.mkdir()
    stderr = StringIO()

    await run_cli(
        ["init", "--root", str(root), "--entrypoint", str(entrypoint)],
        stdout=StringIO(),
    )
    # Corrupt the config so validation fails.
    (root / "config.yaml").write_text("entrypoint: ''\n", encoding="utf-8")

    exit_code = await run_cli(["validate-config", "--root", str(root)], stderr=stderr)

    assert exit_code == int(AsyncOrchestratorCliExitCode.CONFIGURATION_OR_USAGE)
    assert "entrypoint must not be blank" in stderr.getvalue()


async def test_cli_init_writes_worker_revision_prompt(tmp_path: Path) -> None:
    """``tend init --agent tend`` materialises a worker-revision template.

    The on-disk ``prompts/worker-revision.md`` is the source the per-revision
    agent-runner step splices ``{feedback_message}`` into; the worker shim
    consults the substituted result on every ``--resume``. The init step
    leaves ``{feedback_message}`` literally intact so the runner can
    interpolate the latest non-worker discussion message (reviewer
    ``request_changes`` or any orchestrator-injected feedback) at spawn time.
    """
    root = tmp_path / "orch"
    entrypoint = tmp_path / "entrypoint"
    entrypoint.mkdir()

    exit_code = await run_cli(
        ["init", "--root", str(root), "--entrypoint", str(entrypoint), "--agent", "tend"],
        stdout=StringIO(),
    )

    assert exit_code == int(AsyncOrchestratorCliExitCode.SUCCESS)
    revision_template = root / "prompts" / "worker-revision.md"
    assert revision_template.is_file(), (
        "tend init --agent tend must write prompts/worker-revision.md"
    )
    content = revision_template.read_text(encoding="utf-8")
    # The non-feedback placeholders must be resolved to async equivalents.
    assert "{contribution_id}" not in content
    assert "{worktree_path}" not in content
    # ``{feedback_message}`` survives for the agent-runner to substitute later.
    assert "{feedback_message}" in content
    # The shim's worker script references the revision template.
    worker_shim = (root / "bin" / "worker-agent.sh").read_text(encoding="utf-8")
    assert "worker-revision.md" in worker_shim
    assert "revision-prompt.md" in worker_shim  # session-dir target inside the shim
    # The revision-prompt selection block must sit AFTER the resume-detection
    # block (so it's gated on ``${#RESUME_ARGS[@]} -gt 0`` and cannot fire on
    # a non-resume invocation that finds a stale prompt in the session dir).
    resume_block_index = worker_shim.index('if [[ "${1:-}" == "--resume"')
    revision_block_index = worker_shim.index("SESSION_REVISION_PROMPT_FILE")
    assert revision_block_index > resume_block_index, (
        "revision-prompt selection must follow resume detection so a non-resume "
        "invocation cannot pick up a stale revision-prompt.md from the session dir"
    )
    assert "${#RESUME_ARGS[@]}" in worker_shim, (
        "revision-prompt selection must be gated on RESUME_ARGS being non-empty"
    )


async def test_cli_init_reviewer_shim_does_not_select_revision_prompt(
    tmp_path: Path,
) -> None:
    """The reviewer shim is single-prompt; only the worker shim revs on resume."""
    root = tmp_path / "orch"
    entrypoint = tmp_path / "entrypoint"
    entrypoint.mkdir()

    exit_code = await run_cli(
        ["init", "--root", str(root), "--entrypoint", str(entrypoint), "--agent", "tend"],
        stdout=StringIO(),
    )
    assert exit_code == int(AsyncOrchestratorCliExitCode.SUCCESS)

    reviewer_shim = (root / "bin" / "reviewer-agent.sh").read_text(encoding="utf-8")
    assert "worker-revision.md" not in reviewer_shim
    assert "revision-prompt.md" not in reviewer_shim


async def test_init_defaults_merge_validation_worktree_on(tmp_path: Path) -> None:
    root = tmp_path / "orch"
    entrypoint = tmp_path / "repo"
    entrypoint.mkdir()

    await run_cli(
        ["init", "--root", str(root), "--entrypoint", str(entrypoint)],
        stdout=StringIO(),
    )

    config = read_config_model(
        root / "config.yaml", AsyncOrchestratorProjectConfig, kind="async orchestrator config"
    )
    assert config.merge_validation_worktree is True


async def test_init_max_merge_batch_size_writes_config(tmp_path: Path) -> None:
    root = tmp_path / "orch"
    entrypoint = tmp_path / "repo"
    entrypoint.mkdir()

    exit_code = await run_cli(
        [
            "init",
            "--root",
            str(root),
            "--entrypoint",
            str(entrypoint),
            "--max-merge-batch-size",
            "8",
        ],
        stdout=StringIO(),
    )

    assert exit_code == int(AsyncOrchestratorCliExitCode.SUCCESS)
    config = read_config_model(
        root / "config.yaml", AsyncOrchestratorProjectConfig, kind="async orchestrator config"
    )
    assert config.max_merge_batch_size == 8


async def test_init_skip_build_validation_for_task_only_merges_writes_config(
    tmp_path: Path,
) -> None:
    root = tmp_path / "orch"
    entrypoint = tmp_path / "repo"
    entrypoint.mkdir()

    exit_code = await run_cli(
        [
            "init",
            "--root",
            str(root),
            "--entrypoint",
            str(entrypoint),
            "--skip-build-validation-for-task-only-merges",
        ],
        stdout=StringIO(),
    )

    assert exit_code == int(AsyncOrchestratorCliExitCode.SUCCESS)
    config = read_config_model(
        root / "config.yaml", AsyncOrchestratorProjectConfig, kind="async orchestrator config"
    )
    assert config.skip_build_validation_for_task_only_merges is True


async def test_init_defaults_task_only_build_skip_off(tmp_path: Path) -> None:
    root = tmp_path / "orch"
    entrypoint = tmp_path / "repo"
    entrypoint.mkdir()

    await run_cli(
        ["init", "--root", str(root), "--entrypoint", str(entrypoint)],
        stdout=StringIO(),
    )

    config = read_config_model(
        root / "config.yaml", AsyncOrchestratorProjectConfig, kind="async orchestrator config"
    )
    assert config.skip_build_validation_for_task_only_merges is False


async def test_init_rejects_non_positive_max_merge_batch_size(tmp_path: Path) -> None:
    root = tmp_path / "orch"
    stderr = StringIO()

    exit_code = await run_cli(
        ["init", "--root", str(root), "--max-merge-batch-size", "0"],
        stderr=stderr,
    )

    assert exit_code == int(AsyncOrchestratorCliExitCode.CONFIGURATION_OR_USAGE)
    assert "value must be greater than 0" in stderr.getvalue()


async def test_init_no_merge_validation_worktree_opts_out(tmp_path: Path) -> None:
    root = tmp_path / "orch"
    entrypoint = tmp_path / "repo"
    entrypoint.mkdir()

    await run_cli(
        [
            "init",
            "--root",
            str(root),
            "--entrypoint",
            str(entrypoint),
            "--no-merge-validation-worktree",
        ],
        stdout=StringIO(),
    )

    config = read_config_model(
        root / "config.yaml", AsyncOrchestratorProjectConfig, kind="async orchestrator config"
    )
    assert config.merge_validation_worktree is False
