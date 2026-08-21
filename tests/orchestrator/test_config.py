from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from tend._common.config_files import read_config_model
from tend.orchestrator.config import (
    AsyncOrchestratorAgentCommandConfig,
    AsyncOrchestratorBudgetConfig,
    AsyncOrchestratorConfig,
    AsyncOrchestratorProjectConfig,
    AsyncOrchestratorValidationCommandConfig,
    AsyncOrchestratorWorktreeSetupCommandConfig,
)


def test_config_accepts_agent_commands_and_concurrency() -> None:
    config = AsyncOrchestratorConfig.from_paths(
        root="./orch",
        entrypoint=Path("./repo"),
        worker_agent_command=["tend-agent", "--prompt", "worker"],
        reviewer_agent_command=AsyncOrchestratorAgentCommandConfig(
            argv=("tend-agent", "--prompt", "reviewer"),
            resume_argv=("--resume-session",),
        ),
        worktree_setup_command=["cp", "--archive", "{entrypoint}/.lake", "{worktree}/"],
        validation_commands=[
            ["uv", "run", "ruff", "check"],
            AsyncOrchestratorValidationCommandConfig(argv=("uv", "run", "pyright")),
        ],
        pre_merge_validation_commands=[
            ["uv", "run", "pytest", "-m", "not live"],
        ],
        merge_target_branch="integration",
        max_merge_batch_size=8,
        max_concurrent_worker_agents=3,
        max_concurrent_reviewer_agents=2,
    )

    assert config.root.as_posix() == "orch"
    assert config.entrypoint.as_posix() == "repo"
    assert config.worker_agent_command is not None
    assert config.worker_agent_command.argv == ("tend-agent", "--prompt", "worker")
    assert config.reviewer_agent_command is not None
    assert config.reviewer_agent_command.argv == ("tend-agent", "--prompt", "reviewer")
    assert config.reviewer_agent_command.resume_argv == ("--resume-session",)
    assert config.worktree_setup_command is not None
    assert config.worktree_setup_command.argv_for_paths(
        entrypoint=Path("/repo"),
        worktree=Path("/orch/worktrees/w1"),
    ) == ("cp", "--archive", "/repo/.lake", "/orch/worktrees/w1/")
    assert [command.argv for command in config.validation_commands] == [
        ("uv", "run", "ruff", "check"),
        ("uv", "run", "pyright"),
    ]
    assert [command.argv for command in config.pre_merge_validation_commands] == [
        ("uv", "run", "pytest", "-m", "not live"),
    ]
    assert config.merge_target_branch == "integration"
    assert config.max_merge_batch_size == 8
    assert config.max_concurrent_worker_agents == 3
    assert config.max_concurrent_reviewer_agents == 2


def test_project_config_builds_runtime_config_for_root() -> None:
    project_config = AsyncOrchestratorProjectConfig.model_validate(
        {
            "entrypoint": Path("./repo"),
            "worker_agent_command": ("tend-agent", "--prompt", "worker"),
            "worktree_setup_command": ("cp", "{entrypoint}/.lake", "{worktree}/"),
            "validation_commands": [
                {"argv": ["uv", "run", "ruff", "check"]},
                ["uv", "run", "pyright"],
            ],
            "pre_merge_validation_commands": [
                {"argv": ["uv", "run", "pytest", "-m", "not live"]},
            ],
            "merge_target_branch": "release",
            "max_merge_batch_size": 4,
            "max_concurrent_worker_agents": 2,
        }
    )

    config = project_config.to_runtime_config(root=Path("./orch"))

    assert config.root.as_posix() == "orch"
    assert config.entrypoint.as_posix() == "repo"
    assert config.worker_agent_command is not None
    assert config.worker_agent_command.argv == ("tend-agent", "--prompt", "worker")
    assert config.worktree_setup_command is not None
    assert config.worktree_setup_command.argv == ("cp", "{entrypoint}/.lake", "{worktree}/")
    assert [command.argv for command in config.validation_commands] == [
        ("uv", "run", "ruff", "check"),
        ("uv", "run", "pyright"),
    ]
    assert [command.argv for command in config.pre_merge_validation_commands] == [
        ("uv", "run", "pytest", "-m", "not live"),
    ]
    assert config.merge_target_branch == "release"
    assert config.max_merge_batch_size == 4
    assert config.max_concurrent_worker_agents == 2


def test_project_config_defaults_merge_target_branch() -> None:
    project_config = AsyncOrchestratorProjectConfig(entrypoint=Path("./repo"))

    assert project_config.merge_target_branch == "main"
    assert project_config.max_merge_batch_size is None
    assert project_config.pre_merge_validation_commands == ()
    assert project_config.skip_build_validation_for_task_only_merges is False
    runtime_config = project_config.to_runtime_config(root=Path("./orch"))
    assert runtime_config.merge_target_branch == "main"
    assert runtime_config.max_merge_batch_size is None
    assert runtime_config.skip_build_validation_for_task_only_merges is False


def test_project_config_propagates_task_only_build_skip_to_runtime_config() -> None:
    project_config = AsyncOrchestratorProjectConfig.model_validate(
        {
            "entrypoint": Path("./repo"),
            "skip_build_validation_for_task_only_merges": True,
        }
    )

    runtime_config = project_config.to_runtime_config(root=Path("./orch"))

    assert project_config.skip_build_validation_for_task_only_merges is True
    assert runtime_config.skip_build_validation_for_task_only_merges is True


def test_runtime_config_from_paths_accepts_task_only_build_skip() -> None:
    config = AsyncOrchestratorConfig.from_paths(
        root="./orch",
        entrypoint="./repo",
        skip_build_validation_for_task_only_merges=True,
    )

    assert config.skip_build_validation_for_task_only_merges is True


def test_agent_command_rejects_blank_arguments() -> None:
    with pytest.raises(ValidationError, match="agent command arguments"):
        AsyncOrchestratorAgentCommandConfig(argv=("tend-agent", " "))


def test_validation_command_rejects_blank_arguments() -> None:
    with pytest.raises(ValidationError, match="validation command arguments"):
        AsyncOrchestratorValidationCommandConfig(argv=("uv", " "))


def test_validation_command_accepts_optional_timeout() -> None:
    no_timeout = AsyncOrchestratorValidationCommandConfig(argv=("lake", "build"))
    with_timeout = AsyncOrchestratorValidationCommandConfig(
        argv=("lake", "build"), timeout_seconds=1800.0
    )

    assert no_timeout.timeout_seconds is None
    assert with_timeout.timeout_seconds == 1800.0


def test_validation_command_rejects_non_positive_timeout() -> None:
    with pytest.raises(ValidationError):
        AsyncOrchestratorValidationCommandConfig(argv=("lake", "build"), timeout_seconds=0.0)


def test_budget_config_defaults_to_no_ceiling() -> None:
    budget = AsyncOrchestratorBudgetConfig()

    assert budget.max_cost is None
    assert budget.currency == "USD"


def test_budget_config_coerces_string_and_int_max_cost() -> None:
    # Strings and ints are accepted (and coerced to Decimal) via model_validate so
    # money can be expressed exactly in YAML/JSON config without float rounding.
    from_string = AsyncOrchestratorBudgetConfig.model_validate({"max_cost": "50.00"})
    from_int = AsyncOrchestratorBudgetConfig.model_validate({"max_cost": 50})

    assert from_string.max_cost == Decimal("50.00")
    assert from_int.max_cost == Decimal(50)


def test_budget_config_rejects_non_positive_and_non_decimal_max_cost() -> None:
    with pytest.raises(ValidationError):
        AsyncOrchestratorBudgetConfig.model_validate({"max_cost": "0"})
    with pytest.raises(ValidationError):
        AsyncOrchestratorBudgetConfig.model_validate({"max_cost": "not-a-number"})


def test_project_config_propagates_budget_to_runtime_config() -> None:
    project_config = AsyncOrchestratorProjectConfig(
        entrypoint=Path("./repo"),
        budget=AsyncOrchestratorBudgetConfig(max_cost=Decimal("25"), currency="USD"),
    )

    runtime_config = project_config.to_runtime_config(root=Path("./orch"))

    assert runtime_config.budget.max_cost == Decimal("25")
    assert runtime_config.budget.currency == "USD"


def test_project_config_propagates_agent_oom_score_adj_to_runtime_config() -> None:
    default_project_config = AsyncOrchestratorProjectConfig(entrypoint=Path("./repo"))
    disabled_project_config = AsyncOrchestratorProjectConfig(
        entrypoint=Path("./repo"), agent_oom_score_adj=None
    )
    custom_project_config = AsyncOrchestratorProjectConfig(
        entrypoint=Path("./repo"), agent_oom_score_adj=123
    )

    assert (
        default_project_config.to_runtime_config(root=Path("./orch")).agent_oom_score_adj
        == 750
    )
    assert (
        disabled_project_config.to_runtime_config(root=Path("./orch")).agent_oom_score_adj
        is None
    )
    assert (
        custom_project_config.to_runtime_config(root=Path("./orch")).agent_oom_score_adj
        == 123
    )


def test_runtime_config_from_paths_accepts_agent_oom_score_adj() -> None:
    disabled_config = AsyncOrchestratorConfig.from_paths(
        root="./orch", entrypoint="./repo", agent_oom_score_adj=None
    )
    custom_config = AsyncOrchestratorConfig.from_paths(
        root="./orch", entrypoint="./repo", agent_oom_score_adj=321
    )

    assert disabled_config.agent_oom_score_adj is None
    assert custom_config.agent_oom_score_adj == 321


def test_agent_oom_score_adj_rejects_out_of_range_values() -> None:
    with pytest.raises(ValidationError):
        AsyncOrchestratorProjectConfig(entrypoint=Path("./repo"), agent_oom_score_adj=1001)
    with pytest.raises(ValidationError):
        AsyncOrchestratorConfig.from_paths(
            root="./orch", entrypoint="./repo", agent_oom_score_adj=-1001
        )


def test_project_config_reads_budget_from_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "entrypoint: ./repo\nbudget:\n  max_cost: '12.50'\n  currency: USD\n",
        encoding="utf-8",
    )

    project_config = read_config_model(
        config_path,
        AsyncOrchestratorProjectConfig,
        kind="async orchestrator config",
    )

    assert project_config.budget.max_cost == Decimal("12.50")


def test_worktree_setup_command_rejects_unknown_placeholders() -> None:
    with pytest.raises(ValidationError, match="placeholders"):
        AsyncOrchestratorWorktreeSetupCommandConfig(argv=("cp", "{unknown}/.lake"))


def test_project_config_yaml_file_rejects_blank_entrypoint(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text('entrypoint: ""\n', encoding="utf-8")

    with pytest.raises(ValidationError, match="entrypoint must not be blank"):
        read_config_model(
            config_path,
            AsyncOrchestratorProjectConfig,
            kind="async orchestrator config",
        )


def test_project_config_json_file_rejects_blank_entrypoint(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text('{"entrypoint": ""}\n', encoding="utf-8")

    with pytest.raises(ValidationError, match="entrypoint must not be blank"):
        read_config_model(
            config_path,
            AsyncOrchestratorProjectConfig,
            kind="async orchestrator config",
        )


def test_runtime_config_yaml_file_rejects_blank_root(tmp_path: Path) -> None:
    config_path = tmp_path / "runtime.yaml"
    config_path.write_text('root: ""\nentrypoint: ./repo\n', encoding="utf-8")

    with pytest.raises(ValidationError, match="root must not be blank"):
        read_config_model(
            config_path,
            AsyncOrchestratorConfig,
            kind="async orchestrator config",
        )


def test_runtime_config_json_file_rejects_blank_entrypoint(tmp_path: Path) -> None:
    config_path = tmp_path / "runtime.json"
    config_path.write_text('{"root": "./orch", "entrypoint": ""}\n', encoding="utf-8")

    with pytest.raises(ValidationError, match="entrypoint must not be blank"):
        read_config_model(
            config_path,
            AsyncOrchestratorConfig,
            kind="async orchestrator config",
        )


def test_config_rejects_blank_path_strings() -> None:
    with pytest.raises(ValidationError, match="root must not be blank"):
        AsyncOrchestratorConfig.from_paths(root=" ", entrypoint=Path("./repo"))

    with pytest.raises(ValidationError, match="entrypoint must not be blank"):
        AsyncOrchestratorConfig.from_paths(root=Path("./orch"), entrypoint="\t")


def test_config_rejects_nul_paths() -> None:
    with pytest.raises(ValidationError, match="root must not contain NUL"):
        AsyncOrchestratorConfig(root=Path("orch\x00bad"), entrypoint=Path("./repo"))

    with pytest.raises(ValidationError, match="entrypoint must not contain NUL"):
        AsyncOrchestratorConfig.from_paths(root=Path("./orch"), entrypoint="repo\x00bad")


def test_project_config_rejects_nul_entrypoint() -> None:
    with pytest.raises(ValidationError, match="entrypoint must not contain NUL"):
        AsyncOrchestratorProjectConfig(entrypoint=Path("repo\x00bad"))


def test_config_rejects_non_positive_concurrency() -> None:
    with pytest.raises(ValidationError):
        AsyncOrchestratorConfig(
            root=Path("./orch"),
            entrypoint=Path("./repo"),
            max_concurrent_worker_agents=0,
        )


def test_config_rejects_non_positive_max_merge_batch_size() -> None:
    with pytest.raises(ValidationError):
        AsyncOrchestratorConfig.from_paths(
            root="./orch",
            entrypoint="./repo",
            max_merge_batch_size=0,
        )
    with pytest.raises(ValidationError):
        AsyncOrchestratorProjectConfig(
            entrypoint=Path("./repo"),
            max_merge_batch_size=-1,
        )


def test_config_rejects_blank_merge_target_branch() -> None:
    with pytest.raises(ValidationError, match="merge target branch"):
        AsyncOrchestratorConfig(
            root=Path("./orch"),
            entrypoint=Path("./repo"),
            merge_target_branch=" ",
        )
