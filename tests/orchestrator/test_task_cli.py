from __future__ import annotations

from io import StringIO
from pathlib import Path

from tend.orchestrator.task_cli import TaskCliExitCode, run_task_cli
from tend.orchestrator.task_io import write_task
from tend.orchestrator.tasks import Task


def test_task_cli_verify_accepts_valid_task_set(tmp_path: Path) -> None:
    tasks_dir = tmp_path / "tasks"
    write_task(tasks_dir / "001.yaml", _task("task-1"))
    stdout = StringIO()
    stderr = StringIO()

    exit_code = run_task_cli(["verify", str(tasks_dir)], stdout=stdout, stderr=stderr)

    assert exit_code == int(TaskCliExitCode.SUCCESS)
    assert f"task set is valid: {tasks_dir.resolve()}" in stdout.getvalue()
    assert stderr.getvalue() == ""


def test_task_cli_verify_reports_malformed_task_file(tmp_path: Path) -> None:
    tasks_dir = tmp_path / "tasks"
    write_task(tasks_dir / "001-good.yaml", _task("task-1"))
    bad = tasks_dir / "002-bad.yaml"
    bad.write_text("id: task-broken\n  bad: : indentation\n", encoding="utf-8")
    stdout = StringIO()
    stderr = StringIO()

    exit_code = run_task_cli(["verify", str(tasks_dir)], stdout=stdout, stderr=stderr)

    assert exit_code == int(TaskCliExitCode.VALIDATION_FAILED)
    assert stdout.getvalue() == ""
    text = stderr.getvalue()
    assert "error[task_validation_error]: task file failed to parse" in text
    assert str(bad.resolve()) in text
    assert "offending paths:" in text


def test_task_cli_verify_reports_invalid_dependency_graph(tmp_path: Path) -> None:
    tasks_dir = tmp_path / "tasks"
    write_task(tasks_dir / "001-a.yaml", _task("task-a", depends_on=["task-b"]))
    write_task(tasks_dir / "002-b.yaml", _task("task-b", depends_on=["task-a"]))
    stderr = StringIO()

    exit_code = run_task_cli(["verify", str(tasks_dir)], stderr=stderr)

    assert exit_code == int(TaskCliExitCode.VALIDATION_FAILED)
    assert "error[task_validation_error]: task dependency graph is invalid" in stderr.getvalue()
    assert "task dependency cycle detected" in stderr.getvalue()


def test_task_cli_verify_rejects_missing_task_directory(tmp_path: Path) -> None:
    missing = tmp_path / "missing-tasks"
    stderr = StringIO()

    exit_code = run_task_cli(["verify", str(missing)], stderr=stderr)

    assert exit_code == int(TaskCliExitCode.CONFIGURATION_OR_USAGE)
    assert "error[filesystem_error]: task directory does not exist" in stderr.getvalue()


def _task(task_id: str, *, depends_on: list[str] | None = None) -> Task:
    return Task(
        id=task_id,
        title=task_id,
        summary=task_id,
        description=f"{task_id} description.",
        depends_on=[] if depends_on is None else depends_on,
    )
