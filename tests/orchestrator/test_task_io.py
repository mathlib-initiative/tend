from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from tend._common.config_files import ConfigFileError
from tend.orchestrator.task_io import (
    load_entrypoint_task_manager,
    load_entrypoint_tasks,
    load_entrypoint_tasks_strict,
    load_task,
    load_tasks,
    load_tasks_strict,
    task_directory,
    write_task,
)
from tend.orchestrator.tasks import Task, TaskPriority, TaskStatus


def test_task_round_trips_as_yaml(tmp_path: Path) -> None:
    task = Task(
        id="task-1",
        title="Seed task",
        summary="Seed task",
        description="Create the seed.",
        status=TaskStatus.COMPLETE,
    )
    path = tmp_path / "task.yaml"

    write_task(path, task)

    text = path.read_text(encoding="utf-8")
    assert load_task(path) == task
    assert "summary: Seed task" in text
    assert "priority: default" in text
    # The dependency list is written under its YAML key 'depends_on'.
    assert "depends_on:" in text
    assert "dependencies:" not in text
    # An absent hand-off is not serialized (exclude_none keeps task files clean).
    assert "notes:" not in text


def test_task_priority_round_trip_when_present(tmp_path: Path) -> None:
    task = Task(
        id="task-1",
        title="Steering task",
        summary="Steering task",
        description="Guide the queue.",
        priority=TaskPriority.MAX,
    )
    path = tmp_path / "task.yaml"

    write_task(path, task)

    text = path.read_text(encoding="utf-8")
    assert "priority: max" in text
    assert load_task(path) == task


def test_task_notes_round_trip_when_present(tmp_path: Path) -> None:
    task = Task(
        id="task-1",
        title="Seed task",
        summary="Seed task",
        description="Create the seed.",
        notes="Stated the lemma; proof deferred to task-2.",
    )
    path = tmp_path / "task.yaml"

    write_task(path, task)

    text = path.read_text(encoding="utf-8")
    assert "notes:" in text
    assert load_task(path) == task
    assert load_task(path).notes == "Stated the lemma; proof deferred to task-2."


def test_load_entrypoint_tasks_from_tasks_directory(tmp_path: Path) -> None:
    entrypoint = tmp_path / "entrypoint"
    tasks_dir = task_directory(entrypoint)
    first = Task(
        id="task-1",
        title="First", summary="First",
        description="First task.",
        status=TaskStatus.COMPLETE,
    )
    second = Task(
        id="task-2",
        title="Second", summary="Second",
        description="Second task.",
        depends_on=[first.id],
    )
    write_task(tasks_dir / "002-second.yaml", second)
    write_task(tasks_dir / "001-first.yaml", first)

    tasks = load_entrypoint_tasks(entrypoint)
    manager = load_entrypoint_task_manager(entrypoint)

    assert tasks == (first, second)
    assert manager.tasks == [first, second]
    assert manager.ready_tasks() == (second,)


def test_load_tasks_skips_malformed_task_files(tmp_path: Path) -> None:
    tasks_dir = tmp_path / "tasks"
    good = Task(id="task-1", title="Good", summary="Good", description="A valid task.")
    write_task(tasks_dir / "001-good.yaml", good)
    # A worker-written file that is not valid YAML / not a valid task.
    (tasks_dir / "002-bad.yaml").write_text("id: task-2\n  bad: : indentation\n", encoding="utf-8")
    # A structurally valid YAML missing required fields.
    (tasks_dir / "003-incomplete.yaml").write_text("id: task-3\n", encoding="utf-8")

    tasks = load_tasks(tasks_dir)

    # The bad files are skipped; the well-formed task is still loaded.
    assert tasks == (good,)


def test_load_tasks_strict_raises_on_malformed_yaml(tmp_path: Path) -> None:
    """The strict loader is the merge-time gate: a bad file must fail loudly."""

    tasks_dir = tmp_path / "tasks"
    good = Task(id="task-1", title="Good", summary="Good", description="A valid task.")
    write_task(tasks_dir / "001-good.yaml", good)
    (tasks_dir / "002-bad.yaml").write_text("id: task-2\n  bad: : indentation\n", encoding="utf-8")

    with pytest.raises(ConfigFileError):
        load_tasks_strict(tasks_dir)


def test_load_tasks_strict_raises_on_validation_error(tmp_path: Path) -> None:
    """A YAML-parsable file with missing required fields fails validation strictly."""

    tasks_dir = tmp_path / "tasks"
    good = Task(id="task-1", title="Good", summary="Good", description="A valid task.")
    write_task(tasks_dir / "001-good.yaml", good)
    (tasks_dir / "002-incomplete.yaml").write_text("id: task-2\n", encoding="utf-8")

    with pytest.raises(ValidationError):
        load_tasks_strict(tasks_dir)


def test_load_tasks_strict_returns_all_when_all_valid(tmp_path: Path) -> None:
    tasks_dir = tmp_path / "tasks"
    first = Task(id="task-1", title="First", summary="First", description="First.")
    second = Task(
        id="task-2", title="Second", summary="Second", description="Second.", depends_on=["task-1"]
    )
    write_task(tasks_dir / "001-first.yaml", first)
    write_task(tasks_dir / "002-second.yaml", second)

    assert load_tasks_strict(tasks_dir) == (first, second)


def test_load_entrypoint_tasks_strict_reads_tasks_subdirectory(tmp_path: Path) -> None:
    entrypoint = tmp_path / "entrypoint"
    task = Task(id="task-1", title="T", summary="T", description="T.")
    write_task(task_directory(entrypoint) / "001.yaml", task)

    assert load_entrypoint_tasks_strict(entrypoint) == (task,)


def test_load_entrypoint_task_manager_relaxes_a_malformed_graph(tmp_path: Path) -> None:
    entrypoint = tmp_path / "entrypoint"
    tasks_dir = task_directory(entrypoint)
    # A dependency cycle plus a dependency on an unknown task id.
    write_task(
        tasks_dir / "001.yaml",
        Task(
            id="task-1",
            title="One",
            summary="One",
            description="First.",
            depends_on=["task-2", "ghost"],
        ),
    )
    write_task(
        tasks_dir / "002.yaml",
        Task(id="task-2", title="Two", summary="Two", description="Second.", depends_on=["task-1"]),
    )

    # Strict construction would raise; the resilient loader relaxes the graph instead.
    manager = load_entrypoint_task_manager(entrypoint)

    assert manager.task_ids == ("task-1", "task-2")
    # The unknown-id edge is dropped and the cycle is broken (one back edge removed).
    deps = {task.id: task.dependencies for task in manager.tasks}
    assert "ghost" not in deps["task-1"]
    assert not (deps["task-1"] and deps["task-2"])  # cycle broken
