"""Disk I/O helpers for async orchestrator tasks."""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import ValidationError

from tend._common.config_files import ConfigFileError, dump_yaml_data, read_config_model
from tend.orchestrator.task_manager import TaskManager, build_resilient_task_manager
from tend.orchestrator.tasks import Task

DEFAULT_TASK_FILE_GLOB = "*.yaml"
TASKS_DIRECTORY_NAME = "tasks"

_LOGGER = logging.getLogger(__name__)


def task_directory(entrypoint: str | Path) -> Path:
    """Return the conventional task directory for an entrypoint repository."""

    return Path(entrypoint) / TASKS_DIRECTORY_NAME


def load_task(path: str | Path) -> Task:
    """Load one human-readable YAML task file from disk."""

    return read_config_model(path, Task, kind="async orchestrator task")


def dump_task_yaml(task: Task) -> str:
    """Serialize a task as human-readable YAML using the unified field names.

    The dependency list is emitted under its YAML key ``depends_on`` so the
    written file matches the unified task format read by both orchestrators.
    """

    data = task.model_dump(mode="json", exclude_none=True, by_alias=True)
    return dump_yaml_data(data)


def write_task(path: str | Path, task: Task) -> None:
    """Write one human-readable YAML task file to disk."""

    task_path = Path(path)
    task_path.parent.mkdir(parents=True, exist_ok=True)
    task_path.write_text(dump_task_yaml(task), encoding="utf-8")


def load_tasks(directory: str | Path, *, glob: str = DEFAULT_TASK_FILE_GLOB) -> tuple[Task, ...]:
    """Load all task YAML files matching ``glob`` from ``directory``.

    A task file that cannot be parsed (invalid YAML, missing/invalid fields) is
    skipped with a logged warning rather than aborting the whole load, so one bad
    file written by a worker cannot tear down the run. This mirrors the sync
    orchestrator's ``TaskRepositoryScanner.scan``. The directory itself failing to
    list still raises, since that is not a per-file recovery case.
    """

    task_dir = Path(directory)
    try:
        paths = tuple(sorted(path for path in task_dir.glob(glob) if path.is_file()))
    except OSError as exc:
        raise ConfigFileError(
            f"could not list async orchestrator task directory {task_dir}: {exc.strerror or exc}",
            path=task_dir,
            kind="async orchestrator task directory",
        ) from exc
    tasks: list[Task] = []
    for path in paths:
        try:
            tasks.append(load_task(path))
        except (ConfigFileError, ValidationError) as exc:
            _LOGGER.warning("skipping malformed async task file %s: %s", path, exc)
    return tuple(tasks)


def load_entrypoint_tasks(
    entrypoint: str | Path,
    *,
    glob: str = DEFAULT_TASK_FILE_GLOB,
) -> tuple[Task, ...]:
    """Load tasks from ``<entrypoint>/tasks/``."""

    return load_tasks(task_directory(entrypoint), glob=glob)


def load_tasks_strict(
    directory: str | Path,
    *,
    glob: str = DEFAULT_TASK_FILE_GLOB,
) -> tuple[Task, ...]:
    """Load all task YAML files in ``directory`` strictly, raising on first bad file.

    Unlike :func:`load_tasks`, a task file that cannot be parsed (invalid YAML or
    missing/invalid fields) propagates as :class:`ConfigFileError` /
    :class:`pydantic.ValidationError`. This is the merge-time gate counterpart to
    the lenient scheduler-time loader: bad worker output is kept out of the
    scanned tree by failing the merge, not papered over at scheduling time.
    """

    task_dir = Path(directory)
    try:
        paths = tuple(sorted(path for path in task_dir.glob(glob) if path.is_file()))
    except OSError as exc:
        raise ConfigFileError(
            f"could not list async orchestrator task directory {task_dir}: {exc.strerror or exc}",
            path=task_dir,
            kind="async orchestrator task directory",
        ) from exc
    return tuple(load_task(path) for path in paths)


def load_entrypoint_tasks_strict(
    entrypoint: str | Path,
    *,
    glob: str = DEFAULT_TASK_FILE_GLOB,
) -> tuple[Task, ...]:
    """Strictly load tasks from ``<entrypoint>/tasks/``; see :func:`load_tasks_strict`."""

    return load_tasks_strict(task_directory(entrypoint), glob=glob)


def load_entrypoint_task_manager(
    entrypoint: str | Path,
    *,
    glob: str = DEFAULT_TASK_FILE_GLOB,
) -> TaskManager:
    """Load ``<entrypoint>/tasks/`` into a resilient task manager.

    Malformed task files are skipped (see ``load_tasks``) and a malformed task
    *graph* (duplicate IDs, unknown dependency references, cycles, or a complete
    task depending on an open one) is relaxed rather than raised, so a single bad
    worker-written task cannot tear down the live polling loop. Mirrors the sync
    orchestrator's scan-and-relax behavior.
    """

    return build_resilient_task_manager(load_entrypoint_tasks(entrypoint, glob=glob))
