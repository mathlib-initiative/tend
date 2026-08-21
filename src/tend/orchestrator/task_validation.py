"""Shared validation helpers for orchestrator task directories."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from tend._common.config_files import ConfigFileError
from tend.orchestrator.task_io import DEFAULT_TASK_FILE_GLOB, load_task
from tend.orchestrator.task_manager import (
    TaskGraphValidationError,
    TaskManager,
    validate_task_graph,
)
from tend.orchestrator.tasks import Task

_MAX_RENDERED_FILE_FAILURES = 30


@dataclass(frozen=True, slots=True)
class TaskValidationFailure:
    """Captured task-directory validation failure details.

    ``offending_paths`` names the task file(s) implicated by the failure (the
    unparseable file, every file declaring a task named by a dependency-graph
    error, or — via ``resolve_missing_task_id`` — the pre-merge file whose
    deletion orphaned a dependency reference) so merge-gate callers can
    attribute the failure to the contribution that touched those files instead
    of bisecting.
    """

    summary: str
    detail: str
    offending_paths: tuple[str, ...] = ()


def validate_task_directory(
    directory: str | Path,
    *,
    glob: str = DEFAULT_TASK_FILE_GLOB,
    resolve_missing_task_id: Callable[[str], tuple[str, ...]] | None = None,
) -> TaskValidationFailure | None:
    """Validate a task directory using the strict post-merge orchestrator gate.

    The directory is scanned for task YAML files, each matching file is parsed as
    a :class:`tend.orchestrator.tasks.Task`, and the full set is validated as
    a :class:`tend.orchestrator.task_manager.TaskManager` dependency graph.
    Returns ``None`` when the task set is valid, or a structured failure with the
    same summary/detail style used by real orchestrated merge validation.

    ``resolve_missing_task_id`` maps an offending task id that no scanned file
    declares (an unknown-dependency reference) to blameable path(s). The merge
    gate passes a resolver that looks the id up in the *pre-merge* task tree, so
    a dependency orphaned by deleting its declaring file is attributed to the
    contribution that deleted that file.
    """

    task_dir = Path(directory)
    try:
        paths = tuple(sorted(p for p in task_dir.glob(glob) if p.is_file()))
    except OSError as exc:
        return TaskValidationFailure(
            summary=f"task directory could not be read: {_trim_text(str(exc), max_length=500)}",
            detail=_trim_text(str(exc), max_length=2000),
        )
    tasks: list[Task] = []
    paths_by_task_id: dict[str, list[Path]] = {}
    file_failure_count = 0
    rendered_file_failures: list[tuple[str, str]] = []
    file_failure_paths: list[str] = []
    for path in paths:
        # ``load_task`` wraps OSError into ConfigFileError via ``_read_config_text``;
        # invalid UTF-8 remains a UnicodeDecodeError. Treat both as per-file
        # malformed input alongside schema failures. Keep scanning after every
        # failure: attribution needs every independently malformed contribution
        # from the first assembled batch, while only a bounded sample is retained
        # for human-readable output.
        try:
            task = load_task(path)
        except (ConfigFileError, UnicodeDecodeError, ValidationError) as exc:
            file_failure_count += 1
            file_failure_paths.append(str(path))
            if len(rendered_file_failures) < _MAX_RENDERED_FILE_FAILURES:
                failure_phrase = (
                    "failed to parse"
                    if isinstance(exc, ConfigFileError | UnicodeDecodeError)
                    else "failed validation"
                )
                rendered_file_failures.append(
                    (
                        f"task file {failure_phrase} (file: {path}): "
                        f"{_trim_text(str(exc), max_length=500)}",
                        _trim_text(str(exc), max_length=2000),
                    )
                )
            continue
        tasks.append(task)
        paths_by_task_id.setdefault(task.id, []).append(path)

    if file_failure_count:
        if file_failure_count == 1:
            summary, detail = rendered_file_failures[0]
        else:
            summaries = "; ".join(summary for summary, _ in rendered_file_failures)
            omitted = file_failure_count - len(rendered_file_failures)
            if omitted:
                summaries += f"; ... and {omitted} more"
            summary = _trim_text(
                f"{file_failure_count} task files failed parsing or validation: {summaries}",
                max_length=500,
            )
            details = "\n\n".join(
                f"{summary_line}\n{detail_line}"
                for summary_line, detail_line in rendered_file_failures
            )
            if omitted:
                details += f"\n\n... and {omitted} more task file failures"
            detail = _trim_text(details, max_length=2000)
        return TaskValidationFailure(
            summary=summary,
            detail=detail,
            offending_paths=tuple(file_failure_paths),
        )
    # ``validate_task_graph`` is the exact invariant set enforced by strict
    # ``TaskManager`` construction (whose model validator delegates to it),
    # called directly so a graph failure carries the offending task ids and can
    # be attributed to the task files that declared them (issue #128).
    try:
        validate_task_graph(tasks)
    except TaskGraphValidationError as exc:
        return TaskValidationFailure(
            summary=f"task dependency graph is invalid: {_trim_text(str(exc), max_length=500)}",
            detail=_trim_text(str(exc), max_length=2000),
            offending_paths=_paths_for_task_ids(
                exc.offending_task_ids,
                paths_by_task_id,
                resolve_missing_task_id=resolve_missing_task_id,
            ),
        )
    # Safety net: strict ``TaskManager`` construction stays the authoritative
    # gate, so a future ``TaskManager`` validator not mirrored by
    # ``validate_task_graph`` is still reported as a validation failure instead
    # of crashing the caller (pydantic v2 funnels validator ValueErrors into
    # ValidationError, but a future field-validator change could raise
    # TypeError/ValueError directly; widen the catch accordingly).
    try:
        TaskManager(tasks=list(tasks))
    except (ValidationError, ValueError, TypeError) as exc:
        return TaskValidationFailure(
            summary=f"task dependency graph is invalid: {_trim_text(str(exc), max_length=500)}",
            detail=_trim_text(str(exc), max_length=2000),
        )
    return None


def _paths_for_task_ids(
    task_ids: tuple[str, ...],
    paths_by_task_id: dict[str, list[Path]],
    *,
    resolve_missing_task_id: Callable[[str], tuple[str, ...]] | None = None,
) -> tuple[str, ...]:
    """Map offending task ids to the task file(s) that declared them.

    An id declared by multiple files (the duplicate-id failure) maps to every
    declaring file; an id with no file (a dangling dependency reference) maps to
    whatever ``resolve_missing_task_id`` returns for it (the pre-merge declaring
    file, when the caller provides a resolver), else nothing. Order follows
    ``task_ids`` with duplicates dropped.
    """

    out: list[str] = []
    seen: set[str] = set()
    for task_id in task_ids:
        declared = paths_by_task_id.get(task_id)
        if declared is not None:
            candidates: tuple[str, ...] = tuple(str(path) for path in declared)
        elif resolve_missing_task_id is not None:
            candidates = resolve_missing_task_id(task_id)
        else:
            candidates = ()
        for text in candidates:
            if text not in seen:
                seen.add(text)
                out.append(text)
    return tuple(out)


def _trim_text(text: str, *, max_length: int = 4000) -> str:
    if len(text) <= max_length:
        return text
    return f"{text[:max_length]}\n... <truncated>"


__all__ = ("TaskValidationFailure", "validate_task_directory")
