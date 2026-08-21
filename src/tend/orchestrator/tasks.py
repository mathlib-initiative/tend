"""Task models for the orchestrator.

The canonical task format is a YAML file ``tasks/<id>.yaml`` with the fields
``schema_version``, ``id``, ``title``, ``status`` (``open``/``complete``),
``priority`` (``default``/``high``/``max``), ``depends_on``, ``summary``, and
``description``, plus optional ``notes``.
"""

from __future__ import annotations

from enum import StrEnum
from typing import ClassVar, Final, Literal, cast

from pydantic import ConfigDict, Field, field_validator, model_validator

from tend._common.types import StrictModel

_SCHEMA_VERSION: Literal["1"] = "1"


def _empty_dependencies() -> list[str]:
    return []


class TaskStatus(StrEnum):
    """Task lifecycle status for dependency readiness.

    ``open`` means work remains; ``complete`` is the terminal task state.
    """

    OPEN = "open"
    COMPLETE = "complete"


class TaskPriority(StrEnum):
    """Task scheduling priority for ready-task queue admission.

    Higher-priority ready tasks are picked from the task queue first. ``default``
    preserves the historical behavior for task files that omit ``priority``.
    """

    DEFAULT = "default"
    HIGH = "high"
    MAX = "max"


_TASK_PRIORITY_RANK: Final[dict[TaskPriority, int]] = {
    TaskPriority.MAX: 0,
    TaskPriority.HIGH: 1,
    TaskPriority.DEFAULT: 2,
}


def task_priority_rank(priority: TaskPriority) -> int:
    """Return the queue ordering rank for ``priority`` (lower is scheduled first)."""

    return _TASK_PRIORITY_RANK[priority]


class Task(StrictModel):
    """A unit of work for the orchestrator.

    The YAML dependency key is ``depends_on``; the Python attribute is
    ``dependencies`` (both spellings are accepted by the constructor).
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
        populate_by_name=True,
    )

    schema_version: Literal["1"] = _SCHEMA_VERSION
    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    status: TaskStatus = TaskStatus.OPEN
    priority: TaskPriority = TaskPriority.DEFAULT
    dependencies: list[str] = Field(default_factory=_empty_dependencies, alias="depends_on")
    summary: str = Field(min_length=1)
    description: str = Field(min_length=1)
    # ``description`` is the spec — what the task must accomplish, set when the
    # task is created (only corrected if the spec itself is wrong). ``notes`` is
    # the running hand-off: what a worker did, found, or got blocked on this
    # run, written for the next worker that picks the task up. Optional and
    # ``None`` when absent, so ``dump_task_yaml``'s ``exclude_none`` keeps it out
    # of the many task files that never carry a hand-off.
    notes: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _default_title_from_summary(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        mapping = dict(cast("dict[object, object]", data))
        title = mapping.get("title")
        if title is None or (isinstance(title, str) and not title.strip()):
            summary = mapping.get("summary")
            if isinstance(summary, str):
                mapping["title"] = summary
        return mapping

    @field_validator("schema_version", mode="before")
    @classmethod
    def _coerce_schema_version(cls, value: object) -> object:
        # Tolerate a YAML integer ``schema_version: 1`` for the unified task
        # format. Agent-written subtask files vary between ``1`` and ``"1"``;
        # rejecting the integer form (or, in the sync engine, silently skipping
        # the file) loses tasks for what is purely a YAML scalar-type quirk.
        # Bool is checked because ``bool`` is a subclass of ``int`` in Python.
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
        return value

    @field_validator("priority", mode="before")
    @classmethod
    def _coerce_priority(cls, value: object) -> object:
        if isinstance(value, str):
            try:
                return TaskPriority(value)
            except ValueError:
                return value
        return value

    @field_validator("id", "title", "summary", "description")
    @classmethod
    def _validate_non_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("task text fields must not be blank")
        return value

    @field_validator("dependencies")
    @classmethod
    def _validate_dependencies(cls, dependencies: list[str]) -> list[str]:
        seen: set[str] = set()
        for dependency_id in dependencies:
            if not dependency_id.strip():
                raise ValueError("task dependency IDs must not be blank")
            if dependency_id in seen:
                raise ValueError(f"duplicate task dependency id: {dependency_id}")
            seen.add(dependency_id)
        return dependencies

    @property
    def body(self) -> str:
        """Return the free-form task body (alias for ``description``)."""

        return self.description
