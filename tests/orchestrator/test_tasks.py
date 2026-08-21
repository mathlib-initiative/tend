from __future__ import annotations

import pytest
from pydantic import ValidationError

import tend.orchestrator.task_manager as task_manager_module
from tend.orchestrator.task_manager import (
    TaskGraphValidationError,
    TaskManager,
    build_resilient_task_manager,
    validate_task_graph,
)
from tend.orchestrator.tasks import Task, TaskPriority, TaskStatus


def test_task_accepts_dependency_ids_and_defaults_to_open() -> None:
    # Use model_validate (no explicit title) to exercise the title-from-summary default.
    task = Task.model_validate(
        {
            "id": "task-2",
            "summary": "Use seed",
            "description": "Build on the seed.",
            "depends_on": ["task-1"],
        }
    )

    assert task.status is TaskStatus.OPEN
    assert task.priority is TaskPriority.DEFAULT
    assert task.dependencies == ["task-1"]
    # Title defaults from summary when not provided.
    assert task.title == "Use seed"


def test_task_accepts_depends_on_alias_and_explicit_title() -> None:
    task = Task.model_validate(
        {
            "schema_version": "1",
            "id": "task-2",
            "title": "Explicit title",
            "summary": "Use seed",
            "description": "Build on the seed.",
            "depends_on": ["task-1"],
        }
    )

    assert task.title == "Explicit title"
    assert task.dependencies == ["task-1"]


def test_task_accepts_explicit_priority() -> None:
    task = Task.model_validate(
        {
            "id": "task-2",
            "summary": "Steer the run",
            "description": "Create a blocking steering task.",
            "priority": "max",
        }
    )

    assert task.priority is TaskPriority.MAX


@pytest.mark.parametrize("priority", ["low", "", "urgent"])
def test_task_rejects_unknown_priority(priority: str) -> None:
    with pytest.raises(ValidationError):
        Task.model_validate(
            {
                "id": "task-2",
                "summary": "Bad priority",
                "description": "Priority must be one of the supported values.",
                "priority": priority,
            }
        )


def test_task_accepts_integer_schema_version() -> None:
    # YAML scalar quirk: ``schema_version: 1`` parses to the integer 1, while
    # ``schema_version: "1"`` parses to "1". Agent-written subtask files vary
    # between the two; the async Task must accept both so the unified task
    # format parses identically to the sync engine (which already coerces via
    # ``_normalise_scalar``).
    task = Task.model_validate(
        {
            "schema_version": 1,
            "id": "task-2",
            "title": "Integer schema_version",
            "summary": "Use seed",
            "description": "Build on the seed.",
            "depends_on": [],
        }
    )

    assert task.schema_version == "1"


def test_task_rejects_blank_text() -> None:
    with pytest.raises(ValidationError, match="task text fields must not be blank"):
        Task(id=" ", title="Summary", summary="Summary", description="Description")


def test_task_rejects_blank_dependency_ids() -> None:
    with pytest.raises(ValidationError, match="task dependency IDs must not be blank"):
        Task(
            id="task-1",
            title="Task", summary="Task",
            description="Task.",
            depends_on=[" "],
        )


def test_task_notes_defaults_empty_and_round_trips() -> None:
    # ``notes`` is the optional worker hand-off field: absent -> None, and a
    # provided value is preserved (it is not folded into ``description``).
    bare = Task.model_validate(
        {"id": "task-2", "summary": "Use seed", "description": "Build on the seed."}
    )
    assert bare.notes is None

    with_notes = Task.model_validate(
        {
            "id": "task-2",
            "summary": "Use seed",
            "description": "Build on the seed.",
            "notes": "Stated `foo`; `bar` remains, blocked on task-3.",
        }
    )
    assert with_notes.notes == "Stated `foo`; `bar` remains, blocked on task-3."
    assert with_notes.description == "Build on the seed."


def test_task_still_rejects_unknown_top_level_key() -> None:
    # The schema stays strict: hand-off prose has a home (``notes``), so a
    # stray top-level key like ``progress`` remains a hard error — the
    # validation-failure feedback then points the next worker at ``notes``.
    with pytest.raises(ValidationError, match="progress"):
        Task.model_validate(
            {
                "id": "task-2",
                "summary": "Use seed",
                "description": "Build on the seed.",
                "progress": "did some stuff",
            }
        )


def test_task_manager_stores_tasks_as_dependency_dag() -> None:
    first = Task(id="task-1", title="First", summary="First", description="First task.")
    second = Task(
        id="task-2",
        title="Second", summary="Second",
        description="Second task.",
        depends_on=[first.id],
    )

    manager = TaskManager(tasks=[first, second])

    assert manager.tasks == [first, second]
    assert manager.task_ids == ("task-1", "task-2")


def test_task_manager_rejects_duplicate_top_level_task_ids() -> None:
    first = Task(id="task-1", title="First", summary="First", description="First task.")
    duplicate = Task(
        id="task-1", title="Duplicate", summary="Duplicate", description="Duplicate task."
    )

    with pytest.raises(ValidationError, match="duplicate task id"):
        TaskManager(tasks=[first, duplicate])


def test_task_manager_rejects_unknown_dependency_ids() -> None:
    task = Task(
        id="task-1",
        title="Task", summary="Task",
        description="Task.",
        depends_on=["missing"],
    )

    with pytest.raises(ValidationError, match="unknown task id"):
        TaskManager(tasks=[task])


def test_task_manager_rejects_dependency_cycles() -> None:
    task = Task(
        id="task-1",
        title="Task", summary="Task",
        description="Task.",
        depends_on=["task-1"],
    )

    with pytest.raises(ValidationError, match="task dependency cycle detected"):
        TaskManager(tasks=[task])


def test_task_manager_rejects_complete_tasks_with_transitive_open_dependencies() -> None:
    open_dependency = Task(
        id="task-1",
        title="Open dependency", summary="Open dependency",
        description="Still open.",
    )
    complete_middle = Task(
        id="task-2",
        title="Complete middle", summary="Complete middle",
        description="Invalidly complete.",
        status=TaskStatus.COMPLETE,
        depends_on=[open_dependency.id],
    )
    complete_parent = Task(
        id="task-3",
        title="Complete parent", summary="Complete parent",
        description="Also invalid through transitive dependency.",
        status=TaskStatus.COMPLETE,
        depends_on=[complete_middle.id],
    )

    with pytest.raises(ValidationError, match="complete task cannot depend on open task"):
        TaskManager(tasks=[open_dependency, complete_middle, complete_parent])


def _graph_task(
    task_id: str,
    *,
    depends_on: list[str] | None = None,
    status: TaskStatus = TaskStatus.OPEN,
) -> Task:
    return Task(
        id=task_id,
        title=task_id,
        summary=task_id,
        description=f"{task_id} description.",
        depends_on=[] if depends_on is None else depends_on,
        status=status,
    )


def test_validate_task_graph_names_every_cycle_member() -> None:
    """A dependency cycle carries the full cycle path, not just the entry task.

    The offending ids are what the merge gate maps back to task files to bounce
    the responsible contribution without bisecting (issue #128); an innocent
    task outside the cycle must not be implicated.
    """

    innocent = _graph_task("task-innocent")
    cyc_a = _graph_task("task-a", depends_on=["task-b"])
    cyc_b = _graph_task("task-b", depends_on=["task-c"])
    cyc_c = _graph_task("task-c", depends_on=["task-a"])

    with pytest.raises(TaskGraphValidationError) as exc_info:
        validate_task_graph([innocent, cyc_a, cyc_b, cyc_c])

    # The human-readable message is unchanged; the structure is additive.
    assert str(exc_info.value) == "task dependency cycle detected at task id: task-a"
    assert exc_info.value.offending_task_ids == ("task-a", "task-b", "task-c")


def test_validate_task_graph_names_both_ids_of_complete_depends_on_open() -> None:
    open_task = _graph_task("task-open")
    complete_task = _graph_task(
        "task-complete", depends_on=["task-open"], status=TaskStatus.COMPLETE
    )

    with pytest.raises(TaskGraphValidationError) as exc_info:
        validate_task_graph([open_task, complete_task])

    # The direct (two-node) case keeps the historical message shape.
    assert str(exc_info.value) == (
        "complete task cannot depend on open task: task-complete depends on task-open"
    )
    assert exc_info.value.offending_task_ids == ("task-complete", "task-open")


def test_validate_task_graph_names_full_path_of_transitive_complete_depends_on_open() -> None:
    """Every task along the dependency path is offending, not just the endpoints.

    For complete ``task-parent`` -> complete ``task-middle`` -> open
    ``task-open``, the causative edit may be the *intermediate* edge (editing
    middle to depend on open), whose file the endpoint ids never name — so the
    merge gate could not attribute it (round-2 adversarial review of
    issue #128). The path is spelled out in the message; both complete tasks on
    the chain report their own violation (all violations are aggregated).
    """

    open_task = _graph_task("task-open")
    middle = _graph_task("task-middle", depends_on=["task-open"], status=TaskStatus.COMPLETE)
    parent = _graph_task("task-parent", depends_on=["task-middle"], status=TaskStatus.COMPLETE)

    with pytest.raises(TaskGraphValidationError) as exc_info:
        validate_task_graph([parent, middle, open_task])

    message = str(exc_info.value)
    assert message.startswith("task graph has 2 violations: ")
    assert (
        "complete task cannot depend on open task: task-parent depends on task-open "
        "(via task-parent -> task-middle -> task-open)"
    ) in message
    assert (
        "complete task cannot depend on open task: task-middle depends on task-open" in message
    )
    assert exc_info.value.offending_task_ids == ("task-parent", "task-middle", "task-open")


def test_validate_task_graph_names_intermediates_on_alternate_paths_to_same_open_task() -> None:
    """Attribution includes every path even when open endpoints are shared."""

    open_task = _graph_task("task-open")
    middle_a = _graph_task(
        "task-middle-a", depends_on=[open_task.id], status=TaskStatus.COMPLETE
    )
    middle_b = _graph_task(
        "task-middle-b", depends_on=[open_task.id], status=TaskStatus.COMPLETE
    )
    parent = _graph_task(
        "task-parent",
        depends_on=[middle_a.id, middle_b.id],
        status=TaskStatus.COMPLETE,
    )

    with pytest.raises(TaskGraphValidationError) as exc_info:
        validate_task_graph([parent, middle_a, middle_b, open_task])

    assert set(exc_info.value.offending_task_ids) == {
        parent.id,
        middle_a.id,
        middle_b.id,
        open_task.id,
    }


def test_validate_task_graph_names_duplicate_id() -> None:
    with pytest.raises(TaskGraphValidationError) as exc_info:
        validate_task_graph([_graph_task("task-1"), _graph_task("task-1")])

    assert str(exc_info.value) == "duplicate task id: task-1"
    assert exc_info.value.offending_task_ids == ("task-1",)


def test_validate_task_graph_uses_deterministic_scc_cycle_anchor() -> None:
    """Cycle messages use the insertion-first member of the cyclic SCC."""

    task_a = _graph_task("task-a", depends_on=["task-c"])
    task_b = _graph_task("task-b", depends_on=["task-c"])
    task_c = _graph_task("task-c", depends_on=["task-b"])

    with pytest.raises(TaskGraphValidationError) as exc_info:
        validate_task_graph([task_a, task_b, task_c])

    assert str(exc_info.value) == "task dependency cycle detected at task id: task-b"
    assert exc_info.value.offending_task_ids == ("task-b", "task-c")


def test_validate_task_graph_reports_every_member_of_overlapping_cycles() -> None:
    """One SCC can contain overlapping cycles not exposed as DFS back edges."""

    task_a = _graph_task("task-a", depends_on=["task-b", "task-c"])
    task_b = _graph_task("task-b", depends_on=["task-a"])
    task_c = _graph_task("task-c", depends_on=["task-b"])

    with pytest.raises(TaskGraphValidationError) as exc_info:
        validate_task_graph([task_a, task_b, task_c])

    assert "task dependency cycle detected" in str(exc_info.value)
    assert exc_info.value.offending_task_ids == ("task-a", "task-b", "task-c")


def test_open_dependency_path_materialization_is_bounded_for_high_fanout() -> None:
    """A long complete chain into many open leaves retains only capped paths."""

    chain_length = 700
    leaf_count = 20_000
    leaves = [_graph_task(f"leaf-{index:05d}") for index in range(leaf_count)]
    chain: list[Task] = []
    for index in reversed(range(chain_length)):
        dependencies = (
            [leaf.id for leaf in leaves]
            if index == chain_length - 1
            else [f"chain-{index + 1:04d}"]
        )
        chain.append(
            _graph_task(
                f"chain-{index:04d}",
                depends_on=dependencies,
                status=TaskStatus.COMPLETE,
            )
        )
    chain.reverse()
    tasks_by_id = {task.id: task for task in (*chain, *leaves)}

    paths, endpoint_count, implicated_ids = (
        task_manager_module._open_dependency_violations(  # pyright: ignore[reportPrivateUsage]
            chain[0], tasks_by_id, max_rendered_paths=30
        )
    )

    assert endpoint_count == leaf_count
    assert len(paths) == 30
    # At most 30 chain-to-leaf tuples are built, rather than 20,000 tuples
    # containing roughly 14 million Task references.
    assert sum(len(path) for path in paths) == 30 * (chain_length + 1)
    assert len(implicated_ids) == chain_length + leaf_count


def test_validate_task_graph_returns_known_violation_after_deep_acyclic_chain() -> None:
    """Iterative SCC discovery cannot mask a duplicate with RecursionError."""

    chain_length = 5_000
    tasks = [_graph_task("duplicate"), _graph_task("duplicate")]
    tasks.extend(
        _graph_task(
            f"chain-{index:04d}",
            depends_on=[f"chain-{index + 1:04d}"] if index + 1 < chain_length else [],
        )
        for index in range(chain_length)
    )

    with pytest.raises(TaskGraphValidationError) as exc_info:
        validate_task_graph(tasks)

    assert str(exc_info.value) == "duplicate task id: duplicate"
    assert exc_info.value.offending_task_ids == ("duplicate",)


def test_validate_task_graph_bounds_aggregated_error_while_retaining_all_ids() -> None:
    """Large violating graphs never build a multi-megabyte exception string."""

    chain_length = 800
    tasks = [_graph_task("task-open")]
    dependency_id = "task-open"
    for index in range(chain_length):
        task_id = f"task-complete-{index:04d}"
        tasks.append(
            _graph_task(
                task_id,
                depends_on=[dependency_id],
                status=TaskStatus.COMPLETE,
            )
        )
        dependency_id = task_id

    with pytest.raises(TaskGraphValidationError) as exc_info:
        validate_task_graph(list(reversed(tasks)))

    assert len(str(exc_info.value)) < 20_000
    assert len(exc_info.value.offending_task_ids) == chain_length + 1
    assert "and 770 more" in str(exc_info.value)


def test_validate_task_graph_reports_every_cycle_not_just_the_first() -> None:
    """Two independent cycles are both reported in one error.

    Stopping at the first cycle let a batch of N independently cyclic
    contributions be attributed only one member per merge-isolation round
    (round-3 adversarial review of issue #128); the union of all cycle ids
    attributes every implicated file at once.
    """

    a1 = _graph_task("task-a1", depends_on=["task-a2"])
    a2 = _graph_task("task-a2", depends_on=["task-a1"])
    b1 = _graph_task("task-b1", depends_on=["task-b2"])
    b2 = _graph_task("task-b2", depends_on=["task-b1"])

    with pytest.raises(TaskGraphValidationError) as exc_info:
        validate_task_graph([a1, a2, b1, b2])

    message = str(exc_info.value)
    assert message.startswith("task graph has 2 violations: ")
    assert "task dependency cycle detected at task id: task-a1" in message
    assert "task dependency cycle detected at task id: task-b1" in message
    assert exc_info.value.offending_task_ids == ("task-a1", "task-a2", "task-b1", "task-b2")


def test_validate_task_graph_reports_cycle_and_duplicate_together() -> None:
    """Violations of different classes are aggregated, not reported one per raise."""

    duplicated = _graph_task("task-dup")
    cyc_a = _graph_task("task-cyc-a", depends_on=["task-cyc-b"])
    cyc_b = _graph_task("task-cyc-b", depends_on=["task-cyc-a"])

    with pytest.raises(TaskGraphValidationError) as exc_info:
        validate_task_graph([duplicated, _graph_task("task-dup"), cyc_a, cyc_b])

    message = str(exc_info.value)
    assert message.startswith("task graph has 2 violations: ")
    assert "duplicate task id: task-dup" in message
    assert "task dependency cycle detected at task id: task-cyc-a" in message
    assert exc_info.value.offending_task_ids == ("task-dup", "task-cyc-a", "task-cyc-b")


def test_validate_task_graph_scans_past_unknown_dependencies() -> None:
    """An unknown-dependency violation no longer masks later checks: the cycle
    and complete->open scans skip the broken edge and still report their own
    violations in the same aggregated error."""

    dangling = _graph_task("task-dangling", depends_on=["task-missing"])
    open_task = _graph_task("task-open")
    complete_task = _graph_task(
        "task-complete", depends_on=["task-open", "task-missing"], status=TaskStatus.COMPLETE
    )

    with pytest.raises(TaskGraphValidationError) as exc_info:
        validate_task_graph([dangling, open_task, complete_task])

    message = str(exc_info.value)
    assert message.startswith("task graph has 3 violations: ")
    assert "task-dangling depends on task-missing" in message
    assert "task-complete depends on task-missing" in message
    assert "complete task cannot depend on open task: task-complete depends on task-open" in (
        message
    )
    assert exc_info.value.offending_task_ids == (
        "task-dangling",
        "task-missing",
        "task-complete",
        "task-open",
    )


def test_validate_task_graph_names_both_ids_of_unknown_dependency() -> None:
    """Both the depender and the missing id are offending: the missing id has no
    file in the validated set, but when a merge *deleted* its declaring file the
    merge gate resolves it against the pre-merge tree and attributes the
    deletion (adversarial review of issue #128).
    """

    with pytest.raises(TaskGraphValidationError) as exc_info:
        validate_task_graph([_graph_task("task-1", depends_on=["task-missing"])])

    assert str(exc_info.value) == (
        "task dependency references unknown task id: task-1 depends on task-missing"
    )
    assert exc_info.value.offending_task_ids == ("task-1", "task-missing")


def test_validate_task_graph_accepts_valid_dag() -> None:
    first = _graph_task("task-1")
    second = _graph_task("task-2", depends_on=["task-1"])

    validate_task_graph([first, second])


def test_task_manager_returns_ready_open_tasks_without_open_dependencies() -> None:
    complete_dependency = Task(
        id="task-1",
        title="Complete dependency", summary="Complete dependency",
        description="Already done.",
        status=TaskStatus.COMPLETE,
    )
    open_dependency = Task(
        id="task-2",
        title="Open dependency", summary="Open dependency",
        description="Still pending.",
    )
    ready = Task(
        id="task-3",
        title="Ready", summary="Ready",
        description="Can run now.",
        depends_on=[complete_dependency.id],
    )
    blocked = Task(
        id="task-4",
        title="Blocked", summary="Blocked",
        description="Cannot run yet.",
        depends_on=[open_dependency.id],
    )
    already_complete = Task(
        id="task-5",
        title="Complete", summary="Complete",
        description="Should not be returned.",
        status=TaskStatus.COMPLETE,
    )

    manager = TaskManager(
        tasks=[complete_dependency, open_dependency, ready, blocked, already_complete],
    )

    assert manager.ready_tasks() == (open_dependency, ready)


def test_task_manager_orders_ready_tasks_by_priority_then_insertion_order() -> None:
    first_default = Task(
        id="task-1",
        title="Default first",
        summary="Default first",
        description="Default priority.",
    )
    high = Task(
        id="task-2",
        title="High",
        summary="High",
        description="High priority.",
        priority=TaskPriority.HIGH,
    )
    second_default = Task(
        id="task-3",
        title="Default second",
        summary="Default second",
        description="Default priority.",
    )
    max_priority = Task(
        id="task-4",
        title="Max",
        summary="Max",
        description="Max priority.",
        priority=TaskPriority.MAX,
    )
    high_later = Task(
        id="task-5",
        title="High later",
        summary="High later",
        description="High priority but later in file order.",
        priority=TaskPriority.HIGH,
    )

    manager = TaskManager(
        tasks=[first_default, high, second_default, max_priority, high_later]
    )

    assert manager.ready_tasks() == (
        max_priority,
        high,
        high_later,
        first_default,
        second_default,
    )


def _task(task_id: str, *, depends_on: list[str] | None = None, complete: bool = False) -> Task:
    return Task(
        id=task_id,
        title=task_id,
        summary=task_id,
        description=f"{task_id} body.",
        status=TaskStatus.COMPLETE if complete else TaskStatus.OPEN,
        depends_on=depends_on or [],
    )


def test_build_resilient_task_manager_drops_duplicate_ids() -> None:
    first = _task("task-1")
    duplicate = _task("task-1")

    manager = build_resilient_task_manager([first, duplicate])

    # The first occurrence wins; the duplicate is dropped instead of raising.
    assert manager.tasks == [first]


def test_build_resilient_task_manager_drops_unknown_dependency_edges() -> None:
    manager = build_resilient_task_manager([_task("task-1", depends_on=["ghost"])])

    assert manager.task_ids == ("task-1",)
    assert manager.tasks[0].dependencies == []


def test_build_resilient_task_manager_breaks_dependency_cycles() -> None:
    a = _task("task-a", depends_on=["task-b"])
    b = _task("task-b", depends_on=["task-a"])

    manager = build_resilient_task_manager([a, b])

    deps = {task.id: task.dependencies for task in manager.tasks}
    # Exactly one back edge is dropped to break the cycle (deterministically): the
    # sorted traversal starts at task-a -> task-b, and the task-b -> task-a edge
    # closes the cycle, so it is the one removed.
    assert (deps["task-a"], deps["task-b"]) == (["task-b"], [])


def test_build_resilient_task_manager_demotes_complete_task_with_open_dependency() -> None:
    open_dep = _task("task-1")
    complete_child = _task("task-2", depends_on=["task-1"], complete=True)

    manager = build_resilient_task_manager([open_dep, complete_child])

    statuses = {task.id: task.status for task in manager.tasks}
    assert statuses["task-2"] is TaskStatus.OPEN
