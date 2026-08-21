"""Task management for the async orchestrator."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence

from pydantic import Field, model_validator

from tend._common.types import StrictModel
from tend.orchestrator.tasks import Task, TaskStatus, task_priority_rank

_LOGGER = logging.getLogger(__name__)

_MAX_RENDERED_GRAPH_VIOLATIONS = 30
_MAX_RENDERED_PATH_TASK_IDS = 12
_MAX_RENDERED_TASK_ID_LENGTH = 160


class TaskGraphValidationError(ValueError):
    """Task-graph invariant violation(s) naming the offending task id(s).

    ``offending_task_ids`` carries the ordered, de-duplicated union of the task
    ids implicated by *every* violation found — every member of each cyclic
    strongly-connected component, every task on any dependency path of each
    complete-depends-on-open violation (endpoints *and* intermediates, since
    the causative edit may be an intermediate edge), the depender *and* the
    missing id of each unknown-dependency reference, each duplicated id — so
    callers (``task_validation``) can map them back to the task files that
    declared them and let the merge gate probe the touching contribution(s)
    first instead of blind bisection (issue #128). A ``ValueError`` subclass so
    pydantic model validators funnel it into ``ValidationError`` unchanged.
    """

    offending_task_ids: tuple[str, ...]

    def __init__(self, message: str, *, offending_task_ids: tuple[str, ...]) -> None:
        super().__init__(message)
        self.offending_task_ids = offending_task_ids


def validate_task_graph(tasks: Sequence[Task]) -> None:
    """Validate the strict task-dependency-graph invariants of ``TaskManager``.

    Raises :class:`TaskGraphValidationError` — naming the offending task ids —
    on a duplicate task id, a dependency on an unknown task id, a dependency
    cycle, or a ``complete`` task that (transitively) depends on an ``open``
    task. This is the exact invariant set enforced by strict ``TaskManager``
    construction, whose model validator delegates here.

    ALL violations present are aggregated into the single raised error rather
    than stopping at the first: the bounded message uses deterministic
    SCC-based cycle anchors and ``offending_task_ids`` is the ordered,
    de-duplicated union. This is what
    lets the merge gate attribute *every* implicated task file in one round —
    a batch of N independently broken contributions is fully attributed by its
    first gate-1 failure and resolved in O(N log N) build-free validations,
    instead of one violation (and one whole-remainder revalidation) per round
    (round-3 adversarial review of issue #128). Later checks tolerate the
    damage earlier ones found: unknown-dependency edges are skipped (not
    crashed on) by the cycle and complete->open scans, and duplicates keep
    their first declaration, matching ``tasks_by_id`` semantics elsewhere.
    """

    rendered_violations: list[str] = []
    violation_count = 0
    offending: list[str] = []
    offending_seen: set[str] = set()

    def add_offending(task_ids: Iterable[str]) -> None:
        for task_id in task_ids:
            if task_id not in offending_seen:
                offending_seen.add(task_id)
                offending.append(task_id)

    def add_violation(message: str, task_ids: Iterable[str]) -> None:
        nonlocal violation_count
        violation_count += 1
        # Worker-controlled graphs can contain hundreds of thousands of
        # violations. Keep counting and attributing all of them, but retain only
        # a bounded sample for the human-readable exception.
        if len(rendered_violations) < _MAX_RENDERED_GRAPH_VIOLATIONS:
            rendered_violations.append(message)
        add_offending(task_ids)

    tasks_by_id: dict[str, Task] = {}
    for task in tasks:
        if task.id in tasks_by_id:
            add_violation(f"duplicate task id: {_display_task_id(task.id)}", (task.id,))
        else:
            tasks_by_id[task.id] = task

    for task in tasks_by_id.values():
        for dependency_id in task.dependencies:
            if dependency_id not in tasks_by_id:
                # The missing id is offending too: it has no declaring file in
                # the validated set, but when the reference broke because a
                # merge *deleted* the declaring file, the merge gate can map it
                # to that pre-merge file and attribute the deletion.
                add_violation(
                    "task dependency references unknown task id: "
                    f"{_display_task_id(task.id)} depends on "
                    f"{_display_task_id(dependency_id)}",
                    (task.id, dependency_id),
                )

    for component in _cyclic_components(tasks_by_id):
        entry_id = component[0]
        add_violation(
            f"task dependency cycle detected at task id: {_display_task_id(entry_id)}",
            component,
        )

    for task in tasks_by_id.values():
        if task.status is not TaskStatus.COMPLETE:
            continue
        remaining_message_slots = max(
            0, _MAX_RENDERED_GRAPH_VIOLATIONS - len(rendered_violations)
        )
        paths, open_endpoint_count, implicated_ids = _open_dependency_violations(
            task,
            tasks_by_id,
            max_rendered_paths=remaining_message_slots,
        )
        # ``paths`` is only a bounded readable sample. ``implicated_ids`` stays
        # complete and includes nodes on alternate paths, so rendering limits
        # never lose a causative intermediate edit from in-memory attribution.
        for path in paths:
            message = (
                "complete task cannot depend on open task: "
                f"{_display_task_id(task.id)} depends on {_display_task_id(path[-1].id)}"
            )
            if len(path) > 2:
                message += f" (via {_format_task_path(path)})"
            add_violation(message, (step.id for step in path))
        violation_count += open_endpoint_count - len(paths)
        add_offending(implicated_ids)

    if violation_count:
        if violation_count == 1:
            message = rendered_violations[0]
        else:
            message = (
                f"task graph has {violation_count} violations: "
                + "; ".join(rendered_violations)
            )
            omitted = violation_count - len(rendered_violations)
            if omitted:
                message += f"; ... and {omitted} more"
        raise TaskGraphValidationError(message, offending_task_ids=tuple(offending))


def _empty_tasks() -> list[Task]:
    return []


class TaskManager(StrictModel):
    """Stores tasks that are expected to form a dependency DAG."""

    tasks: list[Task] = Field(default_factory=_empty_tasks)

    @model_validator(mode="after")
    def _validate_task_dag(self) -> TaskManager:
        validate_task_graph(self.tasks)
        return self

    @property
    def task_ids(self) -> tuple[str, ...]:
        """Return managed task IDs in insertion order."""

        return tuple(task.id for task in self.tasks)

    def ready_tasks(self) -> tuple[Task, ...]:
        """Return ready open tasks in scheduling order.

        Priority orders ready tasks before file/insertion order: ``max`` before
        ``high`` before ``default``. Tasks with the same priority retain
        ``self.tasks`` order.
        """

        tasks_by_id = {task.id: task for task in self.tasks}
        ready = [
            task
            for task in self.tasks
            if task.status is TaskStatus.OPEN
            and all(
                tasks_by_id[dependency_id].status is TaskStatus.COMPLETE
                for dependency_id in task.dependencies
            )
        ]
        ready.sort(key=lambda task: task_priority_rank(task.priority))
        return tuple(ready)


def build_resilient_task_manager(tasks: Iterable[Task]) -> TaskManager:
    """Build a ``TaskManager`` that tolerates malformed worker-written task graphs.

    The strict ``TaskManager`` constructor raises on duplicate IDs, unknown
    dependency references, dependency cycles, and complete-depends-on-open
    invariants. In the live polling loop those raises tear down the whole run on
    a single bad task file written by a worker. This builder mirrors the sync
    orchestrator's skip-and-relax behavior so one bad graph cannot halt the run:

    - duplicate task IDs: keep the first occurrence, drop later ones (warn);
    - dependencies referencing unknown task IDs: drop the edge (warn);
    - dependency cycles: drop the minimal set of back edges, deterministically
      (sorted traversal), mirroring ``orchestrator.task_graph`` (warn);
    - a ``complete`` task that (transitively) depends on an ``open`` task: demote
      it to ``open`` so work keeps flowing instead of raising (warn).

    The result always satisfies the strict ``TaskManager`` invariants.
    """

    deduped: dict[str, Task] = {}
    for task in tasks:
        if task.id in deduped:
            _LOGGER.warning("dropping duplicate async task id: %s", task.id)
            continue
        deduped[task.id] = task

    known_ids = set(deduped)
    # Drop edges that reference unknown task IDs.
    pruned_dependencies: dict[str, list[str]] = {}
    for task_id, task in deduped.items():
        kept: list[str] = []
        for dependency_id in task.dependencies:
            if dependency_id not in known_ids:
                _LOGGER.warning(
                    "dropping async task dependency on unknown task id: %s depends on %s",
                    task_id,
                    dependency_id,
                )
                continue
            kept.append(dependency_id)
        pruned_dependencies[task_id] = kept

    # Break dependency cycles by dropping back edges (deterministic, sorted).
    pruned_dependencies = _break_dependency_cycles(pruned_dependencies)

    sanitized = {
        task_id: task.model_copy(update={"dependencies": pruned_dependencies[task_id]})
        for task_id, task in deduped.items()
    }

    # Demote complete tasks that (transitively) depend on an open task to open.
    for task_id, task in sanitized.items():
        if task.status is TaskStatus.COMPLETE:
            open_dependency_path = _open_dependency_path(task, sanitized)
            if open_dependency_path is not None:
                _LOGGER.warning(
                    "demoting complete async task to open: %s depends on open task %s",
                    task_id,
                    open_dependency_path[-1].id,
                )
                sanitized[task_id] = task.model_copy(update={"status": TaskStatus.OPEN})

    return TaskManager(tasks=list(sanitized.values()))


def _break_dependency_cycles(dependencies_by_task: dict[str, list[str]]) -> dict[str, list[str]]:
    """Return dependencies with cycle-closing back edges removed.

    Traversal visits tasks and their dependencies in sorted order so the same
    cycle always drops the same edge across runs. Mirrors the sync orchestrator's
    ``task_graph._break_dependency_cycles``.
    """

    dropped: set[tuple[str, str]] = set()
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(task_id: str) -> None:
        visiting.add(task_id)
        for dependency_id in sorted(dependencies_by_task.get(task_id, ())):
            if (task_id, dependency_id) in dropped:
                continue
            if dependency_id in visiting:
                dropped.add((task_id, dependency_id))  # back edge closes a cycle
                _LOGGER.warning(
                    "dropping async task dependency back edge to break a cycle: %s depends on %s",
                    task_id,
                    dependency_id,
                )
                continue
            if dependency_id not in visited:
                visit(dependency_id)
        visiting.discard(task_id)
        visited.add(task_id)

    for task_id in sorted(dependencies_by_task):
        if task_id not in visited:
            visit(task_id)

    if not dropped:
        return dependencies_by_task
    return {
        task_id: [
            dependency_id
            for dependency_id in dependencies
            if (task_id, dependency_id) not in dropped
        ]
        for task_id, dependencies in dependencies_by_task.items()
    }


def _cyclic_components(tasks_by_id: dict[str, Task]) -> tuple[tuple[str, ...], ...]:
    """Return every non-trivial cyclic SCC and every self-loop.

    A DFS back-edge reports only the active slice of one traversal and misses
    members of overlapping cycles once an edge points into an already explored
    node. Tarjan's strongly-connected components capture exactly every cyclic
    member. Components and their members follow task insertion order so errors
    and attribution remain deterministic.
    """

    next_index = 0
    indexes: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    components: list[tuple[str, ...]] = []
    task_order = {task_id: index for index, task_id in enumerate(tasks_by_id)}

    def begin(task_id: str) -> None:
        nonlocal next_index
        indexes[task_id] = next_index
        lowlinks[task_id] = next_index
        next_index += 1
        stack.append(task_id)
        on_stack.add(task_id)

    for root_id in tasks_by_id:
        if root_id in indexes:
            continue
        begin(root_id)
        # Frames hold the node and the next dependency index to process. This
        # exactly mirrors recursive Tarjan while allowing chains far deeper
        # than Python's recursion limit.
        work: list[tuple[str, int]] = [(root_id, 0)]
        while work:
            task_id, dependency_index = work[-1]
            dependencies = tasks_by_id[task_id].dependencies
            if dependency_index < len(dependencies):
                dependency_id = dependencies[dependency_index]
                work[-1] = (task_id, dependency_index + 1)
                if dependency_id not in tasks_by_id:
                    continue
                if dependency_id not in indexes:
                    begin(dependency_id)
                    work.append((dependency_id, 0))
                elif dependency_id in on_stack:
                    lowlinks[task_id] = min(lowlinks[task_id], indexes[dependency_id])
                continue

            work.pop()
            if work:
                parent_id, _ = work[-1]
                lowlinks[parent_id] = min(lowlinks[parent_id], lowlinks[task_id])
            if lowlinks[task_id] != indexes[task_id]:
                continue
            component: list[str] = []
            while True:
                member = stack.pop()
                on_stack.remove(member)
                component.append(member)
                if member == task_id:
                    break
            component.sort(key=task_order.__getitem__)
            if len(component) > 1 or task_id in tasks_by_id[task_id].dependencies:
                components.append(tuple(component))
    components.sort(key=lambda component: task_order[component[0]])
    return tuple(components)


def _display_task_id(task_id: str) -> str:
    """Bound one worker-controlled task id while preserving normal messages."""

    if len(task_id) <= _MAX_RENDERED_TASK_ID_LENGTH:
        return task_id
    kept = _MAX_RENDERED_TASK_ID_LENGTH - len("...<truncated>")
    return f"{task_id[:kept]}...<truncated>"


def _format_task_path(path: tuple[Task, ...]) -> str:
    """Render a bounded sample of a dependency path."""

    if len(path) <= _MAX_RENDERED_PATH_TASK_IDS:
        shown = path
        omitted = 0
    else:
        leading = _MAX_RENDERED_PATH_TASK_IDS // 2
        trailing = _MAX_RENDERED_PATH_TASK_IDS - leading
        shown = (*path[:leading], *path[-trailing:])
        omitted = len(path) - len(shown)
    rendered = [_display_task_id(step.id) for step in shown]
    if omitted:
        rendered.insert(len(rendered) // 2, f"... ({omitted} tasks omitted) ...")
    return " -> ".join(rendered)


def _open_dependency_violations(
    task: Task,
    tasks_by_id: dict[str, Task],
    *,
    max_rendered_paths: int = _MAX_RENDERED_GRAPH_VIOLATIONS,
) -> tuple[tuple[tuple[Task, ...], ...], int, tuple[str, ...]]:
    """Return bounded readable paths, endpoint count, and all implicated ids.

    At most ``max_rendered_paths`` representative paths are materialized; the
    total reachable-open count lets the caller account for omitted violations.
    The final result remains the complete union of *every* node on *every* path
    from ``task`` to any open endpoint, including alternate-path intermediates.
    The traversal stops at an open node: that node already makes the complete
    root invalid, independently of the open node's own graph. Unknown edges are
    skipped so aggregation can continue after reporting them.
    """

    reached: set[str] = {task.id}
    parent: dict[str, str | None] = {task.id: None}
    reverse_edges: dict[str, list[str]] = {}
    open_ids: list[str] = []
    open_seen: set[str] = set()
    pending = [task.id]
    while pending:
        candidate_id = pending.pop()
        candidate = tasks_by_id[candidate_id]
        for dependency_id in candidate.dependencies:
            dependency = tasks_by_id.get(dependency_id)
            if dependency is None:
                continue
            reverse_edges.setdefault(dependency_id, []).append(candidate_id)
            if dependency_id not in parent:
                parent[dependency_id] = candidate_id
            if dependency.status is TaskStatus.OPEN:
                if dependency_id not in open_seen:
                    open_seen.add(dependency_id)
                    open_ids.append(dependency_id)
                continue
            if dependency_id not in reached:
                reached.add(dependency_id)
                pending.append(dependency_id)

    representative_paths: list[tuple[Task, ...]] = []
    for open_id in open_ids[:max_rendered_paths]:
        reversed_path = [open_id]
        cursor = parent[open_id]
        while cursor is not None:
            reversed_path.append(cursor)
            cursor = parent[cursor]
        representative_paths.append(
            tuple(tasks_by_id[task_id] for task_id in reversed(reversed_path))
        )

    productive = set(open_ids)
    pending_productive = list(open_ids)
    while pending_productive:
        candidate_id = pending_productive.pop()
        for depender_id in reverse_edges.get(candidate_id, ()):
            if depender_id not in productive:
                productive.add(depender_id)
                pending_productive.append(depender_id)
    implicated_ids = tuple(task_id for task_id in tasks_by_id if task_id in productive)
    return tuple(representative_paths), len(open_ids), implicated_ids


def _open_dependency_path(task: Task, tasks_by_id: dict[str, Task]) -> tuple[Task, ...] | None:
    """Dependency path from ``task`` to its first open dependency, if any.

    The resilient task-manager repair path needs only one endpoint for its log;
    strict validation uses :func:`_open_dependency_violations` so every endpoint
    and alternate path contributes attribution ids.
    """

    paths, _, _ = _open_dependency_violations(task, tasks_by_id, max_rendered_paths=1)
    return paths[0] if paths else None
