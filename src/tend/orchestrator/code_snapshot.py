"""Freeze the orchestrator's tend checkout into ``<run-root>/code/``.

The orchestrator and its child ``tend-agent`` processes are launched via
``uv run --project <tend-checkout> ...``.  A live run can span hours and many
``resume`` invocations, during which an operator may keep editing that checkout
(we routinely launch experimental, not-yet-committed code).  Every fresh
``uv run`` would otherwise re-resolve against the mutating working tree, so an
in-flight run silently picks up half-finished edits.

The standard layout therefore gives every run directory a ``code/`` subdir: at
launch the working checkout is file-copied into it (including uncommitted edits,
honoring an ignore set so it does not copy build caches or ``.git``), and both
the orchestrator runner and the child ``tend-agent`` ``--project`` are pointed at
``<run-root>/code/``.  ``resume`` reuses the existing ``code/`` unchanged so a
resumed run stays pinned to its launch-time code, and hard-fails if it is gone.
"""

from __future__ import annotations

import fnmatch
import shutil
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Final

from tend._common.errors import FrameworkError

#: Directory name created under each run root to hold the frozen code copy.
CODE_SUBDIR_NAME: Final[str] = "code"

#: Default ignore patterns (``fnmatch`` style) applied while copying the
#: working tree.  Keeps the copy from pulling in git history, build caches, and
#: virtualenvs that would balloon a per-run copy to gigabytes.
DEFAULT_CODE_IGNORE: Final[tuple[str, ...]] = (
    ".git",
    ".lake",
    "__pycache__",
    ".venv",
    "*.pyc",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "node_modules",
)


class OrchestratorCodeSnapshotError(FrameworkError):
    """Raised when the orchestrator code checkout cannot be frozen or reused."""


def code_dir_for_root(run_root: Path) -> Path:
    """Return the ``code/`` directory path for a given run/exp root."""

    return run_root / CODE_SUBDIR_NAME


def code_snapshot_is_present(code_dir: Path) -> bool:
    """Return whether a usable code snapshot already exists at ``code_dir``."""

    return code_dir.is_dir() and any(code_dir.iterdir())


def validate_code_snapshot_location(*, source_checkout: Path, code_dir: Path) -> tuple[Path, Path]:
    """Resolve and validate that source and snapshot directories cannot nest."""

    source = source_checkout.resolve()
    resolved_code_dir = code_dir.resolve()
    if source == resolved_code_dir or source.is_relative_to(resolved_code_dir):
        raise OrchestratorCodeSnapshotError(
            f"orchestrator code checkout {source} must not live inside the snapshot "
            f"directory {resolved_code_dir}"
        )
    if resolved_code_dir.is_relative_to(source):
        raise OrchestratorCodeSnapshotError(
            f"code snapshot directory {resolved_code_dir} must not live inside the "
            f"orchestrator code checkout {source}"
        )
    return source, resolved_code_dir


def create_code_snapshot(
    *,
    source_checkout: Path,
    code_dir: Path,
    ignore: Iterable[str] = DEFAULT_CODE_IGNORE,
) -> Path:
    """File-copy ``source_checkout`` into ``code_dir`` and return ``code_dir``.

    Captures the working tree including uncommitted edits.  Names matching any
    ``ignore`` pattern (``fnmatch`` style, matched against each entry's basename)
    are skipped at every directory level.  Refuses to overwrite an existing
    snapshot so an in-flight run is never clobbered.
    """

    source = source_checkout.resolve()
    if not source.is_dir():
        raise OrchestratorCodeSnapshotError(
            f"orchestrator code checkout is not a directory: {source}"
        )

    validate_code_snapshot_location(source_checkout=source, code_dir=code_dir)

    if code_snapshot_is_present(code_dir):
        raise OrchestratorCodeSnapshotError(
            f"refusing to overwrite existing code snapshot at {code_dir}; "
            "resume reuses it instead of re-copying"
        )

    ignore_patterns = tuple(ignore)

    def _ignore(_dir: str, names: list[str]) -> set[str]:
        return {
            name
            for name in names
            if any(fnmatch.fnmatch(name, pattern) for pattern in ignore_patterns)
        }

    code_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copytree(
            source,
            code_dir,
            ignore=_ignore,
            symlinks=True,
            dirs_exist_ok=True,
        )
    except OSError as exc:
        raise OrchestratorCodeSnapshotError(
            f"could not copy orchestrator checkout {source} -> {code_dir}: {exc}"
        ) from exc
    return code_dir


def require_code_snapshot(code_dir: Path) -> Path:
    """Return ``code_dir`` if a usable snapshot exists, else hard-fail.

    Used on ``resume``: the resumed run is pinned to its launch-time code, so a
    missing ``code/`` is a fatal configuration error rather than an invitation
    to silently re-copy a now-different working tree.
    """

    if not code_snapshot_is_present(code_dir):
        raise OrchestratorCodeSnapshotError(
            f"no orchestrator code snapshot found at {code_dir}; "
            "the run directory must contain a code/ subdir created at launch"
        )
    return code_dir.resolve()


def repoint_uv_project_prefix(argv: Sequence[str], *, project: Path) -> tuple[str, ...]:
    """Rewrite a ``uv run --project <X> ...`` argv to target ``project``.

    Only the value following the first ``--project`` flag is replaced, so a
    command template's ``tend-agent`` invocation can be repointed at the frozen
    ``code/`` snapshot without disturbing the rest of its argv.  argv that does
    not begin with ``uv run`` (e.g. a bare ``tend-agent``) is returned unchanged.
    """

    items = list(argv)
    if items[:2] != ["uv", "run"]:
        return tuple(items)
    project_value = str(project.resolve())
    for index in range(2, len(items)):
        if items[index] == "--project":
            value_index = index + 1
            if value_index < len(items):
                items[value_index] = project_value
            return tuple(items)
    return tuple(items)


__all__ = (
    "CODE_SUBDIR_NAME",
    "DEFAULT_CODE_IGNORE",
    "OrchestratorCodeSnapshotError",
    "code_dir_for_root",
    "code_snapshot_is_present",
    "create_code_snapshot",
    "repoint_uv_project_prefix",
    "require_code_snapshot",
    "validate_code_snapshot_location",
)
