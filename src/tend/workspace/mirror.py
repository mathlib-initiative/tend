"""Workspace mirroring for orchestration git worktrees.

The orchestrator creates worker worktrees through git, then mirrors the
entrypoint workspace into them so untracked/gitignored project-local state
(such as Lean ``.lake/`` caches) is available to agents.  The mirror walks the
filesystem directly rather than consulting git, so hidden and ignored files are
included by default.  The source root's top-level ``.git`` subtree is the one
hard exclusion so the destination worktree keeps its own git metadata; nested
package ``.git`` directories are mirrored unless configured otherwise.
"""

from __future__ import annotations

import fcntl
import os
import shutil
from enum import StrEnum
from os import PathLike
from pathlib import Path, PurePosixPath
from typing import Protocol

from pydantic import Field, field_validator

from tend._common.errors import FrameworkError
from tend._common.types import StrictModel

type PathInput = str | PathLike[str]

_FICLONE_IOCTL: int = 0x40049409
_ROOT_GIT_METADATA_NAME = ".git"


def _empty_string_list() -> list[str]:
    return []


class MirrorReflinkMode(StrEnum):
    """File-copy strategy for project workspace mirroring."""

    AUTO = "auto"
    REQUIRED = "required"
    NEVER = "never"


class MirrorExistingPathPolicy(StrEnum):
    """How to handle existing non-directory destination paths."""

    SKIP = "skip"
    OVERWRITE = "overwrite"
    ERROR = "error"


class WorkspaceMirrorConfig(StrictModel):
    """Configuration for mirroring entrypoint contents into a worktree.

    ``reflink_mode`` provides the Btrfs copy-on-write policy:

    - ``auto`` attempts a reflink clone first and falls back to ``shutil.copy2``;
    - ``required`` raises if a reflink clone cannot be created;
    - ``never`` uses regular file copies.

    The mirror source root's top-level ``.git`` subtree is always excluded,
    independent of the configurable exclusions. Additional ``exclude_names``
    exclude matching path components anywhere in the tree, while
    ``exclude_paths`` exclude relative subtrees from the source root.
    ``symlink_paths`` creates absolute symlinks to matching source files or
    directories instead of copying them.
    """

    reflink_mode: MirrorReflinkMode = MirrorReflinkMode.AUTO
    existing_path_policy: MirrorExistingPathPolicy = MirrorExistingPathPolicy.SKIP
    exclude_names: list[str] = Field(default_factory=_empty_string_list)
    exclude_paths: list[str] = Field(default_factory=_empty_string_list)
    symlink_paths: list[str] = Field(default_factory=_empty_string_list)

    @field_validator("exclude_names")
    @classmethod
    def _validate_exclude_names(cls, names: list[str]) -> list[str]:
        _validate_unique_strings(names, field_name="mirror exclude names")
        for name in names:
            _validate_path_segment(name, field_name="mirror exclude name")
        return names

    @field_validator("exclude_paths")
    @classmethod
    def _validate_exclude_paths(cls, paths: list[str]) -> list[str]:
        normalized: list[str] = []
        for path in paths:
            normalized.append(
                _normalize_relative_path(path, field_name="mirror exclude path").as_posix()
            )
        _validate_unique_strings(normalized, field_name="mirror exclude paths")
        return normalized

    @field_validator("symlink_paths")
    @classmethod
    def _validate_symlink_paths(cls, paths: list[str]) -> list[str]:
        normalized: list[str] = []
        for path in paths:
            normalized.append(
                _normalize_relative_path(path, field_name="mirror symlink path").as_posix()
            )
        _validate_unique_strings(normalized, field_name="mirror symlink paths")
        return normalized


class MirrorResult(StrictModel):
    """Summary of one workspace mirror operation."""

    source_root: Path
    destination_root: Path
    directories_created: tuple[str, ...] = ()
    files_copied: tuple[str, ...] = ()
    symlinks_copied: tuple[str, ...] = ()
    skipped_paths: tuple[str, ...] = ()

    @field_validator("source_root", "destination_root")
    @classmethod
    def _validate_roots(cls, path: Path) -> Path:
        _validate_path(path, field_name="mirror result path")
        return path


class MirrorFileCopier(Protocol):
    """Boundary for copying one regular file into the mirrored worktree."""

    def copy_file(
        self,
        source: Path,
        destination: Path,
        *,
        reflink_mode: MirrorReflinkMode,
    ) -> None:
        """Copy ``source`` to ``destination`` using ``reflink_mode``."""
        ...


class WorkspaceMirrorError(FrameworkError):
    """Base error for workspace mirroring failures."""


class MirrorConflictError(WorkspaceMirrorError):
    """Raised when the destination cannot be reused under the configured policy."""


class MirrorCopyError(WorkspaceMirrorError):
    """Raised when a source path cannot be copied."""


class LocalMirrorFileCopier:
    """Local file copier with Btrfs/Linux reflink support and clear fallback policy."""

    __slots__ = ()

    def copy_file(
        self,
        source: Path,
        destination: Path,
        *,
        reflink_mode: MirrorReflinkMode,
    ) -> None:
        """Copy one regular file, honoring the configured reflink policy."""

        source_path = _to_path(source, field_name="mirror source file")
        destination_path = _to_path(destination, field_name="mirror destination file")
        if not source_path.is_file() or source_path.is_symlink():
            raise MirrorCopyError(f"mirror source is not a regular file: {source_path}")
        destination_path.parent.mkdir(parents=True, exist_ok=True)

        if reflink_mode == MirrorReflinkMode.NEVER:
            _copy_with_shutil(source_path, destination_path)
            return

        try:
            _copy_with_reflink(source_path, destination_path)
        except OSError as exc:
            _unlink_file_if_present(destination_path)
            if reflink_mode == MirrorReflinkMode.REQUIRED:
                raise MirrorCopyError(
                    "reflink copy was required but failed for "
                    f"{source_path} -> {destination_path}: {exc}"
                ) from exc
            _copy_with_shutil(source_path, destination_path)


class WorkspaceMirror:
    """Mirror an entrypoint workspace into a worker git worktree."""

    __slots__ = ("config", "copier")

    config: WorkspaceMirrorConfig
    copier: MirrorFileCopier

    def __init__(
        self,
        *,
        config: WorkspaceMirrorConfig | None = None,
        copier: MirrorFileCopier | None = None,
    ) -> None:
        self.config = WorkspaceMirrorConfig() if config is None else config
        self.copier = LocalMirrorFileCopier() if copier is None else copier

    def mirror(self, source: PathInput, destination: PathInput) -> MirrorResult:
        """Copy source workspace contents into ``destination``.

        The destination root is created if needed.  Existing directories are
        reused so the git-created worktree skeleton remains intact; existing
        files/symlinks follow ``existing_path_policy``.  The source root's
        top-level ``.git`` subtree is always skipped and never overwrites
        destination git metadata.
        """

        source_root = _to_path(source, field_name="mirror source root")
        destination_root = _to_path(destination, field_name="mirror destination root")
        _validate_mirror_roots(source_root, destination_root)
        if not destination_root.exists():
            destination_root.mkdir(parents=True)
        elif not destination_root.is_dir() or destination_root.is_symlink():
            raise MirrorConflictError(
                f"mirror destination root is not a directory: {destination_root}"
            )

        directories_created: list[str] = []
        files_copied: list[str] = []
        symlinks_copied: list[str] = []
        skipped_paths: list[str] = []

        def mirror_directory(
            source_directory: Path,
            relative_directory: PurePosixPath | None,
        ) -> None:
            for child in sorted(source_directory.iterdir(), key=lambda path: path.name):
                relative_path = _child_relative_path(relative_directory, child.name)
                relative_text = relative_path.as_posix()
                if _is_excluded(relative_path, self.config):
                    skipped_paths.append(relative_text)
                    continue

                destination_path = _join_relative(destination_root, relative_path)
                if child.is_symlink():
                    if not _prepare_leaf_destination(
                        destination_path,
                        relative_text=relative_text,
                        policy=self.config.existing_path_policy,
                        skipped_paths=skipped_paths,
                    ):
                        continue
                    target = os.readlink(child)
                    destination_path.symlink_to(target)
                    symlinks_copied.append(relative_text)
                elif child.is_dir():
                    if _should_symlink(relative_path, self.config):
                        if not _symlink_configured_path(
                            child,
                            destination_path,
                            relative_text=relative_text,
                            policy=self.config.existing_path_policy,
                            skipped_paths=skipped_paths,
                        ):
                            continue
                        symlinks_copied.append(relative_text)
                    elif _ensure_destination_directory(
                        destination_path,
                        relative_text=relative_text,
                        policy=self.config.existing_path_policy,
                        directories_created=directories_created,
                        skipped_paths=skipped_paths,
                    ):
                        mirror_directory(child, relative_path)
                elif child.is_file():
                    if _should_symlink(relative_path, self.config):
                        if not _symlink_configured_path(
                            child,
                            destination_path,
                            relative_text=relative_text,
                            policy=self.config.existing_path_policy,
                            skipped_paths=skipped_paths,
                        ):
                            continue
                        symlinks_copied.append(relative_text)
                        continue
                    if not _prepare_leaf_destination(
                        destination_path,
                        relative_text=relative_text,
                        policy=self.config.existing_path_policy,
                        skipped_paths=skipped_paths,
                    ):
                        continue
                    destination_path.parent.mkdir(parents=True, exist_ok=True)
                    self.copier.copy_file(
                        child,
                        destination_path,
                        reflink_mode=self.config.reflink_mode,
                    )
                    files_copied.append(relative_text)
                else:
                    raise MirrorCopyError(f"unsupported mirror source path type: {child}")

        mirror_directory(source_root, None)
        return MirrorResult(
            source_root=source_root,
            destination_root=destination_root,
            directories_created=tuple(directories_created),
            files_copied=tuple(files_copied),
            symlinks_copied=tuple(symlinks_copied),
            skipped_paths=tuple(skipped_paths),
        )


def mirror_workspace(
    source: PathInput,
    destination: PathInput,
    *,
    config: WorkspaceMirrorConfig | None = None,
    copier: MirrorFileCopier | None = None,
) -> MirrorResult:
    """Convenience wrapper around :class:`WorkspaceMirror`."""

    return WorkspaceMirror(config=config, copier=copier).mirror(source, destination)


def should_mirror_path(
    relative_path: str,
    *,
    config: WorkspaceMirrorConfig | None = None,
) -> bool:
    """Return whether ``relative_path`` would be included by mirror rules."""

    mirror_config = WorkspaceMirrorConfig() if config is None else config
    normalized = _normalize_relative_path(relative_path, field_name="mirror relative path")
    return not _is_excluded(normalized, mirror_config)


def _validate_mirror_roots(source_root: Path, destination_root: Path) -> None:
    if not source_root.exists():
        raise WorkspaceMirrorError(f"mirror source root does not exist: {source_root}")
    if not source_root.is_dir() or source_root.is_symlink():
        raise WorkspaceMirrorError(f"mirror source root is not a directory: {source_root}")
    source_resolved = source_root.resolve(strict=True)
    destination_resolved = destination_root.resolve(strict=False)
    if source_resolved == destination_resolved:
        raise WorkspaceMirrorError("mirror source and destination roots must be different")
    try:
        destination_resolved.relative_to(source_resolved)
    except ValueError:
        return
    raise WorkspaceMirrorError(
        "mirror destination root must not be inside the source root: "
        f"{destination_root} is inside {source_root}"
    )


def _ensure_destination_directory(
    destination_path: Path,
    *,
    relative_text: str,
    policy: MirrorExistingPathPolicy,
    directories_created: list[str],
    skipped_paths: list[str],
) -> bool:
    if (
        destination_path.exists()
        and destination_path.is_dir()
        and not destination_path.is_symlink()
    ):
        return True
    if _path_exists(destination_path):
        if policy == MirrorExistingPathPolicy.ERROR:
            raise MirrorConflictError(f"mirror destination already exists: {destination_path}")
        if policy == MirrorExistingPathPolicy.SKIP:
            skipped_paths.append(relative_text)
            return False
        _remove_existing_path(destination_path)
    destination_path.mkdir(parents=True, exist_ok=True)
    directories_created.append(relative_text)
    return True


def _prepare_leaf_destination(
    destination_path: Path,
    *,
    relative_text: str,
    policy: MirrorExistingPathPolicy,
    skipped_paths: list[str],
) -> bool:
    if not _path_exists(destination_path):
        return True
    if policy == MirrorExistingPathPolicy.ERROR:
        raise MirrorConflictError(f"mirror destination already exists: {destination_path}")
    if policy == MirrorExistingPathPolicy.SKIP:
        skipped_paths.append(relative_text)
        return False
    _remove_existing_path(destination_path)
    return True


def _copy_with_reflink(source: Path, destination: Path) -> None:
    with source.open("rb") as source_file, destination.open("xb") as destination_file:
        fcntl.ioctl(destination_file.fileno(), _FICLONE_IOCTL, source_file.fileno())
    shutil.copystat(source, destination, follow_symlinks=False)


def _copy_with_shutil(source: Path, destination: Path) -> None:
    try:
        shutil.copy2(source, destination, follow_symlinks=False)
    except OSError as exc:
        raise MirrorCopyError(
            f"regular file copy failed for {source} -> {destination}: {exc}"
        ) from exc


def _unlink_file_if_present(path: Path) -> None:
    if path.exists() or path.is_symlink():
        path.unlink()


def _remove_existing_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def _symlink_configured_path(
    source_path: Path,
    destination_path: Path,
    *,
    relative_text: str,
    policy: MirrorExistingPathPolicy,
    skipped_paths: list[str],
) -> bool:
    if not _prepare_leaf_destination(
        destination_path,
        relative_text=relative_text,
        policy=policy,
        skipped_paths=skipped_paths,
    ):
        return False
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.symlink_to(source_path.resolve())
    return True


def _should_symlink(relative_path: PurePosixPath, config: WorkspaceMirrorConfig) -> bool:
    for symlink_path_text in config.symlink_paths:
        if relative_path == PurePosixPath(symlink_path_text):
            return True
    return False


def _is_excluded(relative_path: PurePosixPath, config: WorkspaceMirrorConfig) -> bool:
    parts = relative_path.parts
    if parts and parts[0] == _ROOT_GIT_METADATA_NAME:
        return True
    excluded_names = frozenset(config.exclude_names)
    if any(part in excluded_names for part in parts):
        return True
    for exclude_path_text in config.exclude_paths:
        exclude_path = PurePosixPath(exclude_path_text)
        if _is_relative_path_prefix(exclude_path, relative_path):
            return True
    return False


def _is_relative_path_prefix(prefix: PurePosixPath, path: PurePosixPath) -> bool:
    prefix_parts = prefix.parts
    return path.parts[: len(prefix_parts)] == prefix_parts


def _child_relative_path(
    relative_directory: PurePosixPath | None,
    name: str,
) -> PurePosixPath:
    if relative_directory is None:
        return PurePosixPath(name)
    return relative_directory / name


def _join_relative(root: Path, relative_path: PurePosixPath) -> Path:
    return root.joinpath(*relative_path.parts)


def _path_exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _to_path(value: PathInput, *, field_name: str) -> Path:
    if isinstance(value, str) and not value:
        raise ValueError(f"{field_name} path must be non-empty")
    path = Path(value)
    _validate_path(path, field_name=field_name)
    return path


def _validate_path(path: Path, *, field_name: str) -> None:
    if "\x00" in str(path):
        raise ValueError(f"{field_name} must not contain NUL")


def _normalize_relative_path(value: str, *, field_name: str) -> PurePosixPath:
    _validate_non_empty_text(value, field_name=field_name)
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts:
        raise ValueError(f"{field_name} must be a relative path below the mirror root")
    if any(part in {".", ".."} or not part for part in path.parts):
        raise ValueError(f"{field_name} must not contain '.' or '..' path components")
    return path


def _validate_path_segment(value: str, *, field_name: str) -> None:
    _validate_non_empty_text(value, field_name=field_name)
    if value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"{field_name} must be a single relative path segment")


def _validate_non_empty_text(value: str, *, field_name: str) -> None:
    if not value or "\x00" in value:
        raise ValueError(f"{field_name} must be non-empty and must not contain NUL")


def _validate_unique_strings(values: list[str], *, field_name: str) -> None:
    if any(not value for value in values):
        raise ValueError(f"{field_name} must not contain empty strings")
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} must be unique")


__all__ = (
    "LocalMirrorFileCopier",
    "MirrorConflictError",
    "MirrorCopyError",
    "MirrorExistingPathPolicy",
    "MirrorFileCopier",
    "MirrorReflinkMode",
    "MirrorResult",
    "PathInput",
    "WorkspaceMirror",
    "WorkspaceMirrorConfig",
    "WorkspaceMirrorError",
    "mirror_workspace",
    "should_mirror_path",
)
