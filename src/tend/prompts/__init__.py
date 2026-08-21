"""Bundled orchestration prompt files and resolution helpers.

Prompts are markdown files grouped by role and variant under this package:

    tend/prompts/worker/<variant>/{system,task,revision}.md
    tend/prompts/reviewer/<variant>/{system,task}.md

Orchestration configs select a variant by specifying a relative path
(e.g. ``prompts/worker/minimal``); :func:`resolve_prompts_dir` first looks
relative to the orchestration config file, then falls back to the
installed package.  Callers load the individual files with
:func:`load_prompt`.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Final

DEFAULT_WORKER_PROMPTS_DIR: Final[Path] = Path("prompts/worker/minimal")
DEFAULT_REVIEWER_PROMPTS_DIR: Final[Path] = Path("prompts/reviewer/minimal")


class PromptResolutionError(FileNotFoundError):
    """Raised when a configured prompts dir cannot be located."""


def builtin_prompts_root() -> Path:
    """Return the on-disk root of the bundled prompts package."""

    return Path(str(files("tend.prompts"))).resolve()


def resolve_prompts_dir(prompts_dir: Path, *, config_root: Path) -> Path:
    """Resolve a configured prompts dir against config root and the package.

    Absolute paths are returned unchanged after an existence check.  Relative
    paths are first resolved against ``config_root`` (the directory holding
    ``orchestration.yaml``); if that does not exist, the same relative path is
    resolved against the bundled :func:`builtin_prompts_root`.  Raises
    :class:`PromptResolutionError` when neither candidate exists.
    """

    if prompts_dir.is_absolute():
        if prompts_dir.is_dir():
            return prompts_dir
        raise PromptResolutionError(f"prompts dir not found: {prompts_dir}")

    config_candidate = (config_root / prompts_dir).resolve()
    if config_candidate.is_dir():
        return config_candidate

    package_root = builtin_prompts_root()
    package_candidate = (package_root / _strip_package_prefix(prompts_dir)).resolve()
    if package_candidate.is_dir():
        return package_candidate

    raise PromptResolutionError(
        f"prompts dir {prompts_dir!s} not found "
        f"(checked {config_candidate}, {package_candidate})"
    )


def load_prompt(prompts_dir: Path, name: str) -> str:
    """Load and return the text of ``<prompts_dir>/<name>.md``.

    ``prompts_dir`` must already be resolved (see :func:`resolve_prompts_dir`).
    Trailing whitespace is stripped so callers get the same shape that the
    previous Python string constants produced.
    """

    path = prompts_dir / f"{name}.md"
    if not path.is_file():
        raise PromptResolutionError(f"prompt file not found: {path}")
    return path.read_text(encoding="utf-8").rstrip()


def _strip_package_prefix(prompts_dir: Path) -> Path:
    """Drop a leading ``prompts/`` segment so callers can use either form.

    Both ``prompts/worker/minimal`` and ``worker/minimal`` resolve under the bundled
    package; the longer form matches what appears in orchestration.yaml.
    """

    parts = prompts_dir.parts
    if parts and parts[0] == "prompts":
        return Path(*parts[1:]) if len(parts) > 1 else Path()
    return prompts_dir


__all__ = (
    "DEFAULT_REVIEWER_PROMPTS_DIR",
    "DEFAULT_WORKER_PROMPTS_DIR",
    "PromptResolutionError",
    "builtin_prompts_root",
    "load_prompt",
    "resolve_prompts_dir",
)
