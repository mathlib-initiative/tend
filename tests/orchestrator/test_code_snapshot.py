"""Tests for the orchestrator's ``<root>/code/`` snapshot port.

``tend run`` file-copies the operator's tend checkout into
``<root>/code/`` and rewrites each generated tend-agent launcher script's
``UV_PROJECT`` line to point there; resumed runs (auto-detected via saved
state) hard-fail when the snapshot is gone and otherwise repoint scripts at
the existing snapshot unchanged. See also tests/orchestrator/test_cli.py for
related CLI coverage.
"""

from __future__ import annotations

from collections.abc import Iterator
from io import StringIO
from pathlib import Path

import pytest

from tend.orchestrator.cli import AsyncOrchestratorCliExitCode, run_cli
from tend.orchestrator.code_snapshot import (
    DEFAULT_CODE_IGNORE,
    code_dir_for_root,
)
from tend.orchestrator.config import AsyncOrchestratorConfig
from tend.orchestrator.control_store import SQLiteAsyncOrchestratorStore
from tend.orchestrator.orchestrator import AsyncOrchestratorRunResult


def _make_fake_tend(root: Path) -> Path:
    """Create a minimal tend-shaped checkout for snapshot testing.

    Includes one file per default ignore pattern so the create_code_snapshot
    ignore set is exercised end-to-end.
    """

    root.mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text("[project]\nname = 'fake'\n", encoding="utf-8")
    (root / "README.md").write_text("hi\n", encoding="utf-8")
    (root / "src").mkdir()
    (root / "src" / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    # Should be ignored by DEFAULT_CODE_IGNORE.
    (root / ".git").mkdir()
    (root / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (root / ".venv").mkdir()
    (root / ".venv" / "pyvenv.cfg").write_text("home = /usr\n", encoding="utf-8")
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "module.cpython.pyc").write_text("\n", encoding="utf-8")
    return root


def _seed_empty_store(root: Path) -> None:
    SQLiteAsyncOrchestratorStore(root).initialize_state()


class _NoOpOrchestrator:
    """Stand-in orchestrator that does not need to actually run anything."""

    def __init__(
        self,
        config: AsyncOrchestratorConfig,
        *,
        check_resume_health: bool = False,
    ) -> None:
        self.config = config
        self.check_resume_health = check_resume_health

    async def run(self) -> AsyncOrchestratorRunResult:
        return AsyncOrchestratorRunResult(
            root=self.config.root,
            entrypoint=self.config.entrypoint,
        )


def _read_script(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _uv_project_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("UV_PROJECT="):
            return stripped
    raise AssertionError("UV_PROJECT= line not found in script")


@pytest.fixture
def fake_tend(tmp_path: Path) -> Iterator[Path]:
    yield _make_fake_tend(tmp_path / "tend")


async def _init_tend_root(
    tmp_path: Path,
    *,
    tend_project: Path | None,
) -> tuple[Path, Path]:
    """Initialize an async orchestration root with ``--agent tend``.

    Returns ``(root, entrypoint)``.
    """

    root = tmp_path / "orch"
    entrypoint = tmp_path / "repo"
    entrypoint.mkdir()
    args = [
        "init",
        "--root",
        str(root),
        "--entrypoint",
        str(entrypoint),
        "--agent",
        "tend",
        "--no-build-gate",
    ]
    if tend_project is not None:
        args.extend(["--tend-project", str(tend_project)])
    exit_code = await run_cli(args, stdout=StringIO())
    assert exit_code == int(AsyncOrchestratorCliExitCode.SUCCESS)
    return root, entrypoint


async def test_init_tend_project_bakes_uv_project_into_scripts(
    tmp_path: Path,
    fake_tend: Path,
) -> None:
    """``tend init --tend-project`` embeds the path in each launcher."""

    root, _ = await _init_tend_root(tmp_path, tend_project=fake_tend)
    expected = str(fake_tend.resolve())
    for script_name in ("worker-agent.sh", "reviewer-agent.sh"):
        script = root / "bin" / script_name
        text = _read_script(script)
        assert "BEGIN tend UV_PROJECT block" in text
        assert "END tend UV_PROJECT block" in text
        # shlex.quote of an absolute path with no special chars is identity.
        assert _uv_project_line(text) == f"UV_PROJECT={expected}"


async def test_init_without_tend_project_writes_empty_uv_project_block(
    tmp_path: Path,
) -> None:
    """When ``--tend-project`` is omitted the block is present but empty."""

    root, _ = await _init_tend_root(tmp_path, tend_project=None)
    for script_name in ("worker-agent.sh", "reviewer-agent.sh"):
        script = root / "bin" / script_name
        text = _read_script(script)
        assert "BEGIN tend UV_PROJECT block" in text
        assert _uv_project_line(text) == "UV_PROJECT=''"


async def test_init_pi_agent_rejects_tend_project(tmp_path: Path) -> None:
    """``--tend-project`` is meaningful only for the tend-agent scaffold."""

    root = tmp_path / "orch"
    entrypoint = tmp_path / "repo"
    entrypoint.mkdir()
    exit_code = await run_cli(
        [
            "init",
            "--root",
            str(root),
            "--entrypoint",
            str(entrypoint),
            "--agent",
            "pi",
            "--tend-project",
            str(tmp_path / "tend"),
            "--no-build-gate",
        ],
        stdout=StringIO(),
        stderr=StringIO(),
    )
    assert exit_code == int(AsyncOrchestratorCliExitCode.CONFIGURATION_OR_USAGE)


async def test_run_creates_code_snapshot_with_expected_ignore_patterns(
    tmp_path: Path,
    fake_tend: Path,
) -> None:
    """Launch copies the checkout into ``<root>/code/`` honoring DEFAULT_CODE_IGNORE."""

    root, _ = await _init_tend_root(tmp_path, tend_project=fake_tend)
    exit_code = await run_cli(
        ["run", "--root", str(root)],
        orchestrator_factory=_NoOpOrchestrator,
        stdout=StringIO(),
    )
    assert exit_code == int(AsyncOrchestratorCliExitCode.SUCCESS)
    code_dir = code_dir_for_root(root.resolve())
    assert code_dir.is_dir()
    # Source files are copied over.
    assert (code_dir / "pyproject.toml").is_file()
    assert (code_dir / "src" / "module.py").is_file()
    # Default ignore patterns are honored.
    for ignored in DEFAULT_CODE_IGNORE:
        # Only test the literal directory names (not the *.pyc-style globs).
        if "*" in ignored:
            continue
        assert not (code_dir / ignored).exists(), (
            f"snapshot must not include ignored entry {ignored}"
        )


async def test_run_repoints_agent_sh_scripts_to_code_dir(
    tmp_path: Path,
    fake_tend: Path,
) -> None:
    """After launch each launcher's UV_PROJECT line points at ``<root>/code/``."""

    root, _ = await _init_tend_root(tmp_path, tend_project=fake_tend)
    exit_code = await run_cli(
        ["run", "--root", str(root)],
        orchestrator_factory=_NoOpOrchestrator,
        stdout=StringIO(),
    )
    assert exit_code == int(AsyncOrchestratorCliExitCode.SUCCESS)
    expected = str(code_dir_for_root(root.resolve()).resolve())
    for script_name in ("worker-agent.sh", "reviewer-agent.sh"):
        script = root / "bin" / script_name
        line = _uv_project_line(_read_script(script))
        assert line == f"UV_PROJECT={expected}"


async def test_run_hard_fails_when_prewarmed_code_dir_is_inside_source_checkout(
    tmp_path: Path,
    fake_tend: Path,
) -> None:
    """A prewarmed snapshot is still rejected when nested under the source checkout."""

    root = fake_tend / "orch"
    entrypoint = tmp_path / "repo"
    entrypoint.mkdir()
    exit_code = await run_cli(
        [
            "init",
            "--root",
            str(root),
            "--entrypoint",
            str(entrypoint),
            "--agent",
            "tend",
            "--tend-project",
            str(fake_tend),
            "--no-build-gate",
        ],
        stdout=StringIO(),
    )
    assert exit_code == int(AsyncOrchestratorCliExitCode.SUCCESS)
    code_dir = code_dir_for_root(root.resolve())
    code_dir.mkdir(parents=True)
    (code_dir / "PRE_STAGED.txt").write_text("pre-staged contents\n", encoding="utf-8")

    class ShouldNotRunOrchestrator(_NoOpOrchestrator):
        async def run(self) -> AsyncOrchestratorRunResult:
            raise AssertionError("orchestrator must not run with nested code snapshot")

    stderr = StringIO()
    exit_code = await run_cli(
        ["run", "--root", str(root)],
        orchestrator_factory=ShouldNotRunOrchestrator,
        stdout=StringIO(),
        stderr=stderr,
    )

    assert exit_code == int(AsyncOrchestratorCliExitCode.CONFIGURATION_OR_USAGE)
    assert "code_snapshot_error" in stderr.getvalue()
    assert "must not live inside" in stderr.getvalue()


async def test_run_reuses_prepopulated_code_dir_on_launch(
    tmp_path: Path,
    fake_tend: Path,
) -> None:
    """A launch into a root that already has a populated ``code/`` reuses it.

    ``create_code_snapshot`` refuses to overwrite an existing populated
    snapshot, but launch also accepts a pre-staged ``<root>/code/`` (the
    sync-side run-live-once.sh wrapper relies on that). We drop a stray
    non-empty file into ``<root>/code/`` before launch, simulating a
    wrapper-prepared or aborted prior run, and assert that the launch
    does NOT silently re-copy the working tree on top of the populated
    ``code/``. The hard-refusal path against a half-populated snapshot
    is exercised in the shared module tests.
    """

    root, _ = await _init_tend_root(tmp_path, tend_project=fake_tend)
    code_dir = code_dir_for_root(root.resolve())
    code_dir.mkdir(parents=True)
    marker = code_dir / "PRE_STAGED.txt"
    marker.write_text("pre-staged contents\n", encoding="utf-8")

    exit_code = await run_cli(
        ["run", "--root", str(root)],
        orchestrator_factory=_NoOpOrchestrator,
        stdout=StringIO(),
    )

    assert exit_code == int(AsyncOrchestratorCliExitCode.SUCCESS)
    # Pre-staged contents are preserved, i.e. the launch did NOT overwrite.
    assert marker.is_file()
    assert marker.read_text(encoding="utf-8") == "pre-staged contents\n"
    # Scripts still get repointed to <root>/code/.
    expected = str(code_dir.resolve())
    for script_name in ("worker-agent.sh", "reviewer-agent.sh"):
        line = _uv_project_line(_read_script(root / "bin" / script_name))
        assert line == f"UV_PROJECT={expected}"


async def test_resume_hard_fails_when_snapshot_is_missing(
    tmp_path: Path,
    fake_tend: Path,
) -> None:
    """A resume that finds saved state but no ``<root>/code/`` is fatal."""

    root, _ = await _init_tend_root(tmp_path, tend_project=fake_tend)
    # First launch creates <root>/code/.
    exit_code = await run_cli(
        ["run", "--root", str(root)],
        orchestrator_factory=_NoOpOrchestrator,
        stdout=StringIO(),
    )
    assert exit_code == int(AsyncOrchestratorCliExitCode.SUCCESS)
    # Force durable SQLite state to exist so the next run auto-resumes.
    _seed_empty_store(root.resolve())
    # Now blow away the snapshot.
    import shutil

    shutil.rmtree(code_dir_for_root(root.resolve()))

    stderr = StringIO()
    exit_code = await run_cli(
        ["run", "--root", str(root)],
        orchestrator_factory=_NoOpOrchestrator,
        stdout=StringIO(),
        stderr=stderr,
    )
    assert exit_code == int(AsyncOrchestratorCliExitCode.CONFIGURATION_OR_USAGE)
    assert "code_snapshot_error" in stderr.getvalue()


async def test_resume_reuses_existing_snapshot_unchanged(
    tmp_path: Path,
    fake_tend: Path,
) -> None:
    """A resume with an intact ``<root>/code/`` reuses it; no second copy is made."""

    root, _ = await _init_tend_root(tmp_path, tend_project=fake_tend)
    exit_code = await run_cli(
        ["run", "--root", str(root)],
        orchestrator_factory=_NoOpOrchestrator,
        stdout=StringIO(),
    )
    assert exit_code == int(AsyncOrchestratorCliExitCode.SUCCESS)
    code_dir = code_dir_for_root(root.resolve())
    # Drop a marker into the snapshot that is not in the source. If the second
    # run silently re-copies the working tree the marker would be wiped.
    sentinel = code_dir / "RESUME_SENTINEL.txt"
    sentinel.write_text("sentinel\n", encoding="utf-8")

    # Force durable SQLite state so the next launch auto-resumes.
    _seed_empty_store(root.resolve())

    exit_code = await run_cli(
        ["run", "--root", str(root)],
        orchestrator_factory=_NoOpOrchestrator,
        stdout=StringIO(),
    )
    assert exit_code == int(AsyncOrchestratorCliExitCode.SUCCESS)
    assert sentinel.is_file()
    assert sentinel.read_text(encoding="utf-8") == "sentinel\n"
    expected = str(code_dir.resolve())
    for script_name in ("worker-agent.sh", "reviewer-agent.sh"):
        line = _uv_project_line(_read_script(root / "bin" / script_name))
        assert line == f"UV_PROJECT={expected}"


async def test_run_without_uv_project_block_skips_snapshot(
    tmp_path: Path,
) -> None:
    """An init without ``--tend-project`` writes empty blocks; no snapshot is taken."""

    root, _ = await _init_tend_root(tmp_path, tend_project=None)
    exit_code = await run_cli(
        ["run", "--root", str(root)],
        orchestrator_factory=_NoOpOrchestrator,
        stdout=StringIO(),
    )
    assert exit_code == int(AsyncOrchestratorCliExitCode.SUCCESS)
    assert not code_dir_for_root(root.resolve()).exists()
    for script_name in ("worker-agent.sh", "reviewer-agent.sh"):
        line = _uv_project_line(_read_script(root / "bin" / script_name))
        assert line == "UV_PROJECT=''"


async def test_run_pi_agent_skips_snapshot(tmp_path: Path) -> None:
    """The pi-agent scaffold has no UV_PROJECT block, so launch is a no-op."""

    root = tmp_path / "orch"
    entrypoint = tmp_path / "repo"
    entrypoint.mkdir()
    exit_code = await run_cli(
        [
            "init",
            "--root",
            str(root),
            "--entrypoint",
            str(entrypoint),
            "--agent",
            "pi",
            "--no-build-gate",
        ],
        stdout=StringIO(),
    )
    assert exit_code == int(AsyncOrchestratorCliExitCode.SUCCESS)
    exit_code = await run_cli(
        ["run", "--root", str(root)],
        orchestrator_factory=_NoOpOrchestrator,
        stdout=StringIO(),
    )
    assert exit_code == int(AsyncOrchestratorCliExitCode.SUCCESS)
    assert not code_dir_for_root(root.resolve()).exists()
