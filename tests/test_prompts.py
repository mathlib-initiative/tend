from __future__ import annotations

from pathlib import Path

import pytest

from tend.prompts import PromptResolutionError, load_prompt, resolve_prompts_dir


def test_resolve_prompts_dir_prefers_config_relative_directory(tmp_path: Path) -> None:
    config_root = tmp_path / "orch"
    prompt_dir = config_root / "prompts" / "worker" / "custom"
    prompt_dir.mkdir(parents=True)

    resolved = resolve_prompts_dir(
        Path("prompts/worker/custom"),
        config_root=config_root,
    )

    assert resolved == prompt_dir.resolve()


def test_resolve_prompts_dir_falls_back_to_bundled_prompts(tmp_path: Path) -> None:
    resolved = resolve_prompts_dir(Path("prompts/worker/minimal"), config_root=tmp_path)

    assert resolved.name == "minimal"
    assert resolved.parent.name == "worker"
    assert load_prompt(resolved, "system").startswith("You are a Tend orchestration worker.")


def test_resolve_prompts_dir_accepts_package_relative_path(tmp_path: Path) -> None:
    resolved = resolve_prompts_dir(Path("reviewer/minimal"), config_root=tmp_path)

    assert resolved.name == "minimal"
    assert resolved.parent.name == "reviewer"
    assert load_prompt(resolved, "system").startswith("You are a Tend orchestration reviewer.")


def test_resolve_prompts_dir_reports_checked_locations(tmp_path: Path) -> None:
    with pytest.raises(PromptResolutionError) as exc_info:
        resolve_prompts_dir(Path("prompts/worker/missing"), config_root=tmp_path)

    message = str(exc_info.value)
    assert "prompts dir prompts/worker/missing not found" in message
    assert str(tmp_path / "prompts/worker/missing") in message
    assert "worker/missing" in message


def test_load_prompt_strips_trailing_whitespace(tmp_path: Path) -> None:
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "task.md").write_text("custom task prompt\n\n", encoding="utf-8")

    assert load_prompt(prompts_dir, "task") == "custom task prompt"


def test_load_prompt_reports_missing_file(tmp_path: Path) -> None:
    with pytest.raises(PromptResolutionError, match="prompt file not found"):
        load_prompt(tmp_path, "revision")
