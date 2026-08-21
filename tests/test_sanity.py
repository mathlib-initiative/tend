import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import tend


def test_package_imports() -> None:
    assert tend.__version__ == "0.1.0"


def test_console_scripts_use_tend_names() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    project = cast(Mapping[str, object], data["project"])
    scripts = cast(Mapping[str, str], project["scripts"])

    assert scripts == {
        "tend": "tend.orchestrator.cli:main",
        "tend-agent": "tend.agent.cli:main",
        "tend-control": "tend.orchestrator.control_cli:main",
        "tend-task": "tend.orchestrator.task_cli:main",
    }
