from __future__ import annotations

import os

import pytest

_LIVE_ENV_VARS: tuple[str, ...] = ("CF_AIG_TOKEN", "CF_AIG_URL")


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register explicit opt-in flags for tests that may call live providers."""

    parser.addoption(
        "--run-live",
        action="store_true",
        default=False,
        help="run opt-in live provider tests that may call external APIs",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip live tests unless explicitly requested and configured."""

    live_items = [item for item in items if "live" in item.keywords]
    if not live_items:
        return

    if not bool(config.getoption("--run-live", default=False)):
        skip_marker = pytest.mark.skip(reason="live provider tests require --run-live")
    else:
        missing = _missing_live_environment_variables()
        if not missing:
            return
        joined = ", ".join(missing)
        skip_marker = pytest.mark.skip(
            reason=f"live provider tests require environment variables: {joined}"
        )

    for item in live_items:
        item.add_marker(skip_marker)


def _missing_live_environment_variables() -> tuple[str, ...]:
    return tuple(name for name in _LIVE_ENV_VARS if not os.environ.get(name))
