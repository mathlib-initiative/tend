"""Shared environment-variable name validation primitives."""

from __future__ import annotations

from re import Pattern, compile

ENV_NAME_PATTERN: Pattern[str] = compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def validate_env_name(name: str) -> None:
    """Raise ``ValueError`` if ``name`` is not a valid environment variable name."""

    if ENV_NAME_PATTERN.fullmatch(name) is None:
        raise ValueError(f"invalid environment variable name: {name}")


__all__ = ("ENV_NAME_PATTERN", "validate_env_name")
