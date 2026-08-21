"""Typed truncation helpers for model-visible tool output."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import Field, model_validator

from tend._common.types import StrictModel
from tend.llm.artifacts import ArtifactRef

_NonNegativeInt = Annotated[int, Field(ge=0)]


class TruncationPolicy(StrEnum):
    """Output-kind-aware truncation policies."""

    HEAD = "head"
    TAIL = "tail"


class TruncationInfo(StrictModel):
    """Structured metadata describing whether and how text was truncated."""

    truncated: bool
    policy: TruncationPolicy
    original_size_bytes: _NonNegativeInt | None = None
    original_line_count: _NonNegativeInt | None = None
    returned_size_bytes: _NonNegativeInt
    returned_line_count: _NonNegativeInt
    omitted_size_bytes: _NonNegativeInt | None = None
    omitted_line_count: _NonNegativeInt | None = None
    artifact: ArtifactRef | None = None

    @model_validator(mode="after")
    def _validate_truncated_metadata(self) -> TruncationInfo:
        if not self.truncated:
            if self.omitted_size_bytes not in (None, 0):
                raise ValueError("untruncated output must not omit bytes")
            if self.omitted_line_count not in (None, 0):
                raise ValueError("untruncated output must not omit lines")
            if self.artifact is not None:
                raise ValueError("untruncated output must not include an artifact reference")
        return self


class TruncationResult(StrictModel):
    """Text plus explicit truncation metadata returned by helper functions."""

    text: str
    info: TruncationInfo


class _RetainedText(StrictModel):
    text: str
    truncated: bool


def truncate_head(
    text: str,
    *,
    max_bytes: int | None = None,
    max_lines: int | None = None,
    artifact: ArtifactRef | None = None,
) -> TruncationResult:
    """Return ``text`` with deterministic head truncation applied.

    Limits apply to the retained payload portion before the explanatory notice is
    appended. File, list, and search tools should use this policy so the model
    sees the beginning of the output.
    """

    return _truncate(
        text,
        policy=TruncationPolicy.HEAD,
        max_bytes=max_bytes,
        max_lines=max_lines,
        artifact=artifact,
    )


def truncate_tail(
    text: str,
    *,
    max_bytes: int | None = None,
    max_lines: int | None = None,
    artifact: ArtifactRef | None = None,
) -> TruncationResult:
    """Return ``text`` with deterministic tail truncation applied.

    Limits apply to the retained payload portion before the explanatory notice is
    prepended. Bash and log-like tools should use this policy so the model sees
    the most recent output.
    """

    return _truncate(
        text,
        policy=TruncationPolicy.TAIL,
        max_bytes=max_bytes,
        max_lines=max_lines,
        artifact=artifact,
    )


def _truncate(
    text: str,
    *,
    policy: TruncationPolicy,
    max_bytes: int | None,
    max_lines: int | None,
    artifact: ArtifactRef | None,
) -> TruncationResult:
    _validate_limits(max_bytes=max_bytes, max_lines=max_lines)

    original_size_bytes = _size_bytes(text)
    original_line_count = _line_count(text)
    retained = _retain(text, policy=policy, max_bytes=max_bytes, max_lines=max_lines)

    if not retained.truncated:
        return TruncationResult(
            text=text,
            info=TruncationInfo(
                truncated=False,
                policy=policy,
                original_size_bytes=original_size_bytes,
                original_line_count=original_line_count,
                returned_size_bytes=original_size_bytes,
                returned_line_count=original_line_count,
            ),
        )

    retained_size_bytes = _size_bytes(retained.text)
    retained_line_count = _line_count(retained.text)
    notice = _format_notice(
        policy=policy,
        original_size_bytes=original_size_bytes,
        original_line_count=original_line_count,
        retained_size_bytes=retained_size_bytes,
        retained_line_count=retained_line_count,
        artifact=artifact,
    )
    visible_text = _attach_notice(retained.text, notice=notice, policy=policy)

    return TruncationResult(
        text=visible_text,
        info=TruncationInfo(
            truncated=True,
            policy=policy,
            original_size_bytes=original_size_bytes,
            original_line_count=original_line_count,
            returned_size_bytes=_size_bytes(visible_text),
            returned_line_count=_line_count(visible_text),
            omitted_size_bytes=max(original_size_bytes - retained_size_bytes, 0),
            omitted_line_count=max(original_line_count - retained_line_count, 0),
            artifact=artifact,
        ),
    )


def _retain(
    text: str,
    *,
    policy: TruncationPolicy,
    max_bytes: int | None,
    max_lines: int | None,
) -> _RetainedText:
    candidate = text
    truncated = False

    if max_lines is not None:
        line_limited = _retain_lines(candidate, max_lines=max_lines, policy=policy)
        candidate = line_limited.text
        truncated = truncated or line_limited.truncated

    if max_bytes is not None:
        byte_limited = _retain_bytes(candidate, max_bytes=max_bytes, policy=policy)
        candidate = byte_limited.text
        truncated = truncated or byte_limited.truncated

    return _RetainedText(text=candidate, truncated=truncated)


def _retain_lines(text: str, *, max_lines: int, policy: TruncationPolicy) -> _RetainedText:
    lines = text.splitlines(keepends=True)
    if len(lines) <= max_lines:
        return _RetainedText(text=text, truncated=False)
    if policy is TruncationPolicy.HEAD:
        return _RetainedText(text="".join(lines[:max_lines]), truncated=True)
    return _RetainedText(text="".join(lines[-max_lines:]), truncated=True)


def _retain_bytes(text: str, *, max_bytes: int, policy: TruncationPolicy) -> _RetainedText:
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return _RetainedText(text=text, truncated=False)
    if policy is TruncationPolicy.HEAD:
        retained = encoded[:max_bytes]
    else:
        retained = encoded[-max_bytes:]
    return _RetainedText(text=retained.decode("utf-8", errors="ignore"), truncated=True)


def _format_notice(
    *,
    policy: TruncationPolicy,
    original_size_bytes: int,
    original_line_count: int,
    retained_size_bytes: int,
    retained_line_count: int,
    artifact: ArtifactRef | None,
) -> str:
    direction = "first" if policy is TruncationPolicy.HEAD else "last"
    notice = (
        "[Output truncated: "
        f"showing {direction} {retained_line_count} of {original_line_count} lines "
        f"and {retained_size_bytes} of {original_size_bytes} bytes.]"
    )
    if artifact is None:
        return notice
    return f"{notice} Full output artifact: {artifact.artifact_id}."


def _attach_notice(text: str, *, notice: str, policy: TruncationPolicy) -> str:
    if not text:
        return notice
    if policy is TruncationPolicy.HEAD:
        separator = "" if text.endswith("\n") else "\n"
        return f"{text}{separator}{notice}"
    separator = "" if text.startswith("\n") else "\n"
    return f"{notice}{separator}{text}"


def _validate_limits(*, max_bytes: int | None, max_lines: int | None) -> None:
    if max_bytes is not None and max_bytes < 1:
        raise ValueError("max_bytes must be at least 1 when provided")
    if max_lines is not None and max_lines < 1:
        raise ValueError("max_lines must be at least 1 when provided")


def _size_bytes(text: str) -> int:
    return len(text.encode("utf-8"))


def _line_count(text: str) -> int:
    return len(text.splitlines())


__all__ = (
    "TruncationInfo",
    "TruncationPolicy",
    "TruncationResult",
    "truncate_head",
    "truncate_tail",
)
