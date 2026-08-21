"""Artifact reference schemas.

Filesystem artifact writing lives in :mod:`tend.agent.persistence.artifacts`; this
module keeps the serializable reference model independent from storage policy.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from tend._common.types import JsonObject, StrictModel

_NonNegativeInt = Annotated[int, Field(ge=0)]


def _empty_json_object() -> JsonObject:
    return {}


class ArtifactRef(StrictModel):
    """Serializable reference to a payload stored outside an inline result.

    The reference is deliberately storage-neutral: artifact-store code decides
    whether ``path`` is event-ID based, content-addressed, or otherwise laid out
    under a session directory. No filesystem policy is encoded here.
    """

    artifact_id: str = Field(min_length=1)
    kind: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    path: str | None = Field(default=None, min_length=1)
    size_bytes: _NonNegativeInt | None = None
    content_type: str | None = Field(default=None, min_length=1)
    sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    metadata: JsonObject = Field(default_factory=_empty_json_object)


__all__ = ("ArtifactRef",)
