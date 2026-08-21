"""Provider-neutral message and content schemas."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Discriminator, Field

from tend._common.types import JsonObject, StrictModel, new_id


class ContentKind(StrEnum):
    """Provider-neutral content-part discriminator values."""

    TEXT = "text"
    COMPACTION_SUMMARY = "compaction_summary"


class MessageRole(StrEnum):
    """Provider-neutral message roles used by the shared model layer."""

    SYSTEM = "system"
    DEVELOPER = "developer"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class TextContent(StrictModel):
    """Plain text content visible as normal conversation text."""

    kind: Literal[ContentKind.TEXT] = ContentKind.TEXT
    text: str = Field(min_length=1)


class CompactionSummaryContent(StrictModel):
    """Summary content that stands in active context for compacted history."""

    kind: Literal[ContentKind.COMPACTION_SUMMARY] = ContentKind.COMPACTION_SUMMARY
    summary: str = Field(min_length=1)
    covered_message_ids: list[str] = Field(default_factory=list)


_CONTENT_DISCRIMINATOR: Discriminator = Discriminator("kind")

type ContentPart = Annotated[
    TextContent | CompactionSummaryContent,
    _CONTENT_DISCRIMINATOR,
]


def _empty_content_parts() -> list[ContentPart]:
    return []


class _BaseMessage(StrictModel):
    """Common provider-neutral message fields."""

    message_id: str = Field(default_factory=lambda: new_id("msg"), min_length=1)
    sequence: int | None = Field(default=None, ge=0)
    content: list[ContentPart] = Field(default_factory=_empty_content_parts)
    provider_metadata: JsonObject = Field(default_factory=dict)


class SystemMessage(_BaseMessage):
    """System-level instructions for providers that support them."""

    role: Literal[MessageRole.SYSTEM] = MessageRole.SYSTEM


class DeveloperMessage(_BaseMessage):
    """Developer-level instructions kept distinct from system instructions."""

    role: Literal[MessageRole.DEVELOPER] = MessageRole.DEVELOPER


class UserMessage(_BaseMessage):
    """User-authored conversation message."""

    role: Literal[MessageRole.USER] = MessageRole.USER


class AssistantMessage(_BaseMessage):
    """Assistant-authored conversation message."""

    role: Literal[MessageRole.ASSISTANT] = MessageRole.ASSISTANT


_MESSAGE_DISCRIMINATOR: Discriminator = Discriminator("role")

type Message = Annotated[
    SystemMessage | DeveloperMessage | UserMessage | AssistantMessage,
    _MESSAGE_DISCRIMINATOR,
]


__all__ = (
    "AssistantMessage",
    "CompactionSummaryContent",
    "ContentKind",
    "ContentPart",
    "DeveloperMessage",
    "Message",
    "MessageRole",
    "SystemMessage",
    "TextContent",
    "UserMessage",
)
