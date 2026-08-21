import pytest
from pydantic import TypeAdapter, ValidationError

from tend.llm.models import (
    AssistantMessage,
    CompactionSummaryContent,
    ContentKind,
    ContentPart,
    DeveloperMessage,
    Message,
    MessageRole,
    SystemMessage,
    TextContent,
    UserMessage,
)

type ContentValue = TextContent | CompactionSummaryContent
type MessageValue = SystemMessage | DeveloperMessage | UserMessage | AssistantMessage


def test_content_parts_validate_with_discriminators() -> None:
    adapter: TypeAdapter[ContentValue] = TypeAdapter(ContentPart)

    text = adapter.validate_python({"kind": "text", "text": "hello"})
    summary = adapter.validate_python(
        {
            "kind": "compaction_summary",
            "summary": "Earlier history summarized.",
            "covered_message_ids": ["msg_1", "msg_2"],
        }
    )

    assert text == TextContent(text="hello")
    assert summary == CompactionSummaryContent(
        summary="Earlier history summarized.",
        covered_message_ids=["msg_1", "msg_2"],
    )


def test_content_parts_reject_unknown_kind_and_unknown_fields() -> None:
    adapter: TypeAdapter[ContentValue] = TypeAdapter(ContentPart)

    with pytest.raises(ValidationError):
        adapter.validate_python({"kind": "image", "url": "file://example.png"})

    with pytest.raises(ValidationError):
        adapter.validate_python({"kind": "text", "text": "hello", "unexpected": True})


def test_message_roles_represent_provider_neutral_semantics() -> None:
    messages: list[MessageValue] = [
        SystemMessage(message_id="msg_system", content=[TextContent(text="system")]),
        DeveloperMessage(message_id="msg_developer", content=[TextContent(text="developer")]),
        UserMessage(message_id="msg_user", content=[TextContent(text="user")]),
        AssistantMessage(message_id="msg_assistant", content=[TextContent(text="assistant")]),
    ]

    assert [message.role for message in messages] == [
        MessageRole.SYSTEM,
        MessageRole.DEVELOPER,
        MessageRole.USER,
        MessageRole.ASSISTANT,
    ]


def test_message_json_round_trip_through_discriminated_union() -> None:
    adapter: TypeAdapter[MessageValue] = TypeAdapter(Message)
    original = UserMessage(
        message_id="msg_0000000000000001",
        sequence=3,
        content=[TextContent(text="hello")],
        provider_metadata={
            "provider": "scripted",
            "details": {"response_id": "resp_1", "stored": False},
        },
    )

    encoded = adapter.dump_json(original)
    restored = adapter.validate_json(encoded)

    assert restored == original
    assert restored.model_dump(mode="json") == {
        "message_id": "msg_0000000000000001",
        "sequence": 3,
        "content": [{"kind": "text", "text": "hello"}],
        "provider_metadata": {
            "provider": "scripted",
            "details": {"response_id": "resp_1", "stored": False},
        },
        "role": "user",
    }


def test_compaction_summary_message_serializes_as_dedicated_content() -> None:
    message = AssistantMessage(
        message_id="msg_summary",
        sequence=7,
        content=[
            CompactionSummaryContent(
                summary="Goal, progress, and next steps.",
                covered_message_ids=["msg_old_1", "msg_old_2"],
            )
        ],
    )

    assert message.model_dump(mode="json") == {
        "message_id": "msg_summary",
        "sequence": 7,
        "content": [
            {
                "kind": "compaction_summary",
                "summary": "Goal, progress, and next steps.",
                "covered_message_ids": ["msg_old_1", "msg_old_2"],
            }
        ],
        "provider_metadata": {},
        "role": "assistant",
    }


def test_messages_reject_unknown_role_and_unknown_fields() -> None:
    adapter: TypeAdapter[MessageValue] = TypeAdapter(Message)

    with pytest.raises(ValidationError):
        adapter.validate_python({"role": "tool", "content": []})

    with pytest.raises(ValidationError):
        UserMessage.model_validate(
            {
                "role": "user",
                "message_id": "msg_1",
                "content": [{"kind": "text", "text": "hello"}],
                "raw_provider_payload": {},
            }
        )


def test_content_validation_rejects_empty_text_and_empty_summary() -> None:
    with pytest.raises(ValidationError):
        TextContent.model_validate({"kind": ContentKind.TEXT, "text": ""})

    with pytest.raises(ValidationError):
        CompactionSummaryContent.model_validate(
            {"kind": ContentKind.COMPACTION_SUMMARY, "summary": ""}
        )
