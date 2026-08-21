import pytest
from pydantic import Field, ValidationError

from tend._common.types import JsonObject, StrictModel


class StrictExample(StrictModel):
    name: str
    count: int


class MetadataExample(StrictModel):
    provider_metadata: JsonObject = Field(default_factory=dict)


def test_base_model_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        StrictExample.model_validate({"name": "demo", "count": 1, "unexpected": True})


def test_base_model_uses_strict_validation() -> None:
    with pytest.raises(ValidationError):
        StrictExample.model_validate({"name": "demo", "count": "1"})


def test_explicit_metadata_escape_hatch_allows_nested_json() -> None:
    model = MetadataExample.model_validate(
        {
            "provider_metadata": {
                "provider": "openai",
                "response_id": "resp_123",
                "usage": {"cached_tokens": 3, "tags": ["smoke", "scripted"]},
                "stored": False,
            }
        }
    )

    assert model.provider_metadata["provider"] == "openai"
    assert model.provider_metadata["usage"] == {
        "cached_tokens": 3,
        "tags": ["smoke", "scripted"],
    }


def test_metadata_escape_hatch_does_not_allow_unknown_top_level_fields() -> None:
    with pytest.raises(ValidationError):
        MetadataExample.model_validate({"provider_metadata": {}, "raw_provider": {}})
