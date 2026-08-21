import json

import pytest
from pydantic import ValidationError

from tend._common.errors import ConfigurationError
from tend.llm.redaction import (
    Redactor,
    header_names_requiring_redaction,
    redact_mildly_sensitive_url,
    redact_provider_headers,
    redact_resolved_headers,
    redact_text,
)
from tend.llm.secrets import (
    REDACTED_VALUE,
    EnvironmentSecretSource,
    HeaderValueSource,
    LiteralSecretSource,
    ProviderHeaderConfig,
    resolve_provider_header,
    resolve_provider_headers,
)


def test_environment_secret_source_serialization_does_not_include_value() -> None:
    source = EnvironmentSecretSource(env_var="FAKE_API_TOKEN")
    secret = source.resolve({"FAKE_API_TOKEN": "fake-secret-value"})

    assert secret.reveal_value() == "fake-secret-value"
    assert "fake-secret-value" not in repr(secret)
    assert "fake-secret-value" not in secret.model_dump_json()
    assert json.loads(source.model_dump_json()) == {"kind": "env", "env_var": "FAKE_API_TOKEN"}

    with pytest.raises(ConfigurationError, match="missing"):
        source.resolve({})

    with pytest.raises(ConfigurationError, match="empty"):
        source.resolve({"FAKE_API_TOKEN": ""})


def test_literal_secret_source_has_safe_repr_and_serialization() -> None:
    source = LiteralSecretSource.model_validate({"value": "fake-runtime-secret"})
    secret = source.resolve()

    assert secret.reveal_value() == "fake-runtime-secret"
    assert "fake-runtime-secret" not in repr(source)
    assert "fake-runtime-secret" not in source.model_dump_json()
    assert "fake-runtime-secret" not in secret.model_dump_json()



def test_provider_header_resolution_distinguishes_literal_secret_and_env_values() -> None:
    public_header = ProviderHeaderConfig(
        name="x-route",
        source=HeaderValueSource.LITERAL,
        value="public-route",
        secret=False,
    )
    resolved_public = resolve_provider_header(public_header, {})

    assert resolved_public.reveal_value() == "public-route"
    assert resolved_public.model_dump(mode="json")["value"] == "public-route"

    literal_secret = ProviderHeaderConfig(
        name="Authorization",
        source=HeaderValueSource.LITERAL,
        value="Bearer fake-runtime-secret",
        secret=True,
    )
    with pytest.raises(ConfigurationError, match="not allowed"):
        resolve_provider_header(literal_secret, {})

    assert "fake-runtime-secret" not in literal_secret.model_dump_json()

    resolved_literal_secret = resolve_provider_header(
        literal_secret,
        {},
        allow_literal_secrets=True,
    )
    assert resolved_literal_secret.reveal_value() == "Bearer fake-runtime-secret"
    assert "fake-runtime-secret" not in repr(resolved_literal_secret)
    assert "fake-runtime-secret" not in resolved_literal_secret.model_dump_json()
    assert resolved_literal_secret.model_dump(mode="json")["value"] == REDACTED_VALUE

    env_header = ProviderHeaderConfig(
        name="cf-aig-authorization",
        source=HeaderValueSource.ENV,
        env_var="FAKE_GATEWAY_TOKEN",
        secret=False,
    )
    resolved_env = resolve_provider_header(
        env_header,
        {"FAKE_GATEWAY_TOKEN": "Bearer fake-env-secret"},
    )

    assert resolved_env.secret is True
    assert resolved_env.source_name == "FAKE_GATEWAY_TOKEN"
    assert resolved_env.reveal_value() == "Bearer fake-env-secret"
    assert "fake-env-secret" not in resolved_env.model_dump_json()



def test_resolved_headers_wrapper_returns_raw_dict_only_by_explicit_method() -> None:
    configs = [
        ProviderHeaderConfig(
            name="x-public",
            source=HeaderValueSource.LITERAL,
            value="public",
            secret=False,
        ),
        ProviderHeaderConfig(
            name="Authorization",
            source=HeaderValueSource.ENV,
            env_var="FAKE_API_TOKEN",
        ),
    ]

    resolved = resolve_provider_headers(configs, {"FAKE_API_TOKEN": "Bearer fake-secret"})

    assert resolved.as_dict() == {
        "x-public": "public",
        "Authorization": "Bearer fake-secret",
    }
    assert "fake-secret" not in repr(resolved)
    assert "fake-secret" not in resolved.model_dump_json()



def test_header_redaction_uses_secret_and_env_sourced_provider_configs() -> None:
    configs = [
        ProviderHeaderConfig(
            name="Authorization",
            source=HeaderValueSource.LITERAL,
            value="Bearer fake-secret",
            secret=True,
        ),
        ProviderHeaderConfig(
            name="x-env-token",
            source=HeaderValueSource.ENV,
            env_var="FAKE_HEADER_TOKEN",
            secret=False,
        ),
        ProviderHeaderConfig(
            name="x-route",
            source=HeaderValueSource.LITERAL,
            value="route-a",
            secret=False,
        ),
    ]
    headers = {
        "Authorization": "Bearer fake-secret",
        "x-env-token": "fake-env-value",
        "x-route": "route-a",
    }

    redacted = redact_provider_headers(headers, configs)

    assert redacted == {
        "Authorization": REDACTED_VALUE,
        "x-env-token": REDACTED_VALUE,
        "x-route": "route-a",
    }
    assert header_names_requiring_redaction(configs) == ("Authorization", "x-env-token")



def test_resolved_header_redaction_never_returns_secret_values() -> None:
    configs = [
        ProviderHeaderConfig(
            name="Authorization",
            source=HeaderValueSource.ENV,
            env_var="FAKE_API_TOKEN",
        ),
        ProviderHeaderConfig(
            name="x-public",
            source=HeaderValueSource.LITERAL,
            value="public",
            secret=False,
        ),
    ]
    resolved = resolve_provider_headers(configs, {"FAKE_API_TOKEN": "Bearer fake-secret"})

    redacted = redact_resolved_headers(resolved.headers)

    assert redacted == {"Authorization": REDACTED_VALUE, "x-public": "public"}



def test_redaction_patterns_secret_source_assignments_and_mild_urls() -> None:
    url = "https://gateway.ai.cloudflare.com/v1/fake-account/fake-gateway"
    text = (
        "FAKE_API_TOKEN=fake-secret-value "
        f"url={url} "
        "ticket-123 "
        "Authorization: Bearer fake-secret-value"
    )

    redacted = redact_text(
        text,
        secret_values=["Bearer fake-secret-value"],
        secret_source_names=["FAKE_API_TOKEN"],
        patterns=[r"ticket-[0-9]+"],
        mildly_sensitive_urls=[url],
    )

    assert "fake-secret-value" not in redacted
    assert url not in redacted
    assert "fake-account" not in redacted
    assert "ticket-123" not in redacted
    assert "FAKE_API_TOKEN=[REDACTED]" in redacted
    assert redact_mildly_sensitive_url(url) == "https://gateway.ai.cloudflare.com/[REDACTED]"



def test_redaction_can_run_on_nested_event_payloads_and_error_messages() -> None:
    redactor = Redactor(
        secret_values=["fake-secret-value"],
        secret_source_names=["FAKE_API_TOKEN"],
        secret_header_names=["Authorization"],
        patterns=[r"run_[0-9]+"],
        mildly_sensitive_urls=["https://gateway.ai.cloudflare.com/v1/account/gateway"],
    )
    payload = {
        "event": "ModelRequestFailed",
        "env": {"FAKE_API_TOKEN": "fake-secret-value"},
        "headers": {"Authorization": "Bearer fake-secret-value", "x-public": "ok"},
        "messages": [
            "request run_123 failed at https://gateway.ai.cloudflare.com/v1/account/gateway"
        ],
    }

    redacted_payload = redactor.redact_payload(payload)
    redacted_error = redactor.redact_text(
        "error for FAKE_API_TOKEN=fake-secret-value in run_123"
    )

    assert "fake-secret-value" not in repr(redacted_payload)
    assert "account/gateway" not in repr(redacted_payload)
    assert "run_123" not in repr(redacted_payload)
    assert redacted_payload == {
        "event": "ModelRequestFailed",
        "env": {"FAKE_API_TOKEN": REDACTED_VALUE},
        "headers": {"Authorization": REDACTED_VALUE, "x-public": "ok"},
        "messages": ["request [REDACTED] failed at https://gateway.ai.cloudflare.com/[REDACTED]"],
    }
    assert redacted_error == "error for FAKE_API_TOKEN=[REDACTED] in [REDACTED]"



def test_provider_header_config_remains_strict() -> None:
    with pytest.raises(ValidationError, match="literal headers require value"):
        ProviderHeaderConfig(name="Authorization", source=HeaderValueSource.LITERAL)

    with pytest.raises(ValidationError, match="env headers require env_var"):
        ProviderHeaderConfig(
            name="Authorization",
            source=HeaderValueSource.ENV,
            value="not-allowed",
        )
