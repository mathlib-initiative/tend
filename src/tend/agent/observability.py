"""Detailed payload logging boundary.

The canonical persistence events remain minimal and resumable. This module makes
optional detailed payload capture explicit: small redacted payloads may be kept
inline by callers, while large redacted payloads are written as artifacts when
artifact storage is enabled.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import Field, JsonValue, model_validator

from tend._common.types import StrictModel
from tend.agent.config import RuntimeConfig
from tend.agent.persistence.artifacts import (
    ArtifactKind,
    ArtifactNameStrategy,
    ArtifactStore,
    dump_json_payload_bytes,
    json_compatible_payload,
    should_inline_size,
)
from tend.llm.artifacts import ArtifactRef
from tend.llm.models.requests import ModelRequest, ModelResponse
from tend.llm.redaction import Redactor, header_names_requiring_redaction

_NonNegativeInt = Annotated[int, Field(ge=0)]


class DetailedPayloadRecord(StrictModel):
    """Placement decision for one optional detailed payload."""

    kind: ArtifactKind
    size_bytes: _NonNegativeInt
    detailed_logging_enabled: bool
    inlined: bool = False
    inline_payload: JsonValue | None = None
    artifact: ArtifactRef | None = None
    omitted_reason: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _validate_placement(self) -> DetailedPayloadRecord:
        if self.artifact is not None and self.inlined:
            raise ValueError("payload cannot be both inline and artifact-backed")
        if self.artifact is not None and self.omitted_reason is not None:
            raise ValueError("stored payloads must not include an omission reason")
        if self.inlined and self.omitted_reason is not None:
            raise ValueError("inline payloads must not include an omission reason")
        if not self.detailed_logging_enabled and (self.inlined or self.artifact is not None):
            raise ValueError("disabled detailed logging must not store detailed payloads")
        return self


class DetailedPayloadRecorder:
    """Capture optional detailed payloads according to runtime config.

    The recorder never appends canonical events by itself. Turn/session code keeps
    writing minimal event fields such as request IDs and stop reasons regardless
    of detailed logging. Callers may use returned ``inline_payload`` or
    ``artifact`` fields to populate optional event payload details.
    """

    __slots__ = ("_redactor", "artifact_store", "runtime_config")

    runtime_config: RuntimeConfig
    artifact_store: ArtifactStore
    _redactor: Redactor

    def __init__(
        self,
        session_dir: str | Path,
        *,
        runtime_config: RuntimeConfig | None = None,
        redactor: Redactor | None = None,
        secret_values: tuple[str, ...] = (),
        sync_writes: bool = True,
    ) -> None:
        self.runtime_config = runtime_config or RuntimeConfig()
        self.artifact_store = ArtifactStore(
            session_dir,
            config=self.runtime_config.artifacts,
            sync_writes=sync_writes,
        )
        self._redactor = redactor or _redactor_from_runtime(
            self.runtime_config,
            secret_values=secret_values,
        )

    def capture_model_request(
        self,
        request: ModelRequest,
        *,
        event_id: str,
        name_strategy: ArtifactNameStrategy = ArtifactNameStrategy.EVENT_ID,
    ) -> DetailedPayloadRecord:
        """Capture an optional detailed model request payload."""

        return self.capture_json(
            ArtifactKind.MODEL_REQUEST,
            request,
            event_id=event_id,
            name_strategy=name_strategy,
        )

    def capture_model_response(
        self,
        response: ModelResponse,
        *,
        event_id: str,
        name_strategy: ArtifactNameStrategy = ArtifactNameStrategy.EVENT_ID,
    ) -> DetailedPayloadRecord:
        """Capture an optional detailed model response payload."""

        return self.capture_json(
            ArtifactKind.MODEL_RESPONSE,
            response,
            event_id=event_id,
            name_strategy=name_strategy,
        )

    def capture_tool_output(
        self,
        output: object,
        *,
        event_id: str,
        name_strategy: ArtifactNameStrategy = ArtifactNameStrategy.EVENT_ID,
    ) -> DetailedPayloadRecord:
        """Capture an optional detailed tool output payload."""

        if isinstance(output, str):
            return self.capture_text(
                ArtifactKind.TOOL_OUTPUT,
                output,
                event_id=event_id,
                name_strategy=name_strategy,
            )
        return self.capture_json(
            ArtifactKind.TOOL_OUTPUT,
            output,
            event_id=event_id,
            name_strategy=name_strategy,
        )

    def capture_compaction_payload(
        self,
        payload: object,
        *,
        event_id: str,
        name_strategy: ArtifactNameStrategy = ArtifactNameStrategy.EVENT_ID,
    ) -> DetailedPayloadRecord:
        """Capture optional detailed compaction input/output payloads."""

        return self.capture_json(
            ArtifactKind.COMPACTION,
            payload,
            event_id=event_id,
            name_strategy=name_strategy,
        )

    def capture_json(
        self,
        kind: ArtifactKind,
        payload: object,
        *,
        event_id: str,
        name_strategy: ArtifactNameStrategy = ArtifactNameStrategy.EVENT_ID,
    ) -> DetailedPayloadRecord:
        """Capture a JSON-like detailed payload as inline data or an artifact."""

        detailed_enabled = self._detailed_logging_enabled_for(kind)
        if not detailed_enabled:
            return DetailedPayloadRecord(
                kind=kind,
                size_bytes=0,
                detailed_logging_enabled=False,
                omitted_reason="detailed_logging_disabled",
            )
        json_payload = json_compatible_payload(payload)
        redacted_payload = json_compatible_payload(self._redactor.redact_payload(json_payload))
        data = dump_json_payload_bytes(redacted_payload)
        if should_inline_size(
            len(data),
            inline_threshold_bytes=self.runtime_config.artifacts.inline_threshold_bytes,
        ):
            return DetailedPayloadRecord(
                kind=kind,
                size_bytes=len(data),
                detailed_logging_enabled=True,
                inlined=True,
                inline_payload=redacted_payload,
            )
        if not self.runtime_config.artifacts.enabled:
            return DetailedPayloadRecord(
                kind=kind,
                size_bytes=len(data),
                detailed_logging_enabled=True,
                omitted_reason="artifact_storage_disabled_and_payload_exceeds_inline_threshold",
            )
        artifact = self.artifact_store.write_bytes(
            kind,
            data,
            event_id=event_id,
            name_strategy=name_strategy,
            suffix=".json",
            content_type="application/json",
            metadata={"redacted": True},
        )
        return DetailedPayloadRecord(
            kind=kind,
            size_bytes=len(data),
            detailed_logging_enabled=True,
            artifact=artifact,
        )

    def capture_text(
        self,
        kind: ArtifactKind,
        text: str,
        *,
        event_id: str,
        name_strategy: ArtifactNameStrategy = ArtifactNameStrategy.EVENT_ID,
    ) -> DetailedPayloadRecord:
        """Capture a detailed text payload as inline text or an artifact."""

        detailed_enabled = self._detailed_logging_enabled_for(kind)
        if not detailed_enabled:
            return DetailedPayloadRecord(
                kind=kind,
                size_bytes=0,
                detailed_logging_enabled=False,
                omitted_reason="detailed_logging_disabled",
            )
        redacted_text = self._redactor.redact_text(text)
        data = redacted_text.encode("utf-8")
        if should_inline_size(
            len(data),
            inline_threshold_bytes=self.runtime_config.artifacts.inline_threshold_bytes,
        ):
            return DetailedPayloadRecord(
                kind=kind,
                size_bytes=len(data),
                detailed_logging_enabled=True,
                inlined=True,
                inline_payload=redacted_text,
            )
        if not self.runtime_config.artifacts.enabled:
            return DetailedPayloadRecord(
                kind=kind,
                size_bytes=len(data),
                detailed_logging_enabled=True,
                omitted_reason="artifact_storage_disabled_and_payload_exceeds_inline_threshold",
            )
        artifact = self.artifact_store.write_bytes(
            kind,
            data,
            event_id=event_id,
            name_strategy=name_strategy,
            suffix=".txt",
            content_type="text/plain; charset=utf-8",
            metadata={"redacted": True},
        )
        return DetailedPayloadRecord(
            kind=kind,
            size_bytes=len(data),
            detailed_logging_enabled=True,
            artifact=artifact,
        )

    def _detailed_logging_enabled_for(self, kind: ArtifactKind) -> bool:
        logging = self.runtime_config.logging
        if not logging.detailed:
            return False
        if kind in {ArtifactKind.MODEL_REQUEST, ArtifactKind.MODEL_RESPONSE}:
            return logging.include_model_payloads
        if kind is ArtifactKind.TOOL_OUTPUT:
            return logging.include_tool_outputs
        if kind is ArtifactKind.COMPACTION:
            return True
        return False


def _redactor_from_runtime(
    runtime_config: RuntimeConfig,
    *,
    secret_values: tuple[str, ...],
) -> Redactor:
    redaction = runtime_config.redaction
    return Redactor(
        secret_values=secret_values if redaction.redact_secrets else (),
        secret_source_names=(
            runtime_config.secret_source_names() if redaction.redact_secrets else ()
        ),
        secret_header_names=(
            header_names_requiring_redaction(runtime_config.model.extra_headers)
            if redaction.redact_secrets
            else ()
        ),
        patterns=redaction.patterns,
        mildly_sensitive_urls=(
            [runtime_config.model.base_url]
            if redaction.redact_mildly_sensitive_urls and runtime_config.model.base_url is not None
            else []
        ),
    )


__all__ = (
    "DetailedPayloadRecord",
    "DetailedPayloadRecorder",
)
