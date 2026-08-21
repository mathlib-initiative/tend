"""Filesystem-backed session artifact storage.

Artifacts are optional detailed payloads stored under a session directory. The
canonical event log remains sufficient for minimal replay/resume; this module
only writes explicit references that events or tool results may choose to carry.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Final

from pydantic import BaseModel, JsonValue, TypeAdapter

from tend._common.errors import PersistenceError
from tend._common.types import JsonObject
from tend.agent.config import ArtifactConfig
from tend.llm.artifacts import ArtifactRef
from tend.llm.redaction import Redactor

ARTIFACT_DIRECTORY_NAME: Final[str] = "artifacts"

_JSON_VALUE_ADAPTER: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)
_SAFE_FILENAME_STEM = re.compile(r"^[A-Za-z0-9_.-]+$")


class ArtifactKind(StrEnum):
    """Known v1 artifact kinds and event/result reference values."""

    MODEL_REQUEST = "model_request"
    MODEL_RESPONSE = "model_response"
    TOOL_OUTPUT = "tool_output"
    COMPACTION = "compaction"


class ArtifactNameStrategy(StrEnum):
    """Supported deterministic artifact naming strategies."""

    EVENT_ID = "event_id"
    CONTENT_HASH = "content_hash"


ARTIFACT_SUBDIRECTORIES: Final[Mapping[ArtifactKind, str]] = {
    ArtifactKind.MODEL_REQUEST: "model_requests",
    ArtifactKind.MODEL_RESPONSE: "model_responses",
    ArtifactKind.TOOL_OUTPUT: "tool_outputs",
    ArtifactKind.COMPACTION: "compactions",
}


class ArtifactStore:
    """Write/read explicit artifacts under ``<session>/artifacts``.

    References use paths relative to the artifact root, for example
    ``model_requests/evt_0001.json``. All path resolution checks that referenced
    files stay below the artifact root.
    """

    __slots__ = ("artifacts_dir", "config", "session_dir", "sync_writes")

    session_dir: Path
    artifacts_dir: Path
    config: ArtifactConfig
    sync_writes: bool

    def __init__(
        self,
        session_dir: str | Path,
        *,
        config: ArtifactConfig | None = None,
        sync_writes: bool = True,
    ) -> None:
        self.session_dir = Path(session_dir)
        self.config = config or ArtifactConfig()
        self.artifacts_dir = self.session_dir / self.config.directory_name
        self.sync_writes = sync_writes

    def ensure_layout(self) -> None:
        """Create the artifact root and all v1 subdirectories."""

        try:
            for subdirectory in ARTIFACT_SUBDIRECTORIES.values():
                (self.artifacts_dir / subdirectory).mkdir(parents=True, exist_ok=True)
            if self.sync_writes:
                _fsync_directory(self.artifacts_dir)
        except OSError as exc:
            raise PersistenceError(
                f"failed to create artifact directories under {self.artifacts_dir}: {exc}"
            ) from exc

    def write_json(
        self,
        kind: ArtifactKind | str,
        payload: object,
        *,
        event_id: str | None = None,
        name_strategy: ArtifactNameStrategy = ArtifactNameStrategy.EVENT_ID,
        redactor: Redactor | None = None,
        metadata: JsonObject | None = None,
    ) -> ArtifactRef:
        """Redact, serialize, and write one JSON artifact."""

        json_payload = json_compatible_payload(payload)
        if redactor is not None:
            json_payload = json_compatible_payload(redactor.redact_payload(json_payload))
        data = dump_json_payload_bytes(json_payload)
        return self.write_bytes(
            kind,
            data,
            event_id=event_id,
            name_strategy=name_strategy,
            suffix=".json",
            content_type="application/json",
            metadata=metadata,
        )

    def write_text(
        self,
        kind: ArtifactKind | str,
        text: str,
        *,
        event_id: str | None = None,
        name_strategy: ArtifactNameStrategy = ArtifactNameStrategy.EVENT_ID,
        redactor: Redactor | None = None,
        metadata: JsonObject | None = None,
    ) -> ArtifactRef:
        """Redact, UTF-8 encode, and write one text artifact."""

        redacted_text = redactor.redact_text(text) if redactor is not None else text
        return self.write_bytes(
            kind,
            redacted_text.encode("utf-8"),
            event_id=event_id,
            name_strategy=name_strategy,
            suffix=".txt",
            content_type="text/plain; charset=utf-8",
            metadata=metadata,
        )

    def write_bytes(
        self,
        kind: ArtifactKind | str,
        data: bytes,
        *,
        event_id: str | None = None,
        name_strategy: ArtifactNameStrategy = ArtifactNameStrategy.EVENT_ID,
        suffix: str = ".bin",
        content_type: str = "application/octet-stream",
        metadata: JsonObject | None = None,
    ) -> ArtifactRef:
        """Write bytes to a deterministic artifact path and return a reference."""

        artifact_kind = normalize_artifact_kind(kind)
        digest = hashlib.sha256(data).hexdigest()
        stem = _artifact_stem(
            event_id=event_id,
            digest=digest,
            name_strategy=name_strategy,
        )
        safe_suffix = _validate_suffix(suffix)
        relative_path = (
            PurePosixPath(ARTIFACT_SUBDIRECTORIES[artifact_kind]) / f"{stem}{safe_suffix}"
        )
        path = self._resolve_relative_path(relative_path)
        artifact_id = f"art_{artifact_kind.value}_{stem}"
        ref = ArtifactRef(
            artifact_id=artifact_id,
            kind=artifact_kind.value,
            path=relative_path.as_posix(),
            size_bytes=len(data),
            content_type=content_type,
            sha256=digest,
            metadata=metadata or {},
        )

        self.ensure_layout()
        if path.exists():
            existing = _read_file_bytes(path)
            existing_digest = hashlib.sha256(existing).hexdigest()
            if existing_digest != digest:
                raise PersistenceError(
                    f"artifact path already exists with different content: {path}"
                )
            return ref

        _atomic_write(path, data, sync_writes=self.sync_writes)
        return ref

    def read_bytes(self, ref: ArtifactRef) -> bytes:
        """Read bytes referenced by ``ref`` and verify checksum when present."""

        path = self.path_for_ref(ref)
        data = _read_file_bytes(path)
        if ref.sha256 is not None:
            digest = hashlib.sha256(data).hexdigest()
            if digest != ref.sha256:
                raise PersistenceError(f"artifact checksum mismatch for {path}")
        return data

    def read_text(self, ref: ArtifactRef) -> str:
        """Read a UTF-8 text artifact referenced by ``ref``."""

        try:
            return self.read_bytes(ref).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise PersistenceError(f"artifact {ref.artifact_id} is not valid UTF-8") from exc

    def read_json(self, ref: ArtifactRef) -> JsonValue:
        """Read a JSON artifact referenced by ``ref``."""

        data = self.read_bytes(ref)
        try:
            return _JSON_VALUE_ADAPTER.validate_json(data)
        except ValueError as exc:
            raise PersistenceError(f"artifact {ref.artifact_id} is not valid JSON") from exc

    def path_for_ref(self, ref: ArtifactRef) -> Path:
        """Return the absolute path for a stored reference after safety checks."""

        if ref.path is None:
            raise PersistenceError(f"artifact {ref.artifact_id} does not include a path")
        relative_path = PurePosixPath(ref.path)
        return self._resolve_relative_path(relative_path)

    def _resolve_relative_path(self, relative_path: PurePosixPath) -> Path:
        if relative_path.is_absolute() or "" in relative_path.parts:
            raise PersistenceError("artifact paths must be relative paths below the artifact root")
        if any(part in {".", ".."} for part in relative_path.parts):
            raise PersistenceError("artifact paths must not contain . or .. segments")
        if any("\\" in part or "\x00" in part for part in relative_path.parts):
            raise PersistenceError("artifact paths must not contain backslashes or NUL")
        root = self.artifacts_dir.resolve(strict=False)
        path = (self.artifacts_dir / Path(*relative_path.parts)).resolve(strict=False)
        if path != root and root not in path.parents:
            raise PersistenceError("artifact path resolves outside the artifact root")
        return path


def normalize_artifact_kind(kind: ArtifactKind | str) -> ArtifactKind:
    """Validate/normalize a known artifact kind."""

    if isinstance(kind, ArtifactKind):
        return kind
    try:
        return ArtifactKind(kind)
    except ValueError as exc:
        valid = ", ".join(item.value for item in ArtifactKind)
        raise ValueError(f"unknown artifact kind {kind!r}; expected one of: {valid}") from exc


def json_compatible_payload(payload: object) -> JsonValue:
    """Return a strict JSON-compatible value for Pydantic models or plain data."""

    if isinstance(payload, BaseModel):
        payload = payload.model_dump(mode="json")
    return _JSON_VALUE_ADAPTER.validate_python(payload)


def dump_json_payload_bytes(payload: JsonValue) -> bytes:
    """Serialize a JSON-compatible payload as compact UTF-8 bytes."""

    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def payload_size_bytes(payload: object) -> int:
    """Return the compact JSON byte size for a detailed payload candidate."""

    return len(dump_json_payload_bytes(json_compatible_payload(payload)))


def should_inline_size(size_bytes: int, *, inline_threshold_bytes: int) -> bool:
    """Return whether a payload of ``size_bytes`` may be stored inline."""

    if size_bytes < 0:
        raise ValueError("size_bytes must be non-negative")
    if inline_threshold_bytes < 0:
        raise ValueError("inline_threshold_bytes must be non-negative")
    return size_bytes <= inline_threshold_bytes


def _artifact_stem(
    *,
    event_id: str | None,
    digest: str,
    name_strategy: ArtifactNameStrategy,
) -> str:
    if name_strategy is ArtifactNameStrategy.EVENT_ID:
        if event_id is None:
            raise ValueError("event_id is required when naming artifacts by event ID")
        return _safe_filename_stem(event_id)
    if name_strategy is ArtifactNameStrategy.CONTENT_HASH:
        return digest
    raise ValueError(f"unsupported artifact name strategy: {name_strategy}")


def _safe_filename_stem(value: str) -> str:
    if not value or value in {".", ".."} or "\x00" in value:
        raise ValueError("artifact filename stem must be a non-empty safe path segment")
    if not _SAFE_FILENAME_STEM.fullmatch(value):
        raise ValueError("artifact filename stem contains unsupported characters")
    return value


def _validate_suffix(suffix: str) -> str:
    if not suffix.startswith(".") or "/" in suffix or "\\" in suffix or "\x00" in suffix:
        raise ValueError("artifact suffix must be a safe extension starting with '.'")
    if suffix in {".", ".."}:
        raise ValueError("artifact suffix must name a file extension")
    return suffix


def _read_file_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise PersistenceError(f"failed to read artifact {path}: {exc}") from exc


def _atomic_write(path: Path, data: bytes, *, sync_writes: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_fd = -1
    temp_path: Path | None = None
    try:
        temp_fd, temp_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        temp_path = Path(temp_name)
        with os.fdopen(temp_fd, "wb") as file:
            temp_fd = -1
            file.write(data)
            if sync_writes:
                file.flush()
                os.fsync(file.fileno())
        os.replace(temp_path, path)
        temp_path = None
        if sync_writes:
            _fsync_directory(path.parent)
    except OSError as exc:
        raise PersistenceError(f"failed to write artifact {path}: {exc}") from exc
    finally:
        if temp_fd >= 0:
            os.close(temp_fd)
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass


def _fsync_directory(directory: Path) -> None:
    if os.name == "nt":
        return
    try:
        directory_fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


__all__ = (
    "ARTIFACT_DIRECTORY_NAME",
    "ARTIFACT_SUBDIRECTORIES",
    "ArtifactKind",
    "ArtifactNameStrategy",
    "ArtifactStore",
    "dump_json_payload_bytes",
    "json_compatible_payload",
    "normalize_artifact_kind",
    "payload_size_bytes",
    "should_inline_size",
)
