"""Human-authored JSON/YAML configuration file helpers."""

from __future__ import annotations

import json
import math
from enum import StrEnum
from pathlib import Path
from re import Pattern, compile
from typing import Any, cast

import yaml
from pydantic import BaseModel
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode, ScalarNode

from tend._common.errors import ConfigurationError

_BOOL_RESOLVER: Pattern[str] = compile(r"^(?:true|True|TRUE|false|False|FALSE)$")
_JSON_INT_RESOLVER: Pattern[str] = compile(r"^-?(?:0|[1-9][0-9]*)$")
_JSON_FLOAT_RESOLVER: Pattern[str] = compile(
    r"^-?(?:(?:0|[1-9][0-9]*)\.[0-9]+(?:[eE][-+]?[0-9]+)?|"
    r"(?:0|[1-9][0-9]*)[eE][-+]?[0-9]+)$"
)
_YAML_NULL_STRINGS: frozenset[str] = frozenset(("", "~", "null", "Null", "NULL"))
_YAML_BOOL_TAG = "tag:yaml.org,2002:bool"
_YAML_FLOAT_TAG = "tag:yaml.org,2002:float"
_YAML_INT_TAG = "tag:yaml.org,2002:int"
_YAML_MAPPING_TAG = "tag:yaml.org,2002:map"
_YAML_STRING_TAG = "tag:yaml.org,2002:str"
_YAML_TIMESTAMP_TAG = "tag:yaml.org,2002:timestamp"

type ConfigData = None | bool | int | float | str | list[ConfigData] | dict[str, ConfigData]


class ConfigFormat(StrEnum):
    """Supported human-authored config file formats."""

    JSON = "json"
    YAML = "yaml"


class ConfigFileError(ConfigurationError):
    """A config file could not be read or parsed before schema validation."""

    __slots__ = ("path", "kind")

    path: Path
    kind: str

    def __init__(self, message: str, *, path: Path, kind: str) -> None:
        super().__init__(message)
        self.path = path
        self.kind = kind


class _ConfigYamlLoader(yaml.SafeLoader):
    """Safe YAML loader constrained to JSON-like config data."""


class _ConfigYamlDumper(yaml.SafeDumper):
    """Human-readable YAML dumper for generated config files."""

    def ignore_aliases(self, data: object) -> bool:
        del data
        return True

    def increase_indent(self, flow: bool = False, indentless: bool = False) -> None:
        del indentless
        return super().increase_indent(flow=flow, indentless=False)


def config_format_from_path(path: str | Path) -> ConfigFormat:
    """Return the config format implied by a filename suffix."""

    suffix = Path(path).suffix.lower()
    if suffix == ".json":
        return ConfigFormat.JSON
    if suffix in (".yaml", ".yml"):
        return ConfigFormat.YAML
    raise ValueError(
        f"unsupported config file extension {suffix or '<none>'!r}; "
        "use .yaml, .yml, or .json"
    )


def read_config_model[ModelT: BaseModel](
    path: str | Path,
    model_type: type[ModelT],
    *,
    kind: str,
) -> ModelT:
    """Read a JSON/YAML config file and validate it in Pydantic JSON mode."""

    config_path = Path(path)
    try:
        fmt = config_format_from_path(config_path)
    except ValueError as exc:
        raise ConfigFileError(str(exc), path=config_path, kind=kind) from exc

    text = _read_config_text(config_path, kind=kind)
    validation_context = {"config_path": config_path, "config_root": config_path.parent}
    if fmt is ConfigFormat.JSON:
        return model_type.model_validate_json(text, context=validation_context)

    data = read_yaml_config_data(text, path=config_path, kind=kind)
    validation_json = _json_text_for_validation(data, path=config_path, kind=kind)
    return model_type.model_validate_json(validation_json, context=validation_context)


def read_yaml_config_data(text: str, *, path: str | Path | None = None, kind: str) -> ConfigData:
    """Parse YAML text into JSON-compatible data accepted by config schemas."""

    display_path = Path(path) if path is not None else Path("<memory>")
    try:
        loaded = yaml.load(text, Loader=_ConfigYamlLoader)  # noqa: S506 - constrained SafeLoader
    except yaml.YAMLError as exc:
        raise ConfigFileError(
            f"invalid YAML {kind} {display_path}: {exc}",
            path=display_path,
            kind=kind,
        ) from exc
    try:
        return _to_config_data(loaded, path="<root>", seen=set())
    except ConfigFileError as exc:
        raise ConfigFileError(str(exc), path=display_path, kind=kind) from exc


def dump_config_model_yaml(model: BaseModel) -> str:
    """Serialize a Pydantic config model as generated YAML."""

    return dump_yaml_data(model.model_dump(mode="json", exclude_none=True))


def dump_yaml_data(data: object) -> str:
    """Serialize JSON-compatible config data as generated YAML."""

    normalized = _to_config_data(data, path="<root>", seen=set())
    return yaml.dump(
        normalized,
        Dumper=_ConfigYamlDumper,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
        width=100,
    )


def _read_config_text(path: Path, *, kind: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigFileError(
            f"could not read {kind} {path}: {exc.strerror or exc}",
            path=path,
            kind=kind,
        ) from exc


def _json_text_for_validation(data: ConfigData, *, path: Path, kind: str) -> str:
    try:
        return json.dumps(data, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ConfigFileError(
            f"invalid YAML {kind} {path}: parsed data is not JSON-compatible: {exc}",
            path=path,
            kind=kind,
        ) from exc


def _to_config_data(value: object, *, path: str, seen: set[int]) -> ConfigData:
    if value is None or isinstance(value, str | bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ConfigFileError(
                f"invalid YAML value at {path}: non-finite floats are not supported",
                path=Path("<memory>"),
                kind="config",
            )
        return value
    if isinstance(value, list):
        sequence = cast(list[object], value)
        value_id = id(cast(object, sequence))
        if value_id in seen:
            raise ConfigFileError(
                f"invalid YAML value at {path}: recursive sequences are not supported",
                path=Path("<memory>"),
                kind="config",
            )
        seen.add(value_id)
        try:
            return [
                _to_config_data(item, path=f"{path}[{index}]", seen=seen)
                for index, item in enumerate(sequence)
            ]
        finally:
            seen.remove(value_id)
    if isinstance(value, dict):
        mapping = cast(dict[object, object], value)
        value_id = id(cast(object, mapping))
        if value_id in seen:
            raise ConfigFileError(
                f"invalid YAML value at {path}: recursive mappings are not supported",
                path=Path("<memory>"),
                kind="config",
            )
        seen.add(value_id)
        try:
            result: dict[str, ConfigData] = {}
            for raw_key, raw_item in mapping.items():
                if not isinstance(raw_key, str):
                    raise ConfigFileError(
                        f"invalid YAML mapping key at {path}: keys must be strings",
                        path=Path("<memory>"),
                        kind="config",
                    )
                result[raw_key] = _to_config_data(
                    raw_item,
                    path=f"{path}.{raw_key}" if path != "<root>" else raw_key,
                    seen=seen,
                )
            return result
        finally:
            seen.remove(value_id)
    raise ConfigFileError(
        f"invalid YAML value at {path}: unsupported {type(value).__name__}",
        path=Path("<memory>"),
        kind="config",
    )


def _construct_config_int(loader: _ConfigYamlLoader, node: ScalarNode) -> int:
    value = loader.construct_scalar(node)
    if _JSON_INT_RESOLVER.fullmatch(value) is None:
        raise ConstructorError(
            "while constructing an integer",
            node.start_mark,
            f"found unsupported integer spelling {value!r}",
            node.start_mark,
        )
    return int(value)


def _construct_config_float(loader: _ConfigYamlLoader, node: ScalarNode) -> float:
    value = loader.construct_scalar(node)
    if _JSON_FLOAT_RESOLVER.fullmatch(value) is None:
        raise ConstructorError(
            "while constructing a float",
            node.start_mark,
            f"found unsupported float spelling {value!r}",
            node.start_mark,
        )
    return float(value)


def _construct_config_mapping(
    loader: _ConfigYamlLoader,
    node: MappingNode,
    deep: bool = False,
) -> dict[str, object]:
    loader.flatten_mapping(node)
    pairs = cast(list[tuple[object, object]], cast(Any, loader).construct_pairs(node, deep=deep))
    result: dict[str, object] = {}
    for raw_key, value in pairs:
        if not isinstance(raw_key, str):
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found non-string key",
                node.start_mark,
            )
        if raw_key in result:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {raw_key!r}",
                node.start_mark,
            )
        result[raw_key] = value
    return result


def _represent_config_string(dumper: _ConfigYamlDumper, data: str) -> ScalarNode:
    style = "|" if "\n" in data else "'" if _plain_string_would_change_type(data) else None
    return cast(ScalarNode, cast(Any, dumper).represent_scalar(_YAML_STRING_TAG, data, style=style))


def _plain_string_would_change_type(value: str) -> bool:
    return (
        value in _YAML_NULL_STRINGS
        or _BOOL_RESOLVER.fullmatch(value) is not None
        or _JSON_INT_RESOLVER.fullmatch(value) is not None
        or _JSON_FLOAT_RESOLVER.fullmatch(value) is not None
    )


def _yaml_implicit_resolvers_without_yaml_1_1_surprises() -> dict[
    str | None, list[tuple[Any, ...]]
]:
    resolvers: dict[str | None, list[tuple[Any, ...]]] = {}
    original = cast(
        dict[str | None, list[tuple[Any, ...]]],
        yaml.SafeLoader.yaml_implicit_resolvers,
    )
    for first, entries in original.items():
        resolvers[first] = [
            entry
            for entry in entries
            if entry[0]
            not in (_YAML_BOOL_TAG, _YAML_FLOAT_TAG, _YAML_INT_TAG, _YAML_TIMESTAMP_TAG)
        ]
    return resolvers


_ConfigYamlLoader.yaml_implicit_resolvers = _yaml_implicit_resolvers_without_yaml_1_1_surprises()
cast(Any, _ConfigYamlLoader).add_implicit_resolver(_YAML_BOOL_TAG, _BOOL_RESOLVER, list("tTfF"))
cast(Any, _ConfigYamlLoader).add_implicit_resolver(
    _YAML_FLOAT_TAG,
    _JSON_FLOAT_RESOLVER,
    list("-0123456789"),
)
cast(Any, _ConfigYamlLoader).add_implicit_resolver(
    _YAML_INT_TAG,
    _JSON_INT_RESOLVER,
    list("-0123456789"),
)
cast(Any, _ConfigYamlLoader).add_constructor(_YAML_INT_TAG, _construct_config_int)
cast(Any, _ConfigYamlLoader).add_constructor(_YAML_FLOAT_TAG, _construct_config_float)
cast(Any, _ConfigYamlLoader).add_constructor(_YAML_MAPPING_TAG, _construct_config_mapping)
cast(Any, _ConfigYamlDumper).add_representer(str, _represent_config_string)


__all__ = (
    "ConfigData",
    "ConfigFileError",
    "ConfigFormat",
    "config_format_from_path",
    "dump_config_model_yaml",
    "dump_yaml_data",
    "read_config_model",
    "read_yaml_config_data",
)
