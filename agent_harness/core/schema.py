"""Small dependency-free JSON-schema subset for tool contracts."""

from __future__ import annotations

import math
import re
from typing import Any, Mapping, Sequence

from .contracts import HarnessContractError
from .events import canonical_json


_SUPPORTED_KEYWORDS = frozenset(
    {
        "type",
        "description",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "enum",
        "const",
        "minLength",
        "maxLength",
        "pattern",
        "minimum",
        "maximum",
        "minItems",
        "maxItems",
        "uniqueItems",
    }
)
_SUPPORTED_TYPES = frozenset(
    {"object", "array", "string", "integer", "number", "boolean", "null"}
)


def _bounded_non_negative_integer(value: Any, *, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise HarnessContractError(f"{path}: value must be a non-negative integer")
    return value


def validate_schema_definition(
    schema: Mapping[str, Any] | None,
    *,
    path: str = "$schema",
    _depth: int = 0,
) -> None:
    """Validate the complete local Tool schema before it enters the registry."""

    if schema is None:
        return
    if not isinstance(schema, Mapping):
        raise HarnessContractError(f"{path}: schema must be an object")
    if _depth > 16:
        raise HarnessContractError(f"{path}: schema nesting is too deep")
    unknown = sorted(set(schema).difference(_SUPPORTED_KEYWORDS))
    if unknown:
        raise HarnessContractError(
            f"{path}: unsupported schema keywords: {', '.join(unknown)}"
        )
    raw_types = schema.get("type")
    if raw_types is None:
        declared_types: tuple[str, ...] = ()
    elif isinstance(raw_types, str):
        declared_types = (raw_types,)
    elif isinstance(raw_types, Sequence) and not isinstance(raw_types, (str, bytes)):
        declared_types = tuple(raw_types)
        if not declared_types or any(
            not isinstance(item, str) for item in declared_types
        ):
            raise HarnessContractError(f"{path}.type: types must be strings")
        if len(set(declared_types)) != len(declared_types):
            raise HarnessContractError(
                f"{path}.type: types must be non-empty and unique"
            )
    else:
        raise HarnessContractError(f"{path}.type: schema type is invalid")
    if any(
        not isinstance(item, str) or item not in _SUPPORTED_TYPES
        for item in declared_types
    ):
        raise HarnessContractError(f"{path}.type: unsupported schema type")

    description = schema.get("description")
    if description is not None and not isinstance(description, str):
        raise HarnessContractError(f"{path}.description: value must be a string")
    properties = schema.get("properties")
    if properties is not None:
        if not isinstance(properties, Mapping):
            raise HarnessContractError(f"{path}.properties: value must be an object")
        for name, child in properties.items():
            if not isinstance(name, str) or not name:
                raise HarnessContractError(
                    f"{path}.properties: property names must be non-empty strings"
                )
            validate_schema_definition(
                child,
                path=f"{path}.properties.{name}",
                _depth=_depth + 1,
            )
    required = schema.get("required")
    if required is not None:
        if not isinstance(required, Sequence) or isinstance(required, (str, bytes)):
            raise HarnessContractError(f"{path}.required: value must be an array")
        if any(not isinstance(item, str) or not item for item in required):
            raise HarnessContractError(
                f"{path}.required: entries must be non-empty strings"
            )
        if len(required) != len(set(required)):
            raise HarnessContractError(f"{path}.required: entries must be unique")
        if properties is not None and not set(required).issubset(properties):
            raise HarnessContractError(
                f"{path}.required: every entry must have a property schema"
            )
    additional = schema.get("additionalProperties")
    if additional is not None:
        if isinstance(additional, Mapping):
            validate_schema_definition(
                additional,
                path=f"{path}.additionalProperties",
                _depth=_depth + 1,
            )
        elif not isinstance(additional, bool):
            raise HarnessContractError(
                f"{path}.additionalProperties: value must be boolean or a schema"
            )
    items = schema.get("items")
    if items is not None:
        validate_schema_definition(
            items,
            path=f"{path}.items",
            _depth=_depth + 1,
        )
    if "enum" in schema:
        choices = schema["enum"]
        if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)):
            raise HarnessContractError(f"{path}.enum: value must be an array")
        rendered = [canonical_json(item) for item in choices]
        if not rendered or len(rendered) != len(set(rendered)):
            raise HarnessContractError(
                f"{path}.enum: values must be non-empty and unique"
            )
    if "const" in schema:
        canonical_json(schema["const"])
    for name in ("minLength", "maxLength", "minItems", "maxItems"):
        if name in schema:
            _bounded_non_negative_integer(schema[name], path=f"{path}.{name}")
    for minimum_name, maximum_name in (
        ("minLength", "maxLength"),
        ("minItems", "maxItems"),
    ):
        if (
            minimum_name in schema
            and maximum_name in schema
            and schema[minimum_name] > schema[maximum_name]
        ):
            raise HarnessContractError(
                f"{path}: {minimum_name} cannot exceed {maximum_name}"
            )
    pattern = schema.get("pattern")
    if pattern is not None:
        if not isinstance(pattern, str):
            raise HarnessContractError(f"{path}.pattern: value must be a string")
        try:
            re.compile(pattern)
        except re.error as exc:
            raise HarnessContractError(
                f"{path}.pattern: invalid regular expression"
            ) from exc
    for name in ("minimum", "maximum"):
        if name in schema:
            value = schema[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or isinstance(value, float)
                and not math.isfinite(value)
            ):
                raise HarnessContractError(f"{path}.{name}: value must be finite")
    if (
        "minimum" in schema
        and "maximum" in schema
        and schema["minimum"] > schema["maximum"]
    ):
        raise HarnessContractError(f"{path}: minimum cannot exceed maximum")
    if "uniqueItems" in schema and not isinstance(schema["uniqueItems"], bool):
        raise HarnessContractError(f"{path}.uniqueItems: value must be boolean")


def _matches_type(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, Mapping)
    if expected == "array":
        return isinstance(value, Sequence) and not isinstance(value, (str, bytes))
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and (not isinstance(value, float) or math.isfinite(value))
        )
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    raise HarnessContractError(f"unsupported tool schema type: {expected}")


def validate_schema(
    value: Any,
    schema: Mapping[str, Any] | None,
    *,
    path: str = "$",
    _depth: int = 0,
) -> None:
    """Validate the strict subset used by local tool definitions.

    Unsupported keywords fail closed instead of being silently ignored.
    """

    if schema is None:
        return
    if not isinstance(schema, Mapping):
        raise HarnessContractError(f"{path}: schema must be an object")
    if _depth > 16:
        raise HarnessContractError(f"{path}: schema nesting is too deep")
    validate_schema_definition(schema, path=f"{path}.__schema__", _depth=_depth)
    expected = schema.get("type")
    if isinstance(expected, str):
        expected_types = (expected,)
    elif isinstance(expected, Sequence) and not isinstance(expected, (str, bytes)):
        expected_types = tuple(str(item) for item in expected)
    elif expected is None:
        expected_types = ()
    else:
        raise HarnessContractError(f"{path}: schema type is invalid")
    if expected_types and not any(
        _matches_type(value, item) for item in expected_types
    ):
        raise HarnessContractError(
            f"{path}: expected {' or '.join(expected_types)}, got {type(value).__name__}"
        )
    if "const" in schema and value != schema["const"]:
        raise HarnessContractError(f"{path}: value does not match const")
    if "enum" in schema:
        choices = schema["enum"]
        if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)):
            raise HarnessContractError(f"{path}: enum must be an array")
        if value not in choices:
            raise HarnessContractError(f"{path}: value is not in enum")
    if isinstance(value, Mapping):
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        if not isinstance(properties, Mapping):
            raise HarnessContractError(f"{path}: properties must be an object")
        if not isinstance(required, Sequence) or isinstance(required, (str, bytes)):
            raise HarnessContractError(f"{path}: required must be an array")
        for name in required:
            if not isinstance(name, str):
                raise HarnessContractError(f"{path}: required entries must be strings")
            if name not in value:
                raise HarnessContractError(
                    f"{path}.{name}: required property is missing"
                )
        additional = schema.get("additionalProperties", True)
        if not isinstance(additional, (bool, Mapping)):
            raise HarnessContractError(
                f"{path}: additionalProperties must be boolean or schema"
            )
        for name, item in value.items():
            child_path = f"{path}.{name}"
            child_schema = properties.get(name)
            if child_schema is not None:
                validate_schema(
                    item,
                    child_schema,
                    path=child_path,
                    _depth=_depth + 1,
                )
            elif additional is False:
                raise HarnessContractError(
                    f"{child_path}: additional property rejected"
                )
            elif isinstance(additional, Mapping):
                validate_schema(
                    item,
                    additional,
                    path=child_path,
                    _depth=_depth + 1,
                )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        minimum = int(schema.get("minItems", 0))
        maximum = int(schema.get("maxItems", 1_000_000))
        if not minimum <= len(value) <= maximum:
            raise HarnessContractError(
                f"{path}: array length must be in [{minimum}, {maximum}]"
            )
        if schema.get("uniqueItems") is True:
            rendered = [canonical_json(item) for item in value]
            if len(rendered) != len(set(rendered)):
                raise HarnessContractError(f"{path}: array items must be unique")
        item_schema = schema.get("items")
        if item_schema is not None:
            for index, item in enumerate(value):
                validate_schema(
                    item,
                    item_schema,
                    path=f"{path}[{index}]",
                    _depth=_depth + 1,
                )
    if isinstance(value, str):
        minimum = int(schema.get("minLength", 0))
        maximum = int(schema.get("maxLength", 1_000_000))
        if not minimum <= len(value) <= maximum:
            raise HarnessContractError(
                f"{path}: string length must be in [{minimum}, {maximum}]"
            )
        pattern = schema.get("pattern")
        if pattern is not None:
            if not isinstance(pattern, str):
                raise HarnessContractError(f"{path}: pattern must be a string")
            if re.search(pattern, value) is None:
                raise HarnessContractError(f"{path}: string does not match pattern")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise HarnessContractError(f"{path}: number is below minimum")
        if "maximum" in schema and value > schema["maximum"]:
            raise HarnessContractError(f"{path}: number is above maximum")
