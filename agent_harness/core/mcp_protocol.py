"""Strict MCP stdio tools protocol subset.

The Harness intentionally negotiates the widely deployed 2025-06-18
connection-oriented protocol.  It implements only initialize, ping,
tools/list, tools/call and cancellation.  Unsupported schema or content
features fail closed instead of silently weakening local validation.
"""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
import json
import math
import re
from typing import Any, Mapping, Sequence

from .contracts import HarnessContractError
from .events import canonical_json, canonical_sha256
from .schema import validate_schema, validate_schema_definition


MCP_PROTOCOL_VERSION = "2025-06-18"
MCP_CLIENT_NAME = "agent-harness"
MCP_TOOL_RESULT_SCHEMA = "agent_harness.mcp_tool_result.v1"
MAX_MCP_FRAME_BYTES = 1_000_000
MAX_MCP_TOOLS_PER_SERVER = 128
MAX_MCP_LIST_PAGES = 64
MAX_MCP_CATALOG_BYTES = 2_000_000

_RAW_TOOL_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
_SERVER_ID = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")
_SCHEMA_PROPERTY_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,127}$")
_LOCAL_COMPONENT = re.compile(r"[^a-z0-9.-]+")
_IGNORED_SCHEMA_ANNOTATIONS = frozenset(
    {
        "$schema",
        "title",
        "description",
        "default",
        "examples",
        "deprecated",
        "readOnly",
        "writeOnly",
    }
)
_SUPPORTED_SCHEMA_KEYWORDS = frozenset(
    {
        "type",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "enum",
        "const",
        "minLength",
        "maxLength",
        "minimum",
        "maximum",
        "minItems",
        "maxItems",
        "uniqueItems",
    }
)


class McpProtocolError(RuntimeError):
    """Raised when a peer violates the negotiated MCP/JSON-RPC contract."""

    def __init__(self, message: str, *, code: str = "mcp_protocol_error") -> None:
        super().__init__(message)
        self.code = code


class McpRemoteError(McpProtocolError):
    """A validated JSON-RPC error returned by the server."""

    def __init__(self, code_value: int, message: str) -> None:
        self.remote_code = code_value
        super().__init__(message, code="mcp_remote_error")


def _strict_object_pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in values:
        if key in result:
            raise ValueError("duplicate object key")
        result[key] = value
    return result


def _reject_constant(_value: str) -> Any:
    raise ValueError("non-finite JSON number")


def _strict_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("non-finite JSON number")
    return parsed


def _check_complexity(value: Any, *, depth: int = 0) -> tuple[int, int]:
    if depth > 48:
        raise McpProtocolError("MCP JSON nesting is too deep", code="mcp_frame_complexity")
    nodes = 1
    characters = 0
    if isinstance(value, str):
        characters = len(value)
        try:
            value.encode("utf-8", errors="strict")
        except UnicodeEncodeError as exc:
            raise McpProtocolError(
                "MCP string is not valid Unicode",
                code="mcp_frame_invalid",
            ) from exc
        if characters > MAX_MCP_FRAME_BYTES:
            raise McpProtocolError("MCP string is too large", code="mcp_frame_complexity")
    elif isinstance(value, Mapping):
        if len(value) > 10_000:
            raise McpProtocolError("MCP object is too large", code="mcp_frame_complexity")
        for key, item in value.items():
            if not isinstance(key, str):
                raise McpProtocolError("MCP object key is invalid")
            try:
                key.encode("utf-8", errors="strict")
            except UnicodeEncodeError as exc:
                raise McpProtocolError(
                    "MCP object key is not valid Unicode",
                    code="mcp_frame_invalid",
                ) from exc
            characters += len(key)
            child_nodes, child_characters = _check_complexity(item, depth=depth + 1)
            nodes += child_nodes
            characters += child_characters
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) > 10_000:
            raise McpProtocolError("MCP array is too large", code="mcp_frame_complexity")
        for item in value:
            child_nodes, child_characters = _check_complexity(item, depth=depth + 1)
            nodes += child_nodes
            characters += child_characters
    if nodes > 50_000 or characters > MAX_MCP_FRAME_BYTES:
        raise McpProtocolError("MCP frame is too complex", code="mcp_frame_complexity")
    return nodes, characters


def decode_mcp_frame(data: bytes) -> dict[str, Any]:
    """Decode one newline-free strict UTF-8 JSON-RPC object."""

    if not data or len(data) > MAX_MCP_FRAME_BYTES or b"\n" in data or b"\r" in data:
        raise McpProtocolError("MCP frame size or delimiter is invalid", code="mcp_frame_invalid")
    try:
        text = data.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_strict_object_pairs,
            parse_constant=_reject_constant,
            parse_float=_strict_float,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise McpProtocolError("MCP frame is not strict UTF-8 JSON", code="mcp_frame_invalid") from exc
    if not isinstance(value, Mapping):
        raise McpProtocolError("MCP batch and scalar frames are unsupported", code="mcp_frame_invalid")
    result = dict(value)
    _check_complexity(result)
    if result.get("jsonrpc") != "2.0":
        raise McpProtocolError("MCP frame is not JSON-RPC 2.0", code="mcp_jsonrpc_invalid")
    return result


def encode_mcp_frame(value: Mapping[str, Any]) -> bytes:
    try:
        encoded = canonical_json(dict(value)).encode("utf-8", errors="strict")
    except (HarnessContractError, UnicodeEncodeError) as exc:
        raise McpProtocolError(
            "outgoing MCP frame is not canonical JSON",
            code="mcp_frame_invalid",
        ) from exc
    if len(encoded) > MAX_MCP_FRAME_BYTES:
        raise McpProtocolError("outgoing MCP frame is too large", code="mcp_frame_too_large")
    return encoded + b"\n"


def response_result(frame: Mapping[str, Any], request_id: int) -> Mapping[str, Any]:
    if (
        "method" in frame
        or type(frame.get("id")) is not int
        or frame.get("id") != request_id
    ):
        raise McpProtocolError("MCP response id does not match the request", code="mcp_response_mismatch")
    has_result = "result" in frame
    has_error = "error" in frame
    if has_result == has_error:
        raise McpProtocolError("MCP response must contain exactly one result or error")
    if has_error:
        error = frame.get("error")
        if not isinstance(error, Mapping):
            raise McpProtocolError("MCP JSON-RPC error is invalid")
        code = error.get("code")
        message = error.get("message")
        if isinstance(code, bool) or not isinstance(code, int) or not isinstance(message, str):
            raise McpProtocolError("MCP JSON-RPC error is invalid")
        raise McpRemoteError(code, message[:500] or "MCP server error")
    result = frame.get("result")
    if not isinstance(result, Mapping):
        raise McpProtocolError("MCP result must be an object")
    return result


def parse_server_message(
    frame: Mapping[str, Any],
) -> tuple[str, int | str | None, Mapping[str, Any], bool]:
    """Validate one server-initiated JSON-RPC request or notification.

    The negotiated MCP subset uses object-shaped params.  The final boolean
    distinguishes notifications from the valid-but-discouraged explicit JSON
    ``null`` request id.
    """

    if "method" not in frame or "result" in frame or "error" in frame:
        raise McpProtocolError("MCP server message shape is invalid", code="mcp_jsonrpc_invalid")
    method = frame.get("method")
    if not isinstance(method, str) or not 1 <= len(method) <= 256:
        raise McpProtocolError("MCP server method is invalid", code="mcp_jsonrpc_invalid")
    params = frame.get("params", {})
    if not isinstance(params, Mapping):
        raise McpProtocolError("MCP server params must be an object", code="mcp_jsonrpc_invalid")
    request_id: int | str | None = None
    has_request_id = "id" in frame
    if has_request_id:
        raw_id = frame.get("id")
        if (
            isinstance(raw_id, bool)
            or raw_id is not None
            and not isinstance(raw_id, (int, str))
            or isinstance(raw_id, int)
            and not -(2**63) <= raw_id <= 2**63 - 1
            or isinstance(raw_id, str)
            and len(raw_id) > 2_000
        ):
            raise McpProtocolError("MCP server request id is invalid", code="mcp_jsonrpc_invalid")
        request_id = raw_id
    return method, request_id, dict(params), has_request_id


def _normalize_schema(value: Any, *, path: str, depth: int = 0) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise McpProtocolError(f"{path} must be a JSON Schema object", code="mcp_schema_unsupported")
    if depth > 16:
        raise McpProtocolError(f"{path} nesting is too deep", code="mcp_schema_unsupported")
    normalized: dict[str, Any] = {}
    for key, raw in value.items():
        if key in _IGNORED_SCHEMA_ANNOTATIONS:
            continue
        if key not in _SUPPORTED_SCHEMA_KEYWORDS:
            raise McpProtocolError(
                f"{path} uses unsupported JSON Schema keyword {key}",
                code="mcp_schema_unsupported",
            )
        if key == "properties":
            if not isinstance(raw, Mapping):
                raise McpProtocolError(f"{path}.properties must be an object", code="mcp_schema_unsupported")
            properties: dict[str, Any] = {}
            for name, child in raw.items():
                if (
                    not isinstance(name, str)
                    or _SCHEMA_PROPERTY_NAME.fullmatch(name) is None
                ):
                    raise McpProtocolError(
                        f"{path}.properties contains an invalid name",
                        code="mcp_schema_unsupported",
                    )
                properties[name] = _normalize_schema(
                    child,
                    path=f"{path}.properties.{name}",
                    depth=depth + 1,
                )
            normalized[key] = properties
        elif key == "required":
            if (
                not isinstance(raw, list)
                or any(
                    not isinstance(name, str)
                    or _SCHEMA_PROPERTY_NAME.fullmatch(name) is None
                    for name in raw
                )
            ):
                raise McpProtocolError(
                    f"{path}.required contains an invalid name",
                    code="mcp_schema_unsupported",
                )
            normalized[key] = raw
        elif key in {"items", "additionalProperties"} and isinstance(raw, Mapping):
            normalized[key] = _normalize_schema(raw, path=f"{path}.{key}", depth=depth + 1)
        else:
            normalized[key] = raw
    try:
        validate_schema_definition(normalized, path=path)
    except HarnessContractError as exc:
        raise McpProtocolError(str(exc), code="mcp_schema_unsupported") from exc
    return json.loads(canonical_json(normalized))


def local_tool_name(server_id: str, raw_name: str) -> str:
    if _SERVER_ID.fullmatch(server_id) is None:
        raise McpProtocolError("MCP server id is invalid", code="mcp_catalog_invalid")
    slug = _LOCAL_COMPONENT.sub("-", raw_name.casefold()).strip("-.") or "tool"
    slug = slug[:36].rstrip("-.") or "tool"
    suffix = canonical_sha256(raw_name)[:16]
    return f"mcp.{server_id}.{slug}.{suffix}"


@dataclass(frozen=True, slots=True)
class McpToolDefinition:
    raw_name: str
    local_name: str
    input_schema: Mapping[str, Any]
    output_schema: Mapping[str, Any] | None
    definition_sha256: str

    @classmethod
    def from_mapping(cls, server_id: str, raw: Mapping[str, Any]) -> "McpToolDefinition":
        raw_name = raw.get("name")
        if not isinstance(raw_name, str) or _RAW_TOOL_NAME.fullmatch(raw_name) is None:
            raise McpProtocolError("MCP tool name is invalid", code="mcp_tool_invalid")
        input_schema = _normalize_schema(raw.get("inputSchema"), path=f"tool.{raw_name}.input")
        if input_schema.get("type") != "object":
            raise McpProtocolError("MCP tool input schema must have object type", code="mcp_schema_unsupported")
        raw_output = raw.get("outputSchema")
        output_schema = (
            _normalize_schema(raw_output, path=f"tool.{raw_name}.output")
            if raw_output is not None
            else None
        )
        if output_schema is not None and output_schema.get("type") != "object":
            raise McpProtocolError("MCP 2025-06-18 output schema must have object type", code="mcp_schema_unsupported")
        material = {
            "schema": "agent_harness.mcp_remote_tool.v1",
            "raw_name": raw_name,
            "input_schema": input_schema,
            "output_schema": output_schema,
        }
        try:
            definition_sha256 = canonical_sha256(material)
        except (HarnessContractError, UnicodeEncodeError) as exc:
            raise McpProtocolError(
                "MCP tool schema is not canonical Unicode JSON",
                code="mcp_schema_unsupported",
            ) from exc
        return cls(
            raw_name=raw_name,
            local_name=local_tool_name(server_id, raw_name),
            input_schema=input_schema,
            output_schema=output_schema,
            definition_sha256=definition_sha256,
        )

    def to_cache(self) -> dict[str, Any]:
        return {
            "raw_name": self.raw_name,
            "local_name": self.local_name,
            "input_schema": dict(self.input_schema),
            "output_schema": (
                dict(self.output_schema) if self.output_schema is not None else None
            ),
            "definition_sha256": self.definition_sha256,
        }

    @classmethod
    def from_cache(
        cls,
        server_id: str,
        value: Mapping[str, Any],
    ) -> "McpToolDefinition":
        if set(value) != {
            "raw_name",
            "local_name",
            "input_schema",
            "output_schema",
            "definition_sha256",
        }:
            raise McpProtocolError(
                "cached MCP tool shape is invalid",
                code="mcp_catalog_invalid",
            )
        candidate = cls.from_mapping(
            server_id,
            {
                "name": value.get("raw_name"),
                "inputSchema": value.get("input_schema"),
                "outputSchema": value.get("output_schema"),
            },
        )
        if (
            value.get("local_name") != candidate.local_name
            or value.get("definition_sha256") != candidate.definition_sha256
        ):
            raise McpProtocolError("cached MCP tool identity is invalid", code="mcp_catalog_invalid")
        return candidate


def parse_tool_page(server_id: str, result: Mapping[str, Any]) -> tuple[list[McpToolDefinition], str | None, list[dict[str, str]]]:
    raw_tools = result.get("tools")
    if not isinstance(raw_tools, list):
        raise McpProtocolError("MCP tools/list result has no tools array")
    accepted: list[McpToolDefinition] = []
    rejected: list[dict[str, str]] = []
    for raw in raw_tools:
        if not isinstance(raw, Mapping):
            rejected.append({"tool_sha256": canonical_sha256(raw), "reason_code": "mcp_tool_invalid"})
            continue
        try:
            accepted.append(McpToolDefinition.from_mapping(server_id, raw))
        except McpProtocolError as exc:
            rejected.append(
                {
                    "tool_sha256": canonical_sha256(dict(raw)),
                    "reason_code": exc.code,
                }
            )
    cursor = result.get("nextCursor")
    if cursor is not None and (not isinstance(cursor, str) or not cursor or len(cursor) > 2_000):
        raise McpProtocolError("MCP tools/list cursor is invalid")
    return accepted, cursor, rejected


def _omitted_content(item: Mapping[str, Any]) -> dict[str, Any]:
    content_type = str(item.get("type", "unknown"))[:80]
    material = canonical_json(dict(item)).encode("utf-8")
    result: dict[str, Any] = {
        "type": content_type,
        "omitted": True,
        "byte_length": len(material),
        "sha256": canonical_sha256(dict(item)),
    }
    mime = item.get("mimeType")
    if isinstance(mime, str) and 0 < len(mime) <= 200:
        result["mime_type"] = mime
    data = item.get("data")
    if isinstance(data, str) and len(data) <= MAX_MCP_FRAME_BYTES:
        try:
            decoded = base64.b64decode(data, validate=True)
        except (ValueError, binascii.Error):
            pass
        else:
            result["decoded_byte_length"] = len(decoded)
    return result


def normalize_tool_result(
    server_id: str,
    tool: McpToolDefinition,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    result_type = result.get("resultType")
    if result_type not in (None, "complete"):
        raise McpProtocolError("MCP input-required/task results are unsupported", code="mcp_result_unsupported")
    is_error = result.get("isError", False)
    if not isinstance(is_error, bool):
        raise McpProtocolError("MCP tool isError flag is invalid")
    raw_content = result.get("content", [])
    if not isinstance(raw_content, list) or len(raw_content) > 256:
        raise McpProtocolError("MCP tool content is invalid", code="mcp_result_invalid")
    content: list[dict[str, Any]] = []
    total_text = 0
    for raw in raw_content:
        if not isinstance(raw, Mapping):
            raise McpProtocolError("MCP content item is invalid", code="mcp_result_invalid")
        content_type = raw.get("type")
        if content_type == "text":
            text = raw.get("text")
            if not isinstance(text, str):
                raise McpProtocolError("MCP text content is invalid", code="mcp_result_invalid")
            total_text += len(text)
            if total_text > 200_000:
                raise McpProtocolError("MCP text result is too large", code="mcp_result_too_large")
            content.append({"type": "text", "text": text})
        else:
            content.append(_omitted_content(raw))
    structured = result.get("structuredContent")
    if structured is not None:
        if not isinstance(structured, Mapping):
            raise McpProtocolError("MCP structuredContent must be an object", code="mcp_result_invalid")
        if tool.output_schema is not None:
            try:
                validate_schema(structured, tool.output_schema, path="$.structured_content")
            except HarnessContractError as exc:
                raise McpProtocolError("MCP structuredContent violates outputSchema", code="mcp_output_schema_invalid") from exc
    normalized: dict[str, Any] = {
        "schema": MCP_TOOL_RESULT_SCHEMA,
        "server_id": server_id,
        "raw_tool_name": tool.raw_name,
        "is_error": is_error,
        "content": content,
    }
    if structured is not None:
        normalized["structured_content"] = json.loads(canonical_json(structured))
    return normalized


__all__ = [
    "MAX_MCP_CATALOG_BYTES",
    "MAX_MCP_FRAME_BYTES",
    "MAX_MCP_LIST_PAGES",
    "MAX_MCP_TOOLS_PER_SERVER",
    "MCP_CLIENT_NAME",
    "MCP_PROTOCOL_VERSION",
    "MCP_TOOL_RESULT_SCHEMA",
    "McpProtocolError",
    "McpRemoteError",
    "McpToolDefinition",
    "decode_mcp_frame",
    "encode_mcp_frame",
    "local_tool_name",
    "normalize_tool_result",
    "parse_server_message",
    "parse_tool_page",
    "response_result",
]
