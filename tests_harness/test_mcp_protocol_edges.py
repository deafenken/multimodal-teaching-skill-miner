from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
from threading import Event, Thread
import time
from typing import Any, Mapping

import pytest

import agent_harness.mcp as mcp_module
from agent_harness.core import CancellationToken, HarnessCancelled
from agent_harness.core.events import canonical_sha256
from agent_harness.core.mcp_protocol import (
    MCP_PROTOCOL_VERSION,
    McpProtocolError,
    McpToolDefinition,
    decode_mcp_frame,
    parse_server_message,
    response_result,
)
from agent_harness.mcp import (
    MCP_CATALOG_RECORD_SCHEMA,
    MCP_CONFIG_SCHEMA,
    McpLoadError,
    McpServerDefinition,
    McpSnapshot,
    StdioMcpConnection,
    TrustedMcpCatalog,
    load_project_mcp,
    refresh_mcp_catalog,
)


def _tool(server_id: str, raw_name: str = "Echo") -> McpToolDefinition:
    return McpToolDefinition.from_mapping(
        server_id,
        {
            "name": raw_name,
            "inputSchema": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False,
            },
        },
    )


def _definition(tmp_path: Path, script_text: str, *, tool_timeout: float = 1.0) -> McpServerDefinition:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    script = workspace / "server.py"
    script.write_text(script_text, encoding="utf-8")
    script.chmod(0o700)
    config_directory = workspace / ".agent-harness"
    config_directory.mkdir(mode=0o700)
    config = {
        "schema": MCP_CONFIG_SCHEMA,
        "servers": {
            "fixture": {
                "transport": "stdio",
                "command": "server.py",
                "args": [],
                "cwd": ".",
                "pass_env": [],
                "network_access": False,
                "allow_process_fork": False,
                "startup_timeout_seconds": 2,
                "tool_timeout_seconds": tool_timeout,
            }
        },
    }
    config_path = config_directory / "mcp.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    config_path.chmod(0o600)
    return load_project_mcp(workspace).server("fixture")


def _catalog(
    definition: McpServerDefinition,
    tools: list[dict[str, Any]],
    rejected: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    frozen_rejected = rejected or []
    material = {
        "schema": "agent_harness.mcp_catalog.v1",
        "protocol_version": MCP_PROTOCOL_VERSION,
        "server_definition_sha256": definition.definition_sha256,
        "tools": tools,
        "rejected_tools": frozen_rejected,
    }
    return {
        "schema": MCP_CATALOG_RECORD_SCHEMA,
        "definition_sha256": definition.definition_sha256,
        "protocol_version": MCP_PROTOCOL_VERSION,
        "catalog_sha256": canonical_sha256(material),
        "server_info_sha256": canonical_sha256(None),
        "instructions_sha256": canonical_sha256(None),
        "tools": tools,
        "rejected_tools": frozen_rejected,
    }


def _snapshot(definition: McpServerDefinition) -> McpSnapshot:
    return McpSnapshot(
        workspace_root=definition.workspace_root,
        config_present=True,
        config_sha256=definition.config_sha256,
        servers=(definition,),
        snapshot_sha256="f" * 64,
    )


def test_cached_tool_identity_is_bound_to_the_current_server() -> None:
    alpha = _tool("alpha").to_cache()
    assert McpToolDefinition.from_cache("alpha", alpha).local_name == alpha["local_name"]

    beta = _tool("beta").to_cache()
    with pytest.raises(McpProtocolError, match="cached MCP tool identity"):
        McpToolDefinition.from_cache("alpha", beta)


def test_trusted_catalog_recomputes_digest_and_rejects_self_consistent_tampering(
    tmp_path: Path,
) -> None:
    definition = _definition(tmp_path, "#!/bin/sh\nexit 0\n")
    original = _catalog(definition, [_tool("fixture", "Echo").to_cache()])
    trust = {
        "fixture": {
            "action": "trusted",
            "definition_sha256": definition.definition_sha256,
        }
    }
    manager = TrustedMcpCatalog(_snapshot(definition), trust, {"fixture": original})
    assert manager.tool_count == 1

    tampered = deepcopy(original)
    tampered["tools"] = [_tool("fixture", "Different").to_cache()]
    # The attacker leaves the old catalog digest in place.  Every individual
    # tool field is otherwise internally self-consistent.
    with pytest.raises(McpLoadError, match="digest does not match"):
        TrustedMcpCatalog(_snapshot(definition), trust, {"fixture": tampered})
    status = _snapshot(definition).metadata(trust, {"fixture": tampered})
    assert status["servers"][0]["catalog_status"] == "invalid"
    assert status["ready_server_count"] == 0


def test_trusted_catalog_rejects_cross_server_tool_even_with_recomputed_digest(
    tmp_path: Path,
) -> None:
    definition = _definition(tmp_path, "#!/bin/sh\nexit 0\n")
    beta_tool = _tool("beta").to_cache()
    forged = _catalog(definition, [beta_tool])
    trust = {
        "fixture": {
            "action": "trusted",
            "definition_sha256": definition.definition_sha256,
        }
    }

    with pytest.raises(McpLoadError, match="tool identity"):
        TrustedMcpCatalog(_snapshot(definition), trust, {"fixture": forged})


def test_jsonrpc_ids_and_server_messages_are_strict() -> None:
    for invalid_id in (True, 1.0, "1"):
        with pytest.raises(McpProtocolError, match="response id"):
            response_result(
                {"jsonrpc": "2.0", "id": invalid_id, "result": {}},
                1,
            )
    assert parse_server_message(
        {"jsonrpc": "2.0", "id": None, "method": "ping"}
    ) == ("ping", None, {}, True)
    for frame in (
        {"jsonrpc": "2.0", "id": True, "method": "ping"},
        {"jsonrpc": "2.0", "id": 1.0, "method": "ping"},
        {"jsonrpc": "2.0", "id": 2**63, "method": "ping"},
        {"jsonrpc": "2.0", "id": "x" * 2_001, "method": "ping"},
        {"jsonrpc": "2.0", "id": 1, "method": "ping", "params": []},
        {"jsonrpc": "2.0", "id": 1, "method": "ping", "result": {}},
    ):
        with pytest.raises(McpProtocolError):
            parse_server_message(frame)


def test_mcp_json_rejects_exponent_overflow_and_remote_regex_schemas() -> None:
    with pytest.raises(McpProtocolError) as overflow:
        decode_mcp_frame(b'{"jsonrpc":"2.0","id":1,"result":{"value":1e9999}}')
    assert overflow.value.code == "mcp_frame_invalid"
    with pytest.raises(McpProtocolError) as surrogate:
        decode_mcp_frame(b'{"jsonrpc":"2.0","id":1,"result":{"value":"\\ud800"}}')
    assert surrogate.value.code == "mcp_frame_invalid"

    with pytest.raises(McpProtocolError) as pattern:
        McpToolDefinition.from_mapping(
            "fixture",
            {
                "name": "Regex",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "pattern": "^(a+)+$"}
                    },
                    "additionalProperties": False,
                },
            },
        )
    assert pattern.value.code == "mcp_schema_unsupported"


def test_remote_schema_property_and_required_names_are_safe_and_bounded() -> None:
    for bad_name in ("line\nbreak", "x" * 129, "-starts-with-punctuation"):
        with pytest.raises(McpProtocolError) as property_error:
            McpToolDefinition.from_mapping(
                "fixture",
                {
                    "name": "UnsafeProperty",
                    "inputSchema": {
                        "type": "object",
                        "properties": {bad_name: {"type": "string"}},
                        "additionalProperties": False,
                    },
                },
            )
        assert property_error.value.code == "mcp_schema_unsupported"

    with pytest.raises(McpProtocolError) as required_error:
        McpToolDefinition.from_mapping(
            "fixture",
            {
                "name": "UnsafeRequired",
                "inputSchema": {
                    "type": "object",
                    "properties": {"safe": {"type": "string"}},
                    "required": ["unsafe\nname"],
                    "additionalProperties": False,
                },
            },
        )
    assert required_error.value.code == "mcp_schema_unsupported"


def test_active_catalogs_have_a_global_tool_budget(tmp_path: Path) -> None:
    base = _definition(tmp_path, "#!/bin/sh\nexit 0\n")
    definitions = tuple(
        replace(
            base,
            server_id=f"server{server_index}",
            definition_sha256=canonical_sha256({"server": server_index}),
        )
        for server_index in range(3)
    )
    snapshot = McpSnapshot(
        workspace_root=base.workspace_root,
        config_present=True,
        config_sha256=base.config_sha256,
        servers=definitions,
        snapshot_sha256="e" * 64,
    )
    trust: dict[str, dict[str, str]] = {}
    catalogs: dict[str, dict[str, Any]] = {}
    for definition in definitions:
        trust[definition.server_id] = {
            "action": "trusted",
            "definition_sha256": definition.definition_sha256,
        }
        tools = [
            _tool(definition.server_id, f"Tool{index:03d}").to_cache()
            for index in range(100)
        ]
        catalogs[definition.server_id] = _catalog(definition, tools)

    with pytest.raises(McpLoadError, match="catalog budget"):
        TrustedMcpCatalog(snapshot, trust, catalogs)
    status = snapshot.metadata(trust, catalogs)
    assert status["ready_server_count"] == 2
    assert status["servers"][2]["catalog_status"] == "capacity_exceeded"
    assert status["refresh_required_count"] == 1


def test_connection_rejects_requests_outside_initialized_lifecycle(
    tmp_path: Path,
) -> None:
    definition = _definition(tmp_path, "#!/bin/sh\nexit 0\n")
    connection = StdioMcpConnection(definition)
    token = CancellationToken()
    with pytest.raises(McpProtocolError) as request_error:
        connection.request(
            "tools/list",
            {},
            deadline_monotonic=time.monotonic() + 1,
            cancellation_token=token,
        )
    assert request_error.value.code == "mcp_lifecycle_invalid"
    with pytest.raises(McpProtocolError) as initialize_error:
        connection.initialize(
            deadline_monotonic=time.monotonic() + 1,
            cancellation_token=token,
        )
    assert initialize_error.value.code == "mcp_lifecycle_invalid"


def test_initialize_rejects_dynamic_tool_catalog_capability(tmp_path: Path) -> None:
    definition = _definition(tmp_path, "#!/bin/sh\nexit 0\n")
    connection = StdioMcpConnection(definition)
    connection._state = "started"
    connection.request = lambda *_args, **_kwargs: {  # type: ignore[method-assign]
        "protocolVersion": MCP_PROTOCOL_VERSION,
        "capabilities": {"tools": {"listChanged": True}},
        "serverInfo": {"name": "dynamic", "version": "1"},
    }

    with pytest.raises(McpProtocolError) as captured:
        connection.initialize(
            deadline_monotonic=time.monotonic() + 1,
            cancellation_token=CancellationToken(),
        )
    assert captured.value.code == "mcp_dynamic_catalog_unsupported"
    assert connection._state == "failed"


def test_catalog_limit_counts_rejected_tools_and_detects_cursor_cycles(
    tmp_path: Path,
) -> None:
    definition = _definition(tmp_path, "#!/bin/sh\nexit 0\n")
    connection = StdioMcpConnection(definition)
    unsupported = [
        {
            "name": f"bad-{index}",
            "inputSchema": {"type": "object", "anyOf": [{"type": "object"}]},
        }
        for index in range(129)
    ]
    connection.request = lambda *_args, **_kwargs: {"tools": unsupported}  # type: ignore[method-assign]
    with pytest.raises(McpProtocolError) as count_error:
        connection.list_tools(
            deadline_monotonic=time.monotonic() + 1,
            cancellation_token=CancellationToken(),
        )
    assert count_error.value.code == "mcp_catalog_too_large"

    pages = iter(
        (
            {"tools": [], "nextCursor": "same"},
            {"tools": [], "nextCursor": "same"},
        )
    )
    connection.request = lambda *_args, **_kwargs: next(pages)  # type: ignore[method-assign]
    with pytest.raises(McpProtocolError) as cursor_error:
        connection.list_tools(
            deadline_monotonic=time.monotonic() + 1,
            cancellation_token=CancellationToken(),
        )
    assert cursor_error.value.code == "mcp_cursor_cycle"


def test_tool_call_applies_server_timeout_independently_of_run_deadline(
    tmp_path: Path,
) -> None:
    definition = _definition(
        tmp_path,
        "#!/bin/sh\nexit 0\n",
        tool_timeout=1.0,
    )
    connection = StdioMcpConnection(definition)
    observed: list[float] = []

    def request(
        _method: str,
        _params: Mapping[str, Any],
        *,
        deadline_monotonic: float,
        cancellation_token: CancellationToken,
        **_kwargs: Any,
    ) -> Mapping[str, Any]:
        del cancellation_token
        observed.append(deadline_monotonic)
        return {"content": []}

    connection.request = request  # type: ignore[method-assign]
    started = time.monotonic()
    connection.call_tool(
        _tool("fixture"),
        {"text": "hello"},
        deadline_monotonic=started + 100,
        cancellation_token=CancellationToken(),
    )

    assert len(observed) == 1
    assert started < observed[0] <= started + 1.1


def test_cancellation_sends_notification_only_for_active_initialized_request(
    tmp_path: Path,
) -> None:
    definition = _definition(tmp_path, "#!/bin/sh\nexit 0\n")
    connection = StdioMcpConnection(definition)
    connection._state = "ready"
    connection._initialized = True
    token = CancellationToken()
    entered_read = Event()
    writes: list[Mapping[str, Any]] = []
    terminations: list[bool] = []
    failures: list[BaseException] = []

    connection._write = lambda value, **_kwargs: writes.append(value)  # type: ignore[method-assign]
    connection._terminate = lambda: terminations.append(True)  # type: ignore[method-assign]

    def next_frame(**_kwargs: Any) -> Mapping[str, Any]:
        entered_read.set()
        while not token.cancelled:
            time.sleep(0.001)
        token.raise_if_cancelled()
        raise AssertionError("unreachable")

    connection._next_frame = next_frame  # type: ignore[method-assign]

    def invoke() -> None:
        try:
            connection.request(
                "tools/call",
                {"name": "Echo", "arguments": {}},
                deadline_monotonic=time.monotonic() + 2,
                cancellation_token=token,
            )
        except BaseException as exc:
            failures.append(exc)

    thread = Thread(target=invoke)
    thread.start()
    assert entered_read.wait(1)
    assert token.cancel("test cancellation") is True
    thread.join(2)

    assert not thread.is_alive()
    assert len(failures) == 1 and isinstance(failures[0], HarnessCancelled)
    assert [item["method"] for item in writes] == [
        "tools/call",
        "notifications/cancelled",
    ]
    assert terminations


def test_refresh_cleanup_error_does_not_mask_primary_protocol_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    definition = _definition(tmp_path, "#!/bin/sh\nexit 0\n")
    snapshot = _snapshot(definition)

    class Store:
        def mcp_trust_state(self) -> dict[str, dict[str, str]]:
            return {
                "fixture": {
                    "action": "trusted",
                    "definition_sha256": definition.definition_sha256,
                }
            }

    class BrokenConnection:
        catalog_stale = False

        def __init__(self, _definition: McpServerDefinition) -> None:
            pass

        def start(self) -> None:
            pass

        def initialize(self, **_kwargs: Any) -> None:
            raise McpProtocolError("primary failure", code="mcp_primary_failure")

        def _terminate(self) -> None:
            pass

        def close(self) -> None:
            raise McpProtocolError("cleanup failure", code="mcp_cleanup_failed")

    monkeypatch.setattr(mcp_module, "StdioMcpConnection", BrokenConnection)

    with pytest.raises(McpProtocolError) as captured:
        refresh_mcp_catalog(Store(), snapshot, "fixture")  # type: ignore[arg-type]
    assert captured.value.code == "mcp_primary_failure"


_STDERR_SERVER = r'''#!/usr/bin/env python3
import json
import sys

sys.stderr.write("x" * 70000)
sys.stderr.flush()
for line in sys.stdin:
    request = json.loads(line)
    if request.get("method") == "initialize":
        result = {
            "protocolVersion": "2025-06-18",
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "stderr-fixture", "version": "1"},
        }
        print(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}), flush=True)
    elif request.get("method") == "notifications/initialized":
        continue
    elif request.get("method") == "tools/list":
        print(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": {"tools": []}}), flush=True)
'''


def test_stderr_flood_is_drained_and_only_bounded_metadata_is_exposed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    definition = _definition(tmp_path, _STDERR_SERVER)

    def direct_launch(
        _workspace: str,
        _runtime: Path,
        command: list[str],
        **_kwargs: Any,
    ) -> tuple[list[str], dict[str, str]]:
        return command, dict(os.environ)

    monkeypatch.setattr(mcp_module, "mcp_stdio_sandbox_launch", direct_launch)
    connection = StdioMcpConnection(definition)
    connection.start()
    deadline = time.monotonic() + 2
    connection.initialize(
        deadline_monotonic=deadline,
        cancellation_token=CancellationToken(),
    )
    connection.list_tools(
        deadline_monotonic=deadline,
        cancellation_token=CancellationToken(),
    )
    connection.close()

    metadata = connection.stderr_metadata
    assert metadata["byte_length"] == 70_000
    assert metadata["truncated"] is True
    assert len(metadata["sha256"]) == 64


def test_user_owned_command_replacement_after_verification_executes_private_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    definition = _definition(tmp_path, _STDERR_SERVER)
    launched: list[Path] = []

    def replace_after_materialization(
        _workspace: str,
        _runtime: Path,
        command: list[str],
        **_kwargs: Any,
    ) -> tuple[list[str], dict[str, str]]:
        launched.append(Path(command[0]))
        source = Path(definition.command)
        source.write_text("#!/bin/sh\nexit 91\n", encoding="utf-8")
        source.chmod(0o700)
        return command, dict(os.environ)

    monkeypatch.setattr(
        mcp_module,
        "mcp_stdio_sandbox_launch",
        replace_after_materialization,
    )
    connection = StdioMcpConnection(definition)
    connection.start()
    assert len(launched) == 1
    materialized = launched[0]
    assert materialized != Path(definition.command)
    assert materialized.is_file()
    assert materialized.stat().st_mode & 0o777 == 0o500
    deadline = time.monotonic() + 2
    connection.initialize(
        deadline_monotonic=deadline,
        cancellation_token=CancellationToken(),
    )
    connection.list_tools(
        deadline_monotonic=deadline,
        cancellation_token=CancellationToken(),
    )
    connection.close()
    assert not materialized.exists()


def test_root_anchored_fast_path_rejects_acl_visible_write_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = Path("/usr/bin/python3").resolve(strict=True)
    metadata = command.lstat()
    if metadata.st_uid != 0:
        pytest.skip("requires a root-owned system executable")
    real_access = os.access

    def writable(path: os.PathLike[str] | str, mode: int, **kwargs: Any) -> bool:
        if Path(path) == command and mode == os.W_OK:
            return True
        return real_access(path, mode, **kwargs)

    monkeypatch.setattr(mcp_module.os, "access", writable)
    assert mcp_module._is_root_anchored_system_path(command, metadata) is False


@pytest.mark.parametrize(
    "environment_name",
    [
        "BASH_ENV",
        "DYLD_INSERT_LIBRARIES",
        "LD_PRELOAD",
        "PYTHONPATH",
        "PYTHONSTARTUP",
        "NODE_OPTIONS",
        "NODE_PATH",
        "RUBYOPT",
        "PERL5OPT",
        "JAVA_TOOL_OPTIONS",
        "_JAVA_OPTIONS",
        "CLASSPATH",
        "DOTNET_STARTUP_HOOKS",
        "CORECLR_ENABLE_PROFILING",
    ],
)
def test_mcp_pass_env_rejects_runtime_code_loading_controls(
    tmp_path: Path,
    environment_name: str,
) -> None:
    _definition(tmp_path, "#!/bin/sh\nexit 0\n")
    config_path = tmp_path / "workspace" / ".agent-harness" / "mcp.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["servers"]["fixture"]["pass_env"] = [environment_name]
    config_path.write_text(json.dumps(config), encoding="utf-8")
    config_path.chmod(0o600)

    with pytest.raises(McpLoadError, match="code-loading"):
        load_project_mcp(tmp_path / "workspace")


def test_mcp_pass_env_still_allows_explicit_business_values(tmp_path: Path) -> None:
    _definition(tmp_path, "#!/bin/sh\nexit 0\n")
    config_path = tmp_path / "workspace" / ".agent-harness" / "mcp.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["servers"]["fixture"]["pass_env"] = ["BUSINESS_API_TOKEN"]
    config_path.write_text(json.dumps(config), encoding="utf-8")
    config_path.chmod(0o600)

    definition = load_project_mcp(tmp_path / "workspace").server("fixture")
    assert definition.pass_env == ("BUSINESS_API_TOKEN",)


def test_stdout_banner_fails_as_invalid_protocol_frame(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    definition = _definition(
        tmp_path,
        "#!/usr/bin/env python3\nimport sys, time\nprint('not-json', flush=True)\ntime.sleep(2)\n",
    )

    def direct_launch(
        _workspace: str,
        _runtime: Path,
        command: list[str],
        **_kwargs: Any,
    ) -> tuple[list[str], dict[str, str]]:
        return command, dict(os.environ)

    monkeypatch.setattr(mcp_module, "mcp_stdio_sandbox_launch", direct_launch)
    connection = StdioMcpConnection(definition)
    connection.start()
    with pytest.raises(McpProtocolError) as captured:
        connection.initialize(
            deadline_monotonic=time.monotonic() + 1,
            cancellation_token=CancellationToken(),
        )
    assert captured.value.code == "mcp_frame_invalid"
    connection.close()


def test_server_that_does_not_read_stdin_cannot_escape_write_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    definition = _definition(
        tmp_path,
        "#!/usr/bin/env python3\nimport time\ntime.sleep(3)\n",
    )

    def direct_launch(
        _workspace: str,
        _runtime: Path,
        command: list[str],
        **_kwargs: Any,
    ) -> tuple[list[str], dict[str, str]]:
        return command, dict(os.environ)

    monkeypatch.setattr(mcp_module, "mcp_stdio_sandbox_launch", direct_launch)
    connection = StdioMcpConnection(definition)
    connection.start()
    connection._state = "ready"
    connection._initialized = True
    started = time.monotonic()
    try:
        with pytest.raises(McpProtocolError) as captured:
            connection.request(
                "tools/call",
                {"name": "Echo", "arguments": {"text": "x" * 900_000}},
                deadline_monotonic=started + 0.2,
                cancellation_token=CancellationToken(),
            )
        assert captured.value.code == "mcp_timeout"
        assert time.monotonic() - started < 1.0
    finally:
        connection.close()
