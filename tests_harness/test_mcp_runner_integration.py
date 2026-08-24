from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agent_harness.cli import main as cli_main
import agent_harness.tui as tui_module
from agent_harness.core import (
    ApprovalPolicy,
    ApprovalRequest,
    HarnessModelResponse,
    ProviderCapabilities,
    ProviderErrorKind,
    ProviderFailure,
    ProviderModelSpec,
)
from agent_harness.core.events import canonical_sha256
from agent_harness.core.mcp_protocol import MCP_PROTOCOL_VERSION, McpToolDefinition
from agent_harness.mcp import MCP_CONFIG_SCHEMA, McpTrustRequired, load_project_mcp
from agent_harness.runner import AgentRunner
from agent_harness.tui import HarnessTui


class _CaptureFinalModel:
    def __init__(self) -> None:
        self.requests: list[Any] = []

    @property
    def model_spec(self) -> ProviderModelSpec:
        return ProviderModelSpec(
            provider="test",
            model="test-v1",
            capabilities=ProviderCapabilities(
                provider="test",
                model="test-v1",
                structured_output=False,
                native_stream=False,
            ),
            context_window_tokens=8_192,
            maximum_output_tokens=1_024,
        )

    def plan(self, request: Any, **_kwargs: Any) -> HarnessModelResponse:
        self.requests.append(request)
        return HarnessModelResponse(kind="final", output={"message": "done"})

    def classify_error(self, _error: BaseException) -> ProviderFailure:
        return ProviderFailure(
            kind=ProviderErrorKind.UNKNOWN,
            retryable=False,
            safe_code="test_error",
        )


def _workspace(tmp_path: Path) -> tuple[Path, Path, Path]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config_directory = workspace / ".agent-harness"
    config_directory.mkdir(mode=0o700)
    config = {
        "schema": MCP_CONFIG_SCHEMA,
        "servers": {
            "fixture": {
                "transport": "stdio",
                "command": "/usr/bin/true",
                "args": [],
                "cwd": ".",
                "pass_env": [],
                "network_access": False,
                "allow_process_fork": False,
                "startup_timeout_seconds": 2,
                "tool_timeout_seconds": 2,
            }
        },
    }
    config_path = config_directory / "mcp.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    config_path.chmod(0o600)
    key = tmp_path / "key"
    key.write_text("test-api-key", encoding="utf-8")
    key.chmod(0o600)
    return workspace, tmp_path / "state", key


def _runner(
    workspace: Path,
    state_home: Path,
    key: Path,
    *,
    permission_mode: str = "read-only",
) -> tuple[AgentRunner, _CaptureFinalModel]:
    runner = AgentRunner(
        workspace,
        state_home=state_home,
        api_key_file=key,
        permission_mode=permission_mode,
    )
    model = _CaptureFinalModel()
    runner.model = model  # type: ignore[assignment]
    return runner, model


def _trust_and_cache(runner: AgentRunner, *, action: str = "trusted") -> str:
    snapshot = load_project_mcp(runner.workspace)
    definition = snapshot.server("fixture")
    runner.store.set_mcp_trust(
        "fixture",
        definition.definition_sha256,
        action=action,
    )
    if action != "trusted":
        return ""
    tool = McpToolDefinition.from_mapping(
        "fixture",
        {
            "name": "Echo",
            "inputSchema": {
                "type": "object",
                "properties": {"text": {"type": "string", "maxLength": 100}},
                "required": ["text"],
                "additionalProperties": False,
            },
        },
    )
    tools = [tool.to_cache()]
    material = {
        "schema": "agent_harness.mcp_catalog.v1",
        "protocol_version": MCP_PROTOCOL_VERSION,
        "server_definition_sha256": definition.definition_sha256,
        "tools": tools,
        "rejected_tools": [],
    }
    runner.store.set_mcp_catalog(
        "fixture",
        {
            "schema": "agent_harness.mcp_catalog_record.v1",
            "definition_sha256": definition.definition_sha256,
            "protocol_version": MCP_PROTOCOL_VERSION,
            "catalog_sha256": canonical_sha256(material),
            "server_info_sha256": canonical_sha256(None),
            "instructions_sha256": canonical_sha256(None),
            "tools": tools,
            "rejected_tools": [],
        },
    )
    return tool.local_name


def test_unresolved_mcp_trust_blocks_before_provider_or_run_persistence(
    tmp_path: Path,
) -> None:
    workspace, state_home, key = _workspace(tmp_path)
    runner, model = _runner(workspace, state_home, key)
    session = runner.new_session()

    with pytest.raises(McpTrustRequired):
        runner.run_turn(session["session_id"], "inspect")

    assert model.requests == []
    stored = runner.store.load(session["session_id"])
    assert stored["messages"] == []
    assert stored["runs"] == []


def test_disabled_mcp_server_does_not_require_catalog_or_reach_model_tools(
    tmp_path: Path,
) -> None:
    workspace, state_home, key = _workspace(tmp_path)
    runner, model = _runner(workspace, state_home, key, permission_mode="full-access")
    _trust_and_cache(runner, action="disabled")
    session = runner.new_session()

    outcome = runner.run_turn(session["session_id"], "inspect")

    assert outcome.status == "completed"
    assert len(model.requests) == 1
    assert not any(str(item.get("name", "")).startswith("mcp.") for item in model.requests[0].tools)


def test_ready_mcp_tools_are_permission_filtered_and_projected_content_free(
    tmp_path: Path,
) -> None:
    workspace, state_home, key = _workspace(tmp_path)
    privileged, privileged_model = _runner(
        workspace,
        state_home,
        key,
        permission_mode="full-access",
    )
    local_name = _trust_and_cache(privileged)
    privileged_session = privileged.new_session()

    outcome = privileged.run_turn(privileged_session["session_id"], "inspect")

    assert local_name in {str(item["name"]) for item in privileged_model.requests[0].tools}
    policy = privileged_model.requests[0].context["mcp_snapshot"]
    assert policy["servers"][0]["server_id"] == "fixture"
    assert "Echo" not in json.dumps(policy, ensure_ascii=False)
    started = next(
        event for event in outcome.result["events"] if event["type"] == "run.started"
    )
    assert started["payload"]["mcp_sha256"] == canonical_sha256(policy)
    assert started["payload"]["mcp_server_count"] == 1
    assert started["payload"]["mcp_tool_count"] == 1

    restricted, restricted_model = _runner(workspace, state_home, key)
    restricted_session = restricted.new_session()
    restricted.run_turn(restricted_session["session_id"], "inspect")
    assert not any(
        str(item.get("name", "")).startswith("mcp.")
        for item in restricted_model.requests[0].tools
    )


def test_mcp_cli_json_status_and_tui_command_are_provider_free(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    workspace, state_home, _key = _workspace(tmp_path)
    code = cli_main(
        [
            "--cwd",
            str(workspace),
            "--state-home",
            str(state_home),
            "mcp",
            "--json",
        ]
    )
    assert code == 0
    status = json.loads(capsys.readouterr().out)
    assert status["server_count"] == 1
    assert len(status["servers"][0]["definition_sha256"]) == 64

    application = HarnessTui(
        SimpleNamespace(mcp_status=lambda: status),  # type: ignore[arg-type]
        {"session_id": "session_12345678", "messages": []},
    )
    application.command("/mcp")
    rendered = "\n".join(application.state.notices)
    assert "项目 MCP：1 configured" in rendered
    assert "fixture · untrusted/not_active" in rendered


def test_tui_refuses_persistent_keys_for_once_only_mcp_approval() -> None:
    request = ApprovalRequest.for_call(
        approval_id="approval_mcp",
        run_id="run_mcp",
        call_id="call_mcp",
        tool_name="mcp.fixture.echo.1234567890abcdef",
        tool_version="a" * 64,
        arguments={"text": "hello"},
        policy=ApprovalPolicy(),
        risk="high",
        persistent_scope_allowed=False,
    )

    class Store:
        def add_approval_rule(self, *_args: Any, **_kwargs: Any) -> None:
            raise AssertionError("once-only MCP approval must not be persisted")

    application = HarnessTui(
        SimpleNamespace(store=Store()),  # type: ignore[arg-type]
        {"session_id": "session_12345678", "messages": []},
    )
    pending = tui_module._PendingApproval(request, "MCP tool arguments: text=hello")
    application.pending_approval = pending
    application.approval_preview_visible = True

    application.handle_key("s")

    assert pending.event.is_set() is False
    assert application.pending_approval is pending
    assert any("此工具要求逐次审批" in notice for notice in application.state.notices)
