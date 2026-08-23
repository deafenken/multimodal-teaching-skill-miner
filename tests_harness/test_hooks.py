from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any, Mapping

import pytest
import agent_harness.hooks as hooks_module

from agent_harness.cli import main as cli_main
from agent_harness.core import (
    ApprovalPolicy,
    ApprovalRule,
    CancellationToken,
    HarnessCheckpoint,
    HarnessContractError,
    HarnessJournal,
    HarnessLimits,
    HarnessModelResponse,
    ProviderCapabilities,
    ProviderModelSpec,
    RetryPolicy,
    SystemClock,
    ToolCall,
    ToolExecutionError,
    ToolHookDecision,
    ToolHookRequest,
    ToolRegistry,
    ToolSpec,
    public_harness_trace,
    resume_agent_harness,
    run_agent_harness,
)
from agent_harness.core.contracts import HARNESS_SCHEMA
from agent_harness.core.events import HarnessEventEmitter, canonical_sha256
from agent_harness.core.tools import execute_tool_call
from agent_harness.hooks import (
    HOOK_CONFIG_SCHEMA,
    HookLoadError,
    HookTrustRequired,
    TrustedHookRunner,
    load_project_hooks,
)
from agent_harness.session import SessionStore, SessionStoreError
from agent_harness.toolsets import workspace_sandbox_status


_HOOK_SECRET = "HOOK_PRIVATE_OUTPUT_7c49f6"
_TOOL_SECRET = "TOOL_PRIVATE_ARGUMENT_921cf4"


def _hook_record(
    hook_id: str = "guard",
    *,
    matcher: list[str] | None = None,
    entrypoint: str = ".agent-harness/hooks/guard.sh",
    **extra: Any,
) -> dict[str, Any]:
    return {
        "id": hook_id,
        "matcher": matcher or ["test.echo"],
        "entrypoint": entrypoint,
        "timeout_seconds": 2.0,
        **extra,
    }


def _hook_config(
    *,
    pre: list[Mapping[str, Any]] | None = None,
    post: list[Mapping[str, Any]] | None = None,
    failure: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "schema": HOOK_CONFIG_SCHEMA,
        "hooks": {
            "PreToolUse": list(pre or []),
            "PostToolUse": list(post or []),
            "PostToolUseFailure": list(failure or []),
        },
    }


def _write_hook_workspace(
    root: Path,
    *,
    config: Mapping[str, Any] | None = None,
    script: bytes | None = None,
) -> Path:
    workspace = root / "workspace"
    hooks_directory = workspace / ".agent-harness" / "hooks"
    hooks_directory.mkdir(parents=True)
    os.chmod(workspace / ".agent-harness", 0o700)
    os.chmod(hooks_directory, 0o700)
    entrypoint = hooks_directory / "guard.sh"
    entrypoint.write_bytes(
        script
        or (
            b"#!/bin/sh\n"
            b"printf '%s\\n' "
            b"'{\"schema\":\"agent_harness.hook_output.v1\","
            b"\"decision\":\"pass\"}'\n"
        )
    )
    os.chmod(entrypoint, 0o700)
    config_path = workspace / ".agent-harness" / "hooks.json"
    config_path.write_text(
        json.dumps(
            dict(config or _hook_config(pre=[_hook_record()])),
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    os.chmod(config_path, 0o600)
    return workspace


def _event_payload(request: ToolHookRequest, *, invocation_id: str) -> dict[str, Any]:
    return {
        "invocation_id": invocation_id,
        "hook_id": "guard",
        "hook_event_name": request.event_name,
        "hook_sha256": "a" * 64,
        "input_sha256": canonical_sha256(request.to_input()),
        "call_id": request.call_id,
        "tool_name": request.tool_name,
        "ordinal": 1,
    }


class _RecordingHookBroker:
    def __init__(
        self,
        actions: Mapping[str, str] | None = None,
        *,
        definition_sha256: str = "a" * 64,
        trust_status: str = "trusted",
    ) -> None:
        self.actions = dict(actions or {})
        self.requests: list[ToolHookRequest] = []
        self._policy_material = {
            "schema": "agent_harness.hook_policy.v1",
            "snapshot_sha256": definition_sha256,
            "sandbox_protocol": "test-read-only-v1",
            "definitions": [
                {
                    "hook_id": "guard",
                    "event_name": "PreToolUse",
                    "definition_sha256": definition_sha256,
                    "trust_status": trust_status,
                }
            ],
        }

    @property
    def policy_material(self) -> Mapping[str, Any]:
        return deepcopy(self._policy_material)

    def matches(self, event_name: str, tool_name: str) -> bool:
        return tool_name == "test.echo" and event_name in self.actions

    def evaluate(
        self,
        request: ToolHookRequest,
        *,
        cancellation_token: CancellationToken,
        deadline_monotonic: float,
        audit_sink: Any,
    ) -> ToolHookDecision:
        del deadline_monotonic
        cancellation_token.raise_if_cancelled()
        self.requests.append(request)
        invocation_id = f"hook-invocation-{len(self.requests)}"
        common = _event_payload(request, invocation_id=invocation_id)
        audit_sink("hook.started", common)
        audit_sink("hook.effect_started", common)
        action = self.actions[request.event_name]
        audit_sink(
            "hook.completed",
            {
                **common,
                "duration_ms": 1,
                "action": action,
                "output_sha256": canonical_sha256(_HOOK_SECRET),
            },
        )
        return ToolHookDecision(
            action=action,  # type: ignore[arg-type]
            matched_hook_ids=("guard",),
            executed_hook_ids=("guard",),
        )


def _safe_registry(
    executions: list[dict[str, Any]],
    *,
    fail: bool = False,
) -> ToolRegistry:
    registry = ToolRegistry()

    def handler(arguments: Mapping[str, Any], _context: Any) -> Mapping[str, Any]:
        executions.append(dict(arguments))
        if fail:
            raise ToolExecutionError(
                f"normalized failure; {_TOOL_SECRET}",
                code="fixture_failed",
            )
        return {"value": arguments["value"]}

    registry.register(
        ToolSpec(
            name="test.echo",
            version="1.0.0",
            description="A deterministic hook-policy fixture.",
            input_schema={
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
                "additionalProperties": False,
            },
            output_schema={
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
                "additionalProperties": False,
            },
            permission="test.read",
            risk="low",
            replay_policy="safe",
            execution_isolation="trusted_inline",
            trusted_inline_reason="deterministic test-only hook fixture",
            retry_policy=RetryPolicy(max_attempts=1),
        ),
        handler,
    )
    return registry


def _execute(
    broker: _RecordingHookBroker,
    executions: list[dict[str, Any]],
    *,
    fail: bool = False,
    emitter: HarnessEventEmitter | None = None,
    approval_policy: ApprovalPolicy | None = None,
) -> tuple[Any, HarnessEventEmitter]:
    clock = SystemClock()
    resolved_emitter = emitter or HarnessEventEmitter(
        run_id="run-hooks",
        turn_id="turn-hooks",
    )
    result, _receipt = execute_tool_call(
        ToolCall(
            call_id="call-hooks",
            name="test.echo",
            arguments={"value": _TOOL_SECRET},
        ),
        registry=_safe_registry(executions, fail=fail),
        allowed_permissions={"test.read"},
        emitter=resolved_emitter,
        clock=clock,
        cancellation_token=CancellationToken(),
        run_deadline_monotonic=clock.monotonic() + 5.0,
        max_output_chars=12_000,
        idempotency_receipts={},
        approval_policy=approval_policy or ApprovalPolicy(),
        tool_hook_broker=broker,
    )
    return result, resolved_emitter


class _ToolThenFinalModel:
    @property
    def model_spec(self) -> ProviderModelSpec:
        return ProviderModelSpec(
            provider="hook-tests",
            model="tool-then-final-v1",
            capabilities=ProviderCapabilities(
                provider="hook-tests",
                model="tool-then-final-v1",
                structured_output=True,
                native_stream=False,
                cancellation=True,
            ),
            context_window_tokens=8_192,
            maximum_output_tokens=1_024,
        )

    def plan(self, request: Any, **_kwargs: Any) -> HarnessModelResponse:
        if request.observations:
            return HarnessModelResponse(kind="final", output={"message": "done"})
        return HarnessModelResponse(
            kind="tool_calls",
            tool_calls=(
                ToolCall(
                    call_id="call-hooks",
                    name="test.echo",
                    arguments={"value": _TOOL_SECRET},
                ),
            ),
        )


class _FinalModel(_ToolThenFinalModel):
    def plan(self, _request: Any, **_kwargs: Any) -> HarnessModelResponse:
        return HarnessModelResponse(kind="final", output={"message": "done"})


def test_hook_config_is_strict_digest_bound_and_content_free(tmp_path: Path) -> None:
    script = b"#!/bin/sh\n# " + _HOOK_SECRET.encode() + b"\nexit 0\n"
    workspace = _write_hook_workspace(tmp_path, script=script)

    first = load_project_hooks(workspace)
    second = load_project_hooks(workspace)
    definition = first.definitions[0]

    assert first.snapshot_sha256 == second.snapshot_sha256
    assert definition.entrypoint_sha256 == sha256(script).hexdigest()
    assert definition.definition_sha256 == second.definitions[0].definition_sha256
    rendered_metadata = json.dumps(first.metadata({}), sort_keys=True)
    assert _HOOK_SECRET not in rendered_metadata
    assert script.decode() not in rendered_metadata


def test_exact_config_bytes_and_entrypoint_bytes_invalidate_trust(tmp_path: Path) -> None:
    workspace = _write_hook_workspace(tmp_path)
    first = load_project_hooks(workspace)
    trusted = {
        "guard": {
            "action": "trusted",
            "definition_sha256": first.definitions[0].definition_sha256,
        }
    }
    assert first.statuses(trusted) == {"guard": "trusted"}

    config_path = workspace / ".agent-harness" / "hooks.json"
    config_path.write_text(config_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    os.chmod(config_path, 0o600)
    config_changed = load_project_hooks(workspace)
    assert config_changed.config_sha256 != first.config_sha256
    assert config_changed.definitions[0].definition_sha256 != first.definitions[0].definition_sha256
    assert config_changed.statuses(trusted) == {"guard": "modified"}

    entrypoint = workspace / ".agent-harness" / "hooks" / "guard.sh"
    entrypoint.write_bytes(entrypoint.read_bytes() + b"# changed\n")
    os.chmod(entrypoint, 0o700)
    script_changed = load_project_hooks(workspace)
    assert (
        script_changed.definitions[0].entrypoint_sha256
        != first.definitions[0].entrypoint_sha256
    )
    assert script_changed.statuses(trusted) == {"guard": "modified"}


@pytest.mark.parametrize(
    "config",
    [
        {
            "schema": HOOK_CONFIG_SCHEMA,
            "hooks": {"BeforeEverything": [_hook_record()]},
        },
        _hook_config(pre=[_hook_record(extra_field=True)]),
        _hook_config(pre=[_hook_record(matcher=["Not A Tool"])]),
        _hook_config(
            pre=[_hook_record("duplicate")],
            post=[_hook_record("duplicate")],
        ),
        _hook_config(
            pre=[
                _hook_record(
                    entrypoint=".agent-harness/hooks/../outside.sh"
                )
            ]
        ),
    ],
)
def test_invalid_hook_configuration_fails_closed(
    tmp_path: Path,
    config: Mapping[str, Any],
) -> None:
    workspace = _write_hook_workspace(tmp_path, config=config)
    with pytest.raises(HookLoadError):
        load_project_hooks(workspace)


def test_duplicate_json_keys_and_oversized_config_fail_closed(tmp_path: Path) -> None:
    workspace = _write_hook_workspace(tmp_path)
    config_path = workspace / ".agent-harness" / "hooks.json"
    config_path.write_text(
        '{"schema":"agent_harness.hooks.v1",'
        '"schema":"agent_harness.hooks.v1","hooks":{}}',
        encoding="utf-8",
    )
    with pytest.raises(HookLoadError):
        load_project_hooks(workspace)

    config_path.write_bytes(b" " * (64 * 1024 + 1))
    os.chmod(config_path, 0o600)
    with pytest.raises(HookLoadError):
        load_project_hooks(workspace)


def test_symlinks_hardlinks_permissions_and_non_executable_files_are_rejected(
    tmp_path: Path,
) -> None:
    workspace = _write_hook_workspace(tmp_path)
    config_path = workspace / ".agent-harness" / "hooks.json"
    entrypoint = workspace / ".agent-harness" / "hooks" / "guard.sh"

    config_copy = tmp_path / "config-copy.json"
    config_copy.write_bytes(config_path.read_bytes())
    config_path.unlink()
    config_path.symlink_to(config_copy)
    with pytest.raises(HookLoadError):
        load_project_hooks(workspace)

    config_path.unlink()
    config_path.write_bytes(config_copy.read_bytes())
    os.chmod(config_path, 0o622)
    with pytest.raises(HookLoadError):
        load_project_hooks(workspace)

    os.chmod(config_path, 0o600)
    entrypoint.chmod(0o600)
    with pytest.raises(HookLoadError):
        load_project_hooks(workspace)

    entrypoint.chmod(0o700)
    hardlink = tmp_path / "guard-hardlink.sh"
    os.link(entrypoint, hardlink)
    with pytest.raises(HookLoadError):
        load_project_hooks(workspace)


def test_workspace_and_intermediate_directory_symlinks_are_rejected(tmp_path: Path) -> None:
    workspace = _write_hook_workspace(tmp_path)
    workspace_link = tmp_path / "workspace-link"
    workspace_link.symlink_to(workspace, target_is_directory=True)
    with pytest.raises(HookLoadError):
        load_project_hooks(workspace_link)

    agent_directory = workspace / ".agent-harness"
    moved = tmp_path / "agent-harness-real"
    agent_directory.rename(moved)
    agent_directory.symlink_to(moved, target_is_directory=True)
    with pytest.raises(HookLoadError):
        load_project_hooks(workspace)


def test_hook_trust_is_exact_private_revocable_and_outside_workspace(
    tmp_path: Path,
) -> None:
    workspace = _write_hook_workspace(tmp_path)
    snapshot = load_project_hooks(workspace)
    digest = snapshot.definitions[0].definition_sha256
    store = SessionStore(workspace, state_home=tmp_path / "state")

    store.set_hook_trust("guard", digest, action="trusted")
    assert store.hook_trust_state() == {
        "guard": {"action": "trusted", "definition_sha256": digest}
    }
    assert store.hook_trust_path.stat().st_mode & 0o777 == 0o600
    TrustedHookRunner(snapshot, store.hook_trust_state())

    store.set_hook_trust("guard", digest, action="disabled")
    disabled = TrustedHookRunner(snapshot, store.hook_trust_state())
    assert not disabled.matches("PreToolUse", "test.echo")
    assert store.revoke_hook_trust("guard") is True
    assert store.revoke_hook_trust("guard") is False
    with pytest.raises(HookTrustRequired):
        TrustedHookRunner(snapshot, store.hook_trust_state())

    in_workspace = SessionStore(workspace, state_home=workspace / "state")
    with pytest.raises(SessionStoreError, match="outside the workspace"):
        in_workspace.set_hook_trust("guard", digest, action="trusted")


def test_trusted_hooks_have_no_unsupported_host_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _write_hook_workspace(tmp_path)
    snapshot = load_project_hooks(workspace)
    definition = snapshot.definitions[0]
    monkeypatch.setattr(
        hooks_module,
        "workspace_sandbox_status",
        lambda: {
            "available": False,
            "backend": "unavailable",
            "network_denied": False,
        },
    )
    with pytest.raises(HookLoadError, match="Seatbelt"):
        TrustedHookRunner(
            snapshot,
            {
                definition.hook_id: {
                    "action": "trusted",
                    "definition_sha256": definition.definition_sha256,
                }
            },
        )

    disabled = TrustedHookRunner(
        snapshot,
        {
            definition.hook_id: {
                "action": "disabled",
                "definition_sha256": definition.definition_sha256,
            }
        },
    )
    assert not disabled.matches("PreToolUse", "test.echo")


@pytest.mark.skipif(
    not workspace_sandbox_status()["available"],
    reason="trusted command hooks require macOS Seatbelt",
)
def test_trusted_hook_executes_snapshotted_bytes_in_read_only_sandbox(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = (
        b"#!/bin/sh\n"
        b"decision=pass\n"
        b"if printf hacked 2>/dev/null > \"$PWD/hook-sentinel\"; then\n"
        b"  decision=deny\n"
        b"fi\n"
        b"if read git_secret 2>/dev/null < \"$PWD/.git/config\"; then\n"
        b"  decision=deny\n"
        b"fi\n"
        b"if [ -n \"${DEEPSEEK_API_KEY-}\" ]; then decision=deny; fi\n"
        b"printf '%s\\n' \"{\\\"schema\\\":"
        b"\\\"agent_harness.hook_output.v1\\\","
        b"\\\"decision\\\":\\\"$decision\\\"}\"\n"
    )
    workspace = _write_hook_workspace(tmp_path, script=script)
    (workspace / ".git").mkdir()
    (workspace / ".git" / "config").write_text(
        "credential = must-not-be-readable\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "must-not-be-inherited")
    snapshot = load_project_hooks(workspace)
    definition = snapshot.definitions[0]
    runner = TrustedHookRunner(
        snapshot,
        {
            definition.hook_id: {
                "action": "trusted",
                "definition_sha256": definition.definition_sha256,
            }
        },
    )
    # Mutating the project path after discovery must not change the bytes that
    # this frozen runner materializes and executes.
    entrypoint = workspace / definition.entrypoint
    entrypoint.write_text("#!/bin/sh\nexit 2\n", encoding="utf-8")
    entrypoint.chmod(0o700)
    request = ToolHookRequest(
        event_name="PreToolUse",
        run_id="run-hooks",
        turn_id="turn-hooks",
        session_id="session-hooks",
        call_id="call-hooks",
        tool_name="test.echo",
        tool_version="1.0.0",
        tool_input={"value": "safe"},
        arguments_sha256=canonical_sha256({"value": "safe"}),
    )
    events: list[tuple[str, Mapping[str, Any]]] = []
    cancellation = CancellationToken()
    decision = runner.evaluate(
        request,
        cancellation_token=cancellation,
        deadline_monotonic=SystemClock().monotonic() + 5.0,
        audit_sink=lambda event_type, payload: events.append(
            (event_type, dict(payload))
        ),
    )

    assert decision.action == "pass"
    assert not (workspace / "hook-sentinel").exists()
    assert [event_type for event_type, _payload in events] == [
        "hook.started",
        "hook.effect_started",
        "hook.completed",
    ]


def test_hook_trust_file_symlink_and_broad_permissions_fail_closed(tmp_path: Path) -> None:
    workspace = _write_hook_workspace(tmp_path)
    snapshot = load_project_hooks(workspace)
    digest = snapshot.definitions[0].definition_sha256
    store = SessionStore(workspace, state_home=tmp_path / "state")
    store.set_hook_trust("guard", digest, action="trusted")

    store.hook_trust_path.chmod(0o644)
    with pytest.raises(SessionStoreError):
        store.hook_trust_state()

    outside = tmp_path / "outside-trust.json"
    outside.write_text(
        json.dumps({"schema": "agent_harness.hook_trust.v1", "hooks": {}}),
        encoding="utf-8",
    )
    outside.chmod(0o600)
    store.hook_trust_path.unlink()
    store.hook_trust_path.symlink_to(outside)
    with pytest.raises(SessionStoreError):
        store.hook_trust_state()


def test_hook_request_snapshots_and_binds_every_tool_lifecycle_input() -> None:
    mutable = {"nested": {"value": 1}}
    digest = canonical_sha256(mutable)
    request = ToolHookRequest(
        event_name="PreToolUse",
        run_id="run-hooks",
        turn_id="turn-hooks",
        session_id="session-hooks",
        call_id="call-hooks",
        tool_name="test.echo",
        tool_version="1.0.0",
        tool_input=mutable,
        arguments_sha256=digest,
    )
    mutable["nested"]["value"] = 2
    assert request.tool_input == {"nested": {"value": 1}}
    assert request.to_input()["arguments_sha256"] == digest

    with pytest.raises(HarnessContractError, match="digest"):
        ToolHookRequest(
            event_name="PreToolUse",
            run_id="run-hooks",
            turn_id="turn-hooks",
            session_id=None,
            call_id="call-hooks",
            tool_name="test.echo",
            tool_version="1.0.0",
            tool_input={"value": 1},
            arguments_sha256="0" * 64,
        )
    with pytest.raises(HarnessContractError, match="cannot contain"):
        ToolHookRequest(
            event_name="PreToolUse",
            run_id="run-hooks",
            turn_id="turn-hooks",
            session_id=None,
            call_id="call-hooks",
            tool_name="test.echo",
            tool_version="1.0.0",
            tool_input={},
            arguments_sha256=canonical_sha256({}),
            result={"unexpected": True},
        )


def test_hook_decisions_cannot_claim_unmatched_or_both_executed_and_skipped() -> None:
    with pytest.raises(HarnessContractError, match="invalid"):
        ToolHookDecision(
            action="allow",  # type: ignore[arg-type]
            matched_hook_ids=("guard",),
        )
    with pytest.raises(HarnessContractError, match="inconsistent"):
        ToolHookDecision(
            matched_hook_ids=("guard",),
            executed_hook_ids=("guard",),
            skipped_hook_ids=("guard",),
        )


def test_pre_hook_deny_prevents_tool_start_and_handler_execution() -> None:
    executions: list[dict[str, Any]] = []
    result, emitter = _execute(
        _RecordingHookBroker({"PreToolUse": "deny"}),
        executions,
    )

    assert result.ok is False
    assert result.error_code == "hook_denied"
    assert executions == []
    assert [event["type"] for event in emitter.events] == [
        "tool.requested",
        "hook.started",
        "hook.effect_started",
        "hook.completed",
        "tool.rejected",
    ]


def test_pre_hook_ask_forces_approval_for_an_otherwise_low_risk_tool() -> None:
    executions: list[dict[str, Any]] = []
    result, emitter = _execute(
        _RecordingHookBroker({"PreToolUse": "ask"}),
        executions,
    )

    assert result.ok is False
    assert result.error_code == "approval_required"
    assert executions == []
    event_types = [event["type"] for event in emitter.events]
    assert event_types[-3:] == [
        "approval.requested",
        "approval.resolved",
        "tool.rejected",
    ]
    requested = next(
        event for event in emitter.events if event["type"] == "approval.requested"
    )
    assert requested["payload"]["policy_action"] == "ask"
    assert requested["payload"]["hook_action"] == "ask"
    assert requested["payload"]["persistent_scope_allowed"] is False


def test_hook_ask_cannot_be_bypassed_by_an_existing_exact_allow() -> None:
    executions: list[dict[str, Any]] = []
    arguments = {"value": _TOOL_SECRET}
    policy = ApprovalPolicy(
        rules=(
            ApprovalRule(
                action="allow",
                tool_name="test.echo",
                tool_version="1.0.0",
                arguments_sha256=canonical_sha256(arguments),
            ),
        )
    )
    result, emitter = _execute(
        _RecordingHookBroker({"PreToolUse": "ask"}),
        executions,
        approval_policy=policy,
    )

    assert result.error_code == "approval_required"
    assert executions == []
    requested = next(
        event for event in emitter.events if event["type"] == "approval.requested"
    )
    assert requested["payload"]["policy_action"] == "ask"


def test_hook_pass_or_ask_cannot_weaken_a_base_deny() -> None:
    arguments = {"value": _TOOL_SECRET}
    policy = ApprovalPolicy(
        rules=(
            ApprovalRule(
                action="deny",
                tool_name="test.echo",
                tool_version="1.0.0",
                arguments_sha256=canonical_sha256(arguments),
            ),
        )
    )
    for hook_action in ("pass", "ask"):
        executions: list[dict[str, Any]] = []
        result, _emitter = _execute(
            _RecordingHookBroker({"PreToolUse": hook_action}),
            executions,
            approval_policy=policy,
        )
        assert result.error_code == "approval_denied"
        assert executions == []


def test_post_hook_observes_success_before_settlement_without_rewriting_it() -> None:
    executions: list[dict[str, Any]] = []
    broker = _RecordingHookBroker({"PostToolUse": "pass"})
    result, emitter = _execute(broker, executions)

    assert result.ok is True
    assert result.result == {"value": _TOOL_SECRET}
    assert executions == [{"value": _TOOL_SECRET}]
    assert broker.requests[0].event_name == "PostToolUse"
    assert broker.requests[0].result == result.result
    assert broker.requests[0].result_sha256 == result.result_sha256
    assert [event["type"] for event in emitter.events] == [
        "tool.requested",
        "tool.started",
        "hook.started",
        "hook.effect_started",
        "hook.completed",
        "tool.completed",
    ]


def test_post_hook_cannot_change_a_completed_tool_settlement() -> None:
    executions: list[dict[str, Any]] = []
    result, emitter = _execute(
        _RecordingHookBroker({"PostToolUse": "deny"}),
        executions,
    )

    assert executions == [{"value": _TOOL_SECRET}]
    assert result.ok is True
    assert result.result == {"value": _TOOL_SECRET}
    assert emitter.events[-1]["type"] == "tool.completed"
    assert not any(event["type"] == "tool.failed" for event in emitter.events)


def test_post_failure_hook_observes_normalized_error_before_tool_failed() -> None:
    executions: list[dict[str, Any]] = []
    broker = _RecordingHookBroker({"PostToolUseFailure": "pass"})
    result, emitter = _execute(broker, executions, fail=True)

    assert result.ok is False
    assert result.error_code == "fixture_failed"
    assert broker.requests[0].event_name == "PostToolUseFailure"
    assert broker.requests[0].error_code == "fixture_failed"
    assert [event["type"] for event in emitter.events][-4:] == [
        "hook.started",
        "hook.effect_started",
        "hook.completed",
        "tool.failed",
    ]


def test_hook_events_are_durable_ordered_and_public_trace_is_content_free(
    tmp_path: Path,
) -> None:
    journal = HarnessJournal(
        tmp_path / "events.jsonl",
        run_id="run-hooks",
        turn_id="turn-hooks",
    )
    emitter = HarnessEventEmitter(
        run_id=journal.run_id,
        turn_id=journal.turn_id,
        durable_sink=journal.append,
    )
    emitter.emit("run.started", {"resumed": False})
    executions: list[dict[str, Any]] = []
    result, _ = _execute(
        _RecordingHookBroker({"PostToolUse": "pass"}),
        executions,
        emitter=emitter,
    )
    assert result.ok
    emitter.emit("run.completed", {"reason_code": "completed", "duration_ms": 1})

    replayed = HarnessJournal(
        journal.path,
        run_id=journal.run_id,
        turn_id=journal.turn_id,
    ).replay()
    assert replayed == emitter.events
    assert [event["sequence"] for event in replayed] == list(
        range(1, len(replayed) + 1)
    )

    public = public_harness_trace(
        {
            "schema": HARNESS_SCHEMA,
            "run_id": journal.run_id,
            "turn_id": journal.turn_id,
            "status": "completed",
            "steps": 1,
            "model_call_count": 1,
            "tool_call_count": 1,
            "duration_ms": 1,
            "events": replayed,
        }
    )
    rendered = json.dumps(public, sort_keys=True)
    assert _TOOL_SECRET not in rendered
    assert _HOOK_SECRET not in rendered
    completed_hook = next(
        event for event in public["events"] if event["type"] == "hook.completed"
    )
    assert set(completed_hook["payload"]) >= {
        "hook_sha256",
        "input_sha256",
        "output_sha256",
        "action",
    }


def test_runtime_policy_hash_binds_hook_policy_even_without_an_invocation() -> None:
    first = run_agent_harness(
        _FinalModel(),
        ToolRegistry(),
        {"topic": "hooks"},
        allowed_permissions={"test.read"},
        tool_hook_broker=_RecordingHookBroker(definition_sha256="a" * 64),
    )
    second = run_agent_harness(
        _FinalModel(),
        ToolRegistry(),
        {"topic": "hooks"},
        allowed_permissions={"test.read"},
        tool_hook_broker=_RecordingHookBroker(definition_sha256="b" * 64),
    )

    assert first["checkpoint"].policy_sha256 != second["checkpoint"].policy_sha256
    first_started = first["events"][0]["payload"]
    second_started = second["events"][0]["payload"]
    assert first_started["hooks_sha256"] != second_started["hooks_sha256"]
    assert first_started["hook_count"] == 1
    assert first_started["trusted_hook_count"] == 1


def test_resume_rejects_changed_hook_policy_and_never_replays_hook_guarded_intent() -> None:
    checkpoints: list[HarnessCheckpoint] = []
    executions: list[dict[str, Any]] = []

    class SimulatedCrash(BaseException):
        pass

    def crash_on_pending(checkpoint: HarnessCheckpoint) -> None:
        checkpoints.append(checkpoint)
        if checkpoint.pending_effect is not None:
            raise SimulatedCrash()

    original = _RecordingHookBroker(
        {"PreToolUse": "pass"},
        definition_sha256="a" * 64,
    )
    with pytest.raises(SimulatedCrash):
        run_agent_harness(
            _ToolThenFinalModel(),
            _safe_registry(executions),
            {"topic": "hooks"},
            limits=HarnessLimits(deadline_seconds=5.0),
            retry_policy=RetryPolicy(max_attempts=1),
            allowed_permissions={"test.read"},
            checkpoint_sink=crash_on_pending,
            tool_hook_broker=original,
        )
    pending = next(
        checkpoint
        for checkpoint in reversed(checkpoints)
        if checkpoint.pending_effect is not None
    )
    assert pending.pending_effect["hook_guarded"] is True
    assert pending.pending_effect["replay_policy"] == "never"

    with pytest.raises(HarnessContractError, match="execution policy does not match"):
        resume_agent_harness(
            pending,
            _ToolThenFinalModel(),
            _safe_registry(executions),
            {"topic": "hooks"},
            limits=HarnessLimits(deadline_seconds=5.0),
            retry_policy=RetryPolicy(max_attempts=1),
            allowed_permissions={"test.read"},
            tool_hook_broker=_RecordingHookBroker(
                {"PreToolUse": "pass"},
                definition_sha256="b" * 64,
            ),
        )

    resumed = resume_agent_harness(
        pending,
        _ToolThenFinalModel(),
        _safe_registry(executions),
        {"topic": "hooks"},
        limits=HarnessLimits(deadline_seconds=5.0),
        retry_policy=RetryPolicy(max_attempts=1),
        allowed_permissions={"test.read"},
        tool_hook_broker=_RecordingHookBroker(
            {"PreToolUse": "pass"},
            definition_sha256="a" * 64,
        ),
    )
    assert resumed["status"] == "handoff"
    assert resumed["reason"] == "unsafe_hook_replay_blocked"
    assert executions == []


def test_cli_hooks_lists_trusts_disables_and_revokes_exact_definitions(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    workspace = _write_hook_workspace(tmp_path)
    state_home = tmp_path / "state"
    base = ["--cwd", str(workspace), "--state-home", str(state_home), "hooks"]

    assert cli_main([*base, "--json"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed["hooks"][0]["hook_id"] == "guard"
    assert listed["hooks"][0]["trust_status"] == "untrusted"
    digest = listed["hooks"][0]["definition_sha256"]

    assert cli_main([*base, "trust", "guard", "--sha256", "0" * 64]) == 78
    assert "does not match the current definition" in capsys.readouterr().err
    assert SessionStore(workspace, state_home=state_home).hook_trust_state() == {}

    assert cli_main([*base, "trust", "guard", "--sha256", digest]) == 0
    capsys.readouterr()
    store = SessionStore(workspace, state_home=state_home)
    assert store.hook_trust_state()["guard"] == {
        "action": "trusted",
        "definition_sha256": digest,
    }

    assert cli_main([*base, "revoke", "guard"]) == 0
    capsys.readouterr()
    assert store.hook_trust_state() == {}

    assert cli_main([*base, "disable", "guard", "--sha256", digest]) == 0
    capsys.readouterr()
    assert store.hook_trust_state()["guard"] == {
        "action": "disabled",
        "definition_sha256": digest,
    }
