from __future__ import annotations

from dataclasses import asdict, FrozenInstanceError, replace
import json
import unittest
from pathlib import Path
import tempfile

from agent_harness.core import (
    HarnessCheckpoint,
    HarnessLimits,
    HarnessModelResponse,
    ProviderCapabilities,
    ProviderModelSpec,
    RetryPolicy,
    ToolCall,
    ToolRegistry,
    ToolSpec,
    public_harness_trace,
    resume_agent_harness,
    run_agent_harness,
)
from agent_harness.core.approvals import (
    ApprovalContractError,
    ApprovalDecision,
    ApprovalPolicy,
    ApprovalRequest,
    ApprovalRule,
    HeadlessApprovalBroker,
    arguments_sha256,
    resolve_approval,
    validate_decision,
)
from agent_harness.session import SessionStore, SessionStoreError


class _ToolThenFinalModel:
    def __init__(self, arguments: dict[str, object]) -> None:
        self.arguments = arguments

    @property
    def model_spec(self) -> ProviderModelSpec:
        return ProviderModelSpec(
            provider="approval-tests",
            model="tool-then-final-v1",
            capabilities=ProviderCapabilities(
                provider="approval-tests",
                model="tool-then-final-v1",
                structured_output=True,
                native_stream=False,
                cancellation=True,
            ),
            context_window_tokens=8_192,
            maximum_output_tokens=1_024,
        )

    def plan(self, request, **_kwargs):
        if not request.observations:
            return HarnessModelResponse(
                kind="tool_calls",
                tool_calls=(
                    ToolCall(
                        call_id="call-sensitive-1",
                        name="danger.exec",
                        arguments=self.arguments,
                    ),
                ),
            )
        return HarnessModelResponse(kind="final", output={"message": "done"})


class _AllowOnceBroker:
    def decide(self, request, *, preview=""):
        self.preview = preview
        return ApprovalDecision.for_request(
            request,
            verdict="allow",
            reason_code="user_allowed_once",
        )


def _effectful_registry(executions: list[dict[str, object]]) -> ToolRegistry:
    registry = ToolRegistry()

    def handler(arguments, context):
        context.begin_effect()
        executions.append(dict(arguments))
        return {"ok": True}

    registry.register(
        ToolSpec(
            name="danger.exec",
            version="1",
            description="Exercise the approval boundary.",
            input_schema={
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
                "additionalProperties": False,
            },
            permission="danger.exec",
            risk="high",
            replay_policy="never",
            execution_isolation="trusted_inline",
            trusted_inline_reason="Test-only effect handler behind approval boundary",
            retry_policy=RetryPolicy(max_attempts=1),
        ),
        handler,
    )
    return registry


def _safe_high_risk_registry(
    executions: list[dict[str, object]],
) -> ToolRegistry:
    registry = ToolRegistry()

    def handler(arguments, _context):
        executions.append(dict(arguments))
        return {"ok": True}

    registry.register(
        ToolSpec(
            name="danger.exec",
            version="1",
            description="Exercise a replay-safe approval boundary.",
            input_schema={
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
                "additionalProperties": False,
            },
            permission="danger.exec",
            risk="high",
            replay_policy="safe",
            execution_isolation="trusted_inline",
            trusted_inline_reason="Test-only deterministic approval fixture",
            retry_policy=RetryPolicy(max_attempts=1),
        ),
        handler,
    )
    return registry


def _request(
    policy: ApprovalPolicy,
    *,
    risk: str = "high",
    call_id: str = "call-1",
    arguments: dict[str, object] | None = None,
) -> ApprovalRequest:
    return ApprovalRequest.for_call(
        approval_id="approval-1",
        run_id="run-1",
        call_id=call_id,
        tool_name="process.exec",
        tool_version="1.0.0",
        arguments=arguments or {"command": ["git", "status"], "timeout": 10},
        policy=policy,
        risk=risk,
    )


class ApprovalCoreTests(unittest.TestCase):
    def test_arguments_digest_is_canonical_and_raw_arguments_are_not_retained(self) -> None:
        first = {"nested": {"b": 2, "a": 1}, "command": ["git", "status"]}
        second = {"command": ["git", "status"], "nested": {"a": 1, "b": 2}}
        self.assertEqual(arguments_sha256(first), arguments_sha256(second))

        policy = ApprovalPolicy()
        request = _request(policy, arguments={"secret": "must-not-persist"})
        serialized = json.dumps(asdict(request), sort_keys=True)
        self.assertNotIn("must-not-persist", serialized)
        self.assertNotIn("secret", serialized)

    def test_policy_digest_is_stable_across_rule_order(self) -> None:
        digest = arguments_sha256({"command": "git status"})
        allow = ApprovalRule(action="allow", tool_name="workspace.read")
        ask = ApprovalRule(
            action="ask",
            tool_name="process.exec",
            tool_version="1.0.0",
            arguments_sha256=digest,
        )
        self.assertEqual(
            ApprovalPolicy(rules=(allow, ask)).policy_sha256,
            ApprovalPolicy(rules=(ask, allow)).policy_sha256,
        )

    def test_deny_then_ask_then_allow_precedence_is_order_independent(self) -> None:
        digest = arguments_sha256({"command": ["git", "status"]})
        rules = (
            ApprovalRule(action="allow", tool_name="process.exec"),
            ApprovalRule(
                action="ask",
                tool_name="process.exec",
                tool_version="1.0.0",
                arguments_sha256=digest,
            ),
            ApprovalRule(action="deny", tool_name="*"),
        )
        for policy in (ApprovalPolicy(rules=rules), ApprovalPolicy(rules=rules[::-1])):
            self.assertEqual(
                policy.action_for(
                    tool_name="process.exec",
                    tool_version="1.0.0",
                    arguments_sha256=digest,
                    risk="low",
                ),
                "deny",
            )

        without_deny = ApprovalPolicy(rules=rules[:2])
        self.assertEqual(
            without_deny.action_for(
                tool_name="process.exec",
                tool_version="1.0.0",
                arguments_sha256=digest,
                risk="low",
            ),
            "ask",
        )

    def test_tool_wide_and_exact_rules_have_bounded_matching(self) -> None:
        exact_digest = arguments_sha256({"path": "safe.txt"})
        policy = ApprovalPolicy(
            rules=(
                ApprovalRule(action="allow", tool_name="workspace.write"),
                ApprovalRule(
                    action="deny",
                    tool_name="workspace.write",
                    tool_version="2.0.0",
                    arguments_sha256=exact_digest,
                ),
            )
        )
        self.assertEqual(
            policy.action_for(
                tool_name="workspace.write",
                tool_version="2.0.0",
                arguments_sha256=exact_digest,
                risk="high",
            ),
            "deny",
        )
        self.assertEqual(
            policy.action_for(
                tool_name="workspace.write",
                tool_version="2.0.0",
                arguments_sha256=arguments_sha256({"path": "other.txt"}),
                risk="high",
            ),
            "allow",
        )

    def test_default_policy_allows_low_and_asks_for_medium_and_high(self) -> None:
        policy = ApprovalPolicy()
        low = resolve_approval(_request(policy, risk="low"), policy)
        medium = resolve_approval(_request(policy, risk="medium"), policy)
        high = resolve_approval(_request(policy, risk="high"), policy)
        self.assertEqual((low.verdict, low.reason_code), ("allow", "approval_policy_allowed"))
        self.assertEqual(
            (medium.verdict, medium.reason_code),
            ("deny", "approval_unavailable"),
        )
        self.assertEqual(
            (high.verdict, high.reason_code),
            ("deny", "approval_unavailable"),
        )

    def test_stale_cross_call_and_wrong_digest_responses_fail_closed(self) -> None:
        policy = ApprovalPolicy()
        request = _request(policy)
        valid = ApprovalDecision.for_request(
            request,
            verdict="allow",
            reason_code="user_allowed_once",
        )
        invalid_decisions = (
            replace(valid, approval_id="approval-old"),
            replace(valid, run_id="run-other"),
            replace(valid, call_id="call-other"),
            replace(valid, tool_name="workspace.write"),
            replace(valid, tool_version="9.9.9"),
            replace(valid, arguments_sha256="f" * 64),
            replace(valid, policy_sha256="e" * 64),
        )
        for invalid in invalid_decisions:
            with self.assertRaises(ApprovalContractError):
                validate_decision(request, invalid)

            class InvalidBroker:
                def decide(self, _request, *, preview=""):
                    del preview
                    return invalid

            decision = resolve_approval(request, policy, InvalidBroker())
            self.assertEqual(decision.verdict, "deny")
            self.assertEqual(decision.reason_code, "approval_response_invalid")
            self.assertEqual(decision.binding(), request.binding())

    def test_changed_policy_and_broker_exceptions_fail_closed(self) -> None:
        requested_policy = ApprovalPolicy()
        request = _request(requested_policy)
        changed_policy = ApprovalPolicy(
            rules=(ApprovalRule(action="allow", tool_name="process.exec"),)
        )
        changed = resolve_approval(request, changed_policy)
        self.assertEqual(
            (changed.verdict, changed.reason_code),
            ("deny", "approval_policy_changed"),
        )

        class BrokenBroker:
            def decide(self, _request, *, preview=""):
                del preview
                raise RuntimeError("UI disconnected")

        failed = resolve_approval(request, requested_policy, BrokenBroker())
        self.assertEqual(
            (failed.verdict, failed.reason_code),
            ("deny", "approval_broker_failed"),
        )
        self.assertEqual(failed.binding(), request.binding())

    def test_valid_broker_allow_and_deny_decisions_are_preserved(self) -> None:
        policy = ApprovalPolicy()
        request = _request(policy)

        class BoundBroker:
            def __init__(self, verdict: str) -> None:
                self.verdict = verdict

            def decide(self, current, *, preview=""):
                del preview
                return ApprovalDecision.for_request(
                    current,
                    verdict=self.verdict,
                    reason_code={
                        "allow": "user_allowed_once",
                        "deny": "user_denied_once",
                    }[self.verdict],
                )

        allowed = resolve_approval(request, policy, BoundBroker("allow"))
        denied = resolve_approval(request, policy, BoundBroker("deny"))
        self.assertEqual(
            (allowed.verdict, allowed.reason_code),
            ("allow", "user_allowed_once"),
        )
        self.assertEqual(
            (denied.verdict, denied.reason_code),
            ("deny", "user_denied_once"),
        )

    def test_headless_broker_and_policy_deny_are_bound_denials(self) -> None:
        asking_policy = ApprovalPolicy()
        request = _request(asking_policy)
        headless = HeadlessApprovalBroker().decide(request)
        self.assertEqual(
            (headless.verdict, headless.reason_code),
            ("deny", "approval_unavailable"),
        )
        self.assertEqual(headless.binding(), request.binding())

        deny_policy = ApprovalPolicy(
            rules=(ApprovalRule(action="deny", tool_name="process.exec"),)
        )
        denied_request = _request(deny_policy)
        denied = resolve_approval(denied_request, deny_policy)
        self.assertEqual(
            (denied.verdict, denied.reason_code),
            ("deny", "approval_policy_denied"),
        )

    def test_contracts_are_immutable_and_exact_rules_reject_partial_selectors(self) -> None:
        policy = ApprovalPolicy()
        request = _request(policy)
        with self.assertRaises(FrozenInstanceError):
            request.call_id = "changed"  # type: ignore[misc]
        with self.assertRaisesRegex(ApprovalContractError, "require"):
            ApprovalRule(
                action="allow",
                tool_name="process.exec",
                tool_version="1.0.0",
            )

    def test_headless_effectful_call_handoffs_before_tool_start(self) -> None:
        executions: list[dict[str, object]] = []
        result = run_agent_harness(
            _ToolThenFinalModel({"command": "SECRET-COMMAND"}),
            _effectful_registry(executions),
            {"request": "run it"},
            allowed_permissions={"danger.exec"},
            approval_policy=ApprovalPolicy(),
        )

        event_types = [event["type"] for event in result["events"]]
        self.assertEqual(result["status"], "handoff")
        self.assertEqual(result["reason"], "human approval is required for the requested tool")
        self.assertFalse(executions)
        self.assertIn("approval.requested", event_types)
        self.assertIn("approval.resolved", event_types)
        self.assertIn("tool.rejected", event_types)
        self.assertNotIn("tool.started", event_types)
        self.assertNotIn("tool.effect_started", event_types)
        self.assertLess(
            event_types.index("approval.resolved"),
            event_types.index("tool.rejected"),
        )
        approval_events = [
            event
            for event in result["events"]
            if str(event["type"]).startswith("approval.")
        ]
        self.assertNotIn("SECRET-COMMAND", json.dumps(approval_events))

    def test_omitted_approval_policy_is_conservative(self) -> None:
        executions: list[dict[str, object]] = []
        result = run_agent_harness(
            _ToolThenFinalModel({"command": "must-not-run"}),
            _effectful_registry(executions),
            {"request": "run it"},
            allowed_permissions={"danger.exec"},
        )

        self.assertEqual(result["status"], "handoff")
        self.assertFalse(executions)
        self.assertEqual(result["events"][-1]["type"], "run.handoff")
        self.assertEqual(
            result["events"][-1]["payload"]["reason_code"],
            "approval_required",
        )

    def test_model_seam_detaches_arguments_before_approval_and_execution(self) -> None:
        executions: list[dict[str, object]] = []
        retained_call = ToolCall(
            call_id="call-sensitive-1",
            name="danger.exec",
            arguments={"command": "approved-snapshot"},
        )

        class RetainedCallModel(_ToolThenFinalModel):
            def plan(self, request, **_kwargs):
                if not request.observations:
                    return HarnessModelResponse(
                        kind="tool_calls",
                        tool_calls=(retained_call,),
                    )
                return HarnessModelResponse(kind="final", output={"message": "done"})

        class MutatingBroker(_AllowOnceBroker):
            def decide(self, request, *, preview=""):
                retained_call.arguments["command"] = "MUTATED-AFTER-MODEL-SEAM"
                return super().decide(request, preview=preview)

        result = run_agent_harness(
            RetainedCallModel({"command": "unused"}),
            _safe_high_risk_registry(executions),
            {"request": "run it"},
            allowed_permissions={"danger.exec"},
            approval_broker=MutatingBroker(),
        )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(executions, [{"command": "approved-snapshot"}])
        requested = next(
            event for event in result["events"] if event["type"] == "approval.requested"
        )
        self.assertEqual(
            requested["payload"]["arguments_sha256"],
            arguments_sha256({"command": "approved-snapshot"}),
        )

    def test_pending_safe_call_resume_returns_stable_approval_handoff(self) -> None:
        executions: list[dict[str, object]] = []
        checkpoints: list[HarnessCheckpoint] = []

        class SimulatedCrash(BaseException):
            pass

        def crash_on_pending(checkpoint: HarnessCheckpoint) -> None:
            checkpoints.append(checkpoint)
            if checkpoint.pending_effect is not None:
                raise SimulatedCrash()

        context = {"request": "run it"}
        with self.assertRaises(SimulatedCrash):
            run_agent_harness(
                _ToolThenFinalModel({"command": "resume-me"}),
                _safe_high_risk_registry(executions),
                context,
                allowed_permissions={"danger.exec"},
                checkpoint_sink=crash_on_pending,
            )
        pending = next(
            checkpoint
            for checkpoint in reversed(checkpoints)
            if checkpoint.pending_effect is not None
        )

        resumed = resume_agent_harness(
            pending,
            _ToolThenFinalModel({"command": "resume-me"}),
            _safe_high_risk_registry(executions),
            context,
            allowed_permissions={"danger.exec"},
        )

        self.assertEqual(resumed["status"], "handoff")
        self.assertFalse(executions)
        self.assertIsNone(resumed["checkpoint"].pending_effect)
        self.assertEqual(resumed["events"][-1]["type"], "run.handoff")

    def test_large_tool_arguments_are_hashed_not_embedded_in_model_event(self) -> None:
        marker = "PRIVATE-LARGE-ARGUMENT-" + ("x" * 80_000)
        executions: list[dict[str, object]] = []
        registry = ToolRegistry()

        def handler(arguments, _context):
            executions.append({"size": len(arguments["payload"])})
            return {"size": len(arguments["payload"])}

        registry.register(
            ToolSpec(
                name="large.read",
                version="1",
                description="Consume a large but valid model argument.",
                input_schema={
                    "type": "object",
                    "properties": {"payload": {"type": "string"}},
                    "required": ["payload"],
                    "additionalProperties": False,
                },
                permission="large.read",
                risk="low",
                replay_policy="safe",
                execution_isolation="trusted_inline",
                trusted_inline_reason="Test-only deterministic large input fixture",
                retry_policy=RetryPolicy(max_attempts=1),
            ),
            handler,
        )

        class LargeArgumentModel(_ToolThenFinalModel):
            def plan(self, request, **_kwargs):
                if not request.observations:
                    return HarnessModelResponse(
                        kind="tool_calls",
                        tool_calls=(
                            ToolCall(
                                call_id="call-large-1",
                                name="large.read",
                                arguments={"payload": marker},
                            ),
                        ),
                    )
                return HarnessModelResponse(kind="final", output={"message": "done"})

        result = run_agent_harness(
            LargeArgumentModel({}),
            registry,
            {"request": "consume it"},
            allowed_permissions={"large.read"},
            limits=HarnessLimits(max_event_payload_chars=2_000),
        )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(executions, [{"size": len(marker)}])
        completed = [
            event for event in result["events"] if event["type"] == "model.completed"
        ][0]
        serialized = json.dumps(completed, ensure_ascii=False, sort_keys=True)
        self.assertNotIn(marker, serialized)
        self.assertNotIn("model_response", completed["payload"])
        self.assertEqual(completed["payload"]["tool_call_count"], 1)
        self.assertRegex(
            completed["payload"]["model_response_sha256"],
            r"^[0-9a-f]{64}$",
        )

    def test_run_started_projects_only_context_lineage_metadata(self) -> None:
        class FinalModel(_ToolThenFinalModel):
            def plan(self, _request, **_kwargs):
                return HarnessModelResponse(kind="final", output={"message": "done"})

        private_summary = "PRIVATE-COMPACTION-SUMMARY"
        lineage = {
            "compaction_id": "compact_123",
            "source_message_count": 8,
            "source_messages_sha256": "a" * 64,
            "summary_sha256": "b" * 64,
            "active_context_sha256": "c" * 64,
            "active_message_ids": ["message_9", "message_10"],
            "summary": private_summary,
        }
        result = run_agent_harness(
            FinalModel({}),
            ToolRegistry(),
            {"context_lineage": lineage},
            allowed_permissions={"workspace.read"},
        )
        started = next(
            event for event in result["events"] if event["type"] == "run.started"
        )
        self.assertEqual(
            started["payload"],
            {
                "resumed": False,
                "context_sha256": started["payload"]["context_sha256"],
                "policy_sha256": started["payload"]["policy_sha256"],
                "compaction_id": "compact_123",
                "source_message_count": 8,
                "source_messages_sha256": "a" * 64,
                "summary_sha256": "b" * 64,
                "active_context_sha256": "c" * 64,
                "active_message_count": 2,
            },
        )
        public = public_harness_trace(result)
        serialized = json.dumps(public, ensure_ascii=False, sort_keys=True)
        self.assertNotIn(private_summary, serialized)
        public_started = next(
            event for event in public["events"] if event["type"] == "run.started"
        )
        self.assertEqual(public_started["payload"]["active_message_count"], 2)

    def test_bound_allow_occurs_before_effect_boundary(self) -> None:
        executions: list[dict[str, object]] = []
        result = run_agent_harness(
            _ToolThenFinalModel({"command": "approved"}),
            _effectful_registry(executions),
            {"request": "run it"},
            allowed_permissions={"danger.exec"},
            approval_policy=ApprovalPolicy(),
            approval_broker=_AllowOnceBroker(),
        )

        event_types = [event["type"] for event in result["events"]]
        self.assertEqual(result["status"], "completed")
        self.assertEqual(executions, [{"command": "approved"}])
        self.assertLess(
            event_types.index("approval.resolved"),
            event_types.index("tool.started"),
        )
        self.assertLess(
            event_types.index("tool.started"),
            event_types.index("tool.effect_started"),
        )

    def test_process_approval_preview_keeps_dangerous_command_tail(self) -> None:
        executions: list[dict[str, object]] = []
        broker = _AllowOnceBroker()
        dangerous_tail = "\nrm -rf /definitely-visible-tail"
        command = ("echo harmless\n" * 500) + dangerous_tail
        registry = ToolRegistry()

        def handler(arguments, context):
            context.begin_effect()
            executions.append(dict(arguments))
            return {"ok": True}

        registry.register(
            ToolSpec(
                name="process.exec",
                version="1",
                description="Exercise complete command approval previews.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "maxLength": 20_000}
                    },
                    "required": ["command"],
                    "additionalProperties": False,
                },
                permission="process.exec",
                risk="high",
                replay_policy="never",
                execution_isolation="trusted_inline",
                trusted_inline_reason="Test-only process approval preview fixture",
                retry_policy=RetryPolicy(max_attempts=1),
            ),
            handler,
        )

        class ProcessModel(_ToolThenFinalModel):
            def plan(self, request, **_kwargs):
                if not request.observations:
                    return HarnessModelResponse(
                        kind="tool_calls",
                        tool_calls=(
                            ToolCall(
                                call_id="call-process-preview",
                                name="process.exec",
                                arguments={"command": command},
                            ),
                        ),
                    )
                return HarnessModelResponse(kind="final", output={"message": "done"})

        result = run_agent_harness(
            ProcessModel({"command": command}),
            registry,
            {"request": "run it"},
            allowed_permissions={"process.exec"},
            approval_policy=ApprovalPolicy(),
            approval_broker=broker,
        )

        self.assertEqual(result["status"], "completed")
        self.assertTrue(broker.preview.endswith(dangerous_tail))
        self.assertEqual(len(broker.preview), len(command))

    def test_private_session_and_workspace_rules_persist_only_digests(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            workspace = root / "workspace"
            workspace.mkdir()
            state_home = root / "state"
            store = SessionStore(workspace, state_home=state_home)
            session = store.create(
                provider="test",
                model="test",
                permission_mode="workspace-write",
            )
            session_rule = ApprovalRule(
                action="allow",
                tool_name="workspace.patch",
                tool_version="1",
                arguments_sha256=arguments_sha256({"patch": "PRIVATE-PATCH"}),
            )
            workspace_rule = ApprovalRule(
                action="deny",
                tool_name="process.exec_host",
            )

            store.add_approval_rule(
                session_rule,
                scope="session",
                session_id=session["session_id"],
            )
            store.add_approval_rule(
                workspace_rule,
                scope="workspace",
                session_id=session["session_id"],
            )
            loaded = SessionStore(workspace, state_home=state_home).approval_rules(
                session["session_id"]
            )

            self.assertEqual(set(loaded), {session_rule, workspace_rule})
            serialized = store.approval_rules_path.read_text(encoding="utf-8")
            self.assertNotIn("PRIVATE-PATCH", serialized)
            self.assertEqual(store.approval_rules_path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(
                store.clear_approval_rules(
                    scope="session",
                    session_id=session["session_id"],
                ),
                1,
            )
            self.assertEqual(
                store.approval_rules(session["session_id"]),
                (workspace_rule,),
            )

    def test_approval_rules_reject_model_writable_state_directory(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            workspace = Path(raw_root) / "workspace"
            workspace.mkdir()
            store = SessionStore(workspace, state_home=workspace / "state")
            session = store.create(
                provider="test",
                model="test",
                permission_mode="workspace-write",
            )
            with self.assertRaisesRegex(SessionStoreError, "outside the workspace"):
                store.approval_rules(session["session_id"])


if __name__ == "__main__":
    unittest.main()
