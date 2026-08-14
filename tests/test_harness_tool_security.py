from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import time
import unittest

from teaching_skill_miner.harness import (
    CancellationToken,
    HarnessLimits,
    HarnessModelResponse,
    ProviderCapabilities,
    ProviderModelSpec,
    RetryPolicy,
    ToolCall,
    ToolRegistry,
    ToolSpec,
    run_agent_harness,
)


class _ToolThenFinalModel:
    def __init__(self, tool_name: str = "explode") -> None:
        self.calls = 0
        self.tool_name = tool_name

    def plan(self, _request, *, cancellation_token, deadline_monotonic):
        del deadline_monotonic
        cancellation_token.raise_if_cancelled()
        self.calls += 1
        if self.calls == 1:
            return HarnessModelResponse(
                kind="tool_calls",
                tool_calls=(ToolCall(call_id="secret_call", name=self.tool_name),),
            )
        return HarnessModelResponse(kind="final", output={"message": "recovered"})


class HarnessToolSecurityTests(unittest.TestCase):
    def test_trusted_inline_tools_require_an_explicit_audit_reason(self) -> None:
        with self.assertRaisesRegex(Exception, "audit reason"):
            ToolSpec(
                name="unsafe_inline",
                version="1.0.0",
                description="Invalid implicit inline execution fixture.",
                input_schema={"type": "object"},
                execution_isolation="trusted_inline",
            )

    @unittest.skipIf(
        sys.platform == "darwin", "exercises the unsupported-platform boundary"
    )
    def test_isolated_tools_fail_closed_when_kernel_sandbox_is_unavailable(
        self,
    ) -> None:
        calls = 0

        def must_not_run(_arguments, _context):
            nonlocal calls
            calls += 1
            return {"ok": True}

        registry = ToolRegistry()
        registry.register(
            ToolSpec(
                name="isolated_probe",
                version="1.0.0",
                description="Verify unsupported hosts cannot run isolated handlers.",
                input_schema={"type": "object"},
                permission="tool.read",
            ),
            must_not_run,
        )

        result = run_agent_harness(
            _ToolThenFinalModel("isolated_probe"),
            registry,
            {"scope": "unsupported-isolation-test"},
            retry_policy=RetryPolicy(max_attempts=1),
            allowed_permissions={"tool.read"},
        )

        failure = next(
            event for event in result["events"] if event["type"] == "tool.failed"
        )
        self.assertEqual(failure["payload"]["error_code"], "sandbox_unavailable")
        self.assertEqual(calls, 0)

    @unittest.skipUnless(sys.platform == "darwin", "requires macOS Seatbelt")
    def test_isolated_timeout_reaps_worker_before_a_late_side_effect(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            sentinel = Path(raw_root) / "late-side-effect.txt"
            registry = ToolRegistry()

            def late_writer(_arguments, _context):
                time.sleep(0.2)
                sentinel.write_text("must never exist", encoding="utf-8")
                return {"ok": True}

            registry.register(
                ToolSpec(
                    name="late_writer",
                    version="1.0.0",
                    description="Try one delayed side effect after the deadline.",
                    input_schema={"type": "object"},
                    permission="effect.write",
                    timeout_seconds=0.05,
                    replay_policy="never",
                ),
                late_writer,
            )
            result = run_agent_harness(
                _ToolThenFinalModel("late_writer"),
                registry,
                {"scope": "isolated-timeout"},
                limits=HarnessLimits(deadline_seconds=1.0),
                allowed_permissions={"effect.write"},
            )

            self.assertEqual(result["status"], "completed")
            failure = next(
                event for event in result["events"] if event["type"] == "tool.failed"
            )
            self.assertEqual(failure["payload"]["error_code"], "timeout")
            self.assertFalse(sentinel.exists())
            time.sleep(0.25)
            self.assertFalse(sentinel.exists())

    @unittest.skipUnless(sys.platform == "darwin", "requires macOS Seatbelt")
    def test_isolated_worker_kernel_profile_denies_network_and_file_writes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            sentinel = Path(raw_root) / "sandbox-write.txt"
            registry = ToolRegistry()

            def sandbox_probe(_arguments, _context):
                network_denied = False
                write_denied = False
                try:
                    connection = socket.socket()
                    connection.settimeout(0.05)
                    connection.connect(("127.0.0.1", 9))
                except PermissionError:
                    network_denied = True
                finally:
                    try:
                        connection.close()
                    except Exception:
                        pass
                try:
                    sentinel.write_text("blocked", encoding="utf-8")
                except PermissionError:
                    write_denied = True
                return {
                    "network_denied": network_denied,
                    "write_denied": write_denied,
                }

            registry.register(
                ToolSpec(
                    name="sandbox_probe",
                    version="1.0.0",
                    description="Verify the isolated worker kernel policy.",
                    input_schema={"type": "object"},
                    permission="effect.probe",
                ),
                sandbox_probe,
            )
            result = run_agent_harness(
                _ToolThenFinalModel("sandbox_probe"),
                registry,
                {"scope": "isolated-sandbox"},
                allowed_permissions={"effect.probe"},
            )

            observation = next(
                item
                for item in result["checkpoint"].observations
                if item.get("tool_name") == "sandbox_probe"
            )
            self.assertTrue(observation["ok"])
            self.assertEqual(
                observation["result"],
                {"network_denied": True, "write_denied": True},
            )
            self.assertFalse(sentinel.exists())

    @unittest.skipUnless(sys.platform == "darwin", "requires macOS Seatbelt")
    def test_cancelling_isolated_tool_reaps_its_process_group(self) -> None:
        worker_pids: list[int] = []
        token = CancellationToken()
        registry = ToolRegistry()

        def blocking_tool(_arguments, context):
            context.emit_progress("worker_started", {"pid": os.getpid()})
            time.sleep(30)
            return {"ok": True}

        registry.register(
            ToolSpec(
                name="blocking_tool",
                version="1.0.0",
                description="Block until the parent cancellation contract fires.",
                input_schema={"type": "object"},
                permission="effect.block",
                timeout_seconds=5.0,
                replay_policy="never",
            ),
            blocking_tool,
        )

        def observe(event):
            payload = event.get("payload", {})
            progress = payload.get("progress", {})
            if (
                event.get("type") == "tool.progress"
                and isinstance(progress, dict)
                and isinstance(progress.get("pid"), int)
            ):
                worker_pids.append(progress["pid"])
                token.cancel("test_cancel")

        result = run_agent_harness(
            _ToolThenFinalModel("blocking_tool"),
            registry,
            {"scope": "isolated-cancel"},
            limits=HarnessLimits(deadline_seconds=6.0),
            allowed_permissions={"effect.block"},
            cancellation_token=token,
            event_sink=observe,
        )

        self.assertEqual(result["status"], "cancelled")
        self.assertEqual(len(worker_pids), 1)
        with self.assertRaises(ProcessLookupError):
            os.kill(worker_pids[0], 0)

    def test_unexpected_tool_exception_text_is_not_persisted(self) -> None:
        secret = "API_KEY=do-not-persist-this"
        registry = ToolRegistry()
        registry.register(
            ToolSpec(
                name="explode",
                version="1.0.0",
                description="Raise one unexpected test exception.",
                input_schema={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                permission="tool.read",
                execution_isolation="trusted_inline",
                trusted_inline_reason="exercise exception redaction without an OS sandbox",
            ),
            lambda _arguments, _context: (_ for _ in ()).throw(RuntimeError(secret)),
        )

        result = run_agent_harness(
            _ToolThenFinalModel(),
            registry,
            {"scope": "security-test"},
            run_id="run_tool_secret",
            turn_id="turn_tool_secret",
            retry_policy=RetryPolicy(max_attempts=1),
            allowed_permissions={"tool.read"},
        )
        serializable = {
            **result,
            "checkpoint": result["checkpoint"].to_dict(),
        }
        rendered = json.dumps(serializable, ensure_ascii=False)

        self.assertEqual(result["status"], "completed")
        self.assertNotIn(secret, rendered)
        self.assertIn("unexpected tool failure (RuntimeError)", rendered)

    def test_remote_data_scope_and_consent_are_independent_gates(self) -> None:
        calls: list[str] = []
        registry = ToolRegistry()
        registry.register(
            ToolSpec(
                name="web_lookup",
                version="1.0.0",
                description="Look up one public web fact after explicit consent.",
                input_schema={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                permission="web.read",
                data_scope="public_web",
                requires_user_consent=True,
                execution_isolation="trusted_inline",
                trusted_inline_reason="bounded in-memory authorization fixture",
            ),
            lambda _arguments, context: (
                calls.append(context.principal_id) or {"ok": True}
            ),
        )

        missing_scope = run_agent_harness(
            _ToolThenFinalModel("web_lookup"),
            registry,
            {"scope": "data-boundary-test"},
            run_id="run_web_scope",
            turn_id="turn_web_scope",
            allowed_permissions={"web.read"},
        )
        self.assertEqual(calls, [])
        self.assertTrue(
            any(
                event.get("payload", {}).get("error_code")
                == "data_scope_not_authorized"
                for event in missing_scope["events"]
            )
        )

        missing_consent = run_agent_harness(
            _ToolThenFinalModel("web_lookup"),
            registry,
            {"scope": "data-boundary-test"},
            run_id="run_web_consent",
            turn_id="turn_web_consent",
            allowed_permissions={"web.read"},
            trusted_data_scopes={"internal", "public_web"},
        )
        self.assertEqual(calls, [])
        self.assertTrue(
            any(
                event.get("payload", {}).get("error_code")
                == "explicit_user_consent_required"
                for event in missing_consent["events"]
            )
        )

        authorized = run_agent_harness(
            _ToolThenFinalModel("web_lookup"),
            registry,
            {"scope": "data-boundary-test"},
            run_id="run_web_authorized",
            turn_id="turn_web_authorized",
            allowed_permissions={"web.read"},
            principal_id="learner-42",
            session_id="session-7",
            trusted_data_scopes={"internal", "public_web", "remote_consent"},
        )
        self.assertEqual(authorized["status"], "completed")
        self.assertEqual(calls, ["learner-42"])

    def test_unauthorized_scope_is_not_exposed_to_the_model(self) -> None:
        seen_tools: list[tuple[dict, ...]] = []

        class _InspectingModel:
            def plan(self, request, *, cancellation_token, deadline_monotonic):
                del cancellation_token, deadline_monotonic
                seen_tools.append(request.tools)
                return HarnessModelResponse(kind="final", output={"ok": True})

        registry = ToolRegistry()
        registry.register(
            ToolSpec(
                name="web_search",
                version="1.0.0",
                description="Provider-managed public web search.",
                input_schema={"type": "object"},
                permission="chat.web_search",
                execution_mode="provider_managed",
                data_scope="public_web",
                requires_user_consent=True,
            ),
            lambda _arguments, _context: {"ok": True},
        )
        run_agent_harness(
            _InspectingModel(),
            registry,
            {"scope": "model-visible-tools"},
            allowed_permissions={"chat.web_search"},
            trusted_data_scopes={"internal"},
        )
        self.assertEqual(seen_tools, [()])

    def test_resume_cannot_change_data_scope_authorization(self) -> None:
        class _CrashBeforeFinal:
            @property
            def model_spec(self):
                return ProviderModelSpec(
                    provider="test-harness",
                    model="authorization-crash-v1",
                    capabilities=ProviderCapabilities(
                        provider="test-harness",
                        model="authorization-crash-v1",
                        structured_output=True,
                        native_stream=False,
                        cancellation=True,
                    ),
                    context_window_tokens=8_192,
                    maximum_output_tokens=1_024,
                )

            def plan(self, _request, *, cancellation_token, deadline_monotonic):
                del cancellation_token, deadline_monotonic
                raise SystemExit("simulated crash")

        checkpoints = []
        with self.assertRaises(SystemExit):
            run_agent_harness(
                _CrashBeforeFinal(),
                ToolRegistry(),
                {"scope": "authorization-resume"},
                run_id="run_auth_resume",
                turn_id="turn_auth_resume",
                allowed_permissions={"tool.read"},
                principal_id="learner-a",
                trusted_data_scopes={"internal"},
                checkpoint_sink=checkpoints.append,
            )
        self.assertTrue(checkpoints)
        from teaching_skill_miner.harness import resume_agent_harness

        with self.assertRaisesRegex(Exception, "context does not match checkpoint"):
            resume_agent_harness(
                checkpoints[-1],
                _CrashBeforeFinal(),
                ToolRegistry(),
                {"scope": "authorization-resume"},
                allowed_permissions={"tool.read"},
                principal_id="learner-a",
                trusted_data_scopes={"internal", "public_web", "remote_consent"},
            )

        hidden_registry = ToolRegistry()
        hidden_registry.register(
            ToolSpec(
                name="secret_read",
                version="1.0.0",
                description="Read one protected test value.",
                input_schema={"type": "object"},
                permission="secret.read",
            ),
            lambda _arguments, _context: {"secret": True},
        )
        with self.assertRaisesRegex(Exception, "context does not match checkpoint"):
            resume_agent_harness(
                checkpoints[-1],
                _CrashBeforeFinal(),
                hidden_registry,
                {"scope": "authorization-resume"},
                allowed_permissions={"secret.read"},
                principal_id="learner-a",
                trusted_data_scopes={"internal"},
            )


if __name__ == "__main__":
    unittest.main()
