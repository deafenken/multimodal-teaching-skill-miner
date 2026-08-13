from __future__ import annotations

from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch
from typing import Any, Mapping

from teaching_skill_miner.harness import (
    CancellationToken,
    HarnessCheckpoint,
    HarnessContractError,
    HarnessJournal,
    HarnessLimits,
    HarnessModelResponse,
    HarnessRunHandle,
    JournalLifecycleError,
    ProviderCapabilities,
    ProviderModelSpec,
    RetryPolicy,
    ToolCall,
    ToolRegistry,
    ToolSpec,
    resume_agent_harness,
    run_agent_harness,
)
from teaching_skill_miner.harness.events import HarnessEventEmitter


class _ScriptedModel:
    def __init__(
        self,
        responses: list[HarnessModelResponse],
        *,
        model_name: str = "journal-script-v1",
        native_stream: bool = False,
    ) -> None:
        self.responses = list(responses)
        self.requests: list[Any] = []
        self.model_name = model_name
        self.native_stream = native_stream

    @property
    def model_spec(self) -> ProviderModelSpec:
        return ProviderModelSpec(
            provider="test-harness",
            model=self.model_name,
            capabilities=ProviderCapabilities(
                provider="test-harness",
                model=self.model_name,
                structured_output=True,
                native_stream=self.native_stream,
                cancellation=True,
            ),
            context_window_tokens=8_192,
            maximum_output_tokens=1_024,
        )

    def plan(
        self,
        request: Any,
        *,
        cancellation_token: CancellationToken,
        deadline_monotonic: float,
    ) -> HarnessModelResponse:
        del cancellation_token, deadline_monotonic
        self.requests.append(request)
        if not self.responses:
            raise AssertionError("model was called beyond its script")
        return self.responses.pop(0)


def _final(message: str = "done") -> HarnessModelResponse:
    return HarnessModelResponse(kind="final", output={"message": message})


def _tool_plan(call_id: str = "effect_1") -> HarnessModelResponse:
    return HarnessModelResponse(
        kind="tool_calls",
        tool_calls=(
            ToolCall(
                call_id=call_id,
                name="external_effect",
                arguments={"value": 7},
                idempotency_key="effect-key-7",
            ),
        ),
    )


def _registry(
    handler: Any,
    *,
    replay_policy: str = "never",
    timeout_seconds: float = 2.0,
    parallel_safe: bool = False,
    retry_policy: RetryPolicy | None = None,
    output_schema: Mapping[str, Any] | None = None,
) -> ToolRegistry:
    registry = ToolRegistry()
    resolved_output_schema = output_schema or {
        "type": "object",
        "properties": {"value": {"type": "integer"}},
        "required": ["value"],
        "additionalProperties": False,
    }
    registry.register(
        ToolSpec(
            name="external_effect",
            version="1.0.0",
            description="A bounded effect used by crash recovery tests.",
            input_schema={
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
                "additionalProperties": False,
            },
            output_schema=resolved_output_schema,
            permission="effect.write",
            risk="high",
            timeout_seconds=timeout_seconds,
            replay_policy=replay_policy,
            parallel_safe=parallel_safe,
            execution_isolation="trusted_inline",
            trusted_inline_reason="bounded deterministic crash recovery fixture",
            retry_policy=retry_policy or RetryPolicy(max_attempts=1),
        ),
        handler,
    )
    return registry


def _run(
    model: _ScriptedModel,
    registry: ToolRegistry,
    journal: HarnessJournal,
    **kwargs: Any,
) -> dict[str, Any]:
    return run_agent_harness(
        model,
        registry,
        {"topic": "durability"},
        run_id=journal.run_id,
        turn_id=journal.turn_id,
        journal=journal,
        limits=HarnessLimits(deadline_seconds=5.0),
        retry_policy=RetryPolicy(max_attempts=1),
        allowed_permissions={"effect.write"},
        **kwargs,
    )


class HarnessRuntimeJournalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.run_id = "run-runtime-journal"
        self.turn_id = "turn-runtime-journal"

    def journal(self) -> HarnessJournal:
        return HarnessJournal(
            self.root / "events.jsonl",
            run_id=self.run_id,
            turn_id=self.turn_id,
        )

    def test_runtime_durably_appends_before_ui_and_anchors_checkpoint(self) -> None:
        journal = self.journal()
        observed: list[dict[str, Any]] = []

        def ui_sink(event: Mapping[str, Any]) -> None:
            acknowledgement = journal.last_ack
            self.assertIsNotNone(acknowledgement)
            assert acknowledgement is not None
            self.assertTrue(acknowledgement.durable)
            self.assertEqual(acknowledgement.sequence, event["sequence"])
            self.assertEqual(acknowledgement.event_id, event["event_id"])
            observed.append(dict(event))

        result = _run(
            _ScriptedModel([_final()]),
            ToolRegistry(),
            journal,
            event_sink=ui_sink,
        )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(observed, journal.replay())
        self.assertEqual(observed[-1]["type"], "run.completed")
        self.assertIsNotNone(journal.terminal_ack)
        self.assertTrue(journal.terminal_ack.durable)  # type: ignore[union-attr]
        self.assertEqual(journal.load_checkpoint(), result["checkpoint"].to_dict())

    def test_terminal_fsync_failure_never_reaches_ui_or_returns_a_false_ack(
        self,
    ) -> None:
        journal = self.journal()
        observed: list[dict[str, Any]] = []
        original_append = journal.append

        def fail_terminal_fsync(event: Mapping[str, Any]) -> Any:
            if event.get("type") == "run.completed":
                with patch(
                    "teaching_skill_miner.harness.journal.os.fsync",
                    side_effect=OSError("injected terminal fsync failure"),
                ):
                    return original_append(event)
            return original_append(event)

        journal.append = fail_terminal_fsync  # type: ignore[method-assign]
        with self.assertRaises(JournalLifecycleError):
            _run(
                _ScriptedModel([_final()]),
                ToolRegistry(),
                journal,
                event_sink=lambda event: observed.append(dict(event)),
            )

        self.assertFalse(
            any(item["type"] in {"run.completed", "run.failed"} for item in observed)
        )
        self.assertIsNotNone(journal.last_ack)
        self.assertFalse(journal.last_ack.terminal)  # type: ignore[union-attr]
        self.assertIsNone(journal.terminal_ack)
        # The visible but unacknowledged terminal is rolled back to the last
        # acknowledged boundary. A reopened journal must never promote page
        # cache bytes whose durability fence failed.
        recovered = self.journal()
        self.assertIsNone(recovered.terminal_type)
        self.assertFalse(
            any(
                event["type"] in {"run.completed", "run.failed"}
                for event in recovered.replay()
            )
        )

    def test_durable_tool_completion_is_reconciled_without_reexecution(self) -> None:
        journal = self.journal()
        executions = 0

        def effect(arguments: Mapping[str, Any], _context: Any) -> Mapping[str, Any]:
            nonlocal executions
            executions += 1
            return {"value": int(arguments["value"])}

        registry = _registry(effect, replay_policy="never")

        def crash_after_durable_completion(event: Mapping[str, Any]) -> None:
            if event.get("type") == "tool.completed":
                raise SystemExit("crash after durable tool settlement")

        with self.assertRaisesRegex(SystemExit, "durable tool settlement"):
            _run(
                _ScriptedModel([_tool_plan()]),
                registry,
                journal,
                event_sink=crash_after_durable_completion,
            )

        checkpoint_value = journal.load_checkpoint()
        self.assertIsNotNone(checkpoint_value)
        checkpoint = HarnessCheckpoint.from_value(checkpoint_value)  # type: ignore[arg-type]
        self.assertIsNotNone(checkpoint.pending_effect)
        completed_sequence = journal.last_sequence
        self.assertEqual(journal.replay()[-1]["type"], "tool.completed")

        resumed_model = _ScriptedModel([_final("after reconciliation")])
        resumed = resume_agent_harness(
            None,
            resumed_model,
            registry,
            {"topic": "durability"},
            journal=journal,
            limits=HarnessLimits(deadline_seconds=5.0),
            retry_policy=RetryPolicy(max_attempts=1),
            allowed_permissions={"effect.write"},
        )

        self.assertEqual(executions, 1)
        self.assertEqual(resumed["status"], "completed")
        new_events = journal.replay(after_sequence=completed_sequence)
        self.assertEqual(new_events[0]["type"], "run.started")
        self.assertEqual(new_events[0]["payload"]["resumed"], True)
        self.assertEqual(new_events[1]["type"], "tool.reconciled")
        self.assertFalse(
            any(
                item["type"] == "tool.started"
                and item["payload"].get("call_id") == "effect_1"
                for item in new_events
            )
        )
        self.assertTrue(resumed_model.requests[0].observations)
        self.assertEqual(
            resumed_model.requests[0].observations[-1]["result"],
            {"value": 7},
        )

    def test_uncertain_never_replay_effect_hands_off_after_process_death(self) -> None:
        journal = self.journal()
        executions = 0

        def uncertain_effect(
            _arguments: Mapping[str, Any], _context: Any
        ) -> Mapping[str, Any]:
            nonlocal executions
            executions += 1
            raise SystemExit("external effect may have committed")

        registry = _registry(uncertain_effect, replay_policy="never")
        with self.assertRaisesRegex(SystemExit, "may have committed"):
            _run(_ScriptedModel([_tool_plan()]), registry, journal)

        self.assertEqual(journal.replay()[-1]["type"], "tool.started")
        resumed_model = _ScriptedModel([_final("must not run")])
        resumed = resume_agent_harness(
            None,
            resumed_model,
            registry,
            {"topic": "durability"},
            journal=journal,
            limits=HarnessLimits(deadline_seconds=5.0),
            retry_policy=RetryPolicy(max_attempts=1),
            allowed_permissions={"effect.write"},
        )

        self.assertEqual(executions, 1)
        self.assertFalse(resumed_model.requests)
        self.assertEqual(resumed["status"], "handoff")
        self.assertEqual(resumed["reason"], "unsafe_tool_replay_blocked")
        self.assertEqual(journal.terminal_type, "run.handoff")

    def test_recovery_survives_death_after_resumed_marker_before_checkpoint(
        self,
    ) -> None:
        journal = self.journal()
        registry = _registry(
            lambda arguments, _context: dict(arguments), replay_policy="never"
        )

        def die_after_pending_checkpoint(checkpoint: HarnessCheckpoint) -> None:
            if checkpoint.pending_effect is not None:
                raise SystemExit("death after pending checkpoint")

        with self.assertRaisesRegex(SystemExit, "pending checkpoint"):
            _run(
                _ScriptedModel([_tool_plan()]),
                registry,
                journal,
                checkpoint_sink=die_after_pending_checkpoint,
            )
        checkpoint = journal.load_checkpoint()
        self.assertIsNotNone(checkpoint)
        checkpoint_sequence = journal.last_sequence

        def die_after_resume_marker(event: Mapping[str, Any]) -> None:
            if event.get("type") == "run.started" and event["payload"].get("resumed"):
                raise SystemExit("death after resumed marker")

        with self.assertRaisesRegex(SystemExit, "resumed marker"):
            resume_agent_harness(
                checkpoint,
                _ScriptedModel([_final("must not run")]),
                registry,
                {"topic": "durability"},
                journal=journal,
                limits=HarnessLimits(deadline_seconds=5.0),
                retry_policy=RetryPolicy(max_attempts=1),
                allowed_permissions={"effect.write"},
                event_sink=die_after_resume_marker,
            )
        self.assertEqual(journal.last_sequence, checkpoint_sequence + 1)

        resumed = resume_agent_harness(
            None,
            _ScriptedModel([_final("must not run")]),
            registry,
            {"topic": "durability"},
            journal=journal,
            limits=HarnessLimits(deadline_seconds=5.0),
            retry_policy=RetryPolicy(max_attempts=1),
            allowed_permissions={"effect.write"},
        )
        self.assertEqual(resumed["status"], "handoff")
        self.assertEqual(resumed["reason"], "unsafe_tool_replay_blocked")

    def test_nonempty_journal_cannot_be_silently_reused_as_fresh(self) -> None:
        journal = self.journal()
        _run(_ScriptedModel([_final()]), ToolRegistry(), journal)

        with self.assertRaisesRegex(HarnessContractError, "empty journal"):
            _run(_ScriptedModel([_final()]), ToolRegistry(), journal)

    def test_resume_binds_limits_retry_policy_model_and_capabilities(self) -> None:
        journal = self.journal()

        def uncertain_effect(
            _arguments: Mapping[str, Any], _context: Any
        ) -> Mapping[str, Any]:
            raise SystemExit("effect boundary")

        registry = _registry(uncertain_effect, replay_policy="never")
        with self.assertRaisesRegex(SystemExit, "effect boundary"):
            _run(_ScriptedModel([_tool_plan()]), registry, journal)
        checkpoint = HarnessCheckpoint.from_value(journal.load_checkpoint())

        mismatches = (
            {
                "limits": HarnessLimits(deadline_seconds=30.0),
                "retry_policy": RetryPolicy(max_attempts=1),
                "model": _ScriptedModel([_final()]),
            },
            {
                "limits": HarnessLimits(deadline_seconds=5.0),
                "retry_policy": RetryPolicy(max_attempts=2),
                "model": _ScriptedModel([_final()]),
            },
            {
                "limits": HarnessLimits(deadline_seconds=5.0),
                "retry_policy": RetryPolicy(max_attempts=1),
                "model": _ScriptedModel(
                    [_final()], model_name="journal-script-v2"
                ),
            },
            {
                "limits": HarnessLimits(deadline_seconds=5.0),
                "retry_policy": RetryPolicy(max_attempts=1),
                "model": _ScriptedModel([_final()], native_stream=True),
            },
        )
        for candidate in mismatches:
            with self.subTest(candidate=candidate):
                with self.assertRaisesRegex(
                    HarnessContractError, "execution policy does not match"
                ):
                    resume_agent_harness(
                        checkpoint,
                        candidate["model"],
                        registry,
                        {"topic": "durability"},
                        journal=journal,
                        limits=candidate["limits"],
                        retry_policy=candidate["retry_policy"],
                        allowed_permissions={"effect.write"},
                    )

    def test_resume_binds_complete_tool_execution_semantics(self) -> None:
        """Same-version tools cannot weaken replay or retry on recovery."""

        def crash_at_effect_boundary(
            _arguments: Mapping[str, Any], _context: Any
        ) -> Mapping[str, Any]:
            raise SystemExit("tool semantics effect boundary")

        cases = (
            (
                "never_to_safe",
                {"replay_policy": "never"},
                {"replay_policy": "safe"},
            ),
            (
                "tool_max_attempts",
                {"replay_policy": "safe"},
                {
                    "replay_policy": "safe",
                    "retry_policy": RetryPolicy(max_attempts=2),
                },
            ),
            (
                "tool_timeout",
                {"replay_policy": "safe", "timeout_seconds": 2.0},
                {"replay_policy": "safe", "timeout_seconds": 3.0},
            ),
            (
                "parallel_execution",
                {"replay_policy": "safe", "parallel_safe": False},
                {"replay_policy": "safe", "parallel_safe": True},
            ),
            (
                "output_validation",
                {"replay_policy": "safe"},
                {
                    "replay_policy": "safe",
                    "output_schema": {
                        "type": "object",
                        "properties": {
                            "value": {"type": "integer"},
                            "receipt": {"type": "string"},
                        },
                        "required": ["value", "receipt"],
                        "additionalProperties": False,
                    },
                },
            ),
        )

        for index, (label, original_kwargs, resumed_kwargs) in enumerate(cases):
            with self.subTest(semantic=label):
                journal = HarnessJournal(
                    self.root / f"tool-policy-{index}.jsonl",
                    run_id=f"run-tool-policy-{index}",
                    turn_id=f"turn-tool-policy-{index}",
                )
                original_registry = _registry(
                    crash_at_effect_boundary,
                    **original_kwargs,
                )
                with self.assertRaisesRegex(
                    SystemExit, "tool semantics effect boundary"
                ):
                    _run(
                        _ScriptedModel([_tool_plan()]),
                        original_registry,
                        journal,
                    )
                checkpoint = HarnessCheckpoint.from_value(journal.load_checkpoint())
                changed_registry = _registry(
                    lambda arguments, _context: dict(arguments),
                    **resumed_kwargs,
                )

                with self.assertRaisesRegex(
                    HarnessContractError, "execution policy does not match"
                ):
                    resume_agent_harness(
                        checkpoint,
                        _ScriptedModel([_final("must not run")]),
                        changed_registry,
                        {"topic": "durability"},
                        journal=journal,
                        limits=HarnessLimits(deadline_seconds=5.0),
                        retry_policy=RetryPolicy(max_attempts=1),
                        allowed_permissions={"effect.write"},
                    )

    def test_tool_execution_manifest_is_complete_and_canonical(self) -> None:
        first = _registry(
            lambda arguments, _context: dict(arguments),
            replay_policy="safe",
            timeout_seconds=4.0,
            parallel_safe=True,
            retry_policy=RetryPolicy(
                max_attempts=3,
                initial_backoff_seconds=0.25,
                backoff_multiplier=3.0,
                max_backoff_seconds=2.0,
            ),
        )
        manifest = first.execution_manifests({"effect.write"})[0]

        self.assertEqual(manifest["name"], "external_effect")
        self.assertEqual(manifest["version"], "1.0.0")
        self.assertEqual(manifest["permission"], "effect.write")
        self.assertEqual(manifest["data_scope"], "internal")
        self.assertEqual(manifest["replay_policy"], "safe")
        self.assertEqual(manifest["timeout_seconds"], 4.0)
        self.assertTrue(manifest["parallel_safe"])
        self.assertEqual(manifest["retry_policy"]["max_attempts"], 3)
        self.assertIn("input_schema", manifest)
        self.assertIn("output_schema", manifest)
        self.assertEqual(
            first.execution_manifests({"effect.write"}),
            first.execution_manifests({"effect.write"}),
        )

    def test_resume_rejects_a_model_without_stable_identity_without_repr(self) -> None:
        private_marker = "PRIVATE_MODEL_REPR_MUST_NOT_LEAK"

        class UnclassifiedModel:
            def __repr__(self) -> str:
                return private_marker

            def plan(self, _request, **_kwargs):
                return _tool_plan()

        journal = HarnessJournal(
            self.root / "unclassified.jsonl",
            run_id="run-unclassified",
            turn_id="turn-unclassified",
        )

        def uncertain_effect(
            _arguments: Mapping[str, Any], _context: Any
        ) -> Mapping[str, Any]:
            raise SystemExit("unclassified effect boundary")

        registry = _registry(uncertain_effect, replay_policy="never")
        with self.assertRaisesRegex(SystemExit, "unclassified effect boundary"):
            run_agent_harness(
                UnclassifiedModel(),
                registry,
                {"topic": "durability"},
                run_id=journal.run_id,
                turn_id=journal.turn_id,
                journal=journal,
                limits=HarnessLimits(deadline_seconds=5.0),
                retry_policy=RetryPolicy(max_attempts=1),
                allowed_permissions={"effect.write"},
            )
        checkpoint = HarnessCheckpoint.from_value(journal.load_checkpoint())
        with self.assertRaisesRegex(
            HarnessContractError, "stable provider model identity"
        ) as raised:
            resume_agent_harness(
                checkpoint,
                UnclassifiedModel(),
                registry,
                {"topic": "durability"},
                journal=journal,
                limits=HarnessLimits(deadline_seconds=5.0),
                retry_policy=RetryPolicy(max_attempts=1),
                allowed_permissions={"effect.write"},
            )
        self.assertNotIn(private_marker, str(raised.exception))

    def test_secret_bearing_internal_exception_is_normalized(self) -> None:
        journal = self.journal()
        private_marker = "PRIVATE_PROVIDER_PAYLOAD_8dd176"

        def broken_steering_source() -> tuple[Mapping[str, Any], ...]:
            raise RuntimeError(private_marker)

        result = _run(
            _ScriptedModel([_final("must not run")]),
            ToolRegistry(),
            journal,
            steering_source=broken_steering_source,
        )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["reason"], "internal_error")
        self.assertNotIn(private_marker, json.dumps(result, default=str))
        self.assertNotIn(private_marker, journal.path.read_text("utf-8"))


class HarnessBoundedReplayTests(unittest.TestCase):
    def test_emitter_rejects_oversized_identifiers(self) -> None:
        with self.assertRaises(HarnessContractError):
            HarnessEventEmitter(run_id="r" * 161, turn_id="turn")
        with self.assertRaises(HarnessContractError):
            HarnessEventEmitter(run_id="run", turn_id="t" * 161)

    def test_failed_durable_ack_does_not_advance_or_lock_terminal_state(self) -> None:
        class Acknowledgement:
            durable = True

        should_fail = False
        published: list[dict[str, Any]] = []

        def durable_sink(_event: Mapping[str, Any]) -> Acknowledgement:
            if should_fail:
                raise OSError("injected durable sink failure")
            return Acknowledgement()

        emitter = HarnessEventEmitter(
            run_id="run-emitter-rollback",
            turn_id="turn-emitter-rollback",
            durable_sink=durable_sink,
            event_sink=lambda event: published.append(dict(event)),
        )
        emitter.emit("run.started", {"resumed": False})
        should_fail = True
        with self.assertRaisesRegex(OSError, "durable sink failure"):
            emitter.emit("run.completed", {"reason_code": "done"})

        self.assertEqual(emitter.next_sequence, 2)
        self.assertIsNone(emitter.terminal_type)
        self.assertEqual([item["type"] for item in published], ["run.started"])

        should_fail = False
        replacement = emitter.emit(
            "run.failed",
            {"error_code": "storage_error", "reason_code": "storage_error"},
        )
        self.assertEqual(replacement["sequence"], 2)
        self.assertEqual(emitter.terminal_type, "run.failed")

    def test_observer_failure_cannot_reverse_a_committed_event(self) -> None:
        class Acknowledgement:
            durable = True

        persisted: list[dict[str, Any]] = []

        def durable_sink(event: Mapping[str, Any]) -> Acknowledgement:
            persisted.append(dict(event))
            return Acknowledgement()

        def detached_observer(_event: Mapping[str, Any]) -> None:
            raise BrokenPipeError("the UI transport detached")

        emitter = HarnessEventEmitter(
            run_id="run-observer-detached",
            turn_id="turn-observer-detached",
            durable_sink=durable_sink,
            event_sink=detached_observer,
        )

        started = emitter.emit("run.started", {"resumed": False})
        completed = emitter.emit("run.completed", {"reason_code": "done"})

        self.assertEqual([item["sequence"] for item in persisted], [1, 2])
        self.assertEqual([started["sequence"], completed["sequence"]], [1, 2])
        self.assertEqual(emitter.terminal_type, "run.completed")

    def test_handle_without_journal_expires_old_cursors_explicitly(self) -> None:
        handle = HarnessRunHandle(
            run_id="run-memory-window",
            turn_id="turn-memory-window",
            max_in_memory_events=2,
            max_in_memory_event_bytes=100_000,
        )
        emitter = HarnessEventEmitter(
            run_id=handle.run_id,
            turn_id=handle.turn_id,
            event_sink=handle.event_sink,
        )
        emitter.emit("run.started", {"resumed": False})
        emitter.emit("model.started", {"attempt": 1, "step": 1})
        emitter.emit(
            "model.completed", {"attempt": 1, "step": 1, "kind": "final"}
        )

        with self.assertRaisesRegex(HarnessContractError, "cursor expired"):
            handle.events_after(0)
        self.assertEqual(
            [item["sequence"] for item in handle.events_after(1)],
            [2, 3],
        )

    def test_handle_uses_journal_for_events_older_than_memory_window(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            journal = HarnessJournal(
                Path(directory) / "events.jsonl",
                run_id="run-durable-window",
                turn_id="turn-durable-window",
            )
            handle = HarnessRunHandle(
                run_id=journal.run_id,
                turn_id=journal.turn_id,
                journal=journal,
                max_in_memory_events=2,
                max_in_memory_event_bytes=100_000,
            )
            emitter = HarnessEventEmitter(
                run_id=journal.run_id,
                turn_id=journal.turn_id,
                durable_sink=journal.append,
                event_sink=handle.event_sink,
            )
            emitter.emit("run.started", {"resumed": False})
            emitter.emit("model.started", {"attempt": 1, "step": 1})
            emitter.emit(
                "model.completed", {"attempt": 1, "step": 1, "kind": "final"}
            )

            self.assertLessEqual(len(handle._events), 2)
            self.assertEqual(
                [item["sequence"] for item in handle.events_after(0)],
                [1, 2, 3],
            )

    def test_handle_rejects_publish_before_durable_append(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            journal = HarnessJournal(
                Path(directory) / "events.jsonl",
                run_id="run-order-guard",
                turn_id="turn-order-guard",
            )
            handle = HarnessRunHandle(
                run_id=journal.run_id,
                turn_id=journal.turn_id,
                journal=journal,
            )
            event = HarnessEventEmitter(
                run_id=journal.run_id,
                turn_id=journal.turn_id,
            ).emit("run.started", {"resumed": False})

            with self.assertRaisesRegex(HarnessContractError, "durable journal ack"):
                handle.event_sink(event)

    def test_terminal_journal_handle_has_deterministic_recovery_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            journal = HarnessJournal(
                Path(directory) / "events.jsonl",
                run_id="run-terminal-replay",
                turn_id="turn-terminal-replay",
            )
            _run(
                _ScriptedModel([_final()]),
                ToolRegistry(),
                journal,
            )
            handle = HarnessRunHandle(
                run_id=journal.run_id,
                turn_id=journal.turn_id,
                journal=journal,
                max_in_memory_events=2,
            )

            self.assertTrue(handle.settled)
            with self.assertRaisesRegex(HarnessContractError, "durable_status"):
                handle.wait(0.01)
            status = handle.durable_status()
            self.assertEqual(status["terminal_type"], "run.completed")
            self.assertEqual(status["terminal_event"]["type"], "run.completed")
            self.assertEqual(status["last_sequence"], journal.last_sequence)
            self.assertEqual(status["checkpoint"], journal.load_checkpoint())


if __name__ == "__main__":
    unittest.main()
