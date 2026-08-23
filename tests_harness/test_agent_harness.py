from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
import unittest
from typing import Any, Callable, Mapping

from agent_harness.core import (
    CancellationToken,
    HarnessCheckpoint,
    HarnessContractError,
    HarnessLimits,
    HarnessModelResponse,
    ProviderCapabilities,
    ProviderErrorKind,
    ProviderFailure,
    ProviderModelSpec,
    RetryPolicy,
    ToolCall,
    ToolExecutionError,
    ToolPermissionError,
    ToolRegistry,
    ToolSpec,
    public_harness_trace,
    resume_agent_harness,
    run_agent_harness,
)


_TERMINAL_EVENTS = {
    "run.completed",
    "run.cancelled",
    "run.failed",
    "run.handoff",
}


class _FakeClock:
    """Deterministic monotonic clock used by retry and timeout contracts."""

    def __init__(self, now: float = 100.0) -> None:
        self.now = float(now)
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        if seconds < 0:
            raise AssertionError("the harness attempted a negative sleep")
        self.sleeps.append(float(seconds))
        self.now += float(seconds)

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise AssertionError("the fake clock cannot move backwards")
        self.now += float(seconds)


class _ScriptedModel:
    def __init__(
        self,
        responses: list[
            HarnessModelResponse
            | BaseException
            | Callable[..., HarnessModelResponse]
        ],
    ) -> None:
        self.responses = list(responses)
        self.calls = 0
        self.requests: list[Any] = []
        self.deadlines: list[float] = []
        self.tokens: list[CancellationToken] = []

    @property
    def model_spec(self) -> ProviderModelSpec:
        return ProviderModelSpec(
            provider="test-harness",
            model="scripted-v1",
            capabilities=ProviderCapabilities(
                provider="test-harness",
                model="scripted-v1",
                structured_output=True,
                native_stream=False,
                cancellation=True,
            ),
            context_window_tokens=8_192,
            maximum_output_tokens=1_024,
        )

    def classify_error(self, error: BaseException) -> ProviderFailure:
        if isinstance(error, TimeoutError):
            return ProviderFailure(
                kind=ProviderErrorKind.TIMEOUT,
                retryable=True,
                safe_code="test_provider_timeout",
            )
        return ProviderFailure(
            kind=ProviderErrorKind.UNKNOWN,
            retryable=False,
            safe_code="test_provider_unknown",
        )

    def plan(
        self,
        request: Any,
        *,
        cancellation_token: CancellationToken,
        deadline_monotonic: float,
    ) -> HarnessModelResponse:
        self.calls += 1
        self.requests.append(deepcopy(request))
        self.deadlines.append(deadline_monotonic)
        self.tokens.append(cancellation_token)
        if not self.responses:
            raise AssertionError("scripted model was called more often than expected")
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        if callable(response):
            return response(
                request,
                cancellation_token=cancellation_token,
                deadline_monotonic=deadline_monotonic,
            )
        return response


def _tool_response(*calls: ToolCall) -> HarnessModelResponse:
    return HarnessModelResponse(kind="tool_calls", tool_calls=tuple(calls))


def _final_response(message: str = "done") -> HarnessModelResponse:
    return HarnessModelResponse(kind="final", output={"message": message})


def _tool_call(
    call_id: str,
    name: str = "echo",
    arguments: Mapping[str, Any] | None = None,
    *,
    idempotency_key: str | None = None,
) -> ToolCall:
    return ToolCall(
        call_id=call_id,
        name=name,
        arguments=dict(arguments or {"value": 1}),
        idempotency_key=idempotency_key,
    )


def _spec(
    *,
    name: str = "echo",
    permission: str = "workspace.read",
    timeout_seconds: float = 2.0,
    replay_policy: str = "safe",
) -> ToolSpec:
    return ToolSpec(
        name=name,
        version="1.0.0",
        description="Return one bounded integer for harness contract tests.",
        input_schema={
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
            "additionalProperties": False,
        },
        permission=permission,
        risk="low",
        timeout_seconds=timeout_seconds,
        replay_policy=replay_policy,
        execution_isolation="trusted_inline",
        trusted_inline_reason="bounded deterministic harness contract fixture",
    )


def _registry(
    handler: Callable[..., Mapping[str, Any]],
    *,
    spec: ToolSpec | None = None,
) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(spec or _spec(), handler)
    return registry


def _limits(
    *,
    max_steps: int = 8,
    max_repeated_tool_calls: int = 2,
    deadline_seconds: float = 20.0,
) -> HarnessLimits:
    return HarnessLimits(
        max_steps=max_steps,
        max_tool_calls_per_step=3,
        max_total_tool_calls=12,
        max_repeated_tool_calls=max_repeated_tool_calls,
        deadline_seconds=deadline_seconds,
    )


def _retry_policy(
    *,
    max_attempts: int = 1,
    initial_backoff_seconds: float = 0.0,
) -> RetryPolicy:
    return RetryPolicy(
        max_attempts=max_attempts,
        initial_backoff_seconds=initial_backoff_seconds,
        backoff_multiplier=2.0,
        max_backoff_seconds=10.0,
    )


def _run(
    model: _ScriptedModel,
    registry: ToolRegistry,
    *,
    clock: _FakeClock | None = None,
    limits: HarnessLimits | None = None,
    retry_policy: RetryPolicy | None = None,
    cancellation_token: CancellationToken | None = None,
    allowed_permissions: set[str] | None = None,
    context: Mapping[str, Any] | None = None,
    event_sink: Callable[[Mapping[str, Any]], None] | None = None,
    checkpoint_sink: Callable[[HarnessCheckpoint], None] | None = None,
) -> dict[str, Any]:
    return run_agent_harness(
        model,
        registry,
        dict(context or {"task": "inspect the repository"}),
        run_id="run_contract_001",
        turn_id="turn_contract_001",
        limits=limits or _limits(),
        retry_policy=retry_policy or _retry_policy(),
        allowed_permissions=(
            allowed_permissions
            if allowed_permissions is not None
            else {"workspace.read"}
        ),
        clock=clock or _FakeClock(),
        cancellation_token=cancellation_token or CancellationToken(),
        event_sink=event_sink,
        checkpoint_sink=checkpoint_sink,
    )


def _event(result: Mapping[str, Any], event_type: str) -> dict[str, Any]:
    return next(item for item in result["events"] if item["type"] == event_type)


def _nested_keys(value: Any) -> set[str]:
    if isinstance(value, Mapping):
        return {
            *(str(key) for key in value),
            *(
                child
                for item in value.values()
                for child in _nested_keys(item)
            ),
        }
    if isinstance(value, (list, tuple)):
        return {child for item in value for child in _nested_keys(item)}
    return set()


class AgentHarnessContractTests(unittest.TestCase):
    def test_non_safe_validation_failure_does_not_create_an_effect_fence(self) -> None:
        model = _ScriptedModel(
            [
                _tool_response(_tool_call("call_invalid_effect", name="write_value")),
                _final_response("continued after validation failure"),
            ]
        )

        def reject_before_effect(
            _arguments: Mapping[str, Any], _context: Any
        ) -> Mapping[str, Any]:
            raise ToolExecutionError("preflight rejected", code="preflight_rejected")

        result = _run(
            model,
            _registry(
                reject_before_effect,
                spec=_spec(
                    name="write_value",
                    permission="workspace.write",
                    replay_policy="never",
                ),
            ),
            allowed_permissions={"workspace.write"},
        )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(model.calls, 2)
        self.assertFalse(
            any(item["type"] == "tool.effect_started" for item in result["events"])
        )
        self.assertEqual(_event(result, "tool.failed")["payload"]["error_code"], "preflight_rejected")

    def test_unsettled_non_safe_effect_has_one_authoritative_handoff(self) -> None:
        model = _ScriptedModel(
            [
                _tool_response(_tool_call("call_partial_effect", name="write_value")),
                _final_response("must not run after an uncertain effect"),
            ]
        )

        def fail_after_boundary(
            _arguments: Mapping[str, Any], context: Any
        ) -> Mapping[str, Any]:
            context.begin_effect()
            raise ToolExecutionError("partial write", code="partial_write")

        result = _run(
            model,
            _registry(
                fail_after_boundary,
                spec=_spec(
                    name="write_value",
                    permission="workspace.write",
                    replay_policy="never",
                ),
            ),
            allowed_permissions={"workspace.write"},
        )

        self.assertEqual(result["status"], "handoff")
        self.assertEqual(model.calls, 1)
        self.assertEqual(_event(result, "tool.failed")["payload"]["error_code"], "partial_write")
        self.assertTrue(any(item["type"] == "tool.effect_started" for item in result["events"]))
        terminal = [item for item in result["events"] if item["type"] in _TERMINAL_EVENTS]
        self.assertEqual([item["type"] for item in terminal], ["run.handoff"])
        self.assertEqual(
            terminal[0]["payload"]["reason_code"],
            "external_effect_unsettled",
        )
        self.assertIsNotNone(result["checkpoint"].pending_effect)
        self.assertTrue(result["checkpoint"].pending_effect["effect_started"])

    def test_non_replayable_oversized_result_is_not_reported_as_failed(self) -> None:
        model = _ScriptedModel(
            [
                _tool_response(
                    _tool_call("call_large", name="write_large")
                ),
                _final_response("continued after the settled effect"),
            ]
        )
        spec = ToolSpec(
            name="write_large",
            version="1.0.0",
            description="Return an oversized receipt after one external effect.",
            input_schema={
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
                "additionalProperties": False,
            },
            output_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False,
            },
            permission="workspace.write",
            replay_policy="never",
            execution_isolation="trusted_inline",
            trusted_inline_reason="bounded contract fixture with a large synthetic result",
        )
        limits = HarnessLimits(
            max_steps=4,
            max_tool_calls_per_step=2,
            max_total_tool_calls=4,
            max_repeated_tool_calls=2,
            max_tool_output_chars=1_000,
            deadline_seconds=20,
        )

        def oversized_effect(
            _arguments: Mapping[str, Any], context: Any
        ) -> Mapping[str, Any]:
            context.begin_effect()
            return {"text": "x" * 10_000}

        result = _run(
            model,
            _registry(oversized_effect, spec=spec),
            limits=limits,
            allowed_permissions={"workspace.write"},
        )

        completed = _event(result, "tool.completed")
        self.assertTrue(completed["payload"]["result"]["truncated"])
        self.assertFalse(any(item["type"] == "tool.failed" for item in result["events"]))
        self.assertEqual(result["status"], "completed")

    def test_events_are_monotonic_and_have_exactly_one_terminal_event(self) -> None:
        observed_events: list[dict[str, Any]] = []
        observed_checkpoints: list[HarnessCheckpoint] = []
        model = _ScriptedModel(
            [
                _tool_response(_tool_call("call_1")),
                _final_response("completed answer"),
            ]
        )
        registry = _registry(lambda arguments, _context: dict(arguments))

        result = _run(
            model,
            registry,
            event_sink=lambda item: observed_events.append(deepcopy(dict(item))),
            checkpoint_sink=observed_checkpoints.append,
        )

        self.assertEqual(result["status"], "completed")
        self.assertIsInstance(result["checkpoint"], HarnessCheckpoint)
        # Two model completions plus tool pending and tool receipt are all
        # independently resumable boundaries.
        self.assertGreaterEqual(len(observed_checkpoints), 4)
        self.assertTrue(
            all(isinstance(item, HarnessCheckpoint) for item in observed_checkpoints)
        )
        self.assertEqual(observed_events, result["events"])
        self.assertEqual(
            [item["sequence"] for item in result["events"]],
            list(range(1, len(result["events"]) + 1)),
        )
        self.assertTrue(
            all(
                item["run_id"] == "run_contract_001"
                and item["turn_id"] == "turn_contract_001"
                for item in result["events"]
            )
        )
        terminal = [
            item for item in result["events"] if item["type"] in _TERMINAL_EVENTS
        ]
        self.assertEqual(len(terminal), 1)
        self.assertEqual(terminal[0]["type"], "run.completed")
        self.assertEqual(result["events"][-1], terminal[0])

    def test_invalid_tool_arguments_are_rejected_before_handler_execution(self) -> None:
        calls = 0

        def handler(
            arguments: Mapping[str, Any], context: Any
        ) -> Mapping[str, Any]:
            nonlocal calls
            context.begin_effect()
            calls += 1
            return dict(arguments)

        model = _ScriptedModel(
            [
                _tool_response(
                    _tool_call("call_invalid", arguments={"value": "not-an-integer"})
                ),
                _final_response("recovered after validation error"),
            ]
        )

        result = _run(model, _registry(handler))

        self.assertEqual(calls, 0)
        rejected = _event(result, "tool.rejected")
        self.assertEqual(rejected["payload"]["error_code"], "invalid_arguments")
        self.assertEqual(rejected["payload"]["tool_name"], "echo")
        self.assertEqual(result["status"], "completed")

    def test_permission_is_checked_before_tool_handler_execution(self) -> None:
        calls = 0

        def handler(
            arguments: Mapping[str, Any], context: Any
        ) -> Mapping[str, Any]:
            nonlocal calls
            context.begin_effect()
            calls += 1
            return dict(arguments)

        model = _ScriptedModel(
            [
                _tool_response(_tool_call("call_private", name="private_state")),
                _final_response("continued without the forbidden tool"),
            ]
        )
        registry = _registry(
            handler,
            spec=_spec(name="private_state", permission="workspace.secret.read"),
        )

        result = _run(
            model,
            registry,
            allowed_permissions={"workspace.read"},
        )

        self.assertEqual(calls, 0)
        rejected = _event(result, "tool.rejected")
        self.assertEqual(rejected["payload"]["error_code"], "permission_denied")
        self.assertEqual(rejected["payload"]["required_permission"], "workspace.secret.read")

    def test_invalid_external_permission_configuration_fails_closed(self) -> None:
        registry = _registry(
            lambda arguments, _context: dict(arguments),
            spec=_spec(permission="workspace.secret.read"),
        )
        model = _ScriptedModel([_final_response()])

        with self.assertRaises((HarnessContractError, ToolPermissionError)):
            _run(
                model,
                registry,
                allowed_permissions={"*"},
            )
        self.assertEqual(model.calls, 0)

    def test_tool_timeout_is_observed_with_the_deterministic_clock(self) -> None:
        clock = _FakeClock()
        calls = 0

        def slow_tool(
            arguments: Mapping[str, Any], _context: Any
        ) -> Mapping[str, Any]:
            nonlocal calls
            calls += 1
            clock.advance(1.25)
            return dict(arguments)

        model = _ScriptedModel(
            [
                _tool_response(_tool_call("call_slow")),
                _final_response("used the timeout observation"),
            ]
        )
        registry = _registry(
            slow_tool,
            spec=_spec(timeout_seconds=1.0),
        )

        result = _run(model, registry, clock=clock)

        self.assertEqual(calls, 1)
        failed = _event(result, "tool.failed")
        self.assertEqual(failed["payload"]["error_code"], "timeout")
        self.assertEqual(failed["payload"]["tool_name"], "echo")
        self.assertFalse(
            any(
                item["type"] == "tool.completed"
                and item["payload"].get("call_id") == "call_slow"
                for item in result["events"]
            )
        )

    def test_cancellation_wins_over_a_model_response_and_emits_no_action(self) -> None:
        token = CancellationToken()

        def cancel_during_plan(
            _request: Any,
            *,
            cancellation_token: CancellationToken,
            deadline_monotonic: float,
        ) -> HarnessModelResponse:
            del deadline_monotonic
            cancellation_token.cancel("user_requested")
            return _final_response("must not be committed")

        model = _ScriptedModel([cancel_during_plan])
        registry = _registry(lambda arguments, _context: dict(arguments))

        result = _run(
            model,
            registry,
            cancellation_token=token,
        )

        self.assertEqual(result["status"], "cancelled")
        terminal = [
            item for item in result["events"] if item["type"] in _TERMINAL_EVENTS
        ]
        self.assertEqual([item["type"] for item in terminal], ["run.cancelled"])
        self.assertFalse(any(item["type"] == "action.completed" for item in result["events"]))

    def test_repeated_call_budget_uses_tool_and_arguments_not_call_id(self) -> None:
        calls = 0

        def handler(
            arguments: Mapping[str, Any], _context: Any
        ) -> Mapping[str, Any]:
            nonlocal calls
            calls += 1
            return dict(arguments)

        model = _ScriptedModel(
            [
                _tool_response(_tool_call("call_first", arguments={"value": 7})),
                _tool_response(_tool_call("call_second", arguments={"value": 7})),
            ]
        )

        result = _run(
            model,
            _registry(handler),
            limits=_limits(max_repeated_tool_calls=1),
        )

        self.assertEqual(calls, 1)
        self.assertEqual(result["status"], "failed")
        guard = _event(result, "guard.triggered")
        self.assertEqual(guard["payload"]["reason_code"], "repeated_tool_call")
        self.assertNotIn("call_first", guard["payload"].values())
        self.assertEqual(result["events"][-1]["type"], "run.failed")

    def test_call_id_cannot_be_reused_across_model_steps(self) -> None:
        executed: list[int] = []

        def handler(
            arguments: Mapping[str, Any], _context: Any
        ) -> Mapping[str, Any]:
            value = int(arguments["value"])
            executed.append(value)
            return {"value": value}

        model = _ScriptedModel(
            [
                _tool_response(
                    _tool_call("call_reused", arguments={"value": 1})
                ),
                _tool_response(
                    _tool_call("call_reused", arguments={"value": 2})
                ),
                _final_response("must not reach the final step"),
            ]
        )

        result = _run(model, _registry(handler))

        self.assertEqual(executed, [1])
        self.assertEqual(result["status"], "failed")
        starts = [
            item
            for item in result["events"]
            if item["type"] == "tool.started"
            and item["payload"].get("call_id") == "call_reused"
        ]
        self.assertEqual(len(starts), 1)
        self.assertEqual(result["events"][-1]["type"], "run.failed")

    def test_model_retry_never_starts_after_the_run_deadline(self) -> None:
        clock = _FakeClock(now=10.0)

        def consume_one_second_then_fail(
            _request: Any,
            *,
            cancellation_token: CancellationToken,
            deadline_monotonic: float,
        ) -> HarnessModelResponse:
            del cancellation_token
            self.assertEqual(deadline_monotonic, 13.0)
            clock.advance(1.0)
            raise TimeoutError("transient provider timeout")

        model = _ScriptedModel([consume_one_second_then_fail])
        registry = _registry(lambda arguments, _context: dict(arguments))

        result = _run(
            model,
            registry,
            clock=clock,
            limits=_limits(deadline_seconds=3.0),
            retry_policy=_retry_policy(
                max_attempts=5,
                initial_backoff_seconds=5.0,
            ),
        )

        self.assertEqual(model.calls, 1)
        self.assertEqual(result["status"], "deadline_exceeded")
        self.assertLessEqual(clock.monotonic(), 13.0)
        failed = result["events"][-1]
        self.assertEqual(failed["type"], "run.failed")
        self.assertEqual(failed["payload"]["error_code"], "deadline_exceeded")
        self.assertTrue(
            any(item["type"] == "model.retry_suppressed" for item in result["events"])
        )

    def test_same_tool_idempotency_key_reuses_receipt_without_reexecution(self) -> None:
        calls = 0

        def handler(
            arguments: Mapping[str, Any], context: Any
        ) -> Mapping[str, Any]:
            nonlocal calls
            context.begin_effect()
            calls += 1
            return dict(arguments)

        model = _ScriptedModel(
            [
                _tool_response(
                    _tool_call(
                        "call_a",
                        arguments={"value": 9},
                        idempotency_key="idem-9",
                    )
                ),
                _tool_response(
                    _tool_call(
                        "call_b",
                        arguments={"value": 9},
                        idempotency_key="idem-9",
                    )
                ),
                _final_response(),
            ]
        )
        registry = _registry(
            handler,
            spec=_spec(replay_policy="idempotent"),
        )

        result = _run(
            model,
            registry,
            limits=_limits(max_repeated_tool_calls=4),
        )

        self.assertEqual(calls, 1)
        replay = _event(result, "tool.replayed")
        self.assertEqual(replay["payload"]["idempotency_key"], "idem-9")
        self.assertEqual(replay["payload"]["source_call_id"], "call_a")
        self.assertEqual(result["status"], "completed")

    def test_resume_does_not_replay_an_uncertain_never_replay_tool(self) -> None:
        clock = _FakeClock()
        checkpoints: list[HarnessCheckpoint] = []
        executions = 0

        def process_payment(
            arguments: Mapping[str, Any], execution_context: Any
        ) -> Mapping[str, Any]:
            nonlocal executions
            executions += 1
            self.assertEqual(execution_context.idempotency_key, "payment-42")
            self.assertIs(execution_context.cancellation_token, token)
            # A tool receives the tighter of its own timeout and the run deadline.
            self.assertEqual(execution_context.deadline_monotonic, 102.0)
            self.assertEqual(arguments, {"value": 42})
            # Simulate a process death after the external side effect may have
            # committed but before the harness can persist its receipt.
            raise SystemExit("simulated process death after side effect")

        token = CancellationToken()
        first_model = _ScriptedModel(
            [
                _tool_response(
                    _tool_call(
                        "charge_once",
                        name="charge_card",
                        arguments={"value": 42},
                        idempotency_key="payment-42",
                    )
                )
            ]
        )
        registry = _registry(
            process_payment,
            spec=_spec(
                name="charge_card",
                permission="payments.write",
                replay_policy="never",
            ),
        )

        with self.assertRaisesRegex(SystemExit, "simulated process death"):
            _run(
                first_model,
                registry,
                clock=clock,
                cancellation_token=token,
                allowed_permissions={"payments.write"},
                checkpoint_sink=checkpoints.append,
            )

        self.assertEqual(executions, 1)
        self.assertGreaterEqual(len(checkpoints), 2)
        self.assertTrue(all(isinstance(item, HarnessCheckpoint) for item in checkpoints))

        resume_inputs: tuple[HarnessCheckpoint | Mapping[str, Any], ...] = (
            checkpoints[-1],
            asdict(checkpoints[-1]),
        )
        for checkpoint_value in resume_inputs:
            with self.subTest(checkpoint_type=type(checkpoint_value).__name__):
                resumed_events: list[dict[str, Any]] = []
                resumed_model = _ScriptedModel(
                    [_final_response("must not be reached")]
                )
                resumed = resume_agent_harness(
                    checkpoint_value,
                    resumed_model,
                    registry,
                    {"task": "inspect the repository"},
                    limits=_limits(),
                    retry_policy=_retry_policy(),
                    allowed_permissions={"payments.write"},
                    clock=clock,
                    cancellation_token=CancellationToken(),
                    event_sink=lambda item: resumed_events.append(
                        deepcopy(dict(item))
                    ),
                )

                self.assertEqual(executions, 1)
                self.assertEqual(resumed_model.calls, 0)
                self.assertEqual(resumed["status"], "handoff")
                self.assertTrue(resumed_events)
                self.assertEqual(
                    [item["sequence"] for item in resumed_events],
                    list(
                        range(
                            checkpoints[-1].next_sequence,
                            checkpoints[-1].next_sequence
                            + len(resumed_events),
                        )
                    ),
                )
                self.assertEqual(
                    resumed["events"][-len(resumed_events) :],
                    resumed_events,
                )
                self.assertEqual(resumed["events"][-1]["type"], "run.handoff")
                self.assertEqual(
                    resumed["events"][-1]["payload"]["reason_code"],
                    "unsafe_tool_replay_blocked",
                )
                self.assertFalse(
                    any(
                        item["type"] == "tool.started"
                        and item["payload"].get("call_id") == "charge_once"
                        for item in resumed_events
                    )
                )

    def test_public_projection_excludes_context_model_and_tool_payloads(self) -> None:
        private_marker = "PRIVATE_INPUT_DO_NOT_LEAK"
        private_number = 999_999_999_999_997
        model = _ScriptedModel(
            [
                _tool_response(
                    _tool_call(
                        "call_secret",
                        arguments={"value": private_number},
                        idempotency_key="secret-idempotency-key",
                    )
                ),
                _final_response(f"answer containing {private_marker}"),
            ]
        )
        registry = _registry(
            lambda arguments, _context: {
                "value": int(arguments["value"]),
            }
        )

        result = _run(
            model,
            registry,
            context={
                "user_text": private_marker,
                "private_profile": {"email": private_marker},
            },
        )
        public = public_harness_trace(result)
        serialized = json.dumps(public, ensure_ascii=False, sort_keys=True)

        self.assertNotIn(private_marker, serialized)
        self.assertNotIn("secret-idempotency-key", serialized)
        self.assertNotIn(str(private_number), serialized)
        self.assertTrue(public.get("trace_sha256"))
        self.assertTrue(public.get("events"))
        forbidden_keys = {
            "arguments",
            "context",
            "model_request",
            "model_response",
            "output",
            "raw_payload",
            "result",
        }
        self.assertFalse(forbidden_keys & _nested_keys(public))

    def test_tool_spec_rejects_unknown_replay_policy(self) -> None:
        with self.assertRaises(HarnessContractError):
            _registry(
                lambda arguments, _context: dict(arguments),
                spec=_spec(replay_policy="sometimes"),
            )


if __name__ == "__main__":
    unittest.main()
