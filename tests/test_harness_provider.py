from __future__ import annotations

import json
import time
import unittest
from typing import Any, Iterator

from teaching_skill_miner.harness import (
    CancellationToken,
    DeepSeekChatHarnessModel,
    HarnessLimits,
    HarnessModelRequest,
    HarnessModelResponse,
    ProviderCapabilities,
    ProviderErrorKind,
    ProviderFailure,
    ProviderModelSpec,
    ProviderStreamEvent,
    RetryPolicy,
    ToolCall,
    ToolRegistry,
    ToolSpec,
    public_harness_trace,
    run_agent_harness,
)
from teaching_skill_miner.deepseek_client import DeepSeekClient, DeepSeekConfig


class _StreamingModel:
    def plan(self, *_args: Any, **_kwargs: Any) -> HarnessModelResponse:
        raise AssertionError("stream-capable model must use plan_stream")

    def plan_stream(
        self,
        _request: Any,
        *,
        cancellation_token: CancellationToken,
        deadline_monotonic: float,
    ) -> Iterator[ProviderStreamEvent | HarnessModelResponse]:
        del deadline_monotonic
        cancellation_token.raise_if_cancelled()
        yield ProviderStreamEvent(type="message.start")
        yield ProviderStreamEvent(type="message.delta", delta="真实")
        yield ProviderStreamEvent(type="message.delta", delta="流式")
        yield ProviderStreamEvent(
            type="usage.update", payload={"output_tokens": 2}
        )
        yield ProviderStreamEvent(type="message.end")
        yield HarnessModelResponse(
            kind="final", output={"message": "真实流式"}
        )


class _RateLimitError(RuntimeError):
    # Runtime must ignore exception-controlled retry hints.  Only the typed
    # provider classifier may grant retry authority.
    retryable = True


class _FakeClock:
    def __init__(
        self,
        *,
        now: float = 100.0,
        cancel_token: CancellationToken | None = None,
    ) -> None:
        self.now = now
        self.sleeps: list[float] = []
        self.cancel_token = cancel_token

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds
        if self.cancel_token is not None:
            self.cancel_token.cancel("cancel_during_provider_backoff")
            self.cancel_token = None


def _provider_spec(
    *, model: str = "classified-v1", native_stream: bool = False
) -> ProviderModelSpec:
    return ProviderModelSpec(
        provider="test-provider",
        model=model,
        capabilities=ProviderCapabilities(
            provider="test-provider",
            model=model,
            structured_output=True,
            native_stream=native_stream,
            cancellation=True,
        ),
        context_window_tokens=8_192,
        maximum_output_tokens=1_024,
    )


class _ClassifiedModel:
    def __init__(self, responses: list[Any], classifier: Any) -> None:
        self.responses = list(responses)
        self.classifier = classifier
        self.calls = 0

    @property
    def model_spec(self) -> ProviderModelSpec:
        return _provider_spec()

    def plan(
        self,
        _request: Any,
        *,
        cancellation_token: CancellationToken,
        deadline_monotonic: float,
    ) -> HarnessModelResponse:
        del deadline_monotonic
        cancellation_token.raise_if_cancelled()
        self.calls += 1
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    def classify_error(self, error: BaseException) -> ProviderFailure:
        if callable(self.classifier):
            return self.classifier(error)
        return self.classifier


class _DeltaThenFailureModel:
    def __init__(self) -> None:
        self.calls = 0

    @property
    def model_spec(self) -> ProviderModelSpec:
        return _provider_spec(native_stream=True)

    def plan(self, *_args: Any, **_kwargs: Any) -> HarnessModelResponse:
        raise AssertionError("stream-capable model must use plan_stream")

    def plan_stream(
        self,
        _request: Any,
        *,
        cancellation_token: CancellationToken,
        deadline_monotonic: float,
    ) -> Iterator[ProviderStreamEvent | HarnessModelResponse]:
        del deadline_monotonic
        cancellation_token.raise_if_cancelled()
        self.calls += 1
        if self.calls == 1:
            yield ProviderStreamEvent(type="message.start")
            yield ProviderStreamEvent(type="message.delta", delta="partial")
            raise _RateLimitError("429 after partial output")
        yield HarnessModelResponse(kind="final", output={"message": "duplicate"})

    def classify_error(self, _error: BaseException) -> ProviderFailure:
        return ProviderFailure(
            kind=ProviderErrorKind.RATE_LIMIT,
            retryable=True,
            safe_code="test_rate_limit",
        )


class _EffectThenDeepSeekModel:
    def __init__(self, deepseek: DeepSeekChatHarnessModel) -> None:
        self.deepseek = deepseek
        self.calls = 0

    @property
    def model_spec(self) -> ProviderModelSpec:
        return self.deepseek.model_spec

    def classify_error(self, error: BaseException) -> ProviderFailure:
        return self.deepseek.classify_error(error)

    def plan(self, *_args: Any, **_kwargs: Any) -> HarnessModelResponse:
        raise AssertionError("stream-capable model must use plan_stream")

    def plan_stream(
        self,
        request: HarnessModelRequest,
        *,
        cancellation_token: CancellationToken,
        deadline_monotonic: float,
    ) -> Iterator[ProviderStreamEvent | HarnessModelResponse]:
        self.calls += 1
        if self.calls == 1:
            yield HarnessModelResponse(
                kind="tool_calls",
                tool_calls=(
                    ToolCall(
                        call_id="effect-before-provider",
                        name="effect",
                        arguments={"value": 1},
                    ),
                ),
            )
            return
        yield from self.deepseek.plan_stream(
            request,
            cancellation_token=cancellation_token,
            deadline_monotonic=deadline_monotonic,
        )


class HarnessProviderTests(unittest.TestCase):
    def test_capabilities_are_explicit_and_json_safe(self) -> None:
        capabilities = ProviderCapabilities(
            provider="deepseek",
            model="deepseek-v4-flash",
            structured_output=True,
            native_stream=True,
            web_search=True,
            cancellation=True,
        )

        encoded = json.dumps(capabilities.to_dict(), sort_keys=True)
        self.assertIn('"native_stream": true', encoded)
        self.assertIn('"cancellation": true', encoded)

    def test_rate_limit_classifier_drives_retry_after_and_safe_code(self) -> None:
        failure = ProviderFailure(
            kind=ProviderErrorKind.RATE_LIMIT,
            retryable=True,
            safe_code="test_rate_limit",
            retry_after_seconds=0.2,
        )
        model = _ClassifiedModel(
            [
                _RateLimitError("HTTP 429 private provider payload"),
                HarnessModelResponse(
                    kind="final", output={"message": "retried safely"}
                ),
            ],
            failure,
        )
        clock = _FakeClock()

        result = run_agent_harness(
            model,
            ToolRegistry(),
            {"mode": "classified-retry"},
            limits=HarnessLimits(deadline_seconds=2.0),
            retry_policy=RetryPolicy(
                max_attempts=2,
                initial_backoff_seconds=0.05,
                max_backoff_seconds=1.0,
            ),
            allowed_permissions={"tool.read"},
            clock=clock,
        )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(model.calls, 2)
        self.assertAlmostEqual(sum(clock.sleeps), 0.2)
        failed = next(
            event for event in result["events"] if event["type"] == "model.failed"
        )
        self.assertEqual(failed["payload"]["error_kind"], "rate_limit")
        self.assertEqual(failed["payload"]["safe_code"], "test_rate_limit")
        self.assertEqual(failed["payload"]["retry_after_ms"], 200)
        retrying = next(
            event
            for event in result["events"]
            if event["type"] == "model.retrying"
        )
        self.assertEqual(retrying["payload"]["delay_ms"], 200)
        self.assertNotIn(
            "private provider payload", json.dumps(result, default=str)
        )

    def test_invalid_or_throwing_classifier_fails_closed_without_retry(self) -> None:
        invalid_classifiers = (
            None,
            ProviderFailure(
                kind="rate_limit",  # type: ignore[arg-type]
                retryable=True,
                safe_code="invalid_kind",
            ),
            lambda _error: (_ for _ in ()).throw(
                RuntimeError("private classifier failure")
            ),
        )
        for classifier in invalid_classifiers:
            with self.subTest(classifier=type(classifier).__name__):
                model = _ClassifiedModel(
                    [_RateLimitError("private transport failure")], classifier
                )
                result = run_agent_harness(
                    model,
                    ToolRegistry(),
                    {"mode": "invalid-classifier"},
                    limits=HarnessLimits(deadline_seconds=2.0),
                    retry_policy=RetryPolicy(max_attempts=3),
                    allowed_permissions={"tool.read"},
                )
                self.assertEqual(model.calls, 1)
                self.assertEqual(result["status"], "failed")
                self.assertEqual(result["reason"], "provider_error_unclassified")
                failed = next(
                    event
                    for event in result["events"]
                    if event["type"] == "model.failed"
                )
                self.assertFalse(failed["payload"]["classifier_valid"])
                self.assertFalse(failed["payload"]["retryable"])
                self.assertEqual(
                    failed["payload"]["safe_code"],
                    "provider_error_unclassified",
                )
                self.assertFalse(
                    any(
                        event["type"] == "model.retrying"
                        for event in result["events"]
                    )
                )

    def test_model_delta_suppresses_a_classified_transport_retry(self) -> None:
        model = _DeltaThenFailureModel()
        result = run_agent_harness(
            model,
            ToolRegistry(),
            {"mode": "delta-retry-fence"},
            limits=HarnessLimits(deadline_seconds=2.0),
            retry_policy=RetryPolicy(max_attempts=3),
            allowed_permissions={"tool.read"},
        )

        self.assertEqual(model.calls, 1)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["reason"], "test_rate_limit")
        suppressed = next(
            event
            for event in result["events"]
            if event["type"] == "model.retry_suppressed"
        )
        self.assertEqual(
            suppressed["payload"]["reason_code"], "model_delta_emitted"
        )

    def test_external_tool_effect_suppresses_later_model_retry(self) -> None:
        calls = 0

        def effect(arguments: Any, _context: Any) -> dict[str, Any]:
            nonlocal calls
            calls += 1
            return dict(arguments)

        registry = ToolRegistry()
        registry.register(
            ToolSpec(
                name="effect",
                version="1.0.0",
                description="Settle one external effect.",
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
                permission="effect.write",
                replay_policy="never",
                execution_isolation="trusted_inline",
                trusted_inline_reason="bounded in-memory provider fence fixture",
            ),
            effect,
        )
        model = _ClassifiedModel(
            [
                HarnessModelResponse(
                    kind="tool_calls",
                    tool_calls=(
                        ToolCall(
                            call_id="effect-1",
                            name="effect",
                            arguments={"value": 1},
                        ),
                    ),
                ),
                _RateLimitError("429 after effect"),
                HarnessModelResponse(kind="final", output={"message": "unsafe"}),
            ],
            ProviderFailure(
                kind=ProviderErrorKind.RATE_LIMIT,
                retryable=True,
                safe_code="test_rate_limit",
            ),
        )

        result = run_agent_harness(
            model,
            registry,
            {"mode": "effect-retry-fence"},
            limits=HarnessLimits(deadline_seconds=2.0),
            retry_policy=RetryPolicy(max_attempts=3),
            allowed_permissions={"effect.write"},
        )

        self.assertEqual(calls, 1)
        self.assertEqual(model.calls, 2)
        self.assertEqual(result["status"], "failed")
        suppressed = next(
            event
            for event in result["events"]
            if event["type"] == "model.retry_suppressed"
        )
        self.assertEqual(
            suppressed["payload"]["reason_code"], "external_effect_started"
        )

    def test_rejected_tool_does_not_claim_an_external_effect(self) -> None:
        model = _ClassifiedModel(
            [
                HarnessModelResponse(
                    kind="tool_calls",
                    tool_calls=(
                        ToolCall(call_id="unknown-1", name="not_registered"),
                    ),
                ),
                _RateLimitError("429 before any handler started"),
                HarnessModelResponse(
                    kind="final", output={"message": "retried safely"}
                ),
            ],
            ProviderFailure(
                kind=ProviderErrorKind.RATE_LIMIT,
                retryable=True,
                safe_code="test_rate_limit",
            ),
        )
        result = run_agent_harness(
            model,
            ToolRegistry(),
            {"mode": "rejected-tool-is-not-an-effect"},
            limits=HarnessLimits(deadline_seconds=2.0),
            retry_policy=RetryPolicy(max_attempts=2),
            allowed_permissions={"tool.read"},
        )

        self.assertEqual(model.calls, 3)
        self.assertEqual(result["status"], "completed")
        self.assertTrue(
            any(event["type"] == "model.retrying" for event in result["events"])
        )
        self.assertFalse(
            any(
                event["type"] == "model.retry_suppressed"
                and event["payload"].get("reason_code")
                == "external_effect_started"
                for event in result["events"]
            )
        )

    def test_deepseek_harness_disables_nested_retries_after_an_effect(self) -> None:
        transport_calls = 0

        def rate_limited_stream(*_args: Any) -> tuple[int, list[bytes]]:
            nonlocal transport_calls
            transport_calls += 1
            return 429, []

        client = DeepSeekClient(
            DeepSeekConfig(
                allow_remote_student_data=True,
                # This legacy client policy would make four attempts if the
                # harness adapter did not explicitly override it.
                max_retries=3,
            ),
            api_key="test-key",
            stream_transport=rate_limited_stream,
        )
        model = _EffectThenDeepSeekModel(DeepSeekChatHarnessModel(client))
        effect_calls = 0

        def effect(arguments: Any, _context: Any) -> dict[str, Any]:
            nonlocal effect_calls
            effect_calls += 1
            return dict(arguments)

        registry = ToolRegistry()
        registry.register(
            ToolSpec(
                name="effect",
                version="1.0.0",
                description="Settle one effect before the provider call.",
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
                permission="effect.write",
                replay_policy="never",
                execution_isolation="trusted_inline",
                trusted_inline_reason="bounded in-memory provider fence fixture",
            ),
            effect,
        )
        result = run_agent_harness(
            model,
            registry,
            {"messages": [{"role": "user", "content": "answer"}]},
            limits=HarnessLimits(deadline_seconds=2.0),
            retry_policy=RetryPolicy(max_attempts=3),
            allowed_permissions={"effect.write"},
        )

        self.assertEqual(effect_calls, 1)
        self.assertEqual(transport_calls, 1)
        self.assertEqual(model.calls, 2)
        self.assertEqual(result["status"], "failed")
        suppressed = next(
            event
            for event in result["events"]
            if event["type"] == "model.retry_suppressed"
        )
        self.assertEqual(
            suppressed["payload"]["reason_code"], "external_effect_started"
        )

    def test_deepseek_buffered_delta_is_flushed_and_never_retried(self) -> None:
        transport_calls = 0

        def truncated_stream(*_args: Any) -> tuple[int, list[bytes]]:
            nonlocal transport_calls
            transport_calls += 1
            return 200, [
                b'data: {"choices":[{"delta":{"content":"partial"}}]}\n'
            ]

        client = DeepSeekClient(
            DeepSeekConfig(
                allow_remote_student_data=True,
                max_retries=3,
            ),
            api_key="test-key",
            stream_transport=truncated_stream,
        )
        result = run_agent_harness(
            DeepSeekChatHarnessModel(client),
            ToolRegistry(),
            {"messages": [{"role": "user", "content": "answer"}]},
            limits=HarnessLimits(deadline_seconds=2.0),
            retry_policy=RetryPolicy(max_attempts=3),
            allowed_permissions={"tool.read"},
        )

        self.assertEqual(transport_calls, 1)
        self.assertEqual(result["status"], "failed")
        delta = next(
            event
            for event in result["events"]
            if event["type"] == "message.delta"
        )
        self.assertEqual(delta["payload"]["delta"], "partial")
        suppressed = next(
            event
            for event in result["events"]
            if event["type"] == "model.retry_suppressed"
        )
        self.assertEqual(
            suppressed["payload"]["reason_code"], "model_delta_emitted"
        )

    def test_retry_after_respects_deadline_and_cancellation(self) -> None:
        failure = ProviderFailure(
            kind=ProviderErrorKind.RATE_LIMIT,
            retryable=True,
            safe_code="test_rate_limit",
            retry_after_seconds=1.0,
        )
        deadline_model = _ClassifiedModel([_RateLimitError("429")], failure)
        deadline_result = run_agent_harness(
            deadline_model,
            ToolRegistry(),
            {"mode": "deadline-fence"},
            limits=HarnessLimits(deadline_seconds=0.5),
            retry_policy=RetryPolicy(max_attempts=2),
            allowed_permissions={"tool.read"},
            clock=_FakeClock(),
        )
        self.assertEqual(deadline_model.calls, 1)
        self.assertEqual(deadline_result["status"], "deadline_exceeded")
        deadline_suppressed = next(
            event
            for event in deadline_result["events"]
            if event["type"] == "model.retry_suppressed"
        )
        self.assertEqual(
            deadline_suppressed["payload"]["reason_code"], "run_deadline"
        )

        token = CancellationToken()
        cancel_model = _ClassifiedModel([_RateLimitError("429")], failure)
        cancel_result = run_agent_harness(
            cancel_model,
            ToolRegistry(),
            {"mode": "cancel-fence"},
            limits=HarnessLimits(deadline_seconds=2.0),
            retry_policy=RetryPolicy(max_attempts=2),
            allowed_permissions={"tool.read"},
            clock=_FakeClock(cancel_token=token),
            cancellation_token=token,
        )
        self.assertEqual(cancel_model.calls, 1)
        self.assertEqual(cancel_result["status"], "cancelled")
        self.assertTrue(
            any(
                event["type"] == "model.retrying"
                for event in cancel_result["events"]
            )
        )

    def test_native_stream_events_arrive_before_durable_final_action(self) -> None:
        observed: list[dict[str, Any]] = []
        result = run_agent_harness(
            _StreamingModel(),
            ToolRegistry(),
            {"mode": "chat"},
            run_id="run_stream",
            turn_id="turn_stream",
            limits=HarnessLimits(deadline_seconds=2.0),
            retry_policy=RetryPolicy(max_attempts=1),
            allowed_permissions={"tool.read"},
            event_sink=lambda event: observed.append(dict(event)),
        )

        self.assertEqual(result["status"], "completed")
        types = [event["type"] for event in observed]
        self.assertLess(types.index("message.delta"), types.index("action.completed"))
        self.assertEqual(types.count("message.delta"), 2)
        self.assertEqual(result["output"], {"message": "真实流式"})
        public = public_harness_trace(result)
        serialized = json.dumps(public, ensure_ascii=False)
        self.assertNotIn("真实", serialized)
        self.assertNotIn("流式", serialized)

    def test_cancellation_callback_runs_once_and_can_be_unsubscribed(self) -> None:
        token = CancellationToken()
        calls: list[str] = []
        unsubscribe = token.add_callback(lambda: calls.append("closed"))
        unsubscribe()

        self.assertTrue(token.cancel("user_requested"))
        self.assertEqual(calls, [])
        token.add_callback(lambda: calls.append("late-close"))
        self.assertEqual(calls, ["late-close"])
        self.assertFalse(token.cancel("again"))

    def test_deepseek_adapter_streams_real_provider_chunks_through_harness(self) -> None:
        lines = [
            b'data: {"id":"ds_stream","choices":[{"delta":{"content":"alpha "}}]}\n',
            b'data: {"id":"ds_stream","choices":[{"delta":{"content":"beta"},"finish_reason":"stop"}]}\n',
            b'data: {"id":"ds_stream","choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2,"prompt_cache_hit_tokens":2,"prompt_cache_miss_tokens":1}}\n',
            b"data: [DONE]\n",
        ]
        client = DeepSeekClient(
            DeepSeekConfig(allow_remote_student_data=True, max_retries=0),
            api_key="adapter-secret",
            stream_transport=lambda *_args: (200, list(lines)),
        )
        observed: list[dict[str, Any]] = []
        result = run_agent_harness(
            DeepSeekChatHarnessModel(client),
            ToolRegistry(),
            {"messages": [{"role": "user", "content": "stream please"}]},
            run_id="run_deepseek_stream",
            turn_id="turn_deepseek_stream",
            limits=HarnessLimits(deadline_seconds=3.0),
            retry_policy=RetryPolicy(max_attempts=1),
            allowed_permissions={"tool.read"},
            event_sink=lambda event: observed.append(dict(event)),
        )

        self.assertEqual(result["output"], {"message": "alpha beta"})
        deltas = [
            event["payload"]["delta"]
            for event in observed
            if event["type"] == "message.delta"
        ]
        self.assertEqual(deltas, ["alpha beta"])
        completed = next(
            event for event in observed if event["type"] == "model.completed"
        )
        self.assertEqual(completed["payload"]["provider_request_id"], "ds_stream")
        cache_usage = {
            "prompt_cache_hit_tokens": 2,
            "prompt_cache_miss_tokens": 1,
        }
        usage_update = next(
            event for event in observed if event["type"] == "usage.update"
        )
        self.assertEqual(
            {
                key: usage_update["payload"][key]
                for key in cache_usage
            },
            cache_usage,
        )
        self.assertEqual(
            {
                key: completed["payload"]["usage"][key]
                for key in cache_usage
            },
            cache_usage,
        )
        serializable = {
            **result,
            "checkpoint": result["checkpoint"].to_dict(),
        }
        self.assertNotIn(
            "adapter-secret", json.dumps(serializable, ensure_ascii=False)
        )

    def test_deepseek_adapter_coalesces_plain_and_web_provider_deltas(self) -> None:
        pieces = [f"{index % 10}" for index in range(320)]
        expected = "".join(pieces)
        request = HarnessModelRequest(
            run_id="run_coalesced",
            turn_id="turn_coalesced",
            step=1,
            context={"messages": [{"role": "user", "content": "answer"}]},
            observations=(),
            tools=(
                {
                    "name": "web_search",
                    "version": "deepseek-native-v1",
                    "description": "Search public web pages.",
                    "input_schema": {"type": "object"},
                    "permission": "chat.web_search",
                    "risk": "medium",
                    "execution_mode": "provider_managed",
                    "data_scope": "public_web",
                    "requires_user_consent": True,
                },
            ),
            state={},
        )

        plain_lines = [
            b"data: "
            + json.dumps(
                {
                    "id": "plain-coalesced",
                    "choices": [{"delta": {"content": piece}}],
                }
            ).encode("utf-8")
            + b"\n"
            for piece in pieces
        ]
        plain_lines.extend(
            [
                b'data: {"id":"plain-coalesced","choices":[{"delta":{},"finish_reason":"stop"}]}\n',
                b"data: [DONE]\n",
            ]
        )

        web_events: list[dict[str, Any]] = [
            {
                "type": "message_start",
                "message": {
                    "id": "web-coalesced",
                    "stop_reason": None,
                    "usage": {},
                },
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "server_tool_use",
                    "id": "srvtoolu-coalesced",
                    "name": "web_search",
                },
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": '{"query":"private-query"}',
                },
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {
                    "type": "web_search_tool_result",
                    "tool_use_id": "srvtoolu-coalesced",
                    "content": [
                        {
                            "type": "web_search_result",
                            "title": "Current source",
                            "url": "https://example.com/current",
                            "encrypted_content": "private-search-body",
                        }
                    ],
                },
            },
            {"type": "content_block_stop", "index": 1},
            {
                "type": "content_block_start",
                "index": 2,
                "content_block": {"type": "text", "text": ""},
            },
            *[
                {
                    "type": "content_block_delta",
                    "index": 2,
                    "delta": {"type": "text_delta", "text": piece},
                }
                for piece in pieces
            ],
            {"type": "content_block_stop", "index": 2},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": len(pieces)},
            },
            {"type": "message_stop"},
        ]
        web_lines = [
            b"event: "
            + str(event["type"]).encode("utf-8")
            + b"\ndata: "
            + json.dumps(event).encode("utf-8")
            + b"\n\n"
            for event in web_events
        ]

        clients_and_models = {
            "plain": DeepSeekChatHarnessModel(
                DeepSeekClient(
                    DeepSeekConfig(
                        allow_remote_student_data=True,
                        max_retries=0,
                    ),
                    api_key="adapter-secret",
                    stream_transport=lambda *_args: (200, list(plain_lines)),
                ),
                delta_flush_chars=64,
                monotonic=lambda: 0.0,
            ),
            "web": DeepSeekChatHarnessModel(
                DeepSeekClient(
                    DeepSeekConfig(
                        allow_remote_student_data=True,
                        max_retries=0,
                    ),
                    api_key="adapter-secret",
                    stream_transport=lambda *_args: (200, list(web_lines)),
                ),
                web_search=True,
                web_search_system="Search only when current data is needed.",
                delta_flush_chars=64,
                monotonic=lambda: 0.0,
            ),
        }
        for label, model in clients_and_models.items():
            with self.subTest(label=label):
                items = list(
                    model.plan_stream(
                        request,
                        cancellation_token=CancellationToken(),
                        deadline_monotonic=time.monotonic() + 10,
                    )
                )
                deltas = [
                    item.delta
                    for item in items
                    if isinstance(item, ProviderStreamEvent)
                    and item.type == "message.delta"
                ]
                self.assertEqual("".join(deltas), expected)
                self.assertLess(len(deltas), len(pieces) // 8)
                final = next(
                    item
                    for item in items
                    if isinstance(item, HarnessModelResponse)
                )
                self.assertEqual(final.output["message"], expected)
                if label == "web":
                    lifecycle = [
                        item
                        for item in items
                        if isinstance(item, ProviderStreamEvent)
                        and item.type == "tool_call.delta"
                    ]
                    self.assertEqual(
                        [item.payload["phase"] for item in lifecycle],
                        ["started", "progress", "completed"],
                    )
                    serialized = json.dumps(
                        [dict(item.payload) for item in lifecycle]
                    )
                    self.assertNotIn("private-query", serialized)
                    self.assertNotIn("private-search-body", serialized)
                    self.assertEqual(
                        final.output["sources"],
                        [
                            {
                                "title": "Current source",
                                "url": "https://example.com/current",
                            }
                        ],
                    )


if __name__ == "__main__":
    unittest.main()
