"""Bounded, recoverable, event-driven agent harness runtime."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Protocol
from uuid import uuid4

from .cancellation import (
    CancellationToken,
    HarnessClock,
    SystemClock,
    check_deadline,
    interruptible_sleep,
    remaining_seconds,
)
from .checkpoint import HarnessCheckpoint
from .contracts import (
    HARNESS_SCHEMA,
    HarnessCancelled,
    HarnessContractError,
    HarnessDeadlineExceeded,
    HarnessLimits,
    HarnessModelRequest,
    HarnessModelResponse,
    RetryPolicy,
    ToolCall,
)
from .events import (
    HarnessEventEmitter,
    canonical_sha256,
    public_event_projection,
)
from .journal import HarnessJournal
from .provider_registry import (
    ProviderErrorKind,
    ProviderFailure,
    ProviderModelSpec,
)
from .providers import ProviderStreamEvent
from .tools import (
    DEFAULT_TRUSTED_DATA_SCOPES,
    ToolExecutionResult,
    ToolRegistry,
    execute_tool_call,
    validate_allowed_permissions,
    validate_trusted_data_scopes,
)


class HarnessModel(Protocol):
    def plan(
        self,
        request: HarnessModelRequest,
        *,
        cancellation_token: CancellationToken,
        deadline_monotonic: float,
    ) -> HarnessModelResponse | Mapping[str, Any]: ...

    def classify_error(self, error: BaseException) -> ProviderFailure: ...


@dataclass(slots=True)
class _RunState:
    run_id: str
    turn_id: str
    context_sha256: str
    policy_sha256: str
    external_effect_started: bool = False
    next_step: int = 1
    model_calls: int = 0
    tool_calls: int = 0
    repeated_calls: dict[str, int] = None  # type: ignore[assignment]
    observations: list[dict[str, Any]] = None  # type: ignore[assignment]
    idempotency_receipts: dict[str, dict[str, Any]] = None  # type: ignore[assignment]
    completed_call_receipts: dict[str, dict[str, Any]] = None  # type: ignore[assignment]
    pending_effect: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.repeated_calls is None:
            self.repeated_calls = {}
        if self.observations is None:
            self.observations = []
        if self.idempotency_receipts is None:
            self.idempotency_receipts = {}
        if self.completed_call_receipts is None:
            self.completed_call_receipts = {}


@dataclass(frozen=True, slots=True)
class _JournalTailRecovery:
    """Decision derived from verified events newer than a checkpoint."""

    disposition: str = "none"
    reason_code: str = ""
    call_id: str | None = None
    tool_name: str | None = None
    step: int | None = None
    remaining_calls: tuple[ToolCall, ...] = ()


@dataclass(slots=True)
class _ModelAttemptTelemetry:
    """Safety facts observed while one provider attempt is in flight."""

    delta_emitted: bool = False


class _ProviderPlanningFailed(HarnessContractError):
    """A provider failure already reduced to its content-free public code."""

    def __init__(self, failure: ProviderFailure) -> None:
        self.failure = failure.validated()
        super().__init__(self.failure.safe_code)


class _ExternalEffectUncertain(HarnessContractError):
    """A declared effect boundary was crossed without a safe settlement."""


def _pending_step_value(pending: Mapping[str, Any], fallback: int) -> int:
    value = pending.get("step", fallback)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise HarnessContractError("pending effect step is invalid")
    return value


def _pending_remaining_calls(pending: Mapping[str, Any]) -> tuple[ToolCall, ...]:
    raw = pending.get("remaining_calls", [])
    if not isinstance(raw, list) or any(not isinstance(item, Mapping) for item in raw):
        raise HarnessContractError("pending remaining tool plan is invalid")
    return tuple(ToolCall.from_mapping(item) for item in raw)


def _checkpoint(
    state: _RunState,
    emitter: HarnessEventEmitter,
    *,
    status: str = "running",
) -> HarnessCheckpoint:
    return HarnessCheckpoint.create(
        run_id=state.run_id,
        turn_id=state.turn_id,
        status=status,
        next_sequence=emitter.next_sequence,
        next_step=state.next_step,
        model_calls=state.model_calls,
        tool_calls=state.tool_calls,
        context_sha256=state.context_sha256,
        policy_sha256=state.policy_sha256,
        external_effect_started=state.external_effect_started,
        repeated_calls=state.repeated_calls,
        observations=tuple(state.observations),
        idempotency_receipts=state.idempotency_receipts,
        completed_call_receipts=state.completed_call_receipts,
        pending_effect=state.pending_effect,
    )


def _save_checkpoint(
    state: _RunState,
    emitter: HarnessEventEmitter,
    checkpoint_sink: Callable[[HarnessCheckpoint], None] | None,
    *,
    status: str = "running",
) -> HarnessCheckpoint:
    checkpoint = _checkpoint(state, emitter, status=status)
    if checkpoint_sink is not None:
        checkpoint_sink(checkpoint)
    return checkpoint


def _persistence_checkpoint_sink(
    journal: HarnessJournal | None,
    checkpoint_sink: Callable[[HarnessCheckpoint], None] | None,
) -> Callable[[HarnessCheckpoint], None] | None:
    """Persist the journal anchor before exposing a checkpoint elsewhere."""

    if journal is None:
        return checkpoint_sink

    def persist(checkpoint: HarnessCheckpoint) -> None:
        acknowledgement = journal.write_checkpoint(checkpoint)
        if not acknowledgement.durable:
            raise HarnessContractError(
                "journal checkpoint was not durably acknowledged"
            )
        if checkpoint_sink is not None:
            checkpoint_sink(checkpoint)

    return persist


def _tool_signature(call: ToolCall) -> str:
    return canonical_sha256(
        {"tool_name": call.name, "arguments": dict(call.arguments)}
    )[:24]


_UNCLASSIFIED_PROVIDER_FAILURE = ProviderFailure(
    kind=ProviderErrorKind.UNKNOWN,
    retryable=False,
    safe_code="provider_error_unclassified",
).validated()

_MODEL_DELTA_EVENT_TYPES = frozenset(
    {"message.delta", "reasoning.delta", "tool_call.delta"}
)


def _classify_model_error(
    model: HarnessModel, exc: BaseException
) -> tuple[ProviderFailure, bool]:
    """Return one validated provider failure or a fail-closed fallback.

    Exception attributes are deliberately ignored.  Retry authority belongs
    exclusively to the adapter's typed ``classify_error`` boundary; a missing,
    throwing, or malformed classifier can therefore never opt an effect into a
    retry.
    """

    classifier = getattr(model, "classify_error", None)
    if not callable(classifier):
        return _UNCLASSIFIED_PROVIDER_FAILURE, False
    try:
        candidate = classifier(exc)
        if not isinstance(candidate, ProviderFailure):
            return _UNCLASSIFIED_PROVIDER_FAILURE, False
        return candidate.validated(), True
    except Exception:
        return _UNCLASSIFIED_PROVIDER_FAILURE, False


def _external_effect_started(state: _RunState) -> bool:
    """Whether this run has crossed any central-tool effect boundary."""

    return state.external_effect_started


def _stable_model_identity(model: HarnessModel) -> tuple[dict[str, Any], bool]:
    """Project a provider model into a credential-free stable identity."""

    try:
        candidate = getattr(model, "model_spec", None)
    except Exception:
        candidate = None
    if not isinstance(candidate, ProviderModelSpec):
        return {"identity_kind": "unclassified_provider_model"}, False
    try:
        spec = candidate.validated()
        capabilities = spec.capabilities.to_dict()
    except Exception:
        return {"identity_kind": "unclassified_provider_model"}, False
    return (
        {
            "identity_kind": "provider_model_spec_v1",
            "provider": spec.provider,
            "model": spec.model,
            "context_window_tokens": spec.context_window_tokens,
            "maximum_output_tokens": spec.maximum_output_tokens,
            "tokenizer": spec.tokenizer,
            "capabilities": capabilities,
        },
        True,
    )


def _execution_policy_material(
    *,
    limits: HarnessLimits,
    retry_policy: RetryPolicy,
    model_identity: Mapping[str, Any],
    allowed_permissions: set[str],
    trusted_data_scopes: frozenset[str],
    tool_execution_manifests: tuple[Mapping[str, Any], ...],
) -> dict[str, Any]:
    """Return every execution knob which must remain fixed on resume."""

    return {
        "limits": asdict(limits),
        "retry_policy": asdict(retry_policy),
        "model_identity": dict(model_identity),
        "authorization": {
            "allowed_permissions": sorted(allowed_permissions),
            "trusted_data_scopes": sorted(trusted_data_scopes),
        },
        "tool_execution_manifests": [
            dict(manifest) for manifest in tool_execution_manifests
        ],
    }


_SAFE_FAILURE_REASON_CODES = frozenset(
    {
        "model_call_budget_exceeded",
        "duplicate_call_id",
        "repeated_tool_call",
        "step_budget_exceeded",
        "tool_calls_per_step_budget_exceeded",
        "total_tool_call_budget_exceeded",
    }
)


def _safe_failure_reason(exc: Exception) -> str:
    """Map internal failures to a bounded, content-free public reason code."""

    if isinstance(exc, _ProviderPlanningFailed):
        return exc.failure.safe_code
    if isinstance(exc, HarnessContractError):
        candidate = str(exc)
        if candidate in _SAFE_FAILURE_REASON_CODES:
            return candidate
        return "harness_contract_error"
    return "internal_error"


def _run_model_plan(
    model: HarnessModel,
    request: HarnessModelRequest,
    *,
    clock: HarnessClock,
    cancellation_token: CancellationToken,
    deadline_monotonic: float,
    emitter: HarnessEventEmitter,
    attempt_telemetry: _ModelAttemptTelemetry,
) -> Any:
    """Run one provider adapter without creating an abandonable daemon thread.

    Production streaming adapters implement cooperative cancellation.  Every
    adapter executes on the run worker itself, so the Harness can never report
    timeout/cancellation while a hidden provider thread continues consuming
    data or producing effects.  A legacy non-cancellable adapter may delay the
    run, but it cannot outlive a terminal result.  Local unclassified test
    doubles use the same synchronous boundary.
    """

    def accept_stream_event(stream_event: ProviderStreamEvent) -> None:
        stream_event = stream_event.validated()
        if stream_event.type in _MODEL_DELTA_EVENT_TYPES:
            # Mark before persistence/publication: if the sink fails after
            # accepting a delta, replaying the provider call could still
            # duplicate visible output or a provider-managed effect.
            attempt_telemetry.delta_emitted = True
        emitter.emit(
            stream_event.type,
            {
                "channel": stream_event.channel,
                "delta": stream_event.delta,
                **dict(stream_event.payload),
            },
        )

    cancellation_token.raise_if_cancelled()
    check_deadline(clock=clock, deadline_monotonic=deadline_monotonic)
    stream_plan = getattr(model, "plan_stream", None)
    if not callable(stream_plan):
        response = model.plan(
            request,
            cancellation_token=cancellation_token,
            deadline_monotonic=deadline_monotonic,
        )
        cancellation_token.raise_if_cancelled()
        check_deadline(clock=clock, deadline_monotonic=deadline_monotonic)
        return response

    response: Any = None
    for item in stream_plan(
        request,
        cancellation_token=cancellation_token,
        deadline_monotonic=deadline_monotonic,
    ):
        cancellation_token.raise_if_cancelled()
        check_deadline(clock=clock, deadline_monotonic=deadline_monotonic)
        if isinstance(item, ProviderStreamEvent):
            accept_stream_event(item)
            continue
        if isinstance(item, (HarnessModelResponse, Mapping)):
            if response is not None:
                raise HarnessContractError(
                    "provider stream produced more than one final response"
                )
            response = item
            continue
        raise HarnessContractError("provider stream yielded an unsupported item")
    cancellation_token.raise_if_cancelled()
    check_deadline(clock=clock, deadline_monotonic=deadline_monotonic)
    if response is None:
        raise HarnessContractError("provider stream ended without a final response")
    return response


def _invoke_model(
    model: HarnessModel,
    request: HarnessModelRequest,
    *,
    state: _RunState,
    emitter: HarnessEventEmitter,
    retry_policy: RetryPolicy,
    clock: HarnessClock,
    cancellation_token: CancellationToken,
    deadline_monotonic: float,
    max_model_calls: int,
    checkpoint_sink: Callable[[HarnessCheckpoint], None] | None,
) -> HarnessModelResponse:
    last_error: Exception | None = None
    last_failure = _UNCLASSIFIED_PROVIDER_FAILURE
    for attempt in range(1, retry_policy.max_attempts + 1):
        cancellation_token.raise_if_cancelled()
        check_deadline(clock=clock, deadline_monotonic=deadline_monotonic)
        if state.model_calls >= max_model_calls:
            emitter.emit(
                "guard.triggered",
                {"reason_code": "model_call_budget_exceeded"},
            )
            raise HarnessContractError("model_call_budget_exceeded")
        state.model_calls += 1
        emitter.emit(
            "model.started",
            {"attempt": attempt, "step": request.step},
        )
        attempt_telemetry = _ModelAttemptTelemetry()
        try:
            raw = _run_model_plan(
                model,
                request,
                clock=clock,
                cancellation_token=cancellation_token,
                deadline_monotonic=deadline_monotonic,
                emitter=emitter,
                attempt_telemetry=attempt_telemetry,
            )
            response = HarnessModelResponse.from_value(raw)
            emitter.emit(
                "model.completed",
                {
                    "attempt": attempt,
                    "step": request.step,
                    "kind": response.kind,
                    "provider_request_id": response.provider_request_id,
                    "usage": dict(response.usage),
                    "model_response": {
                        "kind": response.kind,
                        "tool_calls": [call.to_dict() for call in response.tool_calls],
                        "output": response.output,
                        "reason": response.reason,
                    },
                },
            )
            _save_checkpoint(state, emitter, checkpoint_sink)
            cancellation_token.raise_if_cancelled()
            return response
        except HarnessCancelled:
            raise
        except HarnessDeadlineExceeded:
            raise
        except Exception as exc:
            last_error = exc
            failure, classifier_valid = _classify_model_error(model, exc)
            last_failure = failure
            emitter.emit(
                "model.failed",
                {
                    "attempt": attempt,
                    "step": request.step,
                    "error_type": type(exc).__name__,
                    "error_kind": failure.kind.value,
                    "safe_code": failure.safe_code,
                    "retryable": failure.retryable,
                    "classifier_valid": classifier_valid,
                    **(
                        {"retry_after_ms": round(failure.retry_after_seconds * 1_000)}
                        if failure.retry_after_seconds is not None
                        else {}
                    ),
                },
            )
            _save_checkpoint(state, emitter, checkpoint_sink)
            if not failure.retryable:
                raise _ProviderPlanningFailed(failure) from exc
            suppression_reason = ""
            if attempt_telemetry.delta_emitted:
                suppression_reason = "model_delta_emitted"
            elif _external_effect_started(state):
                suppression_reason = "external_effect_started"
            elif attempt >= retry_policy.max_attempts:
                suppression_reason = "attempt_limit"
            if suppression_reason:
                emitter.emit(
                    "model.retry_suppressed",
                    {
                        "attempt": attempt,
                        "reason_code": suppression_reason,
                        "error_kind": failure.kind.value,
                        "safe_code": failure.safe_code,
                    },
                )
                raise _ProviderPlanningFailed(failure) from exc
            policy_delay = retry_policy.delay_for_retry(attempt)
            retry_after = failure.retry_after_seconds or 0.0
            delay = max(policy_delay, retry_after)
            remaining = remaining_seconds(
                clock=clock, deadline_monotonic=deadline_monotonic
            )
            if delay >= remaining:
                emitter.emit(
                    "model.retry_suppressed",
                    {
                        "attempt": attempt,
                        "reason_code": "run_deadline",
                        "error_kind": failure.kind.value,
                        "safe_code": failure.safe_code,
                        "required_delay_ms": round(delay * 1_000),
                        "remaining_ms": round(remaining * 1_000),
                    },
                )
                raise HarnessDeadlineExceeded(
                    "model retry would start after the run deadline"
                ) from exc
            emitter.emit(
                "model.retrying",
                {
                    "attempt": attempt,
                    "next_attempt": attempt + 1,
                    "delay_ms": round(delay * 1_000),
                    "error_kind": failure.kind.value,
                    "safe_code": failure.safe_code,
                    "retry_after_ms": round(retry_after * 1_000),
                },
            )
            interruptible_sleep(
                delay,
                clock=clock,
                cancellation_token=cancellation_token,
                deadline_monotonic=deadline_monotonic,
            )
    raise _ProviderPlanningFailed(last_failure) from last_error


def _pending_effect(
    call: ToolCall,
    registry: ToolRegistry,
    *,
    step: int,
    remaining_calls: tuple[ToolCall, ...],
) -> dict[str, Any]:
    entry = registry.get(call.name)
    if entry is None:
        replay_policy = "safe"
        version = "unknown"
    else:
        spec, _handler = entry
        replay_policy = spec.replay_policy
        version = spec.version
    return {
        "kind": "tool",
        "step": step,
        "call": call.to_dict(),
        "tool_version": version,
        "replay_policy": replay_policy,
        "arguments_sha256": canonical_sha256(dict(call.arguments)),
        "effect_started": False,
        # Persist the rest of the approved model plan so recovery never has to
        # guess whether a reconciled call was the last call in the step.
        "remaining_calls": [item.to_dict() for item in remaining_calls],
    }


def _record_tool_result(
    state: _RunState,
    call: ToolCall,
    result: ToolExecutionResult,
    receipt: Mapping[str, Any] | None,
    registry: ToolRegistry,
) -> None:
    observation = result.observation()
    observation["arguments"] = deepcopy(dict(call.arguments))
    observation["arguments_sha256"] = canonical_sha256(dict(call.arguments))
    state.observations.append(observation)
    entry = registry.get(call.name)
    replay_policy = entry[0].replay_policy if entry is not None else "safe"
    state.completed_call_receipts[call.call_id] = {
        "call_id": call.call_id,
        "tool_name": call.name,
        "arguments_sha256": canonical_sha256(dict(call.arguments)),
        "replay_policy": replay_policy,
        "ok": result.ok,
        "result_sha256": result.result_sha256,
    }
    if receipt is not None and call.idempotency_key is not None and entry is not None:
        spec = entry[0]
        receipt_key = f"{spec.name}@{spec.version}:{call.idempotency_key}"
        state.idempotency_receipts[receipt_key] = dict(receipt)


def _execute_call(
    call: ToolCall,
    *,
    step: int,
    state: _RunState,
    registry: ToolRegistry,
    allowed_permissions: set[str],
    emitter: HarnessEventEmitter,
    clock: HarnessClock,
    cancellation_token: CancellationToken,
    deadline_monotonic: float,
    limits: HarnessLimits,
    checkpoint_sink: Callable[[HarnessCheckpoint], None] | None,
    principal_id: str,
    session_id: str | None,
    trusted_data_scopes: frozenset[str],
    remaining_calls: tuple[ToolCall, ...] = (),
) -> ToolExecutionResult:
    if call.call_id in state.completed_call_receipts:
        emitter.emit(
            "guard.triggered",
            {
                "reason_code": "duplicate_call_id",
                "tool_name": call.name,
            },
        )
        raise HarnessContractError("duplicate_call_id")
    signature = _tool_signature(call)
    count = state.repeated_calls.get(signature, 0) + 1
    state.repeated_calls[signature] = count
    if count > limits.max_repeated_tool_calls:
        emitter.emit(
            "guard.triggered",
            {
                "reason_code": "repeated_tool_call",
                "tool_name": call.name,
                "signature": signature,
            },
        )
        raise HarnessContractError("repeated_tool_call")
    if state.tool_calls >= limits.max_total_tool_calls:
        emitter.emit(
            "guard.triggered",
            {
                "reason_code": "total_tool_call_budget_exceeded",
                "tool_name": call.name,
            },
        )
        raise HarnessContractError("total_tool_call_budget_exceeded")
    state.tool_calls += 1
    state.pending_effect = _pending_effect(
        call,
        registry,
        step=step,
        remaining_calls=remaining_calls,
    )
    _save_checkpoint(state, emitter, checkpoint_sink)

    def mark_effect_started() -> None:
        state.external_effect_started = True
        if state.pending_effect is not None:
            state.pending_effect["effect_started"] = True
        # ``tool.effect_started`` precedes this checkpoint. Persist the fence
        # before the handler crosses its commit boundary so recovery can never
        # mistake an uncertain run for effect-free.
        _save_checkpoint(state, emitter, checkpoint_sink)

    result, receipt = execute_tool_call(
        call,
        registry=registry,
        allowed_permissions=allowed_permissions,
        emitter=emitter,
        clock=clock,
        cancellation_token=cancellation_token,
        run_deadline_monotonic=deadline_monotonic,
        max_output_chars=limits.max_tool_output_chars,
        idempotency_receipts=state.idempotency_receipts,
        principal_id=principal_id,
        session_id=session_id,
        trusted_data_scopes=trusted_data_scopes,
        effect_started_sink=mark_effect_started,
    )
    _record_tool_result(state, call, result, receipt, registry)
    if result.effect_started and not result.ok:
        # Keep pending_effect in the checkpoint. The journal contains the
        # concrete failed settlement, but a non-safe handler may have changed
        # state before it failed, so the run must end in one authoritative
        # handoff instead of asking the model to continue or repeat it.
        _save_checkpoint(state, emitter, checkpoint_sink)
        raise _ExternalEffectUncertain(
            result.error_code or "external_effect_unsettled"
        )
    state.pending_effect = None
    _save_checkpoint(state, emitter, checkpoint_sink)
    return result


def _initial_state(
    *,
    run_id: str,
    turn_id: str,
    context_sha256: str,
    policy_sha256: str,
    checkpoint: HarnessCheckpoint | None,
) -> _RunState:
    if checkpoint is None:
        return _RunState(
            run_id=run_id,
            turn_id=turn_id,
            context_sha256=context_sha256,
            policy_sha256=policy_sha256,
        )
    return _RunState(
        run_id=checkpoint.run_id,
        turn_id=checkpoint.turn_id,
        context_sha256=checkpoint.context_sha256,
        policy_sha256=checkpoint.policy_sha256,
        external_effect_started=checkpoint.external_effect_started,
        next_step=checkpoint.next_step,
        model_calls=checkpoint.model_calls,
        tool_calls=checkpoint.tool_calls,
        repeated_calls=dict(checkpoint.repeated_calls),
        observations=[dict(item) for item in checkpoint.observations],
        idempotency_receipts={
            str(key): dict(value)
            for key, value in checkpoint.idempotency_receipts.items()
        },
        completed_call_receipts={
            str(key): dict(value)
            for key, value in checkpoint.completed_call_receipts.items()
        },
        pending_effect=(
            dict(checkpoint.pending_effect) if checkpoint.pending_effect else None
        ),
    )


def _reconcile_journal_tail(
    state: _RunState,
    tail: list[dict[str, Any]],
    registry: ToolRegistry,
) -> _JournalTailRecovery:
    """Reconcile a verified tail without re-running a settled side effect.

    A checkpoint is written after an effect intent and again after settlement.
    Process death can therefore leave a durable ``tool.completed`` event newer
    than the intent checkpoint.  The event includes the bounded result and its
    hash, which is enough to rebuild the observation and receipt.  Any tail we
    cannot prove belongs to exactly that pending call is blocked rather than
    interpreted optimistically.
    """

    if not tail:
        return _JournalTailRecovery()
    if state.pending_effect is None:
        return _JournalTailRecovery(
            disposition="blocked",
            reason_code="durable_tail_without_recoverable_effect",
        )
    pending = state.pending_effect
    raw_call = pending.get("call")
    if not isinstance(raw_call, Mapping):
        return _JournalTailRecovery(
            disposition="blocked",
            reason_code="pending_effect_call_invalid",
        )
    try:
        call = ToolCall.from_mapping(raw_call)
    except HarnessContractError:
        return _JournalTailRecovery(
            disposition="blocked",
            reason_code="pending_effect_call_invalid",
        )

    operational: list[dict[str, Any]] = []
    for event in tail:
        event_type = str(event.get("type", ""))
        payload = event.get("payload")
        if not isinstance(payload, Mapping):
            return _JournalTailRecovery(
                disposition="blocked",
                reason_code="durable_tail_payload_invalid",
                call_id=call.call_id,
                tool_name=call.name,
            )
        if event_type == "run.started" and payload.get("resumed") is True:
            # A prior recovery process may itself have died before checkpoint.
            continue
        if event_type == "tool.reconciled":
            if payload.get("call_id") != call.call_id:
                return _JournalTailRecovery(
                    disposition="blocked",
                    reason_code="durable_tail_call_mismatch",
                    call_id=call.call_id,
                    tool_name=call.name,
                )
            continue
        if event_type not in {
            "tool.requested",
            "tool.started",
            "tool.effect_started",
            "tool.progress",
            "tool.retrying",
            "tool.completed",
            "tool.failed",
            "tool.rejected",
            "tool.replayed",
        }:
            return _JournalTailRecovery(
                disposition="blocked",
                reason_code="durable_tail_not_reconcilable",
                call_id=call.call_id,
                tool_name=call.name,
            )
        if (
            payload.get("call_id") != call.call_id
            or payload.get("tool_name") != call.name
        ):
            return _JournalTailRecovery(
                disposition="blocked",
                reason_code="durable_tail_call_mismatch",
                call_id=call.call_id,
                tool_name=call.name,
            )
        operational.append(event)

    expected_arguments_sha256 = str(pending.get("arguments_sha256", ""))
    if expected_arguments_sha256 != canonical_sha256(dict(call.arguments)):
        return _JournalTailRecovery(
            disposition="blocked",
            reason_code="durable_tail_intent_mismatch",
            call_id=call.call_id,
            tool_name=call.name,
        )

    if "remaining_calls" not in pending or not isinstance(
        pending.get("remaining_calls"), list
    ):
        return _JournalTailRecovery(
            disposition="blocked",
            reason_code="durable_tool_settlement_plan_unknown",
            call_id=call.call_id,
            tool_name=call.name,
        )
    try:
        remaining_calls = _pending_remaining_calls(pending)
        pending_step = _pending_step_value(pending, state.next_step)
    except HarnessContractError:
        return _JournalTailRecovery(
            disposition="blocked",
            reason_code="durable_tool_remaining_plan_invalid",
            call_id=call.call_id,
            tool_name=call.name,
        )
    if not operational:
        return _JournalTailRecovery(
            disposition="unsettled",
            reason_code="durable_tool_intent_unsettled",
            call_id=call.call_id,
            tool_name=call.name,
            step=pending_step,
            remaining_calls=remaining_calls,
        )
    requested = [item for item in operational if item.get("type") == "tool.requested"]
    if (not requested and not state.external_effect_started) or any(
        item["payload"].get("arguments_sha256") != expected_arguments_sha256
        for item in requested
    ):
        return _JournalTailRecovery(
            disposition="blocked",
            reason_code="durable_tail_intent_mismatch",
            call_id=call.call_id,
            tool_name=call.name,
        )
    settlements = [
        item
        for item in operational
        if item.get("type")
        in {"tool.completed", "tool.failed", "tool.rejected", "tool.replayed"}
    ]
    if not settlements:
        return _JournalTailRecovery(
            disposition="unsettled",
            reason_code="durable_tool_intent_unsettled",
            call_id=call.call_id,
            tool_name=call.name,
            step=pending_step,
            remaining_calls=remaining_calls,
        )
    if len(settlements) != 1 or operational[-1] is not settlements[0]:
        return _JournalTailRecovery(
            disposition="blocked",
            reason_code="durable_tail_settlement_ambiguous",
            call_id=call.call_id,
            tool_name=call.name,
        )
    settlement = settlements[0]
    payload = settlement["payload"]
    settlement_type = str(settlement["type"])
    if settlement_type == "tool.failed" and (
        pending.get("effect_started") is True
        or any(item.get("type") == "tool.effect_started" for item in operational)
    ):
        return _JournalTailRecovery(
            disposition="blocked",
            reason_code="durable_external_effect_unsettled",
            call_id=call.call_id,
            tool_name=call.name,
        )
    receipt: Mapping[str, Any] | None = None
    if settlement_type == "tool.completed":
        if "result" not in payload:
            return _JournalTailRecovery(
                disposition="blocked",
                reason_code="durable_tool_result_missing",
                call_id=call.call_id,
                tool_name=call.name,
            )
        result_value = deepcopy(payload["result"])
        result_sha256 = str(payload.get("result_sha256", ""))
        if canonical_sha256(result_value) != result_sha256:
            return _JournalTailRecovery(
                disposition="blocked",
                reason_code="durable_tool_result_hash_mismatch",
                call_id=call.call_id,
                tool_name=call.name,
            )
        entry = registry.get(call.name)
        if entry is None or payload.get("tool_version") != entry[0].version:
            return _JournalTailRecovery(
                disposition="blocked",
                reason_code="durable_tool_version_mismatch",
                call_id=call.call_id,
                tool_name=call.name,
            )
        result = ToolExecutionResult(
            call_id=call.call_id,
            tool_name=call.name,
            ok=True,
            result=result_value,
            result_sha256=result_sha256,
        )
        if call.idempotency_key is not None:
            receipt = {
                "call_id": call.call_id,
                "tool_name": call.name,
                "tool_version": entry[0].version,
                "arguments_sha256": expected_arguments_sha256,
                "result": result_value,
                "result_sha256": result_sha256,
            }
    elif settlement_type == "tool.replayed":
        entry = registry.get(call.name)
        if entry is None or call.idempotency_key is None:
            return _JournalTailRecovery(
                disposition="blocked",
                reason_code="durable_tool_receipt_missing",
                call_id=call.call_id,
                tool_name=call.name,
            )
        receipt_key = f"{entry[0].name}@{entry[0].version}:{call.idempotency_key}"
        existing = state.idempotency_receipts.get(receipt_key)
        if not isinstance(existing, Mapping):
            return _JournalTailRecovery(
                disposition="blocked",
                reason_code="durable_tool_receipt_missing",
                call_id=call.call_id,
                tool_name=call.name,
            )
        existing_result = deepcopy(existing.get("result"))
        existing_result_sha256 = str(existing.get("result_sha256", ""))
        if (
            canonical_sha256(existing_result) != existing_result_sha256
            or payload.get("result_sha256") != existing_result_sha256
            or payload.get("source_call_id") != existing.get("call_id")
        ):
            return _JournalTailRecovery(
                disposition="blocked",
                reason_code="durable_tool_receipt_mismatch",
                call_id=call.call_id,
                tool_name=call.name,
            )
        result = ToolExecutionResult(
            call_id=call.call_id,
            tool_name=call.name,
            ok=True,
            result=existing_result,
            result_sha256=existing_result_sha256,
            source_call_id=str(existing.get("call_id", "")),
        )
    else:
        error_code = str(payload.get("error_code", "tool_execution_failed"))[:160]
        result = ToolExecutionResult(
            call_id=call.call_id,
            tool_name=call.name,
            ok=False,
            error_code=error_code,
            error_message="recovered from a durably recorded tool settlement",
            retryable=False,
        )

    _record_tool_result(state, call, result, receipt, registry)
    state.pending_effect = None
    return _JournalTailRecovery(
        disposition="reconciled",
        reason_code="durable_tool_settlement_reconciled",
        call_id=call.call_id,
        tool_name=call.name,
        step=pending_step,
        remaining_calls=remaining_calls,
    )


def _result(
    *,
    status: str,
    output: Any,
    reason: str,
    state: _RunState,
    emitter: HarnessEventEmitter,
    checkpoint: HarnessCheckpoint,
    started_monotonic: float,
    clock: HarnessClock,
) -> dict[str, Any]:
    return {
        "schema": HARNESS_SCHEMA,
        "run_id": state.run_id,
        "turn_id": state.turn_id,
        "status": status,
        "output": deepcopy(output),
        "reason": reason,
        "steps": max(0, state.next_step - 1),
        "model_call_count": state.model_calls,
        "tool_call_count": state.tool_calls,
        "duration_ms": max(0, round((clock.monotonic() - started_monotonic) * 1_000)),
        "events": emitter.events,
        "events_truncated_through_sequence": emitter.dropped_through_sequence,
        "checkpoint": checkpoint,
    }


def run_agent_harness(
    model: HarnessModel,
    registry: ToolRegistry,
    context: Mapping[str, Any],
    *,
    run_id: str | None = None,
    turn_id: str | None = None,
    limits: HarnessLimits | None = None,
    retry_policy: RetryPolicy | None = None,
    allowed_permissions: set[str],
    clock: HarnessClock | None = None,
    cancellation_token: CancellationToken | None = None,
    event_sink: Callable[[Mapping[str, Any]], None] | None = None,
    checkpoint_sink: Callable[[HarnessCheckpoint], None] | None = None,
    journal: HarnessJournal | None = None,
    steering_source: Callable[[], tuple[Mapping[str, Any], ...]] | None = None,
    principal_id: str = "local-user",
    session_id: str | None = None,
    trusted_data_scopes: frozenset[str] | set[str] = DEFAULT_TRUSTED_DATA_SCOPES,
    _resume_checkpoint: HarnessCheckpoint | None = None,
) -> dict[str, Any]:
    """Execute one bounded operation and return its authoritative event trace."""

    if not isinstance(context, Mapping):
        raise HarnessContractError("harness context must be an object")
    if not isinstance(registry, ToolRegistry):
        raise HarnessContractError("registry must be a ToolRegistry")
    if not hasattr(model, "plan"):
        raise HarnessContractError("model must implement plan()")
    if journal is not None and not isinstance(journal, HarnessJournal):
        raise HarnessContractError("journal must be a HarnessJournal")
    limits = (limits or HarnessLimits()).validated()
    retry_policy = (retry_policy or RetryPolicy()).validated()
    allowed_permissions = validate_allowed_permissions(allowed_permissions)
    if (
        not isinstance(principal_id, str)
        or not principal_id.strip()
        or len(principal_id) > 200
    ):
        raise HarnessContractError("principal_id is invalid")
    principal_id = principal_id.strip()
    if session_id is not None and (
        not isinstance(session_id, str)
        or not session_id.strip()
        or len(session_id) > 200
    ):
        raise HarnessContractError("session_id is invalid")
    session_id = session_id.strip() if isinstance(session_id, str) else None
    trusted_data_scopes = validate_trusted_data_scopes(trusted_data_scopes)
    clock = clock or SystemClock()
    cancellation_token = cancellation_token or CancellationToken()
    authorized_tool_definitions = registry.definitions(
        allowed_permissions, trusted_data_scopes=trusted_data_scopes
    )
    tool_execution_manifests = registry.execution_manifests(
        allowed_permissions, trusted_data_scopes=trusted_data_scopes
    )
    model_identity, model_identity_stable = _stable_model_identity(model)
    policy_hash = canonical_sha256(
        _execution_policy_material(
            limits=limits,
            retry_policy=retry_policy,
            model_identity=model_identity,
            allowed_permissions=allowed_permissions,
            trusted_data_scopes=trusted_data_scopes,
            tool_execution_manifests=tool_execution_manifests,
        )
    )
    context_hash = canonical_sha256(
        {
            "context": dict(context),
            "authorization": {
                "principal_id": principal_id,
                "session_id": session_id,
                "allowed_permissions": sorted(allowed_permissions),
                "trusted_data_scopes": sorted(trusted_data_scopes),
                "authorized_tools": authorized_tool_definitions,
            },
        }
    )
    journal_tail: list[dict[str, Any]] = []
    if _resume_checkpoint is not None:
        checkpoint = _resume_checkpoint.validated()
        if not model_identity_stable:
            raise HarnessContractError(
                "resume requires a stable provider model identity"
            )
        if checkpoint.context_sha256 != context_hash:
            raise HarnessContractError("resume context does not match checkpoint")
        if checkpoint.policy_sha256 != policy_hash:
            raise HarnessContractError(
                "resume execution policy does not match checkpoint"
            )
        if checkpoint.status != "running":
            raise HarnessContractError("only a running checkpoint can be resumed")
        run_id = checkpoint.run_id
        turn_id = checkpoint.turn_id
        if journal is not None:
            if journal.run_id != run_id or journal.turn_id != turn_id:
                raise HarnessContractError(
                    "resume journal identifiers do not match checkpoint"
                )
            if journal.terminal_type is not None:
                raise HarnessContractError("cannot resume a terminal journal")
            anchor_sequence = checkpoint.next_sequence - 1
            if journal.last_sequence < anchor_sequence:
                raise HarnessContractError(
                    "resume journal is behind the checkpoint anchor"
                )
            durable_snapshot = journal.load_checkpoint()
            if durable_snapshot is not None:
                durable_checkpoint = HarnessCheckpoint.from_value(durable_snapshot)
                if durable_checkpoint.checkpoint_sha256 != checkpoint.checkpoint_sha256:
                    raise HarnessContractError(
                        "resume checkpoint does not match the durable journal checkpoint"
                    )
            journal_tail = journal.replay(after_sequence=anchor_sequence)
            next_sequence = journal.next_sequence
        else:
            next_sequence = checkpoint.next_sequence
    else:
        run_id = str(
            run_id or (journal.run_id if journal is not None else f"run_{uuid4().hex}")
        ).strip()
        turn_id = str(
            turn_id
            or (journal.turn_id if journal is not None else f"turn_{uuid4().hex}")
        ).strip()
        if journal is not None:
            if journal.run_id != run_id or journal.turn_id != turn_id:
                raise HarnessContractError(
                    "fresh journal identifiers do not match the requested run"
                )
            if journal.last_sequence != 0:
                raise HarnessContractError(
                    "fresh harness run requires an empty journal"
                )
            if journal.load_checkpoint() is not None:
                raise HarnessContractError(
                    "fresh harness run cannot reuse a journal checkpoint"
                )
        next_sequence = 1
    if not run_id or not turn_id:
        raise HarnessContractError("run_id and turn_id are required")
    state = _initial_state(
        run_id=run_id,
        turn_id=turn_id,
        context_sha256=context_hash,
        policy_sha256=policy_hash,
        checkpoint=_resume_checkpoint,
    )
    recovery = _reconcile_journal_tail(state, journal_tail, registry)
    emitter = HarnessEventEmitter(
        run_id=run_id,
        turn_id=turn_id,
        next_sequence=next_sequence,
        max_payload_chars=limits.max_event_payload_chars,
        max_in_memory_events=limits.max_in_memory_events,
        max_in_memory_event_bytes=limits.max_in_memory_event_bytes,
        durable_sink=journal.append if journal is not None else None,
        event_sink=event_sink,
    )
    checkpoint_sink = _persistence_checkpoint_sink(journal, checkpoint_sink)
    started = clock.monotonic()
    deadline = started + limits.deadline_seconds
    emitter.emit(
        "run.started",
        {
            "resumed": _resume_checkpoint is not None,
            "context_sha256": context_hash,
            "policy_sha256": policy_hash,
        },
    )
    if recovery.disposition == "blocked":
        emitter.emit(
            "run.handoff",
            {
                "reason_code": recovery.reason_code,
                "call_id": recovery.call_id,
                "tool_name": recovery.tool_name,
            },
        )
        latest_checkpoint = _save_checkpoint(
            state, emitter, checkpoint_sink, status="handoff"
        )
        return _result(
            status="handoff",
            output=None,
            reason=recovery.reason_code,
            state=state,
            emitter=emitter,
            checkpoint=latest_checkpoint,
            started_monotonic=started,
            clock=clock,
        )
    if recovery.disposition == "reconciled":
        emitter.emit(
            "tool.reconciled",
            {
                "reason_code": recovery.reason_code,
                "call_id": recovery.call_id,
                "tool_name": recovery.tool_name,
            },
        )
        recovery_step = recovery.step or state.next_step
        for index, remaining_call in enumerate(recovery.remaining_calls):
            _execute_call(
                remaining_call,
                step=recovery_step,
                state=state,
                registry=registry,
                allowed_permissions=allowed_permissions,
                emitter=emitter,
                clock=clock,
                cancellation_token=cancellation_token,
                deadline_monotonic=deadline,
                limits=limits,
                checkpoint_sink=checkpoint_sink,
                principal_id=principal_id,
                session_id=session_id,
                trusted_data_scopes=trusted_data_scopes,
                remaining_calls=recovery.remaining_calls[index + 1 :],
            )
        state.next_step = max(state.next_step, recovery_step + 1)
        latest_checkpoint = _save_checkpoint(state, emitter, checkpoint_sink)
    else:
        latest_checkpoint = _save_checkpoint(state, emitter, checkpoint_sink)

    # Intent-without-settlement recovery: safe/idempotent effects may be
    # replayed, while an uncertain never-replay effect requires a human or a
    # domain-specific reconciliation tool.
    if state.pending_effect is not None:
        pending = state.pending_effect
        replay_policy = str(pending.get("replay_policy", "never"))
        raw_call = pending.get("call")
        if replay_policy == "never" or not isinstance(raw_call, Mapping):
            emitter.emit(
                "run.handoff",
                {
                    "reason_code": "unsafe_tool_replay_blocked",
                    "tool_name": (
                        raw_call.get("name") if isinstance(raw_call, Mapping) else None
                    ),
                },
            )
            latest_checkpoint = _save_checkpoint(
                state, emitter, checkpoint_sink, status="handoff"
            )
            return _result(
                status="handoff",
                output=None,
                reason="unsafe_tool_replay_blocked",
                state=state,
                emitter=emitter,
                checkpoint=latest_checkpoint,
                started_monotonic=started,
                clock=clock,
            )
        resumed_call = ToolCall.from_mapping(raw_call)
        pending_step = _pending_step_value(pending, state.next_step)
        pending_remaining = _pending_remaining_calls(pending)
        # The original attempt already counted toward budgets and repetition;
        # reset those two counters once before a permitted recovery replay.
        state.tool_calls = max(0, state.tool_calls - 1)
        signature = _tool_signature(resumed_call)
        state.repeated_calls[signature] = max(
            0, state.repeated_calls.get(signature, 1) - 1
        )
        try:
            recovery_calls = (resumed_call, *pending_remaining)
            for index, recovery_call in enumerate(recovery_calls):
                _execute_call(
                    recovery_call,
                    step=pending_step,
                    state=state,
                    registry=registry,
                    allowed_permissions=allowed_permissions,
                    emitter=emitter,
                    clock=clock,
                    cancellation_token=cancellation_token,
                    deadline_monotonic=deadline,
                    limits=limits,
                    checkpoint_sink=checkpoint_sink,
                    principal_id=principal_id,
                    session_id=session_id,
                    trusted_data_scopes=trusted_data_scopes,
                    remaining_calls=recovery_calls[index + 1 :],
                )
            state.next_step = max(state.next_step, pending_step + 1)
        except (HarnessCancelled, HarnessDeadlineExceeded):
            raise

    output: Any = None
    reason = ""
    try:
        while state.next_step <= limits.max_steps:
            cancellation_token.raise_if_cancelled()
            check_deadline(clock=clock, deadline_monotonic=deadline)
            if steering_source is not None:
                for raw_input in steering_source():
                    if not isinstance(raw_input, Mapping):
                        raise HarnessContractError(
                            "steering source returned invalid input"
                        )
                    content = raw_input.get("content")
                    input_id = raw_input.get("input_id")
                    if (
                        raw_input.get("kind") != "steer"
                        or not isinstance(content, str)
                        or not content.strip()
                        or len(content) > 4_000
                    ):
                        raise HarnessContractError("steering input contract is invalid")
                    observation = {
                        "kind": "steer",
                        "input_id": str(input_id or "")[:160],
                        "content": content.strip(),
                    }
                    state.observations.append(observation)
                    emitter.emit(
                        "input.steered",
                        {
                            **observation,
                            "content_sha256": canonical_sha256(content.strip()),
                        },
                    )
                _save_checkpoint(state, emitter, checkpoint_sink)
            if state.model_calls >= limits.max_model_calls:
                emitter.emit(
                    "guard.triggered",
                    {"reason_code": "model_call_budget_exceeded"},
                )
                raise HarnessContractError("model_call_budget_exceeded")
            request = HarnessModelRequest(
                run_id=state.run_id,
                turn_id=state.turn_id,
                step=state.next_step,
                context=deepcopy(dict(context)),
                observations=tuple(deepcopy(state.observations)),
                tools=authorized_tool_definitions,
                state={
                    "model_calls": state.model_calls,
                    "tool_calls": state.tool_calls,
                    "remaining_steps": limits.max_steps - state.next_step + 1,
                    "completed_call_receipts": list(
                        state.completed_call_receipts.values()
                    )[-64:],
                },
            )
            response = _invoke_model(
                model,
                request,
                state=state,
                emitter=emitter,
                retry_policy=retry_policy,
                clock=clock,
                cancellation_token=cancellation_token,
                deadline_monotonic=deadline,
                max_model_calls=limits.max_model_calls,
                checkpoint_sink=checkpoint_sink,
            )
            cancellation_token.raise_if_cancelled()
            late_steers: list[Mapping[str, Any]] = []
            if steering_source is not None:
                late_steers = list(steering_source())
            if late_steers and response.kind in {"final", "handoff"}:
                for raw_input in late_steers:
                    content = (
                        raw_input.get("content")
                        if isinstance(raw_input, Mapping)
                        else None
                    )
                    if (
                        not isinstance(content, str)
                        or not content.strip()
                        or len(content) > 4_000
                    ):
                        raise HarnessContractError("late steering input is invalid")
                    state.observations.append(
                        {
                            "kind": "steer",
                            "input_id": str(raw_input.get("input_id", ""))[:160],
                            "content": content.strip()[:4_000],
                        }
                    )
                    emitter.emit(
                        "input.steered",
                        {
                            "input_id": str(raw_input.get("input_id", ""))[:160],
                            "content": content.strip()[:4_000],
                            "content_sha256": canonical_sha256(content.strip()[:4_000]),
                        },
                    )
                emitter.emit(
                    "action.superseded",
                    {"reason_code": "late_steering_input"},
                )
                state.next_step += 1
                _save_checkpoint(state, emitter, checkpoint_sink)
                continue
            if response.kind == "handoff":
                reason = response.reason
                emitter.emit(
                    "run.handoff",
                    {"reason_code": "model_requested_handoff"},
                )
                latest_checkpoint = _save_checkpoint(
                    state, emitter, checkpoint_sink, status="handoff"
                )
                return _result(
                    status="handoff",
                    output=response.output,
                    reason=reason,
                    state=state,
                    emitter=emitter,
                    checkpoint=latest_checkpoint,
                    started_monotonic=started,
                    clock=clock,
                )
            if response.kind == "final":
                output = response.output
                emitter.emit(
                    "action.completed",
                    {
                        "output": output,
                        "output_sha256": canonical_sha256(output),
                    },
                )
                emitter.emit(
                    "run.completed",
                    {
                        "reason_code": "final_output",
                        "duration_ms": max(
                            0, round((clock.monotonic() - started) * 1_000)
                        ),
                    },
                )
                state.next_step += 1
                latest_checkpoint = _save_checkpoint(
                    state, emitter, checkpoint_sink, status="completed"
                )
                return _result(
                    status="completed",
                    output=output,
                    reason="final_output",
                    state=state,
                    emitter=emitter,
                    checkpoint=latest_checkpoint,
                    started_monotonic=started,
                    clock=clock,
                )
            calls = response.tool_calls
            if len(calls) > limits.max_tool_calls_per_step:
                emitter.emit(
                    "guard.triggered",
                    {"reason_code": "tool_calls_per_step_budget_exceeded"},
                )
                raise HarnessContractError("tool_calls_per_step_budget_exceeded")
            # Serial settlement is the safety default.  A bounded parallel
            # dispatcher is added by an adapter only when every ToolSpec says
            # ``parallel_safe``; results remain ordered by model call order.
            for index, call in enumerate(calls):
                _execute_call(
                    call,
                    step=state.next_step,
                    state=state,
                    registry=registry,
                    allowed_permissions=allowed_permissions,
                    emitter=emitter,
                    clock=clock,
                    cancellation_token=cancellation_token,
                    deadline_monotonic=deadline,
                    limits=limits,
                    checkpoint_sink=checkpoint_sink,
                    principal_id=principal_id,
                    session_id=session_id,
                    trusted_data_scopes=trusted_data_scopes,
                    remaining_calls=calls[index + 1 :],
                )
            state.next_step += 1
            latest_checkpoint = _save_checkpoint(state, emitter, checkpoint_sink)
        emitter.emit(
            "guard.triggered",
            {"reason_code": "step_budget_exceeded"},
        )
        raise HarnessContractError("step_budget_exceeded")
    except HarnessCancelled as exc:
        if state.external_effect_started:
            emitter.emit(
                "run.handoff",
                {
                    "reason_code": "cancelled_after_external_effect",
                    "cancellation_reason": cancellation_token.reason,
                },
            )
            latest_checkpoint = _save_checkpoint(
                state,
                emitter,
                checkpoint_sink,
                status="handoff",
            )
            return _result(
                status="handoff",
                output=None,
                reason="cancellation followed a non-replayable external effect",
                state=state,
                emitter=emitter,
                checkpoint=latest_checkpoint,
                started_monotonic=started,
                clock=clock,
            )
        emitter.emit(
            "run.cancelled",
            {"reason_code": cancellation_token.reason},
        )
        latest_checkpoint = _save_checkpoint(
            state, emitter, checkpoint_sink, status="cancelled"
        )
        return _result(
            status="cancelled",
            output=None,
            reason=str(exc),
            state=state,
            emitter=emitter,
            checkpoint=latest_checkpoint,
            started_monotonic=started,
            clock=clock,
        )
    except _ExternalEffectUncertain as exc:
        emitter.emit(
            "run.handoff",
            {
                "reason_code": "external_effect_unsettled",
                "safe_code": str(exc)[:128],
            },
        )
        latest_checkpoint = _save_checkpoint(
            state,
            emitter,
            checkpoint_sink,
            status="handoff",
        )
        return _result(
            status="handoff",
            output=None,
            reason="an external effect did not settle safely",
            state=state,
            emitter=emitter,
            checkpoint=latest_checkpoint,
            started_monotonic=started,
            clock=clock,
        )
    except HarnessDeadlineExceeded as exc:
        if state.external_effect_started:
            emitter.emit(
                "run.handoff",
                {"reason_code": "deadline_after_external_effect"},
            )
            latest_checkpoint = _save_checkpoint(
                state,
                emitter,
                checkpoint_sink,
                status="handoff",
            )
            return _result(
                status="handoff",
                output=None,
                reason="the run deadline followed an external effect",
                state=state,
                emitter=emitter,
                checkpoint=latest_checkpoint,
                started_monotonic=started,
                clock=clock,
            )
        emitter.emit(
            "run.failed",
            {
                "error_code": "deadline_exceeded",
                "reason_code": "deadline_exceeded",
            },
        )
        latest_checkpoint = _save_checkpoint(
            state, emitter, checkpoint_sink, status="failed"
        )
        return _result(
            status="deadline_exceeded",
            output=None,
            reason=str(exc),
            state=state,
            emitter=emitter,
            checkpoint=latest_checkpoint,
            started_monotonic=started,
            clock=clock,
        )
    except Exception as exc:
        reason_code = _safe_failure_reason(exc)
        if state.external_effect_started:
            emitter.emit(
                "run.handoff",
                {
                    "reason_code": "failure_after_external_effect",
                    "safe_code": reason_code,
                },
            )
            latest_checkpoint = _save_checkpoint(
                state,
                emitter,
                checkpoint_sink,
                status="handoff",
            )
            return _result(
                status="handoff",
                output=None,
                reason="a failure followed an external effect",
                state=state,
                emitter=emitter,
                checkpoint=latest_checkpoint,
                started_monotonic=started,
                clock=clock,
            )
        emitter.emit(
            "run.failed",
            {
                "error_code": "harness_failed",
                "reason_code": reason_code,
                "error_type": type(exc).__name__,
            },
        )
        latest_checkpoint = _save_checkpoint(
            state, emitter, checkpoint_sink, status="failed"
        )
        return _result(
            status="failed",
            output=None,
            reason=reason_code,
            state=state,
            emitter=emitter,
            checkpoint=latest_checkpoint,
            started_monotonic=started,
            clock=clock,
        )


def resume_agent_harness(
    checkpoint: HarnessCheckpoint | Mapping[str, Any] | None,
    model: HarnessModel,
    registry: ToolRegistry,
    context: Mapping[str, Any],
    **kwargs: Any,
) -> dict[str, Any]:
    """Resume a validated running checkpoint without guessing unsafe effects."""

    if checkpoint is None:
        journal = kwargs.get("journal")
        if not isinstance(journal, HarnessJournal):
            raise HarnessContractError(
                "resume without a checkpoint requires a HarnessJournal"
            )
        checkpoint = journal.load_checkpoint()
        if checkpoint is None:
            raise HarnessContractError("journal does not contain a checkpoint")
    normalized = HarnessCheckpoint.from_value(checkpoint)
    return run_agent_harness(
        model,
        registry,
        context,
        run_id=normalized.run_id,
        turn_id=normalized.turn_id,
        _resume_checkpoint=normalized,
        **kwargs,
    )


def public_harness_trace(result: Mapping[str, Any]) -> dict[str, Any]:
    """Return a content-free trace suitable for UI diagnostics and telemetry."""

    if not isinstance(result, Mapping) or result.get("schema") != HARNESS_SCHEMA:
        raise HarnessContractError("harness result is invalid")
    events = [
        public_event_projection(item)
        for item in result.get("events", [])
        if isinstance(item, Mapping)
    ]
    public = {
        "schema": HARNESS_SCHEMA,
        "run_id": result.get("run_id"),
        "turn_id": result.get("turn_id"),
        "status": result.get("status"),
        "steps": int(result.get("steps", 0) or 0),
        "model_call_count": int(result.get("model_call_count", 0) or 0),
        "tool_call_count": int(result.get("tool_call_count", 0) or 0),
        "duration_ms": int(result.get("duration_ms", 0) or 0),
        "events": events,
    }
    public["trace_sha256"] = canonical_sha256(public)
    return public
