"""Stable contracts for the TeachLab agent harness.

The harness is deliberately domain-neutral.  Teaching policy, student-state
updates, and gold-answer isolation live in adapters above this layer; this
module only defines execution, tool, event, budget, and recovery contracts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence


HARNESS_SCHEMA = "teaching_skill_miner.agent_harness.v1"
HARNESS_EVENT_SCHEMA = "teaching_skill_miner.agent_harness_event.v1"
HARNESS_CHECKPOINT_SCHEMA = "teaching_skill_miner.agent_harness_checkpoint.v1"
HARNESS_MODEL_REQUEST_SCHEMA = "teaching_skill_miner.agent_harness_model_request.v1"
HARNESS_MODEL_RESPONSE_SCHEMA = "teaching_skill_miner.agent_harness_model_response.v1"

TERMINAL_EVENT_TYPES = frozenset(
    {"run.completed", "run.cancelled", "run.failed", "run.handoff"}
)
HARNESS_EVENT_TYPES = frozenset(
    {
        "run.started",
        *TERMINAL_EVENT_TYPES,
        "model.started",
        "model.completed",
        "model.failed",
        "model.retrying",
        "model.retry_suppressed",
        "message.start",
        "message.delta",
        "message.end",
        "reasoning.delta",
        "usage.update",
        "tool_call.delta",
        "tool.requested",
        "tool.started",
        "tool.progress",
        "tool.retrying",
        "tool.completed",
        "tool.failed",
        "tool.rejected",
        "tool.replayed",
        "tool.reconciled",
        "input.steered",
        "guard.triggered",
        "action.started",
        "action.completed",
        "action.superseded",
        "progress.updated",
        "operation.result",
        "state.committed",
    }
)
REPLAY_POLICIES = frozenset({"safe", "idempotent", "never"})
RISK_LEVELS = frozenset({"low", "medium", "high"})


class HarnessError(RuntimeError):
    """Base class for harness failures."""


class HarnessContractError(HarnessError, ValueError):
    """Raised when a caller, model, event, or tool violates a contract."""


class HarnessCancelled(HarnessError):
    """Raised cooperatively when the current run is cancelled."""


class HarnessDeadlineExceeded(HarnessError, TimeoutError):
    """Raised when the run or one effect exceeds its deadline."""


class ToolPermissionError(HarnessError, PermissionError):
    """Raised internally when a tool permission is not granted."""


class ToolExecutionError(HarnessError):
    """A normalized tool failure which can be returned to the model."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "tool_execution_failed",
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class ToolTransientError(ToolExecutionError):
    """A tool failure that may be retried under its replay policy."""

    def __init__(self, message: str, *, code: str = "tool_transient_error") -> None:
        super().__init__(message, code=code, retryable=True)


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Bounded retry policy shared by models and tools."""

    max_attempts: int = 2
    initial_backoff_seconds: float = 0.1
    backoff_multiplier: float = 2.0
    max_backoff_seconds: float = 2.0

    def validated(self) -> "RetryPolicy":
        if not 1 <= self.max_attempts <= 8:
            raise HarnessContractError("max_attempts must be in [1, 8]")
        values = (
            self.initial_backoff_seconds,
            self.backoff_multiplier,
            self.max_backoff_seconds,
        )
        if any(not math.isfinite(value) for value in values):
            raise HarnessContractError("retry timing values must be finite")
        if self.initial_backoff_seconds < 0 or self.max_backoff_seconds < 0:
            raise HarnessContractError("retry delays must be non-negative")
        if self.backoff_multiplier < 1:
            raise HarnessContractError("retry multiplier must be at least 1")
        return self

    def delay_for_retry(self, completed_attempts: int) -> float:
        """Return the delay before the next attempt.

        ``completed_attempts`` is one-based: after attempt one, the first
        delay is returned.
        """

        self.validated()
        exponent = max(0, completed_attempts - 1)
        return min(
            self.max_backoff_seconds,
            self.initial_backoff_seconds * (self.backoff_multiplier**exponent),
        )


@dataclass(frozen=True, slots=True)
class HarnessLimits:
    """Hard budgets for one run, including all model and tool effects."""

    max_steps: int = 12
    max_model_calls: int = 12
    max_total_tool_calls: int = 24
    max_tool_calls_per_step: int = 8
    max_repeated_tool_calls: int = 2
    deadline_seconds: float = 90.0
    max_tool_output_chars: int = 12_000
    max_event_payload_chars: int = 24_000
    max_in_memory_events: int = 1_024
    max_in_memory_event_bytes: int = 2_000_000

    def validated(self) -> "HarnessLimits":
        integer_bounds = {
            "max_steps": (self.max_steps, 1, 64),
            "max_model_calls": (self.max_model_calls, 1, 128),
            "max_total_tool_calls": (self.max_total_tool_calls, 0, 256),
            "max_tool_calls_per_step": (self.max_tool_calls_per_step, 1, 32),
            "max_repeated_tool_calls": (self.max_repeated_tool_calls, 1, 8),
            "max_tool_output_chars": (self.max_tool_output_chars, 128, 200_000),
            "max_event_payload_chars": (
                self.max_event_payload_chars,
                256,
                500_000,
            ),
            "max_in_memory_events": (self.max_in_memory_events, 8, 20_000),
            "max_in_memory_event_bytes": (
                self.max_in_memory_event_bytes,
                64_000,
                100_000_000,
            ),
        }
        for name, (value, lower, upper) in integer_bounds.items():
            if isinstance(value, bool) or not lower <= value <= upper:
                raise HarnessContractError(f"{name} must be in [{lower}, {upper}]")
        if not math.isfinite(self.deadline_seconds) or not (
            0.05 <= self.deadline_seconds <= 86_400
        ):
            raise HarnessContractError("deadline_seconds must be in [0.05, 86400]")
        return self


@dataclass(frozen=True, slots=True)
class ToolCall:
    """One model-requested tool effect."""

    call_id: str
    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    idempotency_key: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, index: int = 0) -> "ToolCall":
        if not isinstance(value, Mapping):
            raise HarnessContractError("tool call must be an object")
        call_id = str(value.get("call_id", "")).strip() or f"call_{index + 1}"
        name = str(value.get("name", "")).strip()
        arguments = value.get("arguments", {})
        key = value.get("idempotency_key")
        if not name:
            raise HarnessContractError("tool call name is required")
        if not isinstance(arguments, Mapping):
            raise HarnessContractError("tool call arguments must be an object")
        if len(call_id) > 160 or len(name) > 128:
            raise HarnessContractError("tool call identifier is too long")
        if key is not None:
            if not isinstance(key, str) or not key.strip() or len(key) > 200:
                raise HarnessContractError("tool idempotency_key is invalid")
            key = key.strip()
        return cls(
            call_id=call_id,
            name=name,
            arguments=dict(arguments),
            idempotency_key=key,
        )

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "call_id": self.call_id,
            "name": self.name,
            "arguments": dict(self.arguments),
        }
        if self.idempotency_key is not None:
            value["idempotency_key"] = self.idempotency_key
        return value


@dataclass(frozen=True, slots=True)
class HarnessModelResponse:
    """Normalized decision produced by a provider or planner adapter."""

    kind: str
    tool_calls: tuple[ToolCall, ...] = ()
    output: Any = None
    reason: str = ""
    usage: Mapping[str, Any] = field(default_factory=dict)
    provider_request_id: str | None = None
    parallel_tool_calls: bool = False

    @classmethod
    def from_value(cls, value: Any) -> "HarnessModelResponse":
        if isinstance(value, cls):
            return value.validated()
        if not isinstance(value, Mapping):
            raise HarnessContractError("model response must be an object")
        kind = str(value.get("kind", "")).strip()
        raw_calls = value.get("tool_calls", [])
        if not isinstance(raw_calls, Sequence) or isinstance(raw_calls, (str, bytes)):
            raise HarnessContractError("model tool_calls must be an array")
        calls = tuple(
            ToolCall.from_mapping(call, index=index)
            for index, call in enumerate(raw_calls)
        )
        usage = value.get("usage", {})
        if not isinstance(usage, Mapping):
            raise HarnessContractError("model usage must be an object")
        request_id = value.get("provider_request_id")
        if request_id is not None and not isinstance(request_id, str):
            raise HarnessContractError("provider_request_id must be a string")
        return cls(
            kind=kind,
            tool_calls=calls,
            output=value.get("output"),
            reason=str(value.get("reason", "")).strip(),
            usage=dict(usage),
            provider_request_id=request_id,
            parallel_tool_calls=bool(value.get("parallel_tool_calls", False)),
        ).validated()

    def validated(self) -> "HarnessModelResponse":
        if self.kind not in {"tool_calls", "final", "handoff"}:
            raise HarnessContractError("model response kind is unsupported")
        if self.kind == "tool_calls" and not self.tool_calls:
            raise HarnessContractError("tool_calls response must contain calls")
        if self.kind != "tool_calls" and self.tool_calls:
            raise HarnessContractError("terminal model response cannot contain tools")
        if self.kind == "handoff" and not self.reason:
            raise HarnessContractError("handoff reason is required")
        call_ids = [call.call_id for call in self.tool_calls]
        if len(call_ids) != len(set(call_ids)):
            raise HarnessContractError("tool call ids must be unique within a step")
        return self


@dataclass(frozen=True, slots=True)
class HarnessModelRequest:
    """Provider-neutral snapshot for the next planner decision."""

    run_id: str
    turn_id: str
    step: int
    context: Mapping[str, Any]
    observations: tuple[Mapping[str, Any], ...]
    tools: tuple[Mapping[str, Any], ...]
    state: Mapping[str, Any]
    schema: str = HARNESS_MODEL_REQUEST_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "run_id": self.run_id,
            "turn_id": self.turn_id,
            "step": self.step,
            "context": dict(self.context),
            "observations": [dict(item) for item in self.observations],
            "tools": [dict(item) for item in self.tools],
            "state": dict(self.state),
        }
