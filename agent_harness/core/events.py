"""Typed event envelopes and privacy-safe trace projection."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
import re
from threading import RLock
from typing import Any, Callable, Mapping

from .contracts import (
    HARNESS_EVENT_TYPES,
    HARNESS_EVENT_SCHEMA,
    HarnessContractError,
    TERMINAL_EVENT_TYPES,
)


def canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise HarnessContractError("harness value is not canonical JSON") from exc


def canonical_sha256(value: Any) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _required_text(payload: Mapping[str, Any], field: str, *, limit: int = 160) -> None:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise HarnessContractError(f"{field} is required for this event")


def _required_integer(
    payload: Mapping[str, Any], field: str, *, minimum: int = 0
) -> None:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise HarnessContractError(f"{field} is invalid for this event")


def _validate_event_payload(event_type: str, payload: Mapping[str, Any]) -> None:
    """Validate the semantic payload contract before durable persistence.

    The public JSON Schema mirrors the externally consumed stream contracts;
    this fail-closed validator also covers internal execution fields.  It is
    intentionally small and type-specific: an event journal must never accept
    a record which the reducer cannot interpret on replay.
    """

    if event_type == "run.started":
        if type(payload.get("resumed")) is not bool:
            raise HarnessContractError("resumed is required for run.started")
        for field in (
            "instructions_sha256",
            "hooks_sha256",
            "mcp_sha256",
            "source_messages_sha256",
            "summary_sha256",
            "active_context_sha256",
        ):
            if field in payload and re.fullmatch(
                r"[0-9a-f]{64}", str(payload.get(field, ""))
            ) is None:
                raise HarnessContractError(f"{field} is invalid")
        if "compaction_id" in payload:
            _required_text(payload, "compaction_id")
        for field in (
            "instruction_count",
            "instruction_bytes",
            "hook_count",
            "trusted_hook_count",
            "disabled_hook_count",
            "mcp_server_count",
            "mcp_tool_count",
            "source_message_count",
            "active_message_count",
        ):
            if field in payload:
                _required_integer(payload, field, minimum=0)
    elif event_type in {"run.completed", "run.cancelled", "run.handoff"}:
        _required_text(payload, "reason_code", limit=120)
    elif event_type == "run.failed":
        _required_text(payload, "error_code", limit=120)
        _required_text(payload, "reason_code", limit=120)
    elif event_type == "model.started":
        _required_integer(payload, "attempt", minimum=1)
        _required_integer(payload, "step", minimum=1)
    elif event_type == "model.completed":
        _required_integer(payload, "attempt", minimum=1)
        _required_integer(payload, "step", minimum=1)
        _required_text(payload, "kind", limit=80)
        for field in (
            "model_response_sha256",
            "output_sha256",
            "reason_sha256",
            "usage_sha256",
        ):
            if field in payload and re.fullmatch(
                r"[0-9a-f]{64}", str(payload.get(field, ""))
            ) is None:
                raise HarnessContractError(f"{field} is invalid")
        if "tool_call_count" in payload:
            _required_integer(payload, "tool_call_count", minimum=0)
        if "tool_calls" in payload:
            tool_calls = payload.get("tool_calls")
            if not isinstance(tool_calls, list):
                raise HarnessContractError("tool_calls is invalid for model.completed")
            if payload.get("tool_call_count") != len(tool_calls):
                raise HarnessContractError(
                    "tool_call_count does not match model.completed summaries"
                )
            for summary in tool_calls:
                if not isinstance(summary, Mapping):
                    raise HarnessContractError(
                        "tool call summary is invalid for model.completed"
                    )
                _required_text(summary, "call_id")
                _required_text(summary, "tool_name", limit=128)
                if re.fullmatch(
                    r"[0-9a-f]{64}",
                    str(summary.get("arguments_sha256", "")),
                ) is None:
                    raise HarnessContractError(
                        "tool call arguments_sha256 is invalid"
                    )
    elif event_type == "model.failed":
        _required_integer(payload, "attempt", minimum=1)
        _required_integer(payload, "step", minimum=1)
        _required_text(payload, "safe_code", limit=128)
        if type(payload.get("retryable")) is not bool:
            raise HarnessContractError("retryable is required for model.failed")
    elif event_type == "model.retrying":
        _required_integer(payload, "attempt", minimum=1)
        _required_integer(payload, "next_attempt", minimum=2)
        _required_integer(payload, "delay_ms", minimum=0)
    elif event_type == "model.retry_suppressed":
        _required_integer(payload, "attempt", minimum=1)
        _required_text(payload, "reason_code", limit=128)
    elif event_type == "message.start":
        _required_text(payload, "channel", limit=40)
    elif event_type in {"message.delta", "reasoning.delta"}:
        if not isinstance(payload.get("delta"), str):
            raise HarnessContractError("delta is required for this event")
    elif event_type == "message.end":
        message_hash = payload.get("message_sha256")
        if message_hash is not None and (
            not isinstance(message_hash, str)
            or re.fullmatch(r"[0-9a-f]{64}", message_hash) is None
        ):
            raise HarnessContractError("message_sha256 is invalid")
        if "chars" in payload:
            _required_integer(payload, "chars", minimum=0)
    elif event_type == "usage.update":
        counters = {
            key: value
            for key, value in payload.items()
            if key not in {"channel", "delta"}
        }
        if not counters or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0
            for value in counters.values()
        ):
            raise HarnessContractError("usage.update requires non-negative counters")
    elif event_type == "tool_call.delta":
        _required_text(payload, "call_id")
        _required_text(payload, "tool_name", limit=128)
        _required_text(payload, "phase", limit=40)
    elif event_type in {"approval.requested", "approval.resolved"}:
        _required_text(payload, "approval_id")
        _required_text(payload, "call_id")
        _required_text(payload, "tool_name", limit=128)
        _required_text(payload, "tool_version", limit=64)
        _required_text(payload, "arguments_sha256", limit=64)
        _required_text(payload, "policy_sha256", limit=64)
        _required_text(payload, "risk", limit=16)
        if any(
            re.fullmatch(r"[0-9a-f]{64}", str(payload.get(field, ""))) is None
            for field in ("arguments_sha256", "policy_sha256")
        ):
            raise HarnessContractError("approval digest is invalid")
        if payload.get("risk") not in {"low", "medium", "high"}:
            raise HarnessContractError("approval risk is invalid")
        if event_type == "approval.requested":
            _required_text(payload, "policy_action", limit=16)
            if payload.get("policy_action") not in {"allow", "ask", "deny"}:
                raise HarnessContractError("approval policy_action is invalid")
            if "persistent_scope_allowed" in payload and type(
                payload.get("persistent_scope_allowed")
            ) is not bool:
                raise HarnessContractError(
                    "approval persistent_scope_allowed is invalid"
                )
        else:
            _required_text(payload, "verdict", limit=16)
            _required_text(payload, "reason_code", limit=128)
            if payload.get("verdict") not in {"allow", "deny"}:
                raise HarnessContractError("approval verdict is invalid")
    elif event_type.startswith("hook."):
        _required_text(payload, "invocation_id")
        _required_text(payload, "hook_id")
        _required_text(payload, "hook_event_name", limit=40)
        _required_text(payload, "call_id")
        _required_text(payload, "tool_name", limit=128)
        _required_integer(payload, "ordinal", minimum=1)
        for field in ("hook_sha256", "input_sha256"):
            if re.fullmatch(r"[0-9a-f]{64}", str(payload.get(field, ""))) is None:
                raise HarnessContractError(f"{field} is invalid")
        if event_type in {"hook.completed", "hook.failed"}:
            _required_integer(payload, "duration_ms", minimum=0)
            if re.fullmatch(
                r"[0-9a-f]{64}", str(payload.get("output_sha256", ""))
            ) is None:
                raise HarnessContractError("hook output_sha256 is invalid")
        if event_type == "hook.completed":
            _required_text(payload, "action", limit=16)
            if payload.get("action") not in {"pass", "ask", "deny"}:
                raise HarnessContractError("hook action is invalid")
        if event_type == "hook.failed":
            _required_text(payload, "error_code", limit=128)
    elif event_type.startswith("tool."):
        _required_text(payload, "call_id")
        _required_text(payload, "tool_name", limit=128)
        if event_type in {
            "tool.started",
            "tool.effect_started",
            "tool.completed",
            "tool.failed",
        }:
            _required_integer(payload, "attempt", minimum=1)
        if event_type == "tool.progress":
            _required_text(payload, "progress_kind", limit=80)
        if event_type in {"tool.failed", "tool.rejected"}:
            _required_text(payload, "error_code", limit=128)
        if event_type == "tool.completed":
            result_hash = payload.get("result_sha256")
            if (
                not isinstance(result_hash, str)
                or re.fullmatch(r"[0-9a-f]{64}", result_hash) is None
            ):
                raise HarnessContractError("result_sha256 is invalid")
            if canonical_sha256(payload.get("result")) != result_hash:
                raise HarnessContractError("tool result hash does not match payload")
        if event_type == "tool.replayed":
            _required_text(payload, "idempotency_key")
            _required_text(payload, "source_call_id")
    elif event_type == "input.steered":
        _required_text(payload, "input_id")
        content = payload.get("content")
        if not isinstance(content, str) or not content.strip():
            raise HarnessContractError("content is required for input.steered")
    elif event_type in {"guard.triggered", "action.superseded"}:
        _required_text(payload, "reason_code", limit=128)
    elif event_type == "action.started":
        if not any(
            isinstance(payload.get(field), str) and payload.get(field).strip()
            for field in ("action_type", "operation")
        ):
            raise HarnessContractError(
                "action_type or operation is required for action.started"
            )
    elif event_type == "action.completed":
        output_hash = payload.get("output_sha256")
        if (
            not isinstance(output_hash, str)
            or re.fullmatch(r"[0-9a-f]{64}", output_hash) is None
        ):
            raise HarnessContractError("output_sha256 is invalid")
        if (
            "output" in payload
            and canonical_sha256(payload.get("output")) != output_hash
        ):
            raise HarnessContractError("action output hash does not match payload")
    elif event_type == "progress.updated":
        if not any(
            isinstance(payload.get(field), str) and payload.get(field).strip()
            for field in ("stage", "phase")
        ):
            raise HarnessContractError(
                "stage or phase is required for progress.updated"
            )
    elif event_type == "operation.result":
        if "result" not in payload and not isinstance(payload.get("operation"), str):
            raise HarnessContractError("result is required for operation.result")
    elif event_type == "state.committed":
        _required_text(payload, "operation", limit=80)


class HarnessEventEmitter:
    """Serialize event creation and enforce one terminal event."""

    def __init__(
        self,
        *,
        run_id: str,
        turn_id: str,
        next_sequence: int = 1,
        max_payload_chars: int = 24_000,
        max_in_memory_events: int = 1_024,
        max_in_memory_event_bytes: int = 2_000_000,
        durable_sink: Callable[[Mapping[str, Any]], Any] | None = None,
        event_sink: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> None:
        if (
            not isinstance(run_id, str)
            or not run_id.strip()
            or len(run_id.strip()) > 160
            or not isinstance(turn_id, str)
            or not turn_id.strip()
            or len(turn_id.strip()) > 160
        ):
            raise HarnessContractError("run_id and turn_id are required")
        if next_sequence < 1:
            raise HarnessContractError("next event sequence must be positive")
        if max_in_memory_events < 1 or max_in_memory_event_bytes < 1:
            raise HarnessContractError("event history limits must be positive")
        self.run_id = run_id.strip()
        self.turn_id = turn_id.strip()
        self._next_sequence = next_sequence
        self._max_payload_chars = max_payload_chars
        self._max_in_memory_events = max_in_memory_events
        self._max_in_memory_event_bytes = max_in_memory_event_bytes
        self._durable_sink = durable_sink
        self._event_sink = event_sink
        self._events: list[dict[str, Any]] = []
        self._events_bytes = 0
        self._dropped_through_sequence = 0
        self._terminal_type: str | None = None
        self._lock = RLock()

    @property
    def next_sequence(self) -> int:
        with self._lock:
            return self._next_sequence

    @property
    def terminal_type(self) -> str | None:
        with self._lock:
            return self._terminal_type

    @property
    def events(self) -> list[dict[str, Any]]:
        with self._lock:
            return deepcopy(self._events)

    @property
    def dropped_through_sequence(self) -> int:
        with self._lock:
            return self._dropped_through_sequence

    def emit(
        self,
        event_type: str,
        payload: Mapping[str, Any] | None = None,
        *,
        causation_id: str | None = None,
    ) -> dict[str, Any]:
        clean_type = str(event_type).strip()
        if clean_type not in HARNESS_EVENT_TYPES:
            raise HarnessContractError("event type is invalid")
        clean_payload = dict(payload or {})
        _validate_event_payload(clean_type, clean_payload)
        rendered = canonical_json(clean_payload)
        if len(rendered) > self._max_payload_chars:
            raise HarnessContractError("event payload exceeds the configured budget")
        with self._lock:
            if self._terminal_type is not None:
                raise HarnessContractError("event emitted after terminal event")
            sequence = self._next_sequence
            identity = {
                "run_id": self.run_id,
                "turn_id": self.turn_id,
                "sequence": sequence,
                "type": clean_type,
                "payload_sha256": sha256(rendered.encode("utf-8")).hexdigest(),
            }
            event = {
                "schema": HARNESS_EVENT_SCHEMA,
                "event_id": canonical_sha256(identity)[:32],
                "run_id": self.run_id,
                "turn_id": self.turn_id,
                "sequence": sequence,
                "type": clean_type,
                "timestamp": datetime.now(timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
                "payload": clean_payload,
            }
            if causation_id:
                event["causation_id"] = str(causation_id)[:160]
            # Persistence is the authoritative commit boundary.  A durable
            # sink must acknowledge the event before it is made visible in
            # memory or delivered to a transport/UI sink.  If persistence
            # fails, the emitter cursor remains unchanged and callers must
            # fail closed rather than presenting an uncommitted event.
            if self._durable_sink is not None:
                acknowledgement = self._durable_sink(deepcopy(event))
                if not bool(getattr(acknowledgement, "durable", False)):
                    raise HarnessContractError(
                        "durable event sink did not acknowledge persistence"
                    )
            if clean_type in TERMINAL_EVENT_TYPES:
                self._terminal_type = clean_type
            self._next_sequence += 1
            self._events.append(event)
            self._events_bytes += len(canonical_json(event).encode("utf-8"))
            while self._events and (
                len(self._events) > self._max_in_memory_events
                or self._events_bytes > self._max_in_memory_event_bytes
            ):
                removed = self._events.pop(0)
                self._events_bytes -= len(canonical_json(removed).encode("utf-8"))
                self._dropped_through_sequence = int(removed["sequence"])
            if self._event_sink is not None:
                # A transport/UI observer is downstream of the authoritative
                # commit boundary.  It may detach or fail independently, but
                # it must never turn an already-durable event into a failed
                # Agent run.  Reconnect consumers recover from the journal.
                try:
                    self._event_sink(deepcopy(event))
                except Exception:
                    pass
            return deepcopy(event)


_PUBLIC_PAYLOAD_FIELDS: dict[str, tuple[str, ...]] = {
    "run.started": (
        "resumed",
        "instructions_sha256",
        "instruction_count",
        "instruction_bytes",
        "hooks_sha256",
        "hook_count",
        "trusted_hook_count",
        "disabled_hook_count",
        "mcp_sha256",
        "mcp_server_count",
        "mcp_tool_count",
        "compaction_id",
        "source_message_count",
        "source_messages_sha256",
        "summary_sha256",
        "active_context_sha256",
        "active_message_count",
    ),
    "run.completed": ("reason_code", "duration_ms"),
    "run.cancelled": ("reason_code",),
    "run.failed": ("error_code", "reason_code"),
    "run.handoff": ("reason_code",),
    "model.started": ("attempt", "step"),
    "model.completed": (
        "attempt",
        "step",
        "kind",
        "provider_request_id",
        "model_response_sha256",
        "tool_call_count",
        "tool_calls",
        "output_sha256",
        "reason_sha256",
        "usage_sha256",
    ),
    "model.failed": (
        "attempt",
        "step",
        "error_type",
        "error_kind",
        "safe_code",
        "retryable",
        "classifier_valid",
        "retry_after_ms",
    ),
    "model.retrying": (
        "attempt",
        "next_attempt",
        "delay_ms",
        "error_kind",
        "safe_code",
        "retry_after_ms",
    ),
    "model.retry_suppressed": (
        "attempt",
        "reason_code",
        "error_kind",
        "safe_code",
    ),
    "approval.requested": (
        "approval_id",
        "call_id",
        "tool_name",
        "tool_version",
        "risk",
        "policy_action",
    ),
    "approval.resolved": (
        "approval_id",
        "call_id",
        "tool_name",
        "tool_version",
        "risk",
        "verdict",
        "reason_code",
    ),
    "hook.started": (
        "invocation_id",
        "hook_id",
        "hook_event_name",
        "hook_sha256",
        "input_sha256",
        "call_id",
        "tool_name",
        "ordinal",
    ),
    "hook.effect_started": (
        "invocation_id",
        "hook_id",
        "hook_event_name",
        "hook_sha256",
        "input_sha256",
        "call_id",
        "tool_name",
        "ordinal",
    ),
    "hook.completed": (
        "invocation_id",
        "hook_id",
        "hook_event_name",
        "hook_sha256",
        "input_sha256",
        "call_id",
        "tool_name",
        "ordinal",
        "duration_ms",
        "action",
        "output_sha256",
    ),
    "hook.failed": (
        "invocation_id",
        "hook_id",
        "hook_event_name",
        "hook_sha256",
        "input_sha256",
        "call_id",
        "tool_name",
        "ordinal",
        "duration_ms",
        "error_code",
        "output_sha256",
    ),
    "tool.requested": ("call_id", "tool_name", "tool_version"),
    "tool.started": ("call_id", "tool_name", "attempt"),
    "tool.effect_started": ("call_id", "tool_name", "attempt"),
    "tool.progress": ("call_id", "tool_name", "progress_kind"),
    "tool.completed": (
        "call_id",
        "tool_name",
        "attempt",
        "duration_ms",
        "result_sha256",
    ),
    "tool.failed": (
        "call_id",
        "tool_name",
        "attempt",
        "error_code",
        "error_type",
    ),
    "tool.rejected": ("call_id", "tool_name", "error_code"),
    "tool.replayed": ("call_id", "tool_name", "source_call_id"),
    "tool.reconciled": ("call_id", "tool_name", "reason_code"),
    "guard.triggered": ("reason_code", "tool_name", "signature"),
    "action.completed": ("output_sha256", "operation", "result_kind"),
    "action.started": ("operation",),
    "progress.updated": ("phase",),
    "operation.result": (),
    "state.committed": ("operation", "session_id"),
}


def public_event_projection(event: Mapping[str, Any]) -> dict[str, Any]:
    """Remove user/model/tool content while retaining lifecycle evidence."""

    event_type = str(event.get("type", ""))
    source = event.get("payload", {})
    if not isinstance(source, Mapping):
        source = {}
    allowed = _PUBLIC_PAYLOAD_FIELDS.get(event_type, ())
    payload = {name: deepcopy(source[name]) for name in allowed if name in source}
    return {
        "schema": event.get("schema"),
        "event_id": event.get("event_id"),
        "run_id": event.get("run_id"),
        "turn_id": event.get("turn_id"),
        "sequence": event.get("sequence"),
        "type": event_type,
        "payload": payload,
    }
