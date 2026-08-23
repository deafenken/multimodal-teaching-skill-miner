"""Tamper-evident operation checkpoints for crash-safe resume."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Mapping

from .contracts import HARNESS_CHECKPOINT_SCHEMA, HarnessContractError, ToolCall
from .events import canonical_sha256


@dataclass(frozen=True, slots=True)
class HarnessCheckpoint:
    schema: str
    run_id: str
    turn_id: str
    status: str
    next_sequence: int
    next_step: int
    model_calls: int
    tool_calls: int
    context_sha256: str
    policy_sha256: str
    external_effect_started: bool
    repeated_calls: Mapping[str, int] = field(default_factory=dict)
    observations: tuple[Mapping[str, Any], ...] = ()
    idempotency_receipts: Mapping[str, Mapping[str, Any]] = field(
        default_factory=dict
    )
    completed_call_receipts: Mapping[str, Mapping[str, Any]] = field(
        default_factory=dict
    )
    pending_effect: Mapping[str, Any] | None = None
    checkpoint_sha256: str = ""

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        turn_id: str,
        status: str,
        next_sequence: int,
        next_step: int,
        model_calls: int,
        tool_calls: int,
        context_sha256: str,
        policy_sha256: str,
        external_effect_started: bool,
        repeated_calls: Mapping[str, int],
        observations: tuple[Mapping[str, Any], ...],
        idempotency_receipts: Mapping[str, Mapping[str, Any]],
        completed_call_receipts: Mapping[str, Mapping[str, Any]],
        pending_effect: Mapping[str, Any] | None,
    ) -> "HarnessCheckpoint":
        material = {
            "schema": HARNESS_CHECKPOINT_SCHEMA,
            "run_id": run_id,
            "turn_id": turn_id,
            "status": status,
            "next_sequence": next_sequence,
            "next_step": next_step,
            "model_calls": model_calls,
            "tool_calls": tool_calls,
            "context_sha256": context_sha256,
            "policy_sha256": policy_sha256,
            "external_effect_started": external_effect_started,
            "repeated_calls": dict(repeated_calls),
            "observations": [dict(item) for item in observations],
            "idempotency_receipts": {
                str(key): dict(value) for key, value in idempotency_receipts.items()
            },
            "completed_call_receipts": {
                str(key): dict(value)
                for key, value in completed_call_receipts.items()
            },
            "pending_effect": dict(pending_effect) if pending_effect else None,
        }
        return cls(
            schema=material["schema"],
            run_id=material["run_id"],
            turn_id=material["turn_id"],
            status=material["status"],
            next_sequence=material["next_sequence"],
            next_step=material["next_step"],
            model_calls=material["model_calls"],
            tool_calls=material["tool_calls"],
            context_sha256=material["context_sha256"],
            policy_sha256=material["policy_sha256"],
            external_effect_started=material["external_effect_started"],
            repeated_calls=material["repeated_calls"],
            observations=tuple(material["observations"]),
            idempotency_receipts=material["idempotency_receipts"],
            completed_call_receipts=material["completed_call_receipts"],
            pending_effect=material["pending_effect"],
            checkpoint_sha256=canonical_sha256(material),
        ).validated()

    @classmethod
    def from_value(cls, value: "HarnessCheckpoint | Mapping[str, Any]") -> "HarnessCheckpoint":
        if isinstance(value, cls):
            return value.validated()
        if not isinstance(value, Mapping):
            raise HarnessContractError("harness checkpoint must be an object")
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(value).difference(allowed))
        if unknown:
            raise HarnessContractError(
                "harness checkpoint contains unknown fields: " + ", ".join(unknown)
            )
        try:
            checkpoint = cls(
                schema=str(value.get("schema", "")),
                run_id=str(value.get("run_id", "")),
                turn_id=str(value.get("turn_id", "")),
                status=str(value.get("status", "")),
                next_sequence=int(value.get("next_sequence", 0)),
                next_step=int(value.get("next_step", 0)),
                model_calls=int(value.get("model_calls", 0)),
                tool_calls=int(value.get("tool_calls", 0)),
                context_sha256=str(value.get("context_sha256", "")),
                policy_sha256=str(value.get("policy_sha256", "")),
                external_effect_started=value.get("external_effect_started", False),
                repeated_calls=dict(value.get("repeated_calls", {})),
                observations=tuple(
                    dict(item) for item in value.get("observations", ())
                ),
                idempotency_receipts={
                    str(key): dict(item)
                    for key, item in dict(
                        value.get("idempotency_receipts", {})
                    ).items()
                },
                completed_call_receipts={
                    str(key): dict(item)
                    for key, item in dict(
                        value.get("completed_call_receipts", {})
                    ).items()
                },
                pending_effect=(
                    dict(value["pending_effect"])
                    if isinstance(value.get("pending_effect"), Mapping)
                    else None
                ),
                checkpoint_sha256=str(value.get("checkpoint_sha256", "")),
            )
        except (TypeError, ValueError) as exc:
            raise HarnessContractError("harness checkpoint is malformed") from exc
        return checkpoint.validated()

    def _material(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "run_id": self.run_id,
            "turn_id": self.turn_id,
            "status": self.status,
            "next_sequence": self.next_sequence,
            "next_step": self.next_step,
            "model_calls": self.model_calls,
            "tool_calls": self.tool_calls,
            "context_sha256": self.context_sha256,
            "policy_sha256": self.policy_sha256,
            "external_effect_started": self.external_effect_started,
            "repeated_calls": dict(self.repeated_calls),
            "observations": [dict(item) for item in self.observations],
            "idempotency_receipts": {
                str(key): dict(value)
                for key, value in self.idempotency_receipts.items()
            },
            "completed_call_receipts": {
                str(key): dict(value)
                for key, value in self.completed_call_receipts.items()
            },
            "pending_effect": dict(self.pending_effect) if self.pending_effect else None,
        }

    def validated(self) -> "HarnessCheckpoint":
        if self.schema != HARNESS_CHECKPOINT_SCHEMA:
            raise HarnessContractError("harness checkpoint schema is invalid")
        if not self.run_id or not self.turn_id:
            raise HarnessContractError("harness checkpoint identifiers are required")
        if self.status not in {"running", "completed", "cancelled", "failed", "handoff"}:
            raise HarnessContractError("harness checkpoint status is invalid")
        if self.next_sequence < 1 or self.next_step < 1:
            raise HarnessContractError("harness checkpoint cursor is invalid")
        if self.model_calls < 0 or self.tool_calls < 0:
            raise HarnessContractError("harness checkpoint counters are invalid")
        if len(self.context_sha256) != 64:
            raise HarnessContractError("harness checkpoint context hash is invalid")
        if len(self.policy_sha256) != 64:
            raise HarnessContractError("harness checkpoint policy hash is invalid")
        if type(self.external_effect_started) is not bool:
            raise HarnessContractError(
                "harness checkpoint external effect flag is invalid"
            )
        if self.pending_effect is not None:
            required_fields = {
                "kind",
                "step",
                "call",
                "tool_version",
                "replay_policy",
                "arguments_sha256",
                "effect_started",
                "remaining_calls",
            }
            optional_fields = {"hook_guarded"}
            actual_fields = set(self.pending_effect)
            if (
                not required_fields <= actual_fields
                or actual_fields - required_fields - optional_fields
            ):
                raise HarnessContractError(
                    "harness checkpoint pending effect fields are invalid"
                )
            if self.pending_effect.get("kind") != "tool":
                raise HarnessContractError(
                    "harness checkpoint pending effect kind is invalid"
                )
            step = self.pending_effect.get("step")
            effect_started = self.pending_effect.get("effect_started")
            hook_guarded = self.pending_effect.get("hook_guarded", False)
            remaining = self.pending_effect.get("remaining_calls")
            raw_call = self.pending_effect.get("call")
            if (
                isinstance(step, bool)
                or not isinstance(step, int)
                or step < 1
                or type(effect_started) is not bool
                or type(hook_guarded) is not bool
                or not isinstance(raw_call, Mapping)
                or not isinstance(remaining, list)
                or len(remaining) > 8
            ):
                raise HarnessContractError(
                    "harness checkpoint pending effect is invalid"
                )
            ToolCall.from_mapping(raw_call)
            for item in remaining:
                if not isinstance(item, Mapping):
                    raise HarnessContractError(
                        "harness checkpoint remaining tool call is invalid"
                    )
                ToolCall.from_mapping(item)
            if effect_started and not self.external_effect_started:
                raise HarnessContractError(
                    "pending effect contradicts the run effect fence"
                )
        if canonical_sha256(self._material()) != self.checkpoint_sha256:
            raise HarnessContractError("harness checkpoint hash mismatch")
        return self

    def to_dict(self) -> dict[str, Any]:
        return {**self._material(), "checkpoint_sha256": self.checkpoint_sha256}
