"""Durable checkpoints for one externally visible teaching operation.

The generic model/tool checkpoint records the state of a single agent loop.
An interactive Teach request is wider than that loop: it prepares domain
context, performs assessment and routing, validates the final action, commits
the session record, and only then publishes the terminal SSE result.  This
checkpoint is anchored to the *outer* stream journal so that there is one
program counter for that whole operation.  Nested model/tool journals are
diagnostic children and are never a competing source of domain truth.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Mapping

from .contracts import HarnessContractError
from .events import canonical_sha256


TEACH_OPERATION_CHECKPOINT_SCHEMA = (
    "teaching_skill_miner.teach_operation_checkpoint.v1"
)

_PROGRAM_COUNTERS = frozenset(
    {
        "prepared",
        "context_prepared",
        "assessment_pending",
        "route_tool_observed",
        "final_action_validated",
        "domain_effect_pending",
        "domain_committed",
        "stream_commit_pending",
        "message_committed",
        "result_committed",
        "state_committed",
        "completed",
        "handoff",
    }
)
_STATUSES = frozenset({"running", "completed", "handoff"})
_EFFECT_STATES = frozenset({"not_started", "pending", "committed", "unknown"})


@dataclass(frozen=True, slots=True)
class TeachOperationCheckpoint:
    """Tamper-evident program counter for one ``start`` or ``step`` request."""

    schema: str
    run_id: str
    turn_id: str
    operation_id: str
    operation: str
    request_fingerprint: str
    status: str
    program_counter: str
    next_sequence: int
    domain_effect_state: str
    inner_run_id: str
    inner_turn_id: str
    inner_checkpoint_sha256: str | None = None
    session_id: str | None = None
    context_version: int | None = None
    response_sha256: str | None = None
    checkpoint_sha256: str = ""

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        turn_id: str,
        operation_id: str,
        operation: str,
        request_fingerprint: str,
        status: str,
        program_counter: str,
        next_sequence: int,
        domain_effect_state: str,
        inner_run_id: str,
        inner_turn_id: str,
        inner_checkpoint_sha256: str | None = None,
        session_id: str | None = None,
        context_version: int | None = None,
        response_sha256: str | None = None,
    ) -> "TeachOperationCheckpoint":
        material: dict[str, Any] = {
            "schema": TEACH_OPERATION_CHECKPOINT_SCHEMA,
            "run_id": run_id,
            "turn_id": turn_id,
            "operation_id": operation_id,
            "operation": operation,
            "request_fingerprint": request_fingerprint,
            "status": status,
            "program_counter": program_counter,
            "next_sequence": next_sequence,
            "domain_effect_state": domain_effect_state,
            "inner_run_id": inner_run_id,
            "inner_turn_id": inner_turn_id,
            "inner_checkpoint_sha256": inner_checkpoint_sha256,
            "session_id": session_id,
            "context_version": context_version,
            "response_sha256": response_sha256,
        }
        return cls(
            **material,
            checkpoint_sha256=canonical_sha256(material),
        ).validated()

    @classmethod
    def from_value(
        cls, value: "TeachOperationCheckpoint | Mapping[str, Any]"
    ) -> "TeachOperationCheckpoint":
        if isinstance(value, cls):
            return value.validated()
        if not isinstance(value, Mapping):
            raise HarnessContractError("Teach operation checkpoint must be an object")
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(value).difference(allowed))
        if unknown:
            raise HarnessContractError(
                "Teach operation checkpoint contains unknown fields: "
                + ", ".join(unknown)
            )
        try:
            checkpoint = cls(
                schema=str(value.get("schema", "")),
                run_id=str(value.get("run_id", "")),
                turn_id=str(value.get("turn_id", "")),
                operation_id=str(value.get("operation_id", "")),
                operation=str(value.get("operation", "")),
                request_fingerprint=str(value.get("request_fingerprint", "")),
                status=str(value.get("status", "")),
                program_counter=str(value.get("program_counter", "")),
                next_sequence=int(value.get("next_sequence", 0)),
                domain_effect_state=str(value.get("domain_effect_state", "")),
                inner_run_id=str(value.get("inner_run_id", "")),
                inner_turn_id=str(value.get("inner_turn_id", "")),
                inner_checkpoint_sha256=(
                    str(value["inner_checkpoint_sha256"])
                    if value.get("inner_checkpoint_sha256") is not None
                    else None
                ),
                session_id=(
                    str(value["session_id"])
                    if value.get("session_id") is not None
                    else None
                ),
                context_version=(
                    int(value["context_version"])
                    if value.get("context_version") is not None
                    else None
                ),
                response_sha256=(
                    str(value["response_sha256"])
                    if value.get("response_sha256") is not None
                    else None
                ),
                checkpoint_sha256=str(value.get("checkpoint_sha256", "")),
            )
        except (TypeError, ValueError) as exc:
            raise HarnessContractError(
                "Teach operation checkpoint is malformed"
            ) from exc
        return checkpoint.validated()

    def _material(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "run_id": self.run_id,
            "turn_id": self.turn_id,
            "operation_id": self.operation_id,
            "operation": self.operation,
            "request_fingerprint": self.request_fingerprint,
            "status": self.status,
            "program_counter": self.program_counter,
            "next_sequence": self.next_sequence,
            "domain_effect_state": self.domain_effect_state,
            "inner_run_id": self.inner_run_id,
            "inner_turn_id": self.inner_turn_id,
            "inner_checkpoint_sha256": self.inner_checkpoint_sha256,
            "session_id": self.session_id,
            "context_version": self.context_version,
            "response_sha256": self.response_sha256,
        }

    def validated(self) -> "TeachOperationCheckpoint":
        if self.schema != TEACH_OPERATION_CHECKPOINT_SCHEMA:
            raise HarnessContractError("Teach operation checkpoint schema is invalid")
        if not all(
            (
                self.run_id,
                self.turn_id,
                self.operation_id,
                self.inner_run_id,
                self.inner_turn_id,
            )
        ):
            raise HarnessContractError(
                "Teach operation checkpoint identifiers are required"
            )
        if self.operation not in {"start", "step"}:
            raise HarnessContractError("Teach operation is invalid")
        if len(self.request_fingerprint) != 64:
            raise HarnessContractError("Teach request fingerprint is invalid")
        if self.status not in _STATUSES:
            raise HarnessContractError("Teach operation checkpoint status is invalid")
        if self.program_counter not in _PROGRAM_COUNTERS:
            raise HarnessContractError("Teach operation program counter is invalid")
        if self.next_sequence < 1:
            raise HarnessContractError("Teach operation checkpoint cursor is invalid")
        if self.domain_effect_state not in _EFFECT_STATES:
            raise HarnessContractError("Teach domain effect state is invalid")
        if self.context_version is not None and self.context_version < 0:
            raise HarnessContractError("Teach context version is invalid")
        for digest in (
            self.inner_checkpoint_sha256,
            self.response_sha256,
            self.checkpoint_sha256,
        ):
            if digest is not None and len(digest) != 64:
                raise HarnessContractError("Teach operation checkpoint hash is invalid")
        if canonical_sha256(self._material()) != self.checkpoint_sha256:
            raise HarnessContractError("Teach operation checkpoint hash mismatch")
        return self

    def to_dict(self) -> dict[str, Any]:
        return {**self._material(), "checkpoint_sha256": self.checkpoint_sha256}


__all__ = ["TEACH_OPERATION_CHECKPOINT_SCHEMA", "TeachOperationCheckpoint"]
