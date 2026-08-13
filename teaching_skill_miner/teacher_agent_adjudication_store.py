"""Durable local store and facade for Teaching Agent adjudication.

The journal is a canonical JSONL hash chain.  Every mutating facade call takes
an exclusive POSIX file lock, replays the authoritative log, evaluates the
in-memory adjudication reducer, and appends one fsync-fenced transaction.  This
makes optimistic item versions meaningful across processes, not just threads.

Only an incomplete final line is repaired.  A complete line with invalid JSON,
an invalid hash, or an impossible state transition fails closed.  The file is
owned by the current user, mode 0600, and is never opened through a symlink.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat
import threading
from typing import Any, TypeVar

try:
    import fcntl
except ImportError:  # pragma: no cover - durable mode is deliberately POSIX-only.
    fcntl = None  # type: ignore[assignment]

from .teacher_agent_adjudication import (
    ADJUDICATION_AUDIT_SCHEMA,
    ADJUDICATION_INSTRUCTION_SCHEMA,
    ADJUDICATION_ITEM_SCHEMA,
    LOCAL_OPERATOR_IDENTITY,
    AdjudicationEvidenceDeletedError,
    TeacherAgentAdjudicationError,
    TeacherAgentAdjudicationQueue,
    authorize_authenticated_instruction,
    authenticated_instruction_authority_basis,
    canonical_sha256,
    reduce_adjudication,
)
from .teacher_agent_authority import (
    AUTHENTICATED_TEACHER_ACTOR,
    AUTHORITY_ASSURANCE,
    TeacherAuthorityError,
    validate_teacher_authority_revalidation_receipt,
    validate_teacher_authority_verification_receipt,
)


STORE_EVENT_SCHEMA = "teaching_skill_miner.teacher_agent_adjudication_store_event.v1"
STORE_TOMBSTONE_SCHEMA = (
    "teaching_skill_miner.teacher_agent_adjudication_erasure_tombstone.v1"
)
STORE_OPERATIONS = frozenset({"enqueue", "claim", "decide", "cancel_deleted_evidence"})

_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_ITEM_ID = re.compile(r"^adj_[0-9a-f]{24}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,159}$")
_SAFE_KC_ID = re.compile(r"^kc_[a-z0-9][a-z0-9_-]{2,80}$")
_UTC_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$")

_EVENT_KEYS = frozenset(
    {
        "schema",
        "seq",
        "recorded_at_utc",
        "operation",
        "previous_hash",
        "item_versions",
        "audit_receipts",
        "idempotency_bindings",
        "evidence_index_updates",
        "erasure_tombstones",
        "hash",
    }
)
_ITEM_KEYS = frozenset(
    {
        "schema",
        "item_id",
        "version",
        "status",
        "created_at",
        "updated_at",
        "source",
        "target_kc_ids",
        "target_scope",
        "original",
        "before_snapshot",
        "after_snapshot",
        "review_reason",
        "lease",
        "decision",
        "cancellation",
        "previous_version_sha256",
        "version_sha256",
    }
)
_AUDIT_KEYS = frozenset(
    {
        "schema",
        "seq",
        "event_type",
        "item_id",
        "from_version",
        "to_version",
        "result_version_sha256",
        "actor",
        "occurred_at",
        "request_sha256",
        "prior_receipt_sha256",
        "receipt_sha256",
    }
)
_BINDING_KEYS = frozenset(
    {
        "operation",
        "idempotency_key",
        "idempotency_key_sha256",
        "request_sha256",
        "response",
        "response_sha256",
    }
)
_INDEX_KEYS = frozenset(
    {"evidence_id", "evidence_id_sha256", "evidence_sha256", "item_id"}
)
_TOMBSTONE_KEYS = frozenset(
    {"schema", "evidence_id_sha256", "erased_at_utc", "reason", "actor"}
)
_ACTOR_KEYS = frozenset({"identity", "authenticated", "teacher_identity_claimed"})
_AUTHENTICATED_ACTOR_KEYS = frozenset(
    {
        "identity",
        "authenticated",
        "teacher_identity_claimed",
        "principal_sha256",
        "roles_sha256",
        "authority_id",
        "assurance",
    }
)


class TeacherAgentAdjudicationStoreError(RuntimeError):
    """Raised when durable adjudication state cannot be trusted or committed."""


class TeacherAgentAdjudicationStorePoisonedError(TeacherAgentAdjudicationStoreError):
    """Raised after an append and its rollback fence both fail."""


@dataclass(frozen=True, slots=True)
class TeacherAgentAdjudicationRecovery:
    """Defensive state reconstructed by strict journal replay."""

    versions: dict[str, list[dict[str, Any]]]
    audit_receipts: list[dict[str, Any]]
    idempotency_bindings: dict[tuple[str, str], dict[str, Any]]
    evidence_items: dict[str, str]
    erasure_tombstones: dict[str, dict[str, Any]]
    events: tuple[dict[str, Any], ...]
    file_size: int
    last_hash: str | None


_Result = TypeVar("_Result")


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TeacherAgentAdjudicationStoreError(
            "adjudication journal values must contain canonical JSON"
        ) from exc


def _digest(value: Any, *, field: str, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise TeacherAgentAdjudicationStoreError(
            f"adjudication journal {field} must be a lowercase SHA-256"
        )
    return value


def _required_string(value: Any, *, field: str, maximum: int = 512) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > maximum
    ):
        raise TeacherAgentAdjudicationStoreError(
            f"adjudication journal {field} must be a non-empty trimmed string"
        )
    return value


def _positive_integer(value: Any, *, field: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TeacherAgentAdjudicationStoreError(
            f"adjudication journal {field} must be an integer >= {minimum}"
        )
    return value


def _utc_timestamp(value: Any, *, field: str) -> str:
    result = _required_string(value, field=field, maximum=40)
    if _UTC_TIMESTAMP.fullmatch(result) is None:
        raise TeacherAgentAdjudicationStoreError(
            f"adjudication journal {field} must be a UTC timestamp"
        )
    try:
        datetime.fromisoformat(result.replace("Z", "+00:00"))
    except ValueError as exc:
        raise TeacherAgentAdjudicationStoreError(
            f"adjudication journal {field} is not a real timestamp"
        ) from exc
    return result


def _now_utc() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _evidence_id_digest(evidence_id: str) -> str:
    return sha256(evidence_id.encode("utf-8")).hexdigest()


def _validate_actor(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TeacherAgentAdjudicationStoreError(
            f"adjudication journal {field} actor shape is invalid"
        )
    if frozenset(value) == _AUTHENTICATED_ACTOR_KEYS:
        if (
            value.get("identity") != AUTHENTICATED_TEACHER_ACTOR
            or value.get("authenticated") is not True
            or value.get("teacher_identity_claimed") is not True
            or value.get("assurance") != AUTHORITY_ASSURANCE
            or not isinstance(value.get("authority_id"), str)
            or re.fullmatch(r"tauth_[0-9a-f]{24}", value["authority_id"]) is None
        ):
            raise TeacherAgentAdjudicationStoreError(
                f"adjudication journal {field} authenticated actor is invalid"
            )
        for key in ("principal_sha256", "roles_sha256"):
            _digest(value.get(key), field=f"{field}.{key}")
        return deepcopy(dict(value))
    if frozenset(value) != _ACTOR_KEYS:
        raise TeacherAgentAdjudicationStoreError(
            f"adjudication journal {field} actor shape is invalid"
        )
    identity = _required_string(value.get("identity"), field=f"{field}.identity")
    if value.get("authenticated") is not False:
        raise TeacherAgentAdjudicationStoreError(
            f"adjudication journal {field} must not claim authentication"
        )
    if value.get("teacher_identity_claimed") is not False:
        raise TeacherAgentAdjudicationStoreError(
            f"adjudication journal {field} must not claim teacher identity"
        )
    return {
        "identity": identity,
        "authenticated": False,
        "teacher_identity_claimed": False,
    }


def _same_authenticated_principal(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> bool:
    return all(
        left.get(key) == right.get(key)
        for key in (
            "identity",
            "authenticated",
            "teacher_identity_claimed",
            "principal_sha256",
            "roles_sha256",
            "assurance",
        )
    )


def _validate_authority_receipt(value: Any, *, field: str) -> dict[str, Any]:
    try:
        return validate_teacher_authority_verification_receipt(value)
    except (TeacherAuthorityError, TypeError) as exc:
        raise TeacherAgentAdjudicationStoreError(
            f"adjudication journal {field} authority receipt is invalid"
        ) from exc


def _validate_item(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or frozenset(value) != _ITEM_KEYS:
        raise TeacherAgentAdjudicationStoreError(
            f"adjudication journal {field} item shape is invalid"
        )
    item = deepcopy(dict(value))
    if item.get("schema") != ADJUDICATION_ITEM_SCHEMA:
        raise TeacherAgentAdjudicationStoreError(
            f"adjudication journal {field} item schema is unsupported"
        )
    if (
        not isinstance(item.get("item_id"), str)
        or _ITEM_ID.fullmatch(item["item_id"]) is None
    ):
        raise TeacherAgentAdjudicationStoreError(
            f"adjudication journal {field}.item_id is invalid"
        )
    _positive_integer(item.get("version"), field=f"{field}.version")
    if item.get("status") not in {"pending", "claimed", "decided", "cancelled"}:
        raise TeacherAgentAdjudicationStoreError(
            f"adjudication journal {field}.status is invalid"
        )
    _utc_timestamp(item.get("created_at"), field=f"{field}.created_at")
    _utc_timestamp(item.get("updated_at"), field=f"{field}.updated_at")
    _digest(
        item.get("previous_version_sha256"),
        field=f"{field}.previous_version_sha256",
        nullable=True,
    )
    declared = _digest(item.get("version_sha256"), field=f"{field}.version_sha256")
    material = deepcopy(item)
    material.pop("version_sha256")
    if declared != canonical_sha256(material):
        raise TeacherAgentAdjudicationStoreError(
            f"adjudication journal {field} item integrity check failed"
        )

    source = item.get("source")
    if not isinstance(source, Mapping) or frozenset(source) != {
        "session_id",
        "round_number",
        "action_id",
        "question_id",
        "history_event_sha256",
    }:
        raise TeacherAgentAdjudicationStoreError(
            f"adjudication journal {field}.source is invalid"
        )
    for key in ("session_id", "action_id", "question_id"):
        raw = _required_string(source.get(key), field=f"{field}.source.{key}")
        if _SAFE_ID.fullmatch(raw) is None:
            raise TeacherAgentAdjudicationStoreError(
                f"adjudication journal {field}.source.{key} is unsafe"
            )
    _positive_integer(
        source.get("round_number"),
        field=f"{field}.source.round_number",
        minimum=0,
    )
    _digest(
        source.get("history_event_sha256"),
        field=f"{field}.source.history_event_sha256",
    )

    target_kcs = item.get("target_kc_ids")
    if (
        not isinstance(target_kcs, list)
        or not 1 <= len(target_kcs) <= 4
        or len(set(target_kcs)) != len(target_kcs)
        or any(
            not isinstance(kc_id, str) or _SAFE_KC_ID.fullmatch(kc_id) is None
            for kc_id in target_kcs
        )
    ):
        raise TeacherAgentAdjudicationStoreError(
            f"adjudication journal {field}.target_kc_ids is invalid"
        )
    expected_scope = "exact_one" if len(target_kcs) == 1 else "bounded_set"
    if item.get("target_scope") != expected_scope:
        raise TeacherAgentAdjudicationStoreError(
            f"adjudication journal {field}.target_scope is inconsistent"
        )
    if item.get("review_reason") not in {
        "automatic_low_confidence",
        "authority_required",
        "learner_dispute",
        "rubric_conflict",
        "manual_quality_review",
    }:
        raise TeacherAgentAdjudicationStoreError(
            f"adjudication journal {field}.review_reason is invalid"
        )

    original = item.get("original")
    if not isinstance(original, Mapping) or frozenset(original) != {
        "assessment_id",
        "assessment_sha256",
        "evidence_id",
        "evidence_sha256",
        "rubric_id",
        "rubric_authority_sha256",
    }:
        raise TeacherAgentAdjudicationStoreError(
            f"adjudication journal {field}.original is invalid"
        )
    for key in ("assessment_id", "evidence_id", "rubric_id"):
        raw = _required_string(original.get(key), field=f"{field}.original.{key}")
        if _SAFE_ID.fullmatch(raw) is None:
            raise TeacherAgentAdjudicationStoreError(
                f"adjudication journal {field}.original.{key} is unsafe"
            )
    for key in (
        "assessment_sha256",
        "evidence_sha256",
        "rubric_authority_sha256",
    ):
        _digest(original.get(key), field=f"{field}.original.{key}")

    for snapshot_name in ("before_snapshot", "after_snapshot"):
        snapshot = item.get(snapshot_name)
        if not isinstance(snapshot, Mapping) or not set(snapshot).issubset(
            {"content_sha256", "schema", "state_version"}
        ):
            raise TeacherAgentAdjudicationStoreError(
                f"adjudication journal {field}.{snapshot_name} is invalid"
            )
        _digest(
            snapshot.get("content_sha256"),
            field=f"{field}.{snapshot_name}.content_sha256",
        )

    status = item["status"]
    if status == "pending" and any(
        item.get(key) is not None for key in ("lease", "decision", "cancellation")
    ):
        raise TeacherAgentAdjudicationStoreError(
            f"adjudication journal {field} pending state is inconsistent"
        )
    if status == "claimed":
        lease = item.get("lease")
        if not isinstance(lease, Mapping) or frozenset(lease) != {
            "actor",
            "claimed_at",
            "expires_at",
            "claim_token_sha256",
        }:
            raise TeacherAgentAdjudicationStoreError(
                f"adjudication journal {field}.lease is invalid"
            )
        actor = _validate_actor(lease.get("actor"), field=f"{field}.lease.actor")
        if actor["identity"] not in {
            LOCAL_OPERATOR_IDENTITY,
            AUTHENTICATED_TEACHER_ACTOR,
        }:
            raise TeacherAgentAdjudicationStoreError(
                f"adjudication journal {field}.lease actor is unsupported"
            )
        _utc_timestamp(lease.get("claimed_at"), field=f"{field}.lease.claimed_at")
        _utc_timestamp(lease.get("expires_at"), field=f"{field}.lease.expires_at")
        _digest(
            lease.get("claim_token_sha256"),
            field=f"{field}.lease.claim_token_sha256",
        )
        if item.get("decision") is not None or item.get("cancellation") is not None:
            raise TeacherAgentAdjudicationStoreError(
                f"adjudication journal {field} claimed state is inconsistent"
            )
    elif item.get("lease") is not None:
        raise TeacherAgentAdjudicationStoreError(
            f"adjudication journal {field} terminal/pending item retains a lease"
        )
    if status == "decided":
        decision = item.get("decision")
        legacy_decision_keys = {
            "kind",
            "reason_code",
            "actor",
            "evidence_id",
            "evidence_sha256",
            "correction",
            "decided_at",
            "idempotency_key_sha256",
        }
        authenticated_decision_keys = legacy_decision_keys | {"authority_receipt"}
        if not isinstance(decision, Mapping) or frozenset(decision) not in {
            frozenset(legacy_decision_keys),
            frozenset(authenticated_decision_keys),
        }:
            raise TeacherAgentAdjudicationStoreError(
                f"adjudication journal {field}.decision is missing"
            )
        actor = _validate_actor(decision.get("actor"), field=f"{field}.decision.actor")
        authority_receipt: dict[str, Any] | None = None
        if actor["identity"] == AUTHENTICATED_TEACHER_ACTOR:
            if frozenset(decision) != frozenset(authenticated_decision_keys):
                raise TeacherAgentAdjudicationStoreError(
                    f"adjudication journal {field}.decision authority receipt is missing"
                )
            authority_receipt = _validate_authority_receipt(
                decision.get("authority_receipt"),
                field=f"{field}.decision.authority_receipt",
            )
            if (
                authority_receipt.get("path") != "api/adjudication/decide"
                or authority_receipt.get("actor_principal_sha256")
                != actor["principal_sha256"]
                or authority_receipt.get("roles_sha256") != actor["roles_sha256"]
                or authority_receipt.get("authority_id") != actor["authority_id"]
            ):
                raise TeacherAgentAdjudicationStoreError(
                    f"adjudication journal {field}.decision authority binding is invalid"
                )
        elif actor["identity"] != LOCAL_OPERATOR_IDENTITY or frozenset(
            decision
        ) != frozenset(legacy_decision_keys):
            raise TeacherAgentAdjudicationStoreError(
                f"adjudication journal {field}.decision actor is unsupported"
            )
        if decision.get("kind") not in {"approve", "correct", "abstain"}:
            raise TeacherAgentAdjudicationStoreError(
                f"adjudication journal {field}.decision kind is invalid"
            )
        reason_code = decision.get("reason_code")
        if reason_code not in {
            "assessment_confirmed",
            "signal_misclassified",
            "alignment_misclassified",
            "knowledge_component_misaligned",
            "focus_dimension_misaligned",
            "insufficient_evidence",
            "rubric_ambiguous",
            "authority_unavailable",
            "conflict_unresolved",
        }:
            raise TeacherAgentAdjudicationStoreError(
                f"adjudication journal {field}.decision reason is invalid"
            )
        if decision["kind"] == "approve" and reason_code != "assessment_confirmed":
            raise TeacherAgentAdjudicationStoreError(
                f"adjudication journal {field}.decision approval reason is invalid"
            )
        if decision["kind"] == "abstain" and reason_code not in {
            "insufficient_evidence",
            "rubric_ambiguous",
            "authority_unavailable",
            "conflict_unresolved",
        }:
            raise TeacherAgentAdjudicationStoreError(
                f"adjudication journal {field}.decision abstention reason is invalid"
            )
        if decision["kind"] == "correct" and reason_code not in {
            "signal_misclassified",
            "alignment_misclassified",
            "knowledge_component_misaligned",
            "focus_dimension_misaligned",
        }:
            raise TeacherAgentAdjudicationStoreError(
                f"adjudication journal {field}.decision correction reason is invalid"
            )
        if decision.get("evidence_id") != original["evidence_id"]:
            raise TeacherAgentAdjudicationStoreError(
                f"adjudication journal {field}.decision evidence ID mismatches"
            )
        if decision.get("evidence_sha256") != original["evidence_sha256"]:
            raise TeacherAgentAdjudicationStoreError(
                f"adjudication journal {field}.decision evidence hash mismatches"
            )
        _utc_timestamp(decision.get("decided_at"), field=f"{field}.decision.decided_at")
        _digest(
            decision.get("idempotency_key_sha256"),
            field=f"{field}.decision.idempotency_key_sha256",
        )
        correction = decision.get("correction")
        if decision["kind"] == "correct":
            if not isinstance(correction, Mapping) or not correction:
                raise TeacherAgentAdjudicationStoreError(
                    f"adjudication journal {field}.decision correction is missing"
                )
            if not set(correction).issubset(
                {"signal", "answer_alignment", "focus_dimension", "target_kc_ids"}
            ):
                raise TeacherAgentAdjudicationStoreError(
                    f"adjudication journal {field}.decision correction is unbounded"
                )
            if "signal" in correction and correction["signal"] not in {
                "correct",
                "partial",
                "misconception",
            }:
                raise TeacherAgentAdjudicationStoreError(
                    f"adjudication journal {field}.decision correction signal is invalid"
                )
            if "answer_alignment" in correction and correction[
                "answer_alignment"
            ] not in {
                "aligned",
                "partially_aligned",
                "contradicted",
                "ambiguous",
            }:
                raise TeacherAgentAdjudicationStoreError(
                    f"adjudication journal {field}.decision correction alignment is invalid"
                )
            if "focus_dimension" in correction and correction[
                "focus_dimension"
            ] not in {
                "prerequisite",
                "conceptual",
                "procedural",
                "transfer",
            }:
                raise TeacherAgentAdjudicationStoreError(
                    f"adjudication journal {field}.decision correction dimension is invalid"
                )
            if "target_kc_ids" in correction:
                correction_targets = correction["target_kc_ids"]
                if (
                    not isinstance(correction_targets, list)
                    or not 1 <= len(correction_targets) <= 4
                    or any(not isinstance(target, str) for target in correction_targets)
                    or len(set(correction_targets)) != len(correction_targets)
                    or not set(correction_targets).issubset(set(target_kcs))
                ):
                    raise TeacherAgentAdjudicationStoreError(
                        f"adjudication journal {field}.decision correction targets are invalid"
                    )
        elif correction is not None:
            raise TeacherAgentAdjudicationStoreError(
                f"adjudication journal {field}.decision correction is inconsistent"
            )
        if item.get("cancellation") is not None:
            raise TeacherAgentAdjudicationStoreError(
                f"adjudication journal {field} decided state is inconsistent"
            )
    if status == "cancelled":
        cancellation = item.get("cancellation")
        if not isinstance(cancellation, Mapping) or frozenset(cancellation) != {
            "reason",
            "cancelled_at",
            "actor",
        }:
            raise TeacherAgentAdjudicationStoreError(
                f"adjudication journal {field}.cancellation is missing"
            )
        if cancellation.get("reason") != "evidence_deleted":
            raise TeacherAgentAdjudicationStoreError(
                f"adjudication journal {field}.cancellation reason is invalid"
            )
        cancellation_actor = _validate_actor(
            cancellation.get("actor"), field=f"{field}.cancellation.actor"
        )
        if cancellation_actor["identity"] != "system_evidence_purge":
            raise TeacherAgentAdjudicationStoreError(
                f"adjudication journal {field}.cancellation actor is invalid"
            )
        _utc_timestamp(
            cancellation.get("cancelled_at"),
            field=f"{field}.cancellation.cancelled_at",
        )
    return item


def _validate_audit(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or frozenset(value) != _AUDIT_KEYS:
        raise TeacherAgentAdjudicationStoreError(
            f"adjudication journal {field} audit shape is invalid"
        )
    receipt = deepcopy(dict(value))
    if receipt.get("schema") != ADJUDICATION_AUDIT_SCHEMA:
        raise TeacherAgentAdjudicationStoreError(
            f"adjudication journal {field} audit schema is unsupported"
        )
    _positive_integer(receipt.get("seq"), field=f"{field}.seq")
    if receipt.get("event_type") not in {
        "review.enqueued",
        "review.claimed",
        "review.approve",
        "review.correct",
        "review.abstain",
        "review.cancelled_evidence_deleted",
    }:
        raise TeacherAgentAdjudicationStoreError(
            f"adjudication journal {field}.event_type is invalid"
        )
    if (
        not isinstance(receipt.get("item_id"), str)
        or _ITEM_ID.fullmatch(receipt["item_id"]) is None
    ):
        raise TeacherAgentAdjudicationStoreError(
            f"adjudication journal {field}.item_id is invalid"
        )
    if receipt.get("from_version") is not None:
        _positive_integer(receipt["from_version"], field=f"{field}.from_version")
    _positive_integer(receipt.get("to_version"), field=f"{field}.to_version")
    for key in (
        "result_version_sha256",
        "request_sha256",
        "prior_receipt_sha256",
    ):
        _digest(
            receipt.get(key),
            field=f"{field}.{key}",
            nullable=key == "prior_receipt_sha256",
        )
    _validate_actor(receipt.get("actor"), field=f"{field}.actor")
    _utc_timestamp(receipt.get("occurred_at"), field=f"{field}.occurred_at")
    declared = _digest(receipt.get("receipt_sha256"), field=f"{field}.receipt_sha256")
    material = deepcopy(receipt)
    material.pop("receipt_sha256")
    if declared != canonical_sha256(material):
        raise TeacherAgentAdjudicationStoreError(
            f"adjudication journal {field} audit integrity check failed"
        )
    return receipt


def _validate_binding(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or frozenset(value) != _BINDING_KEYS:
        raise TeacherAgentAdjudicationStoreError(
            f"adjudication journal {field} idempotency binding shape is invalid"
        )
    binding = deepcopy(dict(value))
    if binding.get("operation") not in {"enqueue", "claim", "decide"}:
        raise TeacherAgentAdjudicationStoreError(
            f"adjudication journal {field}.operation is invalid"
        )
    key = _required_string(
        binding.get("idempotency_key"), field=f"{field}.idempotency_key", maximum=160
    )
    if _SAFE_ID.fullmatch(key) is None or len(key) < 8:
        raise TeacherAgentAdjudicationStoreError(
            f"adjudication journal {field}.idempotency_key is invalid"
        )
    if binding.get("idempotency_key_sha256") != sha256(key.encode("utf-8")).hexdigest():
        raise TeacherAgentAdjudicationStoreError(
            f"adjudication journal {field} idempotency key hash does not match"
        )
    _digest(binding.get("request_sha256"), field=f"{field}.request_sha256")
    response = binding.get("response")
    if not isinstance(response, Mapping):
        raise TeacherAgentAdjudicationStoreError(
            f"adjudication journal {field}.response must be an object"
        )
    if binding.get("response_sha256") != canonical_sha256(dict(response)):
        raise TeacherAgentAdjudicationStoreError(
            f"adjudication journal {field} response integrity check failed"
        )
    return binding


def _instruction_matches_decision(item: Mapping[str, Any], instruction: Any) -> bool:
    if not isinstance(instruction, Mapping):
        return False
    reduced = reduce_adjudication(item)
    if dict(instruction) == reduced:
        return True
    decision = item.get("decision")
    if (
        not isinstance(decision, Mapping)
        or decision.get("kind") != "correct"
        or not isinstance(decision.get("actor"), Mapping)
        or decision["actor"].get("identity") != AUTHENTICATED_TEACHER_ACTOR
    ):
        return False
    authority = instruction.get("authority")
    if not isinstance(authority, Mapping):
        return False
    try:
        receipt = validate_teacher_authority_revalidation_receipt(
            authority.get("server_revalidation_receipt")
        )
        model_reduced = deepcopy(reduced)
        model_reduced["evidence_sha256"] = receipt.get("model_evidence_sha256")
        model_reduced.pop("instruction_sha256", None)
        model_reduced["instruction_sha256"] = canonical_sha256(model_reduced)
        basis_hash = canonical_sha256(
            authenticated_instruction_authority_basis(model_reduced)
        )
        expected = authorize_authenticated_instruction(
            model_reduced, revalidation_receipt=receipt
        )
    except (TeacherAuthorityError, TeacherAgentAdjudicationError, TypeError):
        return False
    actor = decision["actor"]
    original = item.get("original")
    correction = decision.get("correction")
    authority_receipt = decision.get("authority_receipt")
    if not isinstance(original, Mapping) or not isinstance(authority_receipt, Mapping):
        return False
    return bool(
        dict(instruction) == expected
        and receipt.get("authority_id") == actor.get("authority_id")
        and receipt.get("gateway_verification_receipt_sha256")
        == authority_receipt.get("receipt_sha256")
        and receipt.get("actor_principal_sha256") == actor.get("principal_sha256")
        and receipt.get("roles_sha256") == actor.get("roles_sha256")
        and receipt.get("review_item_id") == item.get("item_id")
        and receipt.get("review_item_version_sha256") == item.get("version_sha256")
        and receipt.get("evidence_id") == original.get("evidence_id")
        and receipt.get("evidence_sha256") == original.get("evidence_sha256")
        and isinstance(receipt.get("model_evidence_sha256"), str)
        and receipt.get("instruction_authority_basis_sha256") == basis_hash
        and receipt.get("correction_sha256") == canonical_sha256(correction)
        and receipt.get("target_kc_ids") == item.get("target_kc_ids")
        and receipt.get("rubric_id") == original.get("rubric_id")
        and receipt.get("rubric_authority_sha256")
        == original.get("rubric_authority_sha256")
        and receipt.get("idempotency_key_sha256")
        == decision.get("idempotency_key_sha256")
    )


def _validate_index(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or frozenset(value) != _INDEX_KEYS:
        raise TeacherAgentAdjudicationStoreError(
            f"adjudication journal {field} evidence index shape is invalid"
        )
    update = deepcopy(dict(value))
    evidence_id = _required_string(
        update.get("evidence_id"), field=f"{field}.evidence_id", maximum=160
    )
    if _SAFE_ID.fullmatch(evidence_id) is None:
        raise TeacherAgentAdjudicationStoreError(
            f"adjudication journal {field}.evidence_id is invalid"
        )
    if update.get("evidence_id_sha256") != _evidence_id_digest(evidence_id):
        raise TeacherAgentAdjudicationStoreError(
            f"adjudication journal {field} evidence ID hash does not match"
        )
    _digest(update.get("evidence_sha256"), field=f"{field}.evidence_sha256")
    if (
        not isinstance(update.get("item_id"), str)
        or _ITEM_ID.fullmatch(update["item_id"]) is None
    ):
        raise TeacherAgentAdjudicationStoreError(
            f"adjudication journal {field}.item_id is invalid"
        )
    return update


def _validate_tombstone(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or frozenset(value) != _TOMBSTONE_KEYS:
        raise TeacherAgentAdjudicationStoreError(
            f"adjudication journal {field} erasure tombstone shape is invalid"
        )
    tombstone = deepcopy(dict(value))
    if tombstone.get("schema") != STORE_TOMBSTONE_SCHEMA:
        raise TeacherAgentAdjudicationStoreError(
            f"adjudication journal {field} erasure tombstone schema is unsupported"
        )
    _digest(
        tombstone.get("evidence_id_sha256"),
        field=f"{field}.evidence_id_sha256",
    )
    _utc_timestamp(tombstone.get("erased_at_utc"), field=f"{field}.erased_at_utc")
    if tombstone.get("reason") != "evidence_deleted":
        raise TeacherAgentAdjudicationStoreError(
            f"adjudication journal {field}.reason is invalid"
        )
    actor = _validate_actor(tombstone.get("actor"), field=f"{field}.actor")
    if actor["identity"] != "system_evidence_purge":
        raise TeacherAgentAdjudicationStoreError(
            f"adjudication journal {field}.actor is invalid"
        )
    return tombstone


def _validate_event(
    value: Any, *, expected_seq: int, previous_hash: str | None, line_number: int
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or frozenset(value) != _EVENT_KEYS:
        raise TeacherAgentAdjudicationStoreError(
            f"adjudication journal line {line_number} event shape is invalid"
        )
    event = deepcopy(dict(value))
    if event.get("schema") != STORE_EVENT_SCHEMA:
        raise TeacherAgentAdjudicationStoreError(
            f"adjudication journal line {line_number} schema is unsupported"
        )
    if event.get("seq") != expected_seq:
        raise TeacherAgentAdjudicationStoreError(
            f"adjudication journal line {line_number} sequence is non-contiguous"
        )
    if event.get("previous_hash") != previous_hash:
        raise TeacherAgentAdjudicationStoreError(
            f"adjudication journal line {line_number} breaks the hash chain"
        )
    _utc_timestamp(
        event.get("recorded_at_utc"), field=f"line {line_number}.recorded_at_utc"
    )
    if event.get("operation") not in STORE_OPERATIONS:
        raise TeacherAgentAdjudicationStoreError(
            f"adjudication journal line {line_number} operation is unsupported"
        )
    for key in (
        "item_versions",
        "audit_receipts",
        "idempotency_bindings",
        "evidence_index_updates",
        "erasure_tombstones",
    ):
        if not isinstance(event.get(key), list):
            raise TeacherAgentAdjudicationStoreError(
                f"adjudication journal line {line_number}.{key} must be an array"
            )
    declared = _digest(event.get("hash"), field=f"line {line_number}.hash")
    material = deepcopy(event)
    material.pop("hash")
    if declared != canonical_sha256(material):
        raise TeacherAgentAdjudicationStoreError(
            f"adjudication journal line {line_number} failed integrity validation"
        )
    return event


def _empty_recovery() -> TeacherAgentAdjudicationRecovery:
    return TeacherAgentAdjudicationRecovery(
        versions={},
        audit_receipts=[],
        idempotency_bindings={},
        evidence_items={},
        erasure_tombstones={},
        events=(),
        file_size=0,
        last_hash=None,
    )


def _replay(
    raw_events: Sequence[Mapping[str, Any]], *, file_size: int
) -> TeacherAgentAdjudicationRecovery:
    versions: dict[str, list[dict[str, Any]]] = {}
    audit: list[dict[str, Any]] = []
    idempotency: dict[tuple[str, str], dict[str, Any]] = {}
    evidence_items: dict[str, str] = {}
    tombstones: dict[str, dict[str, Any]] = {}
    events: list[dict[str, Any]] = []
    previous_hash: str | None = None

    for line_number, raw in enumerate(raw_events, start=1):
        event = _validate_event(
            raw,
            expected_seq=line_number,
            previous_hash=previous_hash,
            line_number=line_number,
        )
        operation = str(event["operation"])
        new_versions = [
            _validate_item(value, field=f"line {line_number}.item_versions[{index}]")
            for index, value in enumerate(event["item_versions"])
        ]
        new_receipts = [
            _validate_audit(value, field=f"line {line_number}.audit_receipts[{index}]")
            for index, value in enumerate(event["audit_receipts"])
        ]
        new_bindings = [
            _validate_binding(
                value, field=f"line {line_number}.idempotency_bindings[{index}]"
            )
            for index, value in enumerate(event["idempotency_bindings"])
        ]
        new_indexes = [
            _validate_index(
                value, field=f"line {line_number}.evidence_index_updates[{index}]"
            )
            for index, value in enumerate(event["evidence_index_updates"])
        ]
        new_tombstones = [
            _validate_tombstone(
                value, field=f"line {line_number}.erasure_tombstones[{index}]"
            )
            for index, value in enumerate(event["erasure_tombstones"])
        ]

        if len(new_versions) != len(new_receipts):
            raise TeacherAgentAdjudicationStoreError(
                f"adjudication journal line {line_number} must bind each item version "
                "to exactly one audit receipt"
            )
        version_by_coordinate: dict[tuple[str, int], dict[str, Any]] = {}
        prior_by_coordinate: dict[tuple[str, int], dict[str, Any] | None] = {}
        for item in new_versions:
            item_id = str(item["item_id"])
            prior_versions = versions.get(item_id, [])
            expected_version = len(prior_versions) + 1
            if item["version"] != expected_version:
                raise TeacherAgentAdjudicationStoreError(
                    f"adjudication journal line {line_number} item version is non-contiguous"
                )
            expected_previous = (
                prior_versions[-1]["version_sha256"] if prior_versions else None
            )
            if item["previous_version_sha256"] != expected_previous:
                raise TeacherAgentAdjudicationStoreError(
                    f"adjudication journal line {line_number} item version chain breaks"
                )
            coordinate = (item_id, expected_version)
            if coordinate in version_by_coordinate:
                raise TeacherAgentAdjudicationStoreError(
                    f"adjudication journal line {line_number} duplicates an item version"
                )
            version_by_coordinate[coordinate] = item
            prior_by_coordinate[coordinate] = (
                deepcopy(prior_versions[-1]) if prior_versions else None
            )
            versions.setdefault(item_id, []).append(item)

        for receipt in new_receipts:
            expected_audit_seq = len(audit) + 1
            if receipt["seq"] != expected_audit_seq:
                raise TeacherAgentAdjudicationStoreError(
                    f"adjudication journal line {line_number} audit sequence is non-contiguous"
                )
            expected_prior = audit[-1]["receipt_sha256"] if audit else None
            if receipt["prior_receipt_sha256"] != expected_prior:
                raise TeacherAgentAdjudicationStoreError(
                    f"adjudication journal line {line_number} audit chain breaks"
                )
            coordinate = (str(receipt["item_id"]), int(receipt["to_version"]))
            item = version_by_coordinate.get(coordinate)
            if item is None:
                raise TeacherAgentAdjudicationStoreError(
                    f"adjudication journal line {line_number} audit has no item version"
                )
            if receipt["result_version_sha256"] != item["version_sha256"]:
                raise TeacherAgentAdjudicationStoreError(
                    f"adjudication journal line {line_number} audit result hash mismatches"
                )
            expected_from = int(item["version"]) - 1 or None
            if receipt["from_version"] != expected_from:
                raise TeacherAgentAdjudicationStoreError(
                    f"adjudication journal line {line_number} audit transition is invalid"
                )
            if receipt["event_type"] == "review.enqueued":
                expected_actor = {
                    "identity": "server_evidence_registry",
                    "authenticated": False,
                    "teacher_identity_claimed": False,
                }
            elif receipt["event_type"] == "review.cancelled_evidence_deleted":
                expected_actor = {
                    "identity": "system_evidence_purge",
                    "authenticated": False,
                    "teacher_identity_claimed": False,
                }
            elif item["status"] == "claimed":
                expected_actor = item["lease"]["actor"]
            elif item["status"] == "decided":
                expected_actor = item["decision"]["actor"]
            else:
                expected_actor = {
                    "identity": LOCAL_OPERATOR_IDENTITY,
                    "authenticated": False,
                    "teacher_identity_claimed": False,
                }
            if receipt["actor"] != expected_actor:
                raise TeacherAgentAdjudicationStoreError(
                    f"adjudication journal line {line_number} audit actor is invalid"
                )
            prior_item = prior_by_coordinate[coordinate]
            if prior_item is None:
                if (
                    item["version"] != 1
                    or item["status"] != "pending"
                    or receipt["event_type"] != "review.enqueued"
                ):
                    raise TeacherAgentAdjudicationStoreError(
                        f"adjudication journal line {line_number} initial review state is invalid"
                    )
            else:
                immutable = {
                    "schema",
                    "item_id",
                    "created_at",
                    "source",
                    "target_kc_ids",
                    "target_scope",
                    "original",
                    "before_snapshot",
                    "after_snapshot",
                    "review_reason",
                }
                if any(item[key] != prior_item[key] for key in immutable):
                    raise TeacherAgentAdjudicationStoreError(
                        f"adjudication journal line {line_number} rewrites immutable review content"
                    )
                if prior_item["status"] == "cancelled":
                    raise TeacherAgentAdjudicationStoreError(
                        f"adjudication journal line {line_number} follows a cancelled item"
                    )
                if item["status"] == "claimed":
                    valid_transition = (
                        prior_item["status"] in {"pending", "claimed"}
                        and receipt["event_type"] == "review.claimed"
                        and item["decision"] == prior_item["decision"]
                        and item["cancellation"] == prior_item["cancellation"]
                    )
                elif item["status"] == "decided":
                    valid_transition = (
                        prior_item["status"] == "claimed"
                        and receipt["event_type"]
                        == f"review.{item['decision']['kind']}"
                        and item["cancellation"] is None
                        and isinstance(prior_item.get("lease"), Mapping)
                        and _same_authenticated_principal(
                            prior_item["lease"]["actor"],
                            item["decision"]["actor"],
                        )
                    )
                elif item["status"] == "cancelled":
                    valid_transition = (
                        receipt["event_type"] == "review.cancelled_evidence_deleted"
                    )
                else:
                    valid_transition = False
                if not valid_transition:
                    raise TeacherAgentAdjudicationStoreError(
                        f"adjudication journal line {line_number} item state transition is invalid"
                    )
            audit.append(receipt)

        for update in new_indexes:
            evidence_id = str(update["evidence_id"])
            if evidence_id in evidence_items:
                raise TeacherAgentAdjudicationStoreError(
                    f"adjudication journal line {line_number} evidence index is duplicated"
                )
            item_versions = versions.get(str(update["item_id"]))
            if not item_versions:
                raise TeacherAgentAdjudicationStoreError(
                    f"adjudication journal line {line_number} index item does not exist"
                )
            original = item_versions[0]["original"]
            if (
                original["evidence_id"] != evidence_id
                or original["evidence_sha256"] != update["evidence_sha256"]
            ):
                raise TeacherAgentAdjudicationStoreError(
                    f"adjudication journal line {line_number} index binding mismatches item"
                )
            evidence_items[evidence_id] = str(update["item_id"])

        for binding in new_bindings:
            binding_key = (
                str(binding["operation"]),
                str(binding["idempotency_key"]),
            )
            if binding_key in idempotency:
                raise TeacherAgentAdjudicationStoreError(
                    f"adjudication journal line {line_number} idempotency binding is duplicated"
                )
            if binding["operation"] != operation:
                raise TeacherAgentAdjudicationStoreError(
                    f"adjudication journal line {line_number} idempotency operation mismatches"
                )
            if (
                len(new_receipts) != 1
                or binding["request_sha256"] != new_receipts[0]["request_sha256"]
            ):
                raise TeacherAgentAdjudicationStoreError(
                    f"adjudication journal line {line_number} request binding mismatches audit"
                )
            response = dict(binding["response"])
            if operation == "enqueue":
                if len(new_versions) != 1 or response != new_versions[0]:
                    raise TeacherAgentAdjudicationStoreError(
                        f"adjudication journal line {line_number} enqueue response mismatches"
                    )
            elif operation == "claim":
                token = response.get("claim_token")
                response_item = response.get("item")
                if (
                    len(new_versions) != 1
                    or response_item != new_versions[0]
                    or not isinstance(token, str)
                    or sha256(token.encode("utf-8")).hexdigest()
                    != new_versions[0].get("lease", {}).get("claim_token_sha256")
                ):
                    raise TeacherAgentAdjudicationStoreError(
                        f"adjudication journal line {line_number} claim response mismatches"
                    )
            elif operation == "decide":
                response_item = response.get("item")
                instruction = response.get("instruction")
                if (
                    len(new_versions) != 1
                    or response_item != new_versions[0]
                    or not isinstance(instruction, Mapping)
                    or instruction.get("schema") != ADJUDICATION_INSTRUCTION_SCHEMA
                    or not _instruction_matches_decision(new_versions[0], instruction)
                ):
                    raise TeacherAgentAdjudicationStoreError(
                        f"adjudication journal line {line_number} decision response mismatches"
                    )
                if (
                    new_versions[0]["decision"]["idempotency_key_sha256"]
                    != binding["idempotency_key_sha256"]
                ):
                    raise TeacherAgentAdjudicationStoreError(
                        f"adjudication journal line {line_number} decision idempotency hash mismatches"
                    )
            idempotency[binding_key] = {
                "request_sha256": binding["request_sha256"],
                "response": deepcopy(response),
            }

        for tombstone in new_tombstones:
            digest = str(tombstone["evidence_id_sha256"])
            if digest in tombstones:
                raise TeacherAgentAdjudicationStoreError(
                    f"adjudication journal line {line_number} erasure tombstone is duplicated"
                )
            matching_id = next(
                (
                    evidence_id
                    for evidence_id in evidence_items
                    if _evidence_id_digest(evidence_id) == digest
                ),
                None,
            )
            if matching_id is not None:
                current = versions[evidence_items[matching_id]][-1]
                if current["status"] != "cancelled":
                    raise TeacherAgentAdjudicationStoreError(
                        f"adjudication journal line {line_number} erases an active review"
                    )
            tombstones[digest] = tombstone

        if operation == "enqueue":
            valid_enqueue = (
                len(new_versions)
                == len(new_receipts)
                == len(new_bindings)
                == len(new_indexes)
                == 1
                and not new_tombstones
                and new_versions[0]["version"] == 1
                and new_versions[0]["status"] == "pending"
            )
            valid_erasure_discovery = (
                not new_bindings
                and not new_indexes
                and len(new_tombstones) == 1
                and (
                    (not new_versions and not new_receipts)
                    or (
                        len(new_versions) == len(new_receipts) == 1
                        and new_versions[0]["status"] == "cancelled"
                    )
                )
            )
            if not (valid_enqueue or valid_erasure_discovery):
                raise TeacherAgentAdjudicationStoreError(
                    f"adjudication journal line {line_number} enqueue transaction is invalid"
                )
        elif operation == "claim":
            if not (
                len(new_versions) == len(new_receipts) == len(new_bindings) == 1
                and not new_indexes
                and not new_tombstones
                and new_versions[0]["status"] == "claimed"
            ):
                raise TeacherAgentAdjudicationStoreError(
                    f"adjudication journal line {line_number} claim transaction is invalid"
                )
        elif operation == "decide":
            valid_decision = (
                len(new_versions) == len(new_receipts) == 1
                and not new_indexes
                and (
                    (
                        len(new_bindings) == 1
                        and not new_tombstones
                        and new_versions[0]["status"] == "decided"
                    )
                    or (
                        not new_bindings
                        and len(new_tombstones) == 1
                        and new_versions[0]["status"] == "cancelled"
                    )
                )
            )
            if not valid_decision:
                raise TeacherAgentAdjudicationStoreError(
                    f"adjudication journal line {line_number} decision transaction is invalid"
                )
        elif operation == "cancel_deleted_evidence":
            if not (
                not new_bindings
                and not new_indexes
                and len(new_tombstones) == 1
                and (
                    (not new_versions and not new_receipts)
                    or (
                        len(new_versions) == len(new_receipts) == 1
                        and new_versions[0]["status"] == "cancelled"
                    )
                )
            ):
                raise TeacherAgentAdjudicationStoreError(
                    f"adjudication journal line {line_number} erasure transaction is invalid"
                )

        events.append(event)
        previous_hash = str(event["hash"])

    return TeacherAgentAdjudicationRecovery(
        versions=versions,
        audit_receipts=audit,
        idempotency_bindings=idempotency,
        evidence_items=evidence_items,
        erasure_tombstones=tombstones,
        events=tuple(events),
        file_size=file_size,
        last_hash=previous_hash,
    )


class TeacherAgentAdjudicationStore:
    """Secure canonical journal used by the durable queue facade."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        self._poisoned = False
        # Construction validates ownership, permissions, hash chains, and state
        # transitions and repairs only a genuinely incomplete final line.
        self.recover()

    @staticmethod
    def _require_flock() -> None:
        if fcntl is None:
            raise TeacherAgentAdjudicationStoreError(
                "durable adjudication requires POSIX flock"
            )

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _open_secure(self) -> tuple[int, bool]:
        self._require_flock()
        if not hasattr(os, "O_NOFOLLOW"):
            raise TeacherAgentAdjudicationStoreError(
                "durable adjudication requires no-follow file opens"
            )
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise TeacherAgentAdjudicationStoreError(
                "adjudication store directory cannot be created"
            ) from exc
        flags = os.O_RDWR
        flags |= os.O_NOFOLLOW
        created = False
        try:
            descriptor = os.open(self.path, flags | os.O_CREAT | os.O_EXCL, 0o600)
            created = True
        except FileExistsError:
            try:
                descriptor = os.open(self.path, flags)
            except OSError as exc:
                raise TeacherAgentAdjudicationStoreError(
                    "adjudication store cannot be opened safely"
                ) from exc
        except OSError as exc:
            raise TeacherAgentAdjudicationStoreError(
                "adjudication store cannot be created safely"
            ) from exc
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise TeacherAgentAdjudicationStoreError(
                    "adjudication store must be a regular file"
                )
            if hasattr(os, "getuid") and metadata.st_uid != os.getuid():
                raise TeacherAgentAdjudicationStoreError(
                    "adjudication store must be owned by the current user"
                )
            if created:
                os.fchmod(descriptor, 0o600)
                os.fsync(descriptor)
                self._fsync_directory(self.path.parent)
            elif stat.S_IMODE(metadata.st_mode) != 0o600:
                raise TeacherAgentAdjudicationStoreError(
                    "adjudication store permissions must be exactly 0600"
                )
            return descriptor, created
        except Exception:
            os.close(descriptor)
            raise

    @staticmethod
    def _flock(descriptor: int) -> None:
        try:
            assert fcntl is not None
            fcntl.flock(descriptor, fcntl.LOCK_EX)
        except OSError as exc:
            raise TeacherAgentAdjudicationStoreError(
                "adjudication store process lock cannot be acquired"
            ) from exc

    @staticmethod
    def _unlock(descriptor: int) -> None:
        if fcntl is None:
            return
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        except OSError:
            pass

    @staticmethod
    def _read_all(descriptor: int) -> bytes:
        os.lseek(descriptor, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)

    def _recover_locked(
        self, descriptor: int, *, repair_torn_tail: bool
    ) -> TeacherAgentAdjudicationRecovery:
        try:
            raw = self._read_all(descriptor)
        except OSError as exc:
            raise TeacherAgentAdjudicationStoreError(
                "adjudication journal cannot be read"
            ) from exc
        final_newline = raw.rfind(b"\n")
        valid_size = final_newline + 1 if final_newline >= 0 else 0
        if valid_size != len(raw):
            if not repair_torn_tail:
                raise TeacherAgentAdjudicationStoreError(
                    "adjudication journal has an incomplete final record"
                )
            try:
                os.ftruncate(descriptor, valid_size)
                os.fsync(descriptor)
            except OSError as exc:
                raise TeacherAgentAdjudicationStoreError(
                    "adjudication journal torn tail cannot be repaired"
                ) from exc
        complete = raw[:valid_size]
        parsed: list[dict[str, Any]] = []
        for line_number, raw_line in enumerate(complete.splitlines(), start=1):
            if not raw_line:
                raise TeacherAgentAdjudicationStoreError(
                    f"adjudication journal line {line_number} is empty"
                )
            try:
                value = json.loads(raw_line)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise TeacherAgentAdjudicationStoreError(
                    f"adjudication journal line {line_number} is invalid JSON"
                ) from exc
            if not isinstance(value, dict):
                raise TeacherAgentAdjudicationStoreError(
                    f"adjudication journal line {line_number} is not an object"
                )
            parsed.append(value)
        return _replay(parsed, file_size=valid_size)

    def recover(self) -> TeacherAgentAdjudicationRecovery:
        """Strictly replay current state, repairing only a torn final line."""

        with self._lock:
            if self._poisoned:
                raise TeacherAgentAdjudicationStorePoisonedError(
                    "adjudication store is poisoned after a failed rollback fence"
                )
            descriptor, _created = self._open_secure()
            try:
                self._flock(descriptor)
                try:
                    return self._recover_locked(descriptor, repair_torn_tail=True)
                finally:
                    self._unlock(descriptor)
            finally:
                os.close(descriptor)

    @property
    def events(self) -> tuple[dict[str, Any], ...]:
        return tuple(deepcopy(self.recover().events))

    @staticmethod
    def _hydrate_core(
        core: TeacherAgentAdjudicationQueue,
        recovery: TeacherAgentAdjudicationRecovery,
    ) -> None:
        # This module is the persistence adapter for the v1 reducer.  Hydration
        # is centralized here so private reducer state never leaks to callers.
        core._versions = deepcopy(recovery.versions)  # noqa: SLF001
        core._audit = deepcopy(recovery.audit_receipts)  # noqa: SLF001
        core._idempotency = deepcopy(recovery.idempotency_bindings)  # noqa: SLF001
        core._evidence_items = deepcopy(recovery.evidence_items)  # noqa: SLF001

    def read(
        self,
        *,
        core_factory: Callable[[], TeacherAgentAdjudicationQueue],
        reader: Callable[[TeacherAgentAdjudicationQueue], _Result],
    ) -> _Result:
        """Run a read against a freshly replayed authoritative snapshot."""

        with self._lock:
            descriptor, _created = self._open_secure()
            try:
                self._flock(descriptor)
                try:
                    recovery = self._recover_locked(descriptor, repair_torn_tail=True)
                    core = core_factory()
                    self._hydrate_core(core, recovery)
                    return deepcopy(reader(core))
                finally:
                    self._unlock(descriptor)
            finally:
                os.close(descriptor)

    @staticmethod
    def _state_delta(
        core: TeacherAgentAdjudicationQueue,
        recovery: TeacherAgentAdjudicationRecovery,
    ) -> tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
    ]:
        new_versions: list[dict[str, Any]] = []
        for item_id, item_versions in core._versions.items():  # noqa: SLF001
            prior = recovery.versions.get(item_id, [])
            if item_versions[: len(prior)] != prior:
                raise TeacherAgentAdjudicationStoreError(
                    "adjudication reducer attempted to rewrite item history"
                )
            new_versions.extend(deepcopy(item_versions[len(prior) :]))
        if set(recovery.versions) - set(core._versions):  # noqa: SLF001
            raise TeacherAgentAdjudicationStoreError(
                "adjudication reducer attempted to delete item history"
            )

        if core._audit[: len(recovery.audit_receipts)] != recovery.audit_receipts:  # noqa: SLF001
            raise TeacherAgentAdjudicationStoreError(
                "adjudication reducer attempted to rewrite audit history"
            )
        new_audit = deepcopy(
            core._audit[len(recovery.audit_receipts) :]  # noqa: SLF001
        )

        new_bindings: list[dict[str, Any]] = []
        for binding_key, value in core._idempotency.items():  # noqa: SLF001
            if binding_key in recovery.idempotency_bindings:
                if value != recovery.idempotency_bindings[binding_key]:
                    raise TeacherAgentAdjudicationStoreError(
                        "adjudication reducer attempted to rewrite idempotency history"
                    )
                continue
            operation, key = binding_key
            response = deepcopy(value["response"])
            new_bindings.append(
                {
                    "operation": operation,
                    "idempotency_key": key,
                    "idempotency_key_sha256": sha256(key.encode("utf-8")).hexdigest(),
                    "request_sha256": value["request_sha256"],
                    "response": response,
                    "response_sha256": canonical_sha256(response),
                }
            )

        new_indexes: list[dict[str, Any]] = []
        for evidence_id, item_id in core._evidence_items.items():  # noqa: SLF001
            if evidence_id in recovery.evidence_items:
                if item_id != recovery.evidence_items[evidence_id]:
                    raise TeacherAgentAdjudicationStoreError(
                        "adjudication reducer attempted to rewrite evidence index"
                    )
                continue
            item = core._versions[item_id][0]  # noqa: SLF001
            new_indexes.append(
                {
                    "evidence_id": evidence_id,
                    "evidence_id_sha256": _evidence_id_digest(evidence_id),
                    "evidence_sha256": item["original"]["evidence_sha256"],
                    "item_id": item_id,
                }
            )
        return new_versions, new_audit, new_bindings, new_indexes

    @staticmethod
    def _write_all(descriptor: int, payload: bytes) -> None:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("adjudication journal write made no progress")
            view = view[written:]

    def _append_locked(
        self,
        descriptor: int,
        *,
        recovery: TeacherAgentAdjudicationRecovery,
        operation: str,
        item_versions: Sequence[Mapping[str, Any]],
        audit_receipts: Sequence[Mapping[str, Any]],
        idempotency_bindings: Sequence[Mapping[str, Any]],
        evidence_index_updates: Sequence[Mapping[str, Any]],
        erasure_tombstones: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        event: dict[str, Any] = {
            "schema": STORE_EVENT_SCHEMA,
            "seq": len(recovery.events) + 1,
            "recorded_at_utc": _now_utc(),
            "operation": operation,
            "previous_hash": recovery.last_hash,
            "item_versions": deepcopy(list(item_versions)),
            "audit_receipts": deepcopy(list(audit_receipts)),
            "idempotency_bindings": deepcopy(list(idempotency_bindings)),
            "evidence_index_updates": deepcopy(list(evidence_index_updates)),
            "erasure_tombstones": deepcopy(list(erasure_tombstones)),
        }
        event["hash"] = canonical_sha256(event)
        # Validate the exact prospective chain and state transition before bytes
        # cross the append boundary.
        _replay([*recovery.events, event], file_size=recovery.file_size)
        payload = _canonical_bytes(event) + b"\n"
        original_size = recovery.file_size
        try:
            os.lseek(descriptor, original_size, os.SEEK_SET)
            self._write_all(descriptor, payload)
            os.fsync(descriptor)
        except OSError as exc:
            try:
                os.ftruncate(descriptor, original_size)
                os.fsync(descriptor)
            except OSError as rollback_exc:
                self._poisoned = True
                raise TeacherAgentAdjudicationStorePoisonedError(
                    "adjudication append and rollback fence both failed"
                ) from rollback_exc
            raise TeacherAgentAdjudicationStoreError(
                "adjudication transaction did not cross its fsync barrier"
            ) from exc
        return event

    def transact(
        self,
        *,
        operation: str,
        core_factory: Callable[[], TeacherAgentAdjudicationQueue],
        mutator: Callable[
            [TeacherAgentAdjudicationQueue, TeacherAgentAdjudicationRecovery],
            _Result,
        ],
        erasure_evidence_id: str | None = None,
        tombstone_on_deleted_error: bool = False,
        tombstone_requires_known_evidence: bool = False,
        after_commit: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> _Result:
        """Evaluate and durably append one reducer transaction under flock."""

        if operation not in STORE_OPERATIONS:
            raise TeacherAgentAdjudicationStoreError(
                "adjudication store operation is unsupported"
            )
        committed_event: dict[str, Any] | None = None
        response: _Result | None = None
        pending_error: BaseException | None = None
        with self._lock:
            if self._poisoned:
                raise TeacherAgentAdjudicationStorePoisonedError(
                    "adjudication store is poisoned after a failed rollback fence"
                )
            descriptor, _created = self._open_secure()
            try:
                self._flock(descriptor)
                try:
                    recovery = self._recover_locked(descriptor, repair_torn_tail=True)
                    core = core_factory()
                    self._hydrate_core(core, recovery)
                    try:
                        response = mutator(core, recovery)
                    except BaseException as exc:  # persist reducer cancellation first.
                        pending_error = exc
                    (
                        new_versions,
                        new_audit,
                        new_bindings,
                        new_indexes,
                    ) = self._state_delta(core, recovery)

                    new_tombstones: list[dict[str, Any]] = []
                    should_tombstone = erasure_evidence_id is not None and (
                        (
                            operation == "cancel_deleted_evidence"
                            and pending_error is None
                        )
                        or (
                            tombstone_on_deleted_error
                            and isinstance(
                                pending_error, AdjudicationEvidenceDeletedError
                            )
                            and (
                                not tombstone_requires_known_evidence
                                or erasure_evidence_id in recovery.evidence_items
                            )
                        )
                    )
                    if should_tombstone:
                        assert erasure_evidence_id is not None
                        evidence_digest = _evidence_id_digest(erasure_evidence_id)
                        if evidence_digest not in recovery.erasure_tombstones:
                            new_tombstones.append(
                                {
                                    "schema": STORE_TOMBSTONE_SCHEMA,
                                    "evidence_id_sha256": evidence_digest,
                                    "erased_at_utc": _now_utc(),
                                    "reason": "evidence_deleted",
                                    "actor": {
                                        "identity": "system_evidence_purge",
                                        "authenticated": False,
                                        "teacher_identity_claimed": False,
                                    },
                                }
                            )
                    has_delta = any(
                        (
                            new_versions,
                            new_audit,
                            new_bindings,
                            new_indexes,
                            new_tombstones,
                        )
                    )
                    if has_delta:
                        committed_event = self._append_locked(
                            descriptor,
                            recovery=recovery,
                            operation=operation,
                            item_versions=new_versions,
                            audit_receipts=new_audit,
                            idempotency_bindings=new_bindings,
                            evidence_index_updates=new_indexes,
                            erasure_tombstones=new_tombstones,
                        )
                finally:
                    self._unlock(descriptor)
            finally:
                os.close(descriptor)
        if committed_event is not None and after_commit is not None:
            after_commit(deepcopy(committed_event))
        if pending_error is not None:
            raise pending_error
        return deepcopy(response)  # type: ignore[return-value]


class DurableTeacherAgentAdjudicationQueue:
    """Store-backed facade with the same public operations as the v1 reducer."""

    def __init__(
        self,
        path: str | Path,
        *,
        evidence_resolver: Callable[[str], Mapping[str, Any] | None],
        clock: Callable[[], datetime] | None = None,
        token_factory: Callable[[], str] | None = None,
        after_commit: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> None:
        if not callable(evidence_resolver):
            raise TeacherAgentAdjudicationError("evidence_resolver must be callable")
        self.store = TeacherAgentAdjudicationStore(path)
        self._evidence_resolver = evidence_resolver
        self._clock = clock
        self._token_factory = token_factory
        self._after_commit = after_commit

    def _core(self) -> TeacherAgentAdjudicationQueue:
        return TeacherAgentAdjudicationQueue(
            evidence_resolver=self._evidence_resolver,
            clock=self._clock,
            token_factory=self._token_factory,
        )

    def enqueue(
        self,
        *,
        evidence_id: str,
        evidence_sha256: str,
        idempotency_key: str,
        review_reason: str = "automatic_low_confidence",
    ) -> dict[str, Any]:
        evidence_digest = _evidence_id_digest(evidence_id)

        def mutate(
            core: TeacherAgentAdjudicationQueue,
            recovery: TeacherAgentAdjudicationRecovery,
        ) -> dict[str, Any]:
            if evidence_digest in recovery.erasure_tombstones:
                raise AdjudicationEvidenceDeletedError(
                    "authoritative evidence has a durable erasure tombstone"
                )
            # Revalidate even an idempotent restart retry.  If an external
            # erasure happened without its callback reaching this process, an
            # old enqueue request must not resurrect the sealed queue item.
            try:
                core._resolve(  # noqa: SLF001 - persistence adapter boundary.
                    evidence_id=evidence_id,
                    evidence_sha256=evidence_sha256,
                )
            except AdjudicationEvidenceDeletedError:
                known_item_id = recovery.evidence_items.get(evidence_id)
                if known_item_id is not None:
                    item = core._latest(known_item_id)  # noqa: SLF001
                    _now, occurred_at = core._now()  # noqa: SLF001
                    core._cancel_locked(  # noqa: SLF001
                        item=item,
                        occurred_at=occurred_at,
                        request_sha256=canonical_sha256(
                            {
                                "evidence_id": evidence_id,
                                "reason": "evidence_deleted",
                            }
                        ),
                    )
                raise
            return core.enqueue(
                evidence_id=evidence_id,
                evidence_sha256=evidence_sha256,
                idempotency_key=idempotency_key,
                review_reason=review_reason,
            )

        return self.store.transact(
            operation="enqueue",
            core_factory=self._core,
            mutator=mutate,
            erasure_evidence_id=evidence_id,
            tombstone_on_deleted_error=True,
            tombstone_requires_known_evidence=True,
            after_commit=self._after_commit,
        )

    def get(self, item_id: str) -> dict[str, Any]:
        return self.store.read(
            core_factory=self._core, reader=lambda core: core.get(item_id)
        )

    def history(self, item_id: str) -> tuple[dict[str, Any], ...]:
        return self.store.read(
            core_factory=self._core, reader=lambda core: core.history(item_id)
        )

    def list_items(self) -> tuple[dict[str, Any], ...]:
        return self.store.read(
            core_factory=self._core, reader=lambda core: core.list_items()
        )

    def committed_decision_response(self, item_id: str) -> dict[str, Any]:
        """Return the single sealed decision response for crash recovery."""

        def read(core: TeacherAgentAdjudicationQueue) -> dict[str, Any]:
            current = core.get(item_id)
            matches = [
                deepcopy(binding["response"])
                for (operation, _key), binding in core._idempotency.items()  # noqa: SLF001
                if operation == "decide"
                and isinstance(binding.get("response"), Mapping)
                and binding["response"].get("item") == current
            ]
            if len(matches) != 1:
                raise TeacherAgentAdjudicationStoreError(
                    "decided adjudication has no unique sealed response"
                )
            return matches[0]

        return self.store.read(core_factory=self._core, reader=read)

    @property
    def audit_receipts(self) -> tuple[dict[str, Any], ...]:
        return self.store.read(
            core_factory=self._core, reader=lambda core: core.audit_receipts
        )

    def claim(
        self,
        item_id: str,
        *,
        expected_version: int,
        idempotency_key: str,
        authority_receipt: Mapping[str, Any] | None = None,
        authority_validator: (
            Callable[[Mapping[str, Any]], Mapping[str, Any]] | None
        ) = None,
    ) -> dict[str, Any]:
        return self.store.transact(
            operation="claim",
            core_factory=self._core,
            mutator=lambda core, _recovery: core.claim(
                item_id,
                expected_version=expected_version,
                idempotency_key=idempotency_key,
                authority_receipt=authority_receipt,
                authority_validator=authority_validator,
            ),
            after_commit=self._after_commit,
        )

    def decide(
        self,
        item_id: str,
        *,
        expected_version: int,
        claim_token: str,
        evidence_id: str,
        evidence_sha256: str,
        idempotency_key: str,
        decision: str,
        reason_code: str,
        correction: Mapping[str, Any] | None = None,
        authority_receipt: Mapping[str, Any] | None = None,
        authority_validator: (
            Callable[[Mapping[str, Any]], Mapping[str, Any]] | None
        ) = None,
        instruction_authorizer: (
            Callable[
                [Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]],
                Mapping[str, Any],
            ]
            | None
        ) = None,
    ) -> dict[str, Any]:
        evidence_digest = _evidence_id_digest(evidence_id)

        def mutate(
            core: TeacherAgentAdjudicationQueue,
            recovery: TeacherAgentAdjudicationRecovery,
        ) -> dict[str, Any]:
            if evidence_digest in recovery.erasure_tombstones:
                raise AdjudicationEvidenceDeletedError(
                    "authoritative evidence has a durable erasure tombstone"
                )
            # Revalidate the authoritative record even for an idempotent retry
            # after process restart.  The in-memory reducer intentionally
            # replays idempotency first; the durable boundary is stricter so a
            # later evidence deletion supersedes an old cached response.
            try:
                core._resolve(  # noqa: SLF001 - persistence adapter boundary.
                    evidence_id=evidence_id,
                    evidence_sha256=evidence_sha256,
                )
            except AdjudicationEvidenceDeletedError:
                item = core._latest(item_id)  # noqa: SLF001
                _now, occurred_at = core._now()  # noqa: SLF001
                core._cancel_locked(  # noqa: SLF001
                    item=item,
                    occurred_at=occurred_at,
                    request_sha256=canonical_sha256(
                        {"evidence_id": evidence_id, "reason": "evidence_deleted"}
                    ),
                )
                raise
            return core.decide(
                item_id,
                expected_version=expected_version,
                claim_token=claim_token,
                evidence_id=evidence_id,
                evidence_sha256=evidence_sha256,
                idempotency_key=idempotency_key,
                decision=decision,
                reason_code=reason_code,
                correction=correction,
                authority_receipt=authority_receipt,
                authority_validator=authority_validator,
                instruction_authorizer=instruction_authorizer,
            )

        return self.store.transact(
            operation="decide",
            core_factory=self._core,
            mutator=mutate,
            erasure_evidence_id=evidence_id,
            tombstone_on_deleted_error=True,
            after_commit=self._after_commit,
        )

    def cancel_deleted_evidence(self, evidence_id: str) -> tuple[dict[str, Any], ...]:
        evidence_digest = _evidence_id_digest(evidence_id)

        def mutate(
            core: TeacherAgentAdjudicationQueue,
            recovery: TeacherAgentAdjudicationRecovery,
        ) -> tuple[dict[str, Any], ...]:
            if evidence_digest in recovery.erasure_tombstones:
                item_id = recovery.evidence_items.get(evidence_id)
                if item_id is None:
                    return ()
                current = core.get(item_id)
                if current["status"] != "cancelled":
                    raise TeacherAgentAdjudicationStoreError(
                        "erasure tombstone points to a non-cancelled review"
                    )
                return (current,)
            return core.cancel_deleted_evidence(evidence_id)

        return self.store.transact(
            operation="cancel_deleted_evidence",
            core_factory=self._core,
            mutator=mutate,
            erasure_evidence_id=evidence_id,
            after_commit=self._after_commit,
        )


DurableAdjudicationQueue = DurableTeacherAgentAdjudicationQueue


__all__ = [
    "STORE_EVENT_SCHEMA",
    "STORE_TOMBSTONE_SCHEMA",
    "DurableAdjudicationQueue",
    "DurableTeacherAgentAdjudicationQueue",
    "TeacherAgentAdjudicationRecovery",
    "TeacherAgentAdjudicationStore",
    "TeacherAgentAdjudicationStoreError",
    "TeacherAgentAdjudicationStorePoisonedError",
]
